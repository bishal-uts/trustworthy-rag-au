"""Confidence scoring + routing decision.

Two modes:

  Heuristic mode (default, v0.1) — transparent linear combiner over three
  signals plus two thresholds. All five numbers live in settings.confidence.

  Calibrated LR mode (v0.2, plan v1 Phase 3) — multinomial logistic regression
  whose coefficients are learned by `python -m eval.calibrate` and written
  back to settings.confidence as lr_intercepts / lr_coefs. Activated by
  `settings.confidence.use_lr_calibration: true`.

Three input signals (both modes):
 1. retrieval_top1_dense  : the top-1 dense similarity score (0..1)
 2. rank_overlap          : Jaccard overlap of top-3 ids between BM25 and dense (0..1)
 3. nli_entail            : NLI entailment score for (top chunk -> "the chunk answers the question")

The top1_dense_floor refusal gate is applied in BOTH modes (it's a safety
check, not part of the routing decision).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import settings
from src.models.manager import manager
from src.retrieval.bm25 import BM25Hit
from src.retrieval.dense import DenseHit
from src.retrieval.hybrid import HybridHit
from src.schemas import ConfidenceReport, ConfidenceSignals


@dataclass
class RoutingDecision:
    state: str  # "confident" | "hedged" | "refused"
    report: ConfidenceReport
    refused_reason: str | None = None


def rank_overlap_top3(bm25_hits: list[BM25Hit], dense_hits: list[DenseHit]) -> float:
    """Jaccard overlap of the chunk-id sets at top-3."""
    bm = {h.chunk_id for h in bm25_hits[:3]}
    de = {h.chunk_id for h in dense_hits[:3]}
    if not bm and not de:
        return 0.0
    return len(bm & de) / len(bm | de)


def nli_entail_top_chunk(
    nli_id: str, question: str, top_chunk_text: str
) -> tuple[float, str]:
    """NLI: does the top chunk entail "this chunk contains an answer to the question"?

    Returns (entail_score, label).
    """
    nli = manager.get_nli(nli_id)
    hypothesis = f"This text contains an answer to the question: {question}"
    res = nli.entail(premise=top_chunk_text, hypothesis=hypothesis)
    return res.entail, res.label


def score_and_route(
    question: str,
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    hybrid_hits: list[HybridHit],
    top_chunk_text: str | None,
    nli_id: str,
    override_floor: float | None = None,
) -> RoutingDecision:
    cfg = settings.confidence
    # override_floor lets the ablation runner disable the hard-floor gate
    # (pass 0.0) without mutating module-level settings.
    floor = override_floor if override_floor is not None else cfg.top1_dense_floor

    # Signal 1: top-1 dense similarity
    top1_dense = dense_hits[0].score if dense_hits else 0.0
    # FAISS cosine is in [-1, 1] for normalised vectors; clamp negative scores to 0 for the heuristic
    top1_dense_clamped = max(0.0, min(1.0, top1_dense))

    # Signal 2: rank overlap top-3
    rank_overlap = rank_overlap_top3(bm25_hits, dense_hits)

    # Signal 3: NLI entailment
    if top_chunk_text and hybrid_hits:
        nli_entail, nli_label = nli_entail_top_chunk(nli_id, question, top_chunk_text)
    else:
        nli_entail, nli_label = 0.0, "unknown"

    # ----- Combine signals into a single value (heuristic mode) -----
    weighted = (
        cfg.weight_top1_dense * top1_dense_clamped
        + cfg.weight_rank_overlap * rank_overlap
        + cfg.weight_nli_entail * nli_entail
    )
    total_weight = cfg.weight_top1_dense + cfg.weight_rank_overlap + cfg.weight_nli_entail
    value = weighted / total_weight if total_weight > 0 else 0.0
    value = max(0.0, min(1.0, value))

    signals = ConfidenceSignals(
        retrieval_top1_dense=top1_dense_clamped,
        retrieval_mean=(sum(h.score for h in dense_hits) / len(dense_hits)) if dense_hits else 0.0,
        rank_overlap=rank_overlap,
        nli_entail=nli_entail,
        nli_label=nli_label,  # type: ignore[arg-type]
    )
    report = ConfidenceReport(value=value, signals=signals)

    # ----- Refusal floor (applies in BOTH heuristic and LR modes) -----
    # Hard floor: weak retrieval alone is sufficient to refuse. NLI can be
    # noisy on out-of-domain queries (sometimes >0.5 entailment for nonsense
    # like "recipe for pavlova" against legal text); retrieval is the more
    # reliable signal. If the top-1 dense score is below the floor, the
    # corpus genuinely doesn't contain a relevant chunk — refuse.
    if top1_dense_clamped < floor:
        return RoutingDecision(
            state="refused",
            report=report,
            refused_reason=(
                f"Top-1 dense score {top1_dense_clamped:.2f} below floor "
                f"{floor} — corpus likely does not contain a relevant chunk."
            ),
        )

    # ----- Routing decision: LR mode if configured, else heuristic mode -----
    if cfg.use_lr_calibration and cfg.lr_coefs and cfg.lr_intercepts:
        state, lr_value = _lr_route(
            features=[top1_dense_clamped, rank_overlap, nli_entail],
            classes=cfg.lr_classes,
            intercepts=cfg.lr_intercepts,
            coefs=cfg.lr_coefs,
        )
        # Replace the heuristic `value` in the report with the LR's max-class
        # probability — gives the UI/output a meaningful "confidence" number.
        report = ConfidenceReport(value=lr_value, signals=signals)
        if state == "refused":
            return RoutingDecision(
                state="refused",
                report=report,
                refused_reason=f"LR-calibrated routing predicted refused (P={lr_value:.2f})",
            )
        return RoutingDecision(state=state, report=report)

    # Heuristic-mode threshold decision
    if value >= cfg.threshold_confident:
        return RoutingDecision(state="confident", report=report)
    if value >= cfg.threshold_hedged:
        return RoutingDecision(state="hedged", report=report)
    return RoutingDecision(
        state="refused",
        report=report,
        refused_reason=f"Confidence {value:.2f} < {cfg.threshold_hedged}",
    )


def _lr_route(
    features: list[float],
    classes: list[str],
    intercepts: list[float],
    coefs: list[list[float]],
) -> tuple[str, float]:
    """Multinomial logistic regression: returns (predicted_class, max_prob)."""
    logits = [
        intercepts[i] + sum(c * f for c, f in zip(coefs[i], features))
        for i in range(len(classes))
    ]
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    z = sum(exp_logits)
    probs = [e / z for e in exp_logits]
    idx = max(range(len(probs)), key=lambda i: probs[i])
    return classes[idx], probs[idx]
