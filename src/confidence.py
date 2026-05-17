"""Heuristic confidence scoring + routing decision.

This is the v0.1 version — a transparent linear combiner over three signals.
Plan v1 Phase 3 replaces this with a logistic regression trained on 50
manually labelled tuning questions. The interface here stays the same so
swapping is one-file work.

Three signals:
 1. retrieval_top1_dense  : the top-1 dense similarity score (0..1)
 2. rank_overlap          : Jaccard overlap of top-3 ids between BM25 and dense (0..1)
 3. nli_entail            : NLI entailment score for (top chunk -> "the chunk answers the question")

Combined as a weighted sum with weights from settings.confidence.
Output value is clamped to [0, 1].

Routing thresholds are also from settings.confidence:
 - value >= threshold_confident   -> "confident"
 - value >= threshold_hedged      -> "hedged"
 - else                           -> "refused"
"""

from __future__ import annotations

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

    # Combine
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

    # Hard floor: weak retrieval alone is sufficient to refuse. NLI can be
    # noisy on out-of-domain queries (sometimes >0.5 entailment for nonsense
    # like "recipe for pavlova" against legal text); retrieval is the more
    # reliable signal. If the top-1 dense score is below the floor, the
    # corpus genuinely doesn't contain a relevant chunk — refuse.
    refused_reason: str | None = None
    if top1_dense_clamped < floor:
        return RoutingDecision(
            state="refused",
            report=report,
            refused_reason=(
                f"Top-1 dense score {top1_dense_clamped:.2f} below floor "
                f"{floor} — corpus likely does not contain a relevant chunk."
            ),
        )

    if value >= cfg.threshold_confident:
        return RoutingDecision(state="confident", report=report)
    if value >= cfg.threshold_hedged:
        return RoutingDecision(state="hedged", report=report)
    return RoutingDecision(
        state="refused",
        report=report,
        refused_reason=f"Confidence {value:.2f} < {cfg.threshold_hedged}",
    )
