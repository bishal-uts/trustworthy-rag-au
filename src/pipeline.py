"""End-to-end pipeline.

Implements the architecture diagram from plan v1 §4:

    query -> hybrid retrieval -> confidence scoring -> route ->
        (refuse | generate -> faithfulness check) -> structured output

Public entry points:
 - load_indexes(embedder_id) -> HybridRetriever  (cached per-embedder)
 - run(question, llm_id, embedder_id, nli_id, top_k=None) -> RAGOutput
 - main()                                        (CLI: python -m src.pipeline)

The CLI is plan v1's first definition-of-done item:
    python -m src.pipeline --question "..."
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal

from rich.console import Console
from rich.pretty import pprint

# I-don't-know detector: the 4th safety net described in v0.2 methodology.
# If the LLM produces a refusal-style answer or returns no citations, the
# pipeline downgrades to "refused" regardless of confidence routing — closes
# the gap where retrieval misses + honest LLM refusal can produce a
# false-confident state (see cross_001 analysis in calibration notes).
_REFUSAL_PATTERNS = re.compile(
    r"(INSUFFICIENT_CONTEXT|"
    r"provided context does not contain|insufficient (?:information|context|details)|"
    r"cannot (?:determine|answer|find)|unable to (?:answer|determine|find)|"
    r"i (?:don[''']t|do not) (?:know|have)|no (?:information|mention) (?:of|about)|"
    r"not (?:mentioned|specified|provided|available) in the (?:context|provided))",
    re.IGNORECASE,
)


def _looks_like_idk(text: str | None, citations: list) -> bool:
    """Return True if the LLM's answer is effectively 'I don't know'.

    Triggers:
     1. Empty citations list — system has no source to back the answer.
     2. Refusal phrase in the first 500 chars of the answer text.
    """
    if not text or not text.strip():
        return True
    if not citations:
        return True
    head = text[:500]
    return bool(_REFUSAL_PATTERNS.search(head))

from src.chunking import load_all_chunks
from src.config import settings
from src.confidence import score_and_route
from src.faithfulness import check as faithfulness_check
from src.generation import generate_answer
from src.retrieval.bm25 import BM25Hit, BM25Index
from src.retrieval.dense import DenseHit, DenseIndex
from src.retrieval.hybrid import HybridHit, HybridRetriever
from src.retrieval.reranking import get_reranker
from src.schemas import (
    Chunk,
    ConfidenceReport,
    ConfidenceSignals,
    Evidence,
    FaithfulnessReport,
    RAGOutput,
)

RetrievalMode = Literal["hybrid", "bm25_only", "dense_only"]

console = Console()


@dataclass
class _Cache:
    bm25: BM25Index | None = None
    chunks_by_id: dict[str, dict] | None = None
    retrievers: dict[str, HybridRetriever] | None = None  # keyed by embedder_id


_cache = _Cache(retrievers={})


def _ensure_chunk_lookup() -> dict[str, dict]:
    if _cache.chunks_by_id is None:
        chunks = load_all_chunks()
        if not chunks:
            raise RuntimeError(
                "No chunks found. Run `python -m src.parsing --all` then "
                "`python -m src.chunking --all` first."
            )
        _cache.chunks_by_id = {c["chunk_id"]: c for c in chunks}
    return _cache.chunks_by_id


def _ensure_bm25() -> BM25Index:
    if _cache.bm25 is None:
        try:
            _cache.bm25 = BM25Index.load()
        except FileNotFoundError as e:
            raise RuntimeError(
                "BM25 index not built. Run `python scripts/build_index.py` first."
            ) from e
    return _cache.bm25


def load_indexes(embedder_id: str) -> HybridRetriever:
    """Load (or get-cached) BM25 + dense indexes for the given embedder."""
    if _cache.retrievers is None:
        _cache.retrievers = {}
    if embedder_id in _cache.retrievers:
        return _cache.retrievers[embedder_id]
    bm25 = _ensure_bm25()
    dense = DenseIndex.load(embedder_id)
    retriever = HybridRetriever(bm25_index=bm25, dense_index=dense)
    _cache.retrievers[embedder_id] = retriever
    return retriever


def _hits_to_chunks(hits: list[HybridHit], lookup: dict[str, dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for h in hits:
        meta = lookup.get(h.chunk_id)
        if not meta:
            continue
        out.append(
            Chunk(
                chunk_id=h.chunk_id,
                doc=meta["doc"],
                section=meta["section"],
                paragraph=meta["paragraph"],
                text=meta["text"],
                bm25_score=h.bm25_score,
                dense_score=h.dense_score,
                rrf_score=h.rrf_score,
                rerank_score=h.rerank_score,
                rank=h.rank,
            )
        )
    return out


def _bm25_only_to_hybrid_hits(bm25_hits: list[BM25Hit], top_k: int) -> list[HybridHit]:
    """Adapter for the bm25_only ablation: build HybridHits from BM25 ranks alone."""
    return [
        HybridHit(
            chunk_id=h.chunk_id,
            bm25_score=h.score,
            bm25_rank=h.rank,
            dense_score=0.0,
            dense_rank=None,
            rrf_score=h.score,  # rrf_score unused; surface raw bm25 score for debugging
            rank=h.rank,
        )
        for h in bm25_hits[:top_k]
    ]


def _dense_only_to_hybrid_hits(dense_hits: list[DenseHit], top_k: int) -> list[HybridHit]:
    """Adapter for the dense_only ablation: build HybridHits from dense ranks alone."""
    return [
        HybridHit(
            chunk_id=h.chunk_id,
            bm25_score=0.0,
            bm25_rank=None,
            dense_score=h.score,
            dense_rank=h.rank,
            rrf_score=h.score,  # rrf_score unused; surface raw dense score for debugging
            rank=h.rank,
        )
        for h in dense_hits[:top_k]
    ]


def run(
    question: str,
    llm_id: str | None = None,
    embedder_id: str | None = None,
    nli_id: str | None = None,
    top_k: int | None = None,
    retrieval_mode: RetrievalMode = "hybrid",
    enable_faithfulness: bool = True,
    enable_floor_gate: bool = True,
    enable_routing: bool = True,
    use_reranker: bool = False,
    reranker_id: str = "bge-reranker-base",
) -> RAGOutput:
    """Full end-to-end pipeline. Returns the structured RAGOutput.

    Ablation parameters (all default-preserving):
     - retrieval_mode: "hybrid" (default), "bm25_only", or "dense_only". Picks
       which retriever's ranking becomes the user-facing retrieved set. Both
       retrievers are always called so confidence signals stay computed.
     - enable_faithfulness: when False, skip the post-gen NLI check entirely.
       The faith report stays empty and no confident→hedged downgrade happens.
     - enable_floor_gate: when False, disable the top-1 dense floor that
       short-circuits to refused. Implemented by passing override_floor=0.0
       to score_and_route.
     - enable_routing: when False, ignore the confidence router's decision
       and always proceed to generation as if "confident". Confidence signals
       are still computed (for transparency in the output) but don't gate
       the answer. Used by always-answer baselines (no_rag, hybrid).
     - use_reranker: when True, pull a larger candidate set (3×top_k) from
       hybrid retrieval and rerank with a cross-encoder before passing
       to the LLM. Confidence signals stay based on the original retrieval.
     - reranker_id: which cross-encoder to use (default: bge-reranker-base).
    """
    t0 = time.time()
    llm_id = llm_id or settings.default_llm
    embedder_id = embedder_id or settings.default_embedder
    nli_id = nli_id or settings.default_nli
    top_k = top_k or settings.retrieval.top_k

    chunk_lookup = _ensure_chunk_lookup()
    retriever = load_indexes(embedder_id)

    # -------- Retrieval --------
    # Always call both retrievers so confidence signals (top1_dense, rank_overlap)
    # are computed identically across ablation modes — the ablation only
    # changes which ranking surfaces to the LLM.
    candidate_k = max(20, top_k * 2)
    bm25_hits = retriever.bm25_index.search(question, top_k=candidate_k)
    dense_hits = retriever.dense_index.search(question, top_k=candidate_k)

    # When the reranker is enabled, pull MORE candidates from hybrid first,
    # then rerank, then keep top_k. The "extra candidates" are only used
    # for reranking — the LLM still sees top_k chunks.
    rerank_pool_k = top_k * 3 if use_reranker else top_k

    if retrieval_mode == "hybrid":
        hybrid_hits = retriever.search(question, top_k=rerank_pool_k)
    elif retrieval_mode == "bm25_only":
        hybrid_hits = _bm25_only_to_hybrid_hits(bm25_hits, rerank_pool_k)
    elif retrieval_mode == "dense_only":
        hybrid_hits = _dense_only_to_hybrid_hits(dense_hits, rerank_pool_k)
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode!r}")

    if use_reranker:
        reranker = get_reranker(reranker_id)
        hybrid_hits = reranker.rerank(question, hybrid_hits, chunk_lookup, top_k=top_k)

    retrieved = _hits_to_chunks(hybrid_hits, chunk_lookup)

    # -------- Confidence + routing --------
    # Signals are always computed (kept in the output for transparency).
    # `enable_routing=False` ignores the decision and forces "confident".
    top_text = retrieved[0].text if retrieved else None
    decision = score_and_route(
        question=question,
        bm25_hits=bm25_hits,
        dense_hits=dense_hits,
        hybrid_hits=hybrid_hits,
        top_chunk_text=top_text,
        nli_id=nli_id,
        override_floor=None if enable_floor_gate else 0.0,
    )
    effective_state = decision.state if enable_routing else "confident"

    elapsed_pre_gen_ms = int((time.time() - t0) * 1000)

    # -------- Refused path: no LLM call --------
    if effective_state == "refused":
        return RAGOutput(
            state="refused",
            answer=None,
            citations=[],
            confidence=decision.report,
            faithfulness=FaithfulnessReport(),
            evidence=Evidence(
                retrieved_chunks=retrieved,
                refused_reason=decision.refused_reason,
            ),
            llm_called=False,
            elapsed_ms=elapsed_pre_gen_ms,
        )

    # -------- Generate (confident or hedged) --------
    hedged_flag = effective_state == "hedged"
    draft = generate_answer(
        question=question,
        chunks=retrieved,
        llm_id=llm_id,
        hedged=hedged_flag,
    )

    # -------- Faithfulness check (optional) --------
    if enable_faithfulness:
        faith = faithfulness_check(
            answer=draft.text,
            citations=draft.citations,
            retrieved_chunks=retrieved,
            nli_id=nli_id,
        )
    else:
        faith = FaithfulnessReport()

    # If any claim fails entailment and policy says so, drop to hedged.
    # Skipped when faithfulness checking is disabled or routing is disabled
    # (no point downgrading when the caller has explicitly opted out of
    # confidence-based gating).
    final_state = effective_state
    if (
        enable_routing
        and enable_faithfulness
        and settings.faithfulness.reroute_to_hedged_on_failure
        and faith.unfaithful_count > 0
        and effective_state == "confident"
    ):
        final_state = "hedged"

    # 4th safety net (v0.2): I-don't-know detector. If the LLM produced a
    # refusal-style answer or empty citations, the system shouldn't claim
    # confident/hedged. This closes the cross_001 gap where retrieval missed,
    # the LLM honestly said "context doesn't contain this", and no other
    # check caught it. Only applies when routing is enabled — `--no-routing`
    # ablation bypasses this for proper baseline comparison.
    refused_reason: str | None = None
    if enable_routing and _looks_like_idk(draft.text, draft.citations):
        final_state = "refused"
        refused_reason = (
            "LLM answer signals 'I don't know' (refusal phrase or empty citations) "
            "despite passing confidence routing"
        )
        return RAGOutput(
            state="refused",
            answer=None,
            citations=[],
            confidence=decision.report,
            faithfulness=faith,
            evidence=Evidence(retrieved_chunks=retrieved, refused_reason=refused_reason),
            llm_called=True,  # we did call the LLM, just discarded its answer
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    return RAGOutput(
        state=final_state,  # type: ignore[arg-type]
        answer=draft.text,
        citations=draft.citations,
        confidence=decision.report,
        faithfulness=faith,
        evidence=Evidence(retrieved_chunks=retrieved, refused_reason=None),
        llm_called=True,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the RAG pipeline on one question.")
    parser.add_argument("--question", required=True, help="The question to answer")
    parser.add_argument("--llm", default=None, help="LLM id (default: settings.default_llm)")
    parser.add_argument("--embedder", default=None, help="Embedder id (default: settings.default_embedder)")
    parser.add_argument("--nli", default=None, help="NLI id (default: settings.default_nli)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k retrieval (default: 10)")
    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25_only", "dense_only"],
        default="hybrid",
        help="Retrieval ablation mode (default: hybrid)",
    )
    parser.add_argument(
        "--no-faithfulness",
        action="store_true",
        help="Disable the post-generation NLI faithfulness check",
    )
    parser.add_argument(
        "--no-floor-gate",
        action="store_true",
        help="Disable the top-1 dense floor that short-circuits to refused",
    )
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Always answer, ignoring the confidence router (always-answer baselines)",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranker after hybrid retrieval",
    )
    parser.add_argument(
        "--reranker-id",
        default="bge-reranker-base",
        help="Cross-encoder reranker id (default: bge-reranker-base)",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    out = run(
        question=args.question,
        llm_id=args.llm,
        embedder_id=args.embedder,
        nli_id=args.nli,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        enable_faithfulness=not args.no_faithfulness,
        enable_floor_gate=not args.no_floor_gate,
        enable_routing=not args.no_routing,
        use_reranker=args.use_reranker,
        reranker_id=args.reranker_id,
    )

    if args.json:
        print(json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
        return 0

    # Human-readable summary
    console.rule(f"[bold]{out.state.upper()}[/bold]")
    if out.answer:
        console.print(f"\n[bold]Answer:[/bold]\n{out.answer}\n")
    else:
        console.print(f"\n[yellow]Refused:[/yellow] {out.evidence.refused_reason}\n")
    console.print(f"[bold]Confidence:[/bold] {out.confidence.value:.2f}")
    console.print(f"  signals: {out.confidence.signals.model_dump()}")
    if out.citations:
        console.print("\n[bold]Citations:[/bold]")
        for c in out.citations:
            console.print(f"  - {c.doc} | {c.section} | ¶{c.paragraph}")
    console.print(f"\n[bold]Top retrieved chunks:[/bold]")
    for ch in out.evidence.retrieved_chunks[:5]:
        console.print(
            f"  #{ch.rank} {ch.doc} | {ch.section} | ¶{ch.paragraph}  "
            f"(rrf={ch.rrf_score:.3f}, bm25={ch.bm25_score:.2f}, dense={ch.dense_score:.2f})"
        )
    if out.faithfulness.claims:
        console.print(f"\n[bold]Faithfulness:[/bold] {out.faithfulness.unfaithful_count} unfaithful / {len(out.faithfulness.claims)} claims")
    console.print(f"\n[dim]elapsed: {out.elapsed_ms} ms · llm_called: {out.llm_called}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
