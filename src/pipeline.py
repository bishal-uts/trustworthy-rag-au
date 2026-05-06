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
import sys
import time
from dataclasses import dataclass

from rich.console import Console
from rich.pretty import pprint

from src.chunking import load_all_chunks
from src.config import settings
from src.confidence import score_and_route
from src.faithfulness import check as faithfulness_check
from src.generation import generate_answer
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseIndex
from src.retrieval.hybrid import HybridHit, HybridRetriever
from src.schemas import (
    Chunk,
    ConfidenceReport,
    ConfidenceSignals,
    Evidence,
    FaithfulnessReport,
    RAGOutput,
)

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
                rank=h.rank,
            )
        )
    return out


def run(
    question: str,
    llm_id: str | None = None,
    embedder_id: str | None = None,
    nli_id: str | None = None,
    top_k: int | None = None,
) -> RAGOutput:
    """Full end-to-end pipeline. Returns the structured RAGOutput."""
    t0 = time.time()
    llm_id = llm_id or settings.default_llm
    embedder_id = embedder_id or settings.default_embedder
    nli_id = nli_id or settings.default_nli
    top_k = top_k or settings.retrieval.top_k

    chunk_lookup = _ensure_chunk_lookup()
    retriever = load_indexes(embedder_id)

    # -------- Hybrid retrieval --------
    hybrid_hits = retriever.search(question, top_k=top_k)
    bm25_hits = retriever.bm25_index.search(question, top_k=max(20, top_k * 2))
    dense_hits = retriever.dense_index.search(question, top_k=max(20, top_k * 2))
    retrieved = _hits_to_chunks(hybrid_hits, chunk_lookup)

    # -------- Confidence + routing --------
    top_text = retrieved[0].text if retrieved else None
    decision = score_and_route(
        question=question,
        bm25_hits=bm25_hits,
        dense_hits=dense_hits,
        hybrid_hits=hybrid_hits,
        top_chunk_text=top_text,
        nli_id=nli_id,
    )

    elapsed_pre_gen_ms = int((time.time() - t0) * 1000)

    # -------- Refused path: no LLM call --------
    if decision.state == "refused":
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
    hedged_flag = decision.state == "hedged"
    draft = generate_answer(
        question=question,
        chunks=retrieved,
        llm_id=llm_id,
        hedged=hedged_flag,
    )

    # -------- Faithfulness check --------
    faith = faithfulness_check(
        answer=draft.text,
        citations=draft.citations,
        retrieved_chunks=retrieved,
        nli_id=nli_id,
    )

    # If any claim fails entailment and policy says so, drop to hedged
    final_state = decision.state
    if (
        settings.faithfulness.reroute_to_hedged_on_failure
        and faith.unfaithful_count > 0
        and decision.state == "confident"
    ):
        final_state = "hedged"

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
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    args = parser.parse_args()

    out = run(
        question=args.question,
        llm_id=args.llm,
        embedder_id=args.embedder,
        nli_id=args.nli,
        top_k=args.top_k,
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
