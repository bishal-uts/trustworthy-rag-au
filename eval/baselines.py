"""Five RAG baselines for head-to-head comparison.

All baselines share retrieval / generation primitives from src/. They differ
only in which trust mechanisms are enabled:

    Baseline              | Retrieval        | Refuse on low conf | NLI faith check
    ----------------------|------------------|--------------------|------------------
    no_rag                | none             | no                 | no
    dense_only            | dense (FAISS)    | no                 | no
    hybrid                | BM25 + dense+RRF | no                 | no
    hybrid_refuse         | BM25 + dense+RRF | yes                | no
    trustworthy (current) | BM25 + dense+RRF | yes                | yes

This makes the contribution clear: each row in the comparison table
isolates the effect of one component.

All baselines return `BaselineOutput` (defined in comparison_metrics.py)
so downstream metrics treat them identically.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal

from eval.comparison_metrics import BaselineOutput
from src.config import settings
from src.confidence import score_and_route
from src.faithfulness import check as faithfulness_check
from src.generation import generate_answer
from src.models.manager import manager
from src.pipeline import (
    _ensure_bm25,
    _ensure_chunk_lookup,
    _hits_to_chunks,
    load_indexes,
)
from src.retrieval.dense import DenseIndex
from src.retrieval.hybrid import HybridRetriever
from src.schemas import Chunk

BaselineName = Literal["no_rag", "dense_only", "hybrid", "hybrid_refuse", "trustworthy"]


# ---------------------------------------------------------------------------
# Helper: convert internal Chunk dataclass list to plain dict list for output
# ---------------------------------------------------------------------------


def _chunks_to_dicts(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "doc": c.doc,
            "section": c.section,
            "paragraph": c.paragraph,
            "text": c.text,
            "bm25_score": c.bm25_score,
            "dense_score": c.dense_score,
            "rrf_score": c.rrf_score,
            "rank": c.rank,
        }
        for c in chunks
    ]


def _citations_to_dicts(citations) -> list[dict]:
    return [
        {"doc": c.doc, "section": c.section, "paragraph": c.paragraph, "span": c.span}
        for c in citations
    ]


# ---------------------------------------------------------------------------
# Baseline interface
# ---------------------------------------------------------------------------


class Baseline(ABC):
    name: BaselineName

    def __init__(self, llm_id: str, embedder_id: str, nli_id: str, top_k: int):
        self.llm_id = llm_id
        self.embedder_id = embedder_id
        self.nli_id = nli_id
        self.top_k = top_k

    @abstractmethod
    def run(self, question_id: str, question: str) -> BaselineOutput:
        ...


# ---------------------------------------------------------------------------
# 1. NoRAG: pure LLM, no retrieval
# ---------------------------------------------------------------------------


class NoRAGBaseline(Baseline):
    """Ask the LLM directly with no retrieved context.

    Sanity baseline: shows how much value retrieval adds on top of raw LLM
    knowledge. Expected to hallucinate heavily on domain-specific questions.
    """

    name: BaselineName = "no_rag"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        # Reuse the answer template with empty chunks list.
        draft = generate_answer(question=question, chunks=[], llm_id=self.llm_id, hedged=False)
        return BaselineOutput(
            question_id=question_id,
            state="confident",  # no_rag always answers
            answer=draft.text,
            citations=_citations_to_dicts(draft.citations),
            retrieved_chunks=[],
            llm_called=True,
            elapsed_ms=int((time.time() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# 2. DenseOnlyRAG: dense retrieval, always answer
# ---------------------------------------------------------------------------


class DenseOnlyRAGBaseline(Baseline):
    """FAISS-only retrieval, no BM25, no routing, no faith check.

    Isolates the value of hybrid retrieval: comparing this to HybridRAG
    shows how much BM25+RRF helps for legal text (exact statute numbers,
    defined terms).
    """

    name: BaselineName = "dense_only"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        dense = DenseIndex.load(self.embedder_id)
        chunk_lookup = _ensure_chunk_lookup()

        dense_hits = dense.search(question, top_k=self.top_k)
        retrieved = []
        for h in dense_hits:
            meta = chunk_lookup.get(h.chunk_id)
            if not meta:
                continue
            retrieved.append(
                Chunk(
                    chunk_id=h.chunk_id,
                    doc=meta["doc"],
                    section=meta["section"],
                    paragraph=meta["paragraph"],
                    text=meta["text"],
                    bm25_score=0.0,
                    dense_score=h.score,
                    rrf_score=0.0,
                    rank=h.rank,
                )
            )

        draft = generate_answer(
            question=question, chunks=retrieved, llm_id=self.llm_id, hedged=False
        )
        return BaselineOutput(
            question_id=question_id,
            state="confident",
            answer=draft.text,
            citations=_citations_to_dicts(draft.citations),
            retrieved_chunks=_chunks_to_dicts(retrieved),
            llm_called=True,
            elapsed_ms=int((time.time() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# 3. HybridRAG: BM25 + dense + RRF, always answer
# ---------------------------------------------------------------------------


class HybridRAGBaseline(Baseline):
    """Full hybrid retrieval but no routing, no faith check.

    Isolates the value of confidence routing: comparing this to
    HybridRAGWithRefuse shows the impact of "knowing when to shut up."
    """

    name: BaselineName = "hybrid"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        retriever: HybridRetriever = load_indexes(self.embedder_id)
        chunk_lookup = _ensure_chunk_lookup()

        hybrid_hits = retriever.search(question, top_k=self.top_k)
        retrieved = _hits_to_chunks(hybrid_hits, chunk_lookup)

        draft = generate_answer(
            question=question, chunks=retrieved, llm_id=self.llm_id, hedged=False
        )
        return BaselineOutput(
            question_id=question_id,
            state="confident",
            answer=draft.text,
            citations=_citations_to_dicts(draft.citations),
            retrieved_chunks=_chunks_to_dicts(retrieved),
            llm_called=True,
            elapsed_ms=int((time.time() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# 4. HybridRAG + Refuse: routing enabled, no faith check
# ---------------------------------------------------------------------------


class HybridRAGWithRefuseBaseline(Baseline):
    """Hybrid retrieval + confidence routing, but no NLI faithfulness check.

    Isolates the value of NLI faithfulness: comparing this to TrustworthyRAG
    shows the impact of the post-generation hallucination guard.
    """

    name: BaselineName = "hybrid_refuse"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        retriever: HybridRetriever = load_indexes(self.embedder_id)
        chunk_lookup = _ensure_chunk_lookup()

        hybrid_hits = retriever.search(question, top_k=self.top_k)
        bm25_hits = retriever.bm25_index.search(question, top_k=max(20, self.top_k * 2))
        dense_hits = retriever.dense_index.search(question, top_k=max(20, self.top_k * 2))
        retrieved = _hits_to_chunks(hybrid_hits, chunk_lookup)

        top_text = retrieved[0].text if retrieved else None
        decision = score_and_route(
            question=question,
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
            hybrid_hits=hybrid_hits,
            top_chunk_text=top_text,
            nli_id=self.nli_id,
        )

        if decision.state == "refused":
            return BaselineOutput(
                question_id=question_id,
                state="refused",
                answer=None,
                citations=[],
                retrieved_chunks=_chunks_to_dicts(retrieved),
                llm_called=False,
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        hedged = decision.state == "hedged"
        draft = generate_answer(
            question=question, chunks=retrieved, llm_id=self.llm_id, hedged=hedged
        )

        return BaselineOutput(
            question_id=question_id,
            state=decision.state,  # confident | hedged
            answer=draft.text,
            citations=_citations_to_dicts(draft.citations),
            retrieved_chunks=_chunks_to_dicts(retrieved),
            llm_called=True,
            elapsed_ms=int((time.time() - t0) * 1000),
        )


# ---------------------------------------------------------------------------
# 5. TrustworthyRAG: the full repo pipeline
# ---------------------------------------------------------------------------


class TrustworthyRAGBaseline(Baseline):
    """Full pipeline: hybrid + routing + NLI faith check + reroute on failure."""

    name: BaselineName = "trustworthy"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        retriever: HybridRetriever = load_indexes(self.embedder_id)
        chunk_lookup = _ensure_chunk_lookup()

        hybrid_hits = retriever.search(question, top_k=self.top_k)
        bm25_hits = retriever.bm25_index.search(question, top_k=max(20, self.top_k * 2))
        dense_hits = retriever.dense_index.search(question, top_k=max(20, self.top_k * 2))
        retrieved = _hits_to_chunks(hybrid_hits, chunk_lookup)

        top_text = retrieved[0].text if retrieved else None
        decision = score_and_route(
            question=question,
            bm25_hits=bm25_hits,
            dense_hits=dense_hits,
            hybrid_hits=hybrid_hits,
            top_chunk_text=top_text,
            nli_id=self.nli_id,
        )

        if decision.state == "refused":
            return BaselineOutput(
                question_id=question_id,
                state="refused",
                answer=None,
                citations=[],
                retrieved_chunks=_chunks_to_dicts(retrieved),
                llm_called=False,
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        hedged_flag = decision.state == "hedged"
        draft = generate_answer(
            question=question, chunks=retrieved, llm_id=self.llm_id, hedged=hedged_flag
        )

        # Faithfulness check
        faith = faithfulness_check(
            answer=draft.text,
            citations=draft.citations,
            retrieved_chunks=retrieved,
            nli_id=self.nli_id,
        )

        # Reroute confident → hedged if any claim failed entailment
        final_state = decision.state
        if (
            settings.faithfulness.reroute_to_hedged_on_failure
            and faith.unfaithful_count > 0
            and decision.state == "confident"
        ):
            final_state = "hedged"

        return BaselineOutput(
            question_id=question_id,
            state=final_state,  # type: ignore[arg-type]
            answer=draft.text,
            citations=_citations_to_dicts(draft.citations),
            retrieved_chunks=_chunks_to_dicts(retrieved),
            llm_called=True,
            elapsed_ms=int((time.time() - t0) * 1000),
            unfaithful_claim_count=faith.unfaithful_count,
            total_claim_count=len(faith.claims),
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


BASELINE_REGISTRY: dict[BaselineName, type[Baseline]] = {
    "no_rag": NoRAGBaseline,
    "dense_only": DenseOnlyRAGBaseline,
    "hybrid": HybridRAGBaseline,
    "hybrid_refuse": HybridRAGWithRefuseBaseline,
    "trustworthy": TrustworthyRAGBaseline,
}


def build_baseline(
    name: BaselineName,
    llm_id: str,
    embedder_id: str,
    nli_id: str,
    top_k: int = 10,
) -> Baseline:
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown baseline '{name}'. Known: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](
        llm_id=llm_id, embedder_id=embedder_id, nli_id=nli_id, top_k=top_k
    )


def list_baseline_names() -> list[str]:
    return list(BASELINE_REGISTRY.keys())
