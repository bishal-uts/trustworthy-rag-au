"""Baseline implementations used by the comparison-eval framework.

Each baseline is a subclass of `Baseline` with a `run()` method that takes
a question and returns a uniform `BaselineOutput`. The runner in
`eval/comparison_eval.py` iterates baselines × questions and feeds the
outputs to the judge + metrics.

Architecture note: four of the five baselines delegate to the canonical
`src.pipeline.run()` with different ablation flags. This collapses what was
previously several parallel retrieve-and-generate code paths into one source
of truth and lets you toggle features at the call site:

    Baseline                Pipeline call
    ─────────────────────   ─────────────────────────────────────────────
    DenseOnlyRAGBaseline    run(retrieval_mode="dense_only",
                                enable_routing=False,
                                enable_faithfulness=False)
    HybridRAGBaseline       run(enable_routing=False,
                                enable_faithfulness=False)
    HybridRAGWithRefuseB.   run(enable_faithfulness=False)
    TrustworthyRAGBaseline  run()                          # defaults

`NoRAGBaseline` stays special-cased — it has no retrieval at all and
therefore can't go through the standard pipeline.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Literal

from eval.comparison_metrics import BaselineOutput
from src.generation import generate_answer
from src.pipeline import run as pipeline_run
from src.schemas import Chunk, Citation, RAGOutput

BaselineName = Literal["no_rag", "dense_only", "hybrid", "hybrid_refuse", "trustworthy"]


# ---------------------------------------------------------------------------
# Adapters: convert internal pipeline types to the plain-dict output shape
# expected by BaselineOutput / the judge.
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


def _citations_to_dicts(citations: list[Citation]) -> list[dict]:
    return [
        {"doc": c.doc, "section": c.section, "paragraph": c.paragraph, "span": c.span}
        for c in citations
    ]


def _rag_output_to_baseline_output(
    question_id: str, out: RAGOutput
) -> BaselineOutput:
    """Adapt the canonical pipeline output to the comparison-eval shape."""
    return BaselineOutput(
        question_id=question_id,
        state=out.state,
        answer=out.answer,
        citations=_citations_to_dicts(out.citations),
        retrieved_chunks=_chunks_to_dicts(out.evidence.retrieved_chunks),
        llm_called=out.llm_called,
        elapsed_ms=out.elapsed_ms,
        unfaithful_claim_count=out.faithfulness.unfaithful_count,
        total_claim_count=len(out.faithfulness.claims),
    )


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
# 1. NoRAG: pure LLM, no retrieval (special-cased — can't use pipeline.run
# because the pipeline always retrieves).
# ---------------------------------------------------------------------------


class NoRAGBaseline(Baseline):
    """Ask the LLM directly with no retrieved context.

    Sanity baseline: shows how much value retrieval adds on top of raw LLM
    knowledge. Expected to hallucinate heavily on domain-specific questions.
    """

    name: BaselineName = "no_rag"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        t0 = time.time()
        draft = generate_answer(
            question=question, chunks=[], llm_id=self.llm_id, hedged=False
        )
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
# 2. DenseOnlyRAG: dense retrieval only, no routing, no faith check.
# Isolates the value of hybrid retrieval (compare to HybridRAG).
# ---------------------------------------------------------------------------


class DenseOnlyRAGBaseline(Baseline):
    """FAISS-only retrieval, always answer. Delegates to pipeline.run."""

    name: BaselineName = "dense_only"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        out = pipeline_run(
            question=question,
            llm_id=self.llm_id,
            embedder_id=self.embedder_id,
            nli_id=self.nli_id,
            top_k=self.top_k,
            retrieval_mode="dense_only",
            enable_routing=False,
            enable_faithfulness=False,
        )
        return _rag_output_to_baseline_output(question_id, out)


# ---------------------------------------------------------------------------
# 3. HybridRAG: BM25 + dense + RRF, no routing, no faith check.
# Isolates the value of confidence routing (compare to HybridRAGWithRefuse).
# ---------------------------------------------------------------------------


class HybridRAGBaseline(Baseline):
    """Full hybrid retrieval, always answer. Delegates to pipeline.run."""

    name: BaselineName = "hybrid"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        out = pipeline_run(
            question=question,
            llm_id=self.llm_id,
            embedder_id=self.embedder_id,
            nli_id=self.nli_id,
            top_k=self.top_k,
            retrieval_mode="hybrid",
            enable_routing=False,
            enable_faithfulness=False,
        )
        return _rag_output_to_baseline_output(question_id, out)


# ---------------------------------------------------------------------------
# 4. HybridRAG + Refuse: routing enabled, no faith check.
# Isolates the value of NLI faithfulness (compare to TrustworthyRAG).
# ---------------------------------------------------------------------------


class HybridRAGWithRefuseBaseline(Baseline):
    """Hybrid retrieval + confidence routing, no faith check. Delegates to pipeline.run."""

    name: BaselineName = "hybrid_refuse"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        out = pipeline_run(
            question=question,
            llm_id=self.llm_id,
            embedder_id=self.embedder_id,
            nli_id=self.nli_id,
            top_k=self.top_k,
            retrieval_mode="hybrid",
            enable_routing=True,
            enable_faithfulness=False,
        )
        return _rag_output_to_baseline_output(question_id, out)


# ---------------------------------------------------------------------------
# 5. TrustworthyRAG: the full repo pipeline.
# ---------------------------------------------------------------------------


class TrustworthyRAGBaseline(Baseline):
    """Full pipeline: hybrid + routing + NLI faith check + reroute on failure."""

    name: BaselineName = "trustworthy"

    def run(self, question_id: str, question: str) -> BaselineOutput:
        out = pipeline_run(
            question=question,
            llm_id=self.llm_id,
            embedder_id=self.embedder_id,
            nli_id=self.nli_id,
            top_k=self.top_k,
        )
        return _rag_output_to_baseline_output(question_id, out)


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
