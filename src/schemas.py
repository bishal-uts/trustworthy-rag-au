"""Output schema for the RAG pipeline. Matches plan v1 §3.4."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

State = Literal["confident", "hedged", "refused"]


class Citation(BaseModel):
    doc: str
    section: str
    paragraph: str
    span: str = ""


class ConfidenceSignals(BaseModel):
    retrieval_top1_dense: float = 0.0
    retrieval_mean: float = 0.0
    rank_overlap: float = 0.0
    nli_entail: float = 0.0
    nli_label: Literal["entailment", "neutral", "contradiction", "unknown"] = "unknown"


class ConfidenceReport(BaseModel):
    value: float
    signals: ConfidenceSignals


class FaithfulnessClaim(BaseModel):
    text: str
    cited: bool = False
    entailed: bool = False
    nli_score: float = 0.0


class FaithfulnessReport(BaseModel):
    claims: list[FaithfulnessClaim] = Field(default_factory=list)
    unfaithful_count: int = 0


class Chunk(BaseModel):
    """A retrieved chunk surfaced to the user."""

    chunk_id: str
    doc: str
    section: str
    paragraph: str
    text: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rank: int = 0


class Evidence(BaseModel):
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    refused_reason: str | None = None


class RAGOutput(BaseModel):
    """Full structured output of the pipeline. Always returned, every state."""

    state: State
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceReport
    faithfulness: FaithfulnessReport = Field(default_factory=FaithfulnessReport)
    evidence: Evidence = Field(default_factory=Evidence)
    # Diagnostics that aren't part of the architectural schema but useful in UI
    llm_called: bool = False
    elapsed_ms: int = 0
