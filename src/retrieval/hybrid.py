"""Hybrid retrieval: BM25 + dense, fused via Reciprocal Rank Fusion (RRF).

RRF score for a chunk = sum over retrievers of 1 / (k + rank). The
`k` constant (default 60) dampens the contribution of high-ranked items
relative to the long tail. This is the standard hybrid-retrieval trick
for legal text — preserves both lexical (statute numbers, defined terms)
and semantic (paraphrases) matches without weight tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import settings
from src.retrieval.bm25 import BM25Hit, BM25Index
from src.retrieval.dense import DenseHit, DenseIndex


@dataclass
class HybridHit:
    chunk_id: str
    bm25_score: float
    bm25_rank: int | None
    dense_score: float
    dense_rank: int | None
    rrf_score: float
    rank: int
    rerank_score: float | None = None  # populated when cross-encoder reranker runs


def _rrf_contribution(rank: int | None, k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def fuse(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    rrf_k: int | None = None,
    top_k: int | None = None,
) -> list[HybridHit]:
    rrf_k = rrf_k if rrf_k is not None else settings.retrieval.rrf_k
    top_k = top_k if top_k is not None else settings.retrieval.top_k

    bm25_by_id = {h.chunk_id: h for h in bm25_hits}
    dense_by_id = {h.chunk_id: h for h in dense_hits}
    all_ids = set(bm25_by_id) | set(dense_by_id)

    fused: list[HybridHit] = []
    for cid in all_ids:
        bm = bm25_by_id.get(cid)
        de = dense_by_id.get(cid)
        rrf = _rrf_contribution(bm.rank if bm else None, rrf_k) + _rrf_contribution(
            de.rank if de else None, rrf_k
        )
        fused.append(
            HybridHit(
                chunk_id=cid,
                bm25_score=bm.score if bm else 0.0,
                bm25_rank=bm.rank if bm else None,
                dense_score=de.score if de else 0.0,
                dense_rank=de.rank if de else None,
                rrf_score=rrf,
                rank=-1,  # set below
            )
        )

    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    fused = fused[:top_k]
    for r, h in enumerate(fused):
        h.rank = r
    return fused


class HybridRetriever:
    """Convenience: holds both indexes and runs a hybrid query in one call."""

    def __init__(self, bm25_index: BM25Index, dense_index: DenseIndex):
        self.bm25_index = bm25_index
        self.dense_index = dense_index

    def search(self, query: str, top_k: int | None = None) -> list[HybridHit]:
        k = top_k if top_k is not None else settings.retrieval.top_k
        # Pull a generous candidate pool from each retriever; RRF picks top k from union.
        candidate_k = max(20, k * 2)
        bm25_hits = self.bm25_index.search(query, top_k=candidate_k)
        dense_hits = self.dense_index.search(query, top_k=candidate_k)
        return fuse(bm25_hits, dense_hits, top_k=k)
