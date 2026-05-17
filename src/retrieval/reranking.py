"""Cross-encoder reranking — v0.2.

A cross-encoder takes (query, chunk_text) pairs and produces a single
relevance score per pair. This is more accurate than the bi-encoder retrieval
used to find the chunks in the first place, at the cost of being slower
(can't pre-compute chunk vectors). The standard pattern is to pull a larger
candidate set with bi-encoder retrieval, then rerank that set with a
cross-encoder, then keep the top-K reranked.

Default model: BAAI/bge-reranker-base (~500 MB). Same family as the
default bi-encoder embedder. Switchable via config/models.yaml.

Caller pattern:

    from src.retrieval.reranking import get_reranker
    reranker = get_reranker("bge-reranker-base")
    reranked_hits = reranker.rerank(query, hybrid_hits, chunk_lookup, top_k=10)

Returns a list of HybridHit with rerank_score populated and re-sorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.hybrid import HybridHit


@dataclass
class RerankerSpec:
    id: str
    hf_id: str
    device: str = "cpu"


_DEFAULT_REGISTRY: dict[str, RerankerSpec] = {
    "bge-reranker-base": RerankerSpec(
        id="bge-reranker-base",
        hf_id="BAAI/bge-reranker-base",
    ),
    "bge-reranker-large": RerankerSpec(
        id="bge-reranker-large",
        hf_id="BAAI/bge-reranker-large",
    ),
}


class CrossEncoderReranker:
    """Wraps sentence_transformers.CrossEncoder with a chunk-aware rerank API."""

    def __init__(self, spec: RerankerSpec):
        # Defer the import so importing src/retrieval/reranking.py doesn't
        # require sentence_transformers if the reranker is unused.
        from sentence_transformers import CrossEncoder

        self.spec = spec
        self.model = CrossEncoder(spec.hf_id, device=spec.device)

    def rerank(
        self,
        query: str,
        hits: list[HybridHit],
        chunk_lookup: dict[str, dict],
        top_k: int | None = None,
    ) -> list[HybridHit]:
        """Cross-encoder rescore + re-sort the candidate hits.

        Returns a new list with `rerank_score` populated and `rank` updated.
        Hits whose chunk_id is missing from lookup are dropped.
        """
        if not hits:
            return []

        # Build (query, text) pairs in input order; track the originating hit.
        pairs: list[tuple[str, str]] = []
        kept: list[HybridHit] = []
        for h in hits:
            meta = chunk_lookup.get(h.chunk_id)
            if not meta:
                continue
            pairs.append((query, meta["text"]))
            kept.append(h)

        if not pairs:
            return []

        scores = self.model.predict(pairs)
        # Attach scores; sort descending; truncate.
        scored: list[tuple[float, HybridHit]] = list(zip([float(s) for s in scores], kept))
        scored.sort(key=lambda x: x[0], reverse=True)
        if top_k is not None:
            scored = scored[:top_k]
        # Rebuild HybridHits with rerank_score + updated rank.
        out: list[HybridHit] = []
        for rank, (s, h) in enumerate(scored):
            out.append(
                HybridHit(
                    chunk_id=h.chunk_id,
                    bm25_score=h.bm25_score,
                    bm25_rank=h.bm25_rank,
                    dense_score=h.dense_score,
                    dense_rank=h.dense_rank,
                    rrf_score=h.rrf_score,
                    rank=rank,
                    rerank_score=s,
                )
            )
        return out


# Module-level cache so a single benchmark run doesn't reload the model
# between questions. Keyed by spec id.
_RERANKER_CACHE: dict[str, CrossEncoderReranker] = {}


def get_reranker(reranker_id: str) -> CrossEncoderReranker:
    if reranker_id in _RERANKER_CACHE:
        return _RERANKER_CACHE[reranker_id]
    if reranker_id not in _DEFAULT_REGISTRY:
        raise KeyError(
            f"Unknown reranker id '{reranker_id}'. Known: {list(_DEFAULT_REGISTRY)}"
        )
    inst = CrossEncoderReranker(_DEFAULT_REGISTRY[reranker_id])
    _RERANKER_CACHE[reranker_id] = inst
    return inst


# Legacy stub kept for back-compat with any caller that imported `rerank`.
def rerank(query: str, hits: list[HybridHit]) -> list[HybridHit]:
    """No-op fallback. Use CrossEncoderReranker.rerank for the real thing."""
    _ = query
    return hits
