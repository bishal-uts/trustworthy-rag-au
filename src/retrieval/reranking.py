"""Reranking — STUB in v0.1.

Plan v1 lists this file in §6 for architectural completeness. v0.1 returns
input unchanged. A cross-encoder reranker (e.g. BAAI/bge-reranker-large)
slots in here in a future version without changing any caller.
"""

from __future__ import annotations

from src.retrieval.hybrid import HybridHit


def rerank(query: str, hits: list[HybridHit]) -> list[HybridHit]:
    """No-op reranker. Returns hits unchanged."""
    _ = query  # explicitly unused
    return hits
