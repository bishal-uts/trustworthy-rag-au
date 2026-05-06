"""BM25 retrieval over the chunked corpus.

Uses rank_bm25.BM25Okapi. Tokeniser is a simple regex word-split with
lowercasing — adequate for legal text where exact term matches matter.
The index is persisted as a single pickle file at
data/indexes/bm25.pkl, since BM25 has no per-embedder dimension.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEXES_DIR = REPO_ROOT / settings.paths.indexes_dir
BM25_PATH = INDEXES_DIR / "bm25.pkl"

TOKEN_RE = re.compile(r"\w+")


def tokenise(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class BM25Hit:
    chunk_id: str
    score: float
    rank: int


class BM25Index:
    def __init__(self, chunk_ids: list[str], bm25: BM25Okapi):
        self.chunk_ids = chunk_ids
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[dict]) -> "BM25Index":
        chunk_ids = [c["chunk_id"] for c in chunks]
        tokenised_corpus = [tokenise(c["text"]) for c in chunks]
        bm25 = BM25Okapi(tokenised_corpus)
        return cls(chunk_ids, bm25)

    def save(self, path: Path = BM25_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: Path = BM25_PATH) -> "BM25Index":
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(chunk_ids=payload["chunk_ids"], bm25=payload["bm25"])

    def search(self, query: str, top_k: int = 10) -> list[BM25Hit]:
        scores = self.bm25.get_scores(tokenise(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            BM25Hit(chunk_id=self.chunk_ids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
        ]
