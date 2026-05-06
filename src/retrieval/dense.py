"""Dense retrieval over the chunked corpus using FAISS.

One FAISS index per embedder (dimensionality + tokenisation differ).
Files: data/indexes/{embedder_id}.faiss (vectors) and
       data/indexes/{embedder_id}.ids.json (parallel chunk_id list).

The pipeline asserts the loaded index matches the active embedder.
Switching embedders triggers a "rebuild needed" banner in the UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.config import settings
from src.models.manager import manager

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
INDEXES_DIR = REPO_ROOT / settings.paths.indexes_dir


def index_paths(embedder_id: str) -> tuple[Path, Path]:
    return (
        INDEXES_DIR / f"{embedder_id}.faiss",
        INDEXES_DIR / f"{embedder_id}.ids.json",
    )


@dataclass
class DenseHit:
    chunk_id: str
    score: float
    rank: int


class DenseIndex:
    def __init__(self, embedder_id: str, chunk_ids: list[str], index: faiss.Index):
        self.embedder_id = embedder_id
        self.chunk_ids = chunk_ids
        self.index = index

    @classmethod
    def build(cls, embedder_id: str, chunks: list[dict]) -> "DenseIndex":
        embedder = manager.get_embedder(embedder_id)
        texts = [c["text"] for c in chunks]
        vecs = embedder.encode_passages(texts, show_progress=True)
        # Cosine via inner product on L2-normalised vectors (encode_passages normalises).
        index = faiss.IndexFlatIP(embedder.dim)
        index.add(vecs)
        return cls(embedder_id=embedder_id, chunk_ids=[c["chunk_id"] for c in chunks], index=index)

    def save(self) -> None:
        faiss_path, ids_path = index_paths(self.embedder_id)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        ids_path.write_text(
            json.dumps({"embedder_id": self.embedder_id, "chunk_ids": self.chunk_ids}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, embedder_id: str) -> "DenseIndex":
        faiss_path, ids_path = index_paths(embedder_id)
        if not faiss_path.exists() or not ids_path.exists():
            raise FileNotFoundError(
                f"No FAISS index for embedder '{embedder_id}'. "
                f"Build it with `python scripts/build_index.py --embedder {embedder_id}`."
            )
        meta = json.loads(ids_path.read_text(encoding="utf-8"))
        if meta["embedder_id"] != embedder_id:
            raise ValueError(
                f"Index file says embedder='{meta['embedder_id']}' but caller asked for '{embedder_id}'."
            )
        index = faiss.read_index(str(faiss_path))
        return cls(embedder_id=embedder_id, chunk_ids=meta["chunk_ids"], index=index)

    @classmethod
    def exists(cls, embedder_id: str) -> bool:
        f, i = index_paths(embedder_id)
        return f.exists() and i.exists()

    def search(self, query: str, top_k: int = 10) -> list[DenseHit]:
        embedder = manager.get_embedder(self.embedder_id)
        qvec = embedder.encode_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(qvec, top_k)
        hits: list[DenseHit] = []
        for rank, (i, s) in enumerate(zip(indices[0], scores[0])):
            if i < 0:
                continue
            hits.append(DenseHit(chunk_id=self.chunk_ids[int(i)], score=float(s), rank=rank))
        return hits
