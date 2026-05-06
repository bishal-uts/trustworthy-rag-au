"""sentence-transformers wrapper.

Wraps a sentence-transformers SentenceTransformer model with the per-model
query/passage prefix protocol from models.yaml. BGE needs a query prefix;
E5 needs both prefixes; MiniLM needs neither. Encoding which prefix to use
in the model spec means the pipeline never has to special-case.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.config import EmbedderSpec, settings

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Honour HF_HOME from .env / shell env; fall back to repo-local cache
os.environ.setdefault("HF_HOME", str(REPO_ROOT / settings.paths.hf_cache_dir))


class Embedder:
    """One loaded sentence-transformers model, with prefix protocol applied."""

    def __init__(self, spec: EmbedderSpec, device: str | None = None):
        # Late import so the package can be imported without torch installed
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.id = spec.id
        self.dim = spec.dim
        device = device or settings.memory.embedder_device
        self.device = device
        self.model = SentenceTransformer(spec.hf_id, device=device)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query string. Returns 1-D array of shape (dim,)."""
        prefixed = self.spec.query_prefix + text
        vec = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.astype(np.float32)

    def encode_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode many passages. Returns 2-D array of shape (len(texts), dim)."""
        prefixed = [self.spec.passage_prefix + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def unload(self) -> None:
        """Release model weights. Call before loading a different embedder if RAM is tight."""
        try:
            import torch

            del self.model
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
