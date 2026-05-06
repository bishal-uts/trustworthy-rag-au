"""ModelManager: lazy-load embedders, NLI models, and LLM clients.

Singleton entry point for the rest of the pipeline. Pipeline code talks
only to this manager — never imports the underlying SDKs directly. This
makes model switching a one-line change anywhere downstream.

Memory policy:
 - Embedders and NLI models live in the Python process. Manager keeps an
   LRU dict and evicts past `settings.memory.ram_budget_mb`.
 - LLMs live in Ollama's process; manager only holds a thin HTTP client,
   so memory accounting for LLMs is a no-op here. Ollama's own keep_alive
   handles its model warmth.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

from rich.console import Console

from src.config import settings
from src.models.registry import (
    get_embedder_spec,
    get_llm_spec,
    get_nli_spec,
)

if TYPE_CHECKING:
    from src.models.embed import Embedder
    from src.models.llm import OllamaLLM
    from src.models.nli import NLIModel

console = Console()


class ModelManager:
    """Lazy loader + LRU cache for all model types.

    Thread-safe via a coarse lock — model loading is rare and slow enough
    that fine-grained locking isn't worth the complexity.
    """

    def __init__(self):
        self._embedders: OrderedDict[str, Embedder] = OrderedDict()
        self._nli: OrderedDict[str, NLIModel] = OrderedDict()
        self._llms: OrderedDict[str, OllamaLLM] = OrderedDict()
        self._ram_used_mb: int = 0
        self._lock = threading.RLock()

    # ---------- LLM ----------

    def get_llm(self, llm_id: str) -> "OllamaLLM":
        with self._lock:
            if llm_id in self._llms:
                self._llms.move_to_end(llm_id)
                return self._llms[llm_id]
            from src.models.llm import OllamaLLM  # late import

            spec = get_llm_spec(llm_id)
            console.print(f"[cyan]Loading LLM client: {spec.display_name}[/cyan]")
            inst = OllamaLLM(spec)
            self._llms[llm_id] = inst
            return inst

    # ---------- Embedder ----------

    def get_embedder(self, embedder_id: str) -> "Embedder":
        with self._lock:
            if embedder_id in self._embedders:
                self._embedders.move_to_end(embedder_id)
                return self._embedders[embedder_id]

            from src.models.embed import Embedder  # late import

            spec = get_embedder_spec(embedder_id)
            self._make_room(spec.ram_estimate_mb)
            console.print(f"[cyan]Loading embedder: {spec.display_name}[/cyan]")
            inst = Embedder(spec)
            self._embedders[embedder_id] = inst
            self._ram_used_mb += spec.ram_estimate_mb
            return inst

    # ---------- NLI ----------

    def get_nli(self, nli_id: str) -> "NLIModel":
        with self._lock:
            if nli_id in self._nli:
                self._nli.move_to_end(nli_id)
                return self._nli[nli_id]

            from src.models.nli import NLIModel  # late import

            spec = get_nli_spec(nli_id)
            self._make_room(spec.ram_estimate_mb)
            console.print(f"[cyan]Loading NLI: {spec.display_name}[/cyan]")
            inst = NLIModel(spec)
            self._nli[nli_id] = inst
            self._ram_used_mb += spec.ram_estimate_mb
            return inst

    # ---------- Eviction ----------

    def _make_room(self, needed_mb: int) -> None:
        budget = settings.memory.ram_budget_mb
        # Evict embedders and NLI in LRU order until we fit
        while self._ram_used_mb + needed_mb > budget:
            evicted = self._evict_one()
            if not evicted:
                break

    def _evict_one(self) -> bool:
        """Evict the LRU non-LLM model. Returns True if something was evicted."""
        # Pick whichever (embedder | nli) has the older LRU entry
        candidates = []
        if self._embedders:
            candidates.append(("embedder", next(iter(self._embedders))))
        if self._nli:
            candidates.append(("nli", next(iter(self._nli))))
        if not candidates:
            return False

        # First entry of an OrderedDict is the LRU
        kind, key = candidates[0]
        if kind == "embedder":
            inst = self._embedders.pop(key)
            spec = get_embedder_spec(key)
        else:
            inst = self._nli.pop(key)
            spec = get_nli_spec(key)
        try:
            inst.unload()
        except Exception:
            pass
        self._ram_used_mb = max(0, self._ram_used_mb - spec.ram_estimate_mb)
        console.print(f"[yellow]Evicted {kind}: {spec.display_name}[/yellow]")
        return True

    # ---------- Explicit unloads ----------

    def unload(self, model_id: str) -> None:
        with self._lock:
            for store, getter in (
                (self._embedders, get_embedder_spec),
                (self._nli, get_nli_spec),
            ):
                if model_id in store:
                    inst = store.pop(model_id)
                    try:
                        inst.unload()
                    except Exception:
                        pass
                    self._ram_used_mb = max(0, self._ram_used_mb - getter(model_id).ram_estimate_mb)
                    return
            if model_id in self._llms:
                self._llms.pop(model_id)

    def unload_all(self) -> None:
        with self._lock:
            for store in (self._embedders, self._nli):
                while store:
                    _, inst = store.popitem(last=False)
                    try:
                        inst.unload()
                    except Exception:
                        pass
            self._llms.clear()
            self._ram_used_mb = 0

    # ---------- Status ----------

    def status(self) -> dict:
        with self._lock:
            return {
                "ram_used_mb": self._ram_used_mb,
                "ram_budget_mb": settings.memory.ram_budget_mb,
                "loaded_embedders": list(self._embedders.keys()),
                "loaded_nli": list(self._nli.keys()),
                "loaded_llms": list(self._llms.keys()),
            }


# Module-level singleton
manager = ModelManager()
