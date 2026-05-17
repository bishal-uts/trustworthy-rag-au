"""Configuration loader.

Two YAML files drive the system:
- config/default.yaml: thresholds, paths, default model selections
- config/models.yaml:  the model catalog

Pydantic models validate both on load. Importing `settings` and
`model_catalog` from this module is the single supported entry point;
do not read the YAML files directly elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


class RetrievalSettings(BaseModel):
    top_k: int = 10
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rrf_k: int = 60


class ChunkingSettings(BaseModel):
    max_tokens: int = 512
    min_tokens: int = 50
    overlap_tokens: int = 50


class ConfidenceSettings(BaseModel):
    threshold_confident: float = 0.75
    threshold_hedged: float = 0.40
    weight_top1_dense: float = 0.4
    weight_rank_overlap: float = 0.2
    weight_nli_entail: float = 0.4
    top1_dense_floor: float = 0.50
    nli_entail_floor: float = 0.50
    # Optional multinomial logistic regression calibration (plan v1 Phase 3).
    # If `use_lr_calibration` is True AND lr_coefs/lr_intercepts are populated,
    # score_and_route replaces the weighted-sum + threshold decision with a
    # softmax over class logits. Features assumed order: [top1_dense,
    # rank_overlap, nli_entail]. Classes order is given by lr_classes.
    use_lr_calibration: bool = False
    lr_classes: list[str] = Field(default_factory=lambda: ["confident", "hedged", "refused"])
    lr_intercepts: list[float] = Field(default_factory=list)
    lr_coefs: list[list[float]] = Field(default_factory=list)
    # v0.4: prediction-confidence gate (repurposed from the v0.1 top1_dense floor).
    # When LR mode is active AND the winning class probability is below this
    # threshold, refuse instead of committing to the LR's borderline guess.
    # This catches the case where LR routing is itself uncertain (e.g.
    # P(confident)=0.42, P(hedged)=0.38, P(refused)=0.20 -> argmax says
    # confident but the model is barely sure).
    # Set to 0.0 to disable. Only applies when use_lr_calibration is True.
    lr_min_prediction_confidence: float = 0.50


class GenerationSettings(BaseModel):
    ollama_format_json: bool = True
    max_tokens: int = 1024
    temperature: float = 0.2
    keep_alive: str = "5m"
    enable_thinking: bool = False  # Qwen 3 / 3.5 thinking mode — off for RAG


class FaithfulnessSettings(BaseModel):
    per_claim_entail_floor: float = 0.50
    reroute_to_hedged_on_failure: bool = True


class MemorySettings(BaseModel):
    ram_budget_mb: int = 8000
    embedder_device: Literal["cpu", "cuda"] = "cpu"
    nli_device: Literal["cpu", "cuda"] = "cpu"


class PathSettings(BaseModel):
    raw_pdfs_dir: str = "data/raw_pdfs"
    chunks_dir: str = "data/chunks"
    indexes_dir: str = "data/indexes"
    hf_cache_dir: str = ".hf_cache"

    def resolve(self, key: str) -> Path:
        return REPO_ROOT / getattr(self, key)


class AppSettings(BaseModel):
    default_llm: str
    default_embedder: str
    default_nli: str
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    confidence: ConfidenceSettings = Field(default_factory=ConfidenceSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    faithfulness: FaithfulnessSettings = Field(default_factory=FaithfulnessSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class LLMSpec(BaseModel):
    id: str
    display_name: str
    backend: Literal["ollama"]
    ollama_tag: str
    ram_estimate_mb: int
    notes: str | None = None


class EmbedderSpec(BaseModel):
    id: str
    display_name: str
    backend: Literal["sentence_transformers"]
    hf_id: str
    dim: int
    ram_estimate_mb: int
    query_prefix: str = ""
    passage_prefix: str = ""
    notes: str | None = None


class NLISpec(BaseModel):
    id: str
    display_name: str
    backend: Literal["hf_transformers"]
    hf_id: str
    ram_estimate_mb: int
    notes: str | None = None


class ModelCatalog(BaseModel):
    llms: list[LLMSpec]
    embedders: list[EmbedderSpec]
    nli: list[NLISpec]

    def llm_by_id(self, id_: str) -> LLMSpec:
        for spec in self.llms:
            if spec.id == id_:
                return spec
        raise KeyError(f"Unknown LLM id: {id_}")

    def embedder_by_id(self, id_: str) -> EmbedderSpec:
        for spec in self.embedders:
            if spec.id == id_:
                return spec
        raise KeyError(f"Unknown embedder id: {id_}")

    def nli_by_id(self, id_: str) -> NLISpec:
        for spec in self.nli:
            if spec.id == id_:
                return spec
        raise KeyError(f"Unknown NLI id: {id_}")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_settings(path: Path | None = None) -> AppSettings:
    p = path or (CONFIG_DIR / "default.yaml")
    return AppSettings.model_validate(_load_yaml(p))


def load_catalog(path: Path | None = None) -> ModelCatalog:
    p = path or (CONFIG_DIR / "models.yaml")
    return ModelCatalog.model_validate(_load_yaml(p))


# Module-level singletons. Re-import after editing YAML to pick up changes.
settings: AppSettings = load_settings()
model_catalog: ModelCatalog = load_catalog()
