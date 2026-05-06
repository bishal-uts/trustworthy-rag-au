"""Catalog accessors. Thin wrapper around the singletons in src.config."""

from __future__ import annotations

from src.config import EmbedderSpec, LLMSpec, NLISpec, model_catalog


def list_llms() -> list[LLMSpec]:
    return list(model_catalog.llms)


def list_embedders() -> list[EmbedderSpec]:
    return list(model_catalog.embedders)


def list_nli() -> list[NLISpec]:
    return list(model_catalog.nli)


def llm_choices() -> list[tuple[str, str]]:
    """List of (display_name, id) suitable for Gradio dropdown choices."""
    return [(s.display_name, s.id) for s in model_catalog.llms]


def embedder_choices() -> list[tuple[str, str]]:
    return [(s.display_name, s.id) for s in model_catalog.embedders]


def nli_choices() -> list[tuple[str, str]]:
    return [(s.display_name, s.id) for s in model_catalog.nli]


def get_llm_spec(id_: str) -> LLMSpec:
    return model_catalog.llm_by_id(id_)


def get_embedder_spec(id_: str) -> EmbedderSpec:
    return model_catalog.embedder_by_id(id_)


def get_nli_spec(id_: str) -> NLISpec:
    return model_catalog.nli_by_id(id_)
