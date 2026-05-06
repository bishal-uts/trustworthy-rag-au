"""Helper functions used by ui/gradio_app.py.

Kept separate so the main app file stays focused on layout + event wiring.
Functions in this module return data structures or formatted strings —
they don't construct Gradio components themselves.
"""

from __future__ import annotations

from src.models.llm import OllamaLLM
from src.models.registry import (
    embedder_choices,
    llm_choices,
    nli_choices,
)
from src.retrieval.dense import DenseIndex
from src.schemas import RAGOutput

STATE_BADGE = {
    "confident": "🟢 CONFIDENT",
    "hedged": "🟡 HEDGED",
    "refused": "🔴 REFUSED",
}


def status_banner(embedder_id: str) -> str:
    """One-line markdown status: Ollama health + index availability."""
    parts: list[str] = []
    if OllamaLLM.health_check():
        parts.append("✅ Ollama: running")
    else:
        parts.append("❌ Ollama: not running — start with `ollama serve`")

    if DenseIndex.exists(embedder_id):
        parts.append(f"✅ Index: `{embedder_id}` ready")
    else:
        parts.append(
            f"⚠️ Index for `{embedder_id}` not built — run "
            f"`python scripts/build_index.py --embedder {embedder_id}`"
        )
    return " · ".join(parts)


def llm_dropdown_choices() -> list[tuple[str, str]]:
    return llm_choices()


def embedder_dropdown_choices() -> list[tuple[str, str]]:
    return embedder_choices()


def nli_dropdown_choices() -> list[tuple[str, str]]:
    return nli_choices()


def format_output_panel(out: RAGOutput) -> str:
    """Markdown for the answer/state/confidence panel."""
    badge = STATE_BADGE.get(out.state, out.state.upper())
    lines = [
        f"### {badge}",
        f"**Confidence:** {out.confidence.value:.2f} "
        f"(top1_dense={out.confidence.signals.retrieval_top1_dense:.2f}, "
        f"rank_overlap={out.confidence.signals.rank_overlap:.2f}, "
        f"nli_entail={out.confidence.signals.nli_entail:.2f})",
        f"**LLM called:** {'yes' if out.llm_called else 'no'} · "
        f"**elapsed:** {out.elapsed_ms} ms",
        "",
    ]
    if out.answer:
        lines.append("**Answer:**")
        lines.append(out.answer)
    elif out.evidence.refused_reason:
        lines.append(f"**Refused.** {out.evidence.refused_reason}")
    if out.citations:
        lines.append("")
        lines.append("**Citations:**")
        for c in out.citations:
            line = f"- `{c.doc}` · {c.section} · ¶{c.paragraph}"
            if c.span:
                line += f" — \"{c.span[:120]}{'…' if len(c.span) > 120 else ''}\""
            lines.append(line)
    return "\n".join(lines)


def format_retrieval_table(out: RAGOutput) -> list[list]:
    """Rows for gr.Dataframe: rank, doc, section, ¶, bm25, dense, rrf, preview."""
    rows: list[list] = []
    for ch in out.evidence.retrieved_chunks:
        preview = ch.text.replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:197] + "…"
        rows.append(
            [
                ch.rank,
                ch.doc,
                ch.section,
                ch.paragraph,
                round(ch.bm25_score, 3),
                round(ch.dense_score, 3),
                round(ch.rrf_score, 4),
                preview,
            ]
        )
    return rows


def format_chunk_detail(out: RAGOutput, row_index: int) -> str:
    if row_index < 0 or row_index >= len(out.evidence.retrieved_chunks):
        return "_Click a row in the retrieval table to see the full chunk text._"
    ch = out.evidence.retrieved_chunks[row_index]
    return (
        f"**`{ch.doc}` · {ch.section} · ¶{ch.paragraph}**  \n"
        f"_chunk_id_: `{ch.chunk_id}` · _bm25_: {ch.bm25_score:.3f} · "
        f"_dense_: {ch.dense_score:.3f} · _rrf_: {ch.rrf_score:.4f}\n\n"
        f"```\n{ch.text}\n```"
    )


def format_faithfulness_panel(out: RAGOutput) -> str:
    if not out.faithfulness.claims:
        return "_No faithfulness check (refused or no answer)._"
    lines = [
        f"**Unfaithful claims: {out.faithfulness.unfaithful_count} / {len(out.faithfulness.claims)}**",
        "",
    ]
    for i, claim in enumerate(out.faithfulness.claims, 1):
        ent_icon = "✅" if claim.entailed else "⚠️"
        lines.append(
            f"{i}. {ent_icon} entail={claim.nli_score:.2f}  · {claim.text}"
        )
    return "\n".join(lines)
