"""Gradio UI — single-tab v0.1.

Layout:
 - Sticky model bar (LLM / embedder / NLI dropdowns)
 - Two-column body: query + output
 - Retrieved-context table (always visible)
 - Click-to-expand chunk detail
 - Collapsible faithfulness panel
 - Collapsible raw JSON panel

Concurrency limit = 1 so two model loads can't fight for RAM.

Run:
    python -m ui.gradio_app
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.config import settings
from src.models.manager import manager
from src.pipeline import run as run_pipeline
from src.schemas import RAGOutput
from ui.components import (
    embedder_dropdown_choices,
    format_chunk_detail,
    format_faithfulness_panel,
    format_output_panel,
    format_retrieval_table,
    llm_dropdown_choices,
    nli_dropdown_choices,
    status_banner,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CSS = """
#status-banner { font-size: 0.85em; color: #555; }
#retrieval-table table { font-size: 0.88em; }
"""


def _empty_output() -> RAGOutput:
    from src.schemas import ConfidenceReport, ConfidenceSignals

    return RAGOutput(
        state="refused",
        confidence=ConfidenceReport(value=0.0, signals=ConfidenceSignals()),
    )


def submit_query(
    question: str,
    llm_id: str,
    embedder_id: str,
    nli_id: str,
    top_k: int,
    progress: gr.Progress = gr.Progress(),
):
    if not question or not question.strip():
        empty = _empty_output()
        return (
            "_Enter a question and click Submit._",
            [],
            "_No retrieval yet._",
            "_No faithfulness check yet._",
            "{}",
            empty,
            status_banner(embedder_id),
        )

    progress(0.1, desc="Loading indexes")
    progress(0.4, desc="Retrieving + scoring")
    out = run_pipeline(
        question=question,
        llm_id=llm_id,
        embedder_id=embedder_id,
        nli_id=nli_id,
        top_k=int(top_k),
    )
    progress(1.0, desc="Done")

    return (
        format_output_panel(out),
        format_retrieval_table(out),
        "_Click a row in the retrieval table to see the full chunk text._",
        format_faithfulness_panel(out),
        json.dumps(out.model_dump(), indent=2, ensure_ascii=False),
        out,  # state
        status_banner(embedder_id),
    )


def show_chunk(out: RAGOutput | None, evt: gr.SelectData) -> str:
    if out is None:
        return "_No retrieval yet._"
    return format_chunk_detail(out, evt.index[0])


def free_memory() -> str:
    manager.unload_all()
    s = manager.status()
    return f"Memory cleared. RAM used: {s['ram_used_mb']} MB / {s['ram_budget_mb']} MB"


def refresh_status(embedder_id: str) -> str:
    return status_banner(embedder_id)


def load_models(
    llm_id: str,
    embedder_id: str,
    nli_id: str,
    progress: gr.Progress = gr.Progress(),
) -> str:
    """Pre-warm all three models so the next query doesn't pay cold-load tax."""
    progress(0.05, desc="Loading embedder")
    manager.get_embedder(embedder_id)

    progress(0.35, desc="Loading NLI")
    manager.get_nli(nli_id)

    progress(0.65, desc="Warming Ollama LLM")
    llm = manager.get_llm(llm_id)
    # Tiny generate call to make Ollama actually load the model into memory.
    # 1-token output is enough to trigger the load.
    try:
        llm.generate("hi")
    except Exception as e:
        return f"⚠️ Embedder + NLI loaded, but Ollama warm-up failed: {e}"

    progress(1.0, desc="Done")
    s = manager.status()
    return (
        f"✅ Models warm. Embedder + NLI: {s['ram_used_mb']} MB. "
        f"LLM `{llm_id}` loaded in Ollama. Next query should be fast (~15-30s)."
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Trustworthy RAG (basic v0.1)", css=CSS) as app:
        gr.Markdown(
            "# Trustworthy RAG · basic v0.1\n"
            "Australian Financial Regulation. Local-only inference. "
            "Switch models on the fly to test retrieval and generation behaviour."
        )

        # Model bar
        with gr.Row():
            llm_dd = gr.Dropdown(
                choices=llm_dropdown_choices(),
                value=settings.default_llm,
                label="LLM",
                scale=2,
            )
            embedder_dd = gr.Dropdown(
                choices=embedder_dropdown_choices(),
                value=settings.default_embedder,
                label="Embedder",
                scale=2,
            )
            nli_dd = gr.Dropdown(
                choices=nli_dropdown_choices(),
                value=settings.default_nli,
                label="NLI",
                scale=2,
            )
            top_k_in = gr.Number(value=settings.retrieval.top_k, label="Top-k", precision=0, scale=1)

        with gr.Row():
            load_btn = gr.Button("Load models", variant="secondary", scale=1)
            free_btn = gr.Button("Free memory", scale=1)
            refresh_btn = gr.Button("Refresh status", scale=1)
            status_md = gr.Markdown(status_banner(settings.default_embedder), elem_id="status-banner")

        # Two-column body
        with gr.Row():
            with gr.Column(scale=1):
                question_in = gr.Textbox(
                    label="Question",
                    lines=4,
                    placeholder="e.g. Within how many hours must an APRA-regulated entity notify APRA of a material information security incident?",
                )
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                output_md = gr.Markdown("_Enter a question and click Submit._")

        # Retrieval table (always visible)
        gr.Markdown("### Retrieved context")
        retrieval_df = gr.Dataframe(
            headers=["rank", "doc", "section", "¶", "bm25", "dense", "rrf", "preview"],
            datatype=["number", "str", "str", "str", "number", "number", "number", "str"],
            interactive=False,
            wrap=True,
            elem_id="retrieval-table",
        )
        chunk_detail_md = gr.Markdown(
            "_Click a row in the retrieval table to see the full chunk text._"
        )

        # Faithfulness + raw JSON (collapsed)
        with gr.Accordion("Faithfulness / NLI breakdown", open=False):
            faith_md = gr.Markdown("_No faithfulness check yet._")
        with gr.Accordion("Raw JSON output", open=False):
            json_box = gr.Code(value="{}", language="json")

        # State
        out_state = gr.State(value=None)

        # Wiring
        submit_btn.click(
            submit_query,
            inputs=[question_in, llm_dd, embedder_dd, nli_dd, top_k_in],
            outputs=[
                output_md,
                retrieval_df,
                chunk_detail_md,
                faith_md,
                json_box,
                out_state,
                status_md,
            ],
            concurrency_limit=1,
        )

        retrieval_df.select(show_chunk, inputs=[out_state], outputs=[chunk_detail_md])
        load_btn.click(
            load_models,
            inputs=[llm_dd, embedder_dd, nli_dd],
            outputs=[status_md],
            concurrency_limit=1,
        )
        free_btn.click(free_memory, inputs=[], outputs=[status_md])
        refresh_btn.click(refresh_status, inputs=[embedder_dd], outputs=[status_md])
        embedder_dd.change(refresh_status, inputs=[embedder_dd], outputs=[status_md])

    return app


def main() -> None:
    app = build_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
