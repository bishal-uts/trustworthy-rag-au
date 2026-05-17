"""Benchmark runner.

Runs every question in eval/benchmark.yaml through the pipeline (with any
ablation flags) and produces a folder of outputs under eval/results/<run_id>/:

  per_question.jsonl   one JSON line per question (full RAGOutput + grading)
  metrics.json         aggregate metrics from eval/metrics.aggregate
  summary.md           human-readable summary table (paste into the report)
  metrics.csv          flat per-question rows for plotting / spreadsheet work
  config.json          the exact pipeline kwargs used for this run

Usage:
    python scripts/run_benchmark.py --run-id baseline
    python scripts/run_benchmark.py --run-id bm25_only --retrieval-mode bm25_only
    python scripts/run_benchmark.py --run-id no_faithfulness --no-faithfulness
    python scripts/run_benchmark.py --run-id quick --limit 5     # iterate fast
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Make `src` and `eval` importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval.metrics import (  # noqa: E402
    Aggregate,
    QuestionResult,
    aggregate,
)
from eval.plot_benchmark import plot_run  # noqa: E402
from src.pipeline import run as run_pipeline  # noqa: E402
from src.schemas import RAGOutput  # noqa: E402

BENCHMARK_FILE = REPO_ROOT / "eval" / "benchmark.yaml"
RESULTS_ROOT = REPO_ROOT / "eval" / "results"

console = Console()


# -------- Loading + validation --------

REQUIRED_FIELDS = {"id", "category", "question", "expected_state"}
VALID_STATES = {"confident", "hedged", "refused"}


def load_benchmark(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    questions = data.get("questions") or []
    if not questions:
        raise ValueError(f"No questions found in {path}")
    seen_ids: set[str] = set()
    for q in questions:
        missing = REQUIRED_FIELDS - set(q.keys())
        if missing:
            raise ValueError(f"Question {q.get('id', '?')} missing fields: {missing}")
        if q["expected_state"] not in VALID_STATES:
            raise ValueError(
                f"Question {q['id']}: expected_state {q['expected_state']!r} "
                f"must be one of {VALID_STATES}"
            )
        if q["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {q['id']}")
        seen_ids.add(q["id"])
    return questions


# -------- Grading: RAGOutput -> QuestionResult --------

def grade(q: dict, out: RAGOutput, elapsed_ms: int) -> QuestionResult:
    expected_doc_ids = sorted({c["doc"] for c in (q.get("expected_citations") or [])})
    cited_doc_ids = sorted({c.doc for c in out.citations})
    retrieved_doc_ids = [c.doc for c in out.evidence.retrieved_chunks]
    has_faith = out.state != "refused" and bool(out.faithfulness.claims)
    return QuestionResult(
        qid=q["id"],
        category=q["category"],
        expected_state=q["expected_state"],
        expected_doc_ids=expected_doc_ids,
        predicted_state=out.state,
        cited_doc_ids=cited_doc_ids,
        retrieved_doc_ids=retrieved_doc_ids,
        confidence_value=out.confidence.value,
        unfaithful_count=out.faithfulness.unfaithful_count,
        has_faithfulness_report=has_faith,
        llm_called=out.llm_called,
        elapsed_ms=elapsed_ms,
    )


def error_result(q: dict, exc: Exception) -> QuestionResult:
    return QuestionResult(
        qid=q["id"],
        category=q["category"],
        expected_state=q["expected_state"],
        expected_doc_ids=sorted({c["doc"] for c in (q.get("expected_citations") or [])}),
        predicted_state="ERROR",
        cited_doc_ids=[],
        retrieved_doc_ids=[],
        confidence_value=0.0,
        unfaithful_count=0,
        has_faithfulness_report=False,
        llm_called=False,
        elapsed_ms=0,
        error=f"{type(exc).__name__}: {exc}",
    )


# -------- Output writers --------

def _confusion_matrix_md(cm: dict[str, dict[str, int]]) -> str:
    states = ["confident", "hedged", "refused"]
    header = "| expected \\ predicted | " + " | ".join(states) + " |"
    sep = "|" + "---|" * (len(states) + 1)
    rows = []
    for e in states:
        rows.append("| " + e + " | " + " | ".join(str(cm[e][p]) for p in states) + " |")
    return "\n".join([header, sep, *rows])


def write_summary_md(
    path: Path, run_id: str, config: dict, agg: Aggregate, n_questions: int
) -> None:
    rr = agg.retrieval_recall
    lat = agg.latency
    refusal = agg.refusal
    lines = [
        f"# Benchmark run: `{run_id}`",
        "",
        f"- Questions: **{n_questions}** ({agg.n_errors} pipeline errors)",
        f"- Config: `{json.dumps(config, sort_keys=True)}`",
        "",
        "## Headline metrics",
        "",
        f"- **State accuracy:** {agg.state_accuracy:.1%}",
        f"- **Citation correctness (confident only):** {agg.citation_correctness:.1%}",
        f"- **Faithfulness rate (answered only):** {agg.faithfulness_rate:.1%}",
        f"- **Refusal P/R/F1:** {refusal['precision']:.1%} / {refusal['recall']:.1%} / {refusal['f1']:.1%}"
        f" (tp={int(refusal['tp'])}, fp={int(refusal['fp'])}, fn={int(refusal['fn'])})",
        f"- **Latency:** mean {lat['mean_ms']:.0f} ms · p50 {lat['p50_ms']:.0f} · p95 {lat['p95_ms']:.0f}"
        f" · LLM-call rate {lat['llm_call_rate']:.1%}",
        "",
        "## Retrieval Recall@K (answerable questions only)",
        "",
        "| K | Recall |",
        "|---|---|",
        *[f"| {k} | {v:.1%} |" for k, v in sorted(rr.items())],
        "",
        "## State confusion matrix",
        "",
        _confusion_matrix_md(agg.state_confusion_matrix),
        "",
        "## Per-category state accuracy",
        "",
        "| Category | Accuracy |",
        "|---|---|",
        *[f"| {cat} | {acc:.1%} |" for cat, acc in agg.per_category_accuracy.items()],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_per_question_jsonl(path: Path, results: list[QuestionResult], outputs: list[RAGOutput | None]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r, out in zip(results, outputs):
            payload: dict = asdict(r)
            if out is not None:
                payload["full_output"] = out.model_dump()
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_metrics_csv(path: Path, results: list[QuestionResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "qid", "category", "expected_state", "predicted_state",
            "state_correct", "confidence_value",
            "expected_docs", "cited_docs", "retrieved_top3_docs",
            "unfaithful_count", "llm_called", "elapsed_ms", "error",
        ])
        for r in results:
            w.writerow([
                r.qid,
                r.category,
                r.expected_state,
                r.predicted_state,
                int(r.predicted_state == r.expected_state),
                f"{r.confidence_value:.4f}",
                "|".join(r.expected_doc_ids),
                "|".join(r.cited_doc_ids),
                "|".join(r.retrieved_doc_ids[:3]),
                r.unfaithful_count,
                int(r.llm_called),
                r.elapsed_ms,
                r.error or "",
            ])


def write_metrics_json(path: Path, agg: Aggregate) -> None:
    path.write_text(json.dumps(asdict(agg), indent=2), encoding="utf-8")


def write_config_json(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


# -------- Main --------

def run_benchmark(
    questions: list[dict],
    *,
    llm_id: str | None,
    embedder_id: str | None,
    nli_id: str | None,
    top_k: int | None,
    retrieval_mode: str,
    enable_faithfulness: bool,
    enable_floor_gate: bool,
    enable_routing: bool = True,
    limit: int | None = None,
) -> tuple[list[QuestionResult], list[RAGOutput | None]]:
    results: list[QuestionResult] = []
    outputs: list[RAGOutput | None] = []
    iter_qs = questions[:limit] if limit else questions
    for i, q in enumerate(iter_qs, start=1):
        console.rule(f"[bold cyan]{i}/{len(iter_qs)}[/bold cyan] {q['id']} · {q['category']}")
        console.print(q["question"])
        t0 = time.time()
        try:
            out = run_pipeline(
                question=q["question"],
                llm_id=llm_id,
                embedder_id=embedder_id,
                nli_id=nli_id,
                top_k=top_k,
                retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
                enable_faithfulness=enable_faithfulness,
                enable_floor_gate=enable_floor_gate,
                enable_routing=enable_routing,
            )
            elapsed = int((time.time() - t0) * 1000)
            r = grade(q, out, elapsed_ms=elapsed)
            outputs.append(out)
            colour = "green" if r.predicted_state == r.expected_state else "yellow"
            console.print(
                f"[{colour}]{r.predicted_state}[/{colour}] (expected {r.expected_state}) · "
                f"conf={r.confidence_value:.2f} · llm={r.llm_called} · {elapsed}ms"
            )
        except Exception as e:
            r = error_result(q, e)
            outputs.append(None)
            console.print(f"[red]ERROR:[/red] {e}")
        results.append(r)
    return results, outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full benchmark and write metrics.")
    parser.add_argument("--run-id", required=True, help="Output folder name under eval/results/")
    parser.add_argument("--benchmark", default=str(BENCHMARK_FILE), help="Benchmark YAML path")
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT), help="Results root dir")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions")

    parser.add_argument("--llm", default=None, help="LLM id")
    parser.add_argument("--embedder", default=None, help="Embedder id")
    parser.add_argument("--nli", default=None, help="NLI id")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k retrieval")

    parser.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25_only", "dense_only"],
        default="hybrid",
    )
    parser.add_argument("--no-faithfulness", action="store_true")
    parser.add_argument("--no-floor-gate", action="store_true")
    parser.add_argument(
        "--no-plots", action="store_true", help="Skip writing the PNG plot panel"
    )

    args = parser.parse_args()

    questions = load_benchmark(Path(args.benchmark))
    console.print(f"[bold]Loaded {len(questions)} questions from {args.benchmark}[/bold]")

    config = {
        "run_id": args.run_id,
        "benchmark": args.benchmark,
        "llm": args.llm,
        "embedder": args.embedder,
        "nli": args.nli,
        "top_k": args.top_k,
        "retrieval_mode": args.retrieval_mode,
        "enable_faithfulness": not args.no_faithfulness,
        "enable_floor_gate": not args.no_floor_gate,
        "limit": args.limit,
    }

    results, outputs = run_benchmark(
        questions,
        llm_id=args.llm,
        embedder_id=args.embedder,
        nli_id=args.nli,
        top_k=args.top_k,
        retrieval_mode=args.retrieval_mode,
        enable_faithfulness=not args.no_faithfulness,
        enable_floor_gate=not args.no_floor_gate,
        limit=args.limit,
    )
    agg = aggregate(results)

    out_dir = Path(args.output_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_question_jsonl(out_dir / "per_question.jsonl", results, outputs)
    write_metrics_json(out_dir / "metrics.json", agg)
    write_metrics_csv(out_dir / "metrics.csv", results)
    write_summary_md(out_dir / "summary.md", args.run_id, config, agg, len(results))
    write_config_json(out_dir / "config.json", config)

    if not args.no_plots:
        # Pass results as dicts (already in per_question format) and agg as dict
        results_dicts = [asdict(r) for r in results]
        plot_run(results_dicts, asdict(agg), args.run_id, out_dir / "plots")

    # Console summary
    table = Table(title=f"Run {args.run_id} · headline metrics")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("state accuracy", f"{agg.state_accuracy:.1%}")
    table.add_row("citation correctness (conf)", f"{agg.citation_correctness:.1%}")
    table.add_row("faithfulness rate (answered)", f"{agg.faithfulness_rate:.1%}")
    table.add_row(
        "refusal P/R/F1",
        f"{agg.refusal['precision']:.1%} / {agg.refusal['recall']:.1%} / {agg.refusal['f1']:.1%}",
    )
    for k, v in sorted(agg.retrieval_recall.items()):
        table.add_row(f"retrieval recall@{k}", f"{v:.1%}")
    table.add_row("latency mean ms", f"{agg.latency['mean_ms']:.0f}")
    table.add_row("LLM call rate", f"{agg.latency['llm_call_rate']:.1%}")
    console.print(table)
    console.print(f"\n[bold green]Wrote outputs to {out_dir}[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
