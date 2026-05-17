"""End-to-end comparison runner.

Runs every selected baseline on every benchmark question, applies the
judge to assign correctness labels, computes comparison metrics, and
writes a JSON results file + prints a Rich table.

Usage:
    # Default: all five baselines, default models, save to eval/results/<timestamp>.json
    python -m eval.comparison_eval

    # Subset of baselines (faster smoke run)
    python -m eval.comparison_eval --baselines hybrid,trustworthy

    # Use a different LLM as judge (avoid self-favourability)
    python -m eval.comparison_eval --judge-llm deepseek-r1-8b

    # Keyword-only judging (no LLM judge call)
    python -m eval.comparison_eval --no-llm-judge

    # Per-category breakdown
    python -m eval.comparison_eval --per-category

The output JSON contains: configuration, per-question raw outputs, per-system
summary metrics, optional per-category breakdown. Plot scripts read this.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table  # noqa: E402

from eval.baselines import build_baseline, list_baseline_names  # noqa: E402
from eval.comparison_metrics import (  # noqa: E402
    BaselineOutput,
    ComparisonRow,
    LossMatrix,
    summarise,
    summarise_by_category,
)
from eval.llm_judge import (  # noqa: E402
    CompositeJudge,
    JudgeInput,
    KeywordJudge,
    LLMJudge,
    NLIJudge,
)
from src.config import settings  # noqa: E402
from src.faithfulness import check as faithfulness_check  # noqa: E402
from src.schemas import Chunk, Citation  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = REPO_ROOT / "eval" / "benchmark.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_benchmark() -> list[dict]:
    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def _populate_faithfulness(out: BaselineOutput, nli_id: str) -> None:
    """For baselines that didn't run faithfulness internally (everything except
    `trustworthy`), run it here so the hallucination metric works uniformly.

    Skipped for refused / no-answer outputs.
    """
    if out.state == "refused" or not out.answer:
        return
    if out.total_claim_count > 0:
        return  # trustworthy already filled it in
    citations = [Citation(**c) for c in out.citations]
    chunks = [Chunk(**c) for c in out.retrieved_chunks]
    if not chunks:
        # No retrieved context to faith-check against (e.g. no_rag baseline).
        # Mark every sentence in the answer as unfaithful by construction:
        # the model is using parametric knowledge, not the corpus.
        sentence_count = max(1, out.answer.count(".") + out.answer.count("?") + out.answer.count("!"))
        out.total_claim_count = sentence_count
        out.unfaithful_claim_count = sentence_count
        return
    report = faithfulness_check(
        answer=out.answer,
        citations=citations,
        retrieved_chunks=chunks,
        nli_id=nli_id,
    )
    out.total_claim_count = len(report.claims)
    out.unfaithful_claim_count = report.unfaithful_count


def _judge_correctness(out: BaselineOutput, q: dict, judge: CompositeJudge) -> None:
    if out.state == "refused" or not out.answer:
        # Refusals don't get judged for correctness — they have no answer.
        # Their correctness is captured by `refusal_precision`.
        return
    inp = JudgeInput(
        question=q["question"],
        expected_answer=q.get("expected_answer"),
        expected_keywords=q.get("expected_answer_keywords") or [],
        actual_answer=out.answer,
    )
    res = judge.judge(inp)
    out.correctness = res.label


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_summary(rows: list[ComparisonRow], title: str) -> None:
    table = Table(title=title)
    table.add_column("System")
    table.add_column("n", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Acc/Ans", justify="right")
    table.add_column("Eff.Acc", justify="right")
    table.add_column("Halluc", justify="right")
    table.add_column("RefP", justify="right")
    table.add_column("Risk Loss", justify="right", style="bold")
    table.add_column("ms/q", justify="right")
    for r in rows:
        table.add_row(
            r.system,
            str(r.n_total),
            f"{r.coverage:.2f}",
            f"{r.accuracy_on_answered:.2f}",
            f"{r.effective_accuracy:.2f}",
            f"{r.hallucination_rate:.2f}",
            f"{r.refusal_precision:.2f}",
            f"{r.risk_adjusted_loss:.1f}",
            f"{r.mean_elapsed_ms:.0f}",
        )
    console.print(table)


def _render_per_category(
    name: str,
    per_cat: dict[str, ComparisonRow],
) -> None:
    table = Table(title=f"Per-category · {name}")
    table.add_column("Category")
    table.add_column("n", justify="right")
    table.add_column("Cov", justify="right")
    table.add_column("Acc/Ans", justify="right")
    table.add_column("Halluc", justify="right")
    table.add_column("Risk", justify="right")
    for cat, r in sorted(per_cat.items()):
        table.add_row(
            cat,
            str(r.n_total),
            f"{r.coverage:.2f}",
            f"{r.accuracy_on_answered:.2f}",
            f"{r.hallucination_rate:.2f}",
            f"{r.risk_adjusted_loss:.1f}",
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Comparative eval: trustworthy vs baseline RAGs.")
    parser.add_argument(
        "--baselines",
        default=",".join(list_baseline_names()),
        help=f"Comma-separated baseline names. Default: all. Available: {list_baseline_names()}",
    )
    parser.add_argument("--llm", default=None, help="LLM under test (default: settings.default_llm)")
    parser.add_argument("--embedder", default=None, help="Embedder (default: settings.default_embedder)")
    parser.add_argument("--nli", default=None, help="NLI model (default: settings.default_nli)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k retrieval (default: 10)")
    parser.add_argument(
        "--judge-llm",
        default=None,
        help="LLM id for LLM-as-judge. SHOULD be different from --llm. "
             "If unset, uses --llm (with bias warning).",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge. Use only keyword + NLI judges (cheaper).",
    )
    parser.add_argument(
        "--no-nli-judge",
        action="store_true",
        help="Skip NLI judge (only keyword + optional LLM).",
    )
    parser.add_argument("--per-category", action="store_true", help="Print per-category breakdown")
    parser.add_argument("--strict-correct", action="store_true", help="Only YES counts as correct (drop PARTIAL)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Default: eval/results/comparison_<timestamp>.json",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark to first N questions (debug)")
    parser.add_argument(
        "--loss-wrong-confident", type=float, default=10.0,
        help="Cost of a wrong confident answer (default 10)",
    )
    parser.add_argument(
        "--loss-refused-answerable", type=float, default=1.0,
        help="Cost of refusing an answerable question (default 1)",
    )
    args = parser.parse_args()

    # --- Resolve config ---
    llm_id = args.llm or settings.default_llm
    embedder_id = args.embedder or settings.default_embedder
    nli_id = args.nli or settings.default_nli
    top_k = args.top_k or settings.retrieval.top_k
    judge_llm_id = args.judge_llm or (None if args.no_llm_judge else llm_id)
    if judge_llm_id == llm_id and not args.no_llm_judge:
        console.print(
            "[yellow]⚠ Judge LLM == system LLM. Self-favourability bias possible. "
            "Pass --judge-llm <other_id> to avoid.[/yellow]"
        )

    baseline_names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    for n in baseline_names:
        if n not in list_baseline_names():
            console.print(f"[red]Unknown baseline: {n}. Known: {list_baseline_names()}[/red]")
            return 2

    questions = load_benchmark()
    if args.limit:
        questions = questions[: args.limit]

    expected_states = {q["id"]: q["expected_state"] for q in questions}
    categories = {q["id"]: q.get("category", "unknown") for q in questions}

    loss = LossMatrix(
        wrong_confident=args.loss_wrong_confident,
        refused_answerable=args.loss_refused_answerable,
    )

    # --- Build judge ---
    sub_judges = [KeywordJudge()]
    if not args.no_nli_judge:
        sub_judges.append(NLIJudge(nli_id=nli_id))
    if judge_llm_id:
        sub_judges.append(LLMJudge(judge_llm_id=judge_llm_id))
    judge = CompositeJudge(sub_judges)

    console.print(f"[bold]Comparing {len(baseline_names)} baselines on {len(questions)} questions[/bold]")
    console.print(
        f"  LLM={llm_id}  embedder={embedder_id}  NLI={nli_id}  top_k={top_k}"
        f"  judge_llm={judge_llm_id or 'none'}\n"
    )

    # --- Run each baseline on every question ---
    all_outputs: dict[str, list[BaselineOutput]] = {n: [] for n in baseline_names}
    t_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        for name in baseline_names:
            try:
                baseline = build_baseline(
                    name=name,  # type: ignore[arg-type]
                    llm_id=llm_id,
                    embedder_id=embedder_id,
                    nli_id=nli_id,
                    top_k=top_k,
                )
            except Exception as e:
                console.print(f"[red]Cannot build {name}: {type(e).__name__}: {e}[/red]")
                continue

            task = progress.add_task(f"{name}", total=len(questions))
            for q in questions:
                try:
                    out = baseline.run(q["id"], q["question"])
                except Exception as e:
                    console.print(f"[red]{name}/{q['id']} exception: {type(e).__name__}: {e}[/red]")
                    out = BaselineOutput(
                        question_id=q["id"],
                        state="refused",  # treat exception as refusal
                        answer=None,
                    )
                # Faith check (uniformly across baselines)
                _populate_faithfulness(out, nli_id=nli_id)
                # Judge correctness
                _judge_correctness(out, q, judge)
                all_outputs[name].append(out)
                progress.update(task, advance=1)

    elapsed_total = time.time() - t_start

    # --- Aggregate ---
    rows: list[ComparisonRow] = []
    per_category_data: dict[str, dict[str, ComparisonRow]] = {}
    for name in baseline_names:
        outs = all_outputs[name]
        if not outs:
            continue
        row = summarise(name, outs, expected_states, loss, strict=args.strict_correct)
        rows.append(row)
        per_category_data[name] = summarise_by_category(
            name, outs, categories, expected_states, loss, strict=args.strict_correct
        )

    # --- Render ---
    title = (
        f"Comparison · n={len(questions)} · k={top_k} · "
        f"LLM={llm_id} · strict={args.strict_correct}"
    )
    _render_summary(rows, title)
    if args.per_category:
        for name in baseline_names:
            if name in per_category_data:
                _render_per_category(name, per_category_data[name])

    console.print(f"\n[dim]Total elapsed: {elapsed_total:.1f}s[/dim]")

    # --- Save JSON ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    payload = {
        "config": {
            "llm": llm_id,
            "embedder": embedder_id,
            "nli": nli_id,
            "top_k": top_k,
            "judge_llm": judge_llm_id,
            "strict_correct": args.strict_correct,
            "loss_matrix": asdict(loss),
            "n_questions": len(questions),
            "baselines": baseline_names,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": [asdict(r) for r in rows],
        "per_category": {
            name: {cat: asdict(r) for cat, r in cats.items()}
            for name, cats in per_category_data.items()
        },
        "per_question": {
            name: [asdict(o) for o in outs] for name, outs in all_outputs.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]Results saved: {out_path.relative_to(REPO_ROOT)}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
