"""End-to-end smoke test.

Runs the five original golden questions (id prefix `smoke_*`) from
eval/benchmark.yaml through the pipeline and asserts:
 - The state matches expected_state.
 - For confident answers: at least one expected citation doc appears in the
   actual citations OR top-3 retrieved chunks (loose match — paragraph
   numbering may differ from the benchmark's manually-recorded value until
   we complete the parsing audit in plan v1 Phase 1).
 - For refused: no LLM call was made.

This is a fast (~30-60s) regression check that the pipeline still works.
For the full 49-question benchmark with metrics + plots, use
`scripts/run_benchmark.py`; for the 5-ablation comparison use
`scripts/run_ablations.py`.

Exit code:
  0 if all selected questions pass
  1 if any selected question fails

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --json   # also write per-question JSON to stdout
    python scripts/smoke_test.py --all    # run the entire benchmark, not just smoke_*
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from src.pipeline import run as run_pipeline  # noqa: E402
from src.schemas import RAGOutput  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = REPO_ROOT / "eval" / "benchmark.yaml"

console = Console()


@dataclass
class Result:
    qid: str
    expected_state: str
    actual_state: str
    expected_docs: list[str]
    cited_docs: list[str]
    retrieved_top_docs: list[str]
    passed: bool
    reason: str


def evaluate(qid: str, q: dict, out: RAGOutput) -> Result:
    expected = q["expected_state"]
    actual = out.state

    expected_docs = sorted({c["doc"] for c in (q.get("expected_citations") or [])})
    cited_docs = sorted({c.doc for c in out.citations})
    retrieved_top_docs = list(
        dict.fromkeys(c.doc for c in out.evidence.retrieved_chunks[:3])
    )

    if expected != actual:
        return Result(
            qid=qid,
            expected_state=expected,
            actual_state=actual,
            expected_docs=expected_docs,
            cited_docs=cited_docs,
            retrieved_top_docs=retrieved_top_docs,
            passed=False,
            reason=f"state mismatch: expected '{expected}', got '{actual}'",
        )

    if expected == "refused":
        if out.llm_called:
            return Result(
                qid=qid,
                expected_state=expected,
                actual_state=actual,
                expected_docs=expected_docs,
                cited_docs=cited_docs,
                retrieved_top_docs=retrieved_top_docs,
                passed=False,
                reason="refused but LLM was called",
            )
        return Result(
            qid=qid,
            expected_state=expected,
            actual_state=actual,
            expected_docs=expected_docs,
            cited_docs=cited_docs,
            retrieved_top_docs=retrieved_top_docs,
            passed=True,
            reason="refused without LLM call",
        )

    # Confident or hedged: at least one expected doc must show up either in
    # the citations or in the top-3 retrieved chunks. Substring match because
    # parsed doc titles are full ("CPS 234 Information Security") but the
    # benchmark uses short ids ("CPS 234").
    if expected_docs:
        def _matches(needle: str, haystack: list[str]) -> bool:
            return any(needle.lower() in h.lower() for h in haystack)

        match_in_citations = any(_matches(d, cited_docs) for d in expected_docs)
        match_in_retrieval = any(_matches(d, retrieved_top_docs) for d in expected_docs)
        if not (match_in_citations or match_in_retrieval):
            return Result(
                qid=qid,
                expected_state=expected,
                actual_state=actual,
                expected_docs=expected_docs,
                cited_docs=cited_docs,
                retrieved_top_docs=retrieved_top_docs,
                passed=False,
                reason=(
                    f"no expected doc {expected_docs} in cited {cited_docs} "
                    f"or top-3 retrieved {retrieved_top_docs}"
                ),
            )

    return Result(
        qid=qid,
        expected_state=expected,
        actual_state=actual,
        expected_docs=expected_docs,
        cited_docs=cited_docs,
        retrieved_top_docs=retrieved_top_docs,
        passed=True,
        reason="ok",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end smoke test.")
    parser.add_argument("--json", action="store_true", help="Print per-question JSON output too")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failure")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every question in benchmark.yaml (not just smoke_*). Slow.",
    )
    args = parser.parse_args()

    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]

    # Default behaviour: only the original 5 golden questions (id smoke_*).
    # Use --all to run the entire benchmark (use run_benchmark.py for metrics + plots).
    if not args.all:
        questions = [q for q in questions if str(q.get("id", "")).startswith("smoke_")]
        if not questions:
            console.print(
                "[red]No questions with id starting with 'smoke_' found.[/red] "
                "Use --all to run the full benchmark instead."
            )
            return 1
    console.print(f"[bold]Running {len(questions)} question(s)[/bold]")

    results: list[Result] = []
    for q in questions:
        qid = q["id"]
        console.rule(f"[bold]{qid}[/bold] · {q['category']}")
        console.print(q["question"])
        try:
            out = run_pipeline(question=q["question"])
        except Exception as e:
            results.append(
                Result(
                    qid=qid,
                    expected_state=q["expected_state"],
                    actual_state="ERROR",
                    expected_docs=[],
                    cited_docs=[],
                    retrieved_top_docs=[],
                    passed=False,
                    reason=f"exception: {type(e).__name__}: {e}",
                )
            )
            console.print(f"[red]EXCEPTION:[/red] {e}")
            if args.fail_fast:
                break
            continue

        result = evaluate(qid, q, out)
        results.append(result)
        colour = "green" if result.passed else "red"
        console.print(
            f"[{colour}]{'PASS' if result.passed else 'FAIL'}[/{colour}] "
            f"{result.reason} · state={out.state} · llm_called={out.llm_called} · {out.elapsed_ms}ms"
        )
        if args.json:
            print(json.dumps(out.model_dump(), indent=2, ensure_ascii=False))
        if args.fail_fast and not result.passed:
            break

    # Summary table
    table = Table(title="Smoke test summary")
    table.add_column("qid")
    table.add_column("expected")
    table.add_column("actual")
    table.add_column("result", style="bold")
    table.add_column("reason")
    for r in results:
        table.add_row(
            r.qid,
            r.expected_state,
            r.actual_state,
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
            r.reason,
        )
    console.print(table)

    n_pass = sum(1 for r in results if r.passed)
    console.print(f"\n[bold]{n_pass}/{len(results)} passed[/bold]")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
