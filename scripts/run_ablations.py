"""Ablation runner.

Runs a fixed set of pipeline configurations against the benchmark and
produces a side-by-side comparison table — the Methodology section's
"why each design choice is justified" table.

Each ablation is its own run under eval/results/<run_id>/, just as if you
had called scripts/run_benchmark.py directly. After all runs complete this
script writes eval/results/ablation_comparison.md that pulls the headline
metrics into one table.

Predefined ablations:
  baseline         hybrid + faithfulness + floor      (reference)
  bm25_only        BM25-only ranking                  (justifies adding dense)
  dense_only       dense-only ranking                 (justifies adding BM25)
  no_faithfulness  hybrid, skip post-gen NLI          (justifies faithfulness check)
  no_floor_gate    hybrid, no top-1 dense floor       (justifies the refusal gate)

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --limit 5             # quick iteration
    python scripts/run_ablations.py --only bm25_only,dense_only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console  # noqa: E402

from eval.metrics import Aggregate, aggregate  # noqa: E402
from eval.plot_benchmark import plot_ablations, plot_run  # noqa: E402
from scripts.run_benchmark import (  # noqa: E402
    BENCHMARK_FILE,
    RESULTS_ROOT,
    load_benchmark,
    run_benchmark,
    write_config_json,
    write_metrics_csv,
    write_metrics_json,
    write_per_question_jsonl,
    write_summary_md,
)

console = Console()


ABLATIONS: list[dict] = [
    {
        "run_id": "baseline",
        "retrieval_mode": "hybrid",
        "enable_faithfulness": True,
        "enable_floor_gate": True,
        "justifies": "reference for all comparisons",
    },
    {
        "run_id": "bm25_only",
        "retrieval_mode": "bm25_only",
        "enable_faithfulness": True,
        "enable_floor_gate": True,
        "justifies": "adding dense retrieval to BM25",
    },
    {
        "run_id": "dense_only",
        "retrieval_mode": "dense_only",
        "enable_faithfulness": True,
        "enable_floor_gate": True,
        "justifies": "adding BM25 to dense retrieval",
    },
    {
        "run_id": "no_faithfulness",
        "retrieval_mode": "hybrid",
        "enable_faithfulness": False,
        "enable_floor_gate": True,
        "justifies": "the post-generation NLI faithfulness check",
    },
    {
        "run_id": "no_floor_gate",
        "retrieval_mode": "hybrid",
        "enable_faithfulness": True,
        "enable_floor_gate": False,
        "justifies": "the top-1 dense floor refusal gate",
    },
]


def comparison_md(rows: list[tuple[dict, Aggregate]]) -> str:
    """Build the side-by-side comparison table."""
    header = (
        "| Run | Retrieval | Faith | Floor | "
        "State acc | Refuse F1 | Recall@5 | Cite acc | Faith rate | "
        "LLM call% | p50 ms |"
    )
    sep = "|" + "---|" * 11
    lines = [
        "# Ablation comparison",
        "",
        "Each row is a separate run over the same benchmark. The `Justifies` column",
        "names the design choice each ablation is designed to test.",
        "",
        header,
        sep,
    ]
    for cfg, agg in rows:
        rec5 = agg.retrieval_recall.get(5, 0.0)
        lines.append(
            "| "
            + " | ".join([
                cfg["run_id"],
                cfg["retrieval_mode"],
                "on" if cfg["enable_faithfulness"] else "off",
                "on" if cfg["enable_floor_gate"] else "off",
                f"{agg.state_accuracy:.1%}",
                f"{agg.refusal['f1']:.1%}",
                f"{rec5:.1%}",
                f"{agg.citation_correctness:.1%}",
                f"{agg.faithfulness_rate:.1%}",
                f"{agg.latency['llm_call_rate']:.1%}",
                f"{agg.latency['p50_ms']:.0f}",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Ablation rationale",
        "",
        "| Run | Justifies |",
        "|---|---|",
    ])
    for cfg, _ in rows:
        lines.append(f"| {cfg['run_id']} | {cfg['justifies']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard ablation suite.")
    parser.add_argument("--benchmark", default=str(BENCHMARK_FILE))
    parser.add_argument("--output-dir", default=str(RESULTS_ROOT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--llm", default=None)
    parser.add_argument("--embedder", default=None)
    parser.add_argument("--nli", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated run_ids to include (default: all five ablations)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing per-run and cross-ablation PNG plots",
    )
    args = parser.parse_args()

    only = {r.strip() for r in args.only.split(",")} if args.only else None
    plan = [a for a in ABLATIONS if (only is None or a["run_id"] in only)]
    if not plan:
        console.print(f"[red]No ablations matched --only={args.only!r}[/red]")
        return 2

    questions = load_benchmark(Path(args.benchmark))
    console.print(f"[bold]Loaded {len(questions)} questions[/bold]")
    console.print(f"[bold]Running {len(plan)} ablation(s):[/bold] {[a['run_id'] for a in plan]}")

    rows: list[tuple[dict, Aggregate]] = []
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for cfg in plan:
        console.rule(f"[bold magenta]Ablation: {cfg['run_id']}[/bold magenta]")
        results, outputs = run_benchmark(
            questions,
            llm_id=args.llm,
            embedder_id=args.embedder,
            nli_id=args.nli,
            top_k=args.top_k,
            retrieval_mode=cfg["retrieval_mode"],
            enable_faithfulness=cfg["enable_faithfulness"],
            enable_floor_gate=cfg["enable_floor_gate"],
            limit=args.limit,
        )
        agg = aggregate(results)

        run_dir = out_root / cfg["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_per_question_jsonl(run_dir / "per_question.jsonl", results, outputs)
        write_metrics_json(run_dir / "metrics.json", agg)
        write_metrics_csv(run_dir / "metrics.csv", results)

        run_config = {
            "run_id": cfg["run_id"],
            "benchmark": args.benchmark,
            "llm": args.llm,
            "embedder": args.embedder,
            "nli": args.nli,
            "top_k": args.top_k,
            "retrieval_mode": cfg["retrieval_mode"],
            "enable_faithfulness": cfg["enable_faithfulness"],
            "enable_floor_gate": cfg["enable_floor_gate"],
            "limit": args.limit,
            "justifies": cfg["justifies"],
        }
        write_summary_md(run_dir / "summary.md", cfg["run_id"], run_config, agg, len(results))
        write_config_json(run_dir / "config.json", run_config)

        # Per-run plots (single-config view) live alongside the per-run outputs.
        if not args.no_plots:
            results_dicts = [asdict(r) for r in results]
            plot_run(results_dicts, asdict(agg), cfg["run_id"], run_dir / "plots")

        rows.append((cfg, agg))

    md = comparison_md(rows)
    comparison_path = out_root / "ablation_comparison.md"
    comparison_path.write_text(md, encoding="utf-8")
    rows_json = [{"config": cfg, "metrics": asdict(agg)} for cfg, agg in rows]
    json_path = out_root / "ablation_comparison.json"
    json_path.write_text(json.dumps(rows_json, indent=2), encoding="utf-8")

    # Cross-ablation comparison plots live next to the markdown summary.
    if not args.no_plots:
        plot_ablations(rows_json, out_root / "ablation_plots")

    console.print(f"\n[bold green]Comparison written to {comparison_path}[/bold green]")
    console.print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
