"""Plot a comparison JSON produced by `python -m eval.comparison_eval`.

Two charts:
 1. Coverage-vs-Accuracy scatter — one point per baseline. Trustworthy
    sits up-and-to-the-left (lower coverage, higher accuracy) vs naive
    baselines that sit on the right edge (always answer).
 2. Risk-adjusted loss bar chart — lower is better. Reorders so visually
    clear which baseline has the best loss.

Optional: --per-category produces one coverage-vs-accuracy chart per
category, side-by-side.

matplotlib is the only extra dep. If not installed:
    pip install matplotlib

Usage:
    python -m eval.plot_results eval/results/comparison_<timestamp>.json
    python -m eval.plot_results eval/results/<file>.json --per-category
    python -m eval.plot_results eval/results/<file>.json --out-dir eval/results/plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _require_matplotlib():
    try:
        import matplotlib  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print(
            "matplotlib not installed. Run: pip install matplotlib",
            file=sys.stderr,
        )
        sys.exit(2)


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_coverage_vs_accuracy(results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = results["summary"]
    if not summary:
        print("No summary rows in results — skip coverage plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, row in enumerate(summary):
        cov = row["coverage"]
        acc = row["accuracy_on_answered"]
        ax.scatter(cov, acc, s=120, color=colours[i % len(colours)], label=row["system"], zorder=3)
        ax.annotate(
            row["system"],
            (cov, acc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Coverage (fraction answered)")
    ax.set_ylabel("Accuracy on answered")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Coverage vs Accuracy · n={results['config']['n_questions']} · "
        f"LLM={results['config']['llm']}"
    )
    # Reference: ideal corner
    ax.scatter([1.0], [1.0], marker="*", color="gold", s=200, zorder=2, label="ideal")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        rel = out_path.resolve().relative_to(REPO_ROOT)
        print(f"saved: {rel}")
    except ValueError:
        print(f"saved: {out_path}")


def plot_risk_loss(results: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    summary = results["summary"]
    if not summary:
        return

    # Sort by loss ascending — best (lowest) on the left
    rows = sorted(summary, key=lambda r: r["risk_adjusted_loss"])
    names = [r["system"] for r in rows]
    losses = [r["risk_adjusted_loss"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, losses, color="steelblue")
    # Highlight best (lowest)
    bars[0].set_color("seagreen")
    # Highlight worst
    bars[-1].set_color("indianred")

    ax.set_ylabel("Risk-adjusted loss (lower = better)")
    ax.set_title(
        f"Risk-adjusted loss · LLM={results['config']['llm']} · "
        f"wrong_confident={results['config']['loss_matrix']['wrong_confident']}, "
        f"refused_answerable={results['config']['loss_matrix']['refused_answerable']}"
    )
    for bar, val in zip(bars, losses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        rel = out_path.resolve().relative_to(REPO_ROOT)
        print(f"saved: {rel}")
    except ValueError:
        print(f"saved: {out_path}")


def plot_per_category(results: dict, out_path: Path) -> None:
    """One subplot per category, coverage-vs-accuracy."""
    import matplotlib.pyplot as plt

    per_category = results.get("per_category", {})
    if not per_category:
        print("No per-category data — skip.", file=sys.stderr)
        return

    # Collect unique categories
    categories = sorted({
        cat for cats in per_category.values() for cat in cats.keys()
    })
    if not categories:
        return

    n_cats = len(categories)
    cols = min(3, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), squeeze=False)
    colours = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, cat in enumerate(categories):
        ax = axes[i // cols][i % cols]
        for j, (system, cats) in enumerate(per_category.items()):
            if cat not in cats:
                continue
            r = cats[cat]
            ax.scatter(
                r["coverage"], r["accuracy_on_answered"],
                s=80, color=colours[j % len(colours)], label=system, zorder=3,
            )
        ax.set_title(f"{cat} (n={next(iter(per_category.values())).get(cat, {}).get('n_total', 0)})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Acc/Ans")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=7)

    # Hide unused subplots
    for k in range(n_cats, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.suptitle(f"Per-category Coverage vs Accuracy · LLM={results['config']['llm']}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    try:
        rel = out_path.resolve().relative_to(REPO_ROOT)
        print(f"saved: {rel}")
    except ValueError:
        print(f"saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a comparison results JSON.")
    parser.add_argument("results_path", help="Path to comparison JSON")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs (default: same dir as results, plus '_plots' suffix)",
    )
    parser.add_argument("--per-category", action="store_true", help="Also plot per-category breakdown")
    args = parser.parse_args()

    _require_matplotlib()

    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"file not found: {results_path}", file=sys.stderr)
        return 1

    results = load_results(results_path)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = results_path.parent / f"{results_path.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_coverage_vs_accuracy(results, out_dir / "coverage_vs_accuracy.png")
    plot_risk_loss(results, out_dir / "risk_adjusted_loss.png")
    if args.per_category:
        plot_per_category(results, out_dir / "per_category.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
