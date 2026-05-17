"""Plotting for scripts/run_benchmark.py and scripts/run_ablations.py.

Two entry points:
  plot_run(results, agg, run_id, out_dir)
      Produces single-run plots (one run, one ablation config):
        - state_confusion.png      3x3 heatmap, expected vs predicted state
        - retrieval_recall.png     line: Recall@K for K=1,3,5,10
        - per_category_accuracy.png horizontal bar: state accuracy by category
        - confidence_by_state.png  overlapping histograms of confidence value
        - latency_distribution.png histogram of per-question elapsed_ms

  plot_ablations(rows, out_dir)
      Produces cross-ablation comparison plots (one row per ablation):
        - headline_metrics.png     grouped bars: state_acc / refusal_f1 /
                                   citation_acc / faith_rate per ablation
        - retrieval_recall_across_ablations.png  line: Recall@K, one line per ablation
        - latency_comparison.png   bars: mean and p95 latency per ablation

matplotlib is the only extra dep. If not installed, plot calls become
warnings rather than errors — the metric/CSV/markdown outputs still land
even if plots can't render.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _matplotlib_or_warn():
    """Return the matplotlib.pyplot module, or None and print a warning."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend; safe for scripts
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print(
            "[warn] matplotlib not installed — skipping plots. "
            "Install with `pip install matplotlib` if you want them.",
            file=sys.stderr,
        )
        return None


def _save(fig, path: Path, plt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# =============================================================================
# Single-run plots — plot_run()
# =============================================================================


def _plot_state_confusion(agg: dict, run_id: str, out_path: Path, plt) -> None:
    states = ["confident", "hedged", "refused"]
    cm = agg["state_confusion_matrix"]
    # Build matrix in fixed order
    import numpy as np

    matrix = np.array([[cm.get(e, {}).get(p, 0) for p in states] for e in states])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(states)
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("Expected state")
    ax.set_title(f"State confusion · run={run_id}")
    # Annotate counts
    max_val = matrix.max() if matrix.max() else 1
    for i in range(len(states)):
        for j in range(len(states)):
            v = int(matrix[i, j])
            color = "white" if v > max_val / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.7, label="count")
    _save(fig, out_path, plt)


def _plot_retrieval_recall(agg: dict, run_id: str, out_path: Path, plt) -> None:
    rr = agg["retrieval_recall"]
    # JSON-deserialized keys may be strings; normalise to int + sort.
    ks = sorted(int(k) for k in rr.keys())
    vals = [float(rr[k] if k in rr else rr[str(k)]) for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, marker="o", linewidth=2, color="steelblue")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (answerable questions only)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Retrieval Recall@K · run={run_id}")
    ax.grid(True, alpha=0.3)
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.1%}", (k, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    _save(fig, out_path, plt)


def _plot_per_category_accuracy(agg: dict, run_id: str, out_path: Path, plt) -> None:
    pca = agg["per_category_accuracy"]
    if not pca:
        return
    cats = list(pca.keys())
    vals = [float(v) for v in pca.values()]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(cats) + 1.5)))
    colors = ["seagreen" if v >= 0.8 else "goldenrod" if v >= 0.5 else "indianred" for v in vals]
    y = range(len(cats))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(cats)
    ax.set_xlabel("State accuracy")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Per-category state accuracy · run={run_id}")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.invert_yaxis()
    _save(fig, out_path, plt)


def _plot_confidence_by_state(results: list[dict], run_id: str, out_path: Path, plt) -> None:
    by_state: dict[str, list[float]] = {"confident": [], "hedged": [], "refused": []}
    for r in results:
        if r.get("error"):
            continue
        s = r.get("expected_state")
        if s in by_state:
            by_state[s].append(float(r.get("confidence_value", 0.0)))

    if not any(by_state.values()):
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"confident": "seagreen", "hedged": "goldenrod", "refused": "indianred"}
    bins = [i / 20 for i in range(21)]  # 0.0..1.0 in 0.05 steps
    for state, vals in by_state.items():
        if vals:
            ax.hist(
                vals, bins=bins, alpha=0.5, label=f"expected={state} (n={len(vals)})",
                color=colors[state], edgecolor="white",
            )
    ax.set_xlabel("Confidence value")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.set_title(f"Confidence value distribution by expected state · run={run_id}")
    # Optional: thresholds as vertical lines (read from default.yaml typical)
    ax.axvline(0.65, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(0.40, color="black", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(0.65, ax.get_ylim()[1] * 0.95, "  confident →", fontsize=8, alpha=0.6, va="top")
    ax.text(0.40, ax.get_ylim()[1] * 0.95, "  hedged →", fontsize=8, alpha=0.6, va="top")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path, plt)


def _plot_latency_distribution(results: list[dict], run_id: str, out_path: Path, plt) -> None:
    ms = [int(r.get("elapsed_ms", 0)) for r in results if not r.get("error") and r.get("elapsed_ms")]
    if not ms:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ms, bins=20, color="steelblue", edgecolor="white")
    ax.set_xlabel("elapsed_ms")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-question latency · run={run_id}")
    ax.grid(True, axis="y", alpha=0.3)
    # Mean + p95 markers
    import statistics

    mean_ms = statistics.mean(ms)
    p95_ms = sorted(ms)[max(0, int(round(0.95 * (len(ms) - 1))))]
    ax.axvline(mean_ms, color="black", linestyle="--", linewidth=1, label=f"mean {mean_ms:.0f}")
    ax.axvline(p95_ms, color="darkred", linestyle="--", linewidth=1, label=f"p95 {p95_ms:.0f}")
    ax.legend(fontsize=9)
    _save(fig, out_path, plt)


def plot_run(results: list[dict], agg: dict, run_id: str, out_dir: Path) -> None:
    """Produce single-run plots into out_dir.

    `results` is a list of per-question dicts (as written to per_question.jsonl,
    minus the heavy `full_output` field — but the field's presence is fine).
    `agg` is the dict form of eval.metrics.Aggregate (matches metrics.json).
    """
    plt = _matplotlib_or_warn()
    if plt is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plots] writing run plots to {out_dir}/")
    _plot_state_confusion(agg, run_id, out_dir / "state_confusion.png", plt)
    _plot_retrieval_recall(agg, run_id, out_dir / "retrieval_recall.png", plt)
    _plot_per_category_accuracy(agg, run_id, out_dir / "per_category_accuracy.png", plt)
    _plot_confidence_by_state(results, run_id, out_dir / "confidence_by_state.png", plt)
    _plot_latency_distribution(results, run_id, out_dir / "latency_distribution.png", plt)


# =============================================================================
# Cross-ablation plots — plot_ablations()
# =============================================================================


def _plot_headline_metrics(rows: list[dict], out_path: Path, plt) -> None:
    """Grouped bar chart: 4 headline metrics per ablation."""
    import numpy as np

    names = [r["config"]["run_id"] for r in rows]
    metric_keys = [
        ("state_accuracy", "State acc."),
        ("citation_correctness", "Citation acc."),
        ("faithfulness_rate", "Faithfulness"),
    ]
    refusal_f1 = [r["metrics"]["refusal"]["f1"] for r in rows]
    bars: dict[str, list[float]] = {label: [] for _, label in metric_keys}
    for r in rows:
        for key, label in metric_keys:
            bars[label].append(float(r["metrics"].get(key, 0.0)))
    bars["Refusal F1"] = refusal_f1

    n_metrics = len(bars)
    n_runs = len(names)
    x = np.arange(n_runs)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n_runs + 2), 4.5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, (label, vals) in enumerate(bars.items()):
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=label, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Headline metrics across ablations (higher = better)")
    ax.legend(fontsize=9, loc="upper center", ncol=n_metrics, bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path, plt)


def _plot_retrieval_recall_across_ablations(rows: list[dict], out_path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    all_ks: set[int] = set()
    for i, r in enumerate(rows):
        rr = r["metrics"]["retrieval_recall"]
        ks = sorted(int(k) for k in rr.keys())
        vals = [float(rr.get(k, rr.get(str(k), 0.0))) for k in ks]
        all_ks.update(ks)
        ax.plot(
            ks, vals,
            marker="o", linewidth=2,
            label=r["config"]["run_id"],
            color=colors[i % len(colors)],
        )

    ax.set_xticks(sorted(all_ks))
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K (answerable questions only)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval Recall@K across ablations")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out_path, plt)


def _plot_latency_comparison(rows: list[dict], out_path: Path, plt) -> None:
    import numpy as np

    names = [r["config"]["run_id"] for r in rows]
    mean_ms = [float(r["metrics"]["latency"]["mean_ms"]) for r in rows]
    p95_ms = [float(r["metrics"]["latency"]["p95_ms"]) for r in rows]

    n = len(names)
    x = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * n + 2), 4.5))
    ax.bar(x - width / 2, mean_ms, width=width, label="mean", color="steelblue")
    ax.bar(x + width / 2, p95_ms, width=width, label="p95", color="indianred")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-question latency across ablations (lower = better)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out_path, plt)


def plot_ablations(rows: list[dict], out_dir: Path) -> None:
    """Produce cross-ablation comparison plots.

    `rows` matches the structure written to ablation_comparison.json — i.e.
    a list of {"config": {...}, "metrics": {...}} where metrics is the
    Aggregate dict (state_accuracy, refusal, retrieval_recall, etc.).
    """
    plt = _matplotlib_or_warn()
    if plt is None or not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plots] writing ablation plots to {out_dir}/")
    _plot_headline_metrics(rows, out_dir / "headline_metrics.png", plt)
    _plot_retrieval_recall_across_ablations(
        rows, out_dir / "retrieval_recall_across_ablations.png", plt
    )
    _plot_latency_comparison(rows, out_dir / "latency_comparison.png", plt)


# =============================================================================
# Self-test (synthetic data, no real run needed)
# =============================================================================


def _self_test(tmp_dir: Path) -> None:
    """Verify both plot pipelines produce files end-to-end with fake data."""
    plt = _matplotlib_or_warn()
    if plt is None:
        print("matplotlib not available — self-test skipped.")
        return

    # Fake single-run inputs
    fake_results = [
        {
            "qid": "q1", "category": "answerable_easy", "expected_state": "confident",
            "predicted_state": "confident", "confidence_value": 0.82,
            "elapsed_ms": 1200, "error": None,
        },
        {
            "qid": "q2", "category": "unanswerable_out_of_scope", "expected_state": "refused",
            "predicted_state": "refused", "confidence_value": 0.18,
            "elapsed_ms": 200, "error": None,
        },
        {
            "qid": "q3", "category": "answerable_hard", "expected_state": "confident",
            "predicted_state": "hedged", "confidence_value": 0.55,
            "elapsed_ms": 1500, "error": None,
        },
    ]
    fake_agg = {
        "state_accuracy": 0.667,
        "state_confusion_matrix": {
            "confident": {"confident": 1, "hedged": 1, "refused": 0},
            "hedged": {"confident": 0, "hedged": 0, "refused": 0},
            "refused": {"confident": 0, "hedged": 0, "refused": 1},
        },
        "refusal": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 1, "fp": 0, "fn": 0},
        "retrieval_recall": {1: 0.5, 3: 0.7, 5: 0.9, 10: 1.0},
        "citation_correctness": 0.8,
        "faithfulness_rate": 0.75,
        "latency": {"mean_ms": 967, "p50_ms": 1200, "p95_ms": 1500, "llm_call_rate": 0.67},
        "per_category_accuracy": {
            "answerable_easy": 1.0,
            "answerable_hard": 0.0,
            "unanswerable_out_of_scope": 1.0,
        },
    }
    plot_run(fake_results, fake_agg, "selftest", tmp_dir / "run")

    # Fake ablation rows
    fake_rows = [
        {
            "config": {"run_id": "baseline"},
            "metrics": {**fake_agg, "state_accuracy": 0.82},
        },
        {
            "config": {"run_id": "bm25_only"},
            "metrics": {**fake_agg, "state_accuracy": 0.71, "retrieval_recall": {1: 0.3, 3: 0.5, 5: 0.7, 10: 0.85}},
        },
        {
            "config": {"run_id": "dense_only"},
            "metrics": {**fake_agg, "state_accuracy": 0.78, "retrieval_recall": {1: 0.4, 3: 0.65, 5: 0.85, 10: 0.95}},
        },
    ]
    plot_ablations(fake_rows, tmp_dir / "ablations")
    print("OK — plot_benchmark self-test passed; PNGs written under", tmp_dir)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        _self_test(Path(td))
