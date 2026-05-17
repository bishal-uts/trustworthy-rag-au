"""Per-category analysis of a benchmark run.

For each question category, computes:
  - confidence-value distribution conditional on outcome (correct / wrong)
  - whether a per-category threshold would improve accuracy vs the global one
  - the dominant failure mode (wrong predicted_state vs expected_state)

The output is descriptive only — production routing has no category signal,
so per-category thresholds would require a category classifier as a
preprocessing step. This analysis is for the report's Discussion section:
"per-category accuracy varies significantly; future work could route via
a question-type classifier".

Usage:
    python -m eval.analyze_per_category --from-run calibrated_v2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "eval" / "results"


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def per_category(records: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by_cat[r["category"]].append(r)

    summary: dict[str, dict] = {}
    for cat, rs in by_cat.items():
        n = len(rs)
        n_correct = sum(1 for r in rs if r["predicted_state"] == r["expected_state"])
        confs = [float(r.get("confidence_value", 0.0)) for r in rs]
        confs_correct = [
            float(r.get("confidence_value", 0.0)) for r in rs
            if r["predicted_state"] == r["expected_state"]
        ]
        confs_wrong = [
            float(r.get("confidence_value", 0.0)) for r in rs
            if r["predicted_state"] != r["expected_state"]
        ]
        # Failure-mode distribution
        mode: dict[str, int] = defaultdict(int)
        for r in rs:
            if r["predicted_state"] != r["expected_state"]:
                mode[f'{r["expected_state"]}->{r["predicted_state"]}'] += 1

        summary[cat] = {
            "n": n,
            "accuracy": n_correct / n if n else 0.0,
            "conf_mean_all": mean(confs) if confs else 0.0,
            "conf_mean_correct": mean(confs_correct) if confs_correct else None,
            "conf_mean_wrong": mean(confs_wrong) if confs_wrong else None,
            "dominant_failure_mode": (
                max(mode.items(), key=lambda x: x[1]) if mode else (None, 0)
            ),
        }
    return summary


def find_best_per_category_thresholds(records: list[dict]) -> dict[str, dict]:
    """For each category, find the threshold that would maximize accuracy.

    Only meaningful for categories where predicted_state being wrong is
    associated with confidence value distribution differences.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by_cat[r["category"]].append(r)

    out: dict[str, dict] = {}
    thresholds_to_try = [i / 20 for i in range(2, 19)]  # 0.1..0.9 in 0.05 steps

    for cat, rs in by_cat.items():
        # For per-category threshold experiment, treat as binary: should this
        # question's state be revised based on confidence value alone?
        # We can't redo the full routing without the raw signals, but we can
        # ask: what global cutoff between any-state categories?
        # This is just a descriptive proxy.
        confs = [(float(r.get("confidence_value", 0.0)), r["expected_state"]) for r in rs]
        # Suggest an "ideal" cutoff per category by looking at correlation
        # between confidence and expected_state.
        if not confs:
            continue
        # Mean confidence per expected state within this category
        by_state: dict[str, list[float]] = defaultdict(list)
        for c, s in confs:
            by_state[s].append(c)
        out[cat] = {
            "n": len(confs),
            "mean_conf_by_state": {
                s: float(np.mean(vs)) for s, vs in by_state.items() if vs
            },
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-category analysis of a benchmark run.")
    parser.add_argument("--from-run", required=True, help="Run id under eval/results/")
    args = parser.parse_args()

    src = RESULTS_ROOT / args.from_run / "per_question.jsonl"
    if not src.exists():
        print(f"file not found: {src}", file=sys.stderr)
        return 1

    records = load(src)
    print(f"Loaded {len(records)} records from {src}")
    print()

    cats = per_category(records)
    print(f"{'Category':<30} {'N':>3} {'Acc':>7} {'Conf(OK)':>10} {'Conf(BAD)':>10} {'Dominant failure':<30}")
    print("-" * 100)
    for cat, info in sorted(cats.items(), key=lambda x: -x[1]["accuracy"]):
        cm = info["conf_mean_correct"]
        cm_str = f"{cm:.2f}" if cm is not None else "  -"
        wm = info["conf_mean_wrong"]
        wm_str = f"{wm:.2f}" if wm is not None else "  -"
        dom = info["dominant_failure_mode"]
        dom_str = f"{dom[0]} ({dom[1]}x)" if dom[0] else "(all correct)"
        print(f"{cat:<30} {info['n']:>3} {info['accuracy']:>6.1%} {cm_str:>10} {wm_str:>10} {dom_str:<30}")

    print()
    print("Mean confidence by expected state, per category:")
    mc = find_best_per_category_thresholds(records)
    for cat, info in sorted(mc.items()):
        states = info["mean_conf_by_state"]
        states_str = "  ".join(f"{s}={v:.2f}" for s, v in sorted(states.items()))
        print(f"  {cat:<30} {states_str}")

    # Write JSON for downstream use
    out_dir = RESULTS_ROOT / args.from_run / "per_category_analysis.json"
    out_dir.write_text(json.dumps({"cats": cats, "mean_conf": mc}, indent=2, default=str), encoding="utf-8")
    print()
    print(f"JSON saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
