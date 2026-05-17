"""Calibrate the confidence-routing parameters from a benchmark run.

Plan v1 Phase 3 deliverable: replace the hand-tuned heuristic weights
(0.4/0.2/0.4) and thresholds (0.65/0.40) with values learned from labelled
benchmark data.

Two calibration methods are compared, both evaluated with 5-fold stratified
cross-validation on the same data:

  Method A — grid search on the 5-parameter heuristic
      (3 weights × 2 thresholds). Faithful to the existing routing
      architecture; the result is a direct drop-in for config/default.yaml.

  Method B — sklearn multinomial LogisticRegression
      The "logistic regression calibration" referred to by plan v1 Phase 3.
      Reported as comparison; would require runtime integration to deploy
      (left as future work — Method A is sufficient for v1 since the
      heuristic is itself linear).

Both methods use the same 3 input signals:
    retrieval_top1_dense, rank_overlap, nli_entail

Inputs:
    eval/results/<run_id>/per_question.jsonl  (must contain full_output)

Outputs (printed to stdout + written under eval/results/calibration_<run_id>/):
    grid_search_best.json   the best 5 params found, with CV accuracy
    logistic_regression.json  LR coefficients, intercepts, CV accuracy
    summary.md              human-readable comparison table

Usage:
    python -m eval.calibrate --from-run baseline
    python -m eval.calibrate --from-run tuned   # use the tuned run's data
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "eval" / "results"

# Order matters: matches sklearn's alphabetical class ordering when classes_ is
# ["confident", "hedged", "refused"] — we'll always re-key explicitly to avoid surprises.
STATES = ["refused", "hedged", "confident"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_signals_from_jsonl(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (signals, labels, qids) from a benchmark per_question.jsonl."""
    X: list[list[float]] = []
    y: list[str] = []
    qids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("error"):
            continue
        out = rec.get("full_output") or {}
        sig = (out.get("confidence") or {}).get("signals") or {}
        if "retrieval_top1_dense" not in sig:
            continue
        X.append([
            float(sig["retrieval_top1_dense"]),
            float(sig["rank_overlap"]),
            float(sig["nli_entail"]),
        ])
        y.append(rec["expected_state"])
        qids.append(rec["qid"])
    return np.asarray(X), np.asarray(y), qids


# ---------------------------------------------------------------------------
# Method A — grid search on (3 weights + 2 thresholds)
# ---------------------------------------------------------------------------


def heuristic_predict(
    X: np.ndarray, weights: tuple[float, float, float], thresholds: tuple[float, float]
) -> np.ndarray:
    """Apply the heuristic combiner + threshold routing."""
    alpha, beta, gamma = weights
    total = alpha + beta + gamma
    if total <= 0:
        return np.full(len(X), "refused", dtype=object)
    values = (alpha * X[:, 0] + beta * X[:, 1] + gamma * X[:, 2]) / total
    t_conf, t_hedge = thresholds
    states = np.where(
        values >= t_conf, "confident",
        np.where(values >= t_hedge, "hedged", "refused"),
    )
    return states


def grid_search(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42
) -> dict:
    """5-fold CV grid search over weights and thresholds.

    Grids are kept coarse to limit overfit risk on 49 examples.
    """
    weight_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    threshold_grid = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(X, y))

    best: dict | None = None
    n_evals = 0
    for alpha, beta, gamma in product(weight_grid, repeat=3):
        if alpha + beta + gamma == 0:
            continue
        for t_conf, t_hedge in product(threshold_grid, repeat=2):
            if t_hedge >= t_conf:
                continue
            n_evals += 1
            fold_accs = []
            for _, val_idx in folds:
                pred = heuristic_predict(X[val_idx], (alpha, beta, gamma), (t_conf, t_hedge))
                fold_accs.append(float(np.mean(pred == y[val_idx])))
            cv_acc = float(np.mean(fold_accs))
            if best is None or cv_acc > best["cv_acc"]:
                best = {
                    "weights": [alpha, beta, gamma],
                    "thresholds": [t_conf, t_hedge],
                    "cv_acc": cv_acc,
                    "cv_std": float(np.std(fold_accs)),
                }

    assert best is not None
    # Also compute fit-on-all accuracy for reference
    pred_all = heuristic_predict(
        X, tuple(best["weights"]), tuple(best["thresholds"])
    )
    best["train_acc"] = float(np.mean(pred_all == y))
    best["n_evals"] = n_evals
    return best


# ---------------------------------------------------------------------------
# Method B — sklearn multinomial LogisticRegression
# ---------------------------------------------------------------------------


def fit_logreg(
    X: np.ndarray, y: np.ndarray, n_folds: int = 5, seed: int = 42, C: float = 1.0
) -> dict:
    """5-fold CV multinomial LR + final fit on all data."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_accs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, C=C, random_state=seed)
        clf.fit(X[train_idx], y[train_idx])
        cv_accs.append(float(clf.score(X[val_idx], y[val_idx])))

    clf_full = LogisticRegression(max_iter=2000, C=C, random_state=seed)
    clf_full.fit(X, y)
    train_acc = float(clf_full.score(X, y))
    return {
        "cv_acc": float(np.mean(cv_accs)),
        "cv_std": float(np.std(cv_accs)),
        "train_acc": train_acc,
        "intercepts": clf_full.intercept_.tolist(),
        "coefs": clf_full.coef_.tolist(),
        "classes": clf_full.classes_.tolist(),
        "C": C,
    }


# ---------------------------------------------------------------------------
# Reporting + outputs
# ---------------------------------------------------------------------------


def current_heuristic_baseline(X: np.ndarray, y: np.ndarray) -> dict:
    """What the existing config produces on this data — for comparison."""
    cfg_path = REPO_ROOT / "config" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["confidence"]
    weights = (cfg["weight_top1_dense"], cfg["weight_rank_overlap"], cfg["weight_nli_entail"])
    thresholds = (cfg["threshold_confident"], cfg["threshold_hedged"])
    pred = heuristic_predict(X, weights, thresholds)
    return {
        "weights": list(weights),
        "thresholds": list(thresholds),
        "fit_on_data_acc": float(np.mean(pred == y)),
    }


def write_summary_md(
    path: Path, src_run_id: str, n: int,
    current: dict, grid: dict, logreg: dict,
) -> None:
    lines = [
        f"# Calibration analysis (source: `{src_run_id}` run, n={n})",
        "",
        "## State accuracy comparison",
        "",
        "| Method | CV accuracy | Fit-on-all accuracy |",
        "|---|---|---|",
        f"| Current config (heuristic, no learning) | — | {current['fit_on_data_acc']:.1%} |",
        f"| Grid-search heuristic (5 params, CV) | {grid['cv_acc']:.1%} ± {grid['cv_std']:.1%} | {grid['train_acc']:.1%} |",
        f"| Multinomial logistic regression (CV) | {logreg['cv_acc']:.1%} ± {logreg['cv_std']:.1%} | {logreg['train_acc']:.1%} |",
        "",
        "## Recommended config (drop-in for `config/default.yaml`)",
        "",
        "```yaml",
        "confidence:",
        f"  threshold_confident: {grid['thresholds'][0]}",
        f"  threshold_hedged: {grid['thresholds'][1]}",
        f"  weight_top1_dense: {grid['weights'][0]}",
        f"  weight_rank_overlap: {grid['weights'][1]}",
        f"  weight_nli_entail: {grid['weights'][2]}",
        "```",
        "",
        "## Logistic regression coefficients (deploy later — needs runtime integration)",
        "",
        f"Classes (sklearn order): `{logreg['classes']}`",
        "",
        "Coefficients (`[class][feature]` where features are `[top1_dense, rank_overlap, nli_entail]`):",
        "",
        "```json",
        json.dumps({"coefs": logreg["coefs"], "intercepts": logreg["intercepts"]}, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- CV accuracy is the unbiased estimate; fit-on-all is upper bound.",
        f"- Sample size n={n} is small; CV accuracy ± std reflects variance.",
        "- Grid-search method drops directly into `config/default.yaml`; multinomial LR would require "
        "rewriting `src.confidence.score_and_route` to use a logits-based decision rule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(current: dict, grid: dict, logreg: dict) -> None:
    print()
    print("=" * 72)
    print(" CALIBRATION SUMMARY")
    print("=" * 72)
    print(f"  Current config (no learning, fit-on-data):  {current['fit_on_data_acc']:.1%}")
    print(f"  Grid-search heuristic (5-fold CV):           {grid['cv_acc']:.1%} ± {grid['cv_std']:.1%}")
    print(f"                                  (fit-on-all): {grid['train_acc']:.1%}")
    print(f"  Multinomial logistic regression (5-fold CV): {logreg['cv_acc']:.1%} ± {logreg['cv_std']:.1%}")
    print(f"                                  (fit-on-all): {logreg['train_acc']:.1%}")
    print()
    print("  Recommended drop-in config (from grid search):")
    print(f"    threshold_confident: {grid['thresholds'][0]}")
    print(f"    threshold_hedged:    {grid['thresholds'][1]}")
    print(f"    weight_top1_dense:   {grid['weights'][0]}")
    print(f"    weight_rank_overlap: {grid['weights'][1]}")
    print(f"    weight_nli_entail:   {grid['weights'][2]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate confidence routing from a benchmark run.")
    parser.add_argument(
        "--from-run", default="baseline",
        help="Run id under eval/results/ whose per_question.jsonl provides the training data",
    )
    parser.add_argument("--folds", type=int, default=5, help="CV folds (default 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression inverse-regularisation")
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.0,
        help="If >0, split this fraction off as a held-out test set "
        "(stratified by label); calibration runs on the remaining train set, "
        "then reports held-out test accuracy as the unbiased estimate.",
    )
    args = parser.parse_args()

    src_path = RESULTS_ROOT / args.from_run / "per_question.jsonl"
    if not src_path.exists():
        print(f"file not found: {src_path}", file=sys.stderr)
        print(f"Run `python scripts/run_benchmark.py --run-id {args.from_run}` first.", file=sys.stderr)
        return 1

    X_full, y_full, qids = load_signals_from_jsonl(src_path)
    print(f"Loaded {len(X_full)} (signals, label) rows from {src_path}")
    print(f"Class balance: {dict(zip(*np.unique(y_full, return_counts=True)))}")

    # Optional train/test split
    holdout_info: dict | None = None
    if args.holdout > 0:
        X_train, X_test, y_train, y_test, qids_train, qids_test = train_test_split(
            X_full, y_full, qids,
            test_size=args.holdout,
            stratify=y_full,
            random_state=args.seed,
        )
        print(f"\nTrain/test split: {len(X_train)} train, {len(X_test)} held-out test ({args.holdout:.0%})")
        holdout_info = {
            "test_size": args.holdout,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_qids": qids_test,
        }
        X, y = X_train, y_train  # calibration runs only on train
    else:
        X, y = X_full, y_full

    if len(X) < args.folds * 2:
        print(f"[warn] only {len(X)} samples — {args.folds}-fold CV may be unstable", file=sys.stderr)

    current = current_heuristic_baseline(X_full, y_full)  # current heuristic is on ALL data
    print(f"\nCurrent config produces {current['fit_on_data_acc']:.1%} accuracy on all data.")

    print(f"\nRunning grid search ({args.folds}-fold CV) on {len(X)} samples...")
    grid = grid_search(X, y, n_folds=args.folds, seed=args.seed)
    print(f"  evaluated {grid['n_evals']} configurations")

    print(f"\nFitting multinomial logistic regression ({args.folds}-fold CV) on {len(X)} samples...")
    logreg = fit_logreg(X, y, n_folds=args.folds, seed=args.seed, C=args.C)

    # If holdout, evaluate the chosen params on the held-out test set
    if holdout_info is not None:
        # Grid-search params: apply heuristic_predict directly
        grid_test_pred = heuristic_predict(X_test, tuple(grid["weights"]), tuple(grid["thresholds"]))
        grid_test_acc = float(np.mean(grid_test_pred == y_test))
        grid["holdout_test_acc"] = grid_test_acc

        # LR: re-fit on train only, predict on test
        clf_train = LogisticRegression(max_iter=2000, C=args.C, random_state=args.seed)
        clf_train.fit(X_train, y_train)
        logreg_test_acc = float(clf_train.score(X_test, y_test))
        logreg["holdout_test_acc"] = logreg_test_acc

        holdout_info["grid_search_test_acc"] = grid_test_acc
        holdout_info["logreg_test_acc"] = logreg_test_acc

        print()
        print(f"  Held-out test accuracy (n={len(X_test)}):")
        print(f"    Grid-search heuristic:       {grid_test_acc:.1%}")
        print(f"    Multinomial LR:              {logreg_test_acc:.1%}")

    print_console_summary(current, grid, logreg)

    out_dir = RESULTS_ROOT / f"calibration_{args.from_run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid_search_best.json").write_text(
        json.dumps(grid, indent=2), encoding="utf-8"
    )
    (out_dir / "logistic_regression.json").write_text(
        json.dumps(logreg, indent=2), encoding="utf-8"
    )
    if holdout_info is not None:
        (out_dir / "holdout_split.json").write_text(
            json.dumps(holdout_info, indent=2), encoding="utf-8"
        )
    write_summary_md(
        out_dir / "summary.md", args.from_run, len(X_full), current, grid, logreg
    )
    print(f"  outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
