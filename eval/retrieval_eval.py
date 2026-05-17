"""Retrieval-only eval over eval/benchmark.yaml.

Loads each benchmark question, runs hybrid retrieval (BM25 + dense + RRF),
and computes Recall@k / Hit@k / MRR against the expected citations. The
LLM is NEVER called — this isolates retrieval quality from generation.

This makes embedder comparison fair: rebuild the dense index for each
embedder (`python scripts/build_index.py --all-embedders`) and then
`python -m eval.retrieval_eval --all-embedders` shows which one finds
the right paragraph most often.

Usage:
    python -m eval.retrieval_eval                              # default embedder, k=10
    python -m eval.retrieval_eval --embedder bge-base --k 5
    python -m eval.retrieval_eval --all-embedders              # compare all
    python -m eval.retrieval_eval --strict                      # exact paragraph match
    python -m eval.retrieval_eval --per-question                # show every question's row
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

# Make `src` and `eval` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from eval.metrics import (  # noqa: E402
    ExpectedCitation,
    first_hit_rank,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.chunking import load_all_chunks  # noqa: E402
from src.config import settings  # noqa: E402
from src.models.registry import list_embedders  # noqa: E402
from src.retrieval.bm25 import BM25Index  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402
from src.retrieval.hybrid import HybridRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = REPO_ROOT / "eval" / "benchmark.yaml"

console = Console()


def load_benchmark() -> list[dict]:
    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def to_expected(raw: list[dict] | None) -> list[ExpectedCitation]:
    if not raw:
        return []
    return [
        ExpectedCitation(
            doc=str(c.get("doc", "")),
            section=str(c.get("section", "")),
            paragraph=str(c.get("paragraph", "")),
        )
        for c in raw
    ]


def build_chunk_lookup() -> dict[str, dict]:
    chunks = load_all_chunks()
    if not chunks:
        raise RuntimeError(
            "No chunks found. Run `python -m src.parsing --all` then "
            "`python -m src.chunking --all` first."
        )
    return {c["chunk_id"]: c for c in chunks}


def eval_one_embedder(
    embedder_id: str,
    k: int,
    strict: bool,
    chunk_lookup: dict[str, dict],
    bm25: BM25Index,
    questions: list[dict],
) -> dict:
    """Run retrieval for every answerable question; aggregate metrics."""
    dense = DenseIndex.load(embedder_id)
    retriever = HybridRetriever(bm25_index=bm25, dense_index=dense)

    per_q: list[dict] = []
    skipped_refused = 0
    skipped_no_expected = 0

    for q in questions:
        if q["expected_state"] == "refused":
            skipped_refused += 1
            continue
        expected = to_expected(q.get("expected_citations"))
        if not expected:
            skipped_no_expected += 1
            continue

        # Pull a generous candidate pool so MRR can see deep matches
        hits = retriever.search(q["question"], top_k=max(k, 20))
        retrieved: list[tuple[str, str, str]] = []
        for h in hits:
            meta = chunk_lookup.get(h.chunk_id)
            if meta:
                retrieved.append(
                    (meta["doc"], meta["section"], str(meta["paragraph"]))
                )

        per_q.append(
            {
                "qid": q["id"],
                "recall": recall_at_k(retrieved, expected, k, strict=strict),
                "hit": hit_at_k(retrieved, expected, k, strict=strict),
                "rr": reciprocal_rank(retrieved, expected, strict=strict),
                "first_rank": first_hit_rank(retrieved, expected, strict=strict),
            }
        )

    n = len(per_q)
    return {
        "embedder": embedder_id,
        "n": n,
        "skipped_refused": skipped_refused,
        "skipped_no_expected": skipped_no_expected,
        "mean_recall": statistics.mean(r["recall"] for r in per_q) if n else 0.0,
        "mean_hit": statistics.mean(1.0 if r["hit"] else 0.0 for r in per_q) if n else 0.0,
        "mrr": statistics.mean(r["rr"] for r in per_q) if n else 0.0,
        "per_question": per_q,
    }


def render_summary(results: list[dict], k: int, strict: bool) -> None:
    title = f"Retrieval eval · k={k} · matching={'strict' if strict else 'loose'}"
    table = Table(title=title)
    table.add_column("embedder")
    table.add_column("n", justify="right")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"Hit@{k}", justify="right")
    table.add_column("MRR", justify="right")
    for r in results:
        table.add_row(
            r["embedder"],
            str(r["n"]),
            f"{r['mean_recall']:.3f}",
            f"{r['mean_hit']:.3f}",
            f"{r['mrr']:.3f}",
        )
    console.print(table)


def render_per_question(result: dict, k: int) -> None:
    table = Table(title=f"Per-question · embedder={result['embedder']}")
    table.add_column("qid")
    table.add_column(f"Recall@{k}", justify="right")
    table.add_column(f"Hit@{k}", justify="right")
    table.add_column("RR", justify="right")
    table.add_column("first_rank", justify="right")
    for r in result["per_question"]:
        table.add_row(
            r["qid"],
            f"{r['recall']:.2f}",
            "✓" if r["hit"] else "✗",
            f"{r['rr']:.2f}",
            str(r["first_rank"]) if r["first_rank"] is not None else "—",
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval-only eval over benchmark.")
    parser.add_argument("--embedder", help="Embedder id (default: settings.default_embedder)")
    parser.add_argument(
        "--all-embedders",
        action="store_true",
        help="Eval every embedder that has an index built",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k cutoff (default: 10)")
    parser.add_argument(
        "--strict", action="store_true",
        help="Require exact (doc, section, paragraph) match instead of loose",
    )
    parser.add_argument(
        "--per-question", action="store_true",
        help="Also print the per-question row for each embedder",
    )
    args = parser.parse_args()

    # Shared setup: BM25 + chunk lookup are not embedder-specific
    try:
        chunk_lookup = build_chunk_lookup()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    try:
        bm25 = BM25Index.load()
    except FileNotFoundError as e:
        console.print(
            f"[red]BM25 index not built. Run `python scripts/build_index.py` first.[/red]\n{e}"
        )
        return 1

    questions = load_benchmark()

    if args.all_embedders:
        candidates = [s.id for s in list_embedders()]
    else:
        candidates = [args.embedder or settings.default_embedder]

    results: list[dict] = []
    for eid in candidates:
        if not DenseIndex.exists(eid):
            console.print(
                f"[yellow]skip {eid}: no FAISS index — run "
                f"`python scripts/build_index.py --embedder {eid}`[/yellow]"
            )
            continue
        try:
            res = eval_one_embedder(eid, args.k, args.strict, chunk_lookup, bm25, questions)
        except Exception as e:
            console.print(f"[red]fail {eid}: {type(e).__name__}: {e}[/red]")
            continue
        results.append(res)

    if not results:
        console.print("[red]No embedder produced results.[/red]")
        return 1

    render_summary(results, args.k, args.strict)
    if args.per_question:
        for r in results:
            render_per_question(r, args.k)

    first = results[0]
    if first["skipped_refused"] or first["skipped_no_expected"]:
        console.print(
            f"\n[dim]Skipped {first['skipped_refused']} refused + "
            f"{first['skipped_no_expected']} no-expected-citations questions.[/dim]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
