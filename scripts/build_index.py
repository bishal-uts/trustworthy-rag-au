"""Build BM25 and FAISS dense indexes from data/chunks/*.chunks.jsonl.

Usage:
    python scripts/build_index.py                      # default embedder
    python scripts/build_index.py --embedder bge-base  # specific embedder
    python scripts/build_index.py --all-embedders      # rebuild all in models.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from src.chunking import load_all_chunks  # noqa: E402
from src.config import settings  # noqa: E402
from src.models.registry import list_embedders  # noqa: E402
from src.retrieval.bm25 import BM25Index  # noqa: E402
from src.retrieval.dense import DenseIndex  # noqa: E402

console = Console()


def build_bm25(chunks: list[dict]) -> None:
    t0 = time.time()
    idx = BM25Index.build(chunks)
    idx.save()
    console.print(f"[green]BM25 index built in {time.time() - t0:.1f}s · {len(chunks)} chunks[/green]")


def build_dense(embedder_id: str, chunks: list[dict]) -> None:
    t0 = time.time()
    console.print(f"[cyan]Building dense index for embedder '{embedder_id}'...[/cyan]")
    idx = DenseIndex.build(embedder_id, chunks)
    idx.save()
    console.print(
        f"[green]Dense index '{embedder_id}' built in {time.time() - t0:.1f}s · {len(chunks)} chunks[/green]"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BM25 + dense indexes.")
    parser.add_argument("--embedder", help="Embedder id (default: settings.default_embedder)")
    parser.add_argument(
        "--all-embedders",
        action="store_true",
        help="Build dense indexes for every embedder in models.yaml",
    )
    parser.add_argument("--bm25-only", action="store_true", help="Skip dense index build")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25 index build")
    args = parser.parse_args()

    chunks = load_all_chunks()
    if not chunks:
        console.print(
            "[red]No chunks found. Run `python -m src.parsing --all` then "
            "`python -m src.chunking --all` first.[/red]"
        )
        return 1
    console.print(f"[bold]Loaded {len(chunks)} chunks[/bold]")

    if not args.dense_only:
        build_bm25(chunks)

    if args.bm25_only:
        return 0

    if args.all_embedders:
        for spec in list_embedders():
            build_dense(spec.id, chunks)
    else:
        embedder_id = args.embedder or settings.default_embedder
        build_dense(embedder_id, chunks)

    return 0


if __name__ == "__main__":
    sys.exit(main())
