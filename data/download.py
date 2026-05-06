"""Download the 7 regulatory PDFs into data/raw_pdfs/.

Usage:
    python -m data.download --all
    python -m data.download --doc cps234
    python -m data.download --check          # verify URLs without downloading

URLs live in data/sources.yaml. If a URL has rotted, edit that file and
re-run. Failures are reported per-document; partial success is fine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES_FILE = REPO_ROOT / "data" / "sources.yaml"
RAW_PDFS_DIR = REPO_ROOT / "data" / "raw_pdfs"

console = Console()


def load_sources() -> list[dict]:
    with SOURCES_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["documents"]


def head_check(url: str) -> tuple[bool, str]:
    """HEAD request to verify the URL responds. Returns (ok, message)."""
    try:
        with httpx.Client(follow_redirects=True, timeout=15.0) as client:
            r = client.head(url)
            if r.status_code == 200:
                size = r.headers.get("content-length", "?")
                ctype = r.headers.get("content-type", "?")
                return True, f"OK · {ctype} · {size} bytes"
            # Some sites disallow HEAD; try GET with stream=True and abort
            r = client.get(url, follow_redirects=True)
            if r.status_code == 200:
                return True, f"OK (via GET) · {r.headers.get('content-type', '?')}"
            return False, f"HTTP {r.status_code}"
    except httpx.HTTPError as e:
        return False, f"HTTP error: {type(e).__name__}: {e}"


def download_one(doc: dict) -> bool:
    """Download a single document's PDF. Returns True on success."""
    doc_id = doc["id"]
    url = doc["url"]
    target = RAW_PDFS_DIR / f"{doc_id}.pdf"

    if target.exists():
        console.print(f"[yellow]skip[/yellow] {doc_id}: already at {target.relative_to(REPO_ROOT)}")
        return True

    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(follow_redirects=True, timeout=60.0) as client, client.stream(
            "GET", url
        ) as r:
            if r.status_code != 200:
                console.print(f"[red]fail[/red] {doc_id}: HTTP {r.status_code} for {url}")
                return False

            content_type = r.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                console.print(
                    f"[yellow]warn[/yellow] {doc_id}: content-type '{content_type}' is not PDF — "
                    f"the URL may serve an HTML download page rather than the PDF itself. "
                    f"Save manually to {target.relative_to(REPO_ROOT)}."
                )
                return False

            total = int(r.headers.get("content-length", 0)) or None

            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(f"  {doc_id}", total=total)
                with target.open("wb") as f:
                    for chunk in r.iter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        size_kb = target.stat().st_size // 1024
        console.print(f"[green]ok  [/green] {doc_id}: {size_kb} KB")
        return True

    except httpx.HTTPError as e:
        console.print(f"[red]fail[/red] {doc_id}: {type(e).__name__}: {e}")
        # Clean up partial download
        if target.exists() and target.stat().st_size == 0:
            target.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download regulatory PDFs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download all 7 documents")
    group.add_argument("--doc", help="Download a single document by id (e.g. cps234)")
    group.add_argument("--check", action="store_true", help="HEAD-check URLs without downloading")
    args = parser.parse_args()

    docs = load_sources()

    if args.check:
        console.print("[bold]Verifying URLs...[/bold]")
        all_ok = True
        for doc in docs:
            ok, msg = head_check(doc["url"])
            colour = "green" if ok else "red"
            console.print(f"  [{colour}]{doc['id']:14}[/{colour}] {msg}  ({doc['url']})")
            all_ok = all_ok and ok
        return 0 if all_ok else 1

    if args.doc:
        matches = [d for d in docs if d["id"] == args.doc]
        if not matches:
            console.print(f"[red]Unknown doc id: {args.doc}[/red]")
            console.print(f"Available: {', '.join(d['id'] for d in docs)}")
            return 2
        ok = download_one(matches[0])
        return 0 if ok else 1

    # --all
    results = [download_one(d) for d in docs]
    succeeded = sum(results)
    total = len(results)
    console.print(f"\n[bold]{succeeded}/{total} downloaded.[/bold]")
    if succeeded < total:
        console.print(
            "[yellow]For failed downloads, visit the source body's website and place the "
            "PDF manually at data/raw_pdfs/<id>.pdf — then re-run the parsing step.[/yellow]"
        )
    return 0 if succeeded == total else 1


if __name__ == "__main__":
    sys.exit(main())
