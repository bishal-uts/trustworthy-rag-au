"""Convenience script: pull Ollama models and pre-warm HuggingFace caches.

Usage:
    python scripts/pull_models.py --essentials    # pull qwen2.5-7b, llama3.1-8b, qwen2.5-3b
    python scripts/pull_models.py --all-llms      # pull every LLM in models.yaml
    python scripts/pull_models.py --llm <id>      # pull one
    python scripts/pull_models.py --hf-warm       # download HF embedder + NLI to local cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from src.config import settings  # noqa: E402
from src.models.llm import OllamaLLM, OllamaNotRunning  # noqa: E402
from src.models.registry import (  # noqa: E402
    get_llm_spec,
    list_embedders,
    list_llms,
    list_nli,
)

console = Console()

ESSENTIALS = ["qwen3.5-9b", "qwen3-8b", "deepseek-r1-8b"]


def pull_llm(llm_id: str) -> bool:
    spec = get_llm_spec(llm_id)
    llm = OllamaLLM(spec)
    if llm.is_pulled():
        console.print(f"[yellow]skip[/yellow] {llm_id}: already pulled ({spec.ollama_tag})")
        return True
    console.print(f"[cyan]Pulling[/cyan] {llm_id} ({spec.ollama_tag})...")
    try:
        last_status = ""
        for evt in llm.pull_model():
            status = evt.get("status", "")
            if status != last_status:
                completed = evt.get("completed", 0)
                total = evt.get("total", 0)
                if total:
                    pct = (completed / total) * 100 if total else 0
                    console.print(f"  {status} ({pct:.0f}%)", end="\r")
                else:
                    console.print(f"  {status}")
                last_status = status
        console.print(f"[green]ok[/green] {llm_id}")
        return True
    except OllamaNotRunning as e:
        console.print(f"[red]fail[/red] {llm_id}: {e}")
        return False


def hf_warm() -> bool:
    """Pre-download the default embedder + NLI model into the local HF cache."""
    import os

    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("HF_HOME", str(repo_root / settings.paths.hf_cache_dir))

    ok = True
    # Default embedder
    try:
        from src.models.embed import Embedder
        from src.models.registry import get_embedder_spec

        spec = get_embedder_spec(settings.default_embedder)
        console.print(f"[cyan]Downloading embedder:[/cyan] {spec.display_name} ({spec.hf_id})")
        Embedder(spec)  # constructor downloads + loads
        console.print("[green]ok[/green] embedder cached")
    except Exception as e:
        console.print(f"[red]fail[/red] embedder warm: {e}")
        ok = False

    # Default NLI
    try:
        from src.models.nli import NLIModel
        from src.models.registry import get_nli_spec

        spec = get_nli_spec(settings.default_nli)
        console.print(f"[cyan]Downloading NLI:[/cyan] {spec.display_name} ({spec.hf_id})")
        NLIModel(spec)
        console.print("[green]ok[/green] NLI cached")
    except Exception as e:
        console.print(f"[red]fail[/red] NLI warm: {e}")
        ok = False

    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Pull Ollama models / warm HF caches.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--essentials", action="store_true", help="Pull the day-one LLM set")
    group.add_argument("--all-llms", action="store_true", help="Pull every LLM in models.yaml")
    group.add_argument("--llm", help="Pull a single LLM by id")
    group.add_argument(
        "--hf-warm",
        action="store_true",
        help="Pre-download default embedder + NLI to the local HF cache",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List available models (LLMs, embedders, NLI) without pulling",
    )
    args = parser.parse_args()

    if args.list:
        console.print("[bold]LLMs:[/bold]")
        for s in list_llms():
            console.print(f"  {s.id:30}  {s.display_name}  ({s.ollama_tag})")
        console.print("\n[bold]Embedders:[/bold]")
        for s in list_embedders():
            console.print(f"  {s.id:30}  {s.display_name}")
        console.print("\n[bold]NLI:[/bold]")
        for s in list_nli():
            console.print(f"  {s.id:30}  {s.display_name}")
        return 0

    if args.hf_warm:
        return 0 if hf_warm() else 1

    if args.llm:
        return 0 if pull_llm(args.llm) else 1

    if args.essentials:
        results = [pull_llm(i) for i in ESSENTIALS]
        return 0 if all(results) else 1

    if args.all_llms:
        results = [pull_llm(s.id) for s in list_llms()]
        return 0 if all(results) else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
