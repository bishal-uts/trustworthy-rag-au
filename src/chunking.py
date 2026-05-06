"""Structure-aware chunking.

Input:  data/chunks/<doc_id>.parsed.json (from parsing.py)
Output: data/chunks/<doc_id>.chunks.jsonl (one JSON object per line)

Strategy:
 1. Each paragraph becomes a chunk if it fits in `max_tokens`.
 2. Paragraphs longer than `max_tokens` split at sentence boundaries with
    `overlap_tokens` overlap between adjacent fragments.
 3. Paragraphs shorter than `min_tokens` merge forward into the next
    paragraph in the same section.

Token counting uses a simple regex word-count as an approximation. We don't
need exact LLM token counts — within a factor of 2x is fine for chunk-size
constraints.

Each chunk records {chunk_id, doc, section, paragraph, text, span_start,
span_end, source_body, token_estimate}. The chunk_id is stable across runs:
"<doc>::<section>::<paragraph>[:<frag_idx>]".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console

from src.config import settings

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"

console = Console()

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
TOKEN_RE = re.compile(r"\w+")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    section: str
    paragraph: str
    text: str
    span_start: int
    span_end: int
    source_body: str
    token_estimate: int


def estimate_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def split_long(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a long paragraph at sentence boundaries with token overlap."""
    sentences = SENTENCE_SPLIT_RE.split(text)
    if not sentences:
        return [text]

    fragments: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens <= max_tokens or not current:
            current.append(sent)
            current_tokens += sent_tokens
        else:
            fragments.append(" ".join(current).strip())
            # Take overlap from end of current fragment
            overlap_sentences: list[str] = []
            overlap_tok = 0
            for s in reversed(current):
                s_tok = estimate_tokens(s)
                if overlap_tok + s_tok > overlap_tokens:
                    break
                overlap_sentences.insert(0, s)
                overlap_tok += s_tok
            current = [*overlap_sentences, sent]
            current_tokens = sum(estimate_tokens(s) for s in current)

    if current:
        fragments.append(" ".join(current).strip())
    return fragments


def chunk_doc(parsed_path: Path) -> list[Chunk]:
    payload = json.loads(parsed_path.read_text(encoding="utf-8"))
    doc_id = payload["doc_id"]
    title = payload["title"]
    source_body = payload["source_body"]

    cfg = settings.chunking
    chunks: list[Chunk] = []

    for section in payload["sections"]:
        section_id = section["section_id"]
        merged: list[dict] = []  # paragraphs after small-merge pass
        for para in section["paragraphs"]:
            text = para["text"].strip()
            if not text:
                continue
            if (
                merged
                and estimate_tokens(text) < cfg.min_tokens
                and estimate_tokens(merged[-1]["text"]) < cfg.max_tokens
            ):
                merged[-1]["text"] = merged[-1]["text"] + "\n\n" + text
                merged[-1]["span_end"] = para["span_end"]
                merged[-1]["para_id"] = f"{merged[-1]['para_id']}+{para['para_id']}"
            else:
                merged.append(dict(para))

        for para in merged:
            base_id = f"{doc_id}::{section_id}::{para['para_id']}"
            tokens = estimate_tokens(para["text"])
            if tokens <= cfg.max_tokens:
                chunks.append(
                    Chunk(
                        chunk_id=base_id,
                        doc=title,
                        section=section_id,
                        paragraph=para["para_id"],
                        text=para["text"],
                        span_start=para["span_start"],
                        span_end=para["span_end"],
                        source_body=source_body,
                        token_estimate=tokens,
                    )
                )
            else:
                fragments = split_long(para["text"], cfg.max_tokens, cfg.overlap_tokens)
                for j, frag in enumerate(fragments):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{base_id}:{j}",
                            doc=title,
                            section=section_id,
                            paragraph=para["para_id"],
                            text=frag,
                            span_start=para["span_start"],  # approximate; fragment-level offsets not tracked
                            span_end=para["span_end"],
                            source_body=source_body,
                            token_estimate=estimate_tokens(frag),
                        )
                    )

    return chunks


def write_chunks(doc_id: str, chunks: list[Chunk]) -> Path:
    out_path = CHUNKS_DIR / f"{doc_id}.chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
    return out_path


def chunk_one(doc_id: str) -> int:
    parsed = CHUNKS_DIR / f"{doc_id}.parsed.json"
    if not parsed.exists():
        console.print(f"[red]Missing parsed file: {parsed}. Run `python -m src.parsing --doc {doc_id}` first.[/red]")
        return 0
    chunks = chunk_doc(parsed)
    out = write_chunks(doc_id, chunks)
    console.print(
        f"[green]ok[/green] {doc_id}: {len(chunks)} chunks -> {out.relative_to(REPO_ROOT)}"
    )
    return len(chunks)


def chunk_all() -> int:
    parsed_files = sorted(CHUNKS_DIR.glob("*.parsed.json"))
    if not parsed_files:
        console.print(f"[yellow]No parsed files in {CHUNKS_DIR.relative_to(REPO_ROOT)}. Run parsing first.[/yellow]")
        return 0
    total = 0
    for p in parsed_files:
        doc_id = p.stem.replace(".parsed", "")
        total += chunk_one(doc_id)
    console.print(f"\n[bold]{total} chunks across {len(parsed_files)} documents.[/bold]")
    return total


def load_all_chunks() -> list[dict]:
    """Load every chunk (across all docs) as a list of dicts. Used by index builder."""
    chunks: list[dict] = []
    for p in sorted(CHUNKS_DIR.glob("*.chunks.jsonl")):
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Structure-aware chunking.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Chunk all parsed documents")
    group.add_argument("--doc", help="Chunk a single document by id")
    args = parser.parse_args()

    if args.doc:
        return 0 if chunk_one(args.doc) > 0 else 1
    return 0 if chunk_all() > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
