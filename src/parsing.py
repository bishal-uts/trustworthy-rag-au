"""PDF -> structured JSON.

Output JSON schema (one file per document at data/chunks/<doc_id>.parsed.json):

    {
      "doc_id": "cps234",
      "title": "...",
      "source_body": "APRA",
      "sections": [
        {
          "section_id": "Notification of incidents",
          "heading": "Notification of incidents",
          "paragraphs": [
            { "para_id": "35", "text": "...", "span_start": 12345, "span_end": 12678 }
          ]
        }
      ]
    }

Parsing strategy:
 1. Extract text per page with pdfplumber (preserves reading order).
 2. Concatenate to a single string with a recorded char offset per line.
 3. Split into sections using heading-detection regexes per source body
    (APRA / ASIC / legislation.gov.au all number sections differently).
 4. Within each section, split into paragraphs using paragraph-numbering
    regex (e.g. APRA uses "(\\d+)\\." at line start). Paragraphs without a
    detectable number get a positional id like "p3" with a provenance flag.
 5. Span offsets index into the original concatenated text so we can later
    surface exact citation spans.

This module is intentionally simple. Per-document quirks (footnote handling,
table extraction, scanned pages) are handled by per-doc strategies that can
be added later if a document doesn't parse cleanly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber
import yaml
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PDFS_DIR = REPO_ROOT / "data" / "raw_pdfs"
CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
SOURCES_FILE = REPO_ROOT / "data" / "sources.yaml"

console = Console()


# Heading-detection patterns per source body. Each pattern matches a line
# that begins a new section. APRA uses bold all-caps or numbered headings;
# ASIC RGs use "Section X" or numbered headings; Acts use "Part X / Division Y".
HEADING_PATTERNS = {
    "APRA": [
        re.compile(r"^\s*(\d+)\.\s+([A-Z][A-Za-z][^\n]{2,80})\s*$"),
        re.compile(r"^\s*([A-Z][A-Z\s]{3,60})\s*$"),
    ],
    "ASIC": [
        re.compile(r"^\s*Section\s+([A-Z]):\s+(.{2,80})\s*$"),
        re.compile(r"^\s*([A-Z][A-Z\s]{3,60})\s*$"),
        re.compile(r"^\s*RG\s*\d+\.\d+\s+(.{2,80})\s*$"),
    ],
    "Federal Register of Legislation": [
        re.compile(r"^\s*(Part|Division|Subdivision)\s+([0-9IVX]+).?\s+(.{2,80})\s*$", re.I),
        re.compile(r"^\s*Section\s+(\d+[A-Z]*)\s+(.{2,80})\s*$", re.I),
        re.compile(r"^\s*Schedule\s+(\d+)\s+(.{2,80})\s*$", re.I),
    ],
}

# Paragraph numbering — the leading "35." style used by APRA and ASIC RGs.
PARA_NUM_RE = re.compile(r"^\s*(\d{1,3})\.\s+(.+)", re.S)
# Acts use "(1)", "(2)" subsection numbering.
ACT_SUBSEC_RE = re.compile(r"^\s*\((\d+[a-z]?)\)\s+(.+)", re.S)


@dataclass
class Paragraph:
    para_id: str
    text: str
    span_start: int
    span_end: int
    para_id_provenance: str = "extracted"  # "extracted" | "positional"


@dataclass
class Section:
    section_id: str
    heading: str
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    source_body: str
    sections: list[Section] = field(default_factory=list)


def extract_full_text(pdf_path: Path) -> str:
    """Concatenate all pages with page-break markers preserved as newlines."""
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
    return "\n\n".join(pages)


def detect_section_starts(lines: list[str], source_body: str) -> list[tuple[int, str]]:
    """Return list of (line_index, heading_text). Always includes 0."""
    patterns = HEADING_PATTERNS.get(source_body, HEADING_PATTERNS["APRA"])
    starts: list[tuple[int, str]] = [(0, "Preamble")]
    for i, line in enumerate(lines):
        for pat in patterns:
            m = pat.match(line)
            if m:
                heading = " ".join(m.groups()).strip()
                # Avoid duplicates on the same line index
                if not starts or starts[-1][0] != i:
                    starts.append((i, heading))
                break
    # Deduplicate consecutive identical headings
    deduped: list[tuple[int, str]] = []
    for s in starts:
        if not deduped or deduped[-1][1] != s[1]:
            deduped.append(s)
    return deduped


def split_paragraphs(
    section_text: str, section_offset: int, source_body: str
) -> list[Paragraph]:
    """Split a section into numbered paragraphs.

    section_offset is the char offset of section_text within the full doc.
    """
    paragraph_re = ACT_SUBSEC_RE if source_body == "Federal Register of Legislation" else PARA_NUM_RE

    # Split on blank lines first to get raw paragraph chunks
    raw_paragraphs = re.split(r"\n\s*\n", section_text)
    out: list[Paragraph] = []
    cursor = 0  # offset within section_text
    positional_idx = 0

    for raw in raw_paragraphs:
        text = raw.strip()
        if not text:
            cursor += len(raw) + 2  # +2 for the blank-line separator we split on
            continue

        # Locate the raw chunk in section_text starting from cursor
        idx = section_text.find(raw, cursor)
        if idx < 0:
            idx = cursor
        cursor = idx + len(raw)

        m = paragraph_re.match(text)
        if m:
            para_id = m.group(1)
            provenance = "extracted"
        else:
            positional_idx += 1
            para_id = f"p{positional_idx}"
            provenance = "positional"

        out.append(
            Paragraph(
                para_id=para_id,
                text=text,
                span_start=section_offset + idx,
                span_end=section_offset + idx + len(raw),
                para_id_provenance=provenance,
            )
        )

    return out


def parse_pdf(doc_meta: dict) -> ParsedDoc:
    pdf_path = RAW_PDFS_DIR / f"{doc_meta['id']}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}. Run `python -m data.download --doc {doc_meta['id']}` first."
        )

    full_text = extract_full_text(pdf_path)
    lines = full_text.split("\n")

    # Map line index -> char offset in full_text
    line_offsets = [0]
    for line in lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    section_starts = detect_section_starts(lines, doc_meta["source_body"])

    sections: list[Section] = []
    for k, (line_idx, heading) in enumerate(section_starts):
        next_line_idx = section_starts[k + 1][0] if k + 1 < len(section_starts) else len(lines)
        offset_start = line_offsets[line_idx]
        offset_end = line_offsets[next_line_idx] if next_line_idx < len(line_offsets) else len(full_text)
        section_text = full_text[offset_start:offset_end]

        paras = split_paragraphs(section_text, offset_start, doc_meta["source_body"])
        if not paras:
            continue

        # Use the heading itself as the section id; uniqify if needed
        section_id = heading
        suffix = 1
        existing_ids = {s.section_id for s in sections}
        while section_id in existing_ids:
            suffix += 1
            section_id = f"{heading} ({suffix})"

        sections.append(Section(section_id=section_id, heading=heading, paragraphs=paras))

    return ParsedDoc(
        doc_id=doc_meta["id"],
        title=doc_meta["title"],
        source_body=doc_meta["source_body"],
        sections=sections,
    )


def write_parsed(doc: ParsedDoc) -> Path:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHUNKS_DIR / f"{doc.doc_id}.parsed.json"
    payload = asdict(doc)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def parse_one(doc_id: str, sources: list[dict]) -> ParsedDoc | None:
    matches = [d for d in sources if d["id"] == doc_id]
    if not matches:
        console.print(f"[red]Unknown doc id: {doc_id}[/red]")
        return None
    try:
        doc = parse_pdf(matches[0])
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return None

    n_sections = len(doc.sections)
    n_paras = sum(len(s.paragraphs) for s in doc.sections)
    out = write_parsed(doc)
    console.print(
        f"[green]ok[/green] {doc.doc_id}: {n_sections} sections, {n_paras} paragraphs -> "
        f"{out.relative_to(REPO_ROOT)}"
    )
    if n_paras == 0:
        console.print(
            f"[yellow]warn[/yellow] {doc.doc_id} produced 0 paragraphs — "
            f"PDF may be scanned or use unusual layout. Inspect the file manually."
        )
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse regulatory PDFs to structured JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Parse all documents in sources.yaml")
    group.add_argument("--doc", help="Parse a single document by id")
    args = parser.parse_args()

    with SOURCES_FILE.open("r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)["documents"]

    if args.doc:
        ok = parse_one(args.doc, sources) is not None
        return 0 if ok else 1

    # --all
    successes = sum(1 for d in sources if parse_one(d["id"], sources) is not None)
    console.print(f"\n[bold]{successes}/{len(sources)} parsed.[/bold]")
    return 0 if successes == len(sources) else 1


if __name__ == "__main__":
    sys.exit(main())
