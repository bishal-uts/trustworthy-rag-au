"""Generation: prompt rendering + Ollama call + JSON parse.

Two entry points:
 - generate_answer(question, chunks, llm_id, hedged=False) -> AnswerDraft
 - render_refusal(reason, nearest_chunks) -> str

The answer prompt asks for strict JSON output. We use Ollama's JSON mode
when supported. JSON parse errors fall back to plain-text answer with
state forced to "hedged" — defensive, never crashes the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.models.manager import manager
from src.schemas import Chunk, Citation

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)


@dataclass
class AnswerDraft:
    text: str                       # the answer text the user sees
    citations: list[Citation]
    raw_json: dict | None           # parsed JSON if successful, else None
    parse_ok: bool                  # whether JSON parse succeeded
    confidence_self_assessment: str # high | medium | low | unknown
    notes: str                      # model's caveat field, if any
    llm_called: str                 # model tag actually used
    eval_count: int                 # tokens generated
    eval_duration_ms: int


def render_answer_prompt(question: str, chunks: list[Chunk], hedged: bool) -> str:
    template = _env.get_template("answer.j2")
    return template.render(question=question, chunks=chunks, hedged=hedged)


def render_refusal(reason: str, nearest_chunks: list[Chunk]) -> str:
    template = _env.get_template("refusal.j2")
    return template.render(reason=reason, nearest_chunks=nearest_chunks)


def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extraction. Models sometimes wrap JSON in fences or prose."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for the first { and the matching final }
    start = text.find("{")
    if start < 0:
        return None
    # Scan forward, tracking brace depth, ignoring braces inside strings
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _parse_citations(raw: list | None) -> list[Citation]:
    if not raw:
        return []
    out: list[Citation] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        try:
            out.append(
                Citation(
                    doc=str(c.get("doc", "")),
                    section=str(c.get("section", "")),
                    paragraph=str(c.get("paragraph", "")),
                    span=str(c.get("span", "")),
                )
            )
        except Exception:
            continue
    return out


def generate_answer(
    question: str,
    chunks: list[Chunk],
    llm_id: str,
    hedged: bool = False,
) -> AnswerDraft:
    prompt = render_answer_prompt(question, chunks, hedged=hedged)
    llm = manager.get_llm(llm_id)
    result = llm.generate(prompt)

    parsed = _extract_json(result.text)
    if parsed is None:
        # Fallback: treat the whole response as plain text answer
        return AnswerDraft(
            text=result.text.strip(),
            citations=[],
            raw_json=None,
            parse_ok=False,
            confidence_self_assessment="unknown",
            notes="JSON parse failed; output may be unstructured.",
            llm_called=result.model_called,
            eval_count=result.eval_count,
            eval_duration_ms=result.eval_duration_ms,
        )

    return AnswerDraft(
        text=str(parsed.get("answer", "")).strip(),
        citations=_parse_citations(parsed.get("citations")),
        raw_json=parsed,
        parse_ok=True,
        confidence_self_assessment=str(parsed.get("confidence_self_assessment", "unknown")).lower(),
        notes=str(parsed.get("notes", "")),
        llm_called=result.model_called,
        eval_count=result.eval_count,
        eval_duration_ms=result.eval_duration_ms,
    )
