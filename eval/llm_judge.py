"""Judges for assigning correctness labels (YES / PARTIAL / NO).

Three judges:

  KeywordJudge   — fast, deterministic. Requires `expected_answer_keywords`
                   in the benchmark entry. No model load, no LLM call.
  LLMJudge       — open-ended correctness. Uses a separate LLM (different from
                   the one being evaluated) to avoid self-favourability bias.
  NLIJudge       — runs the existing NLI model: does expected_answer entail
                   the actual answer? Cheap (already loaded for faithfulness).

`CompositeJudge` tries cheaper judges first and falls back to LLM only when
needed — saves a lot of LLM calls during a 100-question run.

Results are cached to disk so re-running with the same (benchmark+system+answer)
doesn't repay the cost.
"""

from __future__ import annotations

import hashlib
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Allow `python eval/llm_judge.py` (direct) in addition to `python -m eval.llm_judge`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.comparison_metrics import CorrectnessLabel, keyword_correctness  # noqa: E402
from src.models.manager import manager  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "eval" / ".judge_cache"


@dataclass
class JudgeResult:
    label: CorrectnessLabel
    reasoning: str = ""
    judge_name: str = ""
    cached: bool = False


@dataclass
class JudgeInput:
    question: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    actual_answer: str | None = None


def _cache_key(judge_name: str, inp: JudgeInput) -> str:
    payload = json.dumps(
        {
            "judge": judge_name,
            "q": inp.question,
            "expected": inp.expected_answer,
            "kw": sorted(inp.expected_keywords),
            "actual": inp.actual_answer,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_get(key: str) -> JudgeResult | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return JudgeResult(
            label=data["label"],
            reasoning=data.get("reasoning", ""),
            judge_name=data.get("judge_name", ""),
            cached=True,
        )
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_put(key: str, res: JudgeResult) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(
        json.dumps(
            {"label": res.label, "reasoning": res.reasoning, "judge_name": res.judge_name},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------


class Judge(ABC):
    name: str

    @abstractmethod
    def can_judge(self, inp: JudgeInput) -> bool:
        ...

    @abstractmethod
    def _judge_impl(self, inp: JudgeInput) -> JudgeResult:
        ...

    def judge(self, inp: JudgeInput, *, use_cache: bool = True) -> JudgeResult:
        if not self.can_judge(inp):
            return JudgeResult(label="UNKNOWN", judge_name=self.name, reasoning="no signal available")
        key = _cache_key(self.name, inp)
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached
        res = self._judge_impl(inp)
        res.judge_name = self.name
        if use_cache:
            _cache_put(key, res)
        return res


# ---------------------------------------------------------------------------
# Keyword judge: cheapest, fully deterministic
# ---------------------------------------------------------------------------


class KeywordJudge(Judge):
    name = "keyword"

    def can_judge(self, inp: JudgeInput) -> bool:
        return bool(inp.expected_keywords) and bool(inp.actual_answer)

    def _judge_impl(self, inp: JudgeInput) -> JudgeResult:
        label = keyword_correctness(inp.actual_answer, inp.expected_keywords)
        return JudgeResult(
            label=label,
            reasoning=(
                f"matched {sum(1 for k in inp.expected_keywords if k.lower() in (inp.actual_answer or '').lower())}"
                f"/{len(inp.expected_keywords)} keywords"
            ),
        )


# ---------------------------------------------------------------------------
# NLI judge: requires expected_answer; uses existing loaded NLI model
# ---------------------------------------------------------------------------


class NLIJudge(Judge):
    name = "nli"

    def __init__(self, nli_id: str, entail_floor: float = 0.5, partial_floor: float = 0.3):
        self.nli_id = nli_id
        self.entail_floor = entail_floor
        self.partial_floor = partial_floor

    def can_judge(self, inp: JudgeInput) -> bool:
        return bool(inp.expected_answer) and bool(inp.actual_answer)

    def _judge_impl(self, inp: JudgeInput) -> JudgeResult:
        nli = manager.get_nli(self.nli_id)
        # Bidirectional check: do they entail each other?
        forward = nli.entail(premise=inp.expected_answer or "", hypothesis=inp.actual_answer or "")
        backward = nli.entail(premise=inp.actual_answer or "", hypothesis=inp.expected_answer or "")
        mean_entail = (forward.entail + backward.entail) / 2
        if mean_entail >= self.entail_floor:
            label: CorrectnessLabel = "YES"
        elif mean_entail >= self.partial_floor:
            label = "PARTIAL"
        else:
            label = "NO"
        return JudgeResult(
            label=label,
            reasoning=f"NLI mean_entail={mean_entail:.2f} (fwd={forward.entail:.2f}, bwd={backward.entail:.2f})",
        )


# ---------------------------------------------------------------------------
# LLM judge: most expensive, most general
# ---------------------------------------------------------------------------


JUDGE_PROMPT_TEMPLATE = """You are evaluating answers to questions about Australian financial regulation.

QUESTION:
{question}

REFERENCE ANSWER (assumed correct):
{expected}

CANDIDATE ANSWER (being evaluated):
{actual}

Does the CANDIDATE answer convey the same factual content as the REFERENCE answer for the QUESTION?

Reply with a single JSON object, no markdown fences:
{{"label": "YES" | "PARTIAL" | "NO", "reasoning": "<one sentence>"}}

- YES: candidate states the same key facts as reference (paraphrase OK)
- PARTIAL: candidate gets some key facts but misses or adds others
- NO: candidate contradicts or is unrelated to the reference
"""


class LLMJudge(Judge):
    name = "llm"

    def __init__(self, judge_llm_id: str):
        # IMPORTANT: pass an LLM id that's DIFFERENT from the system under test
        # to avoid self-favourability bias.
        self.judge_llm_id = judge_llm_id

    def can_judge(self, inp: JudgeInput) -> bool:
        return bool(inp.expected_answer) and bool(inp.actual_answer)

    def _judge_impl(self, inp: JudgeInput) -> JudgeResult:
        llm = manager.get_llm(self.judge_llm_id)
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=inp.question,
            expected=inp.expected_answer,
            actual=inp.actual_answer,
        )
        result = llm.generate(prompt)
        # Lenient parse — judge LLM may wrap JSON
        text = result.text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract first JSON object
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = {}

        raw_label = str(parsed.get("label", "UNKNOWN")).upper().strip()
        label: CorrectnessLabel = (
            raw_label if raw_label in ("YES", "PARTIAL", "NO") else "UNKNOWN"  # type: ignore[assignment]
        )
        reasoning = str(parsed.get("reasoning", text[:200]))
        return JudgeResult(label=label, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Composite: try keyword → NLI → LLM in order
# ---------------------------------------------------------------------------


class CompositeJudge(Judge):
    """Tries cheaper judges first, falls back to more expensive ones.

    Returns the first concrete (non-UNKNOWN) label. Saves LLM cost on
    questions where keywords suffice.
    """

    name = "composite"

    def __init__(self, judges: list[Judge]):
        if not judges:
            raise ValueError("CompositeJudge needs at least one sub-judge")
        self.judges = judges

    def can_judge(self, inp: JudgeInput) -> bool:
        return any(j.can_judge(inp) for j in self.judges)

    def _judge_impl(self, inp: JudgeInput) -> JudgeResult:
        last: JudgeResult | None = None
        for j in self.judges:
            if not j.can_judge(inp):
                continue
            res = j.judge(inp)
            if res.label != "UNKNOWN":
                # annotate which sub-judge produced the verdict
                return JudgeResult(
                    label=res.label,
                    reasoning=f"[{res.judge_name}] {res.reasoning}",
                )
            last = res
        return last or JudgeResult(label="UNKNOWN", reasoning="no judge could decide")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_default_judge(
    nli_id: str | None,
    llm_judge_id: str | None,
) -> CompositeJudge:
    """Keyword → (optional) NLI → (optional) LLM."""
    judges: list[Judge] = [KeywordJudge()]
    if nli_id:
        judges.append(NLIJudge(nli_id=nli_id))
    if llm_judge_id:
        judges.append(LLMJudge(judge_llm_id=llm_judge_id))
    return CompositeJudge(judges)


# ---------------------------------------------------------------------------
# Self-tests (KeywordJudge only — NLI/LLM need real models)
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
    # Keyword judge
    kj = KeywordJudge()

    inp_yes = JudgeInput(
        question="When must APRA be notified?",
        expected_keywords=["72 hours", "aware"],
        actual_answer="Within 72 hours of becoming aware.",
    )
    res = kj.judge(inp_yes, use_cache=False)
    assert res.label == "YES", f"expected YES, got {res.label}"

    inp_no = JudgeInput(
        question="When must APRA be notified?",
        expected_keywords=["72 hours", "aware"],
        actual_answer="I don't know.",
    )
    res = kj.judge(inp_no, use_cache=False)
    assert res.label == "NO"

    inp_no_kw = JudgeInput(
        question="Q",
        expected_keywords=[],
        actual_answer="anything",
    )
    assert not kj.can_judge(inp_no_kw)
    res = kj.judge(inp_no_kw, use_cache=False)
    assert res.label == "UNKNOWN"

    # Composite fallback (only keyword available)
    composite = CompositeJudge([KeywordJudge()])
    res = composite.judge(inp_yes, use_cache=False)
    assert res.label == "YES"
    assert "[keyword]" in res.reasoning

    # Cache key stability
    k1 = _cache_key("foo", inp_yes)
    k2 = _cache_key("foo", inp_yes)
    assert k1 == k2
    k3 = _cache_key("bar", inp_yes)
    assert k1 != k3

    print("OK — llm_judge self-tests passed (KeywordJudge + Composite).")


if __name__ == "__main__":
    _run_self_tests()
