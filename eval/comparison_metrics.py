"""End-to-end comparison metrics: trustworthy RAG vs baseline RAGs.

These metrics operate on a list of `BaselineOutput` (one per question) plus
the corresponding benchmark entries. They aggregate to scalars suitable for
a comparison table.

Five core metrics:

  coverage              fraction of questions where the system gave an answer
  accuracy_on_answered  among answered, fraction correct (per a correctness fn)
  effective_accuracy    correct / total (combines coverage + accuracy)
  hallucination_rate    fraction of answered questions with >0 unfaithful claims
  risk_adjusted_loss    total loss under a configurable cost matrix

The "correctness function" is pluggable: cheap (keyword overlap) or
expensive (LLM-as-judge). Both produce {"YES", "PARTIAL", "NO"} and we
treat YES+PARTIAL as correct unless `strict_correct=True`.

Run `python eval/comparison_metrics.py` to execute self-tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

CorrectnessLabel = Literal["YES", "PARTIAL", "NO", "UNKNOWN"]


@dataclass
class BaselineOutput:
    """Uniform output of any baseline. Fields beyond `state` are optional."""

    question_id: str
    state: Literal["confident", "hedged", "refused"]  # all baselines mapped to this
    answer: str | None = None
    citations: list[dict] = field(default_factory=list)  # {doc, section, paragraph, span}
    retrieved_chunks: list[dict] = field(default_factory=list)  # raw chunk dicts
    llm_called: bool = False
    elapsed_ms: int = 0
    # Filled in by the judge / faith-check pass
    correctness: CorrectnessLabel = "UNKNOWN"
    unfaithful_claim_count: int = 0
    total_claim_count: int = 0


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _is_answered(out: BaselineOutput) -> bool:
    """An output 'answered' if it produced an answer (any non-refused state)."""
    return out.state != "refused" and bool(out.answer and out.answer.strip())


def _is_correct(out: BaselineOutput, *, strict: bool = False) -> bool:
    if strict:
        return out.correctness == "YES"
    return out.correctness in ("YES", "PARTIAL")


def coverage(outputs: list[BaselineOutput]) -> float:
    """Fraction of total questions that produced an answer."""
    if not outputs:
        return 0.0
    return sum(1 for o in outputs if _is_answered(o)) / len(outputs)


def accuracy_on_answered(outputs: list[BaselineOutput], *, strict: bool = False) -> float:
    """Among answered questions, fraction correct.

    Returns 0.0 if nothing was answered (avoid div-by-zero).
    """
    answered = [o for o in outputs if _is_answered(o)]
    if not answered:
        return 0.0
    correct = sum(1 for o in answered if _is_correct(o, strict=strict))
    return correct / len(answered)


def effective_accuracy(outputs: list[BaselineOutput], *, strict: bool = False) -> float:
    """Fraction of TOTAL questions answered correctly.

    Combines coverage and accuracy: a system that refuses everything scores 0,
    a system that answers everything wrongly scores 0, only correct+answered
    contributes.
    """
    if not outputs:
        return 0.0
    correct = sum(1 for o in outputs if _is_answered(o) and _is_correct(o, strict=strict))
    return correct / len(outputs)


def hallucination_rate(outputs: list[BaselineOutput]) -> float:
    """Fraction of answered questions with at least one unfaithful claim.

    Requires that the judge pass populated `unfaithful_claim_count`.
    """
    answered = [o for o in outputs if _is_answered(o)]
    if not answered:
        return 0.0
    halluc = sum(1 for o in answered if o.unfaithful_claim_count > 0)
    return halluc / len(answered)


def refusal_precision(
    outputs: list[BaselineOutput],
    expected_states: dict[str, str],
) -> float:
    """Of refused outputs, fraction whose expected_state was actually 'refused'.

    Measures whether the system refuses for the RIGHT reasons (legit
    out-of-scope), not just everything it's unsure about.

    Returns 0.0 if nothing was refused.
    """
    refused = [o for o in outputs if o.state == "refused"]
    if not refused:
        return 0.0
    correct_refusal = sum(
        1 for o in refused if expected_states.get(o.question_id) == "refused"
    )
    return correct_refusal / len(refused)


# ---------------------------------------------------------------------------
# Risk-adjusted loss
# ---------------------------------------------------------------------------


@dataclass
class LossMatrix:
    """Cost of each outcome. Higher = more harmful.

    Defaults reflect a legal/regulatory setting:
     - Hallucinating while confident is by far the worst (user acts on it).
     - Refusing an answerable question is annoying but recoverable.
     - Refusing an unanswerable question is correct behaviour.
    """

    correct_confident: float = 0.0
    correct_hedged: float = 0.5            # right answer but tagged uncertain
    wrong_confident: float = 10.0          # the dangerous case
    wrong_hedged: float = 3.0              # wrong but user was warned
    refused_answerable: float = 1.0        # over-refusal
    refused_unanswerable: float = 0.0      # correct refusal

    def for_outcome(
        self,
        out: BaselineOutput,
        expected_state: str,
        *,
        strict: bool = False,
    ) -> float:
        if out.state == "refused":
            return (
                self.refused_unanswerable
                if expected_state == "refused"
                else self.refused_answerable
            )
        correct = _is_correct(out, strict=strict)
        if out.state == "confident":
            return self.correct_confident if correct else self.wrong_confident
        if out.state == "hedged":
            return self.correct_hedged if correct else self.wrong_hedged
        return 0.0  # shouldn't reach


def risk_adjusted_loss(
    outputs: list[BaselineOutput],
    expected_states: dict[str, str],
    loss: LossMatrix | None = None,
    *,
    strict: bool = False,
) -> float:
    """Total loss across all outputs under the given loss matrix."""
    loss = loss or LossMatrix()
    return sum(
        loss.for_outcome(o, expected_states.get(o.question_id, "confident"), strict=strict)
        for o in outputs
    )


# ---------------------------------------------------------------------------
# Aggregation helper for the comparison table
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    system: str
    n_total: int
    coverage: float
    accuracy_on_answered: float
    effective_accuracy: float
    hallucination_rate: float
    refusal_precision: float
    risk_adjusted_loss: float
    mean_elapsed_ms: float


def summarise(
    system: str,
    outputs: list[BaselineOutput],
    expected_states: dict[str, str],
    loss: LossMatrix | None = None,
    *,
    strict: bool = False,
) -> ComparisonRow:
    return ComparisonRow(
        system=system,
        n_total=len(outputs),
        coverage=coverage(outputs),
        accuracy_on_answered=accuracy_on_answered(outputs, strict=strict),
        effective_accuracy=effective_accuracy(outputs, strict=strict),
        hallucination_rate=hallucination_rate(outputs),
        refusal_precision=refusal_precision(outputs, expected_states),
        risk_adjusted_loss=risk_adjusted_loss(outputs, expected_states, loss, strict=strict),
        mean_elapsed_ms=(
            sum(o.elapsed_ms for o in outputs) / len(outputs) if outputs else 0.0
        ),
    )


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------


def summarise_by_category(
    system: str,
    outputs: list[BaselineOutput],
    categories: dict[str, str],          # question_id -> category
    expected_states: dict[str, str],
    loss: LossMatrix | None = None,
    *,
    strict: bool = False,
) -> dict[str, ComparisonRow]:
    """Group outputs by question category, return one ComparisonRow per category."""
    groups: dict[str, list[BaselineOutput]] = {}
    for o in outputs:
        cat = categories.get(o.question_id, "unknown")
        groups.setdefault(cat, []).append(o)
    return {
        cat: summarise(system, group, expected_states, loss, strict=strict)
        for cat, group in groups.items()
    }


# ---------------------------------------------------------------------------
# Built-in cheap correctness function: keyword overlap
# ---------------------------------------------------------------------------


def keyword_correctness(
    answer: str | None,
    keywords: list[str],
    *,
    min_fraction: float = 1.0,
) -> CorrectnessLabel:
    """Label correctness by keyword presence in answer.

    All keywords present     -> YES
    Some keywords present    -> PARTIAL (if fraction >= 0.5)
    None / fewer than 0.5    -> NO
    No keywords specified    -> UNKNOWN
    """
    if not keywords:
        return "UNKNOWN"
    if not answer:
        return "NO"
    answer_l = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_l)
    frac = hits / len(keywords)
    if frac >= min_fraction:
        return "YES"
    if frac >= 0.5:
        return "PARTIAL"
    return "NO"


# ---------------------------------------------------------------------------
# Self-tests. Run: `python eval/comparison_metrics.py`
# ---------------------------------------------------------------------------


def _t(qid: str, state: str, answer: str | None, correctness: CorrectnessLabel = "UNKNOWN",
       unfaithful: int = 0) -> BaselineOutput:
    return BaselineOutput(
        question_id=qid,
        state=state,  # type: ignore[arg-type]
        answer=answer,
        correctness=correctness,
        unfaithful_claim_count=unfaithful,
        total_claim_count=max(1, unfaithful),
    )


def _run_self_tests() -> None:
    # 4 questions, varied outcomes
    outputs = [
        _t("q1", "confident", "72 hours.", correctness="YES"),                 # right
        _t("q2", "confident", "24 hours.", correctness="NO", unfaithful=1),    # wrong + halluc
        _t("q3", "hedged",    "Probably APP 11.", correctness="PARTIAL"),      # partial
        _t("q4", "refused",   None),                                            # refused
    ]
    expected_states = {"q1": "confident", "q2": "confident", "q3": "confident", "q4": "refused"}

    # Coverage
    assert coverage(outputs) == 0.75, f"coverage: {coverage(outputs)}"

    # Accuracy on answered: 3 answered (q1, q2, q3); 2 correct (YES + PARTIAL) → 2/3
    acc = accuracy_on_answered(outputs)
    assert abs(acc - 2 / 3) < 1e-9, f"acc: {acc}"

    # Strict accuracy: only YES → 1/3
    acc_strict = accuracy_on_answered(outputs, strict=True)
    assert abs(acc_strict - 1 / 3) < 1e-9, f"acc_strict: {acc_strict}"

    # Effective accuracy: 2 correct out of 4 total → 0.5
    assert effective_accuracy(outputs) == 0.5

    # Hallucination rate: 1 of 3 answered has unfaithful > 0
    assert abs(hallucination_rate(outputs) - 1 / 3) < 1e-9

    # Refusal precision: q4 refused, expected refused → 1/1
    assert refusal_precision(outputs, expected_states) == 1.0

    # Risk-adjusted loss (default matrix)
    # q1: correct_confident = 0
    # q2: wrong_confident = 10
    # q3: correct_hedged = 0.5 (PARTIAL still counts as correct under default)
    # q4: refused_unanswerable = 0
    total = risk_adjusted_loss(outputs, expected_states)
    assert abs(total - 10.5) < 1e-9, f"risk loss: {total}"

    # Strict mode: PARTIAL is no longer correct → q3 becomes wrong_hedged (3.0)
    total_strict = risk_adjusted_loss(outputs, expected_states, strict=True)
    assert abs(total_strict - 13.0) < 1e-9, f"risk loss strict: {total_strict}"

    # Custom loss matrix
    cheap_matrix = LossMatrix(wrong_confident=1.0, wrong_hedged=1.0, refused_answerable=5.0)
    total_cheap = risk_adjusted_loss(outputs, expected_states, cheap_matrix)
    # q1=0, q2=1, q3=0.5, q4=0 → 1.5
    assert abs(total_cheap - 1.5) < 1e-9

    # ComparisonRow
    row = summarise("TestSystem", outputs, expected_states)
    assert row.system == "TestSystem"
    assert row.n_total == 4
    assert row.coverage == 0.75

    # Per-category
    categories = {"q1": "easy", "q2": "easy", "q3": "hard", "q4": "out_of_scope"}
    per_cat = summarise_by_category("TestSystem", outputs, categories, expected_states)
    assert set(per_cat.keys()) == {"easy", "hard", "out_of_scope"}
    assert per_cat["easy"].n_total == 2
    assert per_cat["out_of_scope"].coverage == 0.0

    # Keyword correctness
    assert keyword_correctness("Within 72 hours of becoming aware.", ["72 hours", "aware"]) == "YES"
    # 2 of 3 keywords present → PARTIAL (>= 0.5)
    assert keyword_correctness(
        "Within 72 hours of becoming aware.", ["72 hours", "aware", "material"]
    ) == "PARTIAL"
    # 1 of 3 keywords present → NO (< 0.5)
    assert keyword_correctness("Within 72 hours.", ["72 hours", "aware", "material"]) == "NO"
    assert keyword_correctness("I don't know.", ["72 hours", "aware"]) == "NO"
    assert keyword_correctness(None, ["72 hours"]) == "NO"
    assert keyword_correctness("Anything.", []) == "UNKNOWN"

    # Edge cases
    assert coverage([]) == 0.0
    assert accuracy_on_answered([]) == 0.0
    assert hallucination_rate([]) == 0.0
    assert refusal_precision([], {}) == 0.0

    print("OK — all comparison_metrics self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
