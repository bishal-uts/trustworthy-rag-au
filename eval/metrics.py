"""Metric helpers for the trustworthy RAG evaluation framework.

Two complementary metric families live here; both are pure functions and
unit-testable without loading a model.

1. RETRIEVAL METRICS (operate on a per-question list of retrieved chunks
   plus the benchmark's expected citations):
     - chunk_matches_expected, first_hit_rank
     - hit_at_k, recall_at_k, reciprocal_rank (MRR per-question)
     - Strict and loose matching modes

2. PIPELINE OUTCOME METRICS (operate on a list of QuestionResult records
   that the benchmark runner builds from RAGOutput + benchmark.yaml):
     - state_accuracy, state_confusion_matrix, refusal_precision_recall
     - retrieval_recall_at_k (doc-level, complements per-chunk recall above)
     - citation_correctness, faithfulness_rate
     - latency_stats, per_category_accuracy
     - Aggregate / aggregate()

Run `python eval/metrics.py` to execute the self-tests for both families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Literal


# =============================================================================
# Section 1 — Retrieval metrics (pure per-question functions)
# =============================================================================
#
# These functions don't depend on the pipeline. Each takes an ordered list of
# retrieved (doc, section, paragraph) tuples plus a list of expected citations
# and returns a score for a single question.
#
# Definitions:
#   Hit@k          binary — at least one expected citation lands in top-k?
#   Recall@k       fraction of expected citations that land in top-k
#   ReciprocalRank 1 / (rank + 1) of first relevant retrieved item; 0 if none
#   MRR            average of ReciprocalRank across queries (caller aggregates)
#
# Two matching modes:
#   loose (default): doc substring match + paragraph exact (if specified).
#                    Benchmark uses short ids ("CPS 234") but parsed chunks
#                    carry full titles ("CPS 234 Information Security") —
#                    substring matching is the practical default.
#   strict:          full equality on (doc, section, paragraph) — useful for
#                    diagnosing paragraph-level retrieval precision.


@dataclass(frozen=True)
class ExpectedCitation:
    """A benchmark's expected (doc, section, paragraph) for one question."""

    doc: str
    section: str = ""
    paragraph: str = ""


def chunk_matches_expected(
    chunk_doc: str,
    chunk_section: str,
    chunk_paragraph: str,
    expected: ExpectedCitation,
    *,
    strict: bool = False,
) -> bool:
    """Does this retrieved chunk satisfy the expected citation?"""
    if strict:
        return (
            chunk_doc.lower() == expected.doc.lower()
            and chunk_section.lower() == expected.section.lower()
            and chunk_paragraph.lower() == expected.paragraph.lower()
        )
    # Loose: doc as substring + paragraph exact (if specified)
    doc_ok = expected.doc.lower() in chunk_doc.lower()
    para_ok = (
        not expected.paragraph
        or chunk_paragraph.lower() == expected.paragraph.lower()
    )
    return doc_ok and para_ok


def first_hit_rank(
    retrieved: list[tuple[str, str, str]],
    expected: list[ExpectedCitation],
    *,
    strict: bool = False,
) -> int | None:
    """0-indexed rank of the first retrieved chunk that matches ANY expected.

    Returns None if no retrieved chunk matches.
    """
    for rank, (d, s, p) in enumerate(retrieved):
        for e in expected:
            if chunk_matches_expected(d, s, p, e, strict=strict):
                return rank
    return None


def hit_at_k(
    retrieved: list[tuple[str, str, str]],
    expected: list[ExpectedCitation],
    k: int,
    *,
    strict: bool = False,
) -> bool:
    """At least one expected citation in top-k retrieved chunks?"""
    if not expected:
        return False
    top_k = retrieved[:k]
    for e in expected:
        if any(
            chunk_matches_expected(d, s, p, e, strict=strict) for d, s, p in top_k
        ):
            return True
    return False


def recall_at_k(
    retrieved: list[tuple[str, str, str]],
    expected: list[ExpectedCitation],
    k: int,
    *,
    strict: bool = False,
) -> float:
    """Fraction of expected citations that land in top-k retrieved chunks."""
    if not expected:
        return 0.0
    top_k = retrieved[:k]
    hit = 0
    for e in expected:
        if any(
            chunk_matches_expected(d, s, p, e, strict=strict) for d, s, p in top_k
        ):
            hit += 1
    return hit / len(expected)


def reciprocal_rank(
    retrieved: list[tuple[str, str, str]],
    expected: list[ExpectedCitation],
    *,
    strict: bool = False,
) -> float:
    """1 / (rank + 1) of the first relevant retrieved item; 0 if none."""
    rank = first_hit_rank(retrieved, expected, strict=strict)
    if rank is None:
        return 0.0
    return 1.0 / (rank + 1)


# =============================================================================
# Section 2 — Pipeline outcome metrics (over QuestionResult records)
# =============================================================================
#
# These metrics operate on the runner's per-question summary. They cover end-
# to-end pipeline behaviour: state classification accuracy, refusal P/R/F1,
# citation correctness, faithfulness rate, latency, per-category breakdown.
#
# They complement the retrieval metrics above: retrieval metrics tell you "did
# we find the right chunk?"; pipeline metrics tell you "given what we found,
# did the system make the right end-to-end decision?".

State = Literal["confident", "hedged", "refused"]
STATES: tuple[State, State, State] = ("confident", "hedged", "refused")

ANSWERABLE_CATEGORIES = {
    "answerable_easy",
    "answerable_hard",
    "answerable_multi_chunk",
    "answerable_crossdoc",
    "answerable_inferential",
    "ambiguous",
}
UNANSWERABLE_CATEGORIES = {
    "unanswerable_in_domain",
    "unanswerable_out_of_scope",
    "adversarial",
    "adversarial_misleading",
}


@dataclass
class QuestionResult:
    """Per-question outcome consumed by every metric helper below."""

    qid: str
    category: str
    expected_state: str
    expected_doc_ids: list[str]  # e.g., ["CPS 234"] — loose, doc-level
    predicted_state: str
    cited_doc_ids: list[str]
    retrieved_doc_ids: list[str]  # ordered top-K
    confidence_value: float
    unfaithful_count: int
    has_faithfulness_report: bool  # False if check was skipped or state was refused
    llm_called: bool
    elapsed_ms: int
    error: str | None = None  # set if the pipeline raised; metrics ignore these rows
    # v0.4: LLM judge verdict (when --with-judge is passed to the runner)
    judge_label: str | None = None  # "YES" | "PARTIAL" | "NO" | "UNKNOWN" | None (not judged)
    judge_reasoning: str = ""
    judge_name: str = ""  # which judge produced the verdict ("keyword" / "nli" / "llm")


# -------- State classification --------

def state_accuracy(results: list[QuestionResult]) -> float:
    """Fraction of questions where predicted state == expected state."""
    valid = [r for r in results if r.error is None]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r.predicted_state == r.expected_state) / len(valid)


def state_confusion_matrix(results: list[QuestionResult]) -> dict[str, dict[str, int]]:
    """3x3 matrix expected[row] -> predicted[col] counts."""
    cm: dict[str, dict[str, int]] = {e: {p: 0 for p in STATES} for e in STATES}
    for r in results:
        if r.error is not None:
            continue
        if r.expected_state in cm and r.predicted_state in cm[r.expected_state]:
            cm[r.expected_state][r.predicted_state] += 1
    return cm


def refusal_precision_recall(results: list[QuestionResult]) -> dict[str, float]:
    """P/R/F1 treating 'refused' as the positive class."""
    valid = [r for r in results if r.error is None]
    tp = sum(1 for r in valid if r.predicted_state == "refused" and r.expected_state == "refused")
    fp = sum(1 for r in valid if r.predicted_state == "refused" and r.expected_state != "refused")
    fn = sum(1 for r in valid if r.predicted_state != "refused" and r.expected_state == "refused")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# -------- Retrieval quality (doc-level, complements per-chunk metrics above) --------

def _doc_match(expected: str, retrieved: list[str]) -> bool:
    """Loose, case-insensitive substring match — benchmark uses short doc ids
    ('CPS 234'), parsed chunks carry full titles ('CPS 234 Information Security')."""
    needle = expected.lower()
    return any(needle in r.lower() for r in retrieved)


def retrieval_recall_at_k(results: list[QuestionResult], k: int) -> float:
    """For answerable questions, fraction where any expected doc appears in top-K retrieved.

    Doc-level. For chunk-level Hit@K / Recall@K / MRR see Section 1.
    """
    answerable = [
        r for r in results
        if r.error is None and r.category in ANSWERABLE_CATEGORIES and r.expected_doc_ids
    ]
    if not answerable:
        return 0.0
    hits = sum(
        1 for r in answerable
        if any(_doc_match(d, r.retrieved_doc_ids[:k]) for d in r.expected_doc_ids)
    )
    return hits / len(answerable)


def retrieval_recall_curve(
    results: list[QuestionResult], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> dict[int, float]:
    return {k: retrieval_recall_at_k(results, k) for k in ks}


# -------- Citation quality --------

def citation_correctness(results: list[QuestionResult]) -> float:
    """For confident answers, fraction where at least one cited doc matches an expected doc."""
    confident = [
        r for r in results
        if r.error is None and r.predicted_state == "confident" and r.expected_doc_ids
    ]
    if not confident:
        return 0.0
    hits = sum(
        1 for r in confident
        if any(_doc_match(d, r.cited_doc_ids) for d in r.expected_doc_ids)
    )
    return hits / len(confident)


# -------- Faithfulness --------

def faithfulness_rate(results: list[QuestionResult]) -> float:
    """Fraction of answered questions (with faithfulness report) that had zero unfaithful claims."""
    answered = [
        r for r in results
        if r.error is None and r.has_faithfulness_report
    ]
    if not answered:
        return 0.0
    return sum(1 for r in answered if r.unfaithful_count == 0) / len(answered)


# -------- Latency --------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    # Nearest-rank percentile; fine for small N where exact interpolation is overkill.
    idx = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[idx]


def latency_stats(results: list[QuestionResult]) -> dict[str, float]:
    valid_ms = [r.elapsed_ms for r in results if r.error is None]
    if not valid_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "llm_call_rate": 0.0}
    llm_calls = sum(1 for r in results if r.error is None and r.llm_called)
    return {
        "mean_ms": mean(valid_ms),
        "p50_ms": _percentile([float(v) for v in valid_ms], 50),
        "p95_ms": _percentile([float(v) for v in valid_ms], 95),
        "llm_call_rate": llm_calls / len(valid_ms),
    }


# -------- Judge-based answer correctness (v0.4) --------

def answer_correctness_rate(
    results: list[QuestionResult], partial_credit: float = 0.5
) -> dict[str, float]:
    """Aggregate the LLM judge verdicts into a single correctness score.

    Only counts answered (non-refused) questions where a judge actually ran.
    YES = 1.0, PARTIAL = partial_credit (default 0.5), NO = 0, UNKNOWN = excluded.

    Returns a dict with: 'rate' (the aggregate), counts of each label, and 'n_judged'.
    """
    judged = [
        r for r in results
        if r.error is None and r.judge_label is not None and r.judge_label != "UNKNOWN"
    ]
    counts: dict[str, int] = {"YES": 0, "PARTIAL": 0, "NO": 0, "UNKNOWN": 0}
    for r in results:
        if r.error is None and r.judge_label:
            counts[r.judge_label] = counts.get(r.judge_label, 0) + 1
    if not judged:
        return {"rate": 0.0, "n_judged": 0, **counts}
    score = sum(
        1.0 if r.judge_label == "YES" else partial_credit if r.judge_label == "PARTIAL" else 0.0
        for r in judged
    )
    return {"rate": score / len(judged), "n_judged": len(judged), **counts}


# -------- Aggregate --------

@dataclass
class Aggregate:
    """All headline metrics in one bag — what the runner writes to metrics.json."""

    n_total: int
    n_errors: int
    state_accuracy: float
    state_confusion_matrix: dict[str, dict[str, int]]
    refusal: dict[str, float]
    retrieval_recall: dict[int, float]
    citation_correctness: float
    faithfulness_rate: float
    latency: dict[str, float]
    per_category_accuracy: dict[str, float] = field(default_factory=dict)
    # v0.4: LLM judge aggregate (zeros when judge wasn't enabled)
    answer_correctness: dict[str, float] = field(default_factory=lambda: {"rate": 0.0, "n_judged": 0})


def per_category_accuracy(results: list[QuestionResult]) -> dict[str, float]:
    by_cat: dict[str, list[QuestionResult]] = {}
    for r in results:
        if r.error is None:
            by_cat.setdefault(r.category, []).append(r)
    return {cat: state_accuracy(rs) for cat, rs in sorted(by_cat.items())}


def aggregate(results: list[QuestionResult]) -> Aggregate:
    return Aggregate(
        n_total=len(results),
        n_errors=sum(1 for r in results if r.error is not None),
        state_accuracy=state_accuracy(results),
        state_confusion_matrix=state_confusion_matrix(results),
        refusal=refusal_precision_recall(results),
        retrieval_recall=retrieval_recall_curve(results),
        citation_correctness=citation_correctness(results),
        faithfulness_rate=faithfulness_rate(results),
        latency=latency_stats(results),
        per_category_accuracy=per_category_accuracy(results),
        answer_correctness=answer_correctness_rate(results),
    )


# =============================================================================
# Self-tests. Run: `python eval/metrics.py`
# =============================================================================


def _run_retrieval_self_tests() -> None:
    # A toy retrieval result for the CPS 234 §35 question.
    retrieved = [
        ("CPS 230 Operational Risk Management", "Incident management", "26"),  # rank 0 — wrong doc
        ("CPS 234 Information Security", "Notification of incidents", "35"),    # rank 1 — exact
        ("CPS 234 Information Security", "Notification of incidents", "36"),    # rank 2 — same section, diff para
        ("Privacy Act", "APP 11", "11.1"),                                       # rank 3
    ]
    expected_cps234 = [
        ExpectedCitation(doc="CPS 234", section="Notification of incidents", paragraph="35")
    ]

    # ---- Loose matching ----
    assert chunk_matches_expected(
        "CPS 234 Information Security", "Notification of incidents", "35",
        expected_cps234[0],
    ), "loose: exact-ish match should succeed"

    assert chunk_matches_expected(
        "CPS 234 Information Security", "WRONG SECTION", "35",
        expected_cps234[0],
    ), "loose: section is ignored when not strict"

    assert not chunk_matches_expected(
        "CPS 230 Operational Risk Management", "anything", "35",
        expected_cps234[0],
    ), "loose: wrong doc should fail"

    assert first_hit_rank(retrieved, expected_cps234) == 1, "first hit at rank 1"
    assert hit_at_k(retrieved, expected_cps234, k=2) is True
    assert hit_at_k(retrieved, expected_cps234, k=1) is False, "top-1 is wrong doc"

    assert recall_at_k(retrieved, expected_cps234, k=10) == 1.0
    assert recall_at_k(retrieved, expected_cps234, k=1) == 0.0

    # Two expected → recall is fraction
    expected_two = [
        ExpectedCitation(doc="CPS 234", paragraph="35"),
        ExpectedCitation(doc="Privacy Act", paragraph="11.1"),
    ]
    assert recall_at_k(retrieved, expected_two, k=4) == 1.0, "both found in top-4"
    assert recall_at_k(retrieved, expected_two, k=2) == 0.5, "only CPS 234 in top-2"

    # ---- MRR (per-question) ----
    assert abs(reciprocal_rank(retrieved, expected_cps234) - 0.5) < 1e-9, "rank-1 → RR=0.5"

    no_match = [ExpectedCitation(doc="Banking Act", paragraph="999")]
    assert reciprocal_rank(retrieved, no_match) == 0.0

    # ---- Strict matching ----
    assert chunk_matches_expected(
        "CPS 234 Information Security",
        "Notification of incidents",
        "35",
        ExpectedCitation(
            doc="CPS 234 Information Security",
            section="Notification of incidents",
            paragraph="35",
        ),
        strict=True,
    ), "strict: exact equality"

    assert not chunk_matches_expected(
        "CPS 234 Information Security",
        "Notification of incidents",
        "35",
        ExpectedCitation(doc="CPS 234", section="Notification of incidents", paragraph="35"),
        strict=True,
    ), "strict: 'CPS 234' != 'CPS 234 Information Security'"

    # ---- Edge cases ----
    assert recall_at_k([], expected_cps234, k=10) == 0.0
    assert recall_at_k(retrieved, [], k=10) == 0.0
    assert first_hit_rank([], expected_cps234) is None

    print("OK — retrieval-metrics self-tests passed.")


def _run_pipeline_outcome_self_tests() -> None:
    fake = [
        QuestionResult(
            qid="q1", category="answerable_easy", expected_state="confident",
            expected_doc_ids=["CPS 234"], predicted_state="confident",
            cited_doc_ids=["CPS 234"], retrieved_doc_ids=["CPS 234", "Privacy Act"],
            confidence_value=0.85, unfaithful_count=0, has_faithfulness_report=True,
            llm_called=True, elapsed_ms=1200,
        ),
        QuestionResult(
            qid="q2", category="unanswerable_out_of_scope", expected_state="refused",
            expected_doc_ids=[], predicted_state="refused",
            cited_doc_ids=[], retrieved_doc_ids=["CPS 234"],
            confidence_value=0.20, unfaithful_count=0, has_faithfulness_report=False,
            llm_called=False, elapsed_ms=200,
        ),
        QuestionResult(
            qid="q3", category="answerable_easy", expected_state="confident",
            expected_doc_ids=["Privacy Act"], predicted_state="hedged",
            cited_doc_ids=["Privacy Act"], retrieved_doc_ids=["Privacy Act"],
            confidence_value=0.55, unfaithful_count=1, has_faithfulness_report=True,
            llm_called=True, elapsed_ms=1500,
        ),
    ]
    agg = aggregate(fake)

    assert abs(agg.state_accuracy - 2 / 3) < 1e-9, f"state_accuracy={agg.state_accuracy}"
    assert agg.refusal["precision"] == 1.0
    assert agg.refusal["recall"] == 1.0
    assert agg.retrieval_recall[5] == 1.0
    assert agg.citation_correctness == 1.0, "q1 (confident) cited matching doc"
    assert agg.faithfulness_rate == 0.5, "1 of 2 answered passed faithfulness"
    assert agg.latency["p50_ms"] == 1200.0
    assert agg.per_category_accuracy == {"answerable_easy": 0.5, "unanswerable_out_of_scope": 1.0}

    print("OK — pipeline-outcome self-tests passed.")


if __name__ == "__main__":
    _run_retrieval_self_tests()
    _run_pipeline_outcome_self_tests()
