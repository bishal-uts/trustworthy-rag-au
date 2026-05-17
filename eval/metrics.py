"""Retrieval evaluation metrics — pure functions.

These functions don't depend on the pipeline. They take an ordered list of
retrieved (doc, section, paragraph) tuples plus a list of expected citations
and return a score. Unit-testable without any model loaded.

Definitions:

  Hit@k         binary  — at least one expected citation lands in top-k?
  Recall@k      float   — fraction of expected citations that land in top-k
  ReciprocalRank float  — 1 / (rank + 1) of the first relevant retrieved item;
                          0 if no expected citation is retrieved at all
  MRR (Mean RR)         — average of ReciprocalRank across queries

Both strict and loose matching are supported:

  loose (default): doc substring match + paragraph exact match if specified.
                   Mirrors the benchmark style — benchmark uses "CPS 234" but
                   the parsed chunks carry the full title "CPS 234 Information
                   Security", so substring match is the practical choice.
  strict:          full equality on (doc, section, paragraph) — useful for
                   diagnosing whether the retriever is finding the correct
                   paragraph, not just the correct document.

Run `python eval/metrics.py` to execute the self-tests below.
"""

from __future__ import annotations

from dataclasses import dataclass


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


# ---------------------------------------------------------------------------
# Self-tests. Run: `python eval/metrics.py`
# ---------------------------------------------------------------------------


def _run_self_tests() -> None:
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

    # ---- MRR ----
    # Rank-1 → RR = 1/2 = 0.5
    assert abs(reciprocal_rank(retrieved, expected_cps234) - 0.5) < 1e-9

    # No match → RR = 0
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

    print("OK — all self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
