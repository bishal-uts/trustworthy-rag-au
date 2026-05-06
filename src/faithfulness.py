"""Post-generation faithfulness check.

For each generated claim (sentence in the answer), pick the most likely
cited chunk and run NLI: does the chunk entail the claim? If any claim
fails to be entailed above the configured floor, the caller can choose
to drop the answer to `hedged` state (default).

v0.1 design choices:
 - Claim decomposition is regex-based sentence splitting. Plan v1 §3.3
   leaves room for LLM-based claim decomposition later.
 - "Cited chunk" matching is loose: search for the citation's doc+section
   in the retrieved chunks. If no specific citation, use the top retrieved
   chunk. We're optimising for "did NLI catch a hallucination", not
   "did the model cite the exact paragraph it should have."
"""

from __future__ import annotations

import re

from src.config import settings
from src.models.manager import manager
from src.schemas import (
    Chunk,
    Citation,
    FaithfulnessClaim,
    FaithfulnessReport,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_into_claims(answer: str) -> list[str]:
    """Split an answer into atomic claims (sentences). Drops empty fragments."""
    if not answer or not answer.strip():
        return []
    parts = SENTENCE_SPLIT_RE.split(answer.strip())
    return [p.strip() for p in parts if p.strip()]


def _match_chunk_for_citation(
    citation: Citation | None, retrieved: list[Chunk]
) -> Chunk | None:
    """Find the retrieved chunk that best matches the citation's doc/section."""
    if not retrieved:
        return None
    if citation is None:
        return retrieved[0]
    doc_l = citation.doc.lower().strip()
    sec_l = citation.section.lower().strip()
    para_l = citation.paragraph.lower().strip()

    # Prefer exact (doc, section, paragraph) match
    for ch in retrieved:
        if (
            ch.doc.lower() == doc_l
            and ch.section.lower() == sec_l
            and ch.paragraph.lower() == para_l
        ):
            return ch
    # Then (doc, section) match
    for ch in retrieved:
        if ch.doc.lower() == doc_l and ch.section.lower() == sec_l:
            return ch
    # Then doc-only match
    for ch in retrieved:
        if doc_l and doc_l in ch.doc.lower():
            return ch
    # Fallback: top retrieved chunk
    return retrieved[0]


def check(
    answer: str,
    citations: list[Citation],
    retrieved_chunks: list[Chunk],
    nli_id: str,
) -> FaithfulnessReport:
    """Run NLI per claim and return a faithfulness report."""
    claims_text = split_into_claims(answer)
    if not claims_text or not retrieved_chunks:
        return FaithfulnessReport(claims=[], unfaithful_count=0)

    nli = manager.get_nli(nli_id)
    floor = settings.faithfulness.per_claim_entail_floor

    # First pass: NLI against the cited chunk (LLM-chosen).
    pairs_cited: list[tuple[str, str]] = []
    cite_for_each: list[Citation | None] = []
    for i, claim in enumerate(claims_text):
        cite = citations[i] if i < len(citations) else (citations[0] if citations else None)
        chunk = _match_chunk_for_citation(cite, retrieved_chunks)
        premise = chunk.text if chunk else retrieved_chunks[0].text
        pairs_cited.append((premise, claim))
        cite_for_each.append(cite)

    cited_results = nli.entail_batch(pairs_cited)

    # Second pass: for any claim that failed the cited-chunk check, also
    # try the top-3 retrieved chunks. The LLM sometimes picks a slightly
    # wrong section name; the actual supporting text is usually in the
    # top retrieval set. Take the best entailment score across those.
    fallback_pool = retrieved_chunks[:3]
    final_scores: list[float] = []
    for claim, cited_res in zip(claims_text, cited_results):
        if cited_res.entail >= floor:
            final_scores.append(cited_res.entail)
            continue
        # Fallback: try top-3
        fallback_pairs = [(ch.text, claim) for ch in fallback_pool]
        fallback_results = nli.entail_batch(fallback_pairs)
        best = max([r.entail for r in fallback_results] + [cited_res.entail])
        final_scores.append(best)

    claim_records: list[FaithfulnessClaim] = []
    unfaithful = 0
    for claim_text, cite, score in zip(claims_text, cite_for_each, final_scores):
        is_entailed = score >= floor
        if not is_entailed:
            unfaithful += 1
        claim_records.append(
            FaithfulnessClaim(
                text=claim_text,
                cited=cite is not None,
                entailed=is_entailed,
                nli_score=score,
            )
        )

    return FaithfulnessReport(claims=claim_records, unfaithful_count=unfaithful)
