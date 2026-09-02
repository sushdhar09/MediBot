"""Decide whether a question belongs to SQL RAG (database) or document RAG.

A cheap keyword pass handles the obvious cases; anything ambiguous is settled by
a one-token LLM classification, with the keyword verdict as the fallback if the
LLM call fails.
"""
from __future__ import annotations

import logging
import re

from .llm import complete

log = logging.getLogger(__name__)

# Signals that the answer lives in claims / maintenance_tickets, not in a PDF.
_DB_ENTITIES = (
    "claim", "claims", "ticket", "tickets", "insurer", "reimbursement",
    "maintenance ticket", "claimed amount", "approved amount", "patient id",
)
_AGGREGATION = (
    "how many", "how much", "count", "total", "sum", "average", "avg", "median",
    "most", "least", "highest", "lowest", "top ", "breakdown", "per department",
    "per month", "trend", "statistics", "number of", "percentage", "share of",
    "rank", "compare",
)
# Phrases that mean "explain the policy", i.e. always a document question.
_DOC_INTENT = (
    "how do i", "how to", "procedure", "protocol", "policy", "guideline",
    "what does the", "explain", "steps", "definition", "checklist", "manual",
)

CLASSIFIER_SYSTEM = """Classify a hospital staff question as either SQL or DOCS.

SQL  - the answer is a number, count, total, ranking or breakdown computed from
       operational records: the `claims` table (billing claims: department,
       insurer, amounts, status, submitted dates) or the `maintenance_tickets`
       table (equipment tickets: category, issue type, status, dates).
DOCS - the answer is written in a policy, protocol, formulary, billing guide or
       equipment manual.

Reply with exactly one word: SQL or DOCS."""


def _keyword_verdict(question: str) -> tuple[bool, bool]:
    """Return (looks_analytical, confident)."""
    text = re.sub(r"\s+", " ", question.lower())
    has_entity = any(word in text for word in _DB_ENTITIES)
    has_aggregation = any(word in text for word in _AGGREGATION)
    has_doc_intent = any(word in text for word in _DOC_INTENT)

    if has_entity and has_aggregation:
        return True, True
    if has_doc_intent and not has_aggregation:
        return False, True
    return has_entity and has_aggregation, False


def is_analytical_question(question: str) -> bool:
    verdict, confident = _keyword_verdict(question)
    if confident:
        return verdict
    try:
        answer = complete(CLASSIFIER_SYSTEM, question, temperature=0.0, max_tokens=5)
        return answer.strip().upper().startswith("SQL")
    except Exception:  # LLM unavailable - fall back to the keyword guess
        log.warning("Router LLM call failed, falling back to keywords", exc_info=True)
        return verdict
