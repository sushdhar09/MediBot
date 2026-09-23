"""User-facing refusals.

Deliberately generic and identical across categories: telling a user *which*
rule they tripped turns the guardrail into an oracle they can probe. The only
varying part is the reference id, which lets support find the log line.
"""
from __future__ import annotations

INPUT_REFUSAL = (
    "I can't help with that request. I'm MediBot, the MediAssist staff assistant - "
    "I answer questions from the hospital documents and operational records you are "
    "authorised to access. Please rephrase your question in those terms.\n\n"
    "_Reference: {reference}_"
)

OUTPUT_REFUSAL = (
    "I'm not able to share an answer to that question. The draft response did not pass "
    "MediBot's safety checks, so it has been withheld. Please rephrase your question, or "
    "contact the IT service desk if you believe this is a mistake.\n\n"
    "_Reference: {reference}_"
)

REDACTION_NOTICE = (
    "\n\n_Some personal identifiers in this answer were masked automatically._"
)


def input_refusal(reference: str) -> str:
    return INPUT_REFUSAL.format(reference=reference)


def output_refusal(reference: str) -> str:
    return OUTPUT_REFUSAL.format(reference=reference)
