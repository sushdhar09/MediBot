"""MediBot guardrail layer.

Two independent gates around the RAG pipeline:

    question --> [ guard_input ] --> router / retrieval / LLM --> [ guard_output ] --> user

Both return a structured `GuardrailVerdict`. Both fail closed on a malformed
verdict. Neither ever tells the user why they were blocked - the reason is
logged with a short reference id, and the user gets a generic refusal carrying
that same id.
"""
from __future__ import annotations

import logging

from . import messages
from .input_guard import check as guard_input
from .output_guard import check as guard_output
from .schemas import GuardrailVerdict

log = logging.getLogger("medibot.guardrails")

__all__ = ["guard_input", "guard_output", "GuardrailVerdict", "messages", "record"]


def record(verdict: GuardrailVerdict, *, username: str, question: str) -> GuardrailVerdict:
    """Log a verdict internally. Blocks are ERROR, redactions WARNING."""
    if verdict.action == "block":
        log.error("%s user=%s question=%r", verdict.log_line(), username, question[:200])
    elif verdict.action == "redact":
        log.warning("%s user=%s", verdict.log_line(), username)
    else:
        log.info("%s user=%s", verdict.log_line(), username)
    return verdict
