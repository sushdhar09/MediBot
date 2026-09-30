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

from .. import observability
from . import messages
from .input_guard import check as guard_input
from .output_guard import check as guard_output
from .schemas import GuardrailVerdict

log = logging.getLogger("medibot.guardrails")

__all__ = ["guard_input", "guard_output", "GuardrailVerdict", "messages", "record"]

_LEVELS = {"block": logging.ERROR, "redact": logging.WARNING}


def record(verdict: GuardrailVerdict, *, username: str, question: str) -> GuardrailVerdict:
    """Log a verdict as a structured `guardrail.decision` event.

    Blocks are ERROR, redactions WARNING. The question text is only logged for
    blocks (the full text is always in the trace); `reference` is the id the
    user was shown, so a user report maps straight to this event.
    """
    decision = {
        "reference": verdict.reference,
        "stage": verdict.stage,
        "action": verdict.action,
        "allowed": verdict.allowed,
        "category": verdict.category,
        "checker": verdict.checker,
        "degraded": verdict.degraded,
    }
    blocked = verdict.action == "block"
    message = f"{verdict.log_line()} user={username}"
    if blocked:
        message += f" question={question[:200]!r}"
    observability.emit(
        "guardrail.decision",
        level=_LEVELS.get(verdict.action, logging.INFO),
        message=message,
        logger=log,
        **decision,
        reason=verdict.reason,
        username=username,
        question=question[:500] if blocked else None,
        question_chars=len(question),
    )

    ctx = observability.current()
    if ctx is not None:
        ctx.guardrails.append({**decision, "reason": verdict.reason})
    observability.trace_metadata(**{f"guardrail_{verdict.stage}_{k}": v for k, v in decision.items()})
    observability.tag(f"guardrail:{verdict.stage}:{verdict.action}")
    return verdict
