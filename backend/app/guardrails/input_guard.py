"""Input guardrail: runs before the router, retrieval or any LLM call.

Order of play:

  1. structural sanity (empty, control characters, oversized, encoded payload)
  2. deterministic rules - an unambiguous hit blocks here, for free
  3. escalation - only if step 2 was inconclusive, the OpenEvals judge (and
     Bedrock, when configured) settles it
  4. fail closed - a malformed verdict blocks; an unreachable judge falls back
     to the deterministic verdict and logs at ERROR
"""
from __future__ import annotations

import logging

from .. import config
from . import bedrock, judge, patterns
from .schemas import GuardrailVerdict, JudgeUnavailable, MalformedVerdict

log = logging.getLogger("medibot.guardrails")

_ALLOW = "allow"
_BLOCK = "block"


def _verdict(
    action: str, category: str, reason: str, checker: str, degraded: bool = False
) -> GuardrailVerdict:
    return GuardrailVerdict(
        action=action, stage="input", category=category, reason=reason,
        checker=checker, degraded=degraded,
    )


def check(question: str, *, role: str) -> GuardrailVerdict:
    """Decide whether `question` may reach the retrieval pipeline."""
    if not config.GUARDRAILS_ENABLED:
        return _verdict(_ALLOW, "disabled", "guardrails disabled by configuration", "none")

    problem = patterns.structural_problem(question)
    if problem:
        return _verdict(_BLOCK, "malformed_input", problem, "deterministic")

    signals = patterns.scan(question, patterns.INPUT_RULES)
    blocking = [s for s in signals if s.severity == "block"]
    if blocking:
        return _verdict(
            _BLOCK,
            blocking[0].category,
            "; ".join(f"{s.category}: {s.detail}" for s in blocking),
            "deterministic",
        )

    suspects = [s for s in signals if s.severity == "suspect"]
    off_topic_doubt = not patterns.is_greeting(question) and not patterns.looks_in_scope(question)

    if not suspects and not off_topic_doubt:
        return _verdict(_ALLOW, "none", "no deterministic signal", "deterministic")

    clean = _verdict(_ALLOW, "none", "inconclusive signals cleared by judge", "openevals")
    fallback_reason = "; ".join(f"{s.category}: {s.detail}" for s in suspects) or "possibly off-topic"

    checks: list[tuple[str, str, dict]] = []
    if bedrock.is_enabled():
        checks.append(("bedrock", "bedrock_guardrail", {"text": question, "source": "INPUT"}))
    if suspects:
        from openevals.prompts import PROMPT_INJECTION_PROMPT

        checks.append(("openevals", "prompt_injection", {
            "prompt": PROMPT_INJECTION_PROMPT, "inputs": question,
        }))
    if off_topic_doubt:
        checks.append(("openevals", "out_of_scope", {
            "prompt": judge.SCOPE_PROMPT, "inputs": question,
        }))

    reachable = False
    for kind, key, params in checks:
        try:
            outcome = (
                bedrock.apply(params["text"], source=params["source"])
                if kind == "bedrock"
                else judge.run(params["prompt"], key, inputs=params["inputs"])
            )
        except MalformedVerdict as exc:
            # Fail closed: we asked, we got an answer we cannot act on.
            return _verdict(_BLOCK, "guardrail_error", str(exc), f"{kind}:{key}")
        except JudgeUnavailable as exc:
            log.error("input guardrail degraded, %s unavailable: %s", key, exc)
            continue
        reachable = True
        if outcome.flagged:
            category = "prompt_injection" if key == "bedrock_guardrail" else key
            return _verdict(_BLOCK, category, outcome.reasoning or key, f"{kind}:{key}")

    if not reachable and not config.GUARDRAIL_FALLBACK_DETERMINISTIC:
        return _verdict(_BLOCK, "guardrail_error", "no guardrail judge reachable", "fail_closed")
    if not reachable:
        log.error(
            "input guardrail fell back to deterministic verdict (allow), signals: %s",
            fallback_reason,
        )
        return _verdict(
            _ALLOW, "degraded",
            f"judge unreachable, deterministic allow ({fallback_reason})",
            "deterministic", degraded=True,
        )
    return clean
