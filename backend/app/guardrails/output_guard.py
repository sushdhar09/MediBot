"""Output guardrail: runs after the pipeline answers, before the user sees it.

Checks, in order of cost:

  1. RBAC cross-check on the citations - a source outside the caller's
     collections is a hard block, no judge needed
  2. deterministic leak rules - verbatim system prompt, internal config keys
  3. PII - high-severity identifiers block, contact/patient identifiers are
     masked in place so a useful answer survives
  4. escalation - restricted-topic wording, numbers absent from the retrieved
     context, missing citations or external-knowledge phrasing send the answer
     to the OpenEvals groundedness / PII judges (and Bedrock, if configured)

Same fail-closed rule as the input layer: a malformed verdict blocks.
"""
from __future__ import annotations

import logging

from .. import config, rbac
from . import bedrock, judge, patterns
from .schemas import GuardrailVerdict, JudgeUnavailable, MalformedVerdict

log = logging.getLogger("medibot.guardrails")

# A grounded answer cannot leak a restricted document, because the context it
# is grounded in was already filtered by RBAC inside Qdrant. So one groundedness
# verdict covers both fabrication and leakage.
_ESCALATION_MIN_TOPIC_HITS = 2


def _verdict(
    action: str,
    category: str,
    reason: str,
    checker: str,
    sanitized: str | None = None,
    degraded: bool = False,
) -> GuardrailVerdict:
    return GuardrailVerdict(
        action=action,
        stage="output",
        category=category,
        reason=reason,
        checker=checker,
        degraded=degraded,
        sanitized_output=sanitized,
    )


def _source_violations(sources: list[dict], role: str) -> list[str]:
    """Citations pointing at material this role may not read."""
    allowed = set(rbac.collections_for_role(role))
    violations = []
    for source in sources:
        collection = source.get("collection", "unknown")
        if collection == "database":
            if not rbac.can_use_sql_rag(role):
                violations.append("database (SQL RAG not permitted for this role)")
        elif collection not in allowed:
            violations.append(collection)
    return violations


def check(
    *,
    question: str,
    answer: str,
    context: str,
    role: str,
    sources: list[dict] | None = None,
    is_refusal: bool = False,
) -> GuardrailVerdict:
    """Decide whether `answer` may be shown, masked or withheld."""
    if not config.GUARDRAILS_ENABLED:
        return _verdict("allow", "disabled", "guardrails disabled by configuration", "none")

    sources = sources or []

    violations = _source_violations(sources, role)
    if violations:
        return _verdict(
            "block", "rbac_violation",
            f"answer cites material outside role={role}: {', '.join(violations)}",
            "deterministic",
        )

    leaks = patterns.scan(answer, patterns.SYSTEM_LEAK_RULES)
    blocking = [s for s in leaks if s.severity == "block"]
    blocking += [s for s in patterns.scan(answer, patterns.PII_BLOCK_RULES) if s.severity == "block"]
    if patterns.card_numbers(answer):
        blocking.append(patterns.Signal("pii_leak", "block", "payment card number (Luhn valid)"))
    if blocking:
        return _verdict(
            "block", blocking[0].category,
            "; ".join(f"{s.category}: {s.detail}" for s in blocking),
            "deterministic",
        )

    safe_answer, redactions = patterns.redact_pii(answer)

    # --- what, if anything, deserves a judge call ---------------------------
    doubts: list[str] = [f"{s.category}: {s.detail}" for s in leaks if s.severity == "suspect"]
    doubts += [
        f"{s.category}: {s.detail}"
        for s in patterns.scan(safe_answer, patterns.UNGROUNDED_HINT_RULES)
    ]

    if not is_refusal:
        restricted = patterns.restricted_topic_hits(safe_answer, role)
        if sum(restricted.values()) >= _ESCALATION_MIN_TOPIC_HITS:
            doubts.append(f"restricted-topic wording for role={role}: {restricted}")
        unsupported = patterns.unsupported_numbers(safe_answer, context)
        if unsupported:
            doubts.append(f"numbers absent from context: {unsupported[:5]}")
        if sources and context and not patterns.has_citations(safe_answer):
            doubts.append("answer cites no passage despite retrieved context")

    pii_doubt = bool(redactions)

    if not doubts and not pii_doubt:
        return _finish(redactions, safe_answer, "no deterministic signal", "deterministic")

    checks: list[tuple[str, str, dict]] = []
    if bedrock.is_enabled():
        checks.append(("bedrock", "bedrock_guardrail", {"text": safe_answer, "source": "OUTPUT"}))
    if doubts and context:
        from openevals.prompts import RAG_GROUNDEDNESS_PROMPT

        checks.append(("openevals", "groundedness", {
            "prompt": RAG_GROUNDEDNESS_PROMPT,
            "params": {"outputs": safe_answer, "context": context},
            # groundedness scores TRUE when the answer *is* grounded
            "block_when": False,
        }))
    if pii_doubt:
        from openevals.prompts import PII_LEAKAGE_PROMPT

        checks.append(("openevals", "pii_leakage", {
            "prompt": PII_LEAKAGE_PROMPT,
            "params": {"inputs": question, "outputs": safe_answer},
            "block_when": True,
        }))

    reachable = False
    for kind, key, spec in checks:
        try:
            if kind == "bedrock":
                outcome, block_when = bedrock.apply(spec["text"], source=spec["source"]), True
            else:
                outcome = judge.run(spec["prompt"], key, **spec["params"])
                block_when = spec["block_when"]
        except MalformedVerdict as exc:
            return _verdict("block", "guardrail_error", str(exc), f"{kind}:{key}")
        except JudgeUnavailable as exc:
            log.error("output guardrail degraded, %s unavailable: %s", key, exc)
            continue
        reachable = True
        if outcome.flagged is block_when:
            category = {"groundedness": "fabrication", "bedrock_guardrail": "unsafe_output"}.get(key, key)
            return _verdict("block", category, outcome.reasoning or key, f"{kind}:{key}")

    if not checks or reachable:
        return _finish(redactions, safe_answer, f"judged clean despite {doubts or 'pii redaction'}", "openevals")
    if not config.GUARDRAIL_FALLBACK_DETERMINISTIC:
        return _verdict("block", "guardrail_error", "no guardrail judge reachable", "fail_closed")
    log.error("output guardrail fell back to deterministic verdict (allow), doubts: %s", doubts)
    return _finish(
        redactions, safe_answer,
        f"judge unreachable, deterministic allow ({doubts})", "deterministic", degraded=True,
    )


def _finish(
    redactions: list, safe_answer: str, reason: str, checker: str, degraded: bool = False
) -> GuardrailVerdict:
    if not redactions:
        return _verdict("allow", "none", reason, checker, degraded=degraded)
    return _verdict(
        "redact", "pii_masked",
        f"masked {', '.join(s.detail for s in redactions)}; {reason}",
        checker,
        sanitized=safe_answer,
        degraded=degraded,
    )
