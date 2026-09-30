"""OpenEvals LLM-as-judge, wrapped so the caller can only get a typed outcome.

OpenEvals drives the judge through `with_structured_output`, so the verdict
arrives as a JSON object with a boolean `score` and a `reasoning` string - not
as free text we have to prefix-match. Two failure modes are distinguished, and
they have deliberately different policies:

  * the judge answered but the verdict is not a boolean -> `MalformedVerdict`,
    which the guards always turn into a block (fail closed)
  * the judge could not be reached at all -> `JudgeUnavailable`, which lets the
    deterministic verdict stand (see config.GUARDRAIL_FALLBACK_DETERMINISTIC)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import get_usage_metadata_callback

from .. import config, observability
from .schemas import JudgeUnavailable, MalformedVerdict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgeOutcome:
    """`flagged` means the judged criterion was met (e.g. injection detected)."""

    key: str
    flagged: bool
    reasoning: str


# --- Custom rubrics ---------------------------------------------------------
# OpenEvals ships prompts for injection, PII and groundedness; scope is
# specific to this deployment, so it gets a rubric in the same shape.

SCOPE_PROMPT = """You are a strict scope filter for MediBot, the internal assistant of the
MediAssist Health Network hospital group. It answers staff questions from internal hospital
documents (HR handbook and leave policy, clinical treatment protocols and drug formulary,
ICU nursing and infection-control procedures, insurance billing and claim guides, medical
equipment manuals) and from an operational database of billing claims and maintenance tickets.

<Rubric>
Score TRUE (out of scope / abusive) when the request:
- Has nothing to do with working at a hospital (general trivia, news, sport, celebrities,
  coding help, creative writing, personal advice unrelated to employment)
- Asks for medical advice on the user's own or a named person's health rather than for a
  documented hospital procedure or protocol
- Is abusive, harassing, or aimed at another member of staff
- Tries to use MediBot as a general-purpose chatbot or model playground

Score FALSE (in scope) when the request:
- Concerns hospital policy, clinical or nursing procedure, billing, equipment, HR, or
  operational figures - even if the asker turns out not to be authorised to see the answer
- Is a normal workplace or facility question (leave balance, shift rota, cafeteria, parking,
  IT service desk)
- Is a greeting, a clarification, or a follow-up about a previous hospital answer
</Rubric>

<Instructions>
Authorisation is NOT your concern - a nurse asking a billing question is in scope, just
unauthorised, and a different layer handles that. Judge only the subject matter.
Assign TRUE if the request is out of scope or abusive, FALSE otherwise.
</Instructions>

<example>
<input>
{inputs}
</input>
</example>
"""


@lru_cache(maxsize=1)
def _chat_model() -> Any:
    if not config.GROQ_API_KEY:
        raise JudgeUnavailable("GROQ_API_KEY is not set, the guardrail judge cannot run")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise JudgeUnavailable(f"langchain-groq is not installed: {exc}") from exc
    return ChatGroq(
        model=config.GUARDRAIL_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        timeout=config.GUARDRAIL_JUDGE_TIMEOUT,
        max_retries=1,
    )


@lru_cache(maxsize=8)
def _evaluator(prompt: str, feedback_key: str) -> Any:
    from openevals.llm import create_llm_as_judge

    return create_llm_as_judge(
        prompt=prompt,
        feedback_key=feedback_key,
        judge=_chat_model(),
        use_reasoning=True,
    )


def is_available() -> bool:
    return config.GUARDRAIL_JUDGE_ENABLED and bool(config.GROQ_API_KEY)


def run(prompt: str, feedback_key: str, **params: Any) -> JudgeOutcome:
    """Run one OpenEvals judge. Raises rather than returning an unusable verdict."""
    if not config.GUARDRAIL_JUDGE_ENABLED:
        raise JudgeUnavailable("guardrail judge disabled by configuration")
    return _run(prompt, feedback_key, langsmith_extra={"name": f"guardrail.judge.{feedback_key}"}, **params)


@observability.traced(
    "guardrail.judge",
    # the rubric is static and long; the ChatGroq child run shows it rendered
    process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "prompt"},
)
def _run(prompt: str, feedback_key: str, **params: Any) -> JudgeOutcome:
    evaluator = _evaluator(prompt, feedback_key)  # raises JudgeUnavailable without a key
    try:
        with get_usage_metadata_callback() as usage:
            result = evaluator(**params)
    except Exception as exc:  # transport, auth, rate limit, timeout
        # Proxies and captive portals answer with whole HTML pages - keep the log readable.
        detail = " ".join(str(exc).split())[:300]
        raise JudgeUnavailable(f"{feedback_key} judge call failed: {detail}") from exc

    for model, counts in usage.usage_metadata.items():
        observability.record_llm_usage(
            model=model,
            purpose=f"judge.{feedback_key}",
            input_tokens=counts.get("input_tokens", 0),
            output_tokens=counts.get("output_tokens", 0),
            total_tokens=counts.get("total_tokens"),
            on_span=False,  # LangSmith already has these on the ChatGroq run
        )

    if not isinstance(result, dict) or "score" not in result:
        raise MalformedVerdict(f"{feedback_key} judge returned no verdict: {result!r}")
    score = result["score"]
    if not isinstance(score, bool):
        raise MalformedVerdict(f"{feedback_key} judge returned a non-boolean score: {score!r}")
    return JudgeOutcome(
        key=feedback_key,
        flagged=score,
        reasoning=(result.get("comment") or "").strip()[:400],
    )
