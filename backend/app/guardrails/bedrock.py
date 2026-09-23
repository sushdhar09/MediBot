"""Optional AWS Bedrock Guardrails layer (ApplyGuardrail API).

Dormant unless MEDIBOT_BEDROCK_GUARDRAIL_ID is set, so the service runs with no
AWS dependency at all by default. When configured it runs *in addition to* the
deterministic checks and the OpenEvals judge, on both the input and the output
path, and its verdict is authoritative: if Bedrock intervenes, we block.

Failure policy matches `judge.py`: an unparseable response fails closed, an
unreachable service falls back to the deterministic verdict.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from .. import config
from .judge import JudgeOutcome
from .schemas import JudgeUnavailable, MalformedVerdict

log = logging.getLogger(__name__)

Source = Literal["INPUT", "OUTPUT"]


def is_enabled() -> bool:
    return bool(config.BEDROCK_GUARDRAIL_ID)


@lru_cache(maxsize=1)
def _client() -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise JudgeUnavailable(f"boto3 is not installed: {exc}") from exc
    return boto3.client("bedrock-runtime", region_name=config.BEDROCK_REGION)


def apply(text: str, *, source: Source) -> JudgeOutcome:
    """Run Bedrock Guardrails over one piece of text.

    `flagged=True` means the guardrail intervened and the text must not pass.
    """
    if not is_enabled():
        raise JudgeUnavailable("Bedrock guardrail is not configured")

    try:
        response = _client().apply_guardrail(
            guardrailIdentifier=config.BEDROCK_GUARDRAIL_ID,
            guardrailVersion=config.BEDROCK_GUARDRAIL_VERSION,
            source=source,
            content=[{"text": {"text": text}}],
        )
    except Exception as exc:
        detail = " ".join(str(exc).split())[:300]
        raise JudgeUnavailable(f"Bedrock ApplyGuardrail call failed: {detail}") from exc

    action = response.get("action") if isinstance(response, dict) else None
    if action not in {"GUARDRAIL_INTERVENED", "NONE"}:
        raise MalformedVerdict(f"Bedrock returned an unrecognised action: {action!r}")

    return JudgeOutcome(
        key="bedrock_guardrail",
        flagged=action == "GUARDRAIL_INTERVENED",
        reasoning=_summarise(response.get("assessments") or []),
    )


def _summarise(assessments: list[dict]) -> str:
    """Flatten Bedrock's assessment blocks into one loggable line."""
    parts: list[str] = []
    for assessment in assessments:
        for policy, payload in assessment.items():
            for key in ("filters", "customWords", "managedWordLists", "piiEntities",
                        "regexes", "topics", "filters"):
                for item in (payload or {}).get(key, []) or []:
                    name = item.get("type") or item.get("name") or item.get("match")
                    if name:
                        parts.append(f"{policy}:{name}")
    return ", ".join(dict.fromkeys(parts))[:400] or "no assessment detail"
