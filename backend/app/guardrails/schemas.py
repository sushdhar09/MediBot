"""The structured verdict every guardrail check must produce.

Nothing in this layer returns a free-text answer that the caller has to
prefix-match. A check either returns a `GuardrailVerdict` or raises, and the
caller turns a raised/absent verdict into a block (see `input_guard` and
`output_guard`).
"""
from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

Action = Literal["allow", "block", "redact"]
Stage = Literal["input", "output"]


class GuardrailVerdict(BaseModel):
    """A single, checkable decision about one request or one response.

    `reason` and `category` are diagnostics: they are logged, never returned to
    the end user. The user only ever sees `reference`, which ties their report
    back to a log line.
    """

    model_config = ConfigDict(frozen=True)

    action: Action
    stage: Stage
    category: str = "none"
    reason: str = ""
    checker: str = "deterministic"
    reference: str = Field(default_factory=lambda: uuid4().hex[:8])
    # True when a judge that should have run could not be reached, so this
    # verdict rests on the deterministic layer alone. Operational signal only -
    # never surfaced to the end user.
    degraded: bool = False
    # Populated only for action="redact": the answer with PII masked out.
    sanitized_output: str | None = None

    @property
    def allowed(self) -> bool:
        return self.action != "block"

    def log_line(self) -> str:
        return (
            f"guardrail ref={self.reference} stage={self.stage} action={self.action} "
            f"category={self.category} checker={self.checker} degraded={self.degraded} "
            f"reason={self.reason!r}"
        )


class JudgeError(RuntimeError):
    """Base class for judge failures. Subclasses decide the fallback policy."""


class JudgeUnavailable(JudgeError):
    """The judge could not be reached (no API key, network error, timeout).

    Policy: fall back to the deterministic verdict and log loudly. This is the
    only path on which a guardrail failure does not block, and it is
    configurable via MEDIBOT_GUARDRAIL_FALLBACK_DETERMINISTIC.
    """


class MalformedVerdict(JudgeError):
    """The judge answered, but not with a verdict we can act on.

    Policy: always fail closed. An unparseable verdict is a blocked request.
    """
