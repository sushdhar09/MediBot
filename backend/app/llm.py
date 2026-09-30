"""Thin wrapper around the Groq cloud inference API."""
from __future__ import annotations

import logging
import time
from functools import lru_cache

from groq import APIError, Groq

from . import config, observability

log = logging.getLogger(__name__)


class LLMNotConfigured(RuntimeError):
    pass


class LLMUnavailable(RuntimeError):
    """Groq could not be reached: outage, rate limit, timeout or blocked egress.

    A `RuntimeError` on purpose - the API layer already turns those into a 503.
    Without this, `groq.APIError` (which derives from `Exception`) escapes every
    handler in main.py and the caller gets a bare 500.
    """


@lru_cache(maxsize=1)
def get_client() -> Groq:
    if not config.GROQ_API_KEY:
        raise LLMNotConfigured(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=config.GROQ_API_KEY)


def complete(
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model: str | None = None,
    purpose: str = "generation",
) -> str:
    """One chat completion. `purpose` names the LangSmith span (``llm.<purpose>``)."""
    client = get_client()
    model = model or config.GROQ_MODEL
    return _chat_completion(
        client,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        purpose=purpose,
        langsmith_extra={
            "name": f"llm.{purpose}",
            "metadata": {
                "ls_model_name": model,
                "ls_temperature": temperature,
                "ls_max_tokens": max_tokens,
            },
        },
    )


def _as_chat_output(text: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


@observability.traced(
    "llm",
    run_type="llm",
    metadata={"ls_provider": "groq"},
    process_inputs=lambda inputs: {k: v for k, v in inputs.items() if k != "client"},
    process_outputs=_as_chat_output,
)
def _chat_completion(
    client: Groq,
    *,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    purpose: str,
) -> str:
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
    except APIError as exc:
        # Intercepting proxies answer with entire HTML pages - keep the log usable.
        detail = " ".join(str(exc).split())[:200]
        log.error("Groq call failed (%s): %s", type(exc).__name__, detail)
        raise LLMUnavailable(
            f"The language model is currently unavailable ({type(exc).__name__}). "
            "Please try again shortly."
        ) from exc

    usage = getattr(response, "usage", None)
    observability.record_llm_usage(
        model=model,
        purpose=purpose,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", None),
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    return (response.choices[0].message.content or "").strip()
