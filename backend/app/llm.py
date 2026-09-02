"""Thin wrapper around the Groq cloud inference API."""
from __future__ import annotations

from functools import lru_cache

from groq import Groq

from . import config


class LLMNotConfigured(RuntimeError):
    pass


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
) -> str:
    response = get_client().chat.completions.create(
        model=model or config.GROQ_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()
