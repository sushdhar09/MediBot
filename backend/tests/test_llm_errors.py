"""Provider failures must surface as a 503, never as an unhandled 500.

`groq.APIError` derives from `Exception`, not `RuntimeError`, so before
`LLMUnavailable` existed a Groq outage escaped every handler in main.py: the
caller got a bare "Internal Server Error" and the proxy's HTML error page was
dumped into the log.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIConnectionError, APIError

from app import auth, config, llm, main
from app.guardrails import input_guard, output_guard
from app.guardrails.schemas import JudgeUnavailable

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _quiet_guardrails(monkeypatch):
    monkeypatch.setattr(config, "BEDROCK_GUARDRAIL_ID", "")

    def _unavailable(*args, **kwargs):
        raise JudgeUnavailable("stubbed: no judge in tests")

    monkeypatch.setattr(input_guard.judge, "run", _unavailable)
    monkeypatch.setattr(output_guard.judge, "run", _unavailable)


def _raise_api_error(**kwargs):
    raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/v1/chat"))


def test_llm_unavailable_is_a_runtime_error():
    """main.py's handlers key off RuntimeError, so the hierarchy matters."""
    assert issubclass(llm.LLMUnavailable, RuntimeError)
    assert not issubclass(APIError, RuntimeError)


def test_complete_wraps_provider_errors(monkeypatch):
    class _FakeClient:
        class chat:  # noqa: N801 - mirrors the groq client shape
            class completions:
                create = staticmethod(_raise_api_error)

    monkeypatch.setattr(llm, "get_client", lambda: _FakeClient)
    with pytest.raises(llm.LLMUnavailable, match="currently unavailable"):
        llm.complete("system", "user")


def test_provider_outage_returns_503_not_500(monkeypatch):
    def _boom(question, role):
        raise llm.LLMUnavailable("The language model is currently unavailable (APIConnectionError).")

    monkeypatch.setattr(main.routing, "is_analytical_question", lambda q: False)
    monkeypatch.setattr(main, "answer_from_documents", _boom)

    token = auth.create_token(auth.DEMO_USERS["nurse.priya"])
    response = client.post(
        "/chat",
        json={"question": "What is the hand hygiene procedure?"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 503
    assert "currently unavailable" in response.json()["detail"]


def test_wrapped_error_message_stays_short(monkeypatch):
    """Intercepting proxies reply with whole HTML pages; they must not reach the client."""
    html = "<!DOCTYPE html>" + ("<div>blocked by firewall</div>" * 500)

    def _raise_html(**kwargs):
        raise APIError(html, request=httpx.Request("POST", "https://api.groq.com"), body=None)

    class _FakeClient:
        class chat:  # noqa: N801
            class completions:
                create = staticmethod(_raise_html)

    monkeypatch.setattr(llm, "get_client", lambda: _FakeClient)
    with pytest.raises(llm.LLMUnavailable) as excinfo:
        llm.complete("system", "user")
    assert len(str(excinfo.value)) < 200
    assert "DOCTYPE" not in str(excinfo.value)
