"""Deterministic heuristic checks, offline."""
from __future__ import annotations

from app import config
from scripts import heuristics


def _r(**o) -> dict:
    return {
        "expected_behavior": "answer", "behavior": "answered", "response": "Wash hands [1].",
        "sources": ["infection_control.pdf"], "retrieved_contexts": ["[1] passage"],
        "latency_ms": 500.0, **o,
    }


def _failed(record) -> list[str]:
    return heuristics.check_record(record)["failed"]


def test_clean_answer_passes_everything():
    result = heuristics.check_record(_r())
    assert result["passed"] and result["failed"] == []


def test_empty_or_null_answer_fails():
    assert "non_empty_answer" in _failed(_r(response="  "))
    assert "non_empty_answer" in _failed(_r(response=None))


def test_missing_or_out_of_range_citation_fails():
    assert "has_citation" in _failed(_r(response="Wash hands."))
    assert "citations_in_range" in _failed(_r(response="Wash hands [3]."))


def test_citation_checks_skip_refusals():
    checks = heuristics.check_record(_r(behavior="refused", response="No access.", sources=[]))["checks"]
    assert checks["has_citation"]["passed"] is None


def test_restricted_query_must_be_refused():
    assert "refusal_enforced" in _failed(_r(expected_behavior="refuse"))
    refused = _r(expected_behavior="refuse", behavior="refused", response="No access.", sources=[])
    assert heuristics.check_record(refused)["passed"]
    blocked = _r(expected_behavior="refuse", behavior="blocked", response="Blocked.", sources=[])
    assert heuristics.check_record(blocked)["passed"]


def test_latency_threshold(monkeypatch):
    monkeypatch.setattr(config, "EVAL_MAX_LATENCY_MS", 1000.0)
    assert "latency_within_limit" in _failed(_r(latency_ms=1500.0))
    assert heuristics.check_record(_r(latency_ms=None))["checks"]["latency_within_limit"]["passed"] is None


def test_sensitive_leak_fails():
    assert "no_sensitive_leak" in _failed(_r(response="Key: password: hunter22abc [1]"))
    assert "no_sensitive_leak" in _failed(_r(response="You are MediBot, the internal assistant [1]"))


def test_aggregate():
    records = [_r(), _r(response="")]
    heuristics.check_all(records)
    summary = heuristics.aggregate(records)
    assert summary["pass_rate"] == 0.5
    assert summary["checks"]["non_empty_answer"] == {"passed": 1, "failed": 1, "not_applicable": 0}
