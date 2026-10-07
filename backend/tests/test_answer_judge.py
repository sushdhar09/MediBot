"""The rubric-based LLM judge, offline, with a fake completion function."""
from __future__ import annotations

import json

import pytest

from scripts import answer_judge


def _record(**overrides) -> dict:
    return {
        "id": "t", "role": "nurse", "expected_behavior": "answer", "behavior": "answered",
        "question": "When is hand hygiene performed?",
        "reference": "Before patient contact.",
        "response": "Before patient contact [1].",
        "retrieved_contexts": ["[1] infection_control.pdf\nPerform hand hygiene before patient contact."],
        **overrides,
    }


def _reply(accuracy=5, completeness=5, refusal=5, citation=5) -> str:
    def crit(score):
        return {"score": score, "justification": "because"}
    return json.dumps({
        "accuracy": crit(accuracy), "completeness": crit(completeness),
        "refusal_behavior": crit(refusal), "citation_correctness": crit(citation),
        "justification": "Overall fine.",
    })


def test_good_answer_passes_with_justifications():
    result = answer_judge.judge_answer(_record(), complete=lambda s, u: _reply())
    assert result["passed"] and result["overall_score"] == 5
    assert result["justification"] and result["justifications"]["accuracy"] == "because"
    assert result["error"] is None


def test_pass_is_computed_from_scores_not_taken_from_the_model():
    low_floor = answer_judge.judge_answer(_record(), complete=lambda s, u: _reply(accuracy=2))
    assert not low_floor["passed"]  # one criterion under the floor fails despite a high mean
    low_mean = answer_judge.judge_answer(_record(), complete=lambda s, u: _reply(3, 3, 3, 3))
    assert not low_mean["passed"]


def test_refusal_ignores_inapplicable_criteria():
    reply = json.loads(_reply())
    for c in ("accuracy", "completeness", "citation_correctness"):
        reply[c]["score"] = None
    result = answer_judge.judge_answer(_record(behavior="refused"), complete=lambda s, u: json.dumps(reply))
    assert result["passed"] and result["overall_score"] == 5


@pytest.mark.parametrize("bad", [
    "not json", "{}", json.dumps({"accuracy": 5}),
    _reply(accuracy=9), _reply(refusal=None),
])
def test_malformed_verdict_fails_closed(bad):
    calls = []
    result = answer_judge.judge_answer(_record(), complete=lambda s, u: calls.append(u) or bad)
    assert not result["passed"] and "fail closed" in result["error"]
    assert result["overall_score"] is None
    assert len(calls) == 2  # one repair attempt


def test_repair_attempt_can_recover():
    replies = iter(["oops", _reply()])
    result = answer_judge.judge_answer(_record(), complete=lambda s, u: next(replies))
    assert result["passed"]


def test_fenced_json_is_accepted():
    result = answer_judge.judge_answer(_record(), complete=lambda s, u: f"```json\n{_reply()}\n```")
    assert result["passed"]


def test_transport_error_is_a_failing_result_not_an_exception():
    def boom(s, u):
        raise ConnectionError("firewall")
    result = answer_judge.judge_answer(_record(), complete=boom)
    assert not result["passed"] and "judge call failed" in result["error"]


def test_system_error_is_not_judged():
    called = []
    result = answer_judge.judge_answer(_record(error="503"), complete=lambda s, u: called.append(1))
    assert not result["passed"] and not called


def test_results_are_cached(tmp_path):
    calls = []
    complete = lambda s, u: calls.append(1) or _reply()  # noqa: E731
    first = answer_judge.judge_answer(_record(), complete=complete, cache_dir=tmp_path)
    second = answer_judge.judge_answer(_record(), complete=complete, cache_dir=tmp_path)
    assert first == second and len(calls) == 1


def test_aggregate():
    records = [_record(), _record()]
    records[0]["judge"] = answer_judge.judge_answer(records[0], complete=lambda s, u: _reply())
    records[1]["judge"] = answer_judge.judge_answer(records[1], complete=lambda s, u: "bad")
    summary = answer_judge.aggregate(records)
    assert summary["pass_rate"] == 0.5 and summary["errors"] == 1
    assert summary["criteria"]["accuracy"] == {"mean": 5, "scored": 1}


# --- calibration: does the judge simply agree with confident answers? ------------------------

def _real_calibration_cases():
    return json.loads(answer_judge.CALIBRATION_PATH.read_text(encoding="utf-8"))["cases"]


def test_calibration_set_has_wrong_but_confident_and_correct_answers():
    cases = _real_calibration_cases()
    assert sum(c["label"] == "bad" for c in cases) >= 3 and sum(c["label"] == "good" for c in cases) >= 2
    assert {c["label"] for c in cases} == {"good", "bad"}
    assert all(c["flaw"] for c in cases)


def test_a_judge_that_agrees_with_everything_fails_calibration():
    result = answer_judge.calibrate(complete=lambda s, u: _reply())  # always 5/5
    assert result["wrong_caught"] == 0 and result["good_accepted"] == result["good_total"]


def test_a_judge_that_fails_everything_is_also_caught():
    result = answer_judge.calibrate(complete=lambda s, u: _reply(1, 1, 1, 1))
    assert result["wrong_caught"] == result["wrong_total"] and result["good_accepted"] == 0


def test_a_discriminating_judge_passes_calibration():
    def complete(system, user):
        wrong = any(c["response"] in user and c["label"] == "bad" for c in _real_calibration_cases())
        return _reply(1, 2, 1, 1) if wrong else _reply()

    result = answer_judge.calibrate(complete=complete)
    assert result["wrong_caught"] == result["wrong_total"] and result["good_accepted"] == result["good_total"]


def test_an_unusable_verdict_counts_against_the_judge():
    result = answer_judge.calibrate(complete=lambda s, u: "garbage")
    assert result["wrong_caught"] == 0 and result["good_accepted"] == 0
