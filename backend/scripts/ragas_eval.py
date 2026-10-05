"""RAGAS evaluation of MediBot against a labeled question set.

Every case in eval/eval_set.json goes through the real /chat pipeline (input
guardrail -> router -> hybrid retrieval or SQL RAG -> LLM -> output guardrail)
as the case's role. The passages that reached the LLM are recorded, and RAGAS
scores each answer:

  faithfulness       is every claim in the answer supported by the passages?
  answer_relevancy   does the answer address the question that was asked?
  context_precision  are the passages that matter ranked above the ones that don't?
  context_recall     do the passages contain everything the reference answer needs?

Each case also has an expected behaviour (answer / refuse / any). That check is
how the adversarial cases are graded: a refusal puts no passages in front of
the LLM, so there is nothing for RAGAS to score.

Repeatable by construction: the judge runs at temperature 0 with a fixed seed,
every judge call is cached on disk keyed by its exact prompt, and embeddings
come from the local ONNX model. An unchanged answer always gets the same
score. Each run is compared with the previous one, so any drift shows up.

    python -m scripts.ragas_eval
    python -m scripts.ragas_eval --ids cli-curb65 adv-rbac-clinical
    python -m scripts.ragas_eval --responses eval/results/<run>/per_question.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from unittest.mock import patch

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from ragas.cache import DiskCacheBackend
from ragas.embeddings import BaseRagasEmbeddings

from app import config, main, observability, rbac
from app.auth import DEMO_USERS
from app.retrieval import embeddings, rag, store
from scripts import answer_judge, heuristics

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
DATASET_PATH = EVAL_DIR / "eval_set.json"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / ".ragas_cache"

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
BEHAVIORS = ("answer", "refuse", "any")
CASE_FIELDS = ("id", "category", "role", "question", "reference", "expected_behavior")
SEED = 42


# --- Dataset ------------------------------------------------------------------

def load_cases(path: Path = DATASET_PATH, ids: list[str] | None = None) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    seen: set[str] = set()
    for case in cases:
        missing = [field for field in CASE_FIELDS if not case.get(field)]
        if missing:
            raise ValueError(f"case {case.get('id', '?')}: missing {', '.join(missing)}")
        if case["id"] in seen:
            raise ValueError(f"duplicate case id {case['id']!r}")
        if case["role"] not in rbac.ROLES:
            raise ValueError(f"case {case['id']}: unknown role {case['role']!r}")
        if case["expected_behavior"] not in BEHAVIORS:
            raise ValueError(f"case {case['id']}: expected_behavior must be one of {BEHAVIORS}")
        seen.add(case["id"])
    if ids:
        unknown = set(ids) - seen
        if unknown:
            raise ValueError(f"unknown case ids: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case["id"] in ids]
    return cases


# --- Running the system -----------------------------------------------------------

def _user(role: str) -> dict:
    user = next(u for u in DEMO_USERS.values() if u.role == role)
    return {"username": user.username, "role": user.role}


def _behavior(response: main.ChatResponse) -> str:
    """What the system did, independent of how good the answer was."""
    if response.guardrail.blocked:
        return "blocked"
    if response.access_denied or not response.sources:
        return "refused"  # RBAC topic, SQL not permitted, or retrieval too weak to answer
    return "answered"


def behavior_ok(expected: str, actual: str) -> bool:
    if actual == "error":
        return False
    return expected == "any" or (expected == "answer") == (actual == "answered")


def run_case(case: dict) -> dict:
    """One question through the full /chat pipeline, as the case's role.

    The API never returns the passages (they stay server-side for the output
    guardrail), so the two places that build the LLM's context are wrapped
    while the question runs.
    """
    user = _user(case["role"])
    contexts: list[str] = []
    format_context = rag._format_context
    sql_rag = main.sql_rag_with_details

    def record_passages(chunks):
        contexts[:] = [format_context([chunk]) for chunk in chunks]
        return format_context(chunks)

    def record_sql(question):
        result = sql_rag(question)
        contexts[:] = [result["context"]]
        return result

    record = {field: case[field] for field in CASE_FIELDS}
    try:
        with observability.request_scope(endpoint="eval", **user) as request, \
                patch.object(rag, "_format_context", record_passages), \
                patch.object(main, "sql_rag_with_details", record_sql):
            record["request_id"] = request.request_id
            started = time.perf_counter()
            response = main._chat(case["question"], user, langsmith_extra=request.langsmith_extra())
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    except Exception as exc:  # e.g. the 503 raised when the LLM is unreachable
        detail = getattr(exc, "detail", None) or exc
        return {**record, "response": "", "retrieved_contexts": [], "behavior": "error",
                "retrieval_type": None, "sources": [], "error": f"{type(exc).__name__}: {detail}"[:300]}
    return {
        **record,
        "response": response.answer,
        "retrieved_contexts": contexts,
        "behavior": _behavior(response),
        "retrieval_type": response.retrieval_type,
        "sources": [source.source_document for source in response.sources],
        "guardrail_reference": response.guardrail.reference,
        "error": None,
    }


# --- Scoring ------------------------------------------------------------------------

class LocalEmbeddings(BaseRagasEmbeddings):
    """The app's own dense model (bge-small, ONNX): local and deterministic.

    answer_relevancy compares the question with questions generated back from
    the answer - text of the same kind - so both sides use the passage encoder
    rather than bge's asymmetric query prefix.
    """

    def embed_query(self, text: str) -> list[float]:
        return embeddings.embed_dense([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embeddings.embed_dense(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def judge_llm(cache: DiskCacheBackend | None):
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    chat = ChatGroq(
        model=config.EVAL_JUDGE_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0.0,
        model_kwargs={"seed": SEED},
        max_retries=6,
        timeout=120,
    )
    # bypass_temperature keeps it at 0 (ragas raises it to 0.3 for multi-sample
    # prompts); bypass_n because Groq only accepts n=1, so ragas sends n requests.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return LangchainLLMWrapper(chat, cache=cache, bypass_temperature=True, bypass_n=True)


def judge_cache() -> DiskCacheBackend:
    # the cache key is the prompt, not the model - so one cache per judge model
    return DiskCacheBackend(str(CACHE_DIR / _slug(config.EVAL_JUDGE_MODEL)))


def _metrics() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    return {
        "faithfulness": Faithfulness(),
        # the judge is deterministic, so extra samples would be identical questions
        "answer_relevancy": ResponseRelevancy(strictness=1),
        "context_precision": LLMContextPrecisionWithReference(),
        "context_recall": LLMContextRecall(),
    }


def score(records: list[dict], *, llm, embeddings_model, workers: int = 2) -> None:
    """Add the four RAGAS scores to every record; None where not computable."""
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate

    for record in records:
        record.update({metric: None for metric in METRICS})
        record["note"] = None
        if record.get("error"):
            record["note"] = f"system error: {record['error']}"
        elif not record["retrieved_contexts"]:
            record["note"] = f"not scored: no passages reached the LLM ({record['behavior']})"

    scorable = [r for r in records if r["retrieved_contexts"]]
    if not scorable:
        return
    metrics = _metrics()
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in scorable
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = evaluate(
            dataset,
            metrics=list(metrics.values()),
            llm=llm,
            embeddings=embeddings_model,
            run_config=RunConfig(timeout=180, max_retries=8, max_wait=90, max_workers=workers, seed=SEED),
            raise_exceptions=False,
        )
    for record, row in zip(scorable, result.scores):
        for key, metric in metrics.items():
            value = row.get(metric.name)
            record[key] = None if value is None or math.isnan(value) else round(float(value), 4)
        failed = [metric for metric in METRICS if record[metric] is None]
        if failed:
            record["note"] = f"judge call failed for: {', '.join(failed)}"


# --- Reporting ----------------------------------------------------------------------

def aggregate(records: list[dict]) -> dict:
    def block(rows: list[dict]) -> dict:
        out = {"cases": len(rows), "behavior_pass_rate": round(mean(r["behavior_pass"] for r in rows), 4)}
        for metric in METRICS:
            values = [r[metric] for r in rows if r[metric] is not None]
            out[metric] = {"mean": round(mean(values), 4) if values else None, "scored": len(values)}
        return out

    categories = sorted({r["category"] for r in records})
    return {
        "overall": block(records),
        "by_category": {c: block([r for r in records if r["category"] == c]) for c in categories},
    }


def previous_run(current: Path) -> Path | None:
    runs = sorted(
        p for p in RESULTS_DIR.glob("*")
        if p.is_dir() and p != current and (p / "per_question.json").exists()
    )
    return runs[-1] if runs else None


def _answer_key(record: dict) -> str:
    """The answer minus the per-request guardrail reference quoted in refusals."""
    reference = record.get("guardrail_reference")
    return record["response"].replace(reference, "<ref>") if reference else record["response"]


def compare(records: list[dict], previous_dir: Path) -> dict:
    """How far this run moved from the last one, answer by answer."""
    before = {
        r["id"]: r
        for r in json.loads((previous_dir / "per_question.json").read_text(encoding="utf-8"))
    }
    common = [r for r in records if r["id"] in before]
    changed = [
        {"id": r["id"], "metric": m, "before": before[r["id"]].get(m), "after": r[m]}
        for r in common for m in METRICS
        if before[r["id"]].get(m) != r[m]
    ]
    max_delta = {}
    for metric in METRICS:
        diffs = [
            abs(r[metric] - before[r["id"]][metric])
            for r in common
            if r[metric] is not None and before[r["id"]].get(metric) is not None
        ]
        max_delta[metric] = round(max(diffs), 4) if diffs else None
    return {
        "previous_run": previous_dir.name,
        "cases_compared": len(common),
        "identical_answers": sum(_answer_key(r) == _answer_key(before[r["id"]]) for r in common),
        "identical_behavior": sum(r["behavior"] == before[r["id"]]["behavior"] for r in common),
        "max_abs_delta": max_delta,
        "changed_scores": changed,
    }


CSV_FIELDS = (
    "id", "category", "role", "expected_behavior", "behavior", "behavior_pass",
    "retrieval_type", "n_contexts", *METRICS, "note", "question", "response",
    "reference", "request_id", "latency_ms", "heuristics_passed", "heuristics_failed", "judge_passed", "judge_overall",
    *(f"judge_{c}" for c in answer_judge.CRITERIA), "judge_justification",
)


def write_results(run_dir: Path, records: list[dict], summary: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "per_question.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (run_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            judge = record.get("judge") or {}
            heur = record.get("heuristics") or {}
            writer.writerow({
                **record, "heuristics_passed": heur.get("passed"),
                "heuristics_failed": ",".join(heur.get("failed", [])), "n_contexts": len(record["retrieved_contexts"]),
                "judge_passed": judge.get("passed"), "judge_overall": judge.get("overall_score"),
                **{f"judge_{c}": (judge.get("scores") or {}).get(c) for c in answer_judge.CRITERIA},
                "judge_justification": judge.get("error") or judge.get("justification"),
            })


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def print_report(records: list[dict], summary: dict) -> None:
    headers = ("faith", "relev", "c.prec", "c.rec")
    print(f"\n{'case':<24}{'category':<12}{'expected':<9}{'got':<9}{'ok':<4}"
          + "".join(f"{h:>8}" for h in headers))
    for r in records:
        print(f"{r['id']:<24}{r['category']:<12}{r['expected_behavior']:<9}{r['behavior']:<9}"
              f"{'Y' if r['behavior_pass'] else 'N':<4}"
              + "".join(f"{_fmt(r[m]):>8}" for m in METRICS))
        if r["note"]:
            print(f"{'':<26}{r['note'][:110]}")

    print(f"\n{'aggregate':<24}{'cases':>6}{'behavior':>10}" + "".join(f"{h:>13}" for h in headers))
    scopes = {"overall": summary["aggregate"]["overall"], **summary["aggregate"]["by_category"]}
    for name, block in scopes.items():
        cells = "".join(
            f"{_fmt(block[m]['mean']) + ' (n=' + str(block[m]['scored']) + ')':>13}" for m in METRICS
        )
        print(f"{name:<24}{block['cases']:>6}{block['behavior_pass_rate']:>10.0%}{cells}")

    heur = summary["heuristics"]
    print(f"\nHeuristic checks: {_fmt(heur['pass_rate'])} of cases pass every applicable check")
    for name, c in heur["checks"].items():
        print(f"  {name:<22}pass {c['passed']:<3} fail {c['failed']:<3} n/a {c['not_applicable']}")
    for r in records:
        for name in r["heuristics"]["failed"]:
            print(f"  FAIL {r['id']:<24}{name}: {r['heuristics']['checks'][name]['detail']}")

    judge = summary.get("judge")
    if judge:
        print(f"\nLLM judge ({summary['judge_model']}): pass rate {_fmt(judge['pass_rate'])}, "
              f"mean overall {_fmt(judge['overall_mean'])}/5, {judge['errors']} unusable verdict(s)")
        print("  " + ", ".join(f"{c} {_fmt(b['mean'])} (n={b['scored']})" for c, b in judge["criteria"].items()))
        for r in records:
            j = r["judge"]
            print(f"  {r['id']:<24}{'PASS' if j['passed'] else 'FAIL':<6}{_fmt(j['overall_score']):>6}  "
                  f"{(j['error'] or j['justification'])[:100]}")

    comparison = summary.get("comparison")
    if comparison:
        deltas = ", ".join(f"{m}={_fmt(d)}" for m, d in comparison["max_abs_delta"].items())
        print(f"\nvs previous run {comparison['previous_run']}: "
              f"{comparison['identical_answers']}/{comparison['cases_compared']} identical answers, "
              f"{comparison['identical_behavior']}/{comparison['cases_compared']} identical behaviour, "
              f"{len(comparison['changed_scores'])} changed scores; max |delta| {deltas}")


# --- Entry point ------------------------------------------------------------------

def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAGAS evaluation of MediBot.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--ids", nargs="+", help="only run these case ids")
    parser.add_argument("--responses", type=Path,
                        help="re-score the answers saved by an earlier run instead of querying the system")
    parser.add_argument("--workers", type=int, default=2,
                        help="concurrent judge calls (keep low on Groq's free tier)")
    parser.add_argument("--no-cache", action="store_true", help="do not read or write the judge cache")
    parser.add_argument("--fail-under", type=float,
                        help="exit 1 if any overall metric mean is below this value")
    parser.add_argument("--no-judge", action="store_true", help="skip the rubric-based LLM judge step")
    parser.add_argument("--verbose", action="store_true", help="show application and ragas logs")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")
    if not args.verbose:
        logging.disable(logging.ERROR)  # guardrail blocks log at ERROR; keep the table readable
    if not config.GROQ_API_KEY:
        print("GROQ_API_KEY is not set - both MediBot and the RAGAS judge need it.")
        return 2

    cases = load_cases(args.dataset, args.ids)
    if args.responses:
        saved = {r["id"]: r for r in json.loads(args.responses.read_text(encoding="utf-8"))}
        missing = [c["id"] for c in cases if c["id"] not in saved]
        if missing:
            print(f"{args.responses} has no saved answer for: {', '.join(missing)}")
            return 2
        # labels come from the current dataset, so an edited reference is re-scored
        records = [{**saved[c["id"]], **{f: c[f] for f in CASE_FIELDS}} for c in cases]
    else:
        try:
            store.count_points()
        except Exception as exc:
            print("Cannot open the Qdrant index. Embedded Qdrant allows one process at a time - "
                  f"stop the API server first.\n  {exc}")
            return 2
        records = []
        try:
            for index, case in enumerate(cases, start=1):
                print(f"[{index}/{len(cases)}] {case['id']} ({case['role']})", flush=True)
                records.append(run_case(case))
        finally:
            store.close_client()  # release the embedded index lock before scoring

    for record in records:
        record["behavior_pass"] = behavior_ok(record["expected_behavior"], record["behavior"])

    cache = None if args.no_cache else judge_cache()
    score(records, llm=judge_llm(cache), embeddings_model=LocalEmbeddings(), workers=args.workers)

    heuristics.check_all(records)
    if not args.no_judge:
        answer_judge.judge_all(
            records, cache_dir=None if args.no_cache else CACHE_DIR / "judge" / _slug(config.EVAL_JUDGE_MODEL)
        )

    run_dir = RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary = {
        "run_id": run_dir.name,
        "generator_model": config.GROQ_MODEL,
        "judge_model": config.EVAL_JUDGE_MODEL,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "responses_from": str(args.responses) if args.responses else None,
        "judge_cache": not args.no_cache,
        "aggregate": aggregate(records),
        "heuristics": heuristics.aggregate(records),
    }
    if not args.no_judge:
        summary["judge"] = answer_judge.aggregate(records)
    previous = previous_run(run_dir)
    if previous:
        summary["comparison"] = compare(records, previous)
    write_results(run_dir, records, summary)
    print_report(records, summary)
    print(f"\nResults written to {run_dir}")

    errors = [r["id"] for r in records if r["behavior"] == "error"]
    if errors:
        print(f"{len(errors)} case(s) could not be run: {', '.join(errors)}")
        return 1
    failing = [r["id"] for r in records if not r["heuristics"]["passed"]]
    if failing:
        print(f"{len(failing)} case(s) failed a heuristic check: {', '.join(failing)}")
        return 1
    if args.fail_under is not None:
        low = {
            m: block["mean"] for m, block in summary["aggregate"]["overall"].items()
            if m in METRICS and block["mean"] is not None and block["mean"] < args.fail_under
        }
        if low:
            print(f"Below --fail-under {args.fail_under}: {low}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
