"""Query the structured event log (logs/medibot.jsonl) - no dashboard required.

    python -m scripts.request_report                    # latency / token / outcome metrics
    python -m scripts.request_report --last 20          # one line per recent request
    python -m scripts.request_report <request_id>       # everything logged for one request
    python -m scripts.request_report --ref 9f31c2ab     # the request behind a guardrail reference

The request id is also the LangSmith trace id: paste it into the LangSmith
search bar (project from LANGSMITH_PROJECT) to open the full trace.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from app import config


def _log_files(path: Path) -> list[Path]:
    """Rotated files first (oldest), so events come out in time order."""
    rotated = sorted(path.parent.glob(f"{path.name}.*"), key=lambda p: -int(p.suffix[1:]))
    return [p for p in (*rotated, path) if p.exists()]


def read_events(path: Path) -> Iterator[dict]:
    for file in _log_files(path):
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _pct(values: list[float], q: int) -> float:
    if len(values) < 2:
        return values[0] if values else 0.0
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def show_request(events: list[dict], request_id: str) -> int:
    story = [e for e in events if e.get("request_id") == request_id]
    if not story:
        print(f"No events for request_id={request_id}")
        return 1
    for event in story:
        extra = {k: v for k, v in event.items()
                 if k not in {"ts", "level", "logger", "event", "request_id", "message"}}
        print(f"{event['ts']} {event['level']:<7} {event['event']}")
        print(f"    {event['message']}")
        if extra and event["event"] != "log":
            print("    " + json.dumps(extra, indent=2, default=str).replace("\n", "\n    "))
    print(f"\nLangSmith trace id: {request_id} (project: {config.LANGSMITH_PROJECT})")
    return 0


def show_recent(requests: list[dict], last: int) -> int:
    for r in requests[-last:]:
        blocked = [f"{g['stage']}:{g['category']}" for g in r.get("guardrails", []) if g["action"] == "block"]
        print(
            f"{r['ts']} {r['request_id']} {r.get('username', '-'):<18} {r.get('status'):<5} "
            f"{r.get('outcome', '-'):<14} {r.get('latency_ms', 0):>8.1f}ms "
            f"{r.get('total_tokens', 0):>6} tok {' '.join(blocked)}"
        )
    return 0


def show_metrics(requests: list[dict], events: list[dict]) -> int:
    if not requests:
        print(f"No completed requests in {config.EVENT_LOG_PATH}")
        return 1
    latencies = [r["latency_ms"] for r in requests]
    tokens = [r.get("total_tokens", 0) for r in requests]
    print(f"requests: {len(requests)}  ({requests[0]['ts']} .. {requests[-1]['ts']})")
    print(f"status:   {dict(Counter(r.get('status') for r in requests))}")
    print(f"outcome:  {dict(Counter(r.get('outcome', 'n/a') for r in requests))}")
    print(f"latency:  p50={_pct(latencies, 50):.0f}ms  p95={_pct(latencies, 95):.0f}ms  max={max(latencies):.0f}ms")
    print(f"tokens:   total={sum(tokens)}  mean={statistics.mean(tokens):.0f}/request  "
          f"input={sum(r.get('input_tokens', 0) for r in requests)}  "
          f"output={sum(r.get('output_tokens', 0) for r in requests)}")

    by_model: dict[str, int] = defaultdict(int)
    stages: dict[str, list[float]] = defaultdict(list)
    for r in requests:
        for model, usage in (r.get("tokens_by_model") or {}).items():
            by_model[model] += usage.get("total_tokens", 0)
        for stage, ms in (r.get("stages_ms") or {}).items():
            stages[stage].append(ms)
    if by_model:
        print(f"by model: {dict(by_model)}")
    print("\nstage latency (ms)        n     p50     p95")
    for stage, values in sorted(stages.items()):
        print(f"  {stage:<22} {len(values):>4} {_pct(values, 50):>7.0f} {_pct(values, 95):>7.0f}")

    decisions = [e for e in events if e.get("event") == "guardrail.decision"]
    blocks = Counter(f"{e['stage']}:{e['category']}" for e in decisions if e.get("action") == "block")
    print(f"\nguardrail decisions: {dict(Counter(e.get('action') for e in decisions))}")
    print(f"blocks by category:  {dict(blocks)}")
    print(f"degraded verdicts:   {sum(1 for e in decisions if e.get('degraded'))}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request_id", nargs="?", help="show every event of one request")
    parser.add_argument("--ref", help="find the request behind a guardrail reference id")
    parser.add_argument("--last", type=int, help="list the N most recent requests")
    parser.add_argument("--file", type=Path, default=config.EVENT_LOG_PATH)
    args = parser.parse_args()

    events = list(read_events(args.file))
    requests = [e for e in events if e.get("event") == "request.completed"]

    if args.ref:
        match = next((e for e in events if e.get("event") == "guardrail.decision"
                      and e.get("reference") == args.ref), None)
        if match is None:
            print(f"No guardrail decision with reference={args.ref}")
            return 1
        return show_request(events, match["request_id"])
    if args.request_id:
        return show_request(events, args.request_id)
    if args.last:
        return show_recent(requests, args.last)
    return show_metrics(requests, events)


if __name__ == "__main__":
    sys.exit(main())
