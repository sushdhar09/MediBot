"""Request-scoped observability: LangSmith traces plus a structured JSON event log.

Every POST /chat gets one request id, and that id is

  * the LangSmith root run id - paste it into the LangSmith UI to open the trace
  * stamped on every log record written while the request runs
  * returned to the client (``X-Request-ID`` header and ``request_id`` field)

so a reviewer can go from a user report to the log lines to the trace without
re-running anything. The trace holds the full payloads (question, retrieved
text, the exact prompt, the answer); the event log holds the decisions and the
numbers (scores, verdicts, latency, tokens) as queryable JSON.
"""
from __future__ import annotations

import functools
import json
import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Iterator
from uuid import UUID

from langsmith import run_trees, utils as ls_utils
from langsmith.run_helpers import get_current_run_tree, traceable
from langsmith.uuid import uuid7

from . import config

log = logging.getLogger(__name__)
_events = logging.getLogger("medibot.events")


# --- Request context ----------------------------------------------------------

@dataclass
class RequestContext:
    request_id: str
    endpoint: str
    username: str
    role: str
    started: float = field(default_factory=time.perf_counter)
    status: str = "ok"
    http_status: int = 200
    error: str | None = None
    # Inclusive span durations, summed when a span runs more than once (judges).
    stages_ms: dict[str, float] = field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    llm_calls: int = 0
    guardrails: list[dict[str, Any]] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)
    root_run: Any = None

    def add_stage(self, name: str, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000
        self.stages_ms[name] = round(self.stages_ms.get(name, 0.0) + elapsed, 1)

    def add_usage(self, model: str, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self.llm_calls += 1
        usage = self.tokens_by_model.setdefault(
            model, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        )
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += total_tokens

    def langsmith_extra(self) -> dict[str, Any]:
        """Makes the request id the LangSmith root run id (and hence the trace id)."""
        return {
            "run_id": UUID(self.request_id),
            "metadata": {"request_id": self.request_id, "username": self.username, "role": self.role},
            "tags": ["medibot", f"role:{self.role}"],
        }

    def summary(self) -> dict[str, Any]:
        totals = {
            key: sum(usage[key] for usage in self.tokens_by_model.values())
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        return {
            "trace_id": self.request_id,
            "langsmith_project": config.LANGSMITH_PROJECT,
            "traced": bool(ls_utils.tracing_is_enabled()),
            "endpoint": self.endpoint,
            "username": self.username,
            "role": self.role,
            "status": self.status,
            "http_status": self.http_status,
            "error": self.error,
            "latency_ms": round((time.perf_counter() - self.started) * 1000, 1),
            "stages_ms": self.stages_ms,
            "llm_calls": self.llm_calls,
            **totals,
            "tokens_by_model": self.tokens_by_model,
            "guardrails": self.guardrails,
            **self.fields,
        }


_current: ContextVar[RequestContext | None] = ContextVar("medibot_request", default=None)


def current() -> RequestContext | None:
    return _current.get()


@contextmanager
def request_scope(*, endpoint: str, username: str, role: str) -> Iterator[RequestContext]:
    """Open a request: one id, one metrics line (``request.completed``) at the end."""
    ctx = RequestContext(request_id=str(uuid7()), endpoint=endpoint, username=username, role=role)
    token = _current.set(ctx)
    try:
        yield ctx
    except Exception as exc:
        ctx.status = "error"
        ctx.http_status = getattr(exc, "status_code", 500)
        ctx.error = f"{type(exc).__name__}: {getattr(exc, 'detail', None) or exc}"[:300]
        raise
    finally:
        summary = ctx.summary()
        headline = " ".join(
            f"{key}={summary[key]}"
            for key in ("status", "http_status", "outcome", "route", "latency_ms",
                        "llm_calls", "total_tokens", "error")
            if summary.get(key) is not None
        )
        emit(
            "request.completed",
            level=logging.ERROR if ctx.status == "error" else logging.INFO,
            message=f"request.completed {headline}",
            **summary,
        )
        _current.reset(token)


# --- Events -------------------------------------------------------------------

def emit(
    event: str,
    *,
    level: int = logging.INFO,
    message: str | None = None,
    logger: logging.Logger | None = None,
    **data: Any,
) -> None:
    """Write one structured event. ``data`` becomes top-level keys in the JSON log."""
    if message is None:
        scalars = " ".join(
            f"{key}={value}" for key, value in data.items()
            if isinstance(value, (str, int, float, bool)) and key not in {"trace_id", "langsmith_project"}
        )
        message = f"{event} {scalars}".strip()
    ctx = current()
    (logger or _events).log(
        level, message,
        extra={"event": event, "data": data, "request_id": ctx.request_id if ctx else "-"},
    )


def annotate(**fields: Any) -> None:
    """Attach request-level facts to the metrics line and the LangSmith root run."""
    ctx = current()
    if ctx is None:
        return
    ctx.fields.update(fields)
    trace_metadata(**fields)


def trace_metadata(**fields: Any) -> None:
    """Metadata on the LangSmith root run only - filterable in the LangSmith UI."""
    ctx = current()
    if ctx is not None and ctx.root_run is not None:
        ctx.root_run.add_metadata(fields)


def tag(*tags: str) -> None:
    """Tag the LangSmith root run, so traces can be filtered by outcome."""
    ctx = current()
    if ctx is not None and ctx.root_run is not None:
        ctx.root_run.add_tags(list(tags))


def span_metadata(**fields: Any) -> None:
    """Attach metadata to the innermost open span (no-op when tracing is off)."""
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(fields)


def bind_root_run() -> None:
    """Call first thing inside the root traced function."""
    ctx = current()
    if ctx is not None:
        ctx.root_run = get_current_run_tree()


def record_llm_usage(
    *,
    model: str,
    purpose: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int | None = None,
    latency_ms: float | None = None,
    on_span: bool = True,
) -> None:
    """Count tokens toward the request and, by default, onto the current LLM span.

    ``on_span=False`` is for LangChain calls, which LangSmith already traces
    with their own usage - counting them again on our span would double it.
    """
    total = total_tokens if total_tokens is not None else input_tokens + output_tokens
    ctx = current()
    if ctx is not None:
        ctx.add_usage(model, input_tokens, output_tokens, total)
    if on_span:
        run = get_current_run_tree()
        if run is not None:
            run.set(usage_metadata={
                "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total,
            })
    emit(
        "llm.call", model=model, purpose=purpose, input_tokens=input_tokens,
        output_tokens=output_tokens, total_tokens=total, latency_ms=latency_ms,
    )


# --- Tracing ------------------------------------------------------------------

def traced(name: str, *, run_type: str = "chain", **kwargs: Any) -> Callable:
    """``langsmith.traceable`` plus per-request stage latency.

    The stage is recorded under the span name, including a per-call override
    passed as ``langsmith_extra={"name": ...}``.
    """
    def decorator(fn: Callable) -> Callable:
        traced_fn = traceable(name=name, run_type=run_type, **kwargs)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kw: Any) -> Any:
            stage = (kw.get("langsmith_extra") or {}).get("name", name)
            started = time.perf_counter()
            try:
                return traced_fn(*args, **kw)
            finally:
                ctx = current()
                if ctx is not None:
                    ctx.add_stage(stage, started)

        return wrapper

    return decorator


def flush() -> None:
    """Push buffered traces before the process exits."""
    if not ls_utils.tracing_is_enabled():
        return
    try:
        run_trees.get_cached_client().flush()
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as exc:  # never let shutdown fail on telemetry
        log.warning("LangSmith flush failed: %s", exc)


# --- Logging setup --------------------------------------------------------------

class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = current()
        record.request_id = ctx.request_id if ctx else "-"
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line - greppable, and loadable with pandas or jq."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "data", None) or {})
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging() -> None:
    """Console (human) + rotating JSONL file (machine). Safe to call twice."""
    root = logging.getLogger()
    if any(getattr(h, "_medibot", False) for h in root.handlers):
        return
    root.setLevel(config.LOG_LEVEL)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s")
    )
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    jsonl = RotatingFileHandler(
        config.EVENT_LOG_PATH, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    jsonl.setFormatter(JsonFormatter())

    for handler in (console, jsonl):
        handler.addFilter(_RequestIdFilter())
        handler._medibot = True  # type: ignore[attr-defined]
        root.addHandler(handler)
