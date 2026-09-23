"""SQL RAG - answering analytical questions from mediassist.db.

Implemented as a plain Python function with three explicit steps:
  1. natural language -> SQL (LLM)
  2. clean the raw LLM output down to a single executable statement
  3. execute, then hand the rows back to the LLM for a natural answer
"""
from __future__ import annotations

import logging
import re
import sqlite3
from functools import lru_cache

from . import config
from .llm import complete

log = logging.getLogger(__name__)

MAX_ROWS = 50

SQL_SYSTEM_PROMPT = """You translate questions about a hospital operations database into a single SQLite SELECT query.

Hard rules:
- Output ONLY the SQL. No prose, no explanation, no markdown fences.
- Exactly one statement, and it must be a SELECT (never INSERT/UPDATE/DELETE/DROP/PRAGMA/ATTACH).
- Dates are stored as TEXT in 'YYYY-MM-DD' format; use strftime() for month/year grouping.
- Text values are lowercase snake_case unless shown otherwise in the schema notes.
- Always alias aggregate columns with readable names.
- Add a LIMIT when the question implies a ranking or a list."""

ANSWER_SYSTEM_PROMPT = """You are MediBot, reporting figures from MediAssist's operations database.

Answer the question using ONLY the query result rows given to you. State the
numbers plainly, keep it to a few sentences or a short list, and mention the
figure came from the operational database. Amounts are Indian Rupees (INR).
If the result set is empty, say no matching records were found."""

_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)


class SQLRagError(RuntimeError):
    pass


def _connect() -> sqlite3.Connection:
    if not config.DB_PATH.exists():
        raise SQLRagError(f"Database not found at {config.DB_PATH}")
    # read-only URI: the analytics path can never mutate operational data
    connection = sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


@lru_cache(maxsize=1)
def get_schema_prompt() -> str:
    """Schema plus the distinct values of low-cardinality columns.

    Showing the LLM the real value vocabulary ('in_progress', not 'In Progress')
    is what stops most of the silently-empty-result-set failures.
    """
    sample_columns = {
        "claims": ("department", "claim_type", "status", "insurer"),
        "maintenance_tickets": ("category", "issue_type", "status", "campus"),
    }
    parts: list[str] = []
    with _connect() as connection:
        tables = [
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
            column_lines = "\n".join(f"  {c['name']} {c['type']}" for c in columns)
            count = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            parts.append(f"TABLE {table} ({count} rows)\n{column_lines}")

            for column in sample_columns.get(table, ()):
                values = [
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL LIMIT 20"
                    )
                ]
                if values:
                    parts.append(f"  distinct {table}.{column}: {', '.join(values)}")
    return "\n".join(parts)


def generate_sql(question: str) -> str:
    """Step 1: natural language -> raw LLM output."""
    return complete(
        SQL_SYSTEM_PROMPT,
        f"Database schema:\n{get_schema_prompt()}\n\nQuestion: {question}\n\nSQL:",
        temperature=0.0,
        max_tokens=400,
    )


def clean_sql(raw: str) -> str:
    """Step 2: strip fences/prose and validate that one read-only SELECT remains."""
    text = raw.strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    match = _SELECT_RE.search(text)
    if not match:
        raise SQLRagError(f"No SELECT statement found in model output: {raw!r}")
    text = text[match.start() :].strip()

    # keep only the first statement
    text = text.split(";")[0].strip()
    # drop any trailing commentary the model tacked on after the query
    text = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith(("--", "#"))
    ).strip()

    if _FORBIDDEN_RE.search(text):
        raise SQLRagError(f"Refusing to execute non-read-only SQL: {text!r}")
    return text


def execute_sql(sql: str) -> tuple[list[str], list[tuple]]:
    """Step 3a: run the query against SQLite."""
    with _connect() as connection:
        cursor = connection.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = [tuple(row) for row in cursor.fetchmany(MAX_ROWS)]
    return columns, rows


def _format_rows(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(no rows returned)"
    header = " | ".join(columns)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    return f"{header}\n{body}"


def summarise(question: str, sql: str, columns: list[str], rows: list[tuple]) -> str:
    """Step 3b: rows -> natural language answer."""
    return complete(
        ANSWER_SYSTEM_PROMPT,
        f"Question: {question}\n\nSQL executed:\n{sql}\n\nResult:\n{_format_rows(columns, rows)}",
        temperature=0.1,
    )


def sql_rag_with_details(question: str) -> dict:
    """Full chain, returning the intermediate artefacts for the API/debugging."""
    raw_sql = generate_sql(question)
    sql = clean_sql(raw_sql)
    log.info("SQL RAG query: %s", sql)
    columns, rows = execute_sql(sql)
    return {
        "answer": summarise(question, sql, columns, rows),
        # grounding context for the output guardrail, never returned to the client
        "context": f"SQL executed:\n{sql}\n\nResult rows:\n{_format_rows(columns, rows)}",
        "sql": sql,
        "raw_sql": raw_sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


def sql_rag_chain(question: str) -> str:
    """Natural language question -> natural language answer, via SQL."""
    return sql_rag_with_details(question)["answer"]


if __name__ == "__main__":  # quick manual check: python -m app.sql_rag
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for q in (
        "How many billing claims are still pending?",
        "Which equipment category has the most open maintenance tickets?",
        "What is the total claimed amount per department, highest first?",
        "How many claims were submitted in December 2024?",
    ):
        result = sql_rag_with_details(q)
        print(f"\nQ: {q}\nSQL: {result['sql']}\nA: {result['answer']}")
