"""FastAPI application exposing MediBot to the Next.js frontend."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, guardrails, observability, rbac, routing
from .auth import DEMO_USERS, AuthError, authenticate, create_token, decode_token
from .llm import LLMNotConfigured, LLMUnavailable
from .retrieval import store
from .retrieval.rag import answer_from_documents
from .sql_rag import SQLRagError, sql_rag_with_details

observability.configure_logging()
log = logging.getLogger("medibot")

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    observability.flush()


app = FastAPI(title="MediBot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    display_name: str
    role: str
    department: str
    collections: list[str]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str


class GuardrailInfo(BaseModel):
    """What the client is allowed to know about a guardrail decision.

    Never the category and never the reason - those stay in the server log,
    keyed by `reference`.
    """

    blocked: bool = False
    redacted: bool = False
    stage: str | None = None
    reference: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str
    access_denied: bool = False
    guardrail: GuardrailInfo = GuardrailInfo()
    debug: dict | None = None
    # Also the LangSmith trace id, and the key of every log line for this request.
    request_id: str | None = None


def current_user(authorization: str = Header(default="")) -> dict:
    """Resolve the caller from the bearer token. The role never comes from the body."""
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        return decode_token(authorization.split(" ", 1)[1].strip())
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc


@app.get("/health")
def health() -> dict:
    try:
        indexed = store.count_points()
    except Exception as exc:  # index not built yet, or locked by the ingest script
        return {"status": "degraded", "indexed_chunks": 0, "detail": str(exc)}
    return {
        "status": "ok" if indexed else "degraded",
        "indexed_chunks": indexed,
        "llm_configured": bool(config.GROQ_API_KEY),
        "model": config.GROQ_MODEL,
    }


@app.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    try:
        user = authenticate(payload.username, payload.password)
    except AuthError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    return LoginResponse(
        token=create_token(user),
        username=user.username,
        display_name=user.display_name,
        role=user.role,
        department=rbac.ROLE_DEPARTMENTS[user.role],
        collections=rbac.collections_for_role(user.role),
    )


@app.get("/collections/{role}")
def collections(role: str) -> dict:
    if role not in rbac.ROLES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown role: {role}")
    accessible = rbac.collections_for_role(role)
    return {
        "role": role,
        "department": rbac.ROLE_DEPARTMENTS[role],
        "collections": [
            {"name": name, "description": rbac.COLLECTION_DESCRIPTIONS[name]}
            for name in accessible
        ],
        "restricted_collections": [c for c in rbac.COLLECTION_ACCESS if c not in accessible],
        "sql_rag_enabled": rbac.can_use_sql_rag(role),
    }


@app.get("/demo-users")
def demo_users() -> list[dict]:
    """Credentials for the login screen - demo dataset only."""
    return [
        {
            "username": u.username,
            "password": u.password,
            "role": u.role,
            "display_name": u.display_name,
            "department": rbac.ROLE_DEPARTMENTS[u.role],
        }
        for u in DEMO_USERS.values()
    ]


def _guarded(
    response: ChatResponse,
    *,
    question: str,
    context: str,
    user: dict,
    is_refusal: bool = False,
) -> ChatResponse:
    """Run the output guardrail over a finished answer before it leaves the API."""
    verdict = guardrails.record(
        guardrails.guard_output(
            question=question,
            answer=response.answer,
            context=context,
            role=response.role,
            sources=[s.model_dump() for s in response.sources],
            is_refusal=is_refusal,
        ),
        username=user["username"],
        question=question,
    )
    info = GuardrailInfo(stage="output", reference=verdict.reference)
    if verdict.action == "block":
        return response.model_copy(update={
            "answer": guardrails.messages.output_refusal(verdict.reference),
            "sources": [],
            "debug": None,
            "guardrail": info.model_copy(update={"blocked": True}),
        })
    if verdict.action == "redact":
        return response.model_copy(update={
            "answer": (verdict.sanitized_output or response.answer)
            + guardrails.messages.REDACTION_NOTICE,
            "guardrail": info.model_copy(update={"redacted": True}),
        })
    return response.model_copy(update={"guardrail": info})


def _outcome(response: ChatResponse) -> str:
    if response.guardrail.blocked:
        return f"blocked_{response.guardrail.stage}"
    if response.access_denied:
        return "access_denied"
    return "redacted" if response.guardrail.redacted else "answered"


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, response: Response, user: dict = Depends(current_user)) -> ChatResponse:
    question = payload.question.strip()
    with observability.request_scope(
        endpoint="/chat", username=user["username"], role=user["role"]
    ) as request:
        response.headers[REQUEST_ID_HEADER] = request.request_id
        try:
            result = _chat(question, user, langsmith_extra=request.langsmith_extra())
        except HTTPException as exc:
            exc.headers = {**(exc.headers or {}), REQUEST_ID_HEADER: request.request_id}
            raise
        return result.model_copy(update={"request_id": request.request_id})


@observability.traced(
    "medibot.chat",
    process_inputs=lambda inputs: {
        "question": inputs["question"],
        "username": inputs["user"]["username"],
        "role": inputs["user"]["role"],
    },
)
def _chat(question: str, user: dict) -> ChatResponse:
    """The whole request path under one LangSmith root run."""
    observability.bind_root_run()
    result = _pipeline(question, user)
    observability.annotate(retrieval_type=result.retrieval_type, outcome=_outcome(result))
    return result


def _pipeline(question: str, user: dict) -> ChatResponse:
    role = user["role"]

    # Gate 1: nothing reaches the router, the index or the LLM until this passes.
    verdict = guardrails.record(
        guardrails.guard_input(question, role=role),
        username=user["username"],
        question=question,
    )
    if not verdict.allowed:
        return ChatResponse(
            answer=guardrails.messages.input_refusal(verdict.reference),
            sources=[],
            retrieval_type="blocked",
            role=role,
            guardrail=GuardrailInfo(blocked=True, stage="input", reference=verdict.reference),
        )

    try:
        analytical = routing.is_analytical_question(question)
        route = "sql_rag" if analytical else "hybrid_rag"
        observability.annotate(route=route)
        observability.emit(
            "route.decision", route=route, sql_permitted=rbac.can_use_sql_rag(role), role=role
        )
        if analytical:
            if not rbac.can_use_sql_rag(role):
                return _guarded(
                    ChatResponse(
                        answer=rbac.sql_denied_message(role),
                        sources=[],
                        retrieval_type="sql_rag",
                        role=role,
                        access_denied=True,
                    ),
                    question=question, context="", user=user, is_refusal=True,
                )
            result = sql_rag_with_details(question)
            return _guarded(
                ChatResponse(
                    answer=result["answer"],
                    sources=[
                        Source(
                            source_document="mediassist.db",
                            section_title=f"SQL over {', '.join(result['columns']) or 'operational tables'}",
                            collection="database",
                        )
                    ],
                    retrieval_type="sql_rag",
                    role=role,
                    debug={"sql": result["sql"], "row_count": result["row_count"]},
                ),
                question=question, context=result["context"], user=user,
            )

        result = answer_from_documents(question, role)
        return _guarded(
            ChatResponse(
                answer=result["answer"],
                sources=[Source(**s) for s in result["sources"]],
                retrieval_type="hybrid_rag",
                role=role,
                access_denied=result["access_denied"],
                debug={
                    "candidates_considered": result["candidates_considered"],
                    "reranked": result["reranked"],
                },
            ),
            question=question,
            context=result["context"],
            user=user,
            is_refusal=result["is_refusal"],
        )
    except LLMNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except LLMUnavailable as exc:
        # Already logged with the underlying provider error in llm.complete.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except SQLRagError as exc:
        log.exception("SQL RAG failed")
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except RuntimeError as exc:
        log.exception("Retrieval failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
