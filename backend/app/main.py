"""FastAPI application exposing MediBot to the Next.js frontend."""
from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config, rbac, routing
from .auth import DEMO_USERS, AuthError, authenticate, create_token, decode_token
from .llm import LLMNotConfigured
from .retrieval import store
from .retrieval.rag import answer_from_documents
from .sql_rag import SQLRagError, sql_rag_with_details

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("medibot")

app = FastAPI(title="MediBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str
    access_denied: bool = False
    debug: dict | None = None


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


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: dict = Depends(current_user)) -> ChatResponse:
    role = user["role"]
    question = payload.question.strip()

    try:
        if routing.is_analytical_question(question):
            if not rbac.can_use_sql_rag(role):
                return ChatResponse(
                    answer=rbac.sql_denied_message(role),
                    sources=[],
                    retrieval_type="sql_rag",
                    role=role,
                    access_denied=True,
                )
            result = sql_rag_with_details(question)
            return ChatResponse(
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
            )

        result = answer_from_documents(question, role)
        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            retrieval_type="hybrid_rag",
            role=role,
            access_denied=result["access_denied"],
            debug={
                "candidates_considered": result["candidates_considered"],
                "reranked": result["reranked"],
            },
        )
    except LLMNotConfigured as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except SQLRagError as exc:
        log.exception("SQL RAG failed")
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except RuntimeError as exc:
        log.exception("Retrieval failed")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
