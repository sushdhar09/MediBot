"""Cross-encoder reranking.

Bi-encoder retrieval scores the query and the chunk independently. A cross
encoder reads them *together*, which is slower but far more accurate - so we
retrieve broad (top-10) and rerank narrow (top-3) before anything reaches the
LLM.
"""
from __future__ import annotations

import logging

from .. import config
from . import embeddings
from .store import RetrievedChunk

log = logging.getLogger(__name__)


def rerank(
    query: str, candidates: list[RetrievedChunk], top_k: int | None = None
) -> list[RetrievedChunk]:
    if not candidates:
        return []

    top_k = top_k or config.RERANK_TOP_K
    scores = list(embeddings.cross_encoder().rerank(query, [c.text for c in candidates]))
    for chunk, score in zip(candidates, scores):
        chunk.rerank_score = float(score)

    ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    if log.isEnabledFor(logging.DEBUG):
        for position, chunk in enumerate(ranked, start=1):
            log.debug(
                "rerank #%d score=%.3f %s :: %s",
                position,
                chunk.rerank_score,
                chunk.source_document,
                chunk.section_title,
            )
    return ranked[:top_k]
