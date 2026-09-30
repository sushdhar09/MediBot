"""Cross-encoder reranking.

Bi-encoder retrieval scores the query and the chunk independently. A cross
encoder reads them *together*, which is slower but far more accurate - so we
retrieve broad (top-10) and rerank narrow (top-3) before anything reaches the
LLM.
"""
from __future__ import annotations

import logging

from .. import config, observability
from . import embeddings
from .store import RetrievedChunk

log = logging.getLogger(__name__)


@observability.traced(
    "rerank",
    process_inputs=lambda inputs: {
        "query": inputs["query"],
        "top_k": inputs.get("top_k"),
        "candidates": [c.summary() for c in inputs["candidates"]],
    },
    process_outputs=lambda kept: {"kept": [c.summary() for c in kept]},
)
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
    # The full ranking, dropped candidates included: "why was X not used?"
    observability.span_metadata(
        model=config.RERANK_MODEL,
        ranking=[{**c.summary(), "kept": i < top_k} for i, c in enumerate(ranked)],
    )
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
