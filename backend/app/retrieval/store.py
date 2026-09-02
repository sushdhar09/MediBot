"""Qdrant vector store: hybrid (dense + BM25) index and RBAC-filtered search.

The two vector types live in the *same* collection as named vectors, so a
single Qdrant Query API call runs both searches server-side and fuses them with
Reciprocal Rank Fusion. Crucially, the `access_roles` filter is attached to the
prefetch branches *and* the fusion query, so restricted chunks are excluded
inside the database - they never reach the application, let alone the LLM.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient, models

from .. import config
from . import embeddings

log = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


@dataclass
class RetrievedChunk:
    text: str
    source_document: str
    section_title: str
    collection: str
    chunk_type: str
    access_roles: list[str]
    page_numbers: list[int]
    fusion_score: float
    rerank_score: float | None = None

    @classmethod
    def from_point(cls, point) -> "RetrievedChunk":
        payload = point.payload or {}
        return cls(
            text=payload.get("text", ""),
            source_document=payload.get("source_document", "unknown"),
            section_title=payload.get("section_title", ""),
            collection=payload.get("collection", "unknown"),
            chunk_type=payload.get("chunk_type", "text"),
            access_roles=payload.get("access_roles", []),
            page_numbers=payload.get("page_numbers", []),
            fusion_score=float(getattr(point, "score", 0.0) or 0.0),
        )

    def citation(self) -> dict:
        return {
            "source_document": self.source_document,
            "section_title": self.section_title,
            "collection": self.collection,
        }


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Local (embedded) Qdrant - persists to disk, no server or Docker needed."""
    config.QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def close_client() -> None:
    if get_client.cache_info().currsize:
        get_client().close()
        get_client.cache_clear()


def recreate_collection(client: QdrantClient | None = None) -> None:
    client = client or get_client()
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=config.DENSE_DIM, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            # IDF modifier is what makes this a real BM25 scorer in Qdrant.
            SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    # Payload indexes keep the RBAC filter fast and exact.
    for field in ("access_roles", "collection", "source_document", "chunk_type"):
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def collection_exists(client: QdrantClient | None = None) -> bool:
    client = client or get_client()
    return client.collection_exists(config.QDRANT_COLLECTION)


def count_points(client: QdrantClient | None = None) -> int:
    client = client or get_client()
    if not collection_exists(client):
        return 0
    return client.count(config.QDRANT_COLLECTION, exact=True).count


def index_chunks(chunks, batch_size: int = 32, client: QdrantClient | None = None) -> int:
    """Embed chunks with both models and upsert them as a single point each."""
    client = client or get_client()
    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.text for c in batch]
        dense_vectors = embeddings.embed_dense(texts)
        sparse_vectors = embeddings.embed_sparse(texts)

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload=chunk.to_payload(),
            )
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors)
        ]
        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        total += len(points)
        log.info("Indexed %d/%d chunks", total, len(chunks))
    return total


def rbac_filter(role: str, collections: list[str] | None = None) -> models.Filter:
    """The security boundary: only chunks tagged with this role can match."""
    conditions = [
        models.FieldCondition(key="access_roles", match=models.MatchAny(any=[role]))
    ]
    if collections:
        conditions.append(
            models.FieldCondition(key="collection", match=models.MatchAny(any=collections))
        )
    return models.Filter(must=conditions)


def hybrid_search(
    query: str,
    role: str,
    limit: int | None = None,
    collections: list[str] | None = None,
    client: QdrantClient | None = None,
) -> list[RetrievedChunk]:
    client = client or get_client()
    if not collection_exists(client):
        raise RuntimeError(
            "Qdrant collection is empty. Run `python -m app.ingestion.ingest` first."
        )

    limit = limit or config.HYBRID_CANDIDATES
    access = rbac_filter(role, collections)
    sparse_query = embeddings.embed_sparse_query(query)

    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=embeddings.embed_dense_query(query),
                using=DENSE_VECTOR,
                filter=access,
                limit=config.PREFETCH_LIMIT,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices.tolist(),
                    values=sparse_query.values.tolist(),
                ),
                using=SPARSE_VECTOR,
                filter=access,
                limit=config.PREFETCH_LIMIT,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=access,
        limit=limit,
        with_payload=True,
    )
    return [RetrievedChunk.from_point(point) for point in response.points]


def dense_only_search(
    query: str, role: str, limit: int | None = None, client: QdrantClient | None = None
) -> list[RetrievedChunk]:
    """Dense-only baseline, used by the hybrid-vs-dense comparison script."""
    client = client or get_client()
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=embeddings.embed_dense_query(query),
        using=DENSE_VECTOR,
        query_filter=rbac_filter(role),
        limit=limit or config.HYBRID_CANDIDATES,
        with_payload=True,
    )
    return [RetrievedChunk.from_point(point) for point in response.points]
