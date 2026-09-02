"""Lazily-loaded embedding and reranking models (FastEmbed / ONNX runtime).

Kept in one place so the models are loaded exactly once per process - they are
expensive to construct and are shared by ingestion and query time.
"""
from __future__ import annotations

from functools import lru_cache

from .. import config


@lru_cache(maxsize=1)
def dense_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=config.DENSE_MODEL)


@lru_cache(maxsize=1)
def sparse_model():
    """BM25 sparse encoder - exact keyword matching for drug names, ICD codes."""
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=config.SPARSE_MODEL)


@lru_cache(maxsize=1)
def cross_encoder():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=config.RERANK_MODEL)


def embed_dense(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in dense_model().embed(texts)]


def embed_dense_query(text: str) -> list[float]:
    return next(iter(dense_model().query_embed(text))).tolist()


def embed_sparse(texts: list[str]):
    return list(sparse_model().embed(texts))


def embed_sparse_query(text: str):
    return next(iter(sparse_model().query_embed(text)))


def warm_up() -> None:
    """Force model download/initialisation ahead of the first request."""
    embed_dense_query("warm up")
    embed_sparse_query("warm up")
    list(cross_encoder().rerank("warm up", ["warm up"]))
