"""Ingestion entrypoint: parse -> chunk -> embed (dense + BM25) -> index.

    python -m app.ingestion.ingest              # full run
    python -m app.ingestion.ingest --reuse-cache  # re-index without re-parsing

Docling downloads its layout models on the first run, so run this once as a
standalone script before demoing.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter

from .. import config
from ..retrieval import store
from .chunker import build_chunks, load_chunks, save_chunks

log = logging.getLogger("ingest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MediBot vector index")
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help=f"re-index from {config.CHUNK_CACHE_PATH} instead of re-parsing the PDFs",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    started = time.time()
    if args.reuse_cache and config.CHUNK_CACHE_PATH.exists():
        chunks = load_chunks()
        log.info("Loaded %d cached chunks", len(chunks))
    else:
        chunks = build_chunks()
        save_chunks(chunks)
        log.info("Parsed %d chunks in %.1fs", len(chunks), time.time() - started)

    if not chunks:
        raise SystemExit(f"No documents found under {config.DATA_DIR}")

    store.recreate_collection()
    indexed = store.index_chunks(chunks)
    store.close_client()

    by_collection = Counter(c.collection for c in chunks)
    by_type = Counter(c.chunk_type for c in chunks)
    log.info("Indexed %d chunks in %.1fs", indexed, time.time() - started)
    log.info("By collection: %s", dict(by_collection))
    log.info("By chunk_type: %s", dict(by_type))


if __name__ == "__main__":
    main()
