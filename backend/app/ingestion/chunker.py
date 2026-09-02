"""Structure-aware parsing (Docling) + hierarchical chunking (HybridChunker).

Why not fixed-size chunking: a drug dosage table split across two chunks is
meaningless, and "25mg twice daily" without its parent heading is useless to
both the embedding model and the LLM. Docling gives us a structured document
(headings, tables, code, lists); HybridChunker splits along that structure
first and only then applies a token-aware size limit.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config
from ..rbac import COLLECTION_ACCESS, roles_for_collection

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".markdown"}

# Docling item labels mapped onto the assignment's chunk_type vocabulary.
_TABLE_LABELS = {"table", "document_index"}
_CODE_LABELS = {"code", "formula"}
_HEADING_LABELS = {"title", "section_header", "page_header", "subtitle-level-1"}


@dataclass
class Chunk:
    """One indexable unit: contextualised text plus the full metadata schema."""

    text: str  # what actually gets embedded (heading context + body)
    raw_text: str  # the body on its own, handy for debugging
    source_document: str
    collection: str
    access_roles: list[str]
    section_title: str
    chunk_type: str
    page_numbers: list[int] = field(default_factory=list)

    def to_payload(self) -> dict:
        return asdict(self)


def discover_documents(data_dir: Path | None = None) -> list[tuple[Path, str]]:
    """Return (file, collection) pairs. The folder name *is* the collection."""
    root = Path(data_dir or config.DATA_DIR)
    found: list[tuple[Path, str]] = []
    for collection in COLLECTION_ACCESS:
        folder = root / collection
        if not folder.is_dir():
            log.warning("Collection folder missing: %s", folder)
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append((path, collection))
    return found


def _build_chunker():
    """HybridChunker across docling versions (the tokenizer API moved in 2.x)."""
    from docling.chunking import HybridChunker

    try:
        from docling_core.transforms.chunker.tokenizer.huggingface import (
            HuggingFaceTokenizer,
        )
        from transformers import AutoTokenizer

        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(config.CHUNK_TOKENIZER),
            max_tokens=config.MAX_CHUNK_TOKENS,
        )
        return HybridChunker(tokenizer=tokenizer, merge_peers=True)
    except ImportError:  # older docling: tokenizer id + max_tokens directly
        return HybridChunker(
            tokenizer=config.CHUNK_TOKENIZER,
            max_tokens=config.MAX_CHUNK_TOKENS,
            merge_peers=True,
        )


def _labels_of(chunk) -> set[str]:
    labels: set[str] = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        label = getattr(item, "label", None)
        if label is not None:
            labels.add(str(getattr(label, "value", label)).lower())
    return labels


def _chunk_type(chunk) -> str:
    labels = _labels_of(chunk)
    if labels & _TABLE_LABELS:
        return "table"
    if labels & _CODE_LABELS:
        return "code"
    if labels and labels <= _HEADING_LABELS:
        return "heading"
    return "text"


def _page_numbers(chunk) -> list[int]:
    pages: set[int] = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            page = getattr(prov, "page_no", None)
            if page is not None:
                pages.add(int(page))
    return sorted(pages)


def chunk_document(path: Path, collection: str, converter=None, chunker=None) -> list[Chunk]:
    from docling.document_converter import DocumentConverter

    converter = converter or DocumentConverter()
    chunker = chunker or _build_chunker()

    document = converter.convert(str(path)).document
    chunks: list[Chunk] = []
    for raw_chunk in chunker.chunk(dl_doc=document):
        # contextualize() prefixes the chunk with its heading hierarchy, so the
        # embedded text always carries its parent section as context.
        contextualised = chunker.contextualize(chunk=raw_chunk).strip()
        body = (raw_chunk.text or "").strip()
        if not contextualised:
            continue

        headings = list(getattr(raw_chunk.meta, "headings", None) or [])
        chunks.append(
            Chunk(
                text=contextualised,
                raw_text=body,
                source_document=path.name,
                collection=collection,
                access_roles=roles_for_collection(collection),
                section_title=" > ".join(headings) if headings else path.stem.replace("_", " ").title(),
                chunk_type=_chunk_type(raw_chunk),
                page_numbers=_page_numbers(raw_chunk),
            )
        )
    return chunks


def build_chunks(data_dir: Path | None = None) -> list[Chunk]:
    """Parse and chunk every document in the dataset."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    chunker = _build_chunker()

    all_chunks: list[Chunk] = []
    for path, collection in discover_documents(data_dir):
        log.info("Parsing %s (%s)", path.name, collection)
        chunks = chunk_document(path, collection, converter=converter, chunker=chunker)
        log.info("  -> %d chunks", len(chunks))
        all_chunks.extend(chunks)
    return all_chunks


def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    target = Path(path or config.CHUNK_CACHE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_payload(), ensure_ascii=False) + "\n")
    return target


def load_chunks(path: Path | None = None) -> list[Chunk]:
    source = Path(path or config.CHUNK_CACHE_PATH)
    with source.open(encoding="utf-8") as handle:
        return [Chunk(**json.loads(line)) for line in handle if line.strip()]
