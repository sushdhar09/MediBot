"""Dense-only vs hybrid vs hybrid+rerank, side by side.

Run this to see where BM25 and the cross-encoder actually earn their keep -
exact medical terminology, drug names, fault codes and ICD codes.

    python -m scripts.compare_retrieval
"""
from __future__ import annotations

from app import config
from app.retrieval import store
from app.retrieval.rerank import rerank

QUERIES = [
    ("doctor", "What is the correct dosage of Amoxicillin for adults?"),
    ("nurse", "IV cannula size for a paediatric patient under 5kg"),
    ("technician", "fault code F-05 on the infusion pump"),
    ("billing_executive", "ICD code I21.4 claim submission requirement"),
    ("doctor", "sepsis management protocol first hour"),
    ("nurse", "hand hygiene five moments"),
]


def show(title: str, chunks, score_attr: str) -> None:
    print(f"  {title}")
    if not chunks:
        print("    (nothing)")
        return
    for position, chunk in enumerate(chunks, start=1):
        score = getattr(chunk, score_attr) or 0.0
        print(
            f"    {position}. {score:>8.4f}  {chunk.source_document} :: "
            f"{chunk.section_title[:60]} [{chunk.chunk_type}]"
        )


def main() -> None:
    for role, query in QUERIES:
        print(f"\n=== [{role}] {query}")
        dense = store.dense_only_search(query, role, limit=5)
        hybrid = store.hybrid_search(query, role, limit=config.HYBRID_CANDIDATES)
        show("dense-only (top 5)", dense, "fusion_score")
        show("hybrid RRF (top 5)", hybrid[:5], "fusion_score")
        show(
            f"after cross-encoder rerank (top {config.RERANK_TOP_K})",
            rerank(query, hybrid),
            "rerank_score",
        )


if __name__ == "__main__":
    main()
