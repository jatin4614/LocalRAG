"""Pure-stdlib scoring functions for the RAG eval harness (P0.1).

Every function is deterministic and free of side effects. ``k`` always means
"look at the first k items of the retrieved list". IDs are compared by equality
after the caller has normalised types (chunk IDs are typically strings/UUIDs,
doc IDs are ints in the live Qdrant payloads but may be stored as strings —
callers should normalise before passing in).
"""
from __future__ import annotations

from typing import Sequence


def chunk_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: set[str],
    k: int,
) -> float:
    """Fraction of gold chunk IDs that appear in the top-k retrieved chunk IDs.

    Returns 0.0 if ``gold_chunk_ids`` is empty (caller should avoid that).
    """
    if not gold_chunk_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top = set(retrieved_chunk_ids[:k])
    hits = sum(1 for gid in gold_chunk_ids if gid in top)
    return hits / len(gold_chunk_ids)


def doc_recall_at_k(
    retrieved_doc_ids: Sequence[int],
    gold_doc_ids: set[int],
    k: int,
) -> float:
    """Fraction of gold doc IDs present somewhere in the top-k retrieved doc IDs."""
    if not gold_doc_ids:
        return 0.0
    if k <= 0:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    hits = sum(1 for did in gold_doc_ids if did in top)
    return hits / len(gold_doc_ids)


def mrr_at_k(
    retrieved_doc_ids: Sequence[int],
    gold_doc_ids: set[int],
    k: int,
) -> float:
    """Reciprocal rank of the first gold doc within top-k. 0.0 if none found.

    Rank is 1-indexed: if the first retrieved doc is a gold doc, returns 1.0.
    """
    if not gold_doc_ids or k <= 0:
        return 0.0
    for idx, did in enumerate(retrieved_doc_ids[:k]):
        if did in gold_doc_ids:
            return 1.0 / (idx + 1)
    return 0.0


def unique_docs_at_k(retrieved_doc_ids: Sequence[int], k: int) -> int:
    """Count of unique doc IDs in the top-k retrieved list. Diversity signal."""
    if k <= 0:
        return 0
    return len(set(retrieved_doc_ids[:k]))
