"""Rerank hits from parallel KB searches.

Strategy:
- Fast path: if raw top-1 score / top-2 score > FAST_PATH_RATIO, return input unchanged.
- Otherwise: per-KB max-normalize scores, then re-sort global list descending.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List

from .vector_store import Hit


FAST_PATH_RATIO = 2.0


def rerank(hits: List[Hit], *, top_k: int = 10) -> List[Hit]:
    if not hits:
        return []

    ordered = sorted(hits, key=lambda h: h.score, reverse=True)
    if len(ordered) >= 2 and ordered[1].score > 0 and (ordered[0].score / ordered[1].score) > FAST_PATH_RATIO:
        return ordered[:top_k]

    max_by_kb: dict[int, float] = defaultdict(float)
    for h in hits:
        kb = int(h.payload.get("kb_id", -1))
        if h.score > max_by_kb[kb]:
            max_by_kb[kb] = h.score

    def normalized(h: Hit) -> float:
        kb = int(h.payload.get("kb_id", -1))
        m = max_by_kb[kb]
        return h.score / m if m > 0 else h.score

    # Sort by: normalized score desc, then raw score desc, then id (stable tiebreaker for determinism in tests)
    ordered2 = sorted(hits, key=lambda h: (-normalized(h), -h.score, str(h.id)))
    return ordered2[:top_k]
