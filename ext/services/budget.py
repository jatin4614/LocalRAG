"""Token-budget the reranked chunks — drop from lowest rank end until we fit."""
from __future__ import annotations

import logging
from typing import List

from .chunker import _encoder
from .vector_store import Hit


logger = logging.getLogger("rag.budget")


def _count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def budget_chunks(hits: List[Hit], *, max_tokens: int = 4000) -> List[Hit]:
    """Assumes hits is pre-sorted best-first. Returns longest prefix that fits."""
    kept: list[Hit] = []
    total = 0
    dropped = 0
    for h in hits:
        t = _count_tokens(str(h.payload.get("text", "")))
        if total + t > max_tokens:
            dropped += 1
            continue
        total += t
        kept.append(h)
    if dropped:
        logger.debug("budget dropped %d of %d chunks (used %d/%d tokens)",
                     dropped, len(hits), total, max_tokens)
    return kept
