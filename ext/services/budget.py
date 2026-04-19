"""Token-budget the reranked chunks — drop from lowest rank end until we fit."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from .vector_store import Hit


logger = logging.getLogger("rag.budget")


_RAG_BUDGET_TOKENIZER = os.environ.get("RAG_BUDGET_TOKENIZER", "cl100k").lower()


@lru_cache(maxsize=1)
def _budget_tokenizer():
    """Return a callable (text -> int) that counts tokens.

    Default ('cl100k'): tiktoken.cl100k_base, matches the old behavior.
    Set RAG_BUDGET_TOKENIZER=qwen to use the actual chat-model tokenizer —
    slower first call (downloads vocab), but accurate budgeting for Qwen2.5.
    """
    if _RAG_BUDGET_TOKENIZER == "qwen":
        from transformers import AutoTokenizer
        tok_name = os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "Qwen/Qwen2.5-14B-Instruct")
        tok = AutoTokenizer.from_pretrained(tok_name)

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        return _count

    # Default: cl100k (status quo — same as chunker's _encoder)
    from ext.services.chunker import _encoder
    enc = _encoder()

    def _count(text: str) -> int:
        return len(enc.encode(text))
    return _count


def _count_tokens(text: str) -> int:
    return _budget_tokenizer()(text)


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
