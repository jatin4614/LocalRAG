"""Token-budget the reranked chunks — drop from lowest rank end until we fit."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

from .vector_store import Hit


logger = logging.getLogger("rag.budget")


# Tokenizer registry: maps RAG_BUDGET_TOKENIZER alias -> backend spec.
#   kind="tiktoken" => use a tiktoken encoding by id
#   kind="hf"       => transformers.AutoTokenizer.from_pretrained(id)
#
# Adding a new chat-model family is a 2-line change: pick an alias, point at
# the HF repo. RAG_BUDGET_TOKENIZER_MODEL (if set) overrides the default id
# for the family aliases noted below.
_TOKENIZER_REGISTRY: dict[str, dict[str, str]] = {
    "cl100k": {"kind": "tiktoken", "id": "cl100k_base"},
    "qwen": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
    },
    "qwen2.5": {"kind": "hf", "id": "Qwen/Qwen2.5-14B-Instruct"},
    "gemma": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "google/gemma-4-31B-it"),
    },
    "gemma-4": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-4-31b": {"kind": "hf", "id": "google/gemma-4-31B-it"},
    "gemma-3": {"kind": "hf", "id": "google/gemma-3-27b-it"},
    "gemma-3-12b": {"kind": "hf", "id": "google/gemma-3-12b-it"},
    "llama": {
        "kind": "hf",
        "id": os.environ.get("RAG_BUDGET_TOKENIZER_MODEL", "meta-llama/Llama-3-8B-Instruct"),
    },
}


def _get_tokenizer_alias() -> str:
    return os.environ.get("RAG_BUDGET_TOKENIZER", "cl100k").lower()


@lru_cache(maxsize=1)
def _budget_tokenizer():
    """Return a callable (text -> int) that counts tokens.

    Alias is chosen via RAG_BUDGET_TOKENIZER (default: cl100k, matches the
    old behavior). Unknown alias logs a warning and falls back to cl100k.

    For HF-backed aliases (qwen / gemma / llama), the first call downloads
    tokenizer vocab. When ``RAG_BUDGET_TOKENIZER_MODEL`` is set, it overrides
    the default id for the family alias (qwen / gemma / llama). Exact-version
    aliases like 'qwen2.5', 'gemma-3', 'gemma-3-12b' always use their pinned id.
    """
    alias = _get_tokenizer_alias()
    spec = _TOKENIZER_REGISTRY.get(alias)

    if spec is None:
        logger.warning(
            "RAG_BUDGET_TOKENIZER=%r is not registered; falling back to cl100k. "
            "Known aliases: %s",
            alias,
            ", ".join(sorted(_TOKENIZER_REGISTRY)),
        )
        spec = _TOKENIZER_REGISTRY["cl100k"]

    kind = spec["kind"]
    ident = spec["id"]

    if kind == "hf":
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(ident)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load HF tokenizer %r for alias %r (%s); falling "
                "back to cl100k. Tip: gated repos (e.g. google/gemma-*) "
                "require HF_TOKEN with accepted license.",
                ident, alias, exc,
            )
            return _cl100k_counter()

        def _count(text: str) -> int:
            return len(tok.encode(text, add_special_tokens=False))
        return _count

    if kind == "tiktoken":
        return _cl100k_counter()

    logger.warning("unknown tokenizer kind %r; falling back to cl100k", kind)
    return _cl100k_counter()


def _cl100k_counter():
    """Shared cl100k counter — reuses the chunker's encoder so we get caching."""
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
