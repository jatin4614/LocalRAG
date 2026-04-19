"""Parallel multi-KB + optional chat-private retrieval."""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Sequence

from .embedder import Embedder
from .vector_store import Hit, VectorStore


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID at call time (not import time) so tests can toggle it.

    Default on as of 2026-04-19 — eval showed +12pp chunk_recall at +3ms.
    Set RAG_HYBRID=0 to force dense-only. Any non-"0" value (including empty,
    "yes", "true") means "on" — more forgiving and matches user intent.
    """
    return os.environ.get("RAG_HYBRID", "1") != "0"


async def retrieve(
    *,
    query: str,
    selected_kbs: Sequence[dict],
    chat_id: Optional[int],
    vector_store: VectorStore,
    embedder: Embedder,
    per_kb_limit: int = 10,
    total_limit: int = 30,
) -> List[Hit]:
    """Run parallel searches against each selected KB and an optional chat namespace.

    selected_kbs shape: [{"kb_id": int, "subtag_ids": [int, ...]}, ...]
        empty subtag_ids → search all subtags in that KB.
    Returns a flat list of Hit objects sorted by raw score descending, trimmed to total_limit.

    Hybrid retrieval is on by default (RAG_HYBRID unset → on). When hybrid is
    enabled AND a given collection was created with sparse-vector support,
    retrieval uses hybrid RRF fusion. Any collection that lacks sparse support
    silently falls back to dense-only search (preserves backward compatibility
    with legacy collections). Set RAG_HYBRID=0 to force dense-only globally.

    Semantic cache (P2.6) is gated behind RAG_SEMCACHE=1. Default OFF — the
    module is only imported when the flag is on, so the default path has zero
    cost and no behavior change. On hit, we skip Qdrant entirely and return
    the cached Hit list. Keyed by pipeline_version + KB selection + quantized
    query vector (so near-identical queries share a cache entry).
    """
    [qvec] = await embedder.embed([query])
    hybrid = _hybrid_enabled()

    # P2.6: semantic retrieval cache — lookup BEFORE any Qdrant call.
    # Lazy import keeps default-off path zero-cost (module never loads).
    if os.environ.get("RAG_SEMCACHE", "0") == "1":
        from ext.services.retrieval_cache import (
            get as _cache_get,
            is_enabled as _semcache_on,
        )
        if _semcache_on():
            cached = _cache_get(qvec, list(selected_kbs), chat_id)
            if cached is not None:
                return [
                    Hit(id=c["id"], score=c["score"], payload=c["payload"])
                    for c in cached
                ][:total_limit]

    async def _search_one(collection: str, subtag_ids: Optional[list[int]] = None) -> List[Hit]:
        use_hybrid = False
        if hybrid:
            # Warm the sparse-support cache for this collection; on failure
            # (e.g., collection doesn't exist) _refresh_sparse_cache returns
            # False and we fall back to dense-only → exception surfaces from
            # vs.search below. getattr guards against minimal vector-store
            # substitutes in tests that don't implement the helper.
            refresh = getattr(vector_store, "_refresh_sparse_cache", None)
            if refresh is not None:
                try:
                    use_hybrid = await refresh(collection)
                except Exception:
                    use_hybrid = False
        try:
            if use_hybrid:
                return await vector_store.hybrid_search(
                    collection, qvec, query,
                    limit=per_kb_limit, subtag_ids=subtag_ids,
                )
            return await vector_store.search(
                collection, qvec, limit=per_kb_limit, subtag_ids=subtag_ids,
            )
        except Exception:
            return []

    async def _search_kb(cfg: dict) -> List[Hit]:
        kb_id = cfg["kb_id"]
        subtag_ids = cfg.get("subtag_ids") or None
        return await _search_one(f"kb_{kb_id}", subtag_ids=subtag_ids)

    async def _search_chat() -> List[Hit]:
        if chat_id is None:
            return []
        return await _search_one(f"chat_{chat_id}")

    tasks = [_search_kb(cfg) for cfg in selected_kbs]
    tasks.append(_search_chat())
    results = await asyncio.gather(*tasks)

    flat: list[Hit] = []
    for lst in results:
        flat.extend(lst)
    flat.sort(key=lambda h: h.score, reverse=True)
    trimmed = flat[:total_limit]

    # P2.6: write cache entry after successful Qdrant fan-out.
    # Same env-gate as above so the module stays unloaded in the default path.
    if os.environ.get("RAG_SEMCACHE", "0") == "1":
        from ext.services.retrieval_cache import (
            put as _cache_put,
            is_enabled as _semcache_on,
        )
        if _semcache_on():
            _cache_put(qvec, list(selected_kbs), chat_id, trimmed)

    return trimmed
