"""Parallel multi-KB + optional chat-private retrieval."""
from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Sequence

from .embedder import Embedder
from .vector_store import CHAT_PRIVATE_COLLECTION, Hit, VectorStore


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
    owner_user_id: Optional[int | str] = None,
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

    P2.2: ``owner_user_id`` is forwarded ONLY to chat-scoped namespace searches
    (``chat_private`` + legacy ``chat_{chat_id}``) — KB collections stay shared
    across all users with access. When None (default) no owner filter is
    applied anywhere, making the default path byte-identical to pre-P2.2.
    This is the per-user isolation invariant for private chat docs: within
    the consolidated ``chat_private`` collection, the filter prevents any
    cross-user bleed.

    P2.3: chat-private reads dual-target BOTH ``chat_private`` (new primary,
    tenant-filtered by ``chat_id`` + ``owner_user_id``) AND the legacy
    ``chat_{chat_id}`` collection (fallback for un-migrated data). Results
    are merged and deduplicated by point id so callers see a flat stream.
    Dual-read is idempotent — once a chat is migrated, the legacy
    collection is empty and contributes nothing.

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

    async def _search_one(
        collection: str,
        subtag_ids: Optional[list[int]] = None,
        *,
        owner_filter: Optional[int | str] = None,
        chat_filter: Optional[int | str] = None,
    ) -> List[Hit]:
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
                    owner_user_id=owner_filter,
                    chat_id=chat_filter,
                )
            return await vector_store.search(
                collection, qvec, limit=per_kb_limit, subtag_ids=subtag_ids,
                owner_user_id=owner_filter,
                chat_id=chat_filter,
            )
        except Exception:
            return []

    async def _search_kb(cfg: dict) -> List[Hit]:
        kb_id = cfg["kb_id"]
        subtag_ids = cfg.get("subtag_ids") or None
        # KB collections are shared — do NOT filter by owner_user_id.
        return await _search_one(f"kb_{kb_id}", subtag_ids=subtag_ids)

    async def _search_chat() -> List[Hit]:
        """Read private chat docs from both chat_private (primary) AND the
        legacy ``chat_{chat_id}`` collection (fallback).

        Migration story (P2.3):
          * New uploads land in ``chat_private``, tenant-filtered by
            ``chat_id`` + ``owner_user_id``.
          * Pre-existing chats still have their data in ``chat_{chat_id}``
            collections — we keep reading from those until an operator
            runs ``scripts/migrate_chat_collections.py --apply``.
          * Dual-read is idempotent: once a chat is migrated, the legacy
            collection is empty (or deleted) and ``_search_one`` returns
            an empty list. Merged+deduped result set is unchanged.
        """
        if chat_id is None:
            return []
        # Primary — consolidated collection, filtered by both chat_id and owner.
        # Both filters are load-bearing: chat_id scopes the tenant, owner enforces
        # per-user isolation (two users in the same chat would be a concurrency
        # bug, but defense-in-depth is cheap with tenant indexes).
        primary = await _search_one(
            CHAT_PRIVATE_COLLECTION,
            owner_filter=owner_user_id,
            chat_filter=chat_id,
        )
        # Legacy fallback — per-chat collection (pre-P2.3 shape). Missing
        # collection surfaces as an exception inside vs.search; _search_one
        # catches and returns []. Owner filter still applies even though the
        # collection is already chat-scoped (defense-in-depth; cheap).
        legacy = await _search_one(
            f"chat_{chat_id}",
            owner_filter=owner_user_id,
        )
        # Merge + dedupe by point id. Keep the higher-scoring copy when the
        # same id appears in both (shouldn't happen if migration ran cleanly,
        # but is the right tie-break if it hasn't).
        merged: dict = {}
        for hit in list(primary) + list(legacy):
            prev = merged.get(hit.id)
            if prev is None or hit.score > prev.score:
                merged[hit.id] = hit
        return list(merged.values())

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
