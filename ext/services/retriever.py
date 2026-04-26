"""Parallel multi-KB + optional chat-private retrieval."""
from __future__ import annotations

import asyncio
import os
from time import perf_counter
from typing import Any, List, Optional, Sequence

from . import flags
from .embedder import Embedder
from .obs import span
from .vector_store import CHAT_PRIVATE_COLLECTION, Hit, VectorStore


# RRF (Reciprocal Rank Fusion) constant. The canonical value 60 from the
# original RRF paper (Cormack et al. 2009) — large enough that the
# contribution of rank-N items decays gradually rather than collapsing
# everything onto rank 0. Tuning this is rarely worthwhile.
RRF_K = 60


def rrf_fuse_heads(
    heads: list[list[tuple[str, int]]],
    *,
    k: int = 60,
    top_k: int,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple retrieval heads.

    Each head is a list of ``(doc_id, rank)`` tuples (rank is 0-indexed
    within that head). The fused score for a doc is::

        score(doc) = Σ_heads 1 / (k + rank + 1)

    Missing from a head → no contribution from that head (this is what
    makes RRF degrade gracefully when one arm — e.g. ColBERT on a legacy
    collection — returns empty: the fusion still ranks the remaining
    arms correctly).

    Used by the per-KB search path (Phase 3.5 tri-fusion) to combine
    dense + sparse + ColBERT result lists when ``RAG_COLBERT=1`` AND the
    collection has the named ``colbert`` vector slot. Pure function — no
    Qdrant or model dependencies, fully unit-testable.
    """
    scores: dict[str, float] = {}
    for head in heads:
        for doc_id, rank in head:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def _field(h: Any, key: str, default: Any = None) -> Any:
    """Read a field from either a plain dict (test fixtures) or a ``Hit``
    dataclass (production). For ``Hit`` objects, ``score`` and ``id`` are
    top-level attributes while everything else lives in ``payload``.
    """
    if isinstance(h, dict):
        return h.get(key, default)
    if key in ("score", "id"):
        return getattr(h, key, default)
    payload = getattr(h, "payload", None) or {}
    return payload.get(key, default)


def merge_kb_results(
    per_kb: dict,
    *,
    rerank_enabled: bool,
    top_k: int,
) -> list:
    """Combine per-KB hit lists into one sorted list.

    With rerank ON: simple global sort by score. The cross-encoder will
    re-score everything against the same query later anyway, so absolute
    Qdrant scores at this stage are just a coarse pre-filter — order
    doesn't have to be perfect across collections.

    With rerank OFF: RRF (Reciprocal Rank Fusion) by within-KB rank, so
    a chatty KB whose Qdrant scores are systematically higher doesn't
    dominate the result set. Each hit gets ``Σ 1/(60 + rank_in_kb + 1)``
    summed across every KB it appears in (typically just one — the same
    chunk lives in exactly one collection — but the formulation is
    rank-fusion-canonical).

    Accepts either plain dicts (test fixtures shape:
    ``{"kb_id", "doc_id", "chunk_index", "score"}``) or ``Hit`` dataclass
    instances; ``_field`` handles the difference. Output type matches
    input type — production callers continue to receive ``Hit`` objects.
    """
    if rerank_enabled:
        flat = [h for hits in per_kb.values() for h in hits]
        flat.sort(key=lambda h: _field(h, "score", 0.0), reverse=True)
        return flat[:top_k]
    # RRF — rank-based fusion across KBs.
    #
    # The dedup key is ``(kb_id, doc_id, chunk_index)`` when those payload
    # fields are present (canonical chunk identity, lets the same chunk in
    # two collections share an RRF bucket — rare but theoretically right);
    # otherwise we fall back to the hit's own id. Without the fallback,
    # production hits with sparse payloads (e.g. chat-private docs that
    # don't carry kb_id) would all collapse to the (None, None, None) key
    # and overwrite each other.
    rrf_scores: dict = {}
    hit_by_key: dict = {}
    for _kb_id, hits in per_kb.items():
        for rank, h in enumerate(hits):
            payload_key = (
                _field(h, "kb_id"),
                _field(h, "doc_id"),
                _field(h, "chunk_index"),
            )
            key = payload_key if any(v is not None for v in payload_key) else _field(h, "id")
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            hit_by_key[key] = h
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [hit_by_key[k] for k in sorted_keys[:top_k]]


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID at call time (not import time) so tests can toggle it.

    Default on as of 2026-04-19 — eval showed +12pp chunk_recall at +3ms.
    Set RAG_HYBRID=0 to force dense-only. Any non-"0" value (including empty,
    "yes", "true") means "on" — more forgiving and matches user intent.

    Reads through ``flags.get`` so per-request KB-config overrides take
    precedence over the process-level env var (P3.0).
    """
    return flags.get("RAG_HYBRID", "1") != "0"


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
    level_filter: Optional[str] = None,
    doc_ids: Optional[list[int]] = None,
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
    # P3.3: optional HyDE (Hypothetical Document Embeddings).
    # When RAG_HYDE=1 we generate N synthetic "excerpt" answers via the
    # chat model, embed them, and average. The resulting vector matches
    # real document chunks (written as declarative statements) better
    # than the raw question's embedding does — a big win on abstract
    # queries over long KBs. Fail-open: on any chat error hyde_embed
    # returns None, and we fall through to the raw-query embed.
    #
    # Default path (flag unset) is byte-identical to pre-P3.3: the
    # ``ext.services.hyde`` module is not imported at all, so there is
    # zero cost on the default path.
    if flags.get("RAG_HYDE", "0") == "1":
        from ext.services.hyde import hyde_embed
        n = int(flags.get("RAG_HYDE_N", "1") or "1")
        hyde_vec = await hyde_embed(
            query,
            embedder,
            n=n,
            chat_url=os.environ.get("OPENAI_API_BASE_URL", "http://vllm-chat:8000/v1"),
            chat_model=os.environ.get("HYDE_MODEL", os.environ.get("CHAT_MODEL", "orgchat-chat")),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        if hyde_vec is not None:
            qvec = hyde_vec
        else:
            [qvec] = await embedder.embed([query])
    else:
        [qvec] = await embedder.embed([query])
    hybrid = _hybrid_enabled()

    # P2.6: semantic retrieval cache — lookup BEFORE any Qdrant call.
    # Lazy import keeps default-off path zero-cost (module never loads).
    # flags.get honors request-scoped KB-config overrides (P3.0).
    if flags.get("RAG_SEMCACHE", "0") == "1":
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
        _filter_summary = []
        if subtag_ids:
            _filter_summary.append(f"subtag_ids={list(subtag_ids)}")
        if owner_filter is not None:
            _filter_summary.append(f"owner={owner_filter}")
        if chat_filter is not None:
            _filter_summary.append(f"chat={chat_filter}")
        _t_qd = perf_counter()
        try:
            with span(
                "qdrant.search",
                collection=collection,
                top_k=per_kb_limit,
                filter=",".join(_filter_summary) or "none",
                hybrid=use_hybrid,
            ):
                if use_hybrid:
                    return await vector_store.hybrid_search(
                        collection, qvec, query,
                        limit=per_kb_limit, subtag_ids=subtag_ids,
                        doc_ids=doc_ids,
                        owner_user_id=owner_filter,
                        chat_id=chat_filter,
                        level=level_filter,
                    )
                return await vector_store.search(
                    collection, qvec, limit=per_kb_limit, subtag_ids=subtag_ids,
                    doc_ids=doc_ids,
                    owner_user_id=owner_filter,
                    chat_id=chat_filter,
                    level=level_filter,
                )
        except Exception:
            return []
        finally:
            try:
                from .metrics import qdrant_search_latency_seconds
                qdrant_search_latency_seconds.labels(collection=collection).observe(
                    perf_counter() - _t_qd
                )
            except Exception:
                pass

    async def _search_kb(cfg: dict) -> List[Hit]:
        kb_id = cfg["kb_id"]
        subtag_ids = cfg.get("subtag_ids") or None
        # KB collections are shared — do NOT filter by owner_user_id.
        hits = await _search_one(f"kb_{kb_id}", subtag_ids=subtag_ids)
        # Tier 1/2: when level_filter is set (e.g. "doc" for global-intent
        # queries) keep only points whose payload carries that level tag.
        # Post-filter in Python so the default path (level_filter=None)
        # remains byte-identical to pre-Tier-2 behaviour — no extra Qdrant
        # filter clause, no extra round-trip. Points predating the
        # doc-summary feature have no "level" field and are treated as
        # chunks (level=="chunk" by convention).
        if level_filter is not None:
            hits = [
                h for h in hits
                if (h.payload.get("level") or "chunk") == level_filter
            ]
        return hits

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

    # Build per-bucket map for ``merge_kb_results``. RRF treats every bucket
    # symmetrically so we key by index, not kb_id — that also means duplicate
    # kb_ids in ``selected_kbs`` (rare, but possible) keep their hits in
    # separate buckets instead of silently overwriting. The trailing entry
    # is the chat-private bucket. With rerank ON the helper just does a
    # global sort by score, byte-equivalent to the previous inlined merge.
    per_kb: dict[int, list[Hit]] = {i: list(lst) for i, lst in enumerate(results)}
    rerank_enabled = flags.get("RAG_RERANK", "0") == "1"
    trimmed = merge_kb_results(
        per_kb,
        rerank_enabled=rerank_enabled,
        top_k=total_limit,
    )

    # P2.6: write cache entry after successful Qdrant fan-out.
    # Same env-gate as above so the module stays unloaded in the default path.
    if flags.get("RAG_SEMCACHE", "0") == "1":
        from ext.services.retrieval_cache import (
            put as _cache_put,
            is_enabled as _semcache_on,
        )
        if _semcache_on():
            _cache_put(qvec, list(selected_kbs), chat_id, trimmed)

    return trimmed


# ----------------------------------------------------------------------
# Plan B Phase 5.6 — temporal-aware level injection (single-collection helpers)
# ----------------------------------------------------------------------
# These helpers operate on ONE collection and use plain dict hits (not Hit
# instances) so they're trivially testable. They live alongside the legacy
# ``retrieve()`` fan-out fn but don't replace it — operators can opt in
# per-call via the chat bridge (Phase 5.6 step 4 wiring).

# Intent → level injection rules (Plan B Phase 5.6).
_INTENT_LEVEL_INJECTION: dict[str, list[int]] = {
    "global": [3, 4],       # yearly + 3-year meta
    "evolution": [2, 3],    # quarterly + yearly
    "specific_date": [],    # filter by shard_key, no level injection
    "specific": [],
    "metadata": [],
}


# Module-level VectorStore singleton for Phase 5.6 helpers. Initialized on
# first call from real production code. Tests monkeypatch
# ``_get_vector_store_singleton`` or pre-set the variable directly.
_vs_singleton: Optional[VectorStore] = None


def _get_vector_store_singleton() -> VectorStore:
    """Return / lazily construct the module-level VectorStore.

    Production callers (chat_rag_bridge) typically inject their own VS
    instance via the legacy ``retrieve()`` fan-out — these helpers exist
    for the temporal level-injection code path and unit tests that want
    a no-network singleton.
    """
    global _vs_singleton
    if _vs_singleton is None:
        _vs_singleton = VectorStore(
            url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            vector_size=int(os.environ.get("RAG_DENSE_DIM", "1024")),
        )
    return _vs_singleton


async def _dense_search(
    *,
    collection: str,
    query_vec: list[float],
    top_k: int,
    qdrant_filter=None,
    **_kwargs,
) -> list[dict]:
    """Run a single dense search; return list of dict-shaped hits.

    Tests monkeypatch this directly. Production wiring (Phase 5.6 step 4)
    constructs a Hit-shaped result from the live ``VectorStore.search``
    and adapts it to dicts for downstream merging.
    """
    vs = _get_vector_store_singleton()
    hits = await vs.search(
        collection,
        query_vec,
        limit=top_k,
    )
    return [
        {"id": h.id, "score": h.score, "payload": dict(h.payload or {})}
        for h in hits
    ]


async def _fetch_temporal_levels(
    collection: str, levels: list[int], top_k: int = 1,
) -> dict[int, list[dict]]:
    """Fetch top_k summary nodes per level from a temporal collection.

    Returns ``{level: [hit, ...]}``. Each hit is the same dict shape as
    a normal retrieval result. Implements the Plan B Phase 5.6 contract
    of "guarantee-include the L3 / L4 nodes when intent demands them".
    """
    from qdrant_client.http.models import (
        FieldCondition, Filter, MatchValue,
    )

    vs = _get_vector_store_singleton()
    out: dict[int, list[dict]] = {}
    for level in levels:
        f = Filter(must=[
            FieldCondition(key="level", match=MatchValue(value=level)),
        ])
        try:
            points, _ = await vs._client.scroll(
                collection_name=collection,
                limit=top_k,
                scroll_filter=f,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            out[level] = []
            continue
        out[level] = [
            {"id": str(p.id), "score": 1.0, "payload": p.payload or {}}
            for p in points
        ]
    return out


def _filter_by_temporal_constraint(constraint: Optional[dict]):
    """Build a Qdrant filter that narrows to matching shard_keys.

    Plan B Phase 5.6.
    """
    if not constraint:
        return None
    from qdrant_client.http.models import (
        FieldCondition, Filter, MatchAny, MatchValue,
    )
    year = constraint.get("year")
    month = constraint.get("month")
    quarter = constraint.get("quarter")

    if month and year:
        sk = f"{year:04d}-{month:02d}"
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchValue(value=sk)),
        ])
    if quarter and year:
        first_month = (quarter - 1) * 3 + 1
        sks = [f"{year:04d}-{first_month + i:02d}" for i in range(3)]
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchAny(any=sks)),
        ])
    if year:
        sks = [f"{year:04d}-{m:02d}" for m in range(1, 13)]
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchAny(any=sks)),
        ])
    return None


async def retrieve_for_kb(
    *,
    collection: str,
    query: str,
    query_vec: list[float],
    top_k: int,
    intent_hint: str = "specific",
    temporal_constraint: Optional[dict] = None,
    **kwargs,
) -> list[dict]:
    """Single-collection retrieval with temporal-aware level injection.

    Plan B Phase 5.6. Behavior:

      * ``RAG_TEMPORAL_LEVELS=0`` (default): byte-equivalent to a plain
        dense search — no shard_key filter, no level injection.
      * ``intent_hint=global``: inject L3 + L4 summary nodes ahead of the
        dense candidates so they survive top-K trimming.
      * ``intent_hint=evolution``: inject L2 + L3 summary nodes.
      * ``intent_hint=specific_date``: apply a shard_key filter derived
        from ``temporal_constraint`` (no level injection).
      * Other intents: pass-through dense search.

    Plan B Phase 5.7 wires time-decay scoring on top of the returned
    base hits.
    """
    temporal_enabled = os.environ.get("RAG_TEMPORAL_LEVELS", "0") == "1"

    qdrant_filter = None
    if temporal_enabled and intent_hint == "specific_date":
        qdrant_filter = _filter_by_temporal_constraint(temporal_constraint)

    base_hits = await _dense_search(
        collection=collection,
        query_vec=query_vec,
        top_k=top_k,
        qdrant_filter=qdrant_filter,
        **kwargs,
    )

    if not temporal_enabled:
        return base_hits

    # Plan B Phase 5.7 — intent-conditional time-decay scoring.
    if os.environ.get("RAG_TIME_DECAY", "0") == "1":
        try:
            from .time_decay import (
                apply_time_decay_to_hits, should_apply_time_decay,
            )
            if should_apply_time_decay(
                query=query,
                intent=intent_hint,
                temporal_constraint=temporal_constraint,
            ):
                apply_time_decay_to_hits(base_hits)
        except Exception:  # noqa: BLE001 — fail-open, don't block retrieval
            pass

    levels_to_fetch = _INTENT_LEVEL_INJECTION.get(intent_hint, [])
    if levels_to_fetch:
        levels_hits = await _fetch_temporal_levels(
            collection=collection,
            levels=levels_to_fetch,
            top_k=2,
        )
        injected: list[dict] = []
        for level in sorted(levels_to_fetch):
            injected.extend(levels_hits.get(level, []))
        return injected + base_hits

    return base_hits
