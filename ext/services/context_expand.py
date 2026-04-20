"""Parent-document context expansion.

After rerank (+ optional MMR), for each top hit fetch the N chunks before
and after it within the same document -- the LLM sees coherent context
rather than isolated fragments. Flag-gated by RAG_CONTEXT_EXPAND.

Deduplicates by (doc_id, chunk_index). Preserves rank order. No-op for
hits lacking chunk_index (legacy payloads).

Collection inference
--------------------
Hits in this pipeline come from multiple collections (KB and private-chat)
and the bridge does not track which collection each hit originated in. The
payload carries ``kb_id`` for KB hits and ``chat_id`` for private-chat
hits, so we reconstruct the collection name the same way
``ext.services.retriever`` does:

* ``kb_{kb_id}`` when ``payload["kb_id"]`` is set
* ``chat_{chat_id}`` when ``payload["chat_id"]`` is set and ``kb_id`` is absent

When neither is available we skip expansion for that hit (fail-open, keep
original).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from qdrant_client.http import models as qm


@dataclass
class ExpandedHit:
    """Hit-shaped wrapper exposing the same fields as VectorStore.Hit.

    ``score`` is set to 0.0 for fetched sibling chunks (they were not scored
    against the query). The bridge only reads ``payload`` downstream so the
    low score is harmless.
    """
    id: Any
    score: float
    payload: dict


def _infer_collection(payload: dict) -> Optional[str]:
    """Return ``kb_{kb_id}`` or ``chat_{chat_id}`` from the payload, or None."""
    kb_id = payload.get("kb_id")
    if kb_id is not None:
        return f"kb_{kb_id}"
    chat_id = payload.get("chat_id")
    if chat_id is not None:
        return f"chat_{chat_id}"
    return None


async def _fetch_neighbors(
    vs: Any,
    collection: str,
    doc_id: Any,
    chat_id: Any,
    center_idx: int,
    window: int,
) -> list[ExpandedHit]:
    """Fetch chunks with chunk_index in [center-window, center+window].

    Matches the same doc scope (doc_id for KB, chat_id for private-chat).
    Honors the ``deleted=False`` filter so soft-deleted siblings are not
    re-surfaced. Returns hits sorted ascending by chunk_index.
    """
    lo, hi = center_idx - window, center_idx + window
    must: list[qm.FieldCondition] = [
        qm.FieldCondition(
            key="chunk_index",
            range=qm.Range(gte=lo, lte=hi),
        ),
    ]
    # Scope filter: KB docs use doc_id; private-chat docs don't have doc_id
    # so scope by chat_id instead. The caller passed whichever is applicable.
    if doc_id is not None:
        must.append(
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        )
    elif chat_id is not None:
        must.append(
            qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))
        )
    else:
        # Nothing to scope on — refuse to scroll the whole collection.
        return []

    must_not = [
        qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True))
    ]

    try:
        res, _ = await vs._client.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(must=must, must_not=must_not),
            limit=2 * window + 1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        return []

    # Qdrant returns records in undefined order; re-sort by chunk_index asc so
    # the LLM sees siblings in natural document order.
    records = sorted(
        res,
        key=lambda r: (r.payload or {}).get("chunk_index", center_idx),
    )
    return [
        ExpandedHit(id=r.id, score=0.0, payload=r.payload or {})
        for r in records
    ]


async def expand_context(
    hits: Sequence[Any],
    *,
    vs: Any,
    window: int = 1,
) -> list[Any]:
    """Expand each hit with +/-window sibling chunks from the same doc.

    Dedupes by ``(doc_id_or_chat_id, chunk_index)``. Preserves rank order of
    the input hits; within each hit's expansion, chunks are ordered by
    ``chunk_index`` ascending. Legacy hits (no ``chunk_index``) pass through
    untouched. On fetch failure for a given hit, the original is kept.

    Args:
        hits: The reranked / MMR-diversified top hits.
        vs: The ``VectorStore`` instance (we use ``vs._client.scroll`` to
            fetch by payload filter — no query vector required).
        window: How many chunks before AND after each hit to fetch. ``0``
            short-circuits and returns the input unchanged.
    """
    if not hits or window <= 0:
        return list(hits)

    # Pass 1: plan the fetches. For each hit, derive (collection, scope_key,
    # chunk_index); enqueue a coroutine. Hits that can't be expanded (no
    # chunk_index, no collection) are recorded so pass 2 can keep them in
    # place.
    plans: list[tuple[int, Optional[asyncio.Future]]] = []
    tasks: list[asyncio.Future] = []
    # Per-hit dedupe key for centers: (scope_kind, scope_val, chunk_index)
    # where scope_kind is "doc" or "chat".
    center_keys: list[Optional[tuple[str, Any, int]]] = []

    for i, h in enumerate(hits):
        payload = getattr(h, "payload", {}) or {}
        chunk_idx = payload.get("chunk_index")
        collection = _infer_collection(payload)
        if chunk_idx is None or collection is None:
            plans.append((i, None))
            center_keys.append(None)
            continue

        doc_id = payload.get("doc_id")
        chat_id = payload.get("chat_id") if doc_id is None else None

        if doc_id is not None:
            center_keys.append(("doc", doc_id, int(chunk_idx)))
        elif chat_id is not None:
            center_keys.append(("chat", chat_id, int(chunk_idx)))
        else:
            # Collection is set but no scope key — still fail open.
            plans.append((i, None))
            center_keys.append(None)
            continue

        task = _fetch_neighbors(
            vs, collection, doc_id, chat_id, int(chunk_idx), window,
        )
        tasks.append(task)  # type: ignore[arg-type]
        plans.append((i, task))  # type: ignore[arg-type]

    fetched = await asyncio.gather(*tasks) if tasks else []
    # Map task -> result. tasks appears in plans in the same order we appended
    # them; rebuild an iterator to pop results sequentially.
    fetched_iter = iter(fetched)

    # Pass 2: stitch results back in rank order, deduping.
    out: list[Any] = []
    seen: set[tuple[str, Any, int]] = set()

    for (i, task), center in zip(plans, center_keys):
        hit = hits[i]
        if task is None:
            # Legacy / un-scopable hit — keep as-is, but avoid re-adding the
            # same object if it happens to recur (shouldn't, but cheap).
            out.append(hit)
            continue

        expansion = next(fetched_iter, [])
        if not expansion:
            # Fetch failed or empty result — keep original hit.
            if center is not None and center not in seen:
                seen.add(center)
                out.append(hit)
            elif center is None:
                out.append(hit)
            continue

        # Emit siblings in doc order. Dedupe on (scope_kind, scope_val, idx).
        scope_kind = center[0] if center else "doc"
        scope_val = center[1] if center else None
        for eh in expansion:
            ci = eh.payload.get("chunk_index")
            if ci is None:
                continue
            key = (scope_kind, scope_val, int(ci))
            if key in seen:
                continue
            seen.add(key)
            out.append(eh)

    return out
