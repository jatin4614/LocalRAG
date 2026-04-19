"""Tests for ``expand_context`` — parent-document retrieval post-rerank.

We test against a stub ``VectorStore``-like object that only exposes
``_client.scroll(...)``. The scroll return shape matches qdrant-client:
``(list[Record], next_page_offset)`` where each Record has ``.id`` and
``.payload``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fixtures: stub hits + stub vector-store-ish client
# ---------------------------------------------------------------------------
@dataclass
class _StubHit:
    id: Any
    score: float
    payload: dict


@dataclass
class _StubRecord:
    """Mimics qdrant_client's Record (id + payload)."""
    id: Any
    payload: dict


class _StubScrollClient:
    """Implements only ``async scroll(...)`` -> (records, None).

    ``records_by_doc`` is a mapping ``{(collection, doc_id_or_chat_id): [records]}``.
    Every scroll call filters by chunk_index range; we apply that server-side
    so the test can assert a single call per hit and the filtered response.

    ``raise_on_collection`` triggers an exception for a specific collection
    name — used to verify fail-open on scroll failure.
    ``empty_for_collection`` returns an empty record list — used to verify
    the "expansion returns nothing; keep original" path.
    """

    def __init__(
        self,
        records_by_scope: dict[tuple[str, Any], list[_StubRecord]],
        *,
        raise_on_collection: str | None = None,
        empty_for_collection: str | None = None,
    ) -> None:
        self._records_by_scope = records_by_scope
        self._raise_on_collection = raise_on_collection
        self._empty_for_collection = empty_for_collection
        self.calls: list[dict] = []

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[_StubRecord], Any]:
        self.calls.append(
            {
                "collection": collection_name,
                "filter": scroll_filter,
                "limit": limit,
            }
        )
        if self._raise_on_collection == collection_name:
            raise RuntimeError("scroll failed for test")
        if self._empty_for_collection == collection_name:
            return [], None

        # Extract the chunk_index range and scope value from the filter
        # (we don't want to hardcode qm's internal repr).
        lo, hi = None, None
        scope_key: Any = None
        for cond in scroll_filter.must or []:
            if cond.key == "chunk_index" and cond.range is not None:
                lo = cond.range.gte
                hi = cond.range.lte
            elif cond.key in ("doc_id", "chat_id") and cond.match is not None:
                scope_key = cond.match.value

        all_records = self._records_by_scope.get((collection_name, scope_key), [])
        out = [
            r for r in all_records
            if lo is not None and hi is not None
            and (r.payload or {}).get("chunk_index") is not None
            and lo <= r.payload["chunk_index"] <= hi
            and not r.payload.get("deleted", False)
        ]
        return out, None


@dataclass
class _StubVectorStore:
    _client: _StubScrollClient


# ---------------------------------------------------------------------------
# Basic short-circuit behavior
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_empty_hits_returns_empty():
    from ext.services.context_expand import expand_context

    vs = _StubVectorStore(_client=_StubScrollClient({}))
    out = await expand_context([], vs=vs, window=1)
    assert out == []
    assert vs._client.calls == []


@pytest.mark.asyncio
async def test_window_zero_returns_hits_unchanged():
    from ext.services.context_expand import expand_context

    hits = [_StubHit(id=1, score=0.9, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5})]
    vs = _StubVectorStore(_client=_StubScrollClient({}))
    out = await expand_context(hits, vs=vs, window=0)
    assert out == hits
    assert vs._client.calls == []


# ---------------------------------------------------------------------------
# Happy path: single center, window=1, fetch 3 siblings
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_center_window_one_returns_three_sorted():
    from ext.services.context_expand import expand_context

    hit = _StubHit(
        id=100,
        score=0.9,
        payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "text": "center"},
    )
    records = [
        _StubRecord(id=6, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 6, "text": "after"}),
        _StubRecord(id=4, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 4, "text": "before"}),
        _StubRecord(id=5, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "text": "center"}),
    ]
    vs = _StubVectorStore(
        _client=_StubScrollClient({("kb_1", 10): records})
    )
    out = await expand_context([hit], vs=vs, window=1)
    # All 3 records returned, sorted ascending by chunk_index.
    assert len(out) == 3
    idxs = [h.payload["chunk_index"] for h in out]
    assert idxs == [4, 5, 6]
    # Exactly one scroll call
    assert len(vs._client.calls) == 1
    assert vs._client.calls[0]["collection"] == "kb_1"


# ---------------------------------------------------------------------------
# Two adjacent centers in same doc: overlap dedupe, union preserved
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_two_adjacent_centers_same_doc_dedupe_overlap():
    from ext.services.context_expand import expand_context

    # Two hits, both in doc 10, at chunk_index 5 and 6. Window=2 => windows
    # [3,4,5,6,7] and [4,5,6,7,8]. Union should be [3..8].
    hits = [
        _StubHit(id=5, score=0.9, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5}),
        _StubHit(id=6, score=0.8, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 6}),
    ]
    # Doc has chunks 3 through 8
    all_recs = [
        _StubRecord(id=i, payload={"kb_id": 1, "doc_id": 10, "chunk_index": i})
        for i in range(3, 9)
    ]
    vs = _StubVectorStore(
        _client=_StubScrollClient({("kb_1", 10): all_recs})
    )
    out = await expand_context(hits, vs=vs, window=2)

    # Expect unique chunk_indices [3,4,5,6,7,8]
    idxs = [h.payload["chunk_index"] for h in out]
    assert idxs == [3, 4, 5, 6, 7, 8]
    # Both scroll calls happened
    assert len(vs._client.calls) == 2


# ---------------------------------------------------------------------------
# Legacy hit: no chunk_index -> pass through untouched, no fetch attempted
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_legacy_hit_no_chunk_index_passes_through():
    from ext.services.context_expand import expand_context

    hit = _StubHit(id=1, score=0.9, payload={"kb_id": 1, "doc_id": 10, "text": "legacy"})
    vs = _StubVectorStore(_client=_StubScrollClient({}))
    out = await expand_context([hit], vs=vs, window=1)
    assert len(out) == 1
    assert out[0] is hit
    assert vs._client.calls == []


# ---------------------------------------------------------------------------
# Fetch raises -> keep original hit, don't crash
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_raises_keeps_original():
    from ext.services.context_expand import expand_context

    hit = _StubHit(
        id=1,
        score=0.9,
        payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "text": "hello"},
    )
    vs = _StubVectorStore(
        _client=_StubScrollClient({}, raise_on_collection="kb_1")
    )
    out = await expand_context([hit], vs=vs, window=1)
    assert len(out) == 1
    assert out[0] is hit


# ---------------------------------------------------------------------------
# Fetch returns empty -> original hit kept, no siblings added
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_empty_keeps_original():
    from ext.services.context_expand import expand_context

    hit = _StubHit(
        id=1,
        score=0.9,
        payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "text": "hello"},
    )
    vs = _StubVectorStore(
        _client=_StubScrollClient({}, empty_for_collection="kb_1")
    )
    out = await expand_context([hit], vs=vs, window=1)
    assert len(out) == 1
    assert out[0] is hit


# ---------------------------------------------------------------------------
# Two different docs: rank order preserved across doc boundaries
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_two_different_docs_preserve_rank_order():
    from ext.services.context_expand import expand_context

    # Rank 1 is doc 10 chunk 5; rank 2 is doc 20 chunk 3. Window=1.
    hits = [
        _StubHit(id=5, score=0.9, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5}),
        _StubHit(id=3, score=0.8, payload={"kb_id": 1, "doc_id": 20, "chunk_index": 3}),
    ]
    doc10 = [
        _StubRecord(id=4, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 4}),
        _StubRecord(id=5, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5}),
        _StubRecord(id=6, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 6}),
    ]
    doc20 = [
        _StubRecord(id=2, payload={"kb_id": 1, "doc_id": 20, "chunk_index": 2}),
        _StubRecord(id=3, payload={"kb_id": 1, "doc_id": 20, "chunk_index": 3}),
        _StubRecord(id=4, payload={"kb_id": 1, "doc_id": 20, "chunk_index": 4}),
    ]
    vs = _StubVectorStore(
        _client=_StubScrollClient(
            {("kb_1", 10): doc10, ("kb_1", 20): doc20}
        )
    )
    out = await expand_context(hits, vs=vs, window=1)

    # doc 10 siblings emitted first (rank 1), then doc 20 siblings (rank 2).
    # Within each doc, ascending by chunk_index.
    emitted = [(h.payload["doc_id"], h.payload["chunk_index"]) for h in out]
    assert emitted == [(10, 4), (10, 5), (10, 6), (20, 2), (20, 3), (20, 4)]


# ---------------------------------------------------------------------------
# Private-chat hit (no doc_id, has chat_id) expands via chat scope
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_private_chat_hit_expands_via_chat_scope():
    from ext.services.context_expand import expand_context

    hit = _StubHit(
        id=1,
        score=0.9,
        payload={"chat_id": 42, "chunk_index": 5, "text": "private"},
    )
    recs = [
        _StubRecord(id=4, payload={"chat_id": 42, "chunk_index": 4}),
        _StubRecord(id=5, payload={"chat_id": 42, "chunk_index": 5}),
        _StubRecord(id=6, payload={"chat_id": 42, "chunk_index": 6}),
    ]
    vs = _StubVectorStore(
        _client=_StubScrollClient({("chat_42", 42): recs})
    )
    out = await expand_context([hit], vs=vs, window=1)
    assert len(out) == 3
    assert [h.payload["chunk_index"] for h in out] == [4, 5, 6]
    assert vs._client.calls[0]["collection"] == "chat_42"


# ---------------------------------------------------------------------------
# Deleted siblings are filtered out (the must_not condition)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_deleted_siblings_excluded():
    from ext.services.context_expand import expand_context

    hit = _StubHit(
        id=5,
        score=0.9,
        payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "deleted": False},
    )
    recs = [
        _StubRecord(id=4, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 4, "deleted": True}),
        _StubRecord(id=5, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 5, "deleted": False}),
        _StubRecord(id=6, payload={"kb_id": 1, "doc_id": 10, "chunk_index": 6, "deleted": False}),
    ]
    vs = _StubVectorStore(
        _client=_StubScrollClient({("kb_1", 10): recs})
    )
    out = await expand_context([hit], vs=vs, window=1)
    # chunk_index 4 is deleted, so only 5 and 6 remain.
    assert len(out) == 2
    assert [h.payload["chunk_index"] for h in out] == [5, 6]
