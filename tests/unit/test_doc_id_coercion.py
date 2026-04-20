"""Unit tests for doc_id coercion in ingest_bytes (issue-2 regression).

Historically the ingest pipeline stamped ``doc_id = str(doc.id)`` on every
chunk, producing string-typed payloads in Qdrant while earlier collections
carried ints — breaking doc-level eval metrics. We now coerce to int at
ingest time when possible, and leave non-numeric values untouched.

These tests verify the structural fix without needing a real Qdrant.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ext.services.embedder import StubEmbedder
from ext.services.ingest import ingest_bytes


class _FakeVS:
    """Minimal VectorStore stand-in that records the last upsert call."""

    def __init__(self) -> None:
        self.upsert = AsyncMock()


def _txt() -> bytes:
    # Large enough to produce at least one chunk from the default 800/100 config.
    return b"The quick brown fox jumps over the lazy dog. " * 4


async def _call_ingest(payload_base: dict) -> list[dict]:
    """Run ingest_bytes with a trivial text payload; return the upserted points."""
    vs = _FakeVS()
    emb = StubEmbedder(dim=16)
    n = await ingest_bytes(
        data=_txt(),
        mime_type="text/plain",
        filename="a.txt",
        collection="kb_1",
        payload_base=payload_base,
        vector_store=vs,
        embedder=emb,
        chunk_tokens=20,
        overlap_tokens=5,
    )
    assert n >= 1, "expected at least one chunk ingested"
    vs.upsert.assert_awaited_once()
    # upsert was called as (collection, points); grab the points list.
    args, _ = vs.upsert.call_args
    _, points = args
    return list(points)


@pytest.mark.asyncio
async def test_string_numeric_doc_id_is_coerced_to_int() -> None:
    """payload_base={'doc_id': '42'} must end up as int 42 in every point."""
    points = await _call_ingest({"kb_id": 1, "subtag_id": 1, "doc_id": "42"})
    assert points, "no points upserted"
    for p in points:
        assert p["payload"]["doc_id"] == 42
        assert isinstance(p["payload"]["doc_id"], int)
        assert not isinstance(p["payload"]["doc_id"], bool)  # guard against True==1


@pytest.mark.asyncio
async def test_int_doc_id_passes_through_unchanged() -> None:
    """payload_base={'doc_id': 42} stays int 42."""
    points = await _call_ingest({"kb_id": 1, "subtag_id": 1, "doc_id": 42})
    for p in points:
        assert p["payload"]["doc_id"] == 42
        assert isinstance(p["payload"]["doc_id"], int)


@pytest.mark.asyncio
async def test_non_numeric_string_doc_id_is_preserved() -> None:
    """payload_base={'doc_id': 'abc'} must not raise and must stay 'abc'."""
    points = await _call_ingest({"kb_id": 1, "subtag_id": 1, "doc_id": "abc"})
    for p in points:
        assert p["payload"]["doc_id"] == "abc"
        assert isinstance(p["payload"]["doc_id"], str)


@pytest.mark.asyncio
async def test_missing_doc_id_does_not_crash() -> None:
    """payload_base without doc_id (private-chat ingest) must work cleanly."""
    points = await _call_ingest({"chat_id": "chat-abc", "owner_user_id": 7})
    for p in points:
        assert "doc_id" not in p["payload"]
        assert p["payload"]["chat_id"] == "chat-abc"


@pytest.mark.asyncio
async def test_none_doc_id_is_left_alone() -> None:
    """payload_base={'doc_id': None} must not crash nor coerce None to 0."""
    points = await _call_ingest({"kb_id": 1, "doc_id": None})
    for p in points:
        # None stays None; we never coerce it to int.
        assert p["payload"]["doc_id"] is None
