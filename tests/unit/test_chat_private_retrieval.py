"""P2.3: dual-read from chat_private + legacy chat_{chat_id}.

Invariants under test:
- A chat-scoped retrieval hits BOTH ``chat_private`` (new primary) and
  ``chat_{chat_id}`` (legacy fallback).
- Results are merged and deduped by point id; higher score wins on tie.
- Missing legacy collection is not fatal — ``_search_one`` catches and
  returns ``[]``, so the retrieval succeeds with just the consolidated
  results.
- ``chat_private`` always gets the chat_id filter (so reads within the
  consolidated collection are properly tenant-scoped).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services.retriever import retrieve
from ext.services.vector_store import CHAT_PRIVATE_COLLECTION, Hit


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


def _make_vs(*, primary_hits=None, legacy_hits=None, legacy_raises=False):
    """VectorStore stub that differentiates primary vs legacy calls by collection name."""
    vs = MagicMock()
    vs._refresh_sparse_cache = AsyncMock(return_value=False)
    vs._collection_has_sparse = MagicMock(return_value=False)

    async def _search(collection, *args, **kwargs):
        if collection == CHAT_PRIVATE_COLLECTION:
            return list(primary_hits or [])
        if legacy_raises and collection.startswith("chat_"):
            raise RuntimeError("collection not found")
        return list(legacy_hits or [])

    vs.search = AsyncMock(side_effect=_search)
    vs.hybrid_search = AsyncMock(side_effect=_search)
    return vs


@pytest.mark.asyncio
async def test_dual_read_hits_both_collections(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs(
        primary_hits=[Hit(id="p1", score=0.9, payload={"text": "primary"})],
        legacy_hits=[Hit(id="l1", score=0.7, payload={"text": "legacy"})],
    )
    hits = await retrieve(
        query="q",
        selected_kbs=[],
        chat_id="abc",
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    cols = [c.args[0] for c in vs.search.await_args_list]
    assert CHAT_PRIVATE_COLLECTION in cols
    assert "chat_abc" in cols
    ids = {h.id for h in hits}
    assert {"p1", "l1"} <= ids


@pytest.mark.asyncio
async def test_dedup_prefers_higher_score(monkeypatch) -> None:
    """Same point id appears in both; higher score wins."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs(
        primary_hits=[Hit(id="shared", score=0.95, payload={"text": "new"})],
        legacy_hits=[Hit(id="shared", score=0.50, payload={"text": "old"})],
    )
    hits = await retrieve(
        query="q", selected_kbs=[], chat_id="abc",
        vector_store=vs, embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    matching = [h for h in hits if h.id == "shared"]
    assert len(matching) == 1
    assert matching[0].score == 0.95
    assert matching[0].payload["text"] == "new"


@pytest.mark.asyncio
async def test_missing_legacy_collection_is_not_fatal(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs(
        primary_hits=[Hit(id="p1", score=0.9, payload={"text": "primary"})],
        legacy_raises=True,
    )
    hits = await retrieve(
        query="q", selected_kbs=[], chat_id="abc",
        vector_store=vs, embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    assert [h.id for h in hits] == ["p1"]


@pytest.mark.asyncio
async def test_chat_private_gets_chat_id_filter(monkeypatch) -> None:
    """chat_private call must carry chat_id; legacy call doesn't need it
    (collection itself is chat-scoped)."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs(primary_hits=[], legacy_hits=[])
    await retrieve(
        query="q", selected_kbs=[], chat_id="abc",
        vector_store=vs, embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    calls = {c.args[0]: c.kwargs for c in vs.search.await_args_list}
    assert calls[CHAT_PRIVATE_COLLECTION].get("chat_id") == "abc"
    assert calls["chat_abc"].get("chat_id") is None  # legacy is already chat-scoped by name


@pytest.mark.asyncio
async def test_chat_id_none_skips_chat_arm(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs()
    await retrieve(
        query="q", selected_kbs=[{"kb_id": 1}], chat_id=None,
        vector_store=vs, embedder=_FakeEmbedder(),
    )
    cols = [c.args[0] for c in vs.search.await_args_list]
    assert cols == ["kb_1"]
