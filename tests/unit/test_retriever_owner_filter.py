"""Unit tests for ``owner_user_id`` propagation in ``retrieve()`` (P2.2).

Invariant under test: the owner filter is forwarded ONLY to chat-scoped
namespace searches (``chat_{chat_id}``). KB collections (``kb_{id}``) remain
shared across all users with kb_access grants — passing the owner filter to
those would silently hide chunks uploaded by other admins, which is the
opposite of what we want.

Default (``owner_user_id=None``) is byte-identical to pre-P2.2 — no owner
argument is passed to ``vs.search`` / ``vs.hybrid_search`` at all.

No real Qdrant — ``VectorStore`` is stubbed as a ``MagicMock`` wrapping
``AsyncMock`` for the query methods.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services.retriever import retrieve
from ext.services.vector_store import Hit


class _FakeEmbedder:
    async def embed(self, texts):
        return [[0.1] * 1024 for _ in texts]


def _make_vs_stub(*, sparse_support: bool = False):
    """VectorStore-like stub with recordable ``search``/``hybrid_search``."""
    vs = MagicMock()
    vs.search = AsyncMock(return_value=[Hit(id="d1", score=0.9, payload={"text": "d"})])
    vs.hybrid_search = AsyncMock(return_value=[Hit(id="h1", score=0.8, payload={"text": "h"})])
    vs._refresh_sparse_cache = AsyncMock(return_value=sparse_support)
    vs._collection_has_sparse = MagicMock(return_value=sparse_support)
    return vs


# ---------- Default: owner_user_id=None stays byte-identical ----------------


@pytest.mark.asyncio
async def test_default_no_owner_passes_none_to_every_search(monkeypatch) -> None:
    """When ``owner_user_id`` is omitted / None, all ``vs.search`` calls
    receive ``owner_user_id=None`` (no filter) — byte-identical to pre-P2.2.

    We check both KB and chat collections. ``None`` is how the retriever
    communicates "no filter" down to ``vs.search``; the vector store then
    omits the owner condition from the Qdrant filter (covered in its own
    test module).
    """
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=False)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=42,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # 1 KB + 2 chat dual-read (chat_private + legacy chat_42) = 3 calls.
    assert vs.search.await_count == 3
    for call in vs.search.await_args_list:
        # owner_user_id must be None (not missing) so vs.search's own default
        # kicks in (== None == no filter).
        assert call.kwargs.get("owner_user_id") is None


@pytest.mark.asyncio
async def test_default_no_owner_on_hybrid_path(monkeypatch) -> None:
    """Same invariant on the hybrid code path."""
    monkeypatch.setenv("RAG_HYBRID", "1")
    vs = _make_vs_stub(sparse_support=True)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=42,
        vector_store=vs,
        embedder=_FakeEmbedder(),
    )
    # 1 KB + 2 chat dual-read = 3 hybrid calls.
    assert vs.hybrid_search.await_count == 3
    for call in vs.hybrid_search.await_args_list:
        assert call.kwargs.get("owner_user_id") is None


# ---------- KB collections never get the owner filter -----------------------


@pytest.mark.asyncio
async def test_owner_filter_never_forwarded_to_kb_search(monkeypatch) -> None:
    """With a real ``owner_user_id=7`` in scope, the KB search must still
    see ``owner_user_id=None`` — KBs are shared tenants. Only the chat
    collection gets the filter.
    """
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=False)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}, {"kb_id": 2}],
        chat_id=99,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    # 2 KB + 2 chat dual-read (chat_private + legacy chat_99) = 4 calls.
    assert vs.search.await_count == 4
    calls_by_collection = {
        call.args[0]: call.kwargs.get("owner_user_id")
        for call in vs.search.await_args_list
    }
    assert calls_by_collection["kb_1"] is None
    assert calls_by_collection["kb_2"] is None
    # Both chat arms get the owner filter.
    assert calls_by_collection["chat_private"] == 7
    assert calls_by_collection["chat_99"] == 7


@pytest.mark.asyncio
async def test_owner_filter_never_forwarded_to_kb_hybrid(monkeypatch) -> None:
    """Hybrid code path: same invariant — KBs never receive the owner filter."""
    monkeypatch.setenv("RAG_HYBRID", "1")
    vs = _make_vs_stub(sparse_support=True)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=99,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    calls_by_collection = {
        call.args[0]: call.kwargs.get("owner_user_id")
        for call in vs.hybrid_search.await_args_list
    }
    assert calls_by_collection["kb_1"] is None
    # Both chat arms (chat_private primary + legacy chat_99) get the filter.
    assert calls_by_collection["chat_private"] == 7
    assert calls_by_collection["chat_99"] == 7


# ---------- Chat collections get the owner filter ---------------------------


@pytest.mark.asyncio
async def test_owner_filter_forwarded_to_chat_search(monkeypatch) -> None:
    """``owner_user_id=7`` lands on the chat-scoped search call."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=False)
    await retrieve(
        query="q",
        selected_kbs=[],
        chat_id=99,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    # Only chat arm runs (no KBs selected) but dual-read = 2 calls.
    assert vs.search.await_count == 2
    cols = [c.args[0] for c in vs.search.await_args_list]
    assert "chat_private" in cols
    assert "chat_99" in cols
    for call in vs.search.await_args_list:
        assert call.kwargs["owner_user_id"] == 7


@pytest.mark.asyncio
async def test_owner_filter_forwarded_as_uuid_string(monkeypatch) -> None:
    """Production owner ids are UUID strings — must propagate unchanged."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=False)
    uuid = "aaaa-bbbb-cccc-dddd"
    await retrieve(
        query="q",
        selected_kbs=[],
        chat_id=99,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=uuid,
    )
    assert vs.search.await_args_list[0].kwargs["owner_user_id"] == uuid


@pytest.mark.asyncio
async def test_owner_filter_not_forwarded_when_chat_id_absent(monkeypatch) -> None:
    """If ``chat_id`` is None, no chat search happens regardless of owner."""
    monkeypatch.setenv("RAG_HYBRID", "0")
    vs = _make_vs_stub(sparse_support=False)
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_FakeEmbedder(),
        owner_user_id=7,
    )
    assert vs.search.await_count == 1
    call = vs.search.await_args_list[0]
    assert call.args[0] == "kb_1"
    assert call.kwargs.get("owner_user_id") is None
