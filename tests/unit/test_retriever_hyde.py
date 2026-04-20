"""Unit tests for HyDE integration in ``ext.services.retriever``.

Verifies:
* Default path (``RAG_HYDE`` unset / ``0``): ``hyde_embed`` is NOT called,
  module is NOT imported — byte-identical to pre-P3.3 behaviour.
* Flag-on path with a working chat stub: ``hyde_embed`` is called and its
  returned vector is used as ``qvec`` (passed to vector-store searches).
* Flag-on path with all-failed HyDE: falls back to embedding the raw query,
  retrieval proceeds as normal.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from ext.services.retriever import retrieve
from ext.services.vector_store import Hit


class _RecordingEmbedder:
    """Records every ``embed`` call so tests can assert what was sent."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1] * 1024 for _ in texts]


def _make_vs_stub():
    """Minimal VectorStore stub — dense search only (hybrid disabled via flag)."""
    vs = MagicMock()
    vs.search = AsyncMock(return_value=[Hit(id="d1", score=0.9, payload={"text": "doc"})])
    vs.hybrid_search = AsyncMock(return_value=[])
    vs._refresh_sparse_cache = AsyncMock(return_value=False)
    vs._collection_has_sparse = MagicMock(return_value=False)
    return vs


# ---------------------------------------------------------------------------
# Default path — HyDE disabled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hyde_disabled_does_not_import_module(monkeypatch) -> None:
    """With RAG_HYDE unset, ``ext.services.hyde`` is NOT imported.

    This is the byte-identical-default-path guarantee. The module stays
    out of ``sys.modules`` so there's zero cost (no import, no network,
    no branch executed beyond a single ``flags.get`` read).
    """
    monkeypatch.delenv("RAG_HYDE", raising=False)
    monkeypatch.setenv("RAG_HYBRID", "0")  # keep search path simple
    # Wipe any prior import so the absence assertion is meaningful.
    sys.modules.pop("ext.services.hyde", None)

    vs = _make_vs_stub()
    embedder = _RecordingEmbedder()
    await retrieve(
        query="abstract definitional question",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=embedder,
    )

    # Module never loaded on the default path.
    assert "ext.services.hyde" not in sys.modules
    # Exactly one embedder call with the raw query.
    assert embedder.calls == [["abstract definitional question"]]


@pytest.mark.asyncio
async def test_hyde_flag_explicitly_zero_no_import(monkeypatch) -> None:
    """RAG_HYDE=0 is treated identically to unset."""
    monkeypatch.setenv("RAG_HYDE", "0")
    monkeypatch.setenv("RAG_HYBRID", "0")
    sys.modules.pop("ext.services.hyde", None)

    vs = _make_vs_stub()
    embedder = _RecordingEmbedder()
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=embedder,
    )

    assert "ext.services.hyde" not in sys.modules
    assert embedder.calls == [["q"]]


# ---------------------------------------------------------------------------
# Flag-on path — working HyDE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hyde_enabled_uses_hyde_vector(monkeypatch) -> None:
    """RAG_HYDE=1 + a working mock → hyde_embed is called and its vector is used."""
    monkeypatch.setenv("RAG_HYDE", "1")
    monkeypatch.setenv("RAG_HYBRID", "0")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "orgchat-chat")

    # Patch hyde_embed at its import site (ext.services.hyde) so the retriever's
    # lazy ``from ext.services.hyde import hyde_embed`` picks up our fake.
    fake_vec = [0.42] * 1024

    # Import so it's already in sys.modules, then swap the function.
    from ext.services import hyde as hyde_mod
    called = {}

    async def fake_hyde_embed(query, embedder, **kwargs):
        called["query"] = query
        called["n"] = kwargs.get("n")
        called["chat_url"] = kwargs.get("chat_url")
        called["chat_model"] = kwargs.get("chat_model")
        return fake_vec

    # Ensure the (possibly popped) module is back in sys.modules so the
    # retriever's lazy ``from ext.services.hyde import hyde_embed`` picks
    # up the monkeypatched attribute rather than re-executing the module.
    import sys as _sys
    _sys.modules["ext.services.hyde"] = hyde_mod
    monkeypatch.setattr(hyde_mod, "hyde_embed", fake_hyde_embed)

    vs = _make_vs_stub()
    embedder = _RecordingEmbedder()
    await retrieve(
        query="what is our refund policy for B2B?",
        selected_kbs=[{"kb_id": 7}],
        chat_id=None,
        vector_store=vs,
        embedder=embedder,
    )

    # HyDE was called with the raw query and the configured chat endpoint.
    assert called["query"] == "what is our refund policy for B2B?"
    assert called["n"] == 1  # default
    assert called["chat_url"] == "http://fake:8000/v1"
    assert called["chat_model"] == "orgchat-chat"

    # The retriever skipped its own ``embedder.embed([query])`` call — the
    # HyDE vector was used directly.
    assert embedder.calls == []

    # vs.search was called with fake_vec as the query vector.
    assert vs.search.await_count == 1
    call_args = vs.search.await_args
    passed_qvec = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("qvec")
    # positional: search(collection, qvec, ...)
    assert passed_qvec == fake_vec


@pytest.mark.asyncio
async def test_hyde_enabled_n_override(monkeypatch) -> None:
    """RAG_HYDE_N propagates into the hyde_embed call."""
    monkeypatch.setenv("RAG_HYDE", "1")
    monkeypatch.setenv("RAG_HYDE_N", "5")
    monkeypatch.setenv("RAG_HYBRID", "0")

    from ext.services import hyde as hyde_mod
    import sys as _sys
    _sys.modules["ext.services.hyde"] = hyde_mod
    seen = {}

    async def fake_hyde_embed(query, embedder, **kwargs):
        seen["n"] = kwargs.get("n")
        return [0.1] * 1024

    monkeypatch.setattr(hyde_mod, "hyde_embed", fake_hyde_embed)

    vs = _make_vs_stub()
    await retrieve(
        query="q",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=_RecordingEmbedder(),
    )

    assert seen["n"] == 5


# ---------------------------------------------------------------------------
# Flag-on path — HyDE fails, fall back to raw query
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hyde_returns_none_falls_back_to_raw_embed(monkeypatch) -> None:
    """When hyde_embed returns None (all generations failed), the retriever
    embeds the raw query like normal. No exception propagates."""
    monkeypatch.setenv("RAG_HYDE", "1")
    monkeypatch.setenv("RAG_HYBRID", "0")

    from ext.services import hyde as hyde_mod
    import sys as _sys
    _sys.modules["ext.services.hyde"] = hyde_mod

    async def fake_hyde_embed(query, embedder, **kwargs):
        return None  # All generations failed.

    monkeypatch.setattr(hyde_mod, "hyde_embed", fake_hyde_embed)

    vs = _make_vs_stub()
    embedder = _RecordingEmbedder()
    hits = await retrieve(
        query="fallback test query",
        selected_kbs=[{"kb_id": 1}],
        chat_id=None,
        vector_store=vs,
        embedder=embedder,
    )

    # Raw-query fallback happened.
    assert embedder.calls == [["fallback test query"]]
    # Retrieval still produced results.
    assert len(hits) == 1
    assert hits[0].id == "d1"


@pytest.mark.asyncio
async def test_hyde_per_kb_overlay_can_flip_on(monkeypatch) -> None:
    """Per-KB overlay (via flags.with_overrides) can enable HyDE without
    the process env flag being set — this is the P3.0 per-KB-config
    integration path."""
    # Make sure the process env flag is OFF.
    monkeypatch.delenv("RAG_HYDE", raising=False)
    monkeypatch.setenv("RAG_HYBRID", "0")

    from ext.services import flags, hyde as hyde_mod
    import sys as _sys
    _sys.modules["ext.services.hyde"] = hyde_mod
    seen = {"called": False}

    async def fake_hyde_embed(query, embedder, **kwargs):
        seen["called"] = True
        return [0.5] * 1024

    monkeypatch.setattr(hyde_mod, "hyde_embed", fake_hyde_embed)

    vs = _make_vs_stub()
    embedder = _RecordingEmbedder()
    # Simulate bridge-layer behaviour: overlay turns HyDE on for this request.
    with flags.with_overrides({"RAG_HYDE": "1"}):
        await retrieve(
            query="q",
            selected_kbs=[{"kb_id": 1}],
            chat_id=None,
            vector_store=vs,
            embedder=embedder,
        )

    assert seen["called"] is True
    # Outside the overlay, flag reverts — no leak to later calls.
    import os
    assert os.environ.get("RAG_HYDE") is None
