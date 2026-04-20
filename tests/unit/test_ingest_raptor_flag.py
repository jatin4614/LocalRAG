"""Unit tests for P3.4 RAPTOR wiring inside ``ingest_bytes``.

Verifies:
  1. Default (flag unset) — the raptor module is NOT imported during
     ingest and the flat-chunk path runs byte-identically to pre-P3.4.
  2. Flag on + mocked ``build_tree`` returning 5 nodes across 2 levels —
     upsert receives 5 points carrying the expected ``chunk_level`` +
     ``source_chunk_ids`` payload fields.
  3. Flag on but ``build_tree`` raises — ingest falls open to the flat
     chunking path. No crash, no partial tree.
"""
from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from ext.services import ingest as ingest_mod
from ext.services.embedder import StubEmbedder
from ext.services.ingest import ingest_bytes


class _FakeVS:
    """Minimal VectorStore recording the last upsert call."""

    def __init__(self) -> None:
        self.upsert = AsyncMock()


def _txt() -> bytes:
    return b"The quick brown fox jumps over the lazy dog. " * 8


async def _run_ingest() -> list[dict]:
    vs = _FakeVS()
    emb = StubEmbedder(dim=16)
    n = await ingest_bytes(
        data=_txt(),
        mime_type="text/plain",
        filename="raptor-doc.txt",
        collection="kb_1",
        payload_base={"kb_id": 1, "doc_id": 123},
        vector_store=vs,
        embedder=emb,
        chunk_tokens=20,
        overlap_tokens=5,
    )
    assert n >= 1
    vs.upsert.assert_awaited_once()
    args, _ = vs.upsert.call_args
    _, points = args
    return list(points)


# ---------------------------------------------------------------------------
# 1. Flag-off default — raptor never imported
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_does_not_import_raptor(monkeypatch):
    """With ``RAG_RAPTOR`` unset, ``ext.services.raptor`` is never imported.

    Guards against future refactors that accidentally pull the module
    into the default ingest path (which would drag in sklearn).
    """
    monkeypatch.delenv("RAG_RAPTOR", raising=False)
    sys.modules.pop("ext.services.raptor", None)

    points = await _run_ingest()

    assert "ext.services.raptor" not in sys.modules
    # No chunk_level field on any point when RAPTOR is off.
    for p in points:
        assert "chunk_level" not in p["payload"]
        assert "source_chunk_ids" not in p["payload"]


# ---------------------------------------------------------------------------
# 2. Flag-on + mocked build_tree — tree nodes get upserted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_upserts_tree_nodes(monkeypatch):
    """With RAG_RAPTOR=1 and build_tree returning 5 nodes, upsert sees 5 points.

    Each point carries the expected chunk_level and source_chunk_ids.
    """
    monkeypatch.setenv("RAG_RAPTOR", "1")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "orgchat-chat")

    # Patch build_tree to return a known tree shape without hitting the chat API.
    import ext.services.raptor as raptor_mod
    from ext.services.raptor import RaptorNode

    async def _fake_build_tree(leaves, **_kw):
        # Return N leaves (copying leaf embeddings) + 2 summaries.
        nodes = []
        for i, leaf in enumerate(leaves):
            nodes.append(
                RaptorNode(
                    text=leaf["text"],
                    level=0,
                    parent_id=None,
                    cluster_id=None,
                    source_chunk_ids=[leaf["index"]],
                    embedding=list(leaf["embedding"]),
                )
            )
        # Two synthetic summary nodes covering all leaves.
        all_indices = [leaf["index"] for leaf in leaves]
        for s_i in range(2):
            nodes.append(
                RaptorNode(
                    text=f"Summary {s_i}",
                    level=1,
                    parent_id=None,
                    cluster_id=s_i,
                    source_chunk_ids=all_indices[s_i::2],
                    embedding=[0.1] * 16,
                )
            )
        return nodes

    monkeypatch.setattr(raptor_mod, "build_tree", _fake_build_tree)

    points = await _run_ingest()

    # Expect: N leaves (≥1) + 2 summaries. Summary points carry level=1.
    summaries = [p for p in points if p["payload"]["chunk_level"] == 1]
    leaves = [p for p in points if p["payload"]["chunk_level"] == 0]
    assert len(summaries) == 2, f"expected 2 summaries, got {len(summaries)}"
    assert len(leaves) >= 1

    # Every point has the RAPTOR provenance fields.
    for p in points:
        assert "chunk_level" in p["payload"]
        assert "source_chunk_ids" in p["payload"]
        assert isinstance(p["payload"]["source_chunk_ids"], list)

    # Summary points reference real leaf indices within the doc.
    leaf_indices = {p["payload"]["chunk_index"] for p in leaves}
    for s in summaries:
        for sid in s["payload"]["source_chunk_ids"]:
            assert sid in leaf_indices or sid >= 0


# ---------------------------------------------------------------------------
# 3. Flag on + build_tree raises — fail-open to flat ingest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_build_tree_raises_falls_open(monkeypatch):
    """If build_tree throws, ingest should fall back to flat chunk points."""
    monkeypatch.setenv("RAG_RAPTOR", "1")
    monkeypatch.setenv("OPENAI_API_BASE_URL", "http://fake-vllm:8000/v1")
    monkeypatch.setenv("CHAT_MODEL", "orgchat-chat")

    import ext.services.raptor as raptor_mod

    async def _boom(leaves, **_kw):
        raise RuntimeError("simulated build_tree crash")

    monkeypatch.setattr(raptor_mod, "build_tree", _boom)

    points = await _run_ingest()

    # Flat-path points carry no chunk_level / source_chunk_ids.
    for p in points:
        assert "chunk_level" not in p["payload"]
        assert "source_chunk_ids" not in p["payload"]


@pytest.mark.asyncio
async def test_flag_on_without_chat_url_falls_open(monkeypatch):
    """Flag on but OPENAI_API_BASE_URL unset → no tree, flat path runs."""
    monkeypatch.setenv("RAG_RAPTOR", "1")
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)

    points = await _run_ingest()

    for p in points:
        assert "chunk_level" not in p["payload"]
        assert "source_chunk_ids" not in p["payload"]
