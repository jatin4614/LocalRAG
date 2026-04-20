"""Unit tests for the P3.4 RAPTOR tree builder.

Covers:
  * ``_cluster_embeddings`` corner cases (≤2 inputs, sklearn absent,
    well-separated vectors).
  * ``_summarize_cluster`` happy path + HTTP error via ``MockTransport``.
  * ``build_tree``: below-cluster-min docs produce only leaves; larger
    docs produce at least one level-1 summary; each node carries
    provenance (``source_chunk_ids``).
  * ``is_enabled`` via ``flags.with_overrides``.

The heavy lifting (GMM) is deterministic thanks to ``random_state=0``.
Chat/TEI calls are mocked via ``httpx.MockTransport`` so the suite runs
offline and has no dependency on a running vLLM/TEI.
"""
from __future__ import annotations

import json
import sys
from typing import Optional
from unittest.mock import patch

import httpx
import pytest

pytest.importorskip("sklearn")

from ext.services import flags, raptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic embedder: vector[i] = i-th basis + small jitter from hash.

    Produces vectors similar enough to differentiate texts but stable
    across test runs so GMM results stay reproducible.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = abs(hash(t))
            vec = [((h >> (i * 3)) & 0xFFF) / 4096.0 for i in range(self._dim)]
            out.append(vec)
        return out


def _chat_transport(
    summary: str = "A concise summary.",
    fail: bool = False,
) -> httpx.MockTransport:
    """Build a MockTransport that answers OpenAI-style chat requests."""

    def _handle(request: httpx.Request) -> httpx.Response:
        if fail:
            return httpx.Response(500, text="boom")
        body = {
            "choices": [
                {"message": {"content": summary}, "index": 0}
            ]
        }
        return httpx.Response(200, content=json.dumps(body))

    return httpx.MockTransport(_handle)


# ---------------------------------------------------------------------------
# _cluster_embeddings
# ---------------------------------------------------------------------------


def test_cluster_embeddings_too_few():
    """≤2 embeddings → single cluster (all zeros) without invoking sklearn."""
    assert raptor._cluster_embeddings([]) == []
    assert raptor._cluster_embeddings([[1.0, 2.0]]) == [0]
    assert raptor._cluster_embeddings([[1.0], [2.0]]) == [0, 0]


def test_cluster_embeddings_sklearn_missing(monkeypatch):
    """When sklearn import fails, fall back to single-cluster output."""
    real_import = __import__

    def _blocked(name, *a, **kw):
        if name.startswith("sklearn"):
            raise ImportError("simulated missing sklearn")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", _blocked)
    out = raptor._cluster_embeddings([[0.1] * 8 for _ in range(10)])
    assert out == [0] * 10


def test_cluster_embeddings_well_separated():
    """Given two clusters of near-identical vectors, GMM picks ≥2 distinct ids."""
    # 10 vectors: 5 near origin, 5 near (10,10,...). Well-separated — GMM
    # should easily pick two modes even with diag covariance.
    a = [[0.0 + i * 0.01] * 5 for i in range(5)]
    b = [[10.0 + i * 0.01] * 5 for i in range(5)]
    ids = raptor._cluster_embeddings(a + b)
    assert len(set(ids)) >= 2


# ---------------------------------------------------------------------------
# _summarize_cluster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_cluster_happy():
    transport = _chat_transport("A very concise summary.")
    out = await raptor._summarize_cluster(
        ["alpha", "beta"],
        chat_url="http://fake/v1",
        chat_model="test",
        transport=transport,
    )
    assert out == "A very concise summary."


@pytest.mark.asyncio
async def test_summarize_cluster_error_returns_none():
    transport = _chat_transport(fail=True)
    out = await raptor._summarize_cluster(
        ["alpha"],
        chat_url="http://fake/v1",
        chat_model="test",
        transport=transport,
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarize_cluster_empty_input():
    """No texts → no call, returns None."""
    # No transport touch expected; pass a transport that would blow up if called.
    def _boom(request):
        raise AssertionError("should not call chat on empty input")

    transport = httpx.MockTransport(_boom)
    out = await raptor._summarize_cluster(
        [],
        chat_url="http://fake/v1",
        chat_model="test",
        transport=transport,
    )
    assert out is None


# ---------------------------------------------------------------------------
# build_tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_tree_below_cluster_min():
    """With 3 leaves and cluster_min=5, no intermediate summaries — leaves only."""
    emb = _FakeEmbedder()
    leaves_vecs = await emb.embed(["one", "two", "three"])
    leaves = [
        {"text": f"leaf {i}", "index": i, "embedding": leaves_vecs[i]}
        for i in range(3)
    ]

    nodes = await raptor.build_tree(
        leaves,
        chat_url="http://fake/v1",
        chat_model="test",
        embedder=emb,
        cluster_min=5,
        max_levels=3,
        transport=_chat_transport("should-not-be-called"),
    )
    assert len(nodes) == 3
    assert all(n.level == 0 for n in nodes)
    for i, n in enumerate(nodes):
        assert n.source_chunk_ids == [i]
        assert n.embedding == leaves_vecs[i]


@pytest.mark.asyncio
async def test_build_tree_produces_summary_when_enough_leaves():
    """With 20 well-separated leaves, at least one level-1 summary appears."""
    # Build two groups of near-identical synthetic embeddings so GMM finds >1 cluster.
    leaves = []
    for i in range(10):
        leaves.append({
            "text": f"group A leaf {i}",
            "index": i,
            "embedding": [0.1 + i * 0.001] * 8,
        })
    for i in range(10):
        leaves.append({
            "text": f"group B leaf {i}",
            "index": 10 + i,
            "embedding": [5.0 + i * 0.001] * 8,
        })

    class _Emb:
        async def embed(self, texts):
            # Summary embeddings: arbitrary but deterministic.
            return [[float(len(t)) % 7] * 8 for t in texts]

    transport = _chat_transport("Cluster summary covering the members.")
    nodes = await raptor.build_tree(
        leaves,
        chat_url="http://fake/v1",
        chat_model="test",
        embedder=_Emb(),
        cluster_min=5,
        max_levels=3,
        transport=transport,
    )

    levels = {n.level for n in nodes}
    assert 0 in levels  # leaves always emitted
    assert any(l >= 1 for l in levels), f"expected at least one summary, got levels={levels}"

    # Every summary must cover at least 1 original leaf index from 0..19.
    for n in nodes:
        if n.level >= 1:
            assert len(n.source_chunk_ids) >= 1
            for leaf_idx in n.source_chunk_ids:
                assert 0 <= leaf_idx < 20
            assert n.embedding is not None
            assert len(n.embedding) == 8


@pytest.mark.asyncio
async def test_build_tree_falls_open_on_chat_failure():
    """All-cluster chat failure → only leaves emitted (no exception)."""
    leaves = [
        {"text": f"leaf {i}", "index": i, "embedding": [float(i)] * 8}
        for i in range(12)
    ]

    class _Emb:
        async def embed(self, texts):
            return [[0.0] * 8 for _ in texts]

    transport = _chat_transport(fail=True)
    nodes = await raptor.build_tree(
        leaves,
        chat_url="http://fake/v1",
        chat_model="test",
        embedder=_Emb(),
        cluster_min=5,
        max_levels=3,
        transport=transport,
    )
    # All 12 leaves present; no summary node (chat failed → returns None).
    assert sum(1 for n in nodes if n.level == 0) == 12
    assert all(n.level == 0 for n in nodes)


# ---------------------------------------------------------------------------
# is_enabled
# ---------------------------------------------------------------------------


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_RAPTOR", raising=False)
    assert raptor.is_enabled() is False


def test_is_enabled_env_on(monkeypatch):
    monkeypatch.setenv("RAG_RAPTOR", "1")
    assert raptor.is_enabled() is True


def test_is_enabled_via_flag_overlay(monkeypatch):
    monkeypatch.delenv("RAG_RAPTOR", raising=False)
    with flags.with_overrides({"RAG_RAPTOR": "1"}):
        assert raptor.is_enabled() is True
    # Overlay exits → back to env default.
    assert raptor.is_enabled() is False
