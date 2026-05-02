"""Phase 6.X — MMR ``max_input_size`` cap (avoid TEI OOM).

Multi-entity decomposition (Method 3) can produce 200+ candidates;
sending all of them to the TEI re-embed batch alongside QU LLM +
reranker on the same GPU OOMs. The cap pre-trims the input list
before the embedder call. Default ``None`` is byte-identical to
pre-Phase-6 behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ext.services import mmr


@dataclass
class _FakeHit:
    id: int
    score: float
    payload: dict


class _FakeEmbedder:
    """Records how many texts hit ``embed`` per call."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(len(texts))
        # Return deterministic dense vectors. mmr_rerank consumes
        # ``vectors[:-1]`` as passages, ``vectors[-1]`` as query.
        return [[float(i % 7), float(i % 11), float(i % 13)] for i in range(len(texts))]


def _mk_hits(n: int) -> list[_FakeHit]:
    return [
        _FakeHit(
            id=i,
            score=1.0 - i * 0.001,
            payload={"text": f"chunk-{i}"},
        )
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_no_cap_default_passes_all_hits_to_embedder() -> None:
    """Backward compat: default ``max_input_size=None`` sends every
    hit to the embedder (pre-Phase-6 behaviour)."""
    emb = _FakeEmbedder()
    hits = _mk_hits(200)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb, top_k=10, lambda_=0.7,
    )
    # 200 passages + 1 query = 201 texts in the single TEI call.
    assert emb.calls == [201]
    assert len(out) == 10


@pytest.mark.asyncio
async def test_cap_50_trims_before_embed() -> None:
    """With ``max_input_size=50`` and 200 hits, only 50 reach TEI."""
    emb = _FakeEmbedder()
    hits = _mk_hits(200)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb,
        top_k=10, lambda_=0.7, max_input_size=50,
    )
    # 50 passages + 1 query = 51 texts.
    assert emb.calls == [51]
    # Still returns top_k results (cap doesn't shrink the output below top_k).
    assert len(out) == 10


@pytest.mark.asyncio
async def test_cap_at_zero_treated_as_unlimited() -> None:
    """Defensive: 0 disables the cap (matches the env-flag convention)."""
    emb = _FakeEmbedder()
    hits = _mk_hits(150)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb,
        top_k=10, lambda_=0.7, max_input_size=0,
    )
    # No cap applied → all 150 + query.
    assert emb.calls == [151]
    assert len(out) == 10


@pytest.mark.asyncio
async def test_cap_above_hit_count_passes_all() -> None:
    """When the cap is above len(hits), no trim happens."""
    emb = _FakeEmbedder()
    hits = _mk_hits(20)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb,
        top_k=10, lambda_=0.7, max_input_size=100,
    )
    # All 20 passages + query.
    assert emb.calls == [21]
    assert len(out) == 10


@pytest.mark.asyncio
async def test_cap_below_top_k_still_returns_top_k() -> None:
    """Edge: cap=5, top_k=10 — MMR scores 5, the result is backfilled
    from the tail to keep the contract that we return up to top_k."""
    emb = _FakeEmbedder()
    hits = _mk_hits(50)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb,
        top_k=10, lambda_=0.7, max_input_size=5,
    )
    assert emb.calls == [6]  # 5 passages + 1 query
    assert len(out) == 10  # contract preserved via tail backfill


@pytest.mark.asyncio
async def test_cap_with_empty_hits_short_circuits() -> None:
    emb = _FakeEmbedder()
    out = await mmr.mmr_rerank_from_hits(
        "query", [], emb, top_k=10, lambda_=0.7, max_input_size=50,
    )
    assert out == []
    assert emb.calls == []


@pytest.mark.asyncio
async def test_top_k_above_hits_returns_all() -> None:
    """Backward compat: top_k >= len(hits) bypasses MMR entirely."""
    emb = _FakeEmbedder()
    hits = _mk_hits(5)
    out = await mmr.mmr_rerank_from_hits(
        "query", hits, emb, top_k=10, lambda_=0.7, max_input_size=50,
    )
    assert len(out) == 5
    assert emb.calls == []  # short-circuited before embed
