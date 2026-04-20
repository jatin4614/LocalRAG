"""Unit tests for ``ext.services.mmr``: pure-function behaviour, no I/O."""
from __future__ import annotations

import math

import pytest

from ext.services.mmr import _cosine, mmr_rerank


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------
def test_cosine_orthogonal_is_zero():
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_identical_is_one():
    v = [0.3, 0.4, -0.5]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_opposite_is_minus_one():
    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_returns_zero():
    assert _cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0
    assert _cosine([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0
    assert _cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_cosine_mixed_angle():
    # 45 degrees between unit x-axis and (1, 1) / sqrt(2)
    s = 1.0 / math.sqrt(2.0)
    assert _cosine([1.0, 0.0], [s, s]) == pytest.approx(s)


# ---------------------------------------------------------------------------
# Shape / edge cases
# ---------------------------------------------------------------------------
class _StubHit:
    def __init__(self, id: int) -> None:
        self.id = id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"_StubHit(id={self.id})"


def test_empty_hits_returns_empty():
    assert mmr_rerank([1.0, 0.0], [], [], top_k=5) == []


def test_top_k_ge_len_hits_returns_input_unchanged():
    """MMR only meaningful when selecting a strict subset."""
    hits = [_StubHit(0), _StubHit(1), _StubHit(2)]
    vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    out = mmr_rerank([1.0, 0.0], vecs, hits, top_k=3)
    assert [h.id for h in out] == [0, 1, 2]

    # top_k greater than len: still byte-identical passthrough
    out = mmr_rerank([1.0, 0.0], vecs, hits, top_k=10)
    assert [h.id for h in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Lambda extremes
# ---------------------------------------------------------------------------
def test_lambda_one_pure_relevance_order():
    """lambda=1.0 means diversity term is zeroed out - order by relevance only."""
    q = [1.0, 0.0]
    # Relevance ordering: hit 0 (dot=1.0) > hit 1 (dot=0.5) > hit 2 (dot=0.1)
    vecs = [[1.0, 0.0], [0.5, 0.0], [0.1, 0.0]]
    hits = [_StubHit(0), _StubHit(1), _StubHit(2)]
    out = mmr_rerank(q, vecs, hits, top_k=2, lambda_=1.0)
    assert [h.id for h in out] == [0, 1]


def test_lambda_zero_pure_diversity_after_first_pick():
    """lambda=0.0: first pick is most relevant, subsequent picks maximise distance."""
    q = [1.0, 0.0]
    # hit 0 very relevant; hit 1 is relevant+near-dup of hit 0; hit 2 is low-relevance but ORTHOGONAL
    vecs = [
        [1.0, 0.0],    # hit 0: high relevance (cos=1.0)
        [0.9, 0.001],  # hit 1: high relevance (cos~=1.0), near-duplicate of hit 0
        [0.0, 1.0],    # hit 2: low relevance (cos=0.0), orthogonal to hit 0
    ]
    hits = [_StubHit(0), _StubHit(1), _StubHit(2)]
    out = mmr_rerank(q, vecs, hits, top_k=2, lambda_=0.0)
    # First pick: hit 0 (seed by relevance); second pick: hit 2 (most dissimilar from hit 0)
    assert [h.id for h in out] == [0, 2]


# ---------------------------------------------------------------------------
# The headline case: MMR deprioritises near-duplicates of selected passages
# ---------------------------------------------------------------------------
def test_lambda_demotes_near_duplicate_of_seed():
    """5-passage setup: a near-duplicate of the top passage is demoted in
    favour of a diverse-but-slightly-less-relevant alternative.

    Cosine is direction-only, so to make ``sim(hit_i, hit_0) != rel(hit_i)``
    we point ``hit_0`` along a direction NOT aligned with the query axis.
    An exact duplicate of ``hit_0`` then has the same relevance as hit_0
    but maximal sim-to-selected (=1.0), while a diverse passage can have
    slightly lower relevance but much lower sim-to-selected.

    Hand-computed MMR scores at lambda=0.5 with hit 0 seeded::

        hit 1 (exact dup of 0):    0.5 * 0.95 - 0.5 * 1.0    = -0.025
        hit 2 (diverse, rel 0.75): 0.5 * 0.75 - 0.5 * 0.7125 =  0.019  <-- winner
        hit 3 (diverse, rel 0.72): 0.5 * 0.72 - 0.5 * 0.684  =  0.018
        hit 4 (low rel 0.10):      0.5 * 0.10 - 0.5 * 0.4054 = -0.153

    So MMR picks hit 2 second, NOT hit 1 (near-duplicate).

    Note on lambda: at the textbook default 0.7, the relevance term of hit 1
    (0.665) is large enough to barely beat hits 2/3 (0.311, 0.299) in this
    same setup. This test uses 0.5 deliberately so the diversity preference
    is unambiguous; lambda=0.7 in production needs LARGER relevance gaps
    between the duplicate and the diverse alternatives to flip the pick.
    """
    q = [1.0, 0.0, 0.0]
    vecs = [
        [0.95, 0.312, 0.0],   # hit 0: cos(q)=0.95, seed
        [0.95, 0.312, 0.0],   # hit 1: exact duplicate of hit 0 -> rel=0.95, sim(0)=1.0
        [0.75, 0.0, 0.661],   # hit 2: rel=0.75, sim(0)=0.7125 (diverse direction)
        [0.72, 0.0, 0.694],   # hit 3: rel=0.72, sim(0)=0.684  (diverse, slightly lower)
        [0.1, 0.995, 0.0],    # hit 4: rel=0.10, sim(0)=0.4054 (low relevance)
    ]
    hits = [_StubHit(i) for i in range(5)]
    out = mmr_rerank(q, vecs, hits, top_k=3, lambda_=0.5)
    assert out[0].id == 0
    # The near-duplicate (hit 1) must be demoted in favour of a diverse pick
    assert out[1].id in {2, 3}
    assert 1 not in [h.id for h in out[:2]]


# ---------------------------------------------------------------------------
# Tie-break determinism
# ---------------------------------------------------------------------------
def test_tie_break_prefers_lower_index():
    """Two passages with identical MMR score: the lower-indexed one wins
    (preserves the reranker's upstream ordering on ties).
    """
    q = [1.0, 0.0]
    # Hits 1 and 2 are IDENTICAL vectors -> identical relevance AND identical
    # sim-to-selected -> MMR score ties. Hit 1 should win (lower idx).
    vecs = [
        [1.0, 0.0],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    hits = [_StubHit(i) for i in range(3)]
    out = mmr_rerank(q, vecs, hits, top_k=2, lambda_=0.7)
    assert out[0].id == 0
    assert out[1].id == 1  # lower-indexed of the two identical candidates
