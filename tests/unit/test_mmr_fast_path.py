"""Tests for the §5.10 MMR fast-path (one-shot pre-normalize + np.dot loop).

The contract:

* MMR output is unchanged within float tolerance vs the pre-fix cosine
  implementation (`_cosine`).
* `mmr_rerank_from_hits` pre-normalizes the embedded query + passages once
  and asserts (with a warning) that TEI returned unit-norm vectors.
* The hot loop inside `mmr_rerank` uses `np.dot` on the pre-normalized
  vectors instead of recomputing `np.linalg.norm` per pair.
* Backward compat: `_cosine` is still exported and callable (other modules
  may import it), it just isn't used in the MMR hot loop.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ext.services.mmr import _cosine, mmr_rerank, mmr_rerank_from_hits


class _StubHit:
    def __init__(self, id: int, text: str = "") -> None:
        self.id = id
        self.payload = {"text": text}


# ---------------------------------------------------------------------------
# _cosine still exported / works (backward compat for callers)
# ---------------------------------------------------------------------------
def test_cosine_still_exported_and_correct():
    """`_cosine` must stay importable and return correct values; only the
    MMR hot loop is allowed to skip it.
    """
    s = 1.0 / math.sqrt(2.0)
    assert _cosine([1.0, 0.0], [s, s]) == pytest.approx(s)
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero-vec guard


# ---------------------------------------------------------------------------
# mmr_rerank produces identical output to the old _cosine-based path
# ---------------------------------------------------------------------------
def _slow_mmr(query_vec, passage_vecs, hits, *, top_k, lambda_):
    """Reference impl using the original Python `_cosine` everywhere — the
    pre-fix behaviour.
    """
    if not hits:
        return []
    n = len(hits)
    if top_k >= n:
        return list(hits)
    relevance = [_cosine(query_vec, passage_vecs[i]) for i in range(n)]
    remaining = set(range(n))
    first = max(remaining, key=lambda i: (relevance[i], -i))
    selected = [first]
    remaining.remove(first)
    while len(selected) < top_k and remaining:
        best_i = -1
        best_score = -float("inf")
        for i in sorted(remaining):
            max_sim = max(_cosine(passage_vecs[i], passage_vecs[s]) for s in selected)
            score = lambda_ * relevance[i] - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return [hits[i] for i in selected]


def test_mmr_output_unchanged_within_tolerance_unit_vectors():
    """For L2-normalized vectors (TEI's actual output) the new and old
    implementations must agree within float tolerance.
    """
    rng = np.random.default_rng(seed=42)
    n, dim = 12, 32
    raw = rng.standard_normal((n + 1, dim))
    # Pre-normalize: simulate TEI's L2-normalized output
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    q = raw[0].tolist()
    pvecs = [raw[i].tolist() for i in range(1, n + 1)]
    hits = [_StubHit(i) for i in range(n)]

    out_new = mmr_rerank(q, pvecs, hits, top_k=5, lambda_=0.7)
    out_old = _slow_mmr(q, pvecs, hits, top_k=5, lambda_=0.7)
    assert [h.id for h in out_new] == [h.id for h in out_old]


def test_mmr_output_unchanged_with_non_unit_vectors_too():
    """Even when caller passes non-normalized vectors directly to the pure
    function, behaviour must match the old _cosine path (the pure function
    handles its own normalization for safety).
    """
    rng = np.random.default_rng(seed=7)
    n, dim = 8, 16
    raw = rng.standard_normal((n + 1, dim)) * 5.0  # explicitly non-unit
    q = raw[0].tolist()
    pvecs = [raw[i].tolist() for i in range(1, n + 1)]
    hits = [_StubHit(i) for i in range(n)]

    out_new = mmr_rerank(q, pvecs, hits, top_k=4, lambda_=0.5)
    out_old = _slow_mmr(q, pvecs, hits, top_k=4, lambda_=0.5)
    assert [h.id for h in out_new] == [h.id for h in out_old]


# ---------------------------------------------------------------------------
# mmr_rerank_from_hits: warns when TEI returns non-unit vectors
# ---------------------------------------------------------------------------
class _NonUnitEmbedder:
    """Returns vectors with norm != 1.0 to trigger the sanity warning."""

    def __init__(self):
        self.call_count = 0

    async def embed(self, texts):
        self.call_count += 1
        # Each text gets a vector with norm = 5.0
        return [[5.0, 0.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_non_unit_embedder_does_not_crash(caplog):
    """Pre-normalize is one-shot and resilient: even if TEI ever returned
    non-unit vectors (it shouldn't), MMR must still work — we drop a
    warning and proceed with the renormalized vectors.
    """
    hits = [_StubHit(0, "a"), _StubHit(1, "b"), _StubHit(2, "c")]
    embedder = _NonUnitEmbedder()
    out = await mmr_rerank_from_hits("q", hits, embedder, top_k=2)
    assert len(out) == 2
    assert embedder.call_count == 1


# ---------------------------------------------------------------------------
# Existing edge cases stay green (re-asserted here for completeness)
# ---------------------------------------------------------------------------
def test_empty_hits_returns_empty_unchanged():
    assert mmr_rerank([1.0, 0.0], [], [], top_k=5) == []


def test_top_k_ge_len_short_circuits():
    hits = [_StubHit(0), _StubHit(1)]
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    assert [h.id for h in mmr_rerank([1.0, 0.0], vecs, hits, top_k=10)] == [0, 1]
