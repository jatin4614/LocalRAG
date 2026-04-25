"""Unit tests for ``ext.services.ood_signal``.

The cosine math + centroid plumbing are tested against mock Qdrant
responses. No live Qdrant / TEI needed.
"""
from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from typing import Any, Iterable

import pytest

from ext.services import ood_signal


# ---------------------------------------------------------------------------
# Pure cosine math
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors() -> None:
    """cos(v, v) == 1 for any non-zero vector."""
    v = [0.3, 0.4, 0.5]
    assert math.isclose(ood_signal._cosine(v, v), 1.0, abs_tol=1e-9)


def test_cosine_opposite_vectors() -> None:
    v = [1.0, 0.0, 0.0]
    w = [-1.0, 0.0, 0.0]
    assert math.isclose(ood_signal._cosine(v, w), -1.0, abs_tol=1e-9)


def test_cosine_orthogonal() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert math.isclose(ood_signal._cosine(a, b), 0.0, abs_tol=1e-9)


def test_cosine_known_value() -> None:
    """Classic textbook example: cos(60°) = 0.5."""
    a = [1.0, 0.0]
    b = [0.5, math.sqrt(3) / 2]
    # cos between is 0.5 regardless of b's norm.
    assert math.isclose(ood_signal._cosine(a, b), 0.5, abs_tol=1e-9)


def test_cosine_empty_returns_one() -> None:
    """Degenerate inputs must return 1.0 (in-domain by default) rather than raise."""
    assert ood_signal._cosine([], [1.0]) == 1.0
    assert ood_signal._cosine([1.0], []) == 1.0
    assert ood_signal._cosine([], []) == 1.0


def test_cosine_shape_mismatch_returns_one() -> None:
    """A query vector of different dim than the centroid → in-domain default."""
    assert ood_signal._cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 1.0


def test_cosine_zero_vector_returns_one() -> None:
    """All-zero denominator must not divide by zero."""
    assert ood_signal._cosine([0.0, 0.0], [1.0, 1.0]) == 1.0


def test_normalize_unit_length() -> None:
    out = ood_signal._normalize([3.0, 4.0])
    norm = math.sqrt(sum(x * x for x in out))
    assert math.isclose(norm, 1.0, abs_tol=1e-9)
    assert math.isclose(out[0], 0.6, abs_tol=1e-9)
    assert math.isclose(out[1], 0.8, abs_tol=1e-9)


def test_normalize_zero_vector_passthrough() -> None:
    out = ood_signal._normalize([0.0, 0.0, 0.0])
    assert out == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Centroid plumbing — uses a mock qdrant client
# ---------------------------------------------------------------------------

class _MockQdrant:
    """Minimal mock exposing ``scroll`` compatible with AsyncQdrantClient."""

    def __init__(self, points: Iterable[Any]) -> None:
        self._points = list(points)
        self.calls = 0

    async def scroll(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: Any,
        with_payload: Any,
        with_vectors: bool,
    ) -> tuple[list[Any], Any]:
        self.calls += 1
        # Deliver everything in the first call, then empty.
        if self.calls == 1:
            return (list(self._points), None)
        return ([], None)

    async def close(self) -> None:
        pass


def _pt(vec: list[float]) -> Any:
    return SimpleNamespace(vector=vec, payload=None)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    ood_signal.clear_cache()


def test_compute_ood_score_in_domain() -> None:
    """A query identical to the centroid returns ~1.0."""
    # 3 unit-vectors along +x → centroid is +x.
    client = _MockQdrant([_pt([1.0, 0.0, 0.0])] * 3)
    score = asyncio.run(
        ood_signal.compute_ood_score([1.0, 0.0, 0.0], kb_id=1, qdrant_client=client)
    )
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_compute_ood_score_ood() -> None:
    """A query orthogonal to the centroid returns ~0.0."""
    client = _MockQdrant([_pt([1.0, 0.0, 0.0])] * 3)
    score = asyncio.run(
        ood_signal.compute_ood_score([0.0, 1.0, 0.0], kb_id=2, qdrant_client=client)
    )
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_compute_ood_score_empty_kb_defaults_in_domain() -> None:
    """An empty collection → 1.0 (in-domain). Never blocks a fresh KB."""
    client = _MockQdrant([])
    score = asyncio.run(
        ood_signal.compute_ood_score([1.0, 0.0, 0.0], kb_id=3, qdrant_client=client)
    )
    assert score == 1.0


def test_compute_ood_score_caches_centroid() -> None:
    """Second call with the same kb_id must not re-scroll Qdrant."""
    client = _MockQdrant([_pt([1.0, 0.0, 0.0])] * 3)
    asyncio.run(
        ood_signal.compute_ood_score([1.0, 0.0, 0.0], kb_id=42, qdrant_client=client)
    )
    calls_after_first = client.calls
    asyncio.run(
        ood_signal.compute_ood_score([0.5, 0.5, 0.0], kb_id=42, qdrant_client=client)
    )
    assert client.calls == calls_after_first, "centroid should have been cached"


def test_compute_ood_score_handles_named_vectors() -> None:
    """Hybrid collections wrap vectors in a dict; extractor pulls 'dense'."""
    pt = SimpleNamespace(vector={"dense": [1.0, 0.0], "bm25": "ignored"}, payload=None)
    client = _MockQdrant([pt] * 3)
    score = asyncio.run(
        ood_signal.compute_ood_score([1.0, 0.0], kb_id=7, qdrant_client=client)
    )
    assert math.isclose(score, 1.0, abs_tol=1e-9)


def test_compute_ood_score_empty_query_returns_one() -> None:
    """An empty query vector never blocks retrieval."""
    client = _MockQdrant([_pt([1.0, 0.0])] * 3)
    score = asyncio.run(
        ood_signal.compute_ood_score([], kb_id=9, qdrant_client=client)
    )
    assert score == 1.0
