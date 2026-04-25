"""Phase 3.4 — ColBERT multi-vector write path round-trip.

These tests stand up a real Qdrant container (via the ``clean_qdrant``
session-scoped fixture in ``conftest.py``) and exercise the full
write path:

1. ``ensure_collection(..., with_colbert=True)`` creates the collection
   with the named ``colbert`` slot wired to ``MultiVectorConfig(MAX_SIM)``.
2. ``upsert`` accepts ``colbert_vector: list[list[float]]`` per point
   and writes it under that slot alongside dense (and sparse if hybrid).
3. Qdrant accepts a query against the named slot and returns hits with
   meaningful MAX_SIM scores — proves the slot is wired correctly.

The tests use SYNTHETIC token-vectors (random-looking but deterministic
floats) instead of calling ``colbert_embed`` so that a missing /
not-yet-cached fastembed model doesn't gate the test. The real
``colbert_embed`` is exercised by ``tests/unit/test_colbert_embed.py``.

The read-side fusion (Task 3.5) will replace the bare ``query_points``
call here with a richer fused query — for now we just prove the slot
exists and round-trips.
"""
from __future__ import annotations

import pytest

from ext.services.vector_store import (
    VectorStore,
    _COLBERT_DIM,
    _COLBERT_NAME,
    _DENSE_NAME,
)

pytestmark = pytest.mark.integration


def _synthetic_colbert(seed_text: str, *, n_tokens: int = 4) -> list[list[float]]:
    """Build a deterministic ``n_tokens × _COLBERT_DIM`` float matrix.

    Hash-based + normalised so two calls with the same seed produce
    byte-identical multi-vectors. Avoids any dependency on the real
    ColBERT model — we just need *valid-shape* multi-vectors to prove
    Qdrant accepts the slot.
    """
    import hashlib
    import struct

    out: list[list[float]] = []
    for t in range(n_tokens):
        digest = hashlib.shake_128(
            f"{seed_text}|{t}".encode()
        ).digest(_COLBERT_DIM * 4)
        raw = struct.unpack(f"<{_COLBERT_DIM}i", digest)
        vec = [x / 2**31 for x in raw]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        out.append([x / norm for x in vec])
    return out


@pytest.mark.asyncio
async def test_collection_has_colbert_slot(clean_qdrant):
    """ensure_collection(with_colbert=True) creates the named colbert slot."""
    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_cb_1", with_colbert=True)

    # Local cache is set during ensure_collection.
    assert vs._collection_has_colbert("kb_cb_1") is True
    # Re-derive from Qdrant directly (clearing the cache) — proves the
    # slot is real on the server, not just optimistically cached.
    vs._colbert_cache.pop("kb_cb_1", None)
    assert await vs._refresh_colbert_cache("kb_cb_1") is True

    # Collections without the opt-in stay legacy / dense-only.
    await vs.ensure_collection("kb_legacy", with_colbert=False)
    assert vs._collection_has_colbert("kb_legacy") is False

    await vs.close()


@pytest.mark.asyncio
async def test_upsert_and_colbert_search_roundtrip(clean_qdrant):
    """Upsert points carrying colbert_vector, then query the colbert slot.

    Note: the colbert read path proper lands in Task 3.5 (vs.search will be
    extended to fuse the slots). For now we drop down to the raw
    qdrant-client to verify the slot exists + accepts queries — the goal of
    Task 3.4 is just the *write* path.
    """
    from qdrant_client import AsyncQdrantClient

    vs = VectorStore(url=clean_qdrant, vector_size=4)
    await vs.ensure_collection("kb_cb_2", with_colbert=True)

    pts = [
        {
            "id": 1,
            "vector": [1.0, 0.0, 0.0, 0.0],
            "colbert_vector": _synthetic_colbert("alpha", n_tokens=3),
            "payload": {"text": "alpha"},
        },
        {
            "id": 2,
            "vector": [0.0, 1.0, 0.0, 0.0],
            "colbert_vector": _synthetic_colbert("beta", n_tokens=5),
            "payload": {"text": "beta"},
        },
    ]
    await vs.upsert("kb_cb_2", pts)

    # Verify both slots round-trip via the raw client. Task 3.4's job is the
    # write path; Task 3.5 will teach ``vs.search`` to route via the named
    # slots and fuse them. For now, dropping to the raw client keeps this
    # test focused on the contract that matters here: the points landed in
    # both slots (dense AND colbert) and Qdrant accepts the multi-vector
    # query.
    raw = AsyncQdrantClient(url=clean_qdrant)
    try:
        # Dense slot still works — query named ``dense`` with a
        # query that matches alpha's vector exactly.
        dense_resp = await raw.query_points(
            collection_name="kb_cb_2",
            query=[1.0, 0.0, 0.0, 0.0],
            using=_DENSE_NAME,
            limit=2,
            with_payload=True,
        )
        assert len(dense_resp.points) == 2
        assert dense_resp.points[0].payload["text"] == "alpha"

        # Now query the colbert slot. Use alpha's own multi-vector — under
        # MAX_SIM this gives perfect per-token matches against itself, so
        # alpha must rank #1 against beta (different seed → orthogonal).
        alpha_tokens = _synthetic_colbert("alpha", n_tokens=3)
        cb_resp = await raw.query_points(
            collection_name="kb_cb_2",
            query=alpha_tokens,
            using=_COLBERT_NAME,
            limit=2,
            with_payload=True,
        )
        cb_hits = cb_resp.points
        assert len(cb_hits) == 2
        assert cb_hits[0].payload["text"] == "alpha"
        assert cb_hits[0].score > cb_hits[1].score
    finally:
        await raw.close()

    await vs.close()
