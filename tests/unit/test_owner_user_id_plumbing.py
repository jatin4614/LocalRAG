"""Unit tests for ``owner_user_id`` plumbing in ``VectorStore`` (P2.2).

The filter builder gains an optional ``owner_user_id`` parameter that adds a
``must`` field condition on the ``owner_user_id`` payload. The signature
keeps the default at ``None`` so that callers that don't pass the argument
produce byte-identical filters to pre-P2.2 — a hard requirement for the
legacy collections (kb_1/3/4/5) whose chunks were upserted before the
payload field existed.

No real Qdrant — ``AsyncQdrantClient`` is mocked throughout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http import models as qm

from ext.services.vector_store import VectorStore


def _make_vs() -> VectorStore:
    """Build a VectorStore without actually connecting to Qdrant."""
    vs = VectorStore.__new__(VectorStore)
    vs._client = MagicMock()
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    return vs


def _owner_conditions(flt: qm.Filter) -> list[qm.FieldCondition]:
    """Return every ``must`` FieldCondition whose key is ``owner_user_id``."""
    return [
        m for m in (flt.must or [])
        if getattr(m, "key", None) == "owner_user_id"
    ]


# ---------- _build_filter ----------------------------------------------------


def test_build_filter_without_owner_matches_legacy_shape() -> None:
    """Default (``owner_user_id=None``) produces no owner condition.

    This is the invariant that lets legacy chunks (no owner_user_id payload)
    still be retrieved: with no filter on the field, Qdrant returns the row.
    """
    flt = VectorStore._build_filter()
    assert _owner_conditions(flt) == [], "no owner condition expected when param omitted"


def test_build_filter_with_none_owner_matches_legacy_shape() -> None:
    """Explicit ``owner_user_id=None`` is identical to omitting the param."""
    flt = VectorStore._build_filter(owner_user_id=None)
    assert _owner_conditions(flt) == []


def test_build_filter_with_int_owner_adds_must_condition() -> None:
    """``owner_user_id=7`` → must-match ``owner_user_id == 7``."""
    flt = VectorStore._build_filter(owner_user_id=7)
    owner_conds = _owner_conditions(flt)
    assert len(owner_conds) == 1
    cond = owner_conds[0]
    assert isinstance(cond.match, qm.MatchValue)
    assert cond.match.value == 7
    assert isinstance(cond.match.value, int)


def test_build_filter_with_numeric_string_owner_coerces_to_int() -> None:
    """Defensive: ``owner_user_id="7"`` coerces to int 7 so a caller that
    stringifies user ids still matches points stamped with the integer.

    Mirrors ``ingest.py``'s ``doc_id`` coercion policy.
    """
    flt = VectorStore._build_filter(owner_user_id="7")
    owner_conds = _owner_conditions(flt)
    assert len(owner_conds) == 1
    assert owner_conds[0].match.value == 7
    assert isinstance(owner_conds[0].match.value, int)


def test_build_filter_with_uuid_string_owner_preserved_verbatim() -> None:
    """Non-numeric strings (UUIDs from upstream Open WebUI) pass through
    unchanged so production retrieval still matches the stored payload."""
    uuid_str = "11111111-2222-3333-4444-555555555555"
    flt = VectorStore._build_filter(owner_user_id=uuid_str)
    owner_conds = _owner_conditions(flt)
    assert len(owner_conds) == 1
    assert owner_conds[0].match.value == uuid_str
    assert isinstance(owner_conds[0].match.value, str)


def test_build_filter_combines_subtag_and_owner() -> None:
    """Both filters coexist — subtag + owner — without one clobbering the other."""
    flt = VectorStore._build_filter(subtag_ids=[1, 2], owner_user_id=7)
    keys = [getattr(m, "key", None) for m in (flt.must or [])]
    assert "subtag_id" in keys
    assert "owner_user_id" in keys


def test_build_filter_subtag_only_unchanged_from_pre_p22() -> None:
    """Legacy call site (``subtag_ids=[1]`` only) is byte-identical to pre-P2.2.

    No owner condition; deleted exclusion preserved.
    """
    flt = VectorStore._build_filter(subtag_ids=[1])
    assert _owner_conditions(flt) == []
    # Still excludes deleted — the other invariant we must not break.
    deleted_conds = [
        m for m in (flt.must_not or [])
        if getattr(m, "key", None) == "deleted"
    ]
    assert len(deleted_conds) == 1


def test_build_filter_always_excludes_deleted_regardless_of_owner() -> None:
    """Owner filter does not weaken the must_not=deleted exclusion."""
    flt = VectorStore._build_filter(owner_user_id=7)
    deleted_conds = [
        m for m in (flt.must_not or [])
        if getattr(m, "key", None) == "deleted"
    ]
    assert len(deleted_conds) == 1


# ---------- search(owner_user_id=...) ---------------------------------------


@pytest.mark.asyncio
async def test_search_without_owner_is_byte_identical_to_pre_p22() -> None:
    """Default call path (no ``owner_user_id``) must pass no owner condition
    to Qdrant — so legacy data that lacks the payload field still returns."""
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.search("kb_99", [0.0] * 1024, limit=5)

    _, kwargs = vs._client.query_points.call_args
    flt: qm.Filter = kwargs["query_filter"]
    assert _owner_conditions(flt) == []


@pytest.mark.asyncio
async def test_search_with_owner_passes_filter_to_qdrant() -> None:
    """``owner_user_id=7`` produces a must-match in the submitted filter."""
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.search("chat_42", [0.0] * 1024, limit=5, owner_user_id=7)

    _, kwargs = vs._client.query_points.call_args
    flt: qm.Filter = kwargs["query_filter"]
    owner_conds = _owner_conditions(flt)
    assert len(owner_conds) == 1
    assert owner_conds[0].match.value == 7


@pytest.mark.asyncio
async def test_search_with_uuid_owner_passes_filter_unchanged() -> None:
    """UUID owner ids survive the trip into Qdrant as strings."""
    vs = _make_vs()
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.search(
        "chat_42", [0.0] * 1024, limit=5,
        owner_user_id="abc-123-uuid",
    )

    _, kwargs = vs._client.query_points.call_args
    flt = kwargs["query_filter"]
    owner_conds = _owner_conditions(flt)
    assert len(owner_conds) == 1
    assert owner_conds[0].match.value == "abc-123-uuid"


# ---------- hybrid_search(owner_user_id=...) --------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_without_owner_matches_pre_p22(monkeypatch) -> None:
    """Default hybrid path (no owner) still emits no owner condition on
    any arm so that legacy hybrid collections (pre-P2.2) keep returning."""
    vs = _make_vs()
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.hybrid_search("kb_new", [0.1] * 1024, "q", limit=3)

    _, kwargs = vs._client.query_points.call_args
    outer = kwargs["query_filter"]
    assert _owner_conditions(outer) == []
    for arm in kwargs["prefetch"]:
        assert _owner_conditions(arm.filter) == []


@pytest.mark.asyncio
async def test_hybrid_search_with_owner_propagates_to_both_arms(monkeypatch) -> None:
    """Owner filter lands on both prefetch arms AND the outer fusion filter.

    If only the outer filter had it, Qdrant's RRF would still surface
    other-tenant candidates during prefetch — the outer filter merely
    trims the fused list. Putting it on each arm is the correct boundary.
    """
    vs = _make_vs()
    monkeypatch.setattr(
        "ext.services.sparse_embedder.embed_sparse_query",
        lambda t: ([1], [1.0]),
    )
    vs._client.query_points = AsyncMock(return_value=SimpleNamespace(points=[]))
    await vs.hybrid_search(
        "chat_42", [0.1] * 1024, "q", limit=3, owner_user_id=7,
    )

    _, kwargs = vs._client.query_points.call_args
    outer = kwargs["query_filter"]
    outer_conds = _owner_conditions(outer)
    assert len(outer_conds) == 1 and outer_conds[0].match.value == 7

    for i, arm in enumerate(kwargs["prefetch"]):
        arm_conds = _owner_conditions(arm.filter)
        assert len(arm_conds) == 1, f"prefetch arm {i} missing owner filter"
        assert arm_conds[0].match.value == 7
