"""Unit tests for ``is_tenant=True`` payload-index wiring in VectorStore.

Qdrant 1.11+ treats keyword fields marked with ``is_tenant=True`` as the
tenant-partitioning boundary. We assert that ``ensure_collection`` registers
``kb_id`` / ``chat_id`` / ``owner_user_id`` with that flag and leaves
``subtag_id`` / ``doc_id`` / ``deleted`` as plain KEYWORD (within-tenant
filters — the tenant marker would only waste memory there).

No real Qdrant — ``AsyncQdrantClient`` is mocked.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http import models as qm

from ext.services.vector_store import _PAYLOAD_FIELDS, VectorStore


def _make_vs() -> VectorStore:
    vs = VectorStore.__new__(VectorStore)
    vs._client = MagicMock()
    vs._vector_size = 1024
    vs._distance = "Cosine"
    vs._known = set()
    vs._sparse_cache = {}
    return vs


@pytest.mark.asyncio
async def test_ensure_collection_marks_tenant_fields_with_is_tenant() -> None:
    """kb_id / chat_id / owner_user_id → KeywordIndexParams(is_tenant=True)."""
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_99")

    # One create_payload_index call per indexed field.
    calls = vs._client.create_payload_index.await_args_list
    # Map field_name → field_schema kwarg.
    by_field: dict[str, object] = {}
    for call in calls:
        kwargs = call.kwargs
        by_field[kwargs["field_name"]] = kwargs["field_schema"]

    # Tenant fields must carry is_tenant=True via KeywordIndexParams.
    for field in ("kb_id", "chat_id", "owner_user_id"):
        schema = by_field.get(field)
        assert isinstance(
            schema, qm.KeywordIndexParams
        ), f"{field} must use KeywordIndexParams, got {type(schema).__name__}"
        assert schema.is_tenant is True, f"{field} must set is_tenant=True"
        # Type field must be the keyword enum (or string 'keyword' — both valid).
        type_val = getattr(schema, "type", None)
        assert type_val in (qm.KeywordIndexType.KEYWORD, "keyword"), (
            f"{field} index type unexpected: {type_val!r}"
        )


@pytest.mark.asyncio
async def test_ensure_collection_keeps_filter_fields_plain_keyword() -> None:
    """subtag_id / doc_id / deleted → PayloadSchemaType.KEYWORD (no tenant flag)."""
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_99")

    by_field = {
        c.kwargs["field_name"]: c.kwargs["field_schema"]
        for c in vs._client.create_payload_index.await_args_list
    }

    for field in ("subtag_id", "doc_id", "deleted"):
        schema = by_field.get(field)
        assert schema == qm.PayloadSchemaType.KEYWORD, (
            f"{field} must stay plain KEYWORD, got {schema!r}"
        )


@pytest.mark.asyncio
async def test_ensure_collection_indexes_all_expected_fields() -> None:
    """Every field — tenant + filter — gets exactly one create_payload_index call."""
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    vs._client.create_payload_index = AsyncMock()

    await vs.ensure_collection("kb_99")

    fields = [c.kwargs["field_name"] for c in vs._client.create_payload_index.await_args_list]
    # Union of tenant + filter fields, exactly once each, no duplicates.
    expected = {"kb_id", "chat_id", "owner_user_id", "subtag_id", "doc_id", "deleted"}
    assert set(fields) == expected, f"unexpected fields indexed: {fields}"
    assert len(fields) == len(expected), f"duplicate indexes: {fields}"


def test_payload_fields_includes_owner_user_id() -> None:
    """P2.2 hook: owner_user_id must be in the with_payload allowlist so the
    tenant-index filter's target field is actually returned in hits."""
    assert "owner_user_id" in _PAYLOAD_FIELDS


@pytest.mark.asyncio
async def test_ensure_collection_swallows_duplicate_index_errors() -> None:
    """Qdrant raises on duplicate create_payload_index — we catch and continue.

    If we didn't, re-running ensure_collection on an existing collection would
    crash. This is the very first call that runs on every ingest, so it must
    be rock-solid idempotent.
    """
    vs = _make_vs()
    vs._client.create_collection = AsyncMock()
    # Every index call raises — mimics "already exists".
    vs._client.create_payload_index = AsyncMock(side_effect=RuntimeError("already exists"))

    # Must not raise.
    await vs.ensure_collection("kb_99")
    # All six fields were attempted even though each one "failed".
    assert vs._client.create_payload_index.await_count == 6
    # And the collection is now marked known → future calls short-circuit.
    assert "kb_99" in vs._known
