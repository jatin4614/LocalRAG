"""Integration tests for ``VectorStore.health_check`` (Phase 1.3).

The integration suite already provides a session-scoped ``qdrant`` fixture
in ``tests/integration/conftest.py`` that yields the URL of a Qdrant
container started via testcontainers. We reuse it directly — the plan
called the fixture ``qdrant_url``, so the integration tests below take
the URL as a positional fixture argument named ``qdrant``.
"""
from __future__ import annotations

import asyncio

import pytest

from ext.services.vector_store import VectorStore

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_health_check_returns_true_when_qdrant_is_up(qdrant):
    vs = VectorStore(url=qdrant, vector_size=1024)
    try:
        assert await vs.health_check() is True
    finally:
        await vs.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_when_qdrant_is_unreachable():
    # Bogus port → must return False within timeout, not hang. ``vector_size``
    # is irrelevant here because no collection is touched.
    vs = VectorStore(url="http://localhost:65535", vector_size=1024)
    try:
        result = await asyncio.wait_for(vs.health_check(), timeout=5.0)
        assert result is False
    finally:
        await vs.close()


@pytest.mark.asyncio
async def test_health_check_is_cached_for_5_seconds(qdrant):
    """First probe hits Qdrant and caches the result; second probe within
    5s returns the cached True even after we flip ``_url`` to a known-bad
    address. This proves the cache is honored, not just that the second
    call happened to succeed."""
    vs = VectorStore(url=qdrant, vector_size=1024)
    try:
        assert await vs.health_check() is True
        # Swap the underlying URL to one that would normally fail. If the
        # cache works, the second call still returns True without probing.
        vs._url = "http://localhost:65535"
        assert await vs.health_check() is True
    finally:
        await vs.close()
