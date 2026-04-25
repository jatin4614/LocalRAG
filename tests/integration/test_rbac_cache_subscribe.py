"""Integration test for the RBAC pub/sub subscriber wiring.

Phase 1.5 follow-up: ``ext.services.rbac_cache.subscribe_invalidations`` is
a long-running coroutine that drops local cache entries when another
replica publishes a ``rbac:invalidate`` event. ``ext.app`` wires it into
the FastAPI lifespan (in ``build_app``) and a startup event on the rag
router (in ``build_ext_routers``) so it actually runs on every replica.

This test verifies the subscriber is started — without it, multi-replica
deploys leak revoked KB grants for up to ``RAG_RBAC_CACHE_TTL_SECS``
(default 30s) before the TTL safety net catches them.

We use a fake redis with an awaitable ``pubsub().subscribe()`` call and
assert that the subscriber actually issued the SUBSCRIBE.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


class _FakePubsub:
    """Just enough to satisfy the subscriber: subscribe + listen()."""

    def __init__(self) -> None:
        self.subscribed_channels: list[str] = []
        self._messages: asyncio.Queue = asyncio.Queue()

    async def subscribe(self, channel: str) -> None:
        self.subscribed_channels.append(channel)

    def listen(self):
        async def _it():
            # Park forever so the subscribe call is observable but the
            # task doesn't churn. The lifespan teardown will cancel it.
            while True:
                msg = await self._messages.get()
                yield msg
        return _it()


class _FakeRedis:
    def __init__(self) -> None:
        self.pubsub_obj = _FakePubsub()

    def pubsub(self) -> _FakePubsub:
        return self.pubsub_obj

    async def get(self, key: str):  # pragma: no cover - not exercised
        return None

    async def setex(self, *_a, **_kw):  # pragma: no cover
        return None

    async def delete(self, *_keys):  # pragma: no cover
        return None

    async def publish(self, *_a, **_kw):  # pragma: no cover
        return None

    async def aclose(self):  # pragma: no cover
        return None


@pytest.mark.asyncio
async def test_subscribe_invalidations_runs_at_lifespan_startup(monkeypatch):
    """build_app's lifespan starts the RBAC subscriber, which must
    issue SUBSCRIBE on ``rbac:invalidate``.
    """
    fake_redis = _FakeRedis()

    # Stub out _redis_client so the lifespan grabs our fake instead of
    # opening a real connection. The lifespan resolves the import at
    # entry, so monkeypatching the module attribute is enough.
    from ext.services import chat_rag_bridge

    monkeypatch.setattr(chat_rag_bridge, "_redis_client", lambda: fake_redis)

    # Set required env so build_app can boot. We need a connectable DB
    # URL — use an SQLite in-memory placeholder; the test only exercises
    # the lifespan, not real DB queries.
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("TEI_URL", "http://localhost:8080")
    monkeypatch.setenv("RAG_VECTOR_SIZE", "1024")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SESSION_SECRET", "x" * 32)

    from ext.app import build_app

    app = build_app()
    # Drive the lifespan via TestClient (sync context — async lifespan
    # is run on an internal loop).
    with TestClient(app) as _client:
        # Give the create_task a tick to actually call subscribe().
        # The subscriber awaits subscribe() before entering the listen
        # loop, so a zero-sleep yield is enough.
        for _ in range(20):
            if fake_redis.pubsub_obj.subscribed_channels:
                break
            await asyncio.sleep(0.05)

    assert fake_redis.pubsub_obj.subscribed_channels == ["rbac:invalidate"], (
        "subscribe_invalidations must run at lifespan startup and "
        "subscribe to the rbac:invalidate channel — without this the "
        "RBAC cache leaks revoked grants on multi-replica deploys."
    )
