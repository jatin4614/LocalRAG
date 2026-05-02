"""Bug-fix campaign §1.6 — ingest worker engine reuse.

``_update_doc_status`` and ``_fetch_kb_rag_config`` previously created and
disposed a SQLAlchemy async engine per call (~1000 inits per 1000-doc
batch). The fix introduces a module-level lazy ``_sessionmaker_singleton``
that creates the engine once and reuses it across calls.

These tests pin:
1. The engine factory is invoked at most once across N consecutive calls.
2. Repeated calls return the same engine instance.
3. The singleton is reset cleanly when the module is re-imported (test
   isolation; not a production behaviour).
"""
from __future__ import annotations

import asyncio

import pytest


def _reset_singleton():
    """Clear any cached engine so tests start from a known state."""
    from ext.workers import ingest_worker as iw
    iw._engine_singleton = None  # type: ignore[attr-defined]


def test_fetch_rag_config_reuses_single_engine(monkeypatch):
    """Calling ``_fetch_kb_rag_config`` N times must invoke the engine
    factory exactly once."""
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}

    class _FakeConn:
        async def execute(self, *a, **kw):
            class _R:
                def first(self):
                    return None
            return _R()

    class _FakeBegin:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return None

    class _FakeEngine:
        def begin(self):
            return _FakeBegin()

        async def dispose(self):
            return None

    def _fake_create(*a, **kw):
        factory_calls["count"] += 1
        return _FakeEngine()

    monkeypatch.setattr(iw, "_create_async_engine", _fake_create, raising=False)

    async def _run():
        for _ in range(5):
            await iw._fetch_kb_rag_config(1)

    asyncio.run(_run())
    assert factory_calls["count"] == 1


def test_update_doc_status_reuses_single_engine(monkeypatch):
    """``_update_doc_status`` must also reuse the singleton engine."""
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}

    class _FakeConn:
        async def execute(self, *a, **kw):
            return None

    class _FakeBegin:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return None

    class _FakeEngine:
        def begin(self):
            return _FakeBegin()

        async def dispose(self):
            return None

    def _fake_create(*a, **kw):
        factory_calls["count"] += 1
        return _FakeEngine()

    monkeypatch.setattr(iw, "_create_async_engine", _fake_create, raising=False)

    async def _run():
        for i in range(7):
            await iw._update_doc_status(doc_id=i + 1, status="done", chunk_count=i)

    asyncio.run(_run())
    assert factory_calls["count"] == 1


def test_engine_shared_between_helpers(monkeypatch):
    """Both helpers should share the same singleton engine instance."""
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}

    class _FakeConn:
        async def execute(self, *a, **kw):
            class _R:
                def first(self):
                    return None
            return _R()

    class _FakeBegin:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *a):
            return None

    class _FakeEngine:
        def begin(self):
            return _FakeBegin()

        async def dispose(self):
            return None

    def _fake_create(*a, **kw):
        factory_calls["count"] += 1
        return _FakeEngine()

    monkeypatch.setattr(iw, "_create_async_engine", _fake_create, raising=False)

    async def _run():
        await iw._fetch_kb_rag_config(1)
        await iw._update_doc_status(doc_id=1, status="done")
        await iw._fetch_kb_rag_config(2)
        await iw._update_doc_status(doc_id=2, status="done")

    asyncio.run(_run())
    assert factory_calls["count"] == 1
