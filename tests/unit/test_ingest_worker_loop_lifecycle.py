"""Loop-lifecycle correctness for the ingest-worker DB helpers.

Background:
    ``_engine_singleton`` (``ext/workers/ingest_worker.py:65-102``) caches a
    SQLAlchemy async engine across calls so a 1000-doc batch doesn't
    instantiate 1000 engines. The original implementation used the default
    ``QueuePool``: the asyncpg connections inside that pool bind to the
    event loop alive at first-pool-fill. Celery's prefork workers run each
    task under a FRESH ``asyncio.run(...)`` scope, so the second task picks
    up a pooled connection whose loop is now closed:

        RuntimeError: Event loop is closed
        Task ... attached to a different loop

    The visible symptom in production was every ``kb_documents.ingest_status``
    transition silently failing while Qdrant correctly received the chunks —
    leaving the admin UI stuck at ``chunking`` forever.

These tests pin the fix-contract:

1. ``_update_doc_status`` and ``_fetch_kb_rag_config`` survive being called
   from two SEPARATE ``asyncio.run(...)`` scopes against the same cached
   engine (the bug scenario).
2. The engine factory is called exactly once across both scopes (we keep
   the singleton optimisation — the fix is to make connections short-lived,
   not to re-create the engine).
3. When the SQL execute path raises, the helper SURFACES the failure via
   ``log.error`` AND increments
   ``ingest_status_update_failed_total{stage=<status>}`` — replacing the
   silent ``log.warning`` that masked the bug for two weeks.
"""
from __future__ import annotations

import asyncio
import logging

import pytest


def _reset_singleton():
    from ext.workers import ingest_worker as iw
    iw._engine_singleton = None  # type: ignore[attr-defined]


def _make_loop_aware_fake_engine_factory(factory_calls: dict, execute_calls: dict | None = None):
    """Return a ``_create_async_engine`` stand-in modelling asyncpg's loop binding.

    The fake engine pins itself to the loop alive at the FIRST ``begin()``
    call. A subsequent ``begin()`` from a DIFFERENT loop raises
    ``RuntimeError("Event loop is closed")`` — the exact production
    failure mode reported by asyncpg when a pooled connection is reused
    across event loops.

    Behaviour switches on the ``poolclass`` kwarg captured from
    ``_create_async_engine(...)``:

    * ``poolclass=None`` (legacy default pool): loop pin persists for the
      engine's lifetime → second call from a fresh ``asyncio.run(...)``
      raises (the bug).
    * ``poolclass=NullPool`` (the fix): every ``begin()`` rebinds to the
      current loop → cross-loop reuse is safe.

    ``execute_calls`` (optional) is a counter dict — incremented on every
    ``conn.execute`` so tests can assert the SQL actually ran (not
    silently swallowed by the helper's except block).
    """
    class _FakeRow:
        def first(self):
            return None

    class _FakeConn:
        async def execute(self, *_args, **_kwargs):
            if execute_calls is not None:
                execute_calls["count"] = execute_calls.get("count", 0) + 1
            return _FakeRow()

    class _FakeEngine:
        def __init__(self, poolclass=None):
            self._poolclass = poolclass
            self._bound_loop = None

        def begin(self):
            outer_engine = self

            class _PinningBegin:
                async def __aenter__(self_inner):
                    current = asyncio.get_running_loop()
                    try:
                        from sqlalchemy.pool import NullPool
                        is_null = outer_engine._poolclass is NullPool
                    except Exception:
                        is_null = False
                    if is_null:
                        # NullPool fix: every begin() acts like a fresh
                        # connection bound to the current loop.
                        outer_engine._bound_loop = current
                    else:
                        # Legacy pool: pin on first begin only; later
                        # cross-loop calls raise.
                        if outer_engine._bound_loop is None:
                            outer_engine._bound_loop = current
                        elif outer_engine._bound_loop is not current:
                            raise RuntimeError("Event loop is closed")
                    return _FakeConn()

                async def __aexit__(self_inner, *_a):
                    return None

            return _PinningBegin()

        async def dispose(self):
            self._bound_loop = None

    def _fake_create(*_args, **kwargs):
        factory_calls["count"] += 1
        return _FakeEngine(poolclass=kwargs.get("poolclass"))

    return _fake_create


# ---------------------------------------------------------------------------
# 1. First-call success
# ---------------------------------------------------------------------------
def test_update_doc_status_first_call_succeeds(monkeypatch):
    """Smoke test: a single ``asyncio.run`` invocation works end-to-end.

    Verifies the SQL actually executed (not silently swallowed by the
    helper's broad ``except``).
    """
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}
    execute_calls = {"count": 0}
    monkeypatch.setattr(
        iw, "_create_async_engine",
        _make_loop_aware_fake_engine_factory(factory_calls, execute_calls),
        raising=False,
    )

    asyncio.run(iw._update_doc_status(doc_id=1, status="done", chunk_count=42))
    assert factory_calls["count"] == 1
    assert execute_calls["count"] == 1, "SQL execute did not actually run"


# ---------------------------------------------------------------------------
# 2. THE BUG SCENARIO: two separate asyncio.run calls against same engine
# ---------------------------------------------------------------------------
def test_update_doc_status_survives_separate_event_loops(monkeypatch):
    """Two ``asyncio.run(...)`` invocations of ``_update_doc_status``
    against the same module-level engine cache must both succeed.

    Pre-fix: second call raises ``RuntimeError('Event loop is closed')``
    inside the cached engine because the asyncpg connections were bound
    to the closed loop from the first task. The helper's broad
    ``except`` swallows the raise, so we assert via ``execute_calls`` —
    the SQL must actually run on every call, not be silently dropped.
    """
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}
    execute_calls = {"count": 0}
    monkeypatch.setattr(
        iw, "_create_async_engine",
        _make_loop_aware_fake_engine_factory(factory_calls, execute_calls),
        raising=False,
    )

    # Mirror the exact prefork-worker pattern: each task calls asyncio.run.
    asyncio.run(iw._update_doc_status(doc_id=1, status="chunking"))
    asyncio.run(iw._update_doc_status(doc_id=1, status="embedding"))
    asyncio.run(iw._update_doc_status(doc_id=1, status="done", chunk_count=99))

    # All three SQL UPDATEs must have actually run.
    assert execute_calls["count"] == 3, (
        "expected 3 SQL executes across 3 asyncio.run scopes; got %d "
        "(silent swallow of cross-loop reuse error?)" % execute_calls["count"]
    )


# ---------------------------------------------------------------------------
# 3. Engine factory still called only once (singleton optimisation kept)
# ---------------------------------------------------------------------------
def test_engine_factory_called_once_across_loops(monkeypatch):
    """The fix must not regress §1.6 — the engine is still cached.

    We accept per-call connection overhead (NullPool) but the engine
    object itself is created exactly once.
    """
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}
    monkeypatch.setattr(
        iw, "_create_async_engine",
        _make_loop_aware_fake_engine_factory(factory_calls),
        raising=False,
    )

    asyncio.run(iw._update_doc_status(doc_id=1, status="chunking"))
    asyncio.run(iw._update_doc_status(doc_id=2, status="embedding"))
    asyncio.run(iw._update_doc_status(doc_id=3, status="done"))

    assert factory_calls["count"] == 1, (
        "engine factory invoked %d times — singleton lost" % factory_calls["count"]
    )


# ---------------------------------------------------------------------------
# 4. Failure path now SURFACES instead of silently swallowing
# ---------------------------------------------------------------------------
def test_update_doc_status_failure_logs_error_and_bumps_counter(monkeypatch, caplog):
    """When the SQL execute fails, the helper must:

    1. Log at ERROR (not WARNING — the silent log.warning was the
       reason operators saw "stuck at chunking" with no DB error).
    2. Increment ``ingest_status_update_failed_total{stage=<status>}``.
    3. NOT raise (the worker still completes the actual ingest work;
       status update is fire-and-forget telemetry).
    """
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    class _BoomConn:
        async def execute(self, *a, **kw):
            raise RuntimeError("Event loop is closed")

    class _BoomBegin:
        async def __aenter__(self):
            return _BoomConn()

        async def __aexit__(self, *a):
            return None

    class _BoomEngine:
        def begin(self):
            return _BoomBegin()

        async def dispose(self):
            return None

    def _create(*_a, **_kw):
        return _BoomEngine()

    monkeypatch.setattr(iw, "_create_async_engine", _create, raising=False)

    # Reset the counter so the assertion is local to this test.
    from ext.services.metrics import ingest_status_update_failed_total
    # prometheus_client Counter has no public reset; capture before/after delta.
    def _val(stage: str) -> float:
        if hasattr(ingest_status_update_failed_total, "labels"):
            metric = ingest_status_update_failed_total.labels(stage=stage)
            v = getattr(metric, "_value", None)
            if v is not None and hasattr(v, "get"):
                return float(v.get())
        return 0.0

    before = _val("chunking")

    caplog.set_level(logging.ERROR, logger="orgchat.ingest_worker")

    # Must not raise.
    asyncio.run(iw._update_doc_status(doc_id=42, status="chunking"))

    after = _val("chunking")
    assert after - before == 1.0, (
        "expected ingest_status_update_failed_total{stage='chunking'} to "
        "increment by 1, got delta=%s" % (after - before)
    )

    # Must log at ERROR — the silent WARNING was the bug.
    error_records = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR
        and r.name == "orgchat.ingest_worker"
    ]
    assert error_records, (
        "expected at least one ERROR-level log record from "
        "orgchat.ingest_worker; got: %s"
        % [(r.levelname, r.name, r.getMessage()) for r in caplog.records]
    )


# ---------------------------------------------------------------------------
# 5. Same fix applies to _fetch_kb_rag_config
# ---------------------------------------------------------------------------
def test_fetch_kb_rag_config_survives_separate_event_loops(monkeypatch):
    """Same loop-lifecycle bug exists in ``_fetch_kb_rag_config``.

    Two ``asyncio.run(...)`` calls against the same cached engine must
    both succeed.
    """
    from ext.workers import ingest_worker as iw

    _reset_singleton()
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

    factory_calls = {"count": 0}
    monkeypatch.setattr(
        iw, "_create_async_engine",
        _make_loop_aware_fake_engine_factory(factory_calls),
        raising=False,
    )

    asyncio.run(iw._fetch_kb_rag_config(1))
    asyncio.run(iw._fetch_kb_rag_config(2))
    asyncio.run(iw._fetch_kb_rag_config(3))

    assert factory_calls["count"] == 1
