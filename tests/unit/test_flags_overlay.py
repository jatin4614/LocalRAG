"""P3.0 — ``ext.services.flags`` contextvar-backed overlay on os.environ.

Key invariants:

1. ``flags.get(key, default)`` behaves like ``os.environ.get`` when no
   overlay is active.
2. Inside ``with_overrides({...})`` the overlay wins; on exit, the
   overlay is restored to the prior state.
3. Nested ``with_overrides`` blocks compose (inner extends outer).
4. Contextvar scoping: an overlay set in one asyncio task does NOT leak
   into another concurrent task that never entered the block.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from ext.services import flags


def test_get_falls_back_to_os_environ_when_no_overlay(monkeypatch):
    monkeypatch.setenv("RAG_UNIT_TEST_FLAG", "from-env")
    assert flags.get("RAG_UNIT_TEST_FLAG", "default") == "from-env"


def test_get_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("RAG_UNIT_TEST_FLAG", raising=False)
    assert flags.get("RAG_UNIT_TEST_FLAG", "default") == "default"


def test_overlay_overrides_os_environ(monkeypatch):
    monkeypatch.setenv("RAG_UNIT_TEST_FLAG", "from-env")
    with flags.with_overrides({"RAG_UNIT_TEST_FLAG": "overridden"}):
        assert flags.get("RAG_UNIT_TEST_FLAG", "default") == "overridden"
    # On exit, falls back to env again.
    assert flags.get("RAG_UNIT_TEST_FLAG", "default") == "from-env"


def test_overlay_does_not_mutate_os_environ():
    """Critical invariant — per-request overlays must NOT leak into the
    process-level ``os.environ`` because that would leak across workers
    and concurrent requests."""
    with flags.with_overrides({"RAG_TEST_NON_LEAK": "1"}):
        assert "RAG_TEST_NON_LEAK" not in os.environ
    assert "RAG_TEST_NON_LEAK" not in os.environ


def test_overlay_restores_prior_state_on_exit(monkeypatch):
    monkeypatch.delenv("RAG_TEST_RESTORE", raising=False)
    assert flags.get("RAG_TEST_RESTORE") is None
    with flags.with_overrides({"RAG_TEST_RESTORE": "active"}):
        assert flags.get("RAG_TEST_RESTORE") == "active"
    assert flags.get("RAG_TEST_RESTORE") is None


def test_nested_overlays_compose(monkeypatch):
    monkeypatch.delenv("RAG_TEST_OUTER", raising=False)
    monkeypatch.delenv("RAG_TEST_INNER", raising=False)
    with flags.with_overrides({"RAG_TEST_OUTER": "outer", "RAG_TEST_INNER": "from-outer"}):
        assert flags.get("RAG_TEST_OUTER") == "outer"
        assert flags.get("RAG_TEST_INNER") == "from-outer"
        with flags.with_overrides({"RAG_TEST_INNER": "from-inner"}):
            # Inner wins for keys it sets.
            assert flags.get("RAG_TEST_INNER") == "from-inner"
            # Outer's keys still visible.
            assert flags.get("RAG_TEST_OUTER") == "outer"
        # Inner reverted.
        assert flags.get("RAG_TEST_INNER") == "from-outer"
    # Outer reverted.
    assert flags.get("RAG_TEST_OUTER") is None


def test_nested_overlays_inner_only_adds_new_key():
    """Inner block only provides a NEW key; outer keys should still win."""
    with flags.with_overrides({"RAG_TEST_A": "outer-a"}):
        with flags.with_overrides({"RAG_TEST_B": "inner-b"}):
            assert flags.get("RAG_TEST_A") == "outer-a"
            assert flags.get("RAG_TEST_B") == "inner-b"


def test_empty_overrides_is_no_op(monkeypatch):
    """Passing ``{}`` should not mutate or mask anything."""
    monkeypatch.setenv("RAG_TEST_EMPTY", "env-value")
    with flags.with_overrides({}):
        assert flags.get("RAG_TEST_EMPTY") == "env-value"


def test_overrides_stringifies_non_string_values():
    """Values passed as ints/floats are coerced to strings so flags.get
    can compare with ``== "1"`` exactly as os.environ.get does."""
    with flags.with_overrides({"RAG_TEST_NUM": 1}):
        assert flags.get("RAG_TEST_NUM") == "1"
    with flags.with_overrides({"RAG_TEST_FLOAT": 0.75}):
        # repr(0.75) == "0.75", str(0.75) == "0.75"
        assert flags.get("RAG_TEST_FLOAT") == "0.75"


# ---------------------------------------------------------------------------
# Concurrent task isolation — the critical correctness property.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overlay_does_not_leak_to_concurrent_task():
    """Two asyncio tasks — one sets the overlay, the other must NOT see
    the override. This is the whole point of using contextvars instead
    of a threadlocal or module global."""
    other_task_saw: list[str | None] = []
    started_other = asyncio.Event()
    done_main = asyncio.Event()

    async def other_task():
        started_other.set()
        # Wait until main enters its with_overrides block, then read flag.
        await asyncio.sleep(0.01)
        other_task_saw.append(flags.get("RAG_TEST_CONCURRENT"))
        done_main.set()

    async def main():
        await started_other.wait()
        with flags.with_overrides({"RAG_TEST_CONCURRENT": "only-main"}):
            # Inside the block, main sees the override.
            assert flags.get("RAG_TEST_CONCURRENT") == "only-main"
            await done_main.wait()

    await asyncio.gather(main(), other_task())
    # The other task, which never entered a with_overrides block, must
    # see None — not "only-main".
    assert other_task_saw == [None]


@pytest.mark.asyncio
async def test_overlay_propagates_into_spawned_child_task():
    """contextvars semantics: asyncio.create_task copies the current
    Context, so a child spawned inside with_overrides() DOES inherit the
    overlay. This is the expected shape for fan-out retrieval (the
    bridge spawns parallel KB searches inside the overlay scope)."""
    results: list[str | None] = []

    async def child():
        results.append(flags.get("RAG_TEST_PROPAGATE"))

    with flags.with_overrides({"RAG_TEST_PROPAGATE": "parent-value"}):
        task = asyncio.create_task(child())
        await task

    assert results == ["parent-value"]


def test_peek_overlay_returns_none_when_no_overlay():
    """Test helper exposed by ``flags.py`` for assertions."""
    from ext.services.flags import _peek_overlay_for_tests
    assert _peek_overlay_for_tests() is None


def test_peek_overlay_returns_copy_when_active():
    from ext.services.flags import _peek_overlay_for_tests
    with flags.with_overrides({"RAG_X": "1"}):
        snap = _peek_overlay_for_tests()
        assert snap == {"RAG_X": "1"}
        # Mutation of the returned copy does NOT leak back into the
        # live overlay (regression guard for helper correctness).
        snap["RAG_Y"] = "2"
        assert flags.get("RAG_Y") is None
