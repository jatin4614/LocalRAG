"""Unit tests for ``ext.workers.snapshot_task``.

Covers:
* ``_run_snapshots`` iterates every collection returned by the client.
* A failure on one collection does not abort the rest (soft-fail).
* The ``rag_snapshot_failure_total{collection=…}`` counter is bumped on
  per-collection failure.
* Beat-schedule cron parser respects ``RAG_QDRANT_SNAPSHOT_CRON`` and
  falls back to the default on any parse error.

The Qdrant client is fully stubbed; no real network calls are made. The
metrics counter is monkey-patched into a captured-call recorder so the
assertion is independent of any prometheus_client side-effects.
"""
from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from ext.workers import snapshot_task


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubCollection:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubGetCollectionsResponse:
    def __init__(self, names: list[str]) -> None:
        self.collections = [_StubCollection(n) for n in names]


class _StubSnapshotDescription:
    def __init__(self, name: str, size: int) -> None:
        self.name = name
        self.size = size


class StubQdrantClient:
    """Async stub — exposes get_collections + create_snapshot.

    ``raise_on`` is a set of collection names that should raise on
    create_snapshot to exercise per-collection soft-fail.
    """

    def __init__(
        self,
        names: list[str],
        *,
        raise_on: set[str] | None = None,
        sizes: dict[str, int] | None = None,
    ) -> None:
        self._names = names
        self._raise_on = raise_on or set()
        self._sizes = sizes or {}
        self.snapshot_calls: list[str] = []
        self.closed = False

    async def get_collections(self) -> _StubGetCollectionsResponse:
        return _StubGetCollectionsResponse(self._names)

    async def create_snapshot(self, collection_name: str, **_kwargs: Any) -> _StubSnapshotDescription:
        self.snapshot_calls.append(collection_name)
        if collection_name in self._raise_on:
            raise RuntimeError(f"simulated snapshot failure for {collection_name}")
        return _StubSnapshotDescription(
            name=f"{collection_name}-snap-1",
            size=self._sizes.get(collection_name, 12345),
        )

    async def close(self) -> None:
        self.closed = True


class _CounterRecorder:
    """Stand-in for the prometheus Counter; records labels(...).inc() calls."""

    def __init__(self) -> None:
        self.inc_calls: list[dict[str, Any]] = []
        self._pending_labels: dict[str, Any] | None = None

    def labels(self, **kwargs: Any) -> "_CounterRecorder":
        self._pending_labels = kwargs
        return self

    def inc(self, amount: float = 1.0) -> None:
        self.inc_calls.append({"labels": self._pending_labels, "amount": amount})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_run_snapshots_iterates_every_collection(monkeypatch) -> None:
    """Even when one snapshot fails, the others must still run."""
    client = StubQdrantClient(
        names=["kb_1", "kb_2_v4", "chat_private"],
        sizes={"kb_1": 1024, "kb_2_v4": 2048, "chat_private": 512},
    )
    recorder = _CounterRecorder()
    monkeypatch.setattr(snapshot_task, "rag_snapshot_failure_total", recorder)

    result = await snapshot_task._run_snapshots(client)

    # Every collection got a snapshot attempt.
    assert sorted(client.snapshot_calls) == ["chat_private", "kb_1", "kb_2_v4"]
    # No failures, no counter increments.
    assert recorder.inc_calls == []
    # Per-collection summary is returned.
    assert result["snapshotted"] == 3
    assert result["failed"] == 0
    assert sorted(c["collection"] for c in result["collections"]) == [
        "chat_private",
        "kb_1",
        "kb_2_v4",
    ]


async def test_one_failed_snapshot_does_not_stop_others(monkeypatch) -> None:
    """A bad snapshot on kb_2 must NOT prevent kb_1 and kb_3 from succeeding."""
    client = StubQdrantClient(
        names=["kb_1", "kb_2_broken", "kb_3"],
        raise_on={"kb_2_broken"},
    )
    recorder = _CounterRecorder()
    monkeypatch.setattr(snapshot_task, "rag_snapshot_failure_total", recorder)

    result = await snapshot_task._run_snapshots(client)

    # All three were attempted (proving we didn't bail at the first failure).
    assert sorted(client.snapshot_calls) == ["kb_1", "kb_2_broken", "kb_3"]
    # Two succeeded, one failed.
    assert result["snapshotted"] == 2
    assert result["failed"] == 1
    # Counter bumped once with the failing collection label.
    assert len(recorder.inc_calls) == 1
    assert recorder.inc_calls[0]["labels"] == {"collection": "kb_2_broken"}
    assert recorder.inc_calls[0]["amount"] == 1.0


async def test_empty_cluster_returns_zero(monkeypatch) -> None:
    """No collections → no failure, no work."""
    client = StubQdrantClient(names=[])
    recorder = _CounterRecorder()
    monkeypatch.setattr(snapshot_task, "rag_snapshot_failure_total", recorder)

    result = await snapshot_task._run_snapshots(client)

    assert client.snapshot_calls == []
    assert result["snapshotted"] == 0
    assert result["failed"] == 0
    assert result["collections"] == []
    assert recorder.inc_calls == []


async def test_get_collections_failure_logs_and_returns_zero(monkeypatch) -> None:
    """If listing collections itself fails, the task must return a stable
    error summary rather than crashing the worker."""

    class BrokenClient:
        async def get_collections(self) -> Any:
            raise RuntimeError("qdrant unreachable")

        async def close(self) -> None:
            return None

    recorder = _CounterRecorder()
    monkeypatch.setattr(snapshot_task, "rag_snapshot_failure_total", recorder)

    result = await snapshot_task._run_snapshots(BrokenClient())

    # No per-collection inc, but the task as a whole flags an error.
    assert result["snapshotted"] == 0
    # We bump a sentinel "_list_collections" failure so operators can
    # distinguish "qdrant down" from "all collections are healthy".
    assert any(
        c["labels"] == {"collection": "_list_collections"}
        for c in recorder.inc_calls
    )
    assert "error" in result


def test_parse_cron_default(monkeypatch) -> None:
    monkeypatch.delenv("RAG_QDRANT_SNAPSHOT_CRON", raising=False)
    cron = snapshot_task._build_cron()
    # Default = daily at 02:30 UTC.
    assert "30" in str(cron._orig_minute)
    assert "2" in str(cron._orig_hour)


def test_parse_cron_override(monkeypatch) -> None:
    """RAG_QDRANT_SNAPSHOT_CRON overrides the default."""
    monkeypatch.setenv("RAG_QDRANT_SNAPSHOT_CRON", "15 4 * * *")
    cron = snapshot_task._build_cron()
    assert "15" in str(cron._orig_minute)
    assert "4" in str(cron._orig_hour)


def test_parse_cron_bad_input_falls_back(monkeypatch) -> None:
    """Garbage cron must not crash the worker; falls back to default."""
    monkeypatch.setenv("RAG_QDRANT_SNAPSHOT_CRON", "this is not a cron")
    cron = snapshot_task._build_cron()
    # Default = daily at 02:30 UTC (proving fallback fired).
    assert "30" in str(cron._orig_minute)
    assert "2" in str(cron._orig_hour)


def test_beat_schedule_registered() -> None:
    """The module-level import must register a beat_schedule entry that
    points at the snapshot task."""
    from ext.workers.celery_app import app

    schedule = getattr(app.conf, "beat_schedule", {})
    assert "qdrant-snapshot-daily" in schedule, (
        "snapshot task must register a beat entry so flipping celery beat on "
        "is a one-env-var change (matches blob_gc pattern)"
    )
    entry = schedule["qdrant-snapshot-daily"]
    assert entry["task"] == "ext.workers.snapshot_task.qdrant_snapshot"
    assert entry["options"]["queue"] == "ingest"
