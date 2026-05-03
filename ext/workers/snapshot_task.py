"""Celery periodic task: per-collection Qdrant snapshots.

This is the in-container side of the campaign rollback architecture's
Layer 4 (data restore from snapshot). The host-side counterpart is
``scripts/backup_qdrant.sh``, which operators run before risky changes.

The Celery task exists so a daily snapshot fires automatically on every
deployment without any operator action — it's the safety net for
"the operator forgot to back up before pushing a migration". Snapshots
are written inside the qdrant container itself (``/qdrant/snapshots/``);
a host-side bind mount on the celery-worker is what makes them visible to
backup tooling — see ``compose/docker-compose.yml`` (``backups:/host-backups``).

Each collection is snapshotted independently. **A failure on one collection
must NOT stop the others** — when ``kb_5`` is corrupt we still want
``kb_1`` and ``chat_private`` snapshots, because those are the ones the
operator will need to restore from.

Environment
-----------
``QDRANT_URL``                  Qdrant REST URL; defaults to ``http://qdrant:6333``.
``QDRANT_API_KEY``              Optional per-request API key (Wave 1a §4.1).
``RAG_QDRANT_SNAPSHOT_CRON``    Beat cron expression
                                ``minute hour day_of_week month day_of_month``
                                (default: daily at 02:30 UTC, offset from
                                blob_gc's 03:17). Used only when beat is active.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from celery.schedules import crontab

from ..services.metrics import rag_snapshot_failure_total
from .celery_app import app

log = logging.getLogger("orgchat.snapshot_task")


async def _run_snapshots(client: Any) -> dict[str, Any]:
    """Snapshot every collection. One failure does not stop the others.

    Returns a summary dict::

        {
          "snapshotted": int,
          "failed": int,
          "collections": [
              {"collection": "kb_1", "snapshot": "kb_1-…", "size": 1234},
              ...
          ],
          # Only present on listing failure:
          "error": "qdrant unreachable",
        }

    The ``rag_snapshot_failure_total{collection=…}`` counter is bumped
    once per per-collection failure (label is the collection name) AND
    once with ``collection="_list_collections"`` if listing itself
    fails.
    """
    summary: dict[str, Any] = {
        "snapshotted": 0,
        "failed": 0,
        "collections": [],
    }

    # 1. List collections. If THIS fails the task is dead — record a
    # sentinel failure label so operators can distinguish "qdrant down"
    # from "all collections healthy" in dashboards.
    try:
        cols_resp = await client.get_collections()
        names = [c.name for c in cols_resp.collections]
    except Exception as exc:  # noqa: BLE001
        log.warning("qdrant_snapshot: get_collections failed: %s", exc)
        try:
            rag_snapshot_failure_total.labels(collection="_list_collections").inc()
        except Exception:  # pragma: no cover - metrics is fail-open
            pass
        summary["error"] = str(exc)
        return summary

    log.info("qdrant_snapshot: snapshotting %d collections", len(names))

    # 2. Per-collection snapshot. Each call independently try/except'd so
    # one bad collection cannot starve the others.
    for name in names:
        try:
            desc = await client.create_snapshot(collection_name=name)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "qdrant_snapshot: snapshot for collection=%s failed: %s",
                name,
                exc,
            )
            try:
                rag_snapshot_failure_total.labels(collection=name).inc()
            except Exception:  # pragma: no cover - metrics is fail-open
                pass
            summary["failed"] += 1
            continue

        # ``create_snapshot`` returns SnapshotDescription with .name + .size
        # (size in bytes). Some test stubs return None for transient cases;
        # treat that as a soft no-op rather than crashing.
        if desc is None:
            log.warning("qdrant_snapshot: collection=%s returned no description", name)
            try:
                rag_snapshot_failure_total.labels(collection=name).inc()
            except Exception:  # pragma: no cover - metrics is fail-open
                pass
            summary["failed"] += 1
            continue

        snap_name = getattr(desc, "name", None)
        snap_size = getattr(desc, "size", None)
        log.info(
            "qdrant_snapshot: collection=%s snapshot=%s bytes=%s",
            name,
            snap_name,
            snap_size,
        )
        summary["snapshotted"] += 1
        summary["collections"].append({
            "collection": name,
            "snapshot": snap_name,
            "size": snap_size,
        })

    return summary


async def _one_pass() -> dict[str, Any]:
    """Open a fresh AsyncQdrantClient, run all snapshots, close it."""
    # Lazy import — keeps the celery_app boot cheap and allows test stubs
    # to bypass the real qdrant-client entirely via patching ``_run_snapshots``.
    from qdrant_client import AsyncQdrantClient

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    timeout = float(os.environ.get("RAG_QDRANT_TIMEOUT", "120.0"))

    client = AsyncQdrantClient(url=qdrant_url, api_key=api_key, timeout=timeout)
    try:
        return await _run_snapshots(client)
    finally:
        try:
            await client.close()
        except Exception:  # pragma: no cover - best-effort close
            pass


@app.task(name="ext.workers.snapshot_task.qdrant_snapshot", queue="ingest")
def qdrant_snapshot() -> dict[str, Any]:
    """Snapshot every Qdrant collection.

    This is the Celery beat-fireable entrypoint; it shares the ingest
    queue (no separate worker needed). Returns the summary from
    :func:`_run_snapshots`.
    """
    return asyncio.run(_one_pass())


def _parse_cron_spec(spec: str) -> crontab:
    """Parse ``"minute hour day_of_week month day_of_month"`` (5 fields).

    Five space-separated tokens, same order as Celery's :class:`crontab`
    positional args. Mirrors :mod:`ext.workers.blob_gc_task` so operators
    learn one cron format for both periodic tasks.
    """
    parts = spec.strip().split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, got {len(parts)}")
    minute, hour, day_of_week, month_of_year, day_of_month = parts
    return crontab(
        minute=minute,
        hour=hour,
        day_of_week=day_of_week,
        month_of_year=month_of_year,
        day_of_month=day_of_month,
    )


def _build_cron() -> crontab:
    """Read RAG_QDRANT_SNAPSHOT_CRON and return a crontab.

    Default: daily at 02:30 UTC. Offset from blob_gc's 03:17 so the two
    tasks never compete for I/O. Falls back to default on any parse
    error so a bad env var never kills the worker.
    """
    default = crontab(minute="30", hour="2")
    raw = os.environ.get("RAG_QDRANT_SNAPSHOT_CRON")
    if not raw:
        return default
    try:
        return _parse_cron_spec(raw)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "snapshot_task: invalid RAG_QDRANT_SNAPSHOT_CRON=%r (%s); using default",
            raw,
            exc,
        )
        return default


_CRON = _build_cron()


# Register a Beat schedule entry. This is inert unless ``celery beat`` is
# actually running; workers without beat will never fire it. The pattern
# mirrors ext/workers/blob_gc_task.py:112-133 exactly — that's the
# canonical beat-schedule registration in this codebase.
app.conf.beat_schedule = {
    **getattr(app.conf, "beat_schedule", {}),
    "qdrant-snapshot-daily": {
        "task": "ext.workers.snapshot_task.qdrant_snapshot",
        "schedule": _CRON,
        "options": {"queue": "ingest"},
    },
}
