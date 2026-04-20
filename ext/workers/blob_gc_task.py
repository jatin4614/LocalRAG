"""Celery periodic task: run :func:`ext.services.blob_gc.run_gc`.

Celery Beat (a separate process from the worker) is required to fire this
task on a schedule. The compose setup does NOT include beat today; this
module only registers the task and a default beat schedule entry so that
flipping beat on later is a one-env-var change.

If you do not run beat, invoke the task manually::

    celery -A ext.workers.celery_app call ext.workers.blob_gc_task.blob_gc

or just run the CLI (``python scripts/blob_gc.py --apply``) from cron.

Environment
-----------
``DATABASE_URL``             PostgreSQL URL (required).
``INGEST_BLOB_ROOT``         Blob store root; defaults to ``/var/ingest``.
``QDRANT_URL``               Qdrant REST URL; defaults to ``http://qdrant:6333``.
``RAG_VECTOR_SIZE``          Embedding dim; defaults to ``1024``.
``RAG_BLOB_RETENTION_DAYS``  Retention in days; defaults to ``30``.
``RAG_BLOB_GC_LIMIT``        Max rows per pass; defaults to ``1000``.
``RAG_BLOB_GC_CRON``         Beat cron expression ``minute hour dow month day_of_month``
                             (default: daily at 03:17 UTC). Used only when
                             beat is active.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from celery.schedules import crontab

from ..services.blob_gc import retention_days as env_retention, run_gc
from ..services.blob_store import BlobStore
from .celery_app import app

log = logging.getLogger("orgchat.blob_gc_task")


async def _one_pass(database_url: str, dry_run: bool, limit: int) -> dict:
    # Local imports so Celery can import the module even if SQLAlchemy /
    # VectorStore are absent at definition time (they're always present at
    # execution time, but this keeps the module lightweight).
    from ..db.session import make_engine, make_sessionmaker
    from ..services.vector_store import VectorStore

    blob_root = os.environ.get("INGEST_BLOB_ROOT", "/var/ingest")
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    vector_size = int(os.environ.get("RAG_VECTOR_SIZE", "1024"))

    engine = make_engine(database_url)
    sm = make_sessionmaker(engine)
    vs = VectorStore(url=qdrant_url, vector_size=vector_size)
    blob_store = BlobStore(blob_root)

    try:
        async with sm() as session:
            return await run_gc(
                session=session,
                blob_store=blob_store,
                vector_store=vs,
                retention_days=env_retention(),
                dry_run=dry_run,
                limit=limit,
            )
    finally:
        try:
            await vs.close()
        except Exception:
            pass
        await engine.dispose()


@app.task(name="ext.workers.blob_gc_task.blob_gc", queue="ingest")
def blob_gc(dry_run: bool = False, limit: int | None = None) -> dict[str, Any]:
    """Run one GC pass.

    Defaults to ``dry_run=False`` because a scheduled task exists to free
    resources, not to just print. Pass ``dry_run=True`` when invoking
    manually to preview.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        log.error("blob_gc_task: DATABASE_URL not set; aborting")
        return {"error": "DATABASE_URL not set"}
    eff_limit = limit if limit is not None else int(os.environ.get("RAG_BLOB_GC_LIMIT", "1000"))
    return asyncio.run(_one_pass(database_url, bool(dry_run), eff_limit))


def _parse_cron_spec(spec: str) -> crontab:
    """Parse ``"minute hour day_of_week month day_of_month"`` into a crontab.

    Five space-separated tokens, same order as Celery's :class:`crontab`
    positional args. Falls back to the default (03:17 UTC daily) on any
    parse error so a bad env var never kills the worker.
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


_DEFAULT_CRON = crontab(minute="17", hour="3")
_RAW_CRON = os.environ.get("RAG_BLOB_GC_CRON")
if _RAW_CRON:
    try:
        _CRON = _parse_cron_spec(_RAW_CRON)
    except Exception as exc:  # noqa: BLE001
        log.warning("blob_gc_task: invalid RAG_BLOB_GC_CRON=%r (%s); using default", _RAW_CRON, exc)
        _CRON = _DEFAULT_CRON
else:
    _CRON = _DEFAULT_CRON


# Register a Beat schedule entry. This is inert unless ``celery beat`` is
# actually running; workers without beat will never fire it.
app.conf.beat_schedule = {
    **getattr(app.conf, "beat_schedule", {}),
    "blob-gc-daily": {
        "task": "ext.workers.blob_gc_task.blob_gc",
        "schedule": _CRON,
        "kwargs": {"dry_run": False},
        "options": {"queue": "ingest"},
    },
}
