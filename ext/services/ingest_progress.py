"""Per-KB ingest progress events over Redis pub/sub.

Celery worker emits one event per state transition (``queued`` → ``processing``
→ ``done``/``failed``). The SSE endpoint at ``/api/kb/{kb_id}/ingest-stream``
subscribes to the matching channel and forwards events to the admin UI so
uploads visibly progress instead of staying "queued" until the next page
refresh.

Channel naming::

    kb:{kb_id}:ingest

Event shape (JSON-encoded)::

    {"doc_id": 155, "filename": "Jan 23.docx",
     "stage": "queued|processing|done|failed",
     "chunks": 219,           # optional, on done
     "error": "..."           # optional, on failed
    }

The emitter is fail-open: if Redis is unreachable the ingest still
runs; we just lose telemetry for that one event.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator

import redis as _sync_redis
import redis.asyncio as _aredis

logger = logging.getLogger("orgchat.ingest_progress")


# Phase 6 followup — the celery broker already lives at db=1 (per the
# CELERY_BROKER_URL default in compose). Reuse the same Redis instance
# at db=0 (which already serves chat sessions, etc.) for ingest progress
# pub/sub so we don't need another container.
_REDIS_URL = os.environ.get(
    "INGEST_PROGRESS_REDIS_URL",
    os.environ.get("REDIS_URL", "redis://redis:6379/0"),
)


def _channel(kb_id: int) -> str:
    return f"kb:{int(kb_id)}:ingest"


def emit_sync(kb_id: int, event: dict) -> None:
    """Publish an event to ``kb:{kb_id}:ingest``. Fail-open.

    Synchronous because the Celery worker runs in a sync process
    (asyncio.run is used for individual stages but the task body itself
    is sync). Per-call client construction keeps the worker stateless;
    Redis connection setup is sub-millisecond.
    """
    if kb_id is None:
        return
    try:
        r = _sync_redis.from_url(_REDIS_URL, decode_responses=True)
        r.publish(_channel(kb_id), json.dumps(event, ensure_ascii=True))
    except Exception as e:  # noqa: BLE001 — telemetry failures must never break ingest
        logger.warning("ingest_progress.emit_sync failed kb=%s: %s", kb_id, e)


async def subscribe_async(kb_id: int) -> AsyncIterator[str]:
    """Async generator yielding raw JSON event payloads from the channel.

    Caller is responsible for closing the generator (FastAPI does this
    when the SSE response is cancelled / disconnected). Each iteration
    blocks until the next event lands.
    """
    r = _aredis.from_url(_REDIS_URL, decode_responses=True)
    pubsub = r.pubsub()
    channel = _channel(kb_id)
    await pubsub.subscribe(channel)
    try:
        async for msg in pubsub.listen():
            # ``listen`` yields control messages (subscribe, etc.) too —
            # only forward actual data frames.
            if msg.get("type") == "message":
                data = msg.get("data")
                if isinstance(data, (bytes, bytearray)):
                    data = data.decode("utf-8", errors="replace")
                yield str(data)
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        except Exception:
            pass
        try:
            await r.close()
        except Exception:
            pass
