"""Celery application factory for the ingest worker.

This module is imported both by the worker entrypoint
(``celery -A ext.workers.celery_app worker``) and by the producer-side shim
in ``ext.routers.upload`` when ``RAG_SYNC_INGEST=0``. In the default
(synchronous) path this module is never imported, so Celery is an optional
runtime dependency.
"""
from __future__ import annotations

import os

from celery import Celery

broker = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/1")
backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/2")

app = Celery(
    "orgchat_ingest",
    broker=broker,
    backend=backend,
    include=[
        "ext.workers.ingest_worker",
        "ext.workers.blob_gc_task",
        # Phase 4: weekly golden-set eval. The module registers its own
        # beat_schedule entry; inert unless celery beat is running.
        "ext.workers.scheduled_eval",
    ],
)

app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Long tasks: no greedy prefetch so a worker crash doesn't drag 10 docs with it.
    worker_prefetch_multiplier=1,
    task_time_limit=60 * 60,        # 1 h hard limit
    task_soft_time_limit=55 * 60,
    # Visibility timeout > task_time_limit so Redis doesn't redeliver mid-task.
    broker_transport_options={"visibility_timeout": 7200},  # 2 h
    result_expires=60 * 60 * 24,
    task_default_queue="ingest",
    task_queues={"ingest": {}, "ingest_dlq": {}},
)
