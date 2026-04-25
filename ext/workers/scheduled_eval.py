"""Celery beat task: run the golden-set eval harness weekly.

Fires every Sunday at 03:00 UTC (``crontab(hour="3", day_of_week="0")``).
Runs ``tests/eval/run_all.py`` as a subprocess — NOT in-process —
because the harness has its own sys.path dance + pulls optional deps
(RAGAS, numpy) that we don't want forcibly loaded into the long-lived
Celery worker. Subprocess also gives us a clean exit-code signal and
protects the worker from any harness-side OOM.

Writes ``tests/eval/results/weekly-YYYY-MM-DD.json`` and pushes the
three headline gauges:

* ``rag_eval_chunk_recall``      — last aggregate chunk_recall@10
* ``rag_eval_faithfulness``      — last aggregate faithfulness
* ``rag_eval_p95_latency_ms``    — last p95 retrieval latency (ms)

A Redis lock prevents two concurrent schedulers (one per host if
someone scales Beat horizontally) from double-running the task.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from celery.schedules import crontab

from .celery_app import app

log = logging.getLogger("orgchat.scheduled_eval")


# Repo root — Celery runs from the container's /app (see Dockerfile); the
# eval harness + results dir live under ``/app/tests/eval``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _REPO_ROOT / "tests" / "eval"
_RESULTS_DIR = _EVAL_DIR / "results"
_RUN_ALL = _EVAL_DIR / "run_all.py"


def _lock_key() -> str:
    return os.environ.get("RAG_EVAL_LOCK_KEY", "orgchat:scheduled_eval:lock")


def _lock_ttl_seconds() -> int:
    """How long the distributed lock is held. Longer than any expected eval
    run — the eval harness has a hard upper bound of an hour in practice.
    """
    try:
        return max(60, int(os.environ.get("RAG_EVAL_LOCK_TTL", "3600")))
    except (TypeError, ValueError):
        return 3600


def _acquire_redis_lock() -> tuple[Any, str] | tuple[None, None]:
    """Try to acquire the weekly-eval Redis lock.

    Returns ``(redis_client, lock_value)`` on success, or ``(None, None)``
    when another scheduler holds the lock. Caller releases with
    :func:`_release_redis_lock`.

    Fail-open: any Redis issue (module missing, connection refused, etc.)
    returns ``("noop", "noop-value")`` so the task proceeds. The rationale:
    the lock is a safety net against double-scheduling — if Redis is down
    we'd rather run the eval than skip it silently.
    """
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        log.info("scheduled_eval: redis module missing; proceeding without lock")
        return ("noop", "noop-value")

    broker = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/1")
    try:
        client = redis.Redis.from_url(broker)
        lock_val = f"host-{os.environ.get('HOSTNAME', 'unknown')}-{int(time.time())}"
        ok = client.set(_lock_key(), lock_val, nx=True, ex=_lock_ttl_seconds())
        if not ok:
            return (None, None)
        return (client, lock_val)
    except Exception as e:
        log.info("scheduled_eval: redis lock acquisition failed (%s); proceeding without lock", e)
        return ("noop", "noop-value")


def _release_redis_lock(client: Any, lock_val: str) -> None:
    """Best-effort release. Compare-and-set so we don't free a lock some
    other scheduler took after ours expired (unlikely given the TTL, but
    correct)."""
    if client is None or client == "noop":
        return
    try:
        # Redis lua for compare-and-set:
        #   if get(KEY) == ARGV[1] then del(KEY) end
        release_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        end
        return 0
        """
        client.eval(release_script, 1, _lock_key(), lock_val)
    except Exception as e:
        log.info("scheduled_eval: redis lock release failed: %s", e)


def _publish_gauges(aggregate: dict, latency: dict) -> None:
    """Push the three headline scores into Prometheus gauges.

    Fail-open: missing keys silently skip that gauge; prometheus_client
    issues are swallowed so the task still returns its status dict.
    """
    try:
        from ..services import metrics as metrics_mod
    except Exception as e:
        log.info("scheduled_eval: metrics import failed: %s", e)
        return

    try:
        if "chunk_recall@10" in aggregate:
            metrics_mod.rag_eval_chunk_recall.set(float(aggregate["chunk_recall@10"]))
        if "faithfulness" in aggregate:
            metrics_mod.rag_eval_faithfulness.set(float(aggregate["faithfulness"]))
        if "p95" in latency:
            metrics_mod.rag_eval_p95_latency.set(float(latency["p95"]))
    except Exception as e:
        log.info("scheduled_eval: gauge publish failed: %s", e)


def _run_harness(output_path: Path) -> tuple[int, str]:
    """Invoke ``tests/eval/run_all.py`` as a subprocess.

    Returns ``(exit_code, stderr_text)``. The harness is expected to
    accept ``--out <path>`` and write JSON to it; we don't parse stdout.
    """
    if not _RUN_ALL.exists():
        msg = f"run_all.py not found at {_RUN_ALL}; Agent A hasn't landed Phase 0"
        log.warning("scheduled_eval: %s", msg)
        return (127, msg)

    # Ensure the results directory exists — first-time deployments may
    # not have it baked in.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(_RUN_ALL),
        "--out",
        str(output_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("RAG_EVAL_TIMEOUT", "3600")),
        )
        return (proc.returncode, proc.stderr or "")
    except subprocess.TimeoutExpired as e:
        return (124, f"eval harness timeout after {e.timeout}s")
    except Exception as e:
        return (1, f"eval harness subprocess error: {e}")


@app.task(name="ext.workers.scheduled_eval.run_weekly_eval", queue="ingest")
def run_weekly_eval() -> dict[str, Any]:
    """Weekly eval entry point (Celery beat-fired).

    Acquires the Redis lock, runs the harness, loads the resulting JSON,
    pushes gauges, and returns a status dict. Returns a ``skipped``
    status if another scheduler holds the lock. The return value is
    Celery's result payload — useful when invoking manually via
    ``celery call``.
    """
    client, lock_val = _acquire_redis_lock()
    if client is None and lock_val is None:
        log.info("scheduled_eval: another scheduler holds the lock; skipping")
        return {"status": "skipped", "reason": "locked"}

    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output_path = _RESULTS_DIR / f"weekly-{today}.json"
        started = time.time()
        code, stderr = _run_harness(output_path)
        duration_s = round(time.time() - started, 2)

        result: dict[str, Any] = {
            "status": "ok" if code == 0 else "error",
            "exit_code": code,
            "output_path": str(output_path),
            "duration_s": duration_s,
        }
        if stderr:
            # Cap the stderr we retain — we don't want a pathological
            # harness dumping an MB of traceback into every task result.
            result["stderr_tail"] = stderr[-2000:]

        if code != 0 or not output_path.exists():
            log.warning(
                "scheduled_eval: harness returned %s (stderr head=%r)",
                code, stderr[:400],
            )
            return result

        try:
            payload = json.loads(output_path.read_text())
            aggregate = payload.get("aggregate", {}) or {}
            latency = payload.get("latency_ms", {}) or {}
            _publish_gauges(aggregate, latency)
            result["aggregate"] = aggregate
        except Exception as e:
            log.warning("scheduled_eval: failed to parse %s: %s", output_path, e)
            result["status"] = "error"
            result["parse_error"] = str(e)

        return result
    finally:
        _release_redis_lock(client, lock_val)


# -----------------------------------------------------------------------
# Beat registration. Default: every Sunday at 03:00 UTC. Operators can
# override via ``RAG_EVAL_CRON`` using the same 5-field format accepted
# by ``ext.workers.blob_gc_task``.
# -----------------------------------------------------------------------

_DEFAULT_CRON = crontab(minute="0", hour="3", day_of_week="0")


def _parse_cron_spec(spec: str) -> crontab:
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


_RAW_CRON = os.environ.get("RAG_EVAL_CRON")
if _RAW_CRON:
    try:
        _CRON = _parse_cron_spec(_RAW_CRON)
    except Exception as exc:  # noqa: BLE001
        log.warning("scheduled_eval: invalid RAG_EVAL_CRON=%r (%s); using default", _RAW_CRON, exc)
        _CRON = _DEFAULT_CRON
else:
    _CRON = _DEFAULT_CRON


app.conf.beat_schedule = {
    **getattr(app.conf, "beat_schedule", {}),
    "weekly-eval": {
        "task": "ext.workers.scheduled_eval.run_weekly_eval",
        "schedule": _CRON,
        "options": {"queue": "ingest"},
    },
}
