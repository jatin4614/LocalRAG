"""Unit tests for ``ext.workers.scheduled_eval``.

We verify:
  * the module imports without a running Redis/Celery broker
    (celery is imported lazily at call time, but the beat schedule is
    registered at module load);
  * the beat schedule entry ``weekly-eval`` is present with the
    expected task name;
  * gauge-publishing does not raise on an empty aggregate.
"""
from __future__ import annotations

import pytest

celery = pytest.importorskip("celery")


def test_module_imports_and_registers_beat_entry() -> None:
    # The celery_app fixture imports celery_app + all of its included
    # tasks, which drives the module-load beat_schedule update.
    from ext.workers import scheduled_eval  # noqa: F401
    from ext.workers.celery_app import app

    schedule = app.conf.beat_schedule or {}
    assert "weekly-eval" in schedule, f"expected weekly-eval entry, got {list(schedule)}"
    entry = schedule["weekly-eval"]
    assert entry["task"] == "ext.workers.scheduled_eval.run_weekly_eval"
    # queue is ingest
    assert entry.get("options", {}).get("queue") == "ingest"


def test_publish_gauges_handles_missing_keys() -> None:
    """Empty aggregate / latency dicts must not raise."""
    prometheus_client = pytest.importorskip("prometheus_client")
    from ext.workers.scheduled_eval import _publish_gauges

    # Should be a no-op, not a raise.
    _publish_gauges({}, {})


def test_publish_gauges_sets_expected_values() -> None:
    """Populated aggregate → gauges carry the right value afterward."""
    prometheus_client = pytest.importorskip("prometheus_client")
    from ext.services import metrics as m
    from ext.workers.scheduled_eval import _publish_gauges

    _publish_gauges(
        {"chunk_recall@10": 0.92, "faithfulness": 0.71},
        {"p50": 120.0, "p95": 230.0},
    )

    # Read back via the registry exposition.
    values: dict[str, float] = {}
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name == "rag_eval_chunk_recall":
            for s in mf.samples:
                values["chunk_recall"] = s.value
        elif mf.name == "rag_eval_faithfulness":
            for s in mf.samples:
                values["faithfulness"] = s.value
        elif mf.name == "rag_eval_p95_latency_ms":
            for s in mf.samples:
                values["p95"] = s.value

    assert values.get("chunk_recall") == pytest.approx(0.92)
    assert values.get("faithfulness") == pytest.approx(0.71)
    assert values.get("p95") == pytest.approx(230.0)
