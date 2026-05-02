"""Unit tests for the scheduled-eval gauge parser (review §8.11).

Verifies that ``_publish_gauges`` accepts both the legacy shape
(``aggregate`` + separate ``latency_ms`` blocks) and the actual harness
schema (``global`` block carrying ``p95_latency_ms`` inline).

The module-level beat-registration tests live in
``test_scheduled_eval_module.py``; this file focuses on the §8.11 fix.
"""
from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")
celery = pytest.importorskip("celery")


def _read_gauge(name: str) -> float | None:
    """Read the latest value of a single-sample gauge from the default registry."""
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != name:
            continue
        for s in mf.samples:
            return float(s.value)
    return None


def test_publish_gauges_legacy_shape() -> None:
    """The pre-fix code path: aggregate + separate latency_ms dict."""
    from ext.workers.scheduled_eval import _publish_gauges

    _publish_gauges(
        {"chunk_recall@10": 0.61, "faithfulness": 0.50},
        {"p95": 410.0},
    )

    assert _read_gauge("rag_eval_chunk_recall") == pytest.approx(0.61)
    assert _read_gauge("rag_eval_faithfulness") == pytest.approx(0.50)
    assert _read_gauge("rag_eval_p95_latency_ms") == pytest.approx(410.0)


def test_publish_gauges_harness_global_shape() -> None:
    """The actual ``tests/eval/harness.py`` shape — ``global`` carries p95 inline.

    Mirrors what ``run_weekly_eval`` now passes after the §8.11 fix:
    ``payload["global"]`` is sent as ``aggregate`` and the latency arg
    is empty because ``p95_latency_ms`` lives inside the same dict.
    """
    from ext.workers.scheduled_eval import _publish_gauges

    harness_global = {
        "n": 20,
        "chunk_recall@10": 0.83,
        "mrr@10": 0.74,
        "ndcg@10": 0.79,
        "p50_latency_ms": 240.0,
        "p95_latency_ms": 612.0,
        "p99_latency_ms": 980.0,
        # No "faithfulness" — harness doesn't compute it; the parser
        # must just skip the corresponding gauge.
    }
    _publish_gauges(harness_global, {})

    assert _read_gauge("rag_eval_chunk_recall") == pytest.approx(0.83)
    assert _read_gauge("rag_eval_p95_latency_ms") == pytest.approx(612.0)


def test_publish_gauges_p95_latency_in_outer_dict() -> None:
    """Mixed shape — ``p95_latency_ms`` provided in the latency dict."""
    from ext.workers.scheduled_eval import _publish_gauges

    _publish_gauges(
        {"chunk_recall@10": 0.5},
        {"p95_latency_ms": 333.0},
    )
    assert _read_gauge("rag_eval_p95_latency_ms") == pytest.approx(333.0)


def test_publish_gauges_missing_keys_no_raise() -> None:
    """Empty payload must not raise — fail-open contract."""
    from ext.workers.scheduled_eval import _publish_gauges

    _publish_gauges({}, {})  # no exception
