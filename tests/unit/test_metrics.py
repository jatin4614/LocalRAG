"""Unit tests for ext.services.metrics.

Skip cleanly if ``prometheus_client`` is not installed in the test env.
"""
from __future__ import annotations

import time

import pytest

prometheus_client = pytest.importorskip("prometheus_client")

from ext.services import metrics  # noqa: E402  — after importorskip


def _collect_samples_named(name_prefix: str) -> dict[str, list]:
    """Return all samples whose metric name starts with ``name_prefix``.

    Keyed by metric name; value is the list of prometheus Sample tuples.
    """
    out: dict[str, list] = {}
    for mf in prometheus_client.REGISTRY.collect():
        if not mf.name.startswith(name_prefix):
            continue
        for s in mf.samples:
            out.setdefault(s.name, []).append(s)
    return out


def test_prom_available():
    """prometheus_client is installed in this env, so flag must be True."""
    assert metrics.prom_available() is True


def test_time_stage_records_positive_duration():
    """After ``with time_stage(stage):`` a _count sample > 0 exists."""
    stage = f"teststage_{int(time.time() * 1000) & 0xffff}"
    before = {}
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_stage_latency_seconds":
            continue
        for s in mf.samples:
            if s.name.endswith("_count") and s.labels.get("stage") == stage:
                before[stage] = s.value

    with metrics.time_stage(stage):
        time.sleep(0.002)  # ensure observation > 0

    # Look for the _count sample for our stage label.
    after_count = None
    after_sum = None
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_stage_latency_seconds":
            continue
        for s in mf.samples:
            if s.labels.get("stage") != stage:
                continue
            if s.name.endswith("_count"):
                after_count = s.value
            elif s.name.endswith("_sum"):
                after_sum = s.value

    assert after_count is not None, "histogram _count sample missing for stage"
    assert after_count >= 1
    assert after_sum is not None and after_sum > 0


def test_retrieval_hits_counter_increments():
    """retrieval_hits_total increments per (kb_count, kb_primary, path) combo (review §8.6)."""
    metrics.retrieval_hits_total.labels(kb_count="2", kb_primary="1", path="hybrid").inc()
    metrics.retrieval_hits_total.labels(kb_count="2", kb_primary="1", path="hybrid").inc(2)

    total = 0.0
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_retrieval_hits":
            continue
        for s in mf.samples:
            if s.name == "rag_retrieval_hits_total" and s.labels == {
                "kb_count": "2", "kb_primary": "1", "path": "hybrid",
            }:
                total = s.value
    assert total >= 3.0


def test_rerank_cache_counter_increments():
    """rerank_cache_total{outcome=hit|miss} is a valid counter."""
    metrics.rerank_cache_total.labels(outcome="hit").inc(5)
    metrics.rerank_cache_total.labels(outcome="miss").inc(2)

    hit_total = 0.0
    miss_total = 0.0
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_rerank_cache":
            continue
        for s in mf.samples:
            if s.name != "rag_rerank_cache_total":
                continue
            if s.labels == {"outcome": "hit"}:
                hit_total = s.value
            elif s.labels == {"outcome": "miss"}:
                miss_total = s.value
    assert hit_total >= 5.0
    assert miss_total >= 2.0


def test_flag_state_gauge_set():
    """flag_state gauge can be toggled on/off."""
    metrics.flag_state.labels(flag="hybrid").set(1)
    metrics.flag_state.labels(flag="rerank").set(0)

    values: dict[str, float] = {}
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_flag_enabled":
            continue
        for s in mf.samples:
            flag = s.labels.get("flag")
            if flag:
                values[flag] = s.value
    assert values.get("hybrid") == 1.0
    assert values.get("rerank") == 0.0


def test_expected_metric_names_present():
    """Required metric families appear in the default registry."""
    names = {mf.name for mf in prometheus_client.REGISTRY.collect()}
    # Prometheus strips the ``_total`` / ``_seconds`` suffix on the family
    # name, so we assert the stem.
    for expected in (
        "rag_stage_latency_seconds",
        "rag_retrieval_hits",
        "rag_rerank_cache",
        "rag_flag_enabled",
        "rag_ingest_chunks",
    ):
        assert expected in names, f"expected metric family {expected!r} missing"


def test_time_stage_preserves_exceptions():
    """Exceptions inside the block still propagate, and metrics are still observed."""
    stage = "teststage_exc"
    with pytest.raises(RuntimeError, match="boom"):
        with metrics.time_stage(stage):
            raise RuntimeError("boom")

    # Histogram _count should still have been incremented despite the raise.
    found = False
    for mf in prometheus_client.REGISTRY.collect():
        if mf.name != "rag_stage_latency_seconds":
            continue
        for s in mf.samples:
            if s.name.endswith("_count") and s.labels.get("stage") == stage and s.value >= 1:
                found = True
    assert found, "time_stage must observe duration even when block raises"
