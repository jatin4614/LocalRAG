"""Per-shard metrics exposed by ext.services.metrics.

Plan B Phase 5.9 — verifies the four per-shard metrics exist and that
``set_shard_tier`` flips the active tier gauge to 1 while resetting
prior tiers to 0 so a Prometheus scrape never observes two active
tiers simultaneously.
"""
from __future__ import annotations

import pytest


def _gauge_value(name: str, labels: dict) -> float:
    from prometheus_client import REGISTRY
    return REGISTRY.get_sample_value(name, labels=labels) or 0.0


def test_metrics_exposed():
    from ext.services import metrics
    for n in (
        "RAG_SHARD_POINT_COUNT",
        "RAG_SHARD_SEARCH_LATENCY",
        "RAG_SHARD_UPSERT_LATENCY",
        "RAG_SHARD_TIER",
    ):
        assert hasattr(metrics, n), f"metric {n} not exposed"


def test_set_shard_tier_gauge():
    pytest.importorskip("prometheus_client")
    from ext.services.metrics import set_shard_tier
    set_shard_tier(collection="kb_1_v4", shard_key="2026-04", tier="hot")
    assert _gauge_value(
        "rag_shard_tier",
        {"collection": "kb_1_v4", "shard_key": "2026-04", "tier": "hot"},
    ) == 1.0
    # Switch tier — old gauge resets to 0
    set_shard_tier(collection="kb_1_v4", shard_key="2026-04", tier="warm")
    assert _gauge_value(
        "rag_shard_tier",
        {"collection": "kb_1_v4", "shard_key": "2026-04", "tier": "hot"},
    ) == 0.0
    assert _gauge_value(
        "rag_shard_tier",
        {"collection": "kb_1_v4", "shard_key": "2026-04", "tier": "warm"},
    ) == 1.0
