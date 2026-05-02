"""Verify that /metrics is exposed by a FastAPI app mounting prometheus's ASGI app.

This mirrors the mount done by ``ext.app.build_app`` but uses a minimal
FastAPI app so we do not need Postgres / Qdrant / TEI up.
"""
from __future__ import annotations

import pytest

prometheus_client = pytest.importorskip("prometheus_client")
httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prom_text():
    """GET /metrics on a minimal app returns 200 and contains rag_* metric lines."""
    from prometheus_client import make_asgi_app

    from ext.services import metrics

    # Prime the registry with at least one observation per metric so the
    # exposition is non-trivial.
    with metrics.time_stage("retrieve"):
        pass
    metrics.retrieval_hits_total.labels(kb_count="1", kb_primary="1", path="dense").inc()
    metrics.rerank_cache_total.labels(outcome="hit").inc()
    metrics.flag_state.labels(flag="hybrid").set(1)
    metrics.ingest_chunks_total.labels(collection="kb_1", path="sync").inc(3)

    app = FastAPI()
    app.mount("/metrics", make_asgi_app())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=True
    ) as client:
        r = await client.get("/metrics")

    assert r.status_code == 200, r.status_code
    body = r.text
    # Check for our metric families in the Prometheus exposition format.
    assert "rag_stage_latency_seconds" in body
    assert "rag_retrieval_hits_total" in body
    assert "rag_rerank_cache_total" in body
    assert "rag_flag_enabled" in body
    assert "rag_ingest_chunks_total" in body
    # HELP line should describe each family.
    assert "# HELP rag_stage_latency_seconds" in body


@pytest.mark.asyncio
async def test_ext_app_build_mounts_metrics(monkeypatch):
    """``ext.app.build_app`` should mount /metrics successfully.

    We cannot call build_app() directly — it wires up DB / Qdrant / TEI
    clients that aren't available in this test harness. Instead, assert
    that the ``_mount_metrics`` helper successfully mounts without raising
    when prometheus_client is installed.
    """
    from ext.app import _mount_metrics

    app = FastAPI()
    _mount_metrics(app)

    # Confirm a /metrics mount point is present on the app's routes.
    mounted_paths = {getattr(r, "path", None) for r in app.routes}
    assert "/metrics" in mounted_paths
