"""Integration tests against a live vllm-qu container.

Plan B Phase 4.9. Requires:
  * vllm-qu running (``docker compose up -d vllm-qu`` — ~90 s to load)
  * ``RAG_QU_URL`` pointing at it (default ``http://localhost:8101/v1``)

Skipped by default (``pytest.mark.integration``). Run explicitly with::

    pytest --integration tests/integration/test_vllm_qu_live.py -v

The harness probes ``/v1/models`` first; if vllm-qu isn't reachable the
test is skipped (not failed) so a developer machine without GPU 1 can
still ship Phase 4 work.
"""
from __future__ import annotations

import os
import time

import httpx
import pytest


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def qu_url() -> str:
    return os.environ.get("RAG_QU_URL", "http://localhost:8101/v1")


@pytest.fixture(scope="module")
def qu_model() -> str:
    return os.environ.get("RAG_QU_MODEL", "qwen3-4b-qu")


@pytest.fixture(scope="module", autouse=True)
def _require_live_vllm_qu(qu_url: str, qu_model: str):
    """Skip the entire module if vllm-qu isn't reachable."""
    try:
        r = httpx.get(f"{qu_url}/models", timeout=5.0)
    except Exception as e:
        pytest.skip(f"vllm-qu unreachable at {qu_url}: {e}")
    if r.status_code != 200:
        pytest.skip(
            f"vllm-qu /models returned {r.status_code} — container not ready"
        )
    body = r.json()
    served = {m.get("id") for m in (body.get("data") or [])}
    if qu_model not in served:
        pytest.skip(
            f"vllm-qu does not serve model {qu_model!r}; got {served!r}"
        )


@pytest.mark.asyncio
async def test_live_basic_classification(qu_url: str, qu_model: str):
    from ext.services.query_understanding import analyze_query

    qu = await analyze_query(
        query="list all reports from January 2026",
        history=[],
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=10_000,
    )
    assert qu is not None, "QU LLM did not respond"
    assert qu.intent in {"metadata", "global", "specific", "specific_date"}
    assert 0.0 <= qu.confidence <= 1.0


@pytest.mark.asyncio
async def test_live_resolves_pronoun_with_history(qu_url: str, qu_model: str):
    from ext.services.query_understanding import analyze_query

    history = [
        {"role": "user", "content": "Tell me about the OFC roadmap"},
        {
            "role": "assistant",
            "content": (
                "OFC roadmap covers 2026-Q1 to 2027-Q1 with milestones..."
            ),
        },
    ]
    qu = await analyze_query(
        query="and what about it in Q2?",
        history=history,
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=10_000,
    )
    assert qu is not None
    # Should resolve "it" -> something about OFC / roadmap in resolved_query
    rq = qu.resolved_query.lower()
    assert "ofc" in rq or "roadmap" in rq, (
        f"resolved_query did not surface antecedent: {qu.resolved_query!r}"
    )


@pytest.mark.asyncio
async def test_live_extracts_temporal_constraint(qu_url: str, qu_model: str):
    from ext.services.query_understanding import analyze_query

    qu = await analyze_query(
        query="outages in Q1 2026",
        history=[],
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=10_000,
    )
    assert qu is not None
    assert qu.temporal_constraint is not None, (
        f"expected temporal_constraint on Q1 query, got: {qu}"
    )
    assert qu.temporal_constraint.get("year") == 2026
    assert qu.temporal_constraint.get("quarter") == 1


@pytest.mark.asyncio
async def test_live_classifies_global_for_compare(qu_url: str, qu_model: str):
    from ext.services.query_understanding import analyze_query

    qu = await analyze_query(
        query="compare budgets across all years",
        history=[],
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=10_000,
    )
    assert qu is not None
    assert qu.intent == "global", f"expected 'global', got: {qu.intent}"


@pytest.mark.asyncio
async def test_live_returns_none_on_timeout(qu_url: str, qu_model: str):
    """1ms timeout is guaranteed to miss even a hot model — must soft-fail."""
    from ext.services.query_understanding import analyze_query

    qu = await analyze_query(
        query="hello",
        history=[],
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=1,
    )
    assert qu is None  # soft-fail by design


@pytest.mark.asyncio
async def test_live_p95_latency_under_budget(qu_url: str, qu_model: str):
    """Run 20 calls; p95 must be under 600ms (the SLO ceiling).

    Uses a small set of representative queries that mix cache-friendly
    repeats with novel queries so prefix-cache effects are realistic.
    """
    from ext.services.query_understanding import analyze_query

    queries = [
        "what changed last quarter",
        "compare budgets",
        "list documents",
        "outages on 5 Jan 2026",
        "summary of march",
    ] * 4  # 20 calls

    durations = []
    for q in queries:
        start = time.monotonic()
        await analyze_query(
            query=q,
            history=[],
            qu_url=qu_url,
            model=qu_model,
            timeout_ms=10_000,
        )
        durations.append(time.monotonic() - start)
    durations.sort()
    p95 = durations[int(0.95 * len(durations))]
    assert p95 < 0.6, f"p95 latency {p95:.3f}s exceeds 600ms SLO"
