"""Plan B Phase 4.7 — QU LLM metrics."""
import pytest


def _registry():
    from prometheus_client import REGISTRY

    return REGISTRY


def _counter_value(name: str, labels: dict | None = None) -> float:
    return _registry().get_sample_value(name, labels=labels or {}) or 0.0


def test_qu_metrics_exposed():
    """The Plan B Phase 4.7 metrics are importable from ext.services.metrics."""
    from ext.services import metrics

    assert hasattr(metrics, "rag_qu_invocations")
    assert hasattr(metrics, "rag_qu_escalations")
    assert hasattr(metrics, "rag_qu_latency")
    assert hasattr(metrics, "rag_qu_schema_violations")
    assert hasattr(metrics, "rag_qu_cache_hits")
    assert hasattr(metrics, "rag_qu_cache_misses")
    assert hasattr(metrics, "rag_qu_cache_hit_ratio")


@pytest.mark.asyncio
async def test_escalation_counter_incremented_on_escalation(monkeypatch):
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.9,
        )

    monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)

    # Plan B Phase 4 followup: the regex-default-fallback predicate fires
    # before comparison_verb when regex hits its default rule (which it
    # does for short queries with no specific pattern). Both routes
    # increment the same family of counters; we just label the reason.
    before = _counter_value(
        "rag_qu_escalations_total", {"reason": "regex_default_fallback"}
    )
    await qi.classify_with_qu("compare budgets", history=[])
    after = _counter_value(
        "rag_qu_escalations_total", {"reason": "regex_default_fallback"}
    )
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_invocations_source_label(monkeypatch):
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global",
            resolved_query="x",
            temporal_constraint=None,
            entities=[],
            confidence=0.9,
        )

    monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)

    before_llm = _counter_value("rag_qu_invocations_total", {"source": "llm"})
    await qi.classify_with_qu("compare budgets", history=[])
    after_llm = _counter_value("rag_qu_invocations_total", {"source": "llm"})
    assert after_llm - before_llm == 1.0


@pytest.mark.asyncio
async def test_invocations_regex_label_when_disabled(monkeypatch):
    from ext.services import query_intent as qi

    monkeypatch.setenv("RAG_QU_ENABLED", "0")

    before = _counter_value("rag_qu_invocations_total", {"source": "regex"})
    await qi.classify_with_qu("compare budgets", history=[])
    after = _counter_value("rag_qu_invocations_total", {"source": "regex"})
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_schema_violation_counted(monkeypatch):
    """Garbage JSON in QU response increments the schema-violation counter
    AND analyze_query returns None (soft-fail)."""
    from ext.services import query_understanding as qu_mod
    import httpx

    class _FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "not json garbage {{"}}]}

    async def fake_post(self, *a, **kw):
        return _FakeResp()

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    before = _counter_value("rag_qu_schema_violations_total")
    result = await qu_mod.analyze_query(
        "x", history=[], qu_url="http://stub", model="m"
    )
    after = _counter_value("rag_qu_schema_violations_total")
    assert result is None
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_cache_hit_counter(fake_redis):
    from ext.services.qu_cache import QUCache
    from ext.services.query_understanding import QueryUnderstanding

    cache = QUCache(redis_client=fake_redis, ttl_secs=300)
    qu = QueryUnderstanding(
        intent="specific",
        resolved_query="x",
        temporal_constraint=None,
        entities=[],
        confidence=0.5,
    )
    await cache.set("q", "t", qu)

    before_hit = _counter_value("rag_qu_cache_hits_total")
    before_miss = _counter_value("rag_qu_cache_misses_total")
    await cache.get("q", "t")  # hit
    await cache.get("q2", "t")  # miss
    after_hit = _counter_value("rag_qu_cache_hits_total")
    after_miss = _counter_value("rag_qu_cache_misses_total")

    assert after_hit - before_hit == 1.0
    assert after_miss - before_miss == 1.0


@pytest.mark.asyncio
async def test_latency_histogram_observed(monkeypatch):
    """analyze_query observes latency on every call (success or failure)."""
    from ext.services import query_understanding as qu_mod
    import httpx

    # Simulate a connection error so the call returns None but still observes latency
    async def fake_post(self, *a, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    before_count = _registry().get_sample_value(
        "rag_qu_latency_seconds_count", labels={}
    ) or 0.0

    await qu_mod.analyze_query("x", history=[], qu_url="http://stub", model="m")

    after_count = _registry().get_sample_value(
        "rag_qu_latency_seconds_count", labels={}
    ) or 0.0
    assert after_count - before_count == 1.0
