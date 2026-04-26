"""Plan B Phase 4.6 — wire QU into chat_rag_bridge."""
import pytest


@pytest.mark.asyncio
async def test_classify_with_qu_returns_regex_when_disabled(monkeypatch):
    """RAG_QU_ENABLED=0 → bridge soft-fails to regex without touching the LLM."""
    from ext.services import chat_rag_bridge as bridge

    monkeypatch.setenv("RAG_QU_ENABLED", "0")
    result = await bridge._classify_with_qu(
        query="compare budgets last quarter",
        history=[],
        last_turn_id="",
    )
    assert result.intent == "specific"
    assert result.source == "regex"


@pytest.mark.asyncio
async def test_classify_with_qu_uses_cache_on_repeat(monkeypatch, fake_redis):
    """Second identical call within TTL hits cache; LLM not invoked twice."""
    from ext.services import chat_rag_bridge as bridge
    from ext.services import qu_cache as cache_mod
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_CACHE_ENABLED", "1")
    # Reset cache singleton + inject fake redis
    monkeypatch.setattr(bridge, "_qu_cache_singleton", None, raising=False)
    monkeypatch.setattr(
        bridge,
        "_get_qu_cache",
        lambda: cache_mod.QUCache(redis_client=fake_redis, ttl_secs=300, enabled=True),
    )

    invoke_count = {"n": 0}

    async def fake_invoke_qu(query, history):
        invoke_count["n"] += 1
        return QueryUnderstanding(
            intent="global",
            resolved_query="compare budgets last quarter, resolved",
            temporal_constraint=None,
            entities=[],
            confidence=0.9,
            source="llm",
        )

    monkeypatch.setattr(qi, "_invoke_qu", fake_invoke_qu)

    # First call — cache miss, LLM invoked
    r1 = await bridge._classify_with_qu(
        query="compare budgets last quarter",
        history=[],
        last_turn_id="t-1",
    )
    # Second identical call — cache hit
    r2 = await bridge._classify_with_qu(
        query="compare budgets last quarter",
        history=[],
        last_turn_id="t-1",
    )
    assert invoke_count["n"] == 1, "QU should only be invoked once"
    assert r1.source == "llm"
    assert r2.cached is True
    assert r2.intent == "global"


@pytest.mark.asyncio
async def test_classify_with_qu_does_not_cache_regex_results(
    monkeypatch, fake_redis
):
    """Regex results (non-escalation) must NOT pollute the cache."""
    from ext.services import chat_rag_bridge as bridge
    from ext.services import qu_cache as cache_mod

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setattr(bridge, "_qu_cache_singleton", None, raising=False)
    cache = cache_mod.QUCache(redis_client=fake_redis, ttl_secs=300, enabled=True)
    monkeypatch.setattr(bridge, "_get_qu_cache", lambda: cache)

    # "list all reports" classifies as global by regex; no escalation
    await bridge._classify_with_qu(
        query="list all reports", history=[], last_turn_id="t-1",
    )
    # Cache should be empty
    assert len(fake_redis._store) == 0


def test_extract_last_turn_id_uses_assistant_id():
    """Extracts the most recent assistant turn's id when present."""
    from ext.services.chat_rag_bridge import _extract_last_turn_id

    history = [
        {"role": "user", "content": "first question", "id": "u-1"},
        {"role": "assistant", "content": "answer one", "id": "a-1"},
        {"role": "user", "content": "second question", "id": "u-2"},
        {"role": "assistant", "content": "answer two", "id": "a-2"},
    ]
    assert _extract_last_turn_id(history) == "a-2"


def test_extract_last_turn_id_falls_back_to_content_hash():
    """No id field → falls back to a content hash so context shifts still
    invalidate the cache."""
    from ext.services.chat_rag_bridge import _extract_last_turn_id

    h1 = [{"role": "assistant", "content": "answer one"}]
    h2 = [{"role": "assistant", "content": "answer two"}]
    id1 = _extract_last_turn_id(h1)
    id2 = _extract_last_turn_id(h2)
    assert id1 != id2
    assert id1 != ""


def test_extract_last_turn_id_returns_empty_for_no_assistant():
    """User-only history (new chat) → empty string is the canonical sentinel."""
    from ext.services.chat_rag_bridge import _extract_last_turn_id

    assert _extract_last_turn_id([]) == ""
    assert _extract_last_turn_id([{"role": "user", "content": "hi"}]) == ""


@pytest.mark.asyncio
async def test_classify_with_qu_falls_back_when_qu_returns_none(monkeypatch):
    """If the LLM call fails (returns None), the bridge surfaces the regex result."""
    from ext.services import chat_rag_bridge as bridge
    from ext.services import query_intent as qi

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setattr(bridge, "_qu_cache_singleton", None, raising=False)
    monkeypatch.setattr(bridge, "_get_qu_cache", lambda: None)

    async def bad_invoke(*a, **kw):
        return None

    monkeypatch.setattr(qi, "_invoke_qu", bad_invoke)

    result = await bridge._classify_with_qu(
        query="compare budgets last quarter",
        history=[],
        last_turn_id="t-1",
    )
    assert result.source == "regex"
    assert result.intent == "specific"
