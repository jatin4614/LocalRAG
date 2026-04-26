"""Plan B Phase 4.8 — shadow-mode A/B harness for the QU LLM."""
import json

import pytest


@pytest.mark.asyncio
async def test_shadow_mode_logs_both_paths(monkeypatch, caplog):
    """RAG_QU_SHADOW_MODE=1 → both regex + LLM logged; production stays regex."""
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global",
            resolved_query="what changed in 2026-Q1",
            temporal_constraint={"year": 2026, "quarter": 1, "month": None},
            entities=[],
            confidence=0.91,
        )

    monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)

    caplog.set_level("INFO", logger="orgchat.qu_shadow")
    result = await qi.classify_with_qu("what changed last quarter", history=[])

    # Production routing is still regex-only in shadow mode
    assert result.source == "regex"
    assert result.intent == "specific"

    # Shadow log emitted
    shadow_records = [r for r in caplog.records if r.name == "orgchat.qu_shadow"]
    assert len(shadow_records) == 1
    payload = json.loads(shadow_records[0].message)
    assert payload["regex_label"] == "specific"
    assert payload["llm_label"] == "global"
    assert payload["agree"] is False
    assert payload["llm_resolved_query"] == "what changed in 2026-Q1"
    assert payload["llm_temporal"] == {"year": 2026, "quarter": 1, "month": None}


@pytest.mark.asyncio
async def test_shadow_mode_off_does_not_invoke_llm_unnecessarily(monkeypatch):
    """Without shadow mode, regex-trustworthy queries skip the LLM call."""
    from ext.services import query_intent as qi

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "0")

    invoked = {"n": 0}

    async def spy_invoke(*a, **kw):
        invoked["n"] += 1
        return None

    monkeypatch.setattr(qi, "_invoke_qu", spy_invoke)
    # "list all reports" matches the regex global path → no escalation.
    await qi.classify_with_qu("list all reports", history=[])
    assert invoked["n"] == 0


@pytest.mark.asyncio
async def test_shadow_mode_invokes_llm_even_without_escalation(
    monkeypatch, caplog
):
    """Shadow mode must run the LLM on EVERY query so we observe the full
    distribution, not only the escalated subset."""
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")

    invoked = {"n": 0}

    async def fake_invoke(*a, **kw):
        invoked["n"] += 1
        return QueryUnderstanding(
            intent="metadata",
            resolved_query="list all reports",
            temporal_constraint=None,
            entities=[],
            confidence=0.9,
        )

    monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)
    caplog.set_level("INFO", logger="orgchat.qu_shadow")

    # A regex-trustworthy query — normal mode wouldn't invoke LLM
    await qi.classify_with_qu("list all reports", history=[])
    assert invoked["n"] == 1, "shadow mode must invoke LLM even on non-escalated queries"


@pytest.mark.asyncio
async def test_shadow_log_handles_llm_returning_none(monkeypatch, caplog):
    """When the LLM call fails (returns None), the shadow log still fires
    with llm_label=None so operators can quantify availability."""
    from ext.services import query_intent as qi

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")

    async def fail(*a, **kw):
        return None

    monkeypatch.setattr(qi, "_invoke_qu", fail)
    caplog.set_level("INFO", logger="orgchat.qu_shadow")
    await qi.classify_with_qu("what changed last quarter", history=[])

    records = [r for r in caplog.records if r.name == "orgchat.qu_shadow"]
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["llm_label"] is None
    assert payload["agree"] is None


def test_analyze_shadow_log_script_exists_and_executable():
    """The operator-facing analyzer script must exist + be executable."""
    import pathlib
    import stat

    script = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_shadow_log.py"
    )
    assert script.exists(), "analyze_shadow_log.py missing"
    assert script.stat().st_mode & stat.S_IXUSR, "script must be executable"
    content = script.read_text()
    assert "agree" in content
    assert "regex_label" in content


def test_analyze_shadow_log_basic_summary(tmp_path):
    """Smoke-test the analyzer produces the expected sections from a small
    synthetic log."""
    import subprocess
    import sys
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "analyze_shadow_log.py"
    )
    log = tmp_path / "shadow.log"
    log.write_text(
        json.dumps(
            {
                "regex_label": "specific",
                "llm_label": "global",
                "agree": False,
                "escalation_reason": "comparison_verb",
                "query": "compare budgets",
            }
        )
        + "\n"
        + json.dumps(
            {
                "regex_label": "metadata",
                "llm_label": "metadata",
                "agree": True,
                "escalation_reason": "none",
                "query": "list reports",
            }
        )
        + "\n"
    )
    out = subprocess.run(
        [sys.executable, str(script), str(log)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Total queries: 2" in out.stdout
    assert "Per-regex-label agreement" in out.stdout
    assert "Escalation reason breakdown" in out.stdout
