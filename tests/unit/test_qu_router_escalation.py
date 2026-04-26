"""Plan B Phase 4.4 — hybrid regex+LLM router escalation predicates."""
import pytest

from ext.services.query_intent import (
    EscalationReason,
    should_escalate_to_llm,
)


class TestEscalationPredicates:
    """should_escalate_to_llm returns (bool, EscalationReason)."""

    def test_no_escalation_for_short_specific_query_with_entity(self):
        # "OFC roadmap" has an entity-shaped token (OFC) and short
        escalate, reason = should_escalate_to_llm(
            query="show me OFC roadmap", regex_label="specific", history=[],
        )
        assert escalate is False
        assert reason is EscalationReason.NONE

    def test_no_escalation_for_metadata_label(self):
        # metadata path is trustworthy — never escalate
        escalate, reason = should_escalate_to_llm(
            query="list all reports", regex_label="metadata", history=[],
        )
        assert escalate is False

    def test_no_escalation_for_global_label(self):
        escalate, reason = should_escalate_to_llm(
            query="summarize everything", regex_label="global", history=[],
        )
        assert escalate is False

    def test_no_escalation_for_specific_date(self):
        escalate, reason = should_escalate_to_llm(
            query="outages on 5 Jan 2026", regex_label="specific_date", history=[],
        )
        assert escalate is False

    def test_escalation_for_pronoun_with_history(self):
        history = [
            {"role": "user", "content": "tell me about OFC roadmap"},
            {"role": "assistant", "content": "OFC roadmap covers..."},
        ]
        escalate, reason = should_escalate_to_llm(
            query="and what about it in Q2?", regex_label="specific", history=history,
        )
        assert escalate is True
        assert reason is EscalationReason.PRONOUN_REF

    def test_no_escalation_for_pronoun_without_history(self):
        # Pronoun without antecedent is meaningless — the pronoun gate is the
        # only predicate that depends on history. We use a capitalized token
        # ("OFC") to suppress the NO_ENTITY predicate so this test isolates
        # the history-aware pronoun gate.
        escalate, reason = should_escalate_to_llm(
            query="show me OFC and tell me about it",
            regex_label="specific",
            history=[],
        )
        assert escalate is False
        assert reason is EscalationReason.NONE

    def test_escalation_for_relative_time(self):
        escalate, reason = should_escalate_to_llm(
            query="what happened last quarter?", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.RELATIVE_TIME

    def test_escalation_for_yesterday(self):
        escalate, reason = should_escalate_to_llm(
            query="yesterday's incidents", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.RELATIVE_TIME

    def test_escalation_for_long_query(self):
        long = " ".join(["word"] * 30)
        escalate, reason = should_escalate_to_llm(
            query=long, regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.LONG_QUERY

    def test_escalation_for_multi_clause(self):
        escalate, reason = should_escalate_to_llm(
            query="show me the roadmap and explain how it changed in Q2",
            regex_label="specific",
            history=[],
        )
        assert escalate is True
        # Either MULTI_CLAUSE / RELATIVE_TIME / PRONOUN_REF — first match wins
        assert reason in (
            EscalationReason.MULTI_CLAUSE,
            EscalationReason.RELATIVE_TIME,
            EscalationReason.PRONOUN_REF,
        )

    def test_escalation_for_question_no_entity(self):
        # "what changed" — question word, no capitalized entity
        escalate, reason = should_escalate_to_llm(
            query="what changed?", regex_label="specific", history=[],
        )
        assert escalate is True

    def test_escalation_for_comparison_verb(self):
        escalate, reason = should_escalate_to_llm(
            query="compare the budgets", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.COMPARISON_VERB

    def test_no_escalation_for_empty_query(self):
        escalate, reason = should_escalate_to_llm(
            query="", regex_label="specific", history=[],
        )
        assert escalate is False
        assert reason is EscalationReason.NONE


class TestHybridClassify:
    """classify_with_qu wraps regex + escalate + analyze_query."""

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_qu_disabled(self, monkeypatch):
        from ext.services.query_intent import classify_with_qu

        monkeypatch.setenv("RAG_QU_ENABLED", "0")
        result = await classify_with_qu("compare budgets", history=[])
        assert result.intent == "specific"
        assert result.source == "regex"

    @pytest.mark.asyncio
    async def test_uses_qu_when_enabled_and_escalated(self, monkeypatch):
        from ext.services import query_intent as qi
        from ext.services.query_understanding import QueryUnderstanding

        async def fake_invoke(query, history, **kw):
            return QueryUnderstanding(
                intent="global",
                resolved_query="compare budgets across all years",
                temporal_constraint=None,
                entities=["budgets"],
                confidence=0.95,
                source="llm",
                cached=False,
            )

        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)
        result = await qi.classify_with_qu("compare budgets", history=[])
        assert result.intent == "global"
        assert result.source == "llm"
        assert "across all years" in result.resolved_query

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_qu_returns_none(self, monkeypatch):
        from ext.services import query_intent as qi

        async def fake_invoke(*a, **kw):
            return None  # simulate timeout / HTTP error

        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)
        result = await qi.classify_with_qu("compare budgets", history=[])
        assert result.intent == "specific"
        assert result.source == "regex"

    @pytest.mark.asyncio
    async def test_does_not_escalate_for_non_specific_label(self, monkeypatch):
        """metadata + global + specific_date results bypass the LLM path."""
        from ext.services import query_intent as qi

        called = {"n": 0}

        async def spy_invoke(*a, **kw):
            called["n"] += 1
            return None

        monkeypatch.setattr(qi, "_invoke_qu", spy_invoke)
        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        # "list all reports" matches the global:list_all_every regex, not
        # metadata — but either way it must NOT escalate to the LLM.
        result = await qi.classify_with_qu("list all reports", history=[])
        assert result.intent == "global"
        assert result.source == "regex"
        assert called["n"] == 0, "QU must not be invoked for non-specific queries"

    @pytest.mark.asyncio
    async def test_keeps_regex_when_qu_confidence_too_low(self, monkeypatch):
        """If the LLM is unsure (confidence < 0.5), trust regex."""
        from ext.services import query_intent as qi
        from ext.services.query_understanding import QueryUnderstanding

        async def fake_invoke(*a, **kw):
            return QueryUnderstanding(
                intent="global",
                resolved_query="compare budgets",
                temporal_constraint=None,
                entities=[],
                confidence=0.3,
                source="llm",
            )

        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        monkeypatch.setattr(qi, "_invoke_qu", fake_invoke)
        result = await qi.classify_with_qu("compare budgets", history=[])
        assert result.source == "regex"
        assert result.intent == "specific"

    def test_classify_with_reason_no_longer_calls_legacy_llm(self, monkeypatch):
        """Plan B Phase 4.4 retires the _llm_classify stub.

        classify_with_reason must still return the regex result when
        RAG_INTENT_LLM=1 and never raise NotImplementedError.
        """
        from ext.services.query_intent import classify_with_reason

        monkeypatch.setenv("RAG_INTENT_LLM", "1")
        label, reason = classify_with_reason("compare budgets")
        assert label == "specific"
        # The reason must be the default (no_pattern_matched), proving the
        # legacy LLM branch is gone.
        assert "no_pattern_matched" in reason
