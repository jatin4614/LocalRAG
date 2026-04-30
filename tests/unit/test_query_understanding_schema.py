"""Plan B Phase 4.3 — query_understanding module schema + parser tests."""
import json

import pytest

from ext.services.query_understanding import (
    QueryUnderstanding,
    QU_OUTPUT_SCHEMA,
    build_qu_prompt,
    parse_qu_response,
)


def test_schema_has_required_fields():
    props = QU_OUTPUT_SCHEMA["properties"]
    for required in (
        "intent",
        "resolved_query",
        "temporal_constraint",
        "entities",
        "confidence",
    ):
        assert required in props, f"schema missing field {required}"


def test_intent_enum_values():
    enum = QU_OUTPUT_SCHEMA["properties"]["intent"]["enum"]
    assert set(enum) == {"metadata", "global", "specific", "specific_date"}


def test_temporal_constraint_nullable():
    tc = QU_OUTPUT_SCHEMA["properties"]["temporal_constraint"]
    types = tc.get("anyOf") or tc.get("oneOf") or [{"type": tc.get("type")}]
    assert any(t.get("type") == "null" for t in types), (
        "temporal_constraint must allow null"
    )


def test_confidence_bounded():
    c = QU_OUTPUT_SCHEMA["properties"]["confidence"]
    assert c["type"] == "number"
    assert c["minimum"] == 0.0 and c["maximum"] == 1.0


def test_required_list_includes_all_fields():
    required = set(QU_OUTPUT_SCHEMA["required"])
    assert required == {
        "intent",
        "resolved_query",
        "temporal_constraint",
        "entities",
        "confidence",
    }


def test_build_qu_prompt_includes_query():
    prompt = build_qu_prompt(query="what changed last quarter", history=[])
    assert "what changed last quarter" in prompt


def test_build_qu_prompt_includes_history_context():
    history = [
        {"role": "user", "content": "tell me about the OFC roadmap"},
        {"role": "assistant", "content": "The OFC roadmap covers 2026-Q1 to 2027-Q1..."},
    ]
    prompt = build_qu_prompt(query="and what about Q2?", history=history)
    assert "OFC roadmap" in prompt or "previous turn" in prompt.lower()


def test_build_qu_prompt_includes_today_date():
    prompt = build_qu_prompt(query="last month", history=[])
    import datetime as dt

    today = dt.date.today().isoformat()
    assert today in prompt


def test_parse_qu_response_happy_path():
    raw = json.dumps(
        {
            "intent": "specific_date",
            "resolved_query": "outages on January 5 2026",
            "temporal_constraint": {"year": 2026, "quarter": None, "month": 1},
            "entities": ["outages"],
            "confidence": 0.92,
        }
    )
    qu = parse_qu_response(raw)
    assert qu.intent == "specific_date"
    assert qu.resolved_query == "outages on January 5 2026"
    # Phase 2.1 — parse_qu_response normalizes missing months/years to [].
    # Legacy single-month input still works; the parser fills in the array
    # fields so downstream filter helpers don't have to special-case shape.
    assert qu.temporal_constraint == {
        "year": 2026, "quarter": None, "month": 1,
        "months": [], "years": [],
    }
    assert qu.entities == ["outages"]
    assert qu.confidence == 0.92


def test_parse_qu_response_rejects_invalid_intent():
    raw = json.dumps(
        {
            "intent": "freeform",  # not in enum
            "resolved_query": "x",
            "temporal_constraint": None,
            "entities": [],
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="invalid intent"):
        parse_qu_response(raw)


def test_parse_qu_response_clamps_confidence():
    raw = json.dumps(
        {
            "intent": "specific",
            "resolved_query": "x",
            "temporal_constraint": None,
            "entities": [],
            "confidence": 1.5,  # out of range
        }
    )
    qu = parse_qu_response(raw)
    assert qu.confidence == 1.0  # clamped


def test_parse_qu_response_handles_garbage_json():
    with pytest.raises(ValueError):
        parse_qu_response("not even json {{")


def test_parse_qu_response_rejects_empty_resolved_query():
    raw = json.dumps(
        {
            "intent": "specific",
            "resolved_query": "   ",
            "temporal_constraint": None,
            "entities": [],
            "confidence": 0.5,
        }
    )
    with pytest.raises(ValueError, match="resolved_query"):
        parse_qu_response(raw)


def test_parse_qu_response_caps_entities_list():
    """The schema bounds maxItems at 10; the parser should defensively trim."""
    many = [f"e{i}" for i in range(50)]
    raw = json.dumps(
        {
            "intent": "specific",
            "resolved_query": "x",
            "temporal_constraint": None,
            "entities": many,
            "confidence": 0.5,
        }
    )
    qu = parse_qu_response(raw)
    assert len(qu.entities) <= 10


def test_query_understanding_default_source_and_cached_flag():
    qu = QueryUnderstanding(
        intent="specific",
        resolved_query="hello",
        temporal_constraint=None,
    )
    assert qu.source == "llm"
    assert qu.cached is False
