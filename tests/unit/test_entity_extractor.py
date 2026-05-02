"""Unit tests for ext.services.entity_extractor (Phase 6.X — Method 5).

Backstop for the multi-entity retrieval failure (32 Inf Bde eviction
case, 2026-05-01). The extractor produces a list of named entities a
multi-query decomposer can fan out on. Two paths:

1. **Regex** — numbered lists, "for the following:" markers, comma+and
   lists. Always available, zero infra dependency.
2. **QU LLM** — re-uses entities already extracted by the existing
   ``analyze_query`` / ``classify_with_qu`` path (Plan B Phase 4).
   No new LLM call.

Composer contract:
- prefers QU entities when present (>=1)
- else falls back to regex
- dedupes by case-insensitive key, preserves first surface form
- caps at 8 entities (bounds asyncio.gather fan-out in Method 3)
"""
from __future__ import annotations

from ext.services import entity_extractor


class TestRegexNumberedList:
    def test_numbered_list_with_period(self) -> None:
        q = "Updates for:\n1. 75 Inf Bde\n2. 5 PoK Bde\n3. 32 Inf Bde\n4. 80 Inf Bde"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["75 Inf Bde", "5 PoK Bde", "32 Inf Bde", "80 Inf Bde"]

    def test_numbered_list_no_period_actual_user_query(self) -> None:
        # The exact pattern that produced the 32 Inf Bde miss (Example 1).
        q = (
            "Give out major updates from the report of apr 2026, for the following :\n"
            "1 75 INf bde\n"
            "2.  5 PoK bde\n"
            "3.  32 Inf Bde\n"
            "4.  80 INfe BDe\n"
            "I want answers under fwg heads"
        )
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["75 INf bde", "5 PoK bde", "32 Inf Bde", "80 INfe BDe"]

    def test_numbered_list_with_paren(self) -> None:
        q = "items:\n1) Alpha\n2) Bravo\n3) Charlie"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["Alpha", "Bravo", "Charlie"]

    def test_single_numbered_item_returns_empty(self) -> None:
        # < 2 entities = not a multi-entity query
        q = "Tell me about:\n1. Just one thing"
        assert entity_extractor.extract_entities_regex(q) == []

    def test_zero_numbered_items_returns_empty(self) -> None:
        q = "What is the weather today?"
        assert entity_extractor.extract_entities_regex(q) == []

    def test_max_8_cap(self) -> None:
        lines = [f"{i}. Entity{i}" for i in range(1, 11)]
        q = "list:\n" + "\n".join(lines)
        out = entity_extractor.extract_entities_regex(q)
        assert len(out) == 8
        assert out[0] == "Entity1"
        assert out[-1] == "Entity8"

    def test_strips_trailing_punctuation(self) -> None:
        q = "items:\n1. First,\n2. Second.\n3. Third;"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["First", "Second", "Third"]


class TestRegexBulletList:
    def test_dash_bullets(self) -> None:
        q = "Items:\n- Apple\n- Banana\n- Cherry"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["Apple", "Banana", "Cherry"]

    def test_asterisk_bullets(self) -> None:
        q = "Items:\n* Apple\n* Banana\n* Cherry"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["Apple", "Banana", "Cherry"]


class TestRegexCommaList:
    def test_comma_with_oxford_and(self) -> None:
        # Triggered by "for/about/regarding" preamble keyword
        q = "Give updates for 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, and 80 Inf Bde"
        out = entity_extractor.extract_entities_regex(q)
        assert "75 Inf Bde" in out
        assert "5 PoK Bde" in out
        assert "32 Inf Bde" in out
        assert "80 Inf Bde" in out

    def test_comma_without_preamble_returns_empty(self) -> None:
        # Without a list-signaling preamble, comma-splitting whole query
        # is too noisy. We require an explicit list marker.
        q = "I went to A, B, C and slept."
        assert entity_extractor.extract_entities_regex(q) == []


class TestRegexNoSignal:
    def test_plain_question_returns_empty(self) -> None:
        assert entity_extractor.extract_entities_regex("How is the weather?") == []

    def test_single_entity_query_returns_empty(self) -> None:
        # Example 2 actual query — single entity, no decomposition needed.
        q = "Give me complete updates of 32 inf bde for the month of Apr 2026"
        assert entity_extractor.extract_entities_regex(q) == []

    def test_empty_query(self) -> None:
        assert entity_extractor.extract_entities_regex("") == []

    def test_none_query(self) -> None:
        assert entity_extractor.extract_entities_regex(None) == []


class TestRegexDedup:
    def test_duplicate_dedupes_case_insensitive(self) -> None:
        q = "items:\n1. 32 Inf Bde\n2. 32 INF BDE\n3. 32 inf bde"
        out = entity_extractor.extract_entities_regex(q)
        # All three collapse to one; dedupe wins.
        assert out == ["32 Inf Bde"]

    def test_preserves_first_surface_form(self) -> None:
        q = "items:\n1. 75 Inf Bde\n2. 75 inf bde"
        out = entity_extractor.extract_entities_regex(q)
        assert out == ["75 Inf Bde"]


class _FakeQU:
    """Stand-in for HybridClassification — we only consume .entities."""

    def __init__(self, entities: list[str]) -> None:
        self.entities = entities


class TestExtractEntitiesComposer:
    def test_qu_entities_used_when_present(self) -> None:
        qu = _FakeQU(entities=["75 Inf Bde", "32 Inf Bde", "80 Inf Bde"])
        # Even with regex-detectable list in query, QU wins.
        q = "Updates for:\n1. Alpha\n2. Bravo"
        out = entity_extractor.extract_entities(q, qu_result=qu)
        assert out == ["75 Inf Bde", "32 Inf Bde", "80 Inf Bde"]

    def test_qu_empty_falls_back_to_regex(self) -> None:
        qu = _FakeQU(entities=[])
        q = "Updates for:\n1. Alpha\n2. Bravo"
        out = entity_extractor.extract_entities(q, qu_result=qu)
        assert out == ["Alpha", "Bravo"]

    def test_qu_none_falls_back_to_regex(self) -> None:
        q = "Updates for:\n1. Alpha\n2. Bravo"
        out = entity_extractor.extract_entities(q, qu_result=None)
        assert out == ["Alpha", "Bravo"]

    def test_qu_capped_at_8(self) -> None:
        ents = [f"E{i}" for i in range(20)]
        qu = _FakeQU(entities=ents)
        out = entity_extractor.extract_entities("anything", qu_result=qu)
        assert len(out) == 8

    def test_qu_dedupes(self) -> None:
        qu = _FakeQU(entities=["32 Inf Bde", "32 INF BDE", "75 Inf Bde"])
        out = entity_extractor.extract_entities("anything", qu_result=qu)
        assert out == ["32 Inf Bde", "75 Inf Bde"]

    def test_qu_strips_whitespace(self) -> None:
        qu = _FakeQU(entities=["  32 Inf Bde  ", "\t75 Inf Bde\n"])
        out = entity_extractor.extract_entities("anything", qu_result=qu)
        assert out == ["32 Inf Bde", "75 Inf Bde"]

    def test_qu_drops_empty_strings(self) -> None:
        qu = _FakeQU(entities=["", "  ", "32 Inf Bde", None])
        out = entity_extractor.extract_entities("anything", qu_result=qu)
        assert out == ["32 Inf Bde"]

    def test_qu_attribute_missing_falls_back_to_regex(self) -> None:
        # Defensive: caller passed something without .entities — soft-fail.
        class Empty:
            pass
        out = entity_extractor.extract_entities(
            "items:\n1. A\n2. B", qu_result=Empty()
        )
        assert out == ["A", "B"]


class TestIsMultiEntityQuery:
    def test_two_or_more_returns_true(self) -> None:
        assert entity_extractor.is_multi_entity_query(
            "items:\n1. A\n2. B"
        ) is True

    def test_zero_or_one_returns_false(self) -> None:
        assert entity_extractor.is_multi_entity_query("plain question") is False
        assert entity_extractor.is_multi_entity_query(
            "items:\n1. only one"
        ) is False

    def test_qu_overrides(self) -> None:
        qu = _FakeQU(entities=["A", "B", "C"])
        assert entity_extractor.is_multi_entity_query(
            "plain question", qu_result=qu
        ) is True
