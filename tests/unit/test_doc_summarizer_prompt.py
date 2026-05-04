"""Unit tests for the structured doc-summary prompt (Phase 2 of the
2026-05-04 multi-entity-elaborate-answers spec).

The prompt asks the chat-LLM to emit two named sections:
  ENTITIES: <comma-separated list of formations named >=2 times>
  SUMMARY: <5-7 sentence paragraph that names every entity from ENTITIES>

The parser separates these into a dict so the ingest path can populate
a Qdrant payload `entities` field alongside the summary text.
"""
from __future__ import annotations

from ext.services import doc_summarizer


class TestPromptShape:
    def test_prompt_mentions_two_sections(self) -> None:
        # The prompt template is module-level; we just check shape.
        assert "ENTITIES:" in doc_summarizer._SUMMARY_PROMPT
        assert "SUMMARY:" in doc_summarizer._SUMMARY_PROMPT
        assert "{filename}" in doc_summarizer._SUMMARY_PROMPT
        assert "{body}" in doc_summarizer._SUMMARY_PROMPT

    def test_max_body_chars_raised(self) -> None:
        # Phase 2 raised this to 32000.
        assert doc_summarizer._MAX_BODY_CHARS == 32000


class TestParseStructuredOutput:
    def test_parses_well_formed_output(self) -> None:
        raw = (
            "ENTITIES: 75 Inf Bde, 5 PoK Bde, 32 Inf Bde, 80 Inf Bde\n\n"
            "SUMMARY: Document Apr 26.docx provides the fourth monthly "
            "update for April 2026. 75 Inf Bde activity centred on "
            "operational reviews. 5 PoK Bde tracked coordination meetings."
        )
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == [
            "75 Inf Bde", "5 PoK Bde", "32 Inf Bde", "80 Inf Bde"
        ]
        assert "April 2026" in out["summary"]
        assert "5 PoK Bde tracked coordination meetings." in out["summary"]

    def test_handles_missing_entities_section(self) -> None:
        # Fail-soft: if LLM only emits the summary, we keep it and emit
        # an empty entities list. Caller's responsibility to log this.
        raw = "SUMMARY: A simple summary with no entities header."
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == []
        assert "simple summary" in out["summary"]

    def test_handles_missing_summary_section(self) -> None:
        # Same — keep the entities, set summary empty.
        raw = "ENTITIES: A, B, C"
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == ["A", "B", "C"]
        assert out["summary"] == ""

    def test_returns_empty_on_blank(self) -> None:
        out = doc_summarizer.parse_structured_summary("")
        assert out == {"entities": [], "summary": ""}

    def test_strips_whitespace(self) -> None:
        raw = "  ENTITIES:   A,  B  \n\n  SUMMARY:   text here.  "
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == ["A", "B"]
        assert out["summary"] == "text here."

    def test_dedupes_entities_case_insensitively(self) -> None:
        raw = "ENTITIES: A, a, A, B\n\nSUMMARY: x"
        out = doc_summarizer.parse_structured_summary(raw)
        # First-surface-form wins — same convention as entity_extractor
        assert out["entities"] == ["A", "B"]

    def test_legacy_unmarked_treated_as_summary(self) -> None:
        """When neither ENTITIES: nor SUMMARY: marker is present, treat
        the whole input as the summary text. This is the legacy fallthrough
        path for pre-Phase-2 chat-LLM responses."""
        raw = "Just three sentences. With no headers. Pre-Phase-2 shape."
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == []
        assert out["summary"] == raw

    def test_entities_none_literal_returns_empty_list(self) -> None:
        """The prompt allows "ENTITIES: none" when no qualifying formations
        appear. Parser must return an empty list, not the literal string."""
        raw = "ENTITIES: none\n\nSUMMARY: brief text"
        out = doc_summarizer.parse_structured_summary(raw)
        assert out["entities"] == []
        assert out["summary"] == "brief text"


import pytest


class TestSummarizeDocumentReturnShape:
    """summarize_document() now returns the structured dict, not a bare string.
    Callers that want the legacy string can read the 'summary' key."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_entities_and_summary(
        self, monkeypatch
    ) -> None:
        async def fake_impl(**kwargs):
            return (
                "ENTITIES: 75 Inf Bde, 5 PoK Bde\n\n"
                "SUMMARY: Apr 26 monthly update covering 75 Inf Bde and 5 PoK Bde."
            )
        monkeypatch.setattr(doc_summarizer, "_summarize_impl", fake_impl)

        out = await doc_summarizer.summarize_document(
            chunks=["chunk 1", "chunk 2"],
            filename="Apr 26.docx",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert isinstance(out, dict)
        assert out["entities"] == ["75 Inf Bde", "5 PoK Bde"]
        assert "Apr 26 monthly update" in out["summary"]

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_shape(
        self, monkeypatch
    ) -> None:
        out = await doc_summarizer.summarize_document(
            chunks=[],
            filename="x",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert out == {"entities": [], "summary": ""}

    @pytest.mark.asyncio
    async def test_impl_failure_returns_empty_shape(
        self, monkeypatch
    ) -> None:
        async def fake_impl(**kwargs):
            return ""
        monkeypatch.setattr(doc_summarizer, "_summarize_impl", fake_impl)

        out = await doc_summarizer.summarize_document(
            chunks=["x"],
            filename="x",
            chat_url="http://fake/v1",
            chat_model="fake",
        )
        assert out == {"entities": [], "summary": ""}
