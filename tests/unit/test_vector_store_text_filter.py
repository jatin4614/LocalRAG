"""Phase 6.X (Method 4) — text_filter pass-through.

VectorStore._build_filter accepts an optional ``text_filter`` arg; when
non-empty it adds a ``must.match.text`` clause to the Qdrant filter.
Default ``None`` is byte-identical to pre-Phase-6 behaviour — every
must clause that was in the filter before is still there, and no new
clause is added.
"""
from __future__ import annotations

from ext.services.vector_store import VectorStore
from qdrant_client.http import models as qm


def _has_text_clause(flt: qm.Filter) -> str | None:
    """Return the matched text if a must.match.text clause is present."""
    for cond in (flt.must or []):
        # FieldCondition with a MatchText match value.
        match = getattr(cond, "match", None)
        if match is None:
            continue
        text = getattr(match, "text", None)
        if isinstance(text, str) and text:
            return text
    return None


class TestTextFilterClause:
    def test_default_none_no_clause(self) -> None:
        flt = VectorStore._build_filter()
        assert _has_text_clause(flt) is None

    def test_explicit_empty_string_no_clause(self) -> None:
        flt = VectorStore._build_filter(text_filter="")
        assert _has_text_clause(flt) is None

    def test_whitespace_only_no_clause(self) -> None:
        # Defensive: " " trimmed to "" → no filter sent
        flt = VectorStore._build_filter(text_filter="   ")
        assert _has_text_clause(flt) is None

    def test_text_filter_added(self) -> None:
        flt = VectorStore._build_filter(text_filter="32 Inf Bde")
        assert _has_text_clause(flt) == "32 Inf Bde"

    def test_text_filter_trimmed(self) -> None:
        flt = VectorStore._build_filter(text_filter="  32 Inf Bde  ")
        assert _has_text_clause(flt) == "32 Inf Bde"

    def test_text_filter_alongside_other_filters(self) -> None:
        flt = VectorStore._build_filter(
            subtag_ids=[1, 2],
            owner_user_id=42,
            text_filter="32 Inf Bde",
        )
        # text filter present
        assert _has_text_clause(flt) == "32 Inf Bde"
        # subtag + owner clauses still present
        keys = {
            getattr(c, "key", None) for c in (flt.must or [])
        }
        assert "subtag_id" in keys
        assert "owner_user_id" in keys
        assert "text" in keys
