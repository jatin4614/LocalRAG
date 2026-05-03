"""Phase 1 / Item 1 — _build_filter must lowercase the entity-text filter
so MatchText matches the same chunks regardless of the user's casing.

Verifies the query-side defense in depth (the index-side fix lives in
scripts/apply_text_index.py — see test_apply_text_index.py).
"""
from ext.services.vector_store import VectorStore


def test_build_filter_lowercases_text_filter():
    """Same entity in three casings must produce the same MatchText payload."""
    vs = VectorStore.__new__(VectorStore)  # bypass __init__ for unit test
    f1 = vs._build_filter(text_filter="75 INF bde")
    f2 = vs._build_filter(text_filter="75 Inf Bde")
    f3 = vs._build_filter(text_filter="75 inf bde")
    # All three must produce a Filter with a single MatchText("75 inf bde").
    for f in (f1, f2, f3):
        # _build_filter returns Filter or None; must contain a text condition
        assert f is not None
        text_conditions = [
            c for c in f.must
            if hasattr(c, "key") and c.key == "text"
        ]
        assert len(text_conditions) == 1
        match_text = text_conditions[0].match.text
        # Post-fix: lowercased; pre-fix: original casing preserved
        assert match_text == "75 inf bde", (
            f"expected lowercased text, got {match_text!r}"
        )


def test_build_filter_text_filter_strips_whitespace_and_lowercases():
    """Trailing/leading whitespace plus mixed case both normalised."""
    vs = VectorStore.__new__(VectorStore)
    f = vs._build_filter(text_filter="  5 PoK Bde  ")
    text_conditions = [c for c in f.must if c.key == "text"]
    assert text_conditions[0].match.text == "5 pok bde"


def test_build_filter_empty_text_filter_no_op():
    """Empty / whitespace-only filter doesn't add a text MUST clause."""
    vs = VectorStore.__new__(VectorStore)
    for empty in ("", "   ", None):
        f = vs._build_filter(text_filter=empty)
        if f is None:
            continue
        text_conditions = [
            c for c in (f.must or [])
            if hasattr(c, "key") and c.key == "text"
        ]
        assert text_conditions == []
