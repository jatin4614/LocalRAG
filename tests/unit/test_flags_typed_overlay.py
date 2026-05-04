"""Phase 3 / item 5 — flags overlay supports dict and list values
without coercing them to strings.

Required for chat_rag_bridge to read per-KB ``subtopic_keywords`` (dict)
and ``synonyms`` (list) out of the rag_config overlay.
"""
from __future__ import annotations

from ext.services import flags


def test_get_dict_returns_dict_value() -> None:
    with flags.with_overrides({"subtopic_keywords": {"visits": ["v"]}}):
        assert flags.get_dict("subtopic_keywords") == {"visits": ["v"]}


def test_get_dict_returns_default_when_unset() -> None:
    assert flags.get_dict("nonexistent_key") is None
    assert flags.get_dict("nonexistent_key", default={}) == {}


def test_get_dict_returns_default_when_value_not_dict() -> None:
    with flags.with_overrides({"foo": "string-value"}):
        # Hardened get filters non-string overlay values to fall through.
        # get_dict requires the value to be a dict.
        assert flags.get_dict("foo") is None


def test_get_list_returns_list_value() -> None:
    with flags.with_overrides({"synonyms": [["A", "B"], ["C"]]}):
        assert flags.get_list("synonyms") == [["A", "B"], ["C"]]


def test_get_list_returns_default_when_unset() -> None:
    assert flags.get_list("nonexistent_key") is None
    assert flags.get_list("nonexistent_key", default=[]) == []


def test_get_string_filters_non_string_overlay() -> None:
    """Hardened flags.get only returns string-typed overlay values;
    a dict in the overlay should fall through to env (which won't have
    the key, so we get default)."""
    with flags.with_overrides({"x": {"y": 1}}):
        assert flags.get("x", "fallback") == "fallback"


def test_with_overrides_preserves_dict_value() -> None:
    """The contextvar overlay must NOT coerce dict/list values via str()."""
    with flags.with_overrides({"subtopic_keywords": {"visits": ["v", "vi"]}}):
        # If with_overrides did `str(value)`, the dict would become its repr
        # and get_dict would return None.
        assert flags.get_dict("subtopic_keywords") == {"visits": ["v", "vi"]}
