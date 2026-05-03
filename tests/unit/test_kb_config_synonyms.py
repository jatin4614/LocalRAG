"""Phase 2 / Item 4 — kb_config.expand_entity() and the new
entity_text_filter_mode / synonyms keys."""
import pytest

from ext.services.kb_config import (
    expand_entity,
    validate_config,
    config_to_env_overrides,
    VALID_BOOL_KEYS,
    VALID_KEYS,
)


def test_expand_entity_basic():
    """Entity in a class returns the whole class."""
    classes = [
        ["5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"],
        ["Inf Bde", "Infantry Brigade"],
    ]
    out = expand_entity("5 PoK", classes)
    assert out == {
        "5 PoK", "5 POK", "5 PoK Bde", "5 POK Bde", "Pakistan-Occupied Kashmir"
    }


def test_expand_entity_case_insensitive_membership():
    """User can type with any casing; class match is case-insensitive."""
    classes = [["5 PoK", "5 POK", "Pakistan-Occupied Kashmir"]]
    assert "Pakistan-Occupied Kashmir" in expand_entity("5 pok", classes)
    assert "5 PoK" in expand_entity("5 POK", classes)


def test_expand_entity_not_in_any_class_returns_self_only():
    """Entity not matched anywhere returns just the input."""
    classes = [["5 PoK", "5 POK"]]
    out = expand_entity("80 Inf Bde", classes)
    assert out == {"80 Inf Bde"}


def test_expand_entity_empty_classes_returns_self_only():
    out = expand_entity("75 Inf", [])
    assert out == {"75 Inf"}


def test_validate_config_accepts_entity_text_filter_mode():
    cfg = validate_config({"entity_text_filter_mode": "boost"})
    assert cfg == {"entity_text_filter_mode": "boost"}


def test_validate_config_drops_invalid_entity_text_filter_mode():
    """Only 'filter' or 'boost' accepted; other values silently dropped."""
    cfg = validate_config({"entity_text_filter_mode": "explode"})
    assert cfg == {}


def test_validate_config_accepts_synonyms_array():
    raw = {"synonyms": [["5 PoK", "5 POK"], ["Inf Bde", "Infantry Brigade"]]}
    cfg = validate_config(raw)
    assert cfg == {"synonyms": [["5 PoK", "5 POK"], ["Inf Bde", "Infantry Brigade"]]}


def test_validate_config_drops_malformed_synonyms():
    """Non-list synonyms or non-list-of-lists shapes are dropped."""
    assert validate_config({"synonyms": "not a list"}) == {}
    assert validate_config({"synonyms": ["not a class"]}) == {}
    assert validate_config({"synonyms": [["valid"], "bad"]}) == {}


def test_config_to_env_overrides_serializes_mode():
    """entity_text_filter_mode flows into RAG_ENTITY_TEXT_FILTER_MODE env."""
    out = config_to_env_overrides({"entity_text_filter_mode": "boost"})
    assert out == {"RAG_ENTITY_TEXT_FILTER_MODE": "boost"}
