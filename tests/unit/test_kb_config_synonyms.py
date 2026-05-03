"""Phase 2 / Item 4 — kb_config.expand_entity() and the new
entity_text_filter_mode / synonyms keys."""
import pytest

from ext.services.kb_config import (
    expand_entity,
    merge_configs,
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
    """User can type with any casing; class match is case-insensitive.
    Returns the entire class plus the input form (set semantics dedup
    when input is already in the class)."""
    classes = [["5 PoK", "5 POK", "Pakistan-Occupied Kashmir"]]
    # Input "5 pok" is NOT literally in the class — gets added to result
    assert expand_entity("5 pok", classes) == {
        "5 pok", "5 PoK", "5 POK", "Pakistan-Occupied Kashmir"
    }
    # Input "5 POK" IS in the class — set dedupes, result equals the class exactly
    assert expand_entity("5 POK", classes) == {
        "5 PoK", "5 POK", "Pakistan-Occupied Kashmir"
    }


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


def test_merge_configs_unions_synonyms_across_kbs():
    """Multi-KB queries with overlapping synonym classes don't lose variants."""
    cfg_a = {"synonyms": [["5 PoK", "5 POK"]]}
    cfg_b = {"synonyms": [["5 PoK", "Pakistan-Occupied Kashmir"]]}
    merged = merge_configs([cfg_a, cfg_b])
    # Both classes' variants must be reachable. Either separate classes
    # in the merged list or merged into one — either is fine, as long as
    # expand_entity("5 PoK", merged["synonyms"]) recovers all 3 variants.
    out = expand_entity("5 PoK", merged["synonyms"])
    assert {"5 PoK", "5 POK", "Pakistan-Occupied Kashmir"} <= out
