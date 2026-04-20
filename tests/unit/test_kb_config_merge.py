"""P3.0 — per-KB rag_config merge + env-override mapping.

Unit tests for ``ext.services.kb_config``:

* ``merge_configs`` — UNION/MAX policy across selected KBs' configs.
* ``config_to_env_overrides`` — JSON key -> RAG_* env var translation.
* ``validate_config`` — whitelist + type coercion on admin PATCH input.
"""
from __future__ import annotations

import pytest

from ext.services.kb_config import (
    INGEST_ONLY_KEYS,
    VALID_KEYS,
    config_to_env_overrides,
    merge_configs,
    validate_config,
)


# ---------------------------------------------------------------------------
# merge_configs
# ---------------------------------------------------------------------------

def test_empty_list_returns_empty_dict():
    assert merge_configs([]) == {}


def test_all_empty_configs_returns_empty_dict():
    assert merge_configs([{}, {}, {}]) == {}


def test_single_config_is_pass_through():
    cfg = {"rerank": True, "context_expand_window": 2}
    assert merge_configs([cfg]) == {"rerank": True, "context_expand_window": 2}


def test_two_configs_one_rerank_true_unions_to_true():
    a = {"rerank": False}
    b = {"rerank": True}
    out = merge_configs([a, b])
    assert out["rerank"] is True


def test_two_configs_both_false_stays_false():
    out = merge_configs([{"rerank": False}, {"rerank": False}])
    # Both explicitly false → merged is explicitly False.
    assert out.get("rerank") is False


def test_numeric_max_wins_for_expand_window():
    a = {"context_expand_window": 1}
    b = {"context_expand_window": 3}
    out = merge_configs([a, b])
    assert out["context_expand_window"] == 3


def test_numeric_max_wins_rerank_top_k():
    out = merge_configs([{"rerank_top_k": 10}, {"rerank_top_k": 50}, {"rerank_top_k": 25}])
    assert out["rerank_top_k"] == 50


def test_float_max_wins_mmr_lambda():
    out = merge_configs([{"mmr_lambda": 0.3}, {"mmr_lambda": 0.8}])
    assert out["mmr_lambda"] == 0.8


def test_unknown_keys_filtered_out():
    cfg = {"rerank": True, "not_a_real_key": "value", "hack": {"nested": True}}
    out = merge_configs([cfg])
    assert out == {"rerank": True}


def test_none_entries_are_skipped():
    out = merge_configs([None, {"rerank": True}, None])  # type: ignore[list-item]
    assert out == {"rerank": True}


def test_merge_preserves_independent_keys():
    a = {"rerank": True, "mmr_lambda": 0.5}
    b = {"mmr": True, "context_expand_window": 2}
    out = merge_configs([a, b])
    assert out == {
        "rerank": True,
        "mmr_lambda": 0.5,
        "mmr": True,
        "context_expand_window": 2,
    }


def test_merge_all_valid_keys():
    cfg = {
        "rerank": True,
        "rerank_top_k": 30,
        "mmr": True,
        "mmr_lambda": 0.7,
        "context_expand": True,
        "context_expand_window": 2,
        "spotlight": True,
        "semcache": True,
        "contextualize_on_ingest": True,
    }
    out = merge_configs([cfg])
    for key in VALID_KEYS:
        assert key in out


# ---------------------------------------------------------------------------
# config_to_env_overrides
# ---------------------------------------------------------------------------

def test_rerank_true_maps_to_rag_rerank_1():
    assert config_to_env_overrides({"rerank": True}) == {"RAG_RERANK": "1"}


def test_rerank_false_maps_to_rag_rerank_0():
    assert config_to_env_overrides({"rerank": False}) == {"RAG_RERANK": "0"}


def test_all_bool_keys_mapping():
    overrides = config_to_env_overrides({
        "rerank": True,
        "mmr": True,
        "context_expand": True,
        "spotlight": True,
        "semcache": True,
    })
    assert overrides == {
        "RAG_RERANK": "1",
        "RAG_MMR": "1",
        "RAG_CONTEXT_EXPAND": "1",
        "RAG_SPOTLIGHT": "1",
        "RAG_SEMCACHE": "1",
    }


def test_int_keys_stringified():
    overrides = config_to_env_overrides({
        "rerank_top_k": 30,
        "context_expand_window": 2,
    })
    assert overrides == {
        "RAG_RERANK_TOP_K": "30",
        "RAG_CONTEXT_EXPAND_WINDOW": "2",
    }


def test_float_keys_stringified():
    overrides = config_to_env_overrides({"mmr_lambda": 0.7})
    assert overrides == {"RAG_MMR_LAMBDA": "0.7"}


def test_ingest_only_keys_not_in_request_overrides():
    """contextualize_on_ingest is an ingest-side flag; the request-scoped
    overlay must NOT carry it (confuses operators + no effect anyway)."""
    overrides = config_to_env_overrides({"contextualize_on_ingest": True})
    assert "RAG_CONTEXTUALIZE_KBS" not in overrides
    assert overrides == {}


def test_empty_config_gives_empty_overrides():
    assert config_to_env_overrides({}) == {}


def test_unknown_keys_ignored_in_mapping():
    """Defense-in-depth: even if validate_config didn't clean the input,
    config_to_env_overrides never produces env vars for unknown keys."""
    out = config_to_env_overrides({"foo": True, "rerank": False})
    assert out == {"RAG_RERANK": "0"}


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def test_validate_strips_unknown_keys():
    out = validate_config({"rerank": True, "malicious": "value"})
    assert out == {"rerank": True}


def test_validate_coerces_string_true_to_bool():
    out = validate_config({"rerank": "true"})
    assert out["rerank"] is True


def test_validate_coerces_int_to_bool():
    out = validate_config({"rerank": 1})
    assert out["rerank"] is True
    out = validate_config({"rerank": 0})
    assert out["rerank"] is False


def test_validate_coerces_int_key_type():
    out = validate_config({"rerank_top_k": "30"})
    assert out["rerank_top_k"] == 30


def test_validate_drops_uncoerceable_int():
    out = validate_config({"rerank_top_k": "abc"})
    assert "rerank_top_k" not in out


def test_validate_drops_uncoerceable_float():
    out = validate_config({"mmr_lambda": "not-a-number"})
    assert "mmr_lambda" not in out


def test_validate_accepts_all_whitelisted_keys():
    raw = {key: (True if key in {"rerank", "mmr", "context_expand", "spotlight",
                                 "semcache", "contextualize_on_ingest"}
                 else 1 if key in {"rerank_top_k", "context_expand_window"}
                 else 0.5)
           for key in VALID_KEYS}
    out = validate_config(raw)
    assert set(out.keys()) == set(VALID_KEYS)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def test_merge_then_to_env_realistic():
    """Two KBs: a small FAQ kb wants everything off; a year-long docs kb
    wants rerank + expand window=2. Result should be 'everything the big
    kb needs'."""
    small_kb = {"rerank": False}
    big_kb = {"rerank": True, "context_expand": True, "context_expand_window": 2}
    merged = merge_configs([small_kb, big_kb])
    overrides = config_to_env_overrides(merged)
    assert overrides["RAG_RERANK"] == "1"
    assert overrides["RAG_CONTEXT_EXPAND"] == "1"
    assert overrides["RAG_CONTEXT_EXPAND_WINDOW"] == "2"


def test_ingest_key_passes_merge_but_not_overlay():
    """contextualize_on_ingest should be preserved in the merged dict
    (admin UIs may want to show it) but filtered out of the request
    overlay."""
    merged = merge_configs([{"contextualize_on_ingest": True, "rerank": True}])
    assert merged["contextualize_on_ingest"] is True
    overrides = config_to_env_overrides(merged)
    assert "RAG_CONTEXTUALIZE_KBS" not in overrides
    assert overrides == {"RAG_RERANK": "1"}


def test_ingest_only_keys_declared():
    """Sanity check: INGEST_ONLY_KEYS is a subset of VALID_KEYS."""
    assert INGEST_ONLY_KEYS <= VALID_KEYS
