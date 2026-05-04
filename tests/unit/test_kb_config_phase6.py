"""Schema tests for Phase 6.X per-KB rag_config keys.

Methods 3 / 4 / 5 each get a per-KB toggle (bool) plus a quota knob
(int). Same contract every other rag_config key follows:

* validate_config drops unknown / out-of-range values silently
* merge_configs uses MAX for ints, OR for bools (strictest wins)
* config_to_env_overrides emits the canonical RAG_* env name
"""
from ext.services import kb_config


class TestMultiEntityDecomposeValidate:
    def test_true_kept(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_decompose": True}
        ) == {"multi_entity_decompose": True}

    def test_false_kept(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_decompose": False}
        ) == {"multi_entity_decompose": False}

    def test_string_truthy_coerced(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_decompose": "1"}
        ) == {"multi_entity_decompose": True}


class TestEntityTextFilterValidate:
    def test_true_kept(self) -> None:
        assert kb_config.validate_config(
            {"entity_text_filter": True}
        ) == {"entity_text_filter": True}


class TestQuEntityExtractValidate:
    def test_true_kept(self) -> None:
        assert kb_config.validate_config(
            {"qu_entity_extract": True}
        ) == {"qu_entity_extract": True}


class TestMinPerEntityValidate:
    def test_in_range_kept(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": 10}
        ) == {"multi_entity_min_per_entity": 10}

    def test_at_floor_kept(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": 1}
        ) == {"multi_entity_min_per_entity": 1}

    def test_at_ceiling_kept(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": 50}
        ) == {"multi_entity_min_per_entity": 50}

    def test_below_floor_dropped(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": 0}
        ) == {}

    def test_above_ceiling_dropped(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": 100}
        ) == {}

    def test_non_numeric_dropped(self) -> None:
        assert kb_config.validate_config(
            {"multi_entity_min_per_entity": "lots"}
        ) == {}


class TestPhase6Merge:
    def test_bool_or_wins(self) -> None:
        merged = kb_config.merge_configs([
            {"multi_entity_decompose": False},
            {"multi_entity_decompose": True},
        ])
        assert merged == {"multi_entity_decompose": True}

    def test_int_max_wins(self) -> None:
        merged = kb_config.merge_configs([
            {"multi_entity_min_per_entity": 5},
            {"multi_entity_min_per_entity": 12},
            {"multi_entity_min_per_entity": 8},
        ])
        assert merged == {"multi_entity_min_per_entity": 12}


class TestPhase6Overlay:
    def test_emits_decompose_env(self) -> None:
        env = kb_config.config_to_env_overrides({"multi_entity_decompose": True})
        assert env == {"RAG_MULTI_ENTITY_DECOMPOSE": "1"}

    def test_emits_text_filter_env(self) -> None:
        env = kb_config.config_to_env_overrides({"entity_text_filter": True})
        assert env == {"RAG_ENTITY_TEXT_FILTER": "1"}

    def test_emits_qu_entity_extract_env(self) -> None:
        env = kb_config.config_to_env_overrides({"qu_entity_extract": True})
        assert env == {"RAG_QU_ENTITY_EXTRACT": "1"}

    def test_emits_min_per_entity_env(self) -> None:
        env = kb_config.config_to_env_overrides(
            {"multi_entity_min_per_entity": 12}
        )
        assert env == {"RAG_MULTI_ENTITY_MIN_PER_ENTITY": "12"}

    def test_combines_all_phase6_keys(self) -> None:
        env = kb_config.config_to_env_overrides({
            "multi_entity_decompose": True,
            "entity_text_filter": True,
            "qu_entity_extract": True,
            "multi_entity_min_per_entity": 12,
        })
        assert env == {
            "RAG_MULTI_ENTITY_DECOMPOSE": "1",
            "RAG_ENTITY_TEXT_FILTER": "1",
            "RAG_QU_ENTITY_EXTRACT": "1",
            "RAG_MULTI_ENTITY_MIN_PER_ENTITY": "12",
        }

    def test_false_emits_zero_not_dropped(self) -> None:
        # Bool serialization preserves explicit OFF — important when an
        # admin disables a globally-on flag for one KB.
        env = kb_config.config_to_env_overrides(
            {"multi_entity_decompose": False}
        )
        assert env == {"RAG_MULTI_ENTITY_DECOMPOSE": "0"}


class TestMultiEntityRerankFloor:
    """Per-KB override of RAG_MULTI_ENTITY_RERANK_FLOOR env. Phase 1 of the
    2026-05-04 multi-entity-elaborate-answers spec."""

    def test_accepts_valid_int(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 8}
        ) == {"multi_entity_rerank_floor": 8}

    def test_accepts_lower_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 1}
        ) == {"multi_entity_rerank_floor": 1}

    def test_accepts_upper_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 50}
        ) == {"multi_entity_rerank_floor": 50}

    def test_rejects_below_lower_bound(self) -> None:
        from ext.services import kb_config
        # 0 silently dropped — out-of-range = inherit env default
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 0}
        ) == {}

    def test_rejects_above_upper_bound(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": 51}
        ) == {}

    def test_rejects_string(self) -> None:
        from ext.services import kb_config
        # Strings without int() coercion drop silently
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": "high"}
        ) == {}

    def test_coerces_string_int(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"multi_entity_rerank_floor": "10"}
        ) == {"multi_entity_rerank_floor": 10}

    def test_emits_rerank_floor_env(self) -> None:
        """Per-KB stamp must round-trip through config_to_env_overrides
        so flags.get sees it at the rerank-stage read site. Without this
        the per-KB override is silently dropped."""
        from ext.services import kb_config
        env = kb_config.config_to_env_overrides(
            {"multi_entity_rerank_floor": 8}
        )
        assert env == {"RAG_MULTI_ENTITY_RERANK_FLOOR": "8"}


class TestSubtopicDecompose:
    """Phase 3 / item 5 — per-KB master gate for two-axis decompose."""

    def test_accepts_true(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_decompose": True}
        ) == {"subtopic_decompose": True}

    def test_accepts_false(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_decompose": False}
        ) == {"subtopic_decompose": False}


class TestSubtopicKeywords:
    """Phase 3 / item 5 — per-KB subtopic-keywords table."""

    def test_accepts_dict_of_str_to_list(self) -> None:
        from ext.services import kb_config
        kw = {
            "visits": ["visit", "vis", "inspection"],
            "operations": ["operation", "exercise"],
        }
        assert kb_config.validate_config(
            {"subtopic_keywords": kw}
        ) == {"subtopic_keywords": kw}

    def test_rejects_non_dict(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": ["a", "b"]}
        ) == {}

    def test_rejects_dict_with_non_string_key(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {1: ["x"]}}
        ) == {}

    def test_rejects_dict_with_non_list_value(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {"visits": "not a list"}}
        ) == {}

    def test_strips_non_string_list_items(self) -> None:
        from ext.services import kb_config
        out = kb_config.validate_config(
            {"subtopic_keywords": {"visits": ["visit", 42, None, "vis"]}}
        )
        assert out == {"subtopic_keywords": {"visits": ["visit", "vis"]}}

    def test_empty_dict_accepted(self) -> None:
        from ext.services import kb_config
        assert kb_config.validate_config(
            {"subtopic_keywords": {}}
        ) == {"subtopic_keywords": {}}
