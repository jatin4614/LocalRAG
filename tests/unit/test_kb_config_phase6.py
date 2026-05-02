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
