"""Unit tests for per-KB ``top_k`` (option B for the 32 Inf Bde
eviction case 2026-05-01). Mirrors the structure of
test_kb_config_chunk_tokens.py.

Schema contract:
- top_k accepted as int, bounded [1, 200]; out-of-range dropped silently
- merges across KBs with MAX (strictest wins, like other ints)
- emitted into the env overlay as RAG_TOP_K
"""
from ext.services import kb_config


class TestTopKValidate:
    def test_in_range_kept(self) -> None:
        assert kb_config.validate_config({"top_k": 24}) == {"top_k": 24}

    def test_at_floor_kept(self) -> None:
        assert kb_config.validate_config({"top_k": 1}) == {"top_k": 1}

    def test_at_ceiling_kept(self) -> None:
        assert kb_config.validate_config({"top_k": 200}) == {"top_k": 200}

    def test_below_floor_dropped(self) -> None:
        assert kb_config.validate_config({"top_k": 0}) == {}

    def test_above_ceiling_dropped(self) -> None:
        assert kb_config.validate_config({"top_k": 500}) == {}

    def test_non_numeric_dropped(self) -> None:
        assert kb_config.validate_config({"top_k": "lots"}) == {}

    def test_coexists_with_rerank_top_k(self) -> None:
        out = kb_config.validate_config({"top_k": 24, "rerank_top_k": 30})
        assert out == {"top_k": 24, "rerank_top_k": 30}


class TestTopKMerge:
    def test_max_wins_across_kbs(self) -> None:
        merged = kb_config.merge_configs([{"top_k": 12}, {"top_k": 24}, {"top_k": 18}])
        assert merged == {"top_k": 24}

    def test_single_kb_passthrough(self) -> None:
        assert kb_config.merge_configs([{"top_k": 24}]) == {"top_k": 24}

    def test_empty_when_no_top_k(self) -> None:
        merged = kb_config.merge_configs([{"rerank": True}])
        assert "top_k" not in merged


class TestTopKOverlay:
    def test_emits_RAG_TOP_K(self) -> None:
        env = kb_config.config_to_env_overrides({"top_k": 24})
        assert env == {"RAG_TOP_K": "24"}

    def test_combines_with_rerank_top_k(self) -> None:
        env = kb_config.config_to_env_overrides({"top_k": 24, "rerank_top_k": 24})
        assert env == {"RAG_TOP_K": "24", "RAG_RERANK_TOP_K": "24"}
