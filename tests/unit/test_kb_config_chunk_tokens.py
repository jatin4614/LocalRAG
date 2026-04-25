"""Unit tests for per-KB chunk_tokens/overlap_tokens in kb_config."""
from __future__ import annotations

import os
from unittest import mock

import pytest

from ext.services import kb_config


class TestValidateConfigChunkKeys:
    def test_chunk_tokens_in_range_is_kept(self) -> None:
        out = kb_config.validate_config({"chunk_tokens": 300})
        assert out == {"chunk_tokens": 300}

    def test_chunk_tokens_below_floor_is_dropped(self) -> None:
        assert kb_config.validate_config({"chunk_tokens": 50}) == {}

    def test_chunk_tokens_above_ceiling_is_dropped(self) -> None:
        assert kb_config.validate_config({"chunk_tokens": 3000}) == {}

    def test_chunk_tokens_at_floor_is_kept(self) -> None:
        assert kb_config.validate_config({"chunk_tokens": 100}) == {"chunk_tokens": 100}

    def test_chunk_tokens_at_ceiling_is_kept(self) -> None:
        assert kb_config.validate_config({"chunk_tokens": 2000}) == {"chunk_tokens": 2000}

    def test_overlap_tokens_zero_is_kept(self) -> None:
        assert kb_config.validate_config({"overlap_tokens": 0}) == {"overlap_tokens": 0}

    def test_overlap_tokens_negative_is_dropped(self) -> None:
        assert kb_config.validate_config({"overlap_tokens": -1}) == {}

    def test_overlap_tokens_way_too_big_is_dropped(self) -> None:
        assert kb_config.validate_config({"overlap_tokens": 5000}) == {}

    def test_non_numeric_chunk_tokens_dropped(self) -> None:
        assert kb_config.validate_config({"chunk_tokens": "not-a-number"}) == {}

    def test_chunk_keys_with_other_valid_keys_coexist(self) -> None:
        out = kb_config.validate_config({
            "rerank": True, "chunk_tokens": 400, "overlap_tokens": 50,
        })
        assert out == {"rerank": True, "chunk_tokens": 400, "overlap_tokens": 50}

    def test_chunk_keys_not_applied_as_env_overrides(self) -> None:
        merged = {"chunk_tokens": 300, "overlap_tokens": 50, "rerank": True}
        env = kb_config.config_to_env_overrides(merged)
        # chunk_tokens is INGEST-ONLY — should NOT land in the overlay.
        assert "chunk_tokens" not in env
        assert "overlap_tokens" not in env
        # Retrieval-time keys still map.
        assert env.get("RAG_RERANK") == "1"


class TestResolveChunkParams:
    def test_empty_config_uses_env_defaults(self) -> None:
        ct, ov = kb_config.resolve_chunk_params(
            None, env_chunk_size=800, env_chunk_overlap=100
        )
        assert (ct, ov) == (800, 100)

    def test_explicit_config_wins_over_env(self) -> None:
        ct, ov = kb_config.resolve_chunk_params(
            {"chunk_tokens": 300, "overlap_tokens": 50},
            env_chunk_size=800, env_chunk_overlap=100,
        )
        assert (ct, ov) == (300, 50)

    def test_partial_config_inherits_other_from_env(self) -> None:
        # Only chunk_tokens set → overlap inherits env.
        ct, ov = kb_config.resolve_chunk_params(
            {"chunk_tokens": 300},
            env_chunk_size=800, env_chunk_overlap=150,
        )
        assert (ct, ov) == (300, 150)

    def test_overlap_greater_than_chunk_is_clipped_down(self) -> None:
        # Safety net — validate_config normally rejects this, but if it
        # slipped through (e.g. hand-edited JSONB), the resolver clips.
        ct, ov = kb_config.resolve_chunk_params(
            {"chunk_tokens": 200, "overlap_tokens": 400},  # invalid pair
        )
        # overlap clipped to chunk_tokens // 4 = 50
        assert ct == 200
        assert ov == 50
        assert ov < ct

    def test_reads_os_env_when_kwargs_not_supplied(self) -> None:
        with mock.patch.dict(
            os.environ, {"CHUNK_SIZE": "400", "CHUNK_OVERLAP": "75"}, clear=False
        ):
            ct, ov = kb_config.resolve_chunk_params(None)
            assert (ct, ov) == (400, 75)

    def test_hard_default_when_env_unset_and_no_kwargs(self) -> None:
        # Remove env vars for this test
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CHUNK_SIZE", None)
            os.environ.pop("CHUNK_OVERLAP", None)
            ct, ov = kb_config.resolve_chunk_params(None)
            assert (ct, ov) == (800, 100)

    def test_invalid_config_values_dropped_before_resolve(self) -> None:
        # validate_config drops chunk_tokens=50 (below 100 floor) → falls
        # back to env default.
        ct, _ = kb_config.resolve_chunk_params(
            {"chunk_tokens": 50}, env_chunk_size=800, env_chunk_overlap=100
        )
        assert ct == 800

    def test_non_numeric_env_falls_back_to_hard_default(self) -> None:
        with mock.patch.dict(
            os.environ, {"CHUNK_SIZE": "not-an-int"}, clear=False
        ):
            os.environ.pop("CHUNK_OVERLAP", None)
            ct, ov = kb_config.resolve_chunk_params(None)
            assert (ct, ov) == (800, 100)


class TestMergeWithChunkKeys:
    def test_chunk_keys_are_not_merged_across_kbs(self) -> None:
        """chunk_tokens is ingest-time per-KB; it does NOT participate in
        the retrieval-time merge when multiple KBs are selected.

        The merge function just runs max() on int keys (including chunk_tokens)
        which is fine for BOOKKEEPING but callers shouldn't rely on the
        merged value for chunking — each KB's chunks already exist at
        whatever size was used at ingest time.
        """
        c1 = {"chunk_tokens": 300}
        c2 = {"chunk_tokens": 600}
        merged = kb_config.merge_configs([c1, c2])
        # Current behaviour: max wins (600). This is documented but not
        # used at retrieval time — each KB's Qdrant chunks are already sized.
        assert merged.get("chunk_tokens") == 600
