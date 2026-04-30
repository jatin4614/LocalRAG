"""Unit tests for H5 — validate_config bounds on retrieval-side knobs.

``validate_config`` silently drops values that fall outside sensible
ranges so a corrupt JSONB row (or a careless admin PATCH) can never
push the retrieval pipeline into an invalid state. The KB then
inherits process defaults instead of running with a bad value.

Bounds enforced:
    mmr_lambda            [0.0, 1.0]
    rerank_top_k          [1, 1000]
    context_expand_window [0, 100]
    hyde_n                [1, 10]
"""
from __future__ import annotations

import pytest

from ext.services import kb_config


class TestValidateConfigClampsMmrLambda:
    def test_mmr_lambda_zero_is_kept(self) -> None:
        assert kb_config.validate_config({"mmr_lambda": 0.0}) == {"mmr_lambda": 0.0}

    def test_mmr_lambda_one_is_kept(self) -> None:
        assert kb_config.validate_config({"mmr_lambda": 1.0}) == {"mmr_lambda": 1.0}

    def test_mmr_lambda_in_range_is_kept(self) -> None:
        assert kb_config.validate_config({"mmr_lambda": 0.7}) == {"mmr_lambda": 0.7}

    def test_mmr_lambda_above_one_is_dropped(self) -> None:
        assert kb_config.validate_config({"mmr_lambda": 1.5}) == {}

    def test_mmr_lambda_negative_is_dropped(self) -> None:
        assert kb_config.validate_config({"mmr_lambda": -0.1}) == {}

    def test_mmr_lambda_zero_kept_negative_dropped_one_kept_overone_dropped(self) -> None:
        # The exact "1.5 dropped, -0.1 dropped, 0.7 kept" cases.
        assert kb_config.validate_config({"mmr_lambda": 1.5}) == {}
        assert kb_config.validate_config({"mmr_lambda": -0.1}) == {}
        assert kb_config.validate_config({"mmr_lambda": 0.7}) == {"mmr_lambda": 0.7}


class TestValidateConfigClampsRerankTopK:
    def test_rerank_top_k_one_is_kept(self) -> None:
        assert kb_config.validate_config({"rerank_top_k": 1}) == {"rerank_top_k": 1}

    def test_rerank_top_k_thousand_is_kept(self) -> None:
        assert kb_config.validate_config({"rerank_top_k": 1000}) == {"rerank_top_k": 1000}

    def test_rerank_top_k_zero_is_dropped(self) -> None:
        assert kb_config.validate_config({"rerank_top_k": 0}) == {}

    def test_rerank_top_k_huge_is_dropped(self) -> None:
        assert kb_config.validate_config({"rerank_top_k": 100000}) == {}

    def test_rerank_top_k_zero_dropped_huge_dropped_fifty_kept(self) -> None:
        assert kb_config.validate_config({"rerank_top_k": 0}) == {}
        assert kb_config.validate_config({"rerank_top_k": 100000}) == {}
        assert kb_config.validate_config({"rerank_top_k": 50}) == {"rerank_top_k": 50}


class TestValidateConfigClampsContextExpandWindow:
    def test_context_expand_window_zero_is_kept(self) -> None:
        assert kb_config.validate_config({"context_expand_window": 0}) == {"context_expand_window": 0}

    def test_context_expand_window_in_range_is_kept(self) -> None:
        assert kb_config.validate_config({"context_expand_window": 5}) == {"context_expand_window": 5}

    def test_context_expand_window_at_ceiling_is_kept(self) -> None:
        assert kb_config.validate_config({"context_expand_window": 100}) == {"context_expand_window": 100}

    def test_context_expand_window_negative_is_dropped(self) -> None:
        assert kb_config.validate_config({"context_expand_window": -1}) == {}

    def test_context_expand_window_above_ceiling_is_dropped(self) -> None:
        assert kb_config.validate_config({"context_expand_window": 101}) == {}

    def test_context_expand_window_huge_is_dropped(self) -> None:
        assert kb_config.validate_config({"context_expand_window": 10000}) == {}


class TestValidateConfigClampsHydeN:
    def test_hyde_n_one_is_kept(self) -> None:
        assert kb_config.validate_config({"hyde_n": 1}) == {"hyde_n": 1}

    def test_hyde_n_ten_is_kept(self) -> None:
        assert kb_config.validate_config({"hyde_n": 10}) == {"hyde_n": 10}

    def test_hyde_n_in_range_is_kept(self) -> None:
        assert kb_config.validate_config({"hyde_n": 3}) == {"hyde_n": 3}

    def test_hyde_n_zero_is_dropped(self) -> None:
        assert kb_config.validate_config({"hyde_n": 0}) == {}

    def test_hyde_n_eleven_is_dropped(self) -> None:
        assert kb_config.validate_config({"hyde_n": 11}) == {}

    def test_hyde_n_huge_is_dropped(self) -> None:
        assert kb_config.validate_config({"hyde_n": 1000}) == {}


class TestBoundsCoexistWithOtherValidKeys:
    def test_out_of_range_drops_only_the_bad_key(self) -> None:
        out = kb_config.validate_config({
            "rerank": True,
            "rerank_top_k": 5000,   # out of range
            "mmr_lambda": 0.7,      # in range
        })
        assert out == {"rerank": True, "mmr_lambda": 0.7}
        assert "rerank_top_k" not in out


# ---------------------------------------------------------------------------
# Top-level test functions matching the names the Agent C spec asks for —
# 1.5 dropped, -0.1 dropped, 0.7 kept, etc. The TestValidateConfig*
# classes above provide the granular assertions; these are the exact
# named tests so a search-by-test-name lookup hits a single function.
# ---------------------------------------------------------------------------

def test_validate_config_clamps_mmr_lambda() -> None:
    """1.5 dropped, -0.1 dropped, 0.7 kept."""
    assert kb_config.validate_config({"mmr_lambda": 1.5}) == {}
    assert kb_config.validate_config({"mmr_lambda": -0.1}) == {}
    assert kb_config.validate_config({"mmr_lambda": 0.7}) == {"mmr_lambda": 0.7}


def test_validate_config_clamps_rerank_top_k() -> None:
    """0 dropped, 100000 dropped, 50 kept."""
    assert kb_config.validate_config({"rerank_top_k": 0}) == {}
    assert kb_config.validate_config({"rerank_top_k": 100000}) == {}
    assert kb_config.validate_config({"rerank_top_k": 50}) == {"rerank_top_k": 50}


def test_validate_config_clamps_context_expand_window() -> None:
    """-1 dropped, 200 dropped, 5 kept."""
    assert kb_config.validate_config({"context_expand_window": -1}) == {}
    assert kb_config.validate_config({"context_expand_window": 200}) == {}
    assert kb_config.validate_config({"context_expand_window": 5}) == {
        "context_expand_window": 5,
    }


def test_validate_config_clamps_hyde_n() -> None:
    """0 dropped, 11 dropped, 3 kept."""
    assert kb_config.validate_config({"hyde_n": 0}) == {}
    assert kb_config.validate_config({"hyde_n": 11}) == {}
    assert kb_config.validate_config({"hyde_n": 3}) == {"hyde_n": 3}
