"""Tests for §5.2 — RAG_BUDGET_INCLUDES_PROMPT pre-deducts non-chunk prompt parts.

Default OFF — ``budget_chunks`` ignores ``reserved_tokens`` and behaviour is
byte-identical. When ON, the caller passes ``reserved_tokens`` (system prompt
+ catalog preamble + datetime preamble + spotlight wrap overhead) and the
function only keeps chunks whose cumulative token count fits in
``max_tokens - reserved_tokens``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from ext.services.budget import budget_chunks


@dataclass
class _Hit:
    score: float
    payload: dict


def _hit_with_text(text: str) -> _Hit:
    return _Hit(score=1.0, payload={"text": text})


def test_reserved_tokens_param_default_off_matches_legacy(monkeypatch):
    """Without the env flag, ``reserved_tokens`` is silently ignored.
    Existing call sites (no kwarg) and the legacy contract are preserved.
    """
    monkeypatch.delenv("RAG_BUDGET_INCLUDES_PROMPT", raising=False)
    hits = [_hit_with_text("a " * 100) for _ in range(10)]

    legacy = budget_chunks(hits, max_tokens=200)
    new_with_kwarg = budget_chunks(hits, max_tokens=200, reserved_tokens=150)

    # Same kept count — kwarg silently ignored when flag is off.
    assert len(legacy) == len(new_with_kwarg)


def test_reserved_tokens_zero_matches_legacy(monkeypatch):
    """With flag ON but reserved_tokens=0, kept set equals the legacy result."""
    monkeypatch.setenv("RAG_BUDGET_INCLUDES_PROMPT", "1")
    hits = [_hit_with_text("alpha " * 50) for _ in range(8)]

    legacy = budget_chunks(hits, max_tokens=400)
    with_zero = budget_chunks(hits, max_tokens=400, reserved_tokens=0)

    assert len(with_zero) == len(legacy)


def test_reserved_tokens_subtracts_from_budget(monkeypatch):
    """Flag ON + reserved_tokens=N → effective budget is (max - N)."""
    monkeypatch.setenv("RAG_BUDGET_INCLUDES_PROMPT", "1")

    # Build distinct-text hits so each chunk has its own (cached) token count.
    hits = [_hit_with_text(f"chunk-{i} " + "x " * 20) for i in range(10)]

    # Calibrate: with 1000-token budget all hits fit easily.
    full = budget_chunks(hits, max_tokens=1000)
    assert len(full) == 10

    # Reserve nearly all the budget — only the smallest prefix should fit.
    tight = budget_chunks(hits, max_tokens=1000, reserved_tokens=950)
    assert 0 <= len(tight) < len(full)

    # Sum of kept-chunk token counts must be <= (max_tokens - reserved).
    from ext.services.budget import _count_tokens
    total = sum(_count_tokens(str(h.payload.get("text", ""))) for h in tight)
    assert total <= (1000 - 950)


def test_reserved_tokens_negative_capacity_keeps_zero(monkeypatch):
    """Reserved tokens >= max_tokens → effective budget is 0 (or negative);
    no chunks fit so the kept list is empty."""
    monkeypatch.setenv("RAG_BUDGET_INCLUDES_PROMPT", "1")

    hits = [_hit_with_text("payload " * 20) for _ in range(5)]
    out = budget_chunks(hits, max_tokens=100, reserved_tokens=200)
    assert out == []


def test_flag_off_ignores_reserved_tokens(monkeypatch):
    """Even when reserved_tokens is huge, flag OFF keeps the original budget."""
    monkeypatch.delenv("RAG_BUDGET_INCLUDES_PROMPT", raising=False)

    hits = [_hit_with_text("xy " * 5) for _ in range(20)]

    full = budget_chunks(hits, max_tokens=1000)
    with_huge_reserve = budget_chunks(hits, max_tokens=1000, reserved_tokens=999999)
    # Flag off → reserved_tokens is ignored.
    assert len(with_huge_reserve) == len(full)
