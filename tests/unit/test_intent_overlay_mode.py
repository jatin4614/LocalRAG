"""Tests for the RAG_INTENT_OVERLAY_MODE flag (B3 design call).

Default mode (``intent``): intent overlay shadows env for the keys it owns
(``RAG_MMR``, ``RAG_CONTEXT_EXPAND``). Per-KB ``rag_config`` always wins.

Alternative mode (``env``): when an env var is set for a key the overlay
would otherwise stamp, drop that key from the overlay so the env value
flows through ``flags.get`` unshadowed. Per-KB still wins.

Memory note: A/B both modes against real production queries before locking
the default. See `~/.claude/projects/-home-vogic-LocalRAG/memory/intent_overlay_ab.md`.
"""
from __future__ import annotations

import pytest

from ext.services.chat_rag_bridge import resolve_intent_flags


# ---------------------------------------------------------------------------
# Default mode = "intent" (current Phase 2.2 behaviour)
# ---------------------------------------------------------------------------


def test_default_mode_is_intent_when_unset(monkeypatch):
    """Unset RAG_INTENT_OVERLAY_MODE → default 'intent' behaviour."""
    monkeypatch.delenv("RAG_INTENT_OVERLAY_MODE", raising=False)
    monkeypatch.setenv("RAG_MMR", "1")  # operator tries to force MMR on
    f = resolve_intent_flags(intent="specific", per_kb_overrides={})
    # Intent overlay stamps RAG_MMR=0 for specific, env value gets shadowed
    assert f["RAG_MMR"] == "0"
    assert f["RAG_CONTEXT_EXPAND"] == "1"


def test_explicit_intent_mode_shadows_env(monkeypatch):
    """RAG_INTENT_OVERLAY_MODE=intent → same as default."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "intent")
    monkeypatch.setenv("RAG_MMR", "1")
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "0")
    f = resolve_intent_flags(intent="global", per_kb_overrides={})
    # global → MMR=1, EXPAND=0; env-set RAG_CONTEXT_EXPAND=0 happens to match,
    # but RAG_MMR=1 happens to match too — still shadowed (deterministic policy)
    assert f["RAG_MMR"] == "1"
    assert f["RAG_CONTEXT_EXPAND"] == "0"


def test_intent_mode_with_env_disagreement_intent_wins(monkeypatch):
    """Intent default disagrees with env → intent wins."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "intent")
    monkeypatch.setenv("RAG_MMR", "1")  # operator wants MMR globally
    monkeypatch.setenv("RAG_CONTEXT_EXPAND", "1")  # operator wants expand globally
    f = resolve_intent_flags(intent="metadata", per_kb_overrides={})
    # metadata intent forces both off
    assert f["RAG_MMR"] == "0"
    assert f["RAG_CONTEXT_EXPAND"] == "0"


# ---------------------------------------------------------------------------
# Alternative mode = "env"
# ---------------------------------------------------------------------------


def test_env_mode_drops_overlay_when_env_set(monkeypatch):
    """RAG_INTENT_OVERLAY_MODE=env + env RAG_MMR=1 → overlay drops RAG_MMR."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_MMR", "1")
    f = resolve_intent_flags(intent="specific", per_kb_overrides={})
    # specific intent would normally stamp RAG_MMR=0; env mode drops it so
    # the env value (1) flows through downstream flags.get unshadowed.
    assert "RAG_MMR" not in f, (
        "in env mode, when env has RAG_MMR set, the overlay must drop "
        "RAG_MMR so flags.get falls through to env"
    )
    # RAG_CONTEXT_EXPAND env not set → intent default still applies
    assert f["RAG_CONTEXT_EXPAND"] == "1"


def test_env_mode_preserves_intent_when_env_unset(monkeypatch):
    """Env mode but no env vars set → intent defaults still apply."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.delenv("RAG_MMR", raising=False)
    monkeypatch.delenv("RAG_CONTEXT_EXPAND", raising=False)
    f = resolve_intent_flags(intent="global", per_kb_overrides={})
    # global → MMR=1, EXPAND=0
    assert f["RAG_MMR"] == "1"
    assert f["RAG_CONTEXT_EXPAND"] == "0"


def test_env_mode_per_kb_overrides_still_win(monkeypatch):
    """Per-KB rag_config wins over both intent AND env."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "env")
    monkeypatch.setenv("RAG_MMR", "1")  # operator wants MMR
    f = resolve_intent_flags(
        intent="global",
        per_kb_overrides={"RAG_MMR": "0"},  # KB admin says no
    )
    # KB admin wins — explicit per-collection statement.
    assert f["RAG_MMR"] == "0"


def test_env_mode_case_insensitive(monkeypatch):
    """Mode parsing is case-insensitive."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "ENV")
    monkeypatch.setenv("RAG_MMR", "1")
    f = resolve_intent_flags(intent="specific", per_kb_overrides={})
    assert "RAG_MMR" not in f


def test_unknown_mode_falls_back_to_intent(monkeypatch):
    """Typo / unknown mode → safe default (intent)."""
    monkeypatch.setenv("RAG_INTENT_OVERLAY_MODE", "wat")
    monkeypatch.setenv("RAG_MMR", "1")
    f = resolve_intent_flags(intent="specific", per_kb_overrides={})
    # Unknown mode treats as intent → shadows env
    assert f["RAG_MMR"] == "0"


# ---------------------------------------------------------------------------
# Cross-cutting: function purity
# ---------------------------------------------------------------------------


def test_resolve_intent_flags_does_not_mutate_inputs(monkeypatch):
    monkeypatch.delenv("RAG_INTENT_OVERLAY_MODE", raising=False)
    overrides = {"RAG_MMR": "1"}
    overrides_before = dict(overrides)
    resolve_intent_flags(intent="specific", per_kb_overrides=overrides)
    assert overrides == overrides_before, "per_kb_overrides was mutated"
