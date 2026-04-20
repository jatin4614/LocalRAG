"""Registry-driven tokenizer resolution.

The tests exercise the alias lookup in ``ext.services.budget`` without
downloading real tokenizer vocabs — ``transformers.AutoTokenizer`` is
patched to return a stub. This keeps the suite deterministic and
network-free while still verifying that unknown aliases warn + fall
back, and that RAG_BUDGET_TOKENIZER_MODEL overrides are honored.
"""
from __future__ import annotations

import importlib
import logging
import sys
from unittest.mock import MagicMock

import pytest


def _reload_budget(monkeypatch, **env):
    """Reload ext.services.budget with the requested env set (or cleared)."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import ext.services.budget as budget  # noqa: WPS433 (late import is the point)
    importlib.reload(budget)
    # Defensive: lru_cache is module-scoped, reload replaces it, but if the
    # runtime kept any reference, clear it.
    budget._budget_tokenizer.cache_clear()
    return budget


def _patch_auto_tokenizer(monkeypatch, *, token_count: int = 7, captured: dict | None = None):
    """Replace transformers.AutoTokenizer.from_pretrained with a stub.

    The stub returns an object whose .encode(..., add_special_tokens=False)
    returns a list of token_count ints. If ``captured`` is provided, the
    from_pretrained call is recorded there.
    """
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed")

    stub_tok = MagicMock()
    stub_tok.encode.return_value = list(range(token_count))
    calls: list[tuple[tuple, dict]] = []

    def _from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        if captured is not None:
            captured["args"] = args
            captured["kwargs"] = kwargs
        return stub_tok

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        _from_pretrained,
    )
    return calls


def test_default_unset_falls_back_to_cl100k(monkeypatch):
    budget = _reload_budget(monkeypatch, RAG_BUDGET_TOKENIZER=None)
    # cl100k encodes "hello world" into 2 tokens
    assert budget._count_tokens("hello world") == 2


def test_alias_gemma_resolves_to_hf_tokenizer(monkeypatch):
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, token_count=11, captured=captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    n = budget._count_tokens("arbitrary text here")
    assert n == 11
    # Verify the stub was asked to load the Gemma-family repo (default).
    called_with = captured["args"][0]
    assert "gemma" in called_with.lower()


def test_alias_gemma_3_pin_ignores_env_override(monkeypatch):
    """'gemma-3' is a pinned alias; RAG_BUDGET_TOKENIZER_MODEL should not affect it."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured=captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-3",
        RAG_BUDGET_TOKENIZER_MODEL="some/other-repo",
    )
    budget._count_tokens("x")
    called_with = captured["args"][0]
    assert called_with == "google/gemma-3-27b-it"


def test_alias_llama_resolves_to_hf_tokenizer(monkeypatch):
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, token_count=5, captured=captured)
    budget = _reload_budget(monkeypatch, RAG_BUDGET_TOKENIZER="llama")
    n = budget._count_tokens("anything")
    assert n == 5
    assert "llama" in captured["args"][0].lower()


def test_unknown_alias_warns_and_falls_back(monkeypatch, caplog):
    # No transformers patch needed — we should never reach the HF branch.
    with caplog.at_level(logging.WARNING, logger="rag.budget"):
        budget = _reload_budget(monkeypatch, RAG_BUDGET_TOKENIZER="banana")
        n = budget._count_tokens("hello world")
    assert n >= 1
    joined = " ".join(rec.message for rec in caplog.records)
    assert "banana" in joined.lower() or "not registered" in joined.lower()


def test_env_override_applies_to_gemma_family_alias(monkeypatch):
    """'gemma' is a family alias; RAG_BUDGET_TOKENIZER_MODEL overrides its id."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured=captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma",
        RAG_BUDGET_TOKENIZER_MODEL="google/gemma-3-12b-it",
    )
    budget._count_tokens("x")
    called_with = captured["args"][0]
    assert called_with == "google/gemma-3-12b-it"


def test_hf_load_failure_falls_back_to_cl100k(monkeypatch):
    """Simulate a gated-repo 401 / network error — we should warn + fall back."""
    pytest.importorskip("transformers")

    def _raise(*args, **kwargs):
        raise OSError("401 Unauthorized: gated repo, need HF_TOKEN")

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _raise)
    budget = _reload_budget(monkeypatch, RAG_BUDGET_TOKENIZER="gemma")
    # Still counts tokens via cl100k fallback — doesn't raise.
    assert budget._count_tokens("hello world") == 2


def test_cl100k_explicit_still_works(monkeypatch):
    budget = _reload_budget(monkeypatch, RAG_BUDGET_TOKENIZER="cl100k")
    assert budget._count_tokens("hello world") == 2


def test_registry_contains_expected_aliases():
    import ext.services.budget as budget
    importlib.reload(budget)
    assert "cl100k" in budget._TOKENIZER_REGISTRY
    assert "qwen" in budget._TOKENIZER_REGISTRY
    assert "gemma" in budget._TOKENIZER_REGISTRY
    assert "llama" in budget._TOKENIZER_REGISTRY
