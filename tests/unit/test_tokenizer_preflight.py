"""Phase 1.1 — tokenizer preflight at startup.

The preflight enforces a contract: when the operator explicitly sets
``RAG_BUDGET_TOKENIZER`` to a non-cl100k HF alias, failure to load the
backing tokenizer at startup CRASHES the process. Silent fallback to
cl100k drifts token counts ~10-15% which can evict relevant chunks
from the context window without anyone noticing.

Unknown aliases (operator typos) and the unset/default cl100k case are
non-fatal — the existing fallback path handles those safely.
"""
import os
import pytest
from unittest.mock import patch

from ext.services.budget import preflight_tokenizer, TokenizerPreflightError


def test_preflight_passes_for_cl100k(monkeypatch):
    monkeypatch.setenv("RAG_BUDGET_TOKENIZER", "cl100k")
    preflight_tokenizer()  # must not raise


def test_preflight_crashes_when_explicit_hf_tokenizer_falls_back(monkeypatch):
    monkeypatch.setenv("RAG_BUDGET_TOKENIZER", "gemma-4")
    # Force AutoTokenizer.from_pretrained to raise
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("no cache")):
        with pytest.raises(TokenizerPreflightError) as excinfo:
            preflight_tokenizer()
        assert "gemma-4" in str(excinfo.value)


def test_preflight_allows_unset_tokenizer(monkeypatch):
    monkeypatch.delenv("RAG_BUDGET_TOKENIZER", raising=False)
    preflight_tokenizer()  # default → cl100k, must not raise


def test_preflight_warns_on_unknown_alias_but_does_not_crash(monkeypatch, caplog):
    import logging
    caplog.set_level(logging.WARNING, logger="ext.services.budget")
    monkeypatch.setenv("RAG_BUDGET_TOKENIZER", "unknown-alias")
    preflight_tokenizer()  # unknown alias → cl100k fallback is acceptable (not explicit HF)
    assert any("unknown-alias" in rec.message.lower() or "cl100k" in rec.message.lower()
               for rec in caplog.records)
