import importlib
import pytest


def _reload_budget(monkeypatch, tokenizer: str):
    monkeypatch.setenv("RAG_BUDGET_TOKENIZER", tokenizer)
    import ext.services.budget as budget
    importlib.reload(budget)
    # Clear lru_cache since it's bound to module identity
    budget._budget_tokenizer.cache_clear()
    return budget


def test_default_is_cl100k(monkeypatch):
    monkeypatch.delenv("RAG_BUDGET_TOKENIZER", raising=False)
    budget = _reload_budget(monkeypatch, "cl100k")
    # cl100k encodes "hello world" into 2 tokens (well-known)
    assert budget._count_tokens("hello world") == 2


def test_qwen_tokenizer_loads(monkeypatch):
    transformers = pytest.importorskip("transformers")
    budget = _reload_budget(monkeypatch, "qwen")
    # Smoke: just check it returns a plausible positive int
    n = budget._count_tokens("hello world")
    assert n >= 1 and n < 20

def test_qwen_counts_differ_from_cl100k(monkeypatch):
    transformers = pytest.importorskip("transformers")
    budget_cl = _reload_budget(monkeypatch, "cl100k")
    n_cl = budget_cl._count_tokens("繁體中文 and العربية mixed.")
    budget_qw = _reload_budget(monkeypatch, "qwen")
    n_qw = budget_qw._count_tokens("繁體中文 and العربية mixed.")
    # For CJK + Arabic, tokenizers diverge significantly
    assert n_cl != n_qw, f"expected divergence: cl={n_cl} qwen={n_qw}"


def test_invalid_tokenizer_falls_back_gracefully(monkeypatch):
    # Unknown value should degrade to cl100k rather than crash
    budget = _reload_budget(monkeypatch, "banana")
    n = budget._count_tokens("hello")
    assert n >= 1
