"""B8 — RAG_BUDGET_TOKENIZER_MODEL applies to every hf alias when set.

Before this fix the env override only flowed into the family aliases
(qwen / gemma / llama). Versioned aliases like ``gemma-4``,
``gemma-4-31b``, ``gemma-3``, ``gemma-3-12b``, ``qwen2.5`` ignored the
override even when the operator wanted to point at a local mirror — for
example, the air-gapped LocalRAG stack ships
``QuantTrio/gemma-4-31B-it-AWQ`` in ``volumes/models``, which uses the
SAME Gemma tokenizer as the Google upstream but is the only repo
actually present on disk.

These tests pin the new contract: with ``RAG_BUDGET_TOKENIZER_MODEL``
set, the override wins; with it unset, the pinned default still wins.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _reset_tokenizer_caches() -> None:
    for mod_name in ("ext.services.budget", "ext.services.chunker"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr in ("get_tokenizer", "_budget_tokenizer", "_encoder"):
            fn = getattr(mod, attr, None)
            cc = getattr(fn, "cache_clear", None)
            if callable(cc):
                cc()


def _reload_budget(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import ext.services.budget as budget
    importlib.reload(budget)
    _reset_tokenizer_caches()
    return budget


def _patch_auto_tokenizer(monkeypatch, captured: dict):
    pytest.importorskip("transformers")

    stub_tok = MagicMock()
    stub_tok.encode.return_value = [1, 2, 3]
    stub_tok.decode.return_value = "x"

    def _from_pretrained(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return stub_tok

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        _from_pretrained,
    )


# -- gemma-4: the alias the live chat model belongs to. ---------------------


def test_gemma_4_with_override_resolves_to_awq_id(monkeypatch):
    """gemma-4 + RAG_BUDGET_TOKENIZER_MODEL=QuantTrio/...-AWQ → AWQ id wins.

    The on-disk model in ``volumes/models`` is the AWQ build; the stock
    google/gemma-4-31B-it tokenizer file isn't downloaded. Without this
    fix the chunker would fail to load any tokenizer in air-gapped mode.
    """
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4",
        RAG_BUDGET_TOKENIZER_MODEL="QuantTrio/gemma-4-31B-it-AWQ",
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "QuantTrio/gemma-4-31B-it-AWQ"


def test_gemma_4_without_override_keeps_pinned_default(monkeypatch):
    """gemma-4 with no override → still loads google/gemma-4-31B-it.

    Operators who haven't opted into an override get deterministic
    behavior — pinned defaults remain authoritative.
    """
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "google/gemma-4-31B-it"


# -- Other versioned aliases also respect the override. --------------------


def test_gemma_3_override_applies(monkeypatch):
    """The fix is uniform: gemma-3 also respects the env override when set."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-3",
        RAG_BUDGET_TOKENIZER_MODEL="foo",
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "foo"


def test_qwen_2_5_override_applies(monkeypatch):
    """qwen2.5 — the other versioned alias — also respects the override."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="qwen2.5",
        RAG_BUDGET_TOKENIZER_MODEL="local/qwen2.5-mirror",
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "local/qwen2.5-mirror"


def test_gemma_4_31b_override_applies(monkeypatch):
    """gemma-4-31b alias mirrors gemma-4 — override applies here too."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4-31b",
        RAG_BUDGET_TOKENIZER_MODEL="QuantTrio/gemma-4-31B-it-AWQ",
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "QuantTrio/gemma-4-31B-it-AWQ"


# -- Family aliases keep working (regression guard). -----------------------


def test_family_alias_gemma_still_respects_override(monkeypatch):
    """Pre-existing behavior: the 'gemma' family alias respected the env
    var. This must keep working — B8 must NOT regress family aliases."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma",
        RAG_BUDGET_TOKENIZER_MODEL="local/gemma-mirror",
    )
    budget._count_tokens("hello")
    assert captured["args"][0] == "local/gemma-mirror"


# -- Override is read at lookup time, not at module import. ----------------


def test_override_does_not_affect_cl100k(monkeypatch):
    """RAG_BUDGET_TOKENIZER=cl100k ignores the model override (tiktoken
    has no notion of a HF repo id).
    """
    pytest.importorskip("tiktoken")
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="cl100k",
        RAG_BUDGET_TOKENIZER_MODEL="QuantTrio/gemma-4-31B-it-AWQ",
    )
    # cl100k tokenizes "hello world" into 2 tokens; if the override had
    # wrongly leaked, we'd be loading a HF tokenizer here and would not
    # get exactly 2.
    assert budget._count_tokens("hello world") == 2


def test_empty_override_treated_as_unset(monkeypatch):
    """Empty string env var must NOT clobber the pinned default — common
    docker-compose footgun where ``RAG_BUDGET_TOKENIZER_MODEL=`` is
    written but no value is supplied."""
    captured: dict = {}
    _patch_auto_tokenizer(monkeypatch, captured)
    monkeypatch.setenv("RAG_BUDGET_TOKENIZER_MODEL", "")
    budget = _reload_budget(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4",
    )
    budget._count_tokens("hello")
    # Empty string falsy → fall back to pinned default, NOT load id="".
    assert captured["args"][0] == "google/gemma-4-31B-it"
