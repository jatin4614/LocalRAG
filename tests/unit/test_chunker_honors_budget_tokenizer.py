"""B2 — chunker shares the budget enforcer's tokenizer.

Before this fix the chunker hardcoded ``tiktoken.cl100k_base`` while the
prompt-budget pass honored ``RAG_BUDGET_TOKENIZER``. With a chat model
like Gemma-4 selected by the operator, chunks were sized in cl100k tokens
but counted in Gemma tokens at retrieval — the ~10-15% drift evicted
relevant chunks. The chunker now routes through
``ext.services.budget.get_tokenizer`` so a single token vocabulary
governs both passes.

These tests use ``transformers.AutoTokenizer`` patched to a stub so the
suite stays network-free; the real Gemma tokenizer file isn't needed.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


def _reset_tokenizer_caches() -> None:
    """Clear the lru_caches that pin the resolved tokenizer per process."""
    for mod_name in ("ext.services.budget", "ext.services.chunker"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr in ("get_tokenizer", "_budget_tokenizer", "_encoder"):
            fn = getattr(mod, attr, None)
            cc = getattr(fn, "cache_clear", None)
            if callable(cc):
                cc()


def _reload_with_env(monkeypatch, **env):
    """Reload budget + chunker so the registry sees the requested env."""
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    import ext.services.budget as budget
    import ext.services.chunker as chunker
    importlib.reload(budget)
    importlib.reload(chunker)
    _reset_tokenizer_caches()
    return budget, chunker


def _patch_hf_tokenizer(monkeypatch, *, tokens_per_call: int = 5):
    """Patch transformers.AutoTokenizer with a stub.

    The stub mimics a Gemma-style tokenizer: ``encode()`` returns a list
    of distinct ints sized roughly proportional to the input. We choose
    a deliberately different ratio than cl100k so token-count comparisons
    diverge.
    """
    pytest.importorskip("transformers")

    stub = MagicMock()

    def _encode(text: str, add_special_tokens: bool = True):
        # Aggressive split: one token per character. cl100k typically
        # produces far fewer tokens for the same text → counts diverge.
        return list(range(len(text)))

    def _decode(ids, skip_special_tokens: bool = False):
        # Round-trip: emit a placeholder string proportional to id count
        # so the chunker's hard-split path can reassemble something.
        return "x" * len(list(ids))

    stub.encode.side_effect = _encode
    stub.decode.side_effect = _decode
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: stub,
    )
    return stub


def test_default_chunker_uses_cl100k(monkeypatch):
    """With RAG_BUDGET_TOKENIZER unset, the chunker keeps using cl100k.

    Smoke check: the existing chunker tests already pin this behavior;
    we re-verify here from the chunker→budget seam to make sure the
    refactor didn't silently swap defaults.
    """
    budget, chunker = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER=None,
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    handle = chunker._encoder()
    # cl100k handle round-trips "hello world" identically
    ids = handle.encode("hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert handle.decode(ids) == "hello world"
    # Chunker still produces output
    chunks = chunker.chunk_text("hello world", chunk_tokens=800, overlap_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_chunker_handle_is_same_object_as_budget_handle(monkeypatch):
    """Single tokenizer instance — chunk boundaries and budget counts share it."""
    budget, chunker = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER=None,
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    assert chunker._encoder() is budget.get_tokenizer()


def test_chunker_uses_gemma_when_alias_set(monkeypatch):
    """RAG_BUDGET_TOKENIZER=gemma-4 → chunker tokenizes via the Gemma stub.

    We patch transformers.AutoTokenizer so no model file is downloaded.
    The stub returns one token per character; cl100k would tokenize
    "hello world" into 2 tokens, the stub returns 11. The divergence
    proves the chunker is no longer hardcoded to cl100k.
    """
    _patch_hf_tokenizer(monkeypatch)
    _budget, chunker = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    handle = chunker._encoder()
    n_gemma = len(handle.encode("hello world"))
    # Stub returns 1 token per char → 11 for "hello world"
    assert n_gemma == 11

    # cl100k baseline for the same text — must differ.
    _budget_cl, chunker_cl = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="cl100k",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    n_cl = len(chunker_cl._encoder().encode("hello world"))
    assert n_gemma != n_cl, (
        f"chunker still using cl100k under gemma-4 alias "
        f"(gemma={n_gemma}, cl100k={n_cl})"
    )


def test_chunker_boundaries_shift_when_tokenizer_changes(monkeypatch):
    """End-to-end: chunk boundaries differ between cl100k and the Gemma stub.

    For the same text + chunk_tokens budget, the two tokenizers pack a
    different number of sentences per chunk because their token counts
    diverge. The chunker_text output reflects that.
    """
    text = (
        "Sentence one is here. Sentence two follows. Sentence three trails. "
        "Sentence four ends. Sentence five caps. Sentence six wraps. "
        "Sentence seven extends. Sentence eight closes. Sentence nine sits. "
        "Sentence ten finishes."
    )

    # cl100k path
    _b1, chunker_cl = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="cl100k",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    cl_chunks = chunker_cl.chunk_text(text, chunk_tokens=20, overlap_tokens=4)
    cl_token_counts = [c.token_count for c in cl_chunks]

    # Gemma stub path: 1 token per char → very different boundaries
    _patch_hf_tokenizer(monkeypatch)
    _b2, chunker_gm = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER="gemma-4",
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    gm_chunks = chunker_gm.chunk_text(text, chunk_tokens=20, overlap_tokens=4)
    gm_token_counts = [c.token_count for c in gm_chunks]

    # Boundaries differ in chunk count, token packing, or both.
    assert (cl_chunks != gm_chunks) or (cl_token_counts != gm_token_counts), (
        "chunker output identical across tokenizers — boundaries didn't shift "
        f"(cl={cl_token_counts}, gemma_stub={gm_token_counts})"
    )


def test_encoder_lru_cache_is_singleton(monkeypatch):
    """Performance guard: ``_encoder()`` must return the same handle every call.

    The chunker's hot path calls ``_encoder()`` once per chunk_text call;
    if the lru_cache regressed, every chunk_text would re-resolve (and
    on the HF path, re-load) the tokenizer.
    """
    _budget, chunker = _reload_with_env(
        monkeypatch,
        RAG_BUDGET_TOKENIZER=None,
        RAG_BUDGET_TOKENIZER_MODEL=None,
    )
    a = chunker._encoder()
    b = chunker._encoder()
    c = chunker._encoder()
    assert a is b is c
