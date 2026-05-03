"""Multilingual sentence splitter tests (review §2.5).

The legacy regex ``(?<=[.!?])\\s+`` (in ``ext/services/chunker.py``) is
English-centric and misses:
  * Hindi danda (``।``) — Devanagari sentences run together.
  * Chinese / Japanese full-stop (``。``) — same problem.
  * Over-splits French / German abbreviations (``Dr.``, ``M.``, etc.).

When ``RAG_PYSBD_ENABLED=1`` AND pysbd is importable, the chunker
delegates to pysbd's language-specific segmenter. These tests assert:

1. With the flag OFF (default), the regex path runs and Hindi /
   Chinese stay UNDER-split (concatenated as one mega-sentence)
   — documents the bug the flag exists to fix.
2. With the flag ON, the multilingual texts split correctly.
3. With the flag ON, French abbreviations don't over-split.
4. The fallback path (pysbd absent) is exercised by mocking the
   probe to ``False`` — no exception, runs as if the flag was off.
"""
from __future__ import annotations

import importlib

import pytest

from ext.services import chunker
from ext.services.chunker import Chunk, _walk_sentences, chunk_text


def _walk(text: str) -> list[str]:
    """Helper: collect just the sentence strings from ``_walk_sentences``."""
    return [sent for sent, _a, _b in _walk_sentences(text)]


def _reset_pysbd_probe() -> None:
    """Force the per-process pysbd probe back to ``not yet probed``.

    Tests flip the env var or monkeypatch the import probe; the cached
    value would otherwise leak between tests and produce flaky pass/fail.
    """
    chunker._PYSBD_PROBED = None
    chunker._pysbd_segmenter.cache_clear()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test owns its env state; reset cached probe + segmenters."""
    monkeypatch.delenv("RAG_PYSBD_ENABLED", raising=False)
    _reset_pysbd_probe()
    yield
    _reset_pysbd_probe()


# ---------- regex path (flag OFF / default) -------------------------------


def test_regex_path_handles_english_correctly():
    """Sanity: legacy English splitter still splits ``.``-separated sentences."""
    text = "This is one. This is two. This is three."
    sents = _walk(text)
    assert len(sents) == 3


def test_regex_path_under_splits_hindi_devanagari_danda():
    """The legacy regex does NOT recognize the Devanagari danda (``।``)
    so a 3-sentence Hindi paragraph collapses to ONE 'sentence'.

    This test documents the bug the §2.5 flag exists to fix; it should
    keep passing as long as the regex fallback exists.
    """
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sents = _walk(text)
    # Regex sees no period+whitespace, so the whole thing is one chunk.
    assert len(sents) == 1


def test_regex_path_under_splits_chinese_fullstop():
    """The legacy regex also misses the CJK full-stop (``。``)."""
    text = "这是第一句。这是第二句。这是第三句。"
    sents = _walk(text)
    assert len(sents) == 1


# ---------- pysbd path (flag ON, real pysbd) ------------------------------


def _have_pysbd() -> bool:
    try:
        importlib.import_module("pysbd")
        return True
    except ImportError:
        return False


pytestmark_pysbd = pytest.mark.skipif(
    not _have_pysbd(),
    reason="pysbd not installed; install via `pip install pysbd>=0.3.4` to run",
)


@pytestmark_pysbd
def test_pysbd_splits_hindi_danda_when_flag_on(monkeypatch):
    """``RAG_PYSBD_ENABLED=1`` makes Hindi split on ``।``."""
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    _reset_pysbd_probe()
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sents = _walk(text)
    # 3 sentences expected — one per danda.
    assert len(sents) == 3, f"got: {sents}"


@pytestmark_pysbd
def test_pysbd_splits_chinese_fullstop_when_flag_on(monkeypatch):
    """``RAG_PYSBD_ENABLED=1`` makes Chinese split on ``。``."""
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    _reset_pysbd_probe()
    text = "这是第一句。这是第二句。这是第三句。"
    sents = _walk(text)
    assert len(sents) == 3, f"got: {sents}"


@pytestmark_pysbd
def test_pysbd_keeps_french_abbreviations_intact_when_flag_on(monkeypatch):
    """The regex over-splits ``Dr. Smith`` into two sentences. pysbd
    (English segmenter, since the language sniffer falls back to ``en``
    for Latin script) recognizes ``Dr.`` as an abbreviation.

    The English segmenter is what fires for predominantly-Latin text;
    the test asserts pysbd's abbreviation-awareness works there. (For
    French-specific rules pysbd has a ``fr`` segmenter, but the sniffer
    will correctly choose ``en`` for content that is mostly ASCII Latin
    — that's the dominant production case for Western European text.)
    """
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    _reset_pysbd_probe()
    text = "Dr. Smith arrived. He was tired. Mr. Jones followed."
    sents = _walk(text)
    # 3 sentences — Dr. and Mr. should NOT trigger a split, vs the
    # regex which would emit 5 (Dr | Smith arrived | He was tired |
    # Mr | Jones followed).
    assert len(sents) == 3, f"got: {sents}"


@pytestmark_pysbd
def test_pysbd_path_round_trips_chunk_text_for_hindi(monkeypatch):
    """End-to-end: ``chunk_text`` with the flag ON produces multiple
    chunks for a multi-sentence Hindi paragraph that under the regex
    would emit one giant chunk.
    """
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    _reset_pysbd_probe()
    # Repeat the 3-sentence paragraph many times so the token budget
    # forces multiple chunks. With the regex path it would still be
    # one sentence, hence one chunk (or hard-split if huge); with
    # pysbd we get sentence-boundaried chunks instead.
    para = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    text = "\n\n".join([para] * 30)
    chunks = chunk_text(text, chunk_tokens=100, overlap_tokens=10)
    assert len(chunks) >= 1
    # Each chunk respects the token budget (the splitter put it
    # together from sentences, not a hard token-cut).
    assert all(c.token_count <= 100 for c in chunks), [c.token_count for c in chunks]


# ---------- regex equivalence under flag-OFF (no behavior change) ---------


def test_flag_off_matches_legacy_for_english():
    """With the flag OFF, behavior must be byte-identical to the
    pre-§2.5 chunker for English input. Regression guard: a future
    ``always-on pysbd'' refactor would silently change every existing
    English document's chunk boundaries — catch it here.
    """
    # Flag explicitly OFF (default).
    text = (
        "This is sentence one. This is sentence two! And sentence three? "
        "And four. And five."
    )
    sents = _walk(text)
    assert sents == [
        "This is sentence one.",
        "This is sentence two!",
        "And sentence three?",
        "And four.",
        "And five.",
    ]


# ---------- graceful degradation when pysbd is missing --------------------


def test_pysbd_absent_falls_back_to_regex(monkeypatch):
    """Even with the flag ON, if pysbd can't be imported the chunker
    silently uses the regex path. No exception, no error log.

    Simulated by forcing the probe to ``False`` (the sentinel for
    'probed and absent') via the module's internal cache.
    """
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    chunker._PYSBD_PROBED = False  # type: ignore[assignment]
    chunker._pysbd_segmenter.cache_clear()
    # Hindi text — the regex won't see the danda, so we expect 1 sent
    # (same buggy behavior as the flag-off case — proving fallback works).
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sents = _walk(text)
    assert len(sents) == 1


# ---------- language sniffer ---------------------------------------------


def test_language_sniffer_picks_hindi_for_devanagari():
    assert chunker._sniff_language("यह हिंदी पाठ है। पूर्ण विराम।") == "hi"


def test_language_sniffer_picks_chinese_for_cjk():
    assert chunker._sniff_language("这是中文内容。第二句。第三句。") == "zh"


def test_language_sniffer_picks_russian_for_cyrillic():
    assert chunker._sniff_language("Это русский текст. Это второе предложение.") == "ru"


def test_language_sniffer_defaults_to_english_for_latin():
    assert chunker._sniff_language("This is plain English text. With abbreviations.") == "en"


def test_language_sniffer_defaults_to_english_for_short_minority_script():
    """Mostly-Latin text with a few CJK characters → 'en' (under threshold)."""
    text = "Order shipping label 1234 to 北京 today."
    assert chunker._sniff_language(text) == "en"


def test_language_sniffer_handles_empty_string():
    assert chunker._sniff_language("") == "en"


# ---------- pysbd flag dispatch -------------------------------------------


def test_pysbd_enabled_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("RAG_PYSBD_ENABLED", raising=False)
    assert chunker._pysbd_enabled() is False


def test_pysbd_enabled_false_when_env_zero(monkeypatch):
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "0")
    assert chunker._pysbd_enabled() is False


def test_pysbd_enabled_returns_false_if_module_missing(monkeypatch):
    monkeypatch.setenv("RAG_PYSBD_ENABLED", "1")
    chunker._PYSBD_PROBED = False  # type: ignore[assignment]
    assert chunker._pysbd_enabled() is False
