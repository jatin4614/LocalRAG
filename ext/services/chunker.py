"""Structure- and tokenizer-aware chunker.

Offsets are tracked during a single forward regex walk, so chunking is
O(N) in the input length. Previously the code computed char offsets via
``enc.decode(ids[:k])`` inside the loop, which was O(N^2) and could wedge
the service on 20 MB uploads.

Downstream modules (``ingest.py``, ``budget.py``) depend on:
- ``chunk_text`` signature: (text, *, chunk_tokens, overlap_tokens) -> List[Chunk]
- ``Chunk.text`` and ``Chunk.index`` fields
- ``_encoder`` module-level helper (kept for back-compat; now delegates
  to :func:`ext.services.budget.get_tokenizer` so chunk boundaries and
  the prompt-budget pass share one token vocabulary).

The active tokenizer is whatever ``RAG_BUDGET_TOKENIZER`` selects in
``budget.py`` (default: ``cl100k``). When the operator sets it to a
chat-model alias (e.g. ``gemma-4``), chunk sizing tracks the real prompt
token count instead of being off by ~10-15% — which previously evicted
relevant chunks from the budget pass.

Multilingual sentence splitting (review §2.5):
The default sentence splitter is the English-centric regex ``_SENT``,
which only sees ``.``, ``!``, ``?`` followed by whitespace. That misses
the Devanagari danda (``।``), the Chinese / Japanese full-stop (``。``),
the Arabic full-stop, etc. — and over-splits French / German "Dr.",
"M.", "z.B." abbreviations.

When the operator sets ``RAG_PYSBD_ENABLED=1`` (default OFF), the walker
delegates to :mod:`pysbd` (Python Sentence Boundary Disambiguation),
which has language-specific rules for the 23 languages it supports.
The language is sniffed cheaply from the first ~512 chars by counting
Unicode-block hits (Devanagari → ``hi``, CJK → ``zh``, Cyrillic →
``ru``, Arabic → ``ar``, Hangul → ``ja`` heuristic, otherwise ``en``).

The flag ships default-OFF because pysbd's segmentation differs from
the regex by a few-percent of edge cases — a good audit trail to keep
even when the team eventually flips it on. If pysbd is missing from
the environment, the import is wrapped in try/except so the chunker
silently falls back to the regex (graceful degradation; no hard
runtime dependency).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional


# Paragraph break: two (or more) newlines, possibly with whitespace between.
_PARA = re.compile(r"\n\s*\n")
# Sentence break: whitespace following a terminator .!?  (English-centric; good enough for P0.)
# Wave 3-f review §2.5: superseded by pysbd when ``RAG_PYSBD_ENABLED=1`` AND
# the import succeeds. See ``_split_sentences`` for the dispatch.
_SENT = re.compile(r"(?<=[.!?])\s+")


# --- pysbd integration (§2.5) ----------------------------------------------
#
# We probe the import lazily so the regex path doesn't pay any cost if the
# operator hasn't installed pysbd. The import is also wrapped in try/except
# so a missing wheel in an air-gapped deploy degrades gracefully to the
# regex rather than blocking ingest.

# Sentinels — None means "not yet probed", False means "probed and absent".
_PYSBD_PROBED: Optional[object] = None  # the module if loaded, False if missing


def _pysbd_module():
    """Return the pysbd module, or False if unavailable. Cached per process."""
    global _PYSBD_PROBED
    if _PYSBD_PROBED is not None:
        return _PYSBD_PROBED
    try:
        import pysbd  # type: ignore[import-not-found]

        _PYSBD_PROBED = pysbd
    except Exception:
        # ImportError, AttributeError, anything from a borked wheel — treat
        # as absent. Logging here would spam every chunk_text call; the
        # operator sees "not splitting multilingually" as a behavior, not
        # an error, which is the correct contract for a soft dependency.
        _PYSBD_PROBED = False  # type: ignore[assignment]
    return _PYSBD_PROBED


def _pysbd_enabled() -> bool:
    """Truthy when the operator opted in via env AND pysbd is importable."""
    if os.environ.get("RAG_PYSBD_ENABLED", "0").strip().lower() not in {"1", "true", "yes"}:
        return False
    return bool(_pysbd_module())


# Unicode-block to pysbd-language-code mapping. The first matching block
# wins; the order is roughly population-weighted so the common cases (zh,
# hi) hit early. We deliberately do NOT call out to a real language
# detector (langdetect / lingua) — those are 10-50 MB deps and overkill
# for picking between 23 segmenters at chunk time. The block sniff is
# free, deterministic, and good enough; misclassified text just gets the
# English segmenter, which is also fine.
_LANG_RANGES: tuple[tuple[str, range], ...] = (
    ("hi", range(0x0900, 0x097F + 1)),  # Devanagari (Hindi, Marathi)
    ("zh", range(0x4E00, 0x9FFF + 1)),  # CJK Unified Ideographs
    ("ja", range(0x3040, 0x30FF + 1)),  # Hiragana + Katakana (Japanese)
    ("ar", range(0x0600, 0x06FF + 1)),  # Arabic
    ("ru", range(0x0400, 0x04FF + 1)),  # Cyrillic
    ("el", range(0x0370, 0x03FF + 1)),  # Greek
    ("hy", range(0x0530, 0x058F + 1)),  # Armenian
    ("am", range(0x1200, 0x137F + 1)),  # Ethiopic (Amharic)
)


def _sniff_language(text: str, *, sample_chars: int = 512) -> str:
    """Return a pysbd language code (``en`` default).

    Scans up to ``sample_chars`` of ``text`` and tallies hits per
    Unicode block. Picks the highest-count block; ties go to the order
    in ``_LANG_RANGES``. Returns ``en`` when nothing notable is found.

    O(min(N, sample_chars)). Deliberately tiny — runs once per
    ``chunk_text`` call, not once per sentence.
    """
    sample = text[:sample_chars]
    if not sample:
        return "en"
    counts: dict[str, int] = {}
    for ch in sample:
        cp = ord(ch)
        for code, rng in _LANG_RANGES:
            if cp in rng:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "en"
    # Highest count wins; on tie, _LANG_RANGES order is preserved by the
    # max() with a key that breaks ties via the table's index.
    order = {code: i for i, (code, _) in enumerate(_LANG_RANGES)}
    best = max(counts.items(), key=lambda kv: (kv[1], -order.get(kv[0], 99)))
    # Require at least 5% of the *actual sampled text* to be in this
    # script before switching off English. Threshold scales with sample
    # length, not the cap, so a short Hindi passage isn't dismissed
    # for failing an absolute character-count floor. Below 5% the bulk
    # is Latin and pysbd's English rules will perform better than any
    # minority language.
    threshold = max(3, len(sample) // 20)
    if best[1] < threshold:
        return "en"
    return best[0]


@lru_cache(maxsize=32)
def _pysbd_segmenter(language: str):
    """Return a cached pysbd Segmenter for ``language``.

    Constructing a segmenter loads regex tables; cache per language so
    repeat calls are cheap. ``clean=False`` keeps the returned strings
    untouched (we want to preserve trailing whitespace for offset math).
    """
    pysbd = _pysbd_module()
    if not pysbd:
        return None
    try:
        return pysbd.Segmenter(language=language, clean=False)
    except Exception:
        # Unknown / unsupported language — fall back to English so the
        # caller doesn't have to special-case it.
        try:
            return pysbd.Segmenter(language="en", clean=False)
        except Exception:
            return None


def _pysbd_split_paragraph(para_text: str):
    """Yield ``(rel_start, rel_end, sentence_text)`` from ``para_text``
    using pysbd. Mirrors the shape ``_iter_regions`` returns so the
    walker doesn't care which path produced the offsets.

    Falls back to the regex path on any pysbd failure or when pysbd
    isn't installed — the caller can rely on the same surface.
    """
    seg = _pysbd_segmenter(_sniff_language(para_text))
    if seg is None:
        # pysbd unavailable; fall through.
        for region in _iter_regions(_SENT, para_text):
            yield region
        return
    try:
        # pysbd returns the sentences in order with their leading
        # whitespace preserved (because clean=False). We re-derive
        # offsets by walking the original paragraph text and matching
        # each returned sentence as a substring from the cursor onward.
        # That avoids depending on pysbd's internal char-tracking which
        # has changed between versions.
        cursor = 0
        for sent in seg.segment(para_text):
            if not sent:
                continue
            stripped = sent.strip()
            if not stripped:
                continue
            try:
                idx = para_text.index(stripped, cursor)
            except ValueError:
                # Defensive: if pysbd mutated the text in any way,
                # fall back to the regex split for this paragraph.
                for region in _iter_regions(_SENT, para_text):
                    yield region
                return
            yield idx, idx + len(stripped), stripped
            cursor = idx + len(stripped)
    except Exception:
        # Any pysbd runtime failure → regex fallback. Don't let a
        # third-party splitter take down ingest.
        for region in _iter_regions(_SENT, para_text):
            yield region


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start: int = 0
    end: int = 0
    token_count: int = 0


@lru_cache(maxsize=1)
def _encoder():
    """Shared tokenizer. Returns the same handle :mod:`ext.services.budget`
    uses for token counting, so chunk boundaries align with the prompt
    budget pass.

    Cached for the life of the process — tokenizer load is the dominant
    first-call cost. The handle exposes ``encode(text) -> list[int]`` and
    ``decode(ids) -> str``, matching the surface this module relied on
    when it talked to ``tiktoken.Encoding`` directly.
    """
    from ext.services.budget import get_tokenizer  # local: avoid import cycle
    return get_tokenizer()


def _iter_regions(regex: "re.Pattern[str]", text: str):
    """Yield (abs_start, abs_end, slice) for regions between regex matches.

    The whole input is covered exactly once. Runs in O(N) via ``finditer``.
    """
    pos = 0
    for m in regex.finditer(text):
        yield pos, m.start(), text[pos:m.start()]
        pos = m.end()
    # Always yield the final region (may be empty on trailing separator).
    yield pos, len(text), text[pos:]


def _walk_sentences(text: str):
    """Yield (sentence_text, abs_start, abs_end) in order. O(N) total.

    Splits on paragraph breaks first, then sentence terminators. Preserves
    absolute character offsets in the original ``text`` so chunk offsets
    can be computed without any decode-and-count tricks.

    Sentence-splitter dispatch (review §2.5): when ``RAG_PYSBD_ENABLED=1``
    AND pysbd is importable, paragraphs are routed through pysbd's
    language-specific segmenter (Hindi danda ``।``, Chinese ``。``,
    French abbreviations, etc.). Otherwise the legacy English-centric
    regex ``_SENT`` runs unchanged. The pysbd path is always allowed to
    fall back to the regex on a per-paragraph basis if a runtime error
    pops up — see ``_pysbd_split_paragraph``.
    """
    use_pysbd = _pysbd_enabled()
    for para_start, _para_end, para_text in _iter_regions(_PARA, text):
        if use_pysbd:
            sent_iter = _pysbd_split_paragraph(para_text)
        else:
            sent_iter = _iter_regions(_SENT, para_text)
        for rel_start, _rel_end, sent_text in sent_iter:
            # Strip leading whitespace but keep the offset accurate.
            leading = len(sent_text) - len(sent_text.lstrip())
            stripped = sent_text.strip()
            if not stripped:
                continue
            a = para_start + rel_start + leading
            yield stripped, a, a + len(stripped)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[Chunk]:
    """Split ``text`` into chunks of roughly ``chunk_tokens`` tokens with
    ``overlap_tokens`` of tail-overlap between adjacent chunks.

    Sentence-aware: packs whole sentences until the next one would overflow
    the budget, then emits a chunk. A sentence that is itself larger than
    ``chunk_tokens`` is hard-split on token boundaries (rare — only happens
    for pathological input such as a single huge "word").

    Complexity: O(N) in input length — each token range is decoded at most
    O(1) times (never cumulatively).
    """
    if not text:
        return []
    if chunk_tokens <= overlap_tokens:
        raise ValueError("chunk_tokens must exceed overlap_tokens")

    enc = _encoder()

    # Single forward pass: collect (sentence_text, char_start, char_end, token_count).
    # Each sentence is encoded exactly once.
    meta: list[tuple[str, int, int, int]] = []
    for sent, a, b in _walk_sentences(text):
        meta.append((sent, a, b, len(enc.encode(sent))))
    if not meta:
        return []

    chunks: List[Chunk] = []
    cidx = 0
    i = 0
    n = len(meta)
    while i < n:
        # Greedy pack sentences [i, j) while we fit under the budget.
        j = i
        total = 0
        while j < n and total + meta[j][3] <= chunk_tokens:
            total += meta[j][3]
            j += 1

        if j == i:
            # One sentence is itself larger than chunk_tokens — hard token split.
            # This path also handles sentence-free input (e.g. one giant "word"),
            # which the sentence walker emits as a single region.
            sent, a, _b, _tc = meta[i]
            ids = enc.encode(sent)
            stride = chunk_tokens - overlap_tokens
            for k in range(0, len(ids), stride):
                sub_ids = ids[k : k + chunk_tokens]
                sub_text = enc.decode(sub_ids)
                # Best-effort char offsets: we know the sentence starts at ``a``
                # but without a char-by-char token alignment (expensive on long
                # tokens) we can't pinpoint sub-offsets exactly. Anchor at ``a``
                # for start and step by decoded-length for end.
                sub_start = a + k  # approximation: token index as byte offset
                sub_end = sub_start + len(sub_text)
                chunks.append(
                    Chunk(
                        index=cidx,
                        text=sub_text,
                        start=sub_start,
                        end=sub_end,
                        token_count=len(sub_ids),
                    )
                )
                cidx += 1
            i += 1
            continue

        # Emit a chunk covering sentences [i, j).
        start = meta[i][1]
        end = meta[j - 1][2]
        chunks.append(
            Chunk(
                index=cidx,
                text=text[start:end],
                start=start,
                end=end,
                token_count=total,
            )
        )
        cidx += 1

        if j >= n:
            break

        # Overlap: walk back from j while the trailing sentences fit in the
        # overlap budget. Advance i to that cutoff so the next chunk re-covers
        # those tail sentences. Guarantee forward progress: ``ni > i`` always.
        ni = j
        back = 0
        while ni > i + 1 and back + meta[ni - 1][3] <= overlap_tokens:
            ni -= 1
            back += meta[ni][3]
        i = ni if ni > i else j
    return chunks
