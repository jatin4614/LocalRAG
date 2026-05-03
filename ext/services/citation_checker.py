"""Inline citation enforcement for LLM responses (review §6.10).

The system prompt forbids inline citations (analyst-voice style guide), so
operators cannot tell whether the model cited the retrieved chunks or
hallucinated facts. This module provides a post-LLM guardrail:

  * For each factual sentence in the response,
  * check whether ANY retrieved source contains a 3-token shingle overlap.
  * Sentences with NO overlap are tagged ``[unverified]``.

Flag-gated by ``RAG_ENFORCE_CITATIONS`` (default ``"0"`` — pass-through).
When the flag is OFF, ``enforce_citations`` returns the input response
byte-identical and never touches sources.

Fail-open: any exception during the check is logged and the original
response is returned unchanged. The check must never break the response.

Why 3-token shingles (not embedding similarity):
  * Pure-Python, no GPU dependency on the response path.
  * Deterministic and explainable — operators can read the matching n-gram.
  * Cheap enough to run inline (<1ms for typical responses).

Why a noun-phrase + verb / date / number / proper-noun heuristic for
"factual claim":
  * Avoid tagging fillers ("Yes.", "I see.", "Got it.").
  * Avoid tagging hedges and meta-statements.
  * Catch the dangerous cases — claims with specific named entities,
    numbers, or dates that operators would actually want to verify.

Wired into the response boundary by callers (e.g. ``ext/routers/rag_stream.py``);
this module is content-only, no I/O.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, Sequence, Union

from . import flags
from .metrics import Counter

logger = logging.getLogger("orgchat.citation_checker")


# ---------------------------------------------------------------------------
# Counter — incremented per tagged sentence so dashboards can show the
# unverified-claim rate by intent.
# ---------------------------------------------------------------------------
rag_unverified_sentences_total = Counter(
    "rag_unverified_sentences_total",
    "Sentences in LLM responses that lacked any source-overlap match "
    "(post-check by citation_checker.enforce_citations); labelled by intent.",
    labelnames=("intent",),
)


# ---------------------------------------------------------------------------
# Sentence splitter — same regex semantics as ``ext.services.chunker._SENT``
# so the boundary behavior between chunker and post-LLM check matches.
# ---------------------------------------------------------------------------
# Sentence terminator followed by whitespace.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# ---------------------------------------------------------------------------
# Factual-claim heuristics. A sentence is treated as a factual claim if ANY of:
#   * Contains a digit (number, date, percentage, etc.).
#   * Contains an ISO-8601 date (extra-strong signal).
#   * Contains 2+ alphabetic tokens AND at least one capitalized token that
#     is NOT the leading word (proper noun heuristic — leading-word capitals
#     are sentence starts, not informative).
#   * Contains a verb-like token (length>=3, ending in -s, -ed, -ing, -ate)
#     AND at least one noun-like token (alphabetic, length>=4).
# ---------------------------------------------------------------------------
_DIGIT = re.compile(r"\d")
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TOKEN = re.compile(r"\w+")
_VERB_LIKE = re.compile(r"\b[a-z]{3,}(?:s|ed|ing|ate)\b")


def _looks_factual(sentence: str) -> bool:
    """Return True if ``sentence`` makes a factual claim (heuristic).

    Errs on the side of treating ambiguous sentences as factual — we'd
    rather tag too many than tag too few. False positives just add the
    visible ``[unverified]`` tag; false negatives let hallucinations
    through unflagged.
    """
    if not sentence or not sentence.strip():
        return False
    # Strong signal: any digit (covers years, percentages, IDs, counts).
    if _DIGIT.search(sentence):
        return True
    tokens = _TOKEN.findall(sentence)
    if len(tokens) < 2:
        # Single-word "Yes." / "No." / "What?" — not a factual claim.
        return False
    # Proper noun heuristic: a capitalized token NOT in leading position
    # (leading word capitalization = sentence-start, not a proper noun).
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if tok and tok[0].isupper() and tok[1:].islower():
            return True
        # All-caps acronyms (>=2 chars) also count as proper nouns.
        if len(tok) >= 2 and tok.isupper() and tok.isalpha():
            return True
    # Verb-like + a longish noun-like token: weak factual signal.
    has_verb = bool(_VERB_LIKE.search(sentence.lower()))
    has_noun = any(len(t) >= 4 and t.isalpha() for t in tokens)
    if has_verb and has_noun:
        return True
    return False


def _iter_sentences(text: str) -> list[tuple[str, int, int]]:
    """Yield ``(sentence, start_char, end_char)`` triples in document order.

    Indices are absolute offsets into ``text`` so the caller can re-assemble
    the response with in-place tags. Empty / whitespace-only fragments are
    skipped.
    """
    if not text:
        return []
    out: list[tuple[str, int, int]] = []
    pos = 0
    for m in _SENT_SPLIT.finditer(text):
        # Sentence is text[pos:m.start()] (inclusive of terminator).
        sent = text[pos:m.start()]
        if sent.strip():
            out.append((sent, pos, m.start()))
        pos = m.end()
    # Final sentence (no trailing whitespace+terminator).
    if pos < len(text):
        sent = text[pos:]
        if sent.strip():
            out.append((sent, pos, len(text)))
    return out


# ---------------------------------------------------------------------------
# Source-text extraction — sources may be passed as:
#   * list[str] (raw text)
#   * list[dict] with "text" key (legacy)
#   * list[dict] with "document" key (chat_rag_bridge sources_out shape)
#
# Strip ``<source id="…">…</source>`` XML wrapping if present.
# ---------------------------------------------------------------------------
_SOURCE_TAG = re.compile(r"<source\b[^>]*>(.*?)</source>", re.DOTALL | re.IGNORECASE)


def _extract_source_texts(sources: Sequence[Union[str, dict[str, Any]]]) -> list[str]:
    """Extract the raw text payload from a heterogeneous sources sequence.

    Strips XML ``<source>`` wrapping if present. Empty / non-string entries
    are skipped silently.
    """
    out: list[str] = []
    for s in sources or []:
        if isinstance(s, str):
            text = s
        elif isinstance(s, dict):
            # Try several known shapes:
            text = s.get("text") or ""
            if not text:
                docs = s.get("document") or []
                if isinstance(docs, list):
                    text = "\n".join(str(d) for d in docs if d)
                elif isinstance(docs, str):
                    text = docs
        else:
            continue
        if not text:
            continue
        # Strip <source id="…">payload</source> wrappers (keep payload).
        try:
            stripped = _SOURCE_TAG.sub(lambda m: m.group(1), str(text))
        except Exception:
            stripped = str(text)
        out.append(stripped)
    return out


# ---------------------------------------------------------------------------
# Shingle overlap — 3-token shingles, lowercased, alpha-num only.
# ---------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    """Tokenize lowercased alpha-num word boundaries."""
    return [t.lower() for t in _TOKEN.findall(text)]


def _shingles(tokens: Sequence[str], n: int = 3) -> set[tuple[str, ...]]:
    """Return the set of ``n``-token shingles for ``tokens``.

    Empty / shorter-than-n input yields the empty set.
    """
    if len(tokens) < n:
        # For very short sentences, fall back to the whole token sequence
        # as a single "shingle" so 1-2 token claims can still match.
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _has_overlap(sentence: str, source_token_sets: Sequence[set[tuple[str, ...]]]) -> bool:
    """Return True if any 3-token shingle of ``sentence`` appears in any
    source's shingle set.
    """
    sent_shingles = _shingles(_tokens(sentence), n=3)
    if not sent_shingles:
        return False
    for src_shingles in source_token_sets:
        if sent_shingles & src_shingles:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def is_enabled() -> bool:
    """Return True iff ``RAG_ENFORCE_CITATIONS`` env / overlay == "1"."""
    return flags.get("RAG_ENFORCE_CITATIONS", "0") == "1"


def enforce_citations(
    response: str,
    sources: Sequence[Union[str, dict[str, Any]]],
    *,
    intent: str = "specific",
) -> str:
    """Tag uncited factual sentences in ``response`` with ``[unverified]``.

    Returns the response unchanged when ``RAG_ENFORCE_CITATIONS`` is off,
    when ``response`` is empty, or when an internal error occurs (fail-open).

    Args:
      response: The full LLM response text. Streaming callers should
        accumulate the stream and call this once at the end.
      sources: The retrieved sources used to build the prompt — either
        raw strings, ``{"text": "..."}`` dicts, or chat_rag_bridge
        sources_out shape ``{"document": ["..."], ...}``.
      intent: Pipeline intent label (``"specific"`` | ``"global"`` |
        ``"metadata"`` | ``"specific_date"``) — used as the
        ``rag_unverified_sentences_total`` counter label.

    Returns:
      A new string with ``[unverified] `` prepended to each factual
      sentence that lacked source overlap. Non-factual sentences and
      cited sentences are left untouched.
    """
    # Flag check first — the cheapest possible exit when off.
    if not is_enabled():
        return response
    if not response:
        return response

    try:
        # Pre-compute source shingle sets (cost amortized across all
        # sentences). Heavy sources (~50KB context) → ~5K tokens →
        # ~5K shingles per source, set-membership is O(1).
        src_texts = _extract_source_texts(sources)
        src_token_sets: list[set[tuple[str, ...]]] = []
        for st in src_texts:
            toks = _tokens(st)
            src_token_sets.append(_shingles(toks, n=3))

        # Walk sentences and decide tag-or-skip per sentence.
        sentences = _iter_sentences(response)
        if not sentences:
            return response

        # Build the output: for each sentence we either keep as-is or
        # prepend ``[unverified] ``. Non-sentence whitespace/separators
        # between sentences are preserved exactly — we re-assemble using
        # the (start, end) offsets.
        pieces: list[str] = []
        cursor = 0
        tagged_count = 0
        for sent, a, b in sentences:
            # Inter-sentence whitespace / leading newlines.
            if a > cursor:
                pieces.append(response[cursor:a])
            if not _looks_factual(sent):
                pieces.append(sent)
            elif _has_overlap(sent, src_token_sets):
                pieces.append(sent)
            else:
                pieces.append(f"[unverified] {sent}")
                tagged_count += 1
            cursor = b
        # Trailing chunk (post-final-sentence whitespace).
        if cursor < len(response):
            pieces.append(response[cursor:])

        # Counter: bump per tagged sentence (best-effort).
        if tagged_count:
            try:
                rag_unverified_sentences_total.labels(intent=str(intent)).inc(tagged_count)
            except Exception:
                # Metric back-end missing or label-cardinality issue → never
                # break the response.
                pass

        return "".join(pieces)

    except Exception as exc:
        # Fail-open: log + return input unchanged.
        try:
            logger.warning(
                "citation_checker: enforce_citations failed (%s): %r",
                type(exc).__name__,
                exc,
            )
        except Exception:
            pass
        return response


__all__ = [
    "enforce_citations",
    "is_enabled",
    "rag_unverified_sentences_total",
]
