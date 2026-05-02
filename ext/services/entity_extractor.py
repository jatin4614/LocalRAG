"""Multi-entity query — entity list extractor (Phase 6.X — Method 5).

Inputs to:

* **Method 3** — multi-query decomposition (one sub-query per entity).
* **Method 4** — entity-aware Qdrant text filter (must.match.text per entity).

Two extraction paths, neither of which adds an LLM call beyond what
already runs:

1. **Regex** (always available, zero infra dependency). Detects
   numbered lists (``1. A`` / ``1) A`` / ``1 A``), bullet lists
   (``- A`` / ``* A``), and "for/about/regarding ... A, B, C, and D"
   comma+and lists. Single-item lists are rejected (not "multi-entity").

2. **QU LLM** (opt-in via ``RAG_QU_ENTITY_EXTRACT`` and only when
   ``RAG_QU_ENABLED=1``). Re-uses entities from
   :class:`ext.services.query_intent.HybridClassification`, which the
   QU LLM already populates as part of intent classification (see
   ``QU_OUTPUT_SCHEMA`` in ``query_understanding.py``). No second LLM
   round-trip.

The composer :func:`extract_entities` prefers QU output when present
(``len(qu.entities) >= 1``); otherwise it falls back to regex. Output
is deduped by case-insensitive key but the first surface form is kept
(so ``"32 Inf Bde"`` does not become ``"32 inf bde"``). Output is
capped at 8 entries — that's the asyncio.gather fan-out ceiling
Method 3 will respect, matched to the existing httpx pool size of 32.
"""
from __future__ import annotations

import re
from typing import Any, Optional


# Numbered list lines: ``1. X``, ``1) X``, ``1 X``. Matches the line
# verbatim so multiline ``re.findall`` returns the entity payload only.
# We require the digit + at most three chars of separator to avoid
# matching "1 small thing happened, then 2 things later" mid-prose.
_NUMBERED_RE = re.compile(
    r"^\s*(\d{1,2})[.\)]?\s+(.+?)\s*$",
    re.MULTILINE,
)

# Bullet list lines: ``- X`` / ``* X`` / ``• X``.
_BULLET_RE = re.compile(
    r"^\s*[-*•]\s+(.+?)\s*$",
    re.MULTILINE,
)

# Comma+and list. Triggered only when one of these preamble keywords is
# present — comma-splitting a free-form sentence is too noisy
# ("I went to A, B, C and slept" has no entities). The regex captures
# the run from the preamble word to end-of-sentence-or-line.
_LIST_PREAMBLE_RE = re.compile(
    r"\b(?:for|about|regarding|on|of|covering|including)\b"
    r"\s+([^.\n?!]+)",
    re.IGNORECASE,
)

# Trailing punctuation we strip from each entity surface form.
_TRAIL_PUNCT_RE = re.compile(r"[,.;:]+$")

# Tokens that are clearly not entities — used to filter comma-split
# results. Conservative; we'd rather miss an entity than hallucinate one.
_STOPWORDS = frozenset({
    "the", "a", "an", "this", "that", "those", "these",
    "and", "or", "with", "from", "into", "between",
    "month", "months", "year", "years", "report", "reports",
    "update", "updates", "summary", "details", "information",
})

# Hard cap on entity-list length. 8 chosen to bound asyncio.gather
# fan-out: the bridge's httpx pool is 32 connections shared across all
# concurrent requests; 8 sub-queries × ~4 concurrent requests = 32
# in-flight ceiling. Bigger fan-out would saturate the pool.
_MAX_ENTITIES = 8

# Minimum entity-list length to count a query as "multi-entity".
# A single-entity query goes through the existing single-query path
# (no decomposition).
_MIN_ENTITIES = 2


def _clean_surface(s: str) -> str:
    """Strip leading/trailing whitespace and trailing punctuation.

    Preserves internal casing/whitespace so ``"32 Inf Bde"`` is kept
    as written. Returns an empty string for ``None`` / non-string
    inputs.
    """
    if not isinstance(s, str):
        return ""
    out = s.strip()
    out = _TRAIL_PUNCT_RE.sub("", out)
    return out.strip()


def _dedupe_preserve_first(items: list[str]) -> list[str]:
    """Case-insensitive dedup, preserving the first surface form.

    ``["32 Inf Bde", "32 INF BDE", "75 Inf Bde"]``
        → ``["32 Inf Bde", "75 Inf Bde"]``
    """
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key = s.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _looks_like_entity(s: str) -> bool:
    """Heuristic: does this string look like a named entity?

    Conservative — used to filter comma-split candidates only. We accept
    anything not entirely a stopword and not too short/long.
    """
    if not s:
        return False
    if len(s) < 2 or len(s) > 80:
        return False
    if s.lower() in _STOPWORDS:
        return False
    return True


def _extract_numbered(text: str) -> list[str]:
    matches = _NUMBERED_RE.findall(text)
    return [_clean_surface(m[1]) for m in matches if _clean_surface(m[1])]


def _extract_bullets(text: str) -> list[str]:
    matches = _BULLET_RE.findall(text)
    return [_clean_surface(m) for m in matches if _clean_surface(m)]


def _extract_comma_and(text: str) -> list[str]:
    """Find ``for/about/regarding A, B, C, and D`` patterns.

    Walks each preamble match and splits the captured run on commas
    (and the optional Oxford ``and``). Each candidate is filtered by
    :func:`_looks_like_entity` so stopwords like "the" don't sneak in.
    Returns the longest candidate list found across all preamble
    matches — heuristic, but works on real queries.
    """
    best: list[str] = []
    for m in _LIST_PREAMBLE_RE.finditer(text):
        run = m.group(1)
        # Replace " and " near the end with "," so " A, B, and C" splits cleanly.
        run = re.sub(r"\s*,?\s+and\s+", ", ", run, flags=re.IGNORECASE)
        parts = [p.strip() for p in run.split(",")]
        cands = [p for p in parts if _looks_like_entity(p)]
        # Only keep if it's actually list-shaped (≥ 2 candidates).
        if len(cands) >= _MIN_ENTITIES and len(cands) > len(best):
            best = cands
    return best


def extract_entities_regex(query: str | None) -> list[str]:
    """Pure-regex entity extractor — no LLM, no I/O.

    Returns an empty list when:
      * input is empty / None
      * fewer than :data:`_MIN_ENTITIES` candidates are detected
        (single-entity queries go through the existing single-query path)
      * no list-shaped pattern is present

    Output is deduped (case-insensitive) and capped at :data:`_MAX_ENTITIES`.
    Pattern priority: numbered > bullets > comma+and. The first pattern
    that yields ≥ :data:`_MIN_ENTITIES` candidates wins; we don't merge
    across patterns to avoid double-counting the same list rendered two
    ways.
    """
    if not query or not isinstance(query, str):
        return []

    for extractor in (_extract_numbered, _extract_bullets, _extract_comma_and):
        cands = extractor(query)
        if len(cands) >= _MIN_ENTITIES:
            return _dedupe_preserve_first(cands)[:_MAX_ENTITIES]
    return []


def _entities_from_qu(qu_result: Any) -> list[str]:
    """Pull `.entities` off a QU result object, defensively.

    Returns ``[]`` when the object lacks the attribute or it's not
    list-like — caller will fall back to regex. Handles None entries,
    non-string entries, and whitespace.
    """
    raw = getattr(qu_result, "entities", None)
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        s = _clean_surface(item) if isinstance(item, str) else ""
        if s:
            cleaned.append(s)
    return cleaned


def _record_metric(source: str, count: int) -> None:
    """Bump ``rag_entity_extract_*`` counters; never raise.

    Local import keeps the regex hot-path import-graph free of
    prometheus_client when callers run tests without it. Mirrors the
    pattern in :mod:`ext.services.query_understanding`.
    """
    try:
        from .metrics import (
            rag_entity_extract_count,
            rag_entity_extract_total,
        )

        rag_entity_extract_total.labels(source=source).inc()
        rag_entity_extract_count.observe(min(count, 8))
    except Exception:
        pass


def extract_entities(
    query: str | None,
    qu_result: Optional[Any] = None,
) -> list[str]:
    """Compose QU + regex extraction into one entity list.

    Decision tree:
      1. If ``qu_result`` exposes a non-empty ``.entities`` list, use
         it. The QU LLM has already done the work as part of intent
         classification — no second LLM call.
      2. Otherwise fall back to :func:`extract_entities_regex`.

    Output is deduped case-insensitively (preserving first surface
    form) and capped at :data:`_MAX_ENTITIES`. Whitespace inside
    entity strings is preserved.

    Records ``rag_entity_extract_total{source="qu|regex|empty"}`` on
    every call. Metric failures are swallowed.
    """
    qu_entities = _entities_from_qu(qu_result)
    if qu_entities:
        out = _dedupe_preserve_first(qu_entities)[:_MAX_ENTITIES]
        _record_metric("qu", len(out))
        return out
    out = extract_entities_regex(query)
    _record_metric("regex" if out else "empty", len(out))
    return out


def is_multi_entity_query(
    query: str | None,
    qu_result: Optional[Any] = None,
) -> bool:
    """Convenience predicate the bridge uses to decide whether to run
    the multi-query decomposer.

    Returns ``True`` iff :func:`extract_entities` yields ≥
    :data:`_MIN_ENTITIES` results.
    """
    return len(extract_entities(query, qu_result)) >= _MIN_ENTITIES


__all__ = [
    "extract_entities",
    "extract_entities_regex",
    "is_multi_entity_query",
]
