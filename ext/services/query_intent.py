"""Query-intent classifier (regex fast path + LLM hybrid router).

Classifies the user's query into one of four labels that drive the
retrieval pipeline:

  * ``metadata`` — the user wants the catalog itself ("list documents",
    "what files do I have"). These are answered by the KB catalog
    preamble; no chunk retrieval is needed.
  * ``global`` — the user wants aggregation/coverage across the corpus
    ("list all dates", "every report", "summarize the entire KB").
    Best served by doc-level summary points (``level="doc"``) so every
    document contributes exactly one point to top-k.
  * ``specific_date`` — the user pinpoints a date ("gun area outages of
    5 Jan 2026", "what did the 03 Feb report say"). The bridge extracts
    the date tuple via ``extract_date_tuple`` and narrows retrieval to
    matching ``doc_id``s. Guarantees the right doc when ranking signals
    can't disambiguate "5 Jan" from "5 Feb" / "4 Jan".
  * ``specific`` — everything else. Standard single-doc / content-anchored
    query; the current top-k chunk pipeline handles it well.

The fast path is pure regex against a lowercased, stripped query —
sub-millisecond, zero I/O.

Plan B Phase 4 added a hybrid LLM tiebreaker exposed via
:func:`classify_with_qu`. It escalates ``specific``-labelled queries to
the QU LLM (``ext.services.query_understanding``) when one of six
predicates fires (pronoun reference, relative time, multi-clause, long
query, comparison verb, question-word with no entity). The legacy
``RAG_INTENT_LLM`` flag is retired (Plan B Phase 4.10); use
``RAG_QU_ENABLED`` instead.
"""
from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from .query_understanding import QueryUnderstanding


Intent = Literal["metadata", "global", "specific_date", "specific"]


# Normalized month lookup — keys are lowercase 3-letter or full month name,
# values are the canonical 3-letter form used by the corpus filenames
# ("05 Jan 2026.docx"). A future-proofing note: corpus filenames use
# mixed capitalization ("17 JAn 2026.docx"); we always compare
# case-insensitively in SQL via ILIKE, so the canonical string here is
# cosmetic, but stable.
_MONTH_CANONICAL: dict[str, str] = {
    "jan": "Jan", "january": "Jan",
    "feb": "Feb", "february": "Feb",
    "mar": "Mar", "march": "Mar",
    "apr": "Apr", "april": "Apr",
    "may": "May",
    "jun": "Jun", "june": "Jun",
    "jul": "Jul", "july": "Jul",
    "aug": "Aug", "august": "Aug",
    "sep": "Sep", "sept": "Sep", "september": "Sep",
    "oct": "Oct", "october": "Oct",
    "nov": "Nov", "november": "Nov",
    "dec": "Dec", "december": "Dec",
}

_MONTH_GROUP = (
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
)

# "5 Jan 2026", "05 Jan 2026", "5-jan-2026", "05 JAN 26", "5th Jan 2026"
_DATE_RE_DMY = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[\s/\-\.]+{_MONTH_GROUP}[\s/\-\.,]+(?P<year>\d{{2,4}})\b",
    re.IGNORECASE,
)

# "Jan 5 2026", "January 5, 2026", "Jan 5th, 2026"
_DATE_RE_MDY = re.compile(
    rf"\b{_MONTH_GROUP}[\s/\-\.]+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[\s/\-\.,]+(?P<year>\d{{2,4}})\b",
    re.IGNORECASE,
)

# ISO: "2026-01-05"
_DATE_RE_ISO = re.compile(r"\b(?P<year>\d{4})-(?P<month_num>\d{1,2})-(?P<day>\d{1,2})\b")

_NUM_MONTH = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# --------------------------------------------------------------------------
# Plan B Phase 4.4 — escalation predicates for the hybrid regex+LLM router.
#
# These detect query shapes the regex fast path can't disambiguate. Each
# predicate is independent; the first match (in fixed order) becomes the
# escalation reason so shadow-mode A/B logging stays deterministic.
# --------------------------------------------------------------------------
_PRONOUN_RE = re.compile(
    r"\b(it|that|this|those|these|they|them)\b", re.IGNORECASE
)
_RELATIVE_TIME_RE = re.compile(
    r"\b(last|previous|next|coming|prior)\s+(week|month|quarter|year|day)\b|"
    r"\b(yesterday|tomorrow|today)\b",
    re.IGNORECASE,
)
_MULTI_CLAUSE_CONNECTOR_RE = re.compile(
    r"\b(and|or|but|also|while|whereas)\b", re.IGNORECASE
)
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|how|when|where|why|which|who|do|did|is|was|are|were)\b",
    re.IGNORECASE,
)
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_COMPARISON_VERB_RE = re.compile(
    r"\b(compare|contrast|differ|change|evolve|trend)\w*\b", re.IGNORECASE
)

_LONG_QUERY_TOKEN_THRESHOLD = 25
_MULTI_CLAUSE_TOKEN_THRESHOLD = 8


class EscalationReason(enum.Enum):
    """Why the hybrid router decided to escalate to the LLM (or didn't)."""

    NONE = "none"
    PRONOUN_REF = "pronoun_ref"
    RELATIVE_TIME = "relative_time"
    MULTI_CLAUSE = "multi_clause"
    LONG_QUERY = "long_query"
    NO_ENTITY = "no_entity_question"
    COMPARISON_VERB = "comparison_verb"


def should_escalate_to_llm(
    query: str,
    regex_label: str,
    history: list[dict] | None,
) -> Tuple[bool, EscalationReason]:
    """Decide whether the hybrid router should consult the QU LLM.

    Returns ``(escalate, reason)``. Escalation is only considered when the
    regex result is ``"specific"`` — the other labels (``metadata``,
    ``global``, ``specific_date``) are trusted as-is. Predicates fire in
    fixed order so the same input always picks the same reason; that
    determinism is required by shadow-mode A/B logging.
    """
    if regex_label != "specific":
        return False, EscalationReason.NONE
    if not query:
        return False, EscalationReason.NONE

    history = history or []
    tokens = query.split()
    n_tokens = len(tokens)

    # 1. Pronoun reference — only meaningful with history
    if history and _PRONOUN_RE.search(query):
        return True, EscalationReason.PRONOUN_REF

    # 2. Relative time
    if _RELATIVE_TIME_RE.search(query):
        return True, EscalationReason.RELATIVE_TIME

    # 3. Multi-clause: connector + non-trivial length
    if (
        n_tokens > _MULTI_CLAUSE_TOKEN_THRESHOLD
        and _MULTI_CLAUSE_CONNECTOR_RE.search(query)
    ):
        return True, EscalationReason.MULTI_CLAUSE

    # 4. Long query
    if n_tokens > _LONG_QUERY_TOKEN_THRESHOLD:
        return True, EscalationReason.LONG_QUERY

    # 5. Comparison verb
    if _COMPARISON_VERB_RE.search(query):
        return True, EscalationReason.COMPARISON_VERB

    # 6. Question word + no capitalized entity
    if _QUESTION_WORD_RE.search(query) and not _CAPITALIZED_TOKEN_RE.search(query):
        return True, EscalationReason.NO_ENTITY

    return False, EscalationReason.NONE


def _normalize_year(raw: str) -> Optional[int]:
    """Coerce 2- and 4-digit years to a 4-digit int.

    Two-digit years resolve to 2000+ (20xx) on the assumption that the
    corpus is dated in the current century. Filenames in this codebase
    use both "26" and "2026" forms, so both must map to 2026. Returns
    None if the string can't be parsed.
    """
    try:
        n = int(raw)
    except (ValueError, TypeError):
        return None
    if n < 100:
        return 2000 + n
    if 1000 <= n <= 9999:
        return n
    return None


def _normalize_day(raw: str) -> Optional[int]:
    try:
        n = int(raw)
    except (ValueError, TypeError):
        return None
    return n if 1 <= n <= 31 else None


def extract_date_tuple(query: str) -> Optional[Tuple[int, str, int]]:
    """Best-effort extraction of a (day, month_3letter, year) tuple.

    Returns ``None`` when no unambiguous date pattern is found. Handles:
      * "5 Jan 2026" / "05 Jan 2026" / "05 JAN 26"
      * "5th Jan, 2026" / "05/Jan/2026"
      * "Jan 5 2026" / "January 5, 2026"
      * ISO "2026-01-05"

    Does NOT handle "5/1/2026" (DD/MM vs MM/DD is ambiguous without
    locale context) or bare "5 Jan" (year inferred — unsafe for a
    corpus that spans multiple years). The regex is intentionally
    conservative: better to fall through to ``specific`` intent than to
    route a query to the wrong doc.
    """
    if not query:
        return None

    # Try each pattern in preference order (most specific first).
    for rx in (_DATE_RE_DMY, _DATE_RE_MDY):
        m = rx.search(query)
        if m:
            day = _normalize_day(m.group("day"))
            year = _normalize_year(m.group("year"))
            month_key = m.group("month").lower()
            month = _MONTH_CANONICAL.get(month_key)
            if day and year and month:
                return (day, month, year)

    m = _DATE_RE_ISO.search(query)
    if m:
        day = _normalize_day(m.group("day"))
        year = _normalize_year(m.group("year"))
        try:
            month_num = int(m.group("month_num"))
            month = _NUM_MONTH.get(month_num)
        except (ValueError, TypeError):
            month = None
        if day and year and month:
            return (day, month, year)

    return None


# --------------------------------------------------------------------------
# Pattern tables
# --------------------------------------------------------------------------
# Each tuple is ``(label_for_reason, compiled_regex)``. Order matters
# inside a list — first match wins. We compile at import time so the hot
# path is a bare ``regex.search`` call.
#
# METADATA patterns match queries where the user wants the catalog of
# available items, not their content. These return an instant answer
# from the KB catalog preamble.
_METADATA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("metadata:what_files_do_i_have",
     re.compile(r"\bwhat\s+(files|reports|documents|docs)\s+do\s+(i|we|you)\s+have\b")),
    ("metadata:do_you_have",
     re.compile(r"\bdo\s+you\s+have\b")),
    # Bare enumeration: "list reports", "what files", "show docs", optionally
    # preceded by "the". Does NOT match "list all/every <docs>" — that form
    # is a global aggregation and routes to the doc-summary index (see the
    # global:list_all_every rule below).
    ("metadata:enumerate_docs",
     re.compile(r"^(what|which|list|show)\s+(the\s+)?(reports?|files?|documents?|docs?)\b")),
    ("metadata:how_many_docs",
     re.compile(r"\bhow\s+many\s+(reports?|files?|documents?|docs?)\b")),
    ("metadata:give_me_the_list",
     re.compile(r"\bgive\s+me\s+the\s+list\b")),
    ("metadata:catalog_keyword",
     re.compile(r"\b(catalog|inventory)\b")),
]

# GLOBAL patterns match queries that want aggregation / coverage across
# the whole corpus. These get routed to the doc-summary index.
_GLOBAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("global:list_all_every",
     re.compile(r"^list\s+(all|every)\b")),
    ("global:every_x",
     re.compile(r"\bevery\s+(report|file|date|month|entry|document|doc)\b")),
    ("global:all_the_x",
     re.compile(r"\ball\s+the\s+(reports?|dates?|entries|documents?|files?|months?)\b")),
    ("global:across_all",
     re.compile(r"\bacross\s+all\b")),
    ("global:summarize_all",
     re.compile(r"\bsummari[zs]e\s+(all|the\s+entire|everything)\b")),
    ("global:summarise_broad",
     re.compile(r"\bsummari[zs]e\s+(the\s+)?(communications?|reports?|events?|"
                r"month|week|quarter|year|day|period|jan(?:uary)?|feb(?:ruary)?|"
                r"mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
                r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b")),
    ("global:full_list",
     re.compile(r"\bfull\s+list\b")),
    ("global:enumerate",
     re.compile(r"\benumerate\b")),
    # Aggregation phrasings where the user wants a cross-doc synthesis.
    # Vague-but-broad queries like "overview of the March month",
    # "summary of the quarter", "state of communications", "recap of
    # last week" are aggregation queries: they span many docs and the
    # top chunks usually won't rank well (dense similarity on broad
    # language is noisy). The doc-summary index is the right level.
    ("global:overview_of",
     re.compile(r"\b(complete\s+|full\s+|entire\s+)?overview\s+of\b")),
    ("global:summary_of",
     re.compile(r"\b(complete\s+|full\s+|entire\s+)?summary\s+of\b")),
    ("global:recap_of",
     re.compile(r"\brecap\s+of\b")),
    ("global:state_of",
     re.compile(r"\bstate\s+of\b")),
    ("global:of_the_time_window",
     re.compile(r"\b(for|in|during|over|of)\s+(the\s+)?"
                r"(month|week|quarter|year|day|period|"
                r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
                r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
                r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(month|"
                r"week|quarter|year)\b")),
    # Catch "highlights/key/main/major <topics|points|themes> in/from/of <period>"
    # The noun group is optional ("highlights of the week" works too).
    ("global:highlights_of",
     re.compile(r"\b(highlights?|key|main|major)\s+"
                r"(points?|topics?|themes?|events?|findings?)?\s*(of|in|from|for)\b")),
]


_DEFAULT_REASON = "default:no_pattern_matched"


def classify(query: str) -> Intent:
    """Return the intent label for ``query``.

    Safe on empty/None input (returns ``"specific"``). The LLM
    tiebreaker (flag-gated) is consulted only when the fast path
    returned ``"specific"`` — the two positive classes are trusted as-is.
    """
    label, _ = classify_with_reason(query)
    return label


def classify_with_reason(query: str) -> Tuple[Intent, str]:
    """Like ``classify`` but also returns the pattern label that matched.

    The reason string is intended for observability (logs, SSE events)
    so an operator can tell why a query was routed where it was.
    Format: ``"<label>:<rule_name>"`` for positive matches,
    ``"default:no_pattern_matched"`` for the fallback.
    """
    if not query:
        return "specific", _DEFAULT_REASON
    q = query.strip().lower()
    if not q:
        return "specific", _DEFAULT_REASON

    # Order matters.
    #
    # 1. ``metadata`` first — "list files" is answered by the catalog, not
    #    by cross-doc aggregation or content retrieval.
    # 2. ``global`` second — aggregation/coverage queries win over
    #    date-matching, since "list every report from January 2026"
    #    would otherwise get pinned to whichever "January 2026" doc
    #    matched the regex.
    # 3. ``specific_date`` third — if the query contains a parseable date
    #    (eg "gun area outages of 5 Jan 2026"), the bridge uses the
    #    extracted tuple to narrow retrieval to matching ``doc_id``s.
    #    The bridge re-extracts the tuple via ``extract_date_tuple`` so
    #    we don't have to change the signature of this function.
    # 4. Fallback: ``specific``. The optional LLM tiebreaker has moved to
    #    the async :func:`classify_with_qu` helper (Plan B Phase 4) and is
    #    no longer reachable from this synchronous function.
    for reason, pat in _METADATA_PATTERNS:
        if pat.search(q):
            return "metadata", reason
    for reason, pat in _GLOBAL_PATTERNS:
        if pat.search(q):
            return "global", reason

    date_tuple = extract_date_tuple(query)
    if date_tuple:
        day, month, year = date_tuple
        return "specific_date", f"specific_date:extracted={day} {month} {year}"

    # Plan B Phase 4 — the legacy ``RAG_INTENT_LLM`` tiebreaker is retired.
    # The async hybrid classifier (:func:`classify_with_qu`) is the only
    # supported LLM path and is governed by ``RAG_QU_ENABLED``. Sync
    # callers (logging hooks, debug endpoints) keep regex-only behaviour.
    return "specific", _DEFAULT_REASON


@dataclass
class HybridClassification:
    """Result of :func:`classify_with_qu`.

    ``intent`` and ``resolved_query`` are the primary signals consumed by
    the chat bridge. ``source`` is one of ``"regex"`` / ``"llm"`` —
    metric labels and shadow-mode logging key off it. ``escalation_reason``
    records which predicate fired (or :attr:`EscalationReason.NONE` for a
    non-escalated regex hit). ``regex_reason`` is the rule label from
    :func:`classify_with_reason` for observability.
    """

    intent: str
    resolved_query: str
    temporal_constraint: Optional[dict]
    entities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "regex"
    escalation_reason: EscalationReason = EscalationReason.NONE
    regex_reason: str = ""
    cached: bool = False


async def _invoke_qu(
    query: str, history: list[dict],
) -> Optional["QueryUnderstanding"]:
    """Indirection so tests can monkey-patch the LLM call.

    Local import keeps the regex hot-path free of httpx + dataclass
    overhead; the import is only paid when the router escalates.
    """
    from .query_understanding import analyze_query

    return await analyze_query(query=query, history=history)


async def classify_with_qu(
    query: str, history: list[dict] | None = None,
) -> HybridClassification:
    """Hybrid regex+LLM classifier (Plan B Phase 4).

    Always runs regex first. Escalates to the QU LLM only when:
      * ``RAG_QU_ENABLED=1``, AND
      * regex returned ``"specific"``, AND
      * an escalation predicate fired (see :func:`should_escalate_to_llm`).

    On QU failure (timeout, HTTP error, schema violation) or when the
    LLM's confidence is below 0.5, the regex result is returned. The
    bridge can rely on this never raising — it's safe to call on every
    query.

    Increments :data:`ext.services.metrics.rag_qu_invocations` exactly
    once per call (with the source that ultimately won) and
    :data:`rag_qu_escalations` once per LLM escalation.
    """
    # Local import keeps the metrics dep out of import-time fast paths.
    from .metrics import rag_qu_escalations, rag_qu_invocations

    regex_label, regex_reason = classify_with_reason(query)
    history = history or []

    # Default: regex result wins
    result = HybridClassification(
        intent=regex_label,
        resolved_query=query,
        temporal_constraint=None,
        entities=[],
        confidence=1.0,
        source="regex",
        escalation_reason=EscalationReason.NONE,
        regex_reason=regex_reason,
    )

    if os.environ.get("RAG_QU_ENABLED", "0") != "1":
        try:
            rag_qu_invocations.labels(source="regex").inc()
        except Exception:
            pass
        return result

    escalate, reason = should_escalate_to_llm(query, regex_label, history)
    result.escalation_reason = reason
    if not escalate:
        try:
            rag_qu_invocations.labels(source="regex").inc()
        except Exception:
            pass
        return result

    try:
        rag_qu_escalations.labels(reason=reason.value).inc()
    except Exception:
        pass

    qu = await _invoke_qu(query, history)
    if qu is None or qu.confidence < 0.5:
        try:
            rag_qu_invocations.labels(source="regex").inc()
        except Exception:
            pass
        return result

    try:
        rag_qu_invocations.labels(source="llm").inc()
    except Exception:
        pass
    return HybridClassification(
        intent=qu.intent,
        resolved_query=qu.resolved_query,
        temporal_constraint=qu.temporal_constraint,
        entities=qu.entities,
        confidence=qu.confidence,
        source="llm",
        escalation_reason=reason,
        regex_reason=regex_reason,
        cached=qu.cached,
    )


__all__ = [
    "Intent",
    "classify",
    "classify_with_reason",
    "classify_with_qu",
    "extract_date_tuple",
    "should_escalate_to_llm",
    "EscalationReason",
    "HybridClassification",
]
