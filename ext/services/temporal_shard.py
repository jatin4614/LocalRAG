"""Temporal shard_key derivation for Qdrant custom sharding.

Plan B Phase 5.2. The shard_key partitions a KB by month — the
ingest pipeline derives it once per document and the same key is used
for every chunk of that document.

Format: ``"YYYY-MM"`` (zero-padded, ASCII). Always 7 chars. This makes
shard_keys sortable lexicographically and trivial to enumerate.

Priority order for derivation:
  1. Filename pattern (uses existing query_intent.extract_date_tuple)
  2. YAML frontmatter ``date:`` field
  3. First date in the body's first 1000 chars
  4. Current month at ingest (fallback; tagged with
     ShardKeyOrigin.INGEST_DEFAULT in payload for observability)
"""
from __future__ import annotations

import datetime as _dt
import enum
import re
from typing import Iterable, Tuple

from .query_intent import extract_date_tuple


class ShardKeyOrigin(enum.Enum):
    FILENAME = "filename"
    FRONTMATTER = "frontmatter"
    BODY = "body"
    INGEST_DEFAULT = "ingest_default"


# YAML front matter ``date:`` line — only matches at start-of-doc and
# requires a YAML fence on either side (to avoid grabbing arbitrary
# ``date:`` mentions deeper in markdown).
_FRONTMATTER_DATE_RE = re.compile(
    r"^---\s*\n(?:.*\n)*?date:\s*(\d{4})-(\d{1,2})-\d{1,2}",
    re.MULTILINE | re.IGNORECASE,
)

# 3-letter abbreviated month → numeric month. Mirrors the canonical form
# returned by ext.services.query_intent.extract_date_tuple.
_MONTH_NUM: dict[str, int] = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Month-year-only filename pattern (no day) — matches things like
# "Jan 23.docx", "feB 23.docx", "March 2023 report.pdf". Used for
# monthly-report archives where each filename is one whole month.
# extract_date_tuple in query_intent rejects this case because day-less
# dates are ambiguous in free text; for filenames the convention is
# unambiguous so we accept it here.
_FILENAME_MONTH_YEAR_RE = re.compile(
    r"(?P<month>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)"
    r"[\s_\-/\.]+"
    r"(?P<year>\d{2}|\d{4})"
    r"(?=$|[\s_\-/\.\(\)])",
    re.IGNORECASE,
)

_FILE_EXT_TAILS = (
    ".docx", ".doc", ".pdf", ".txt", ".md", ".html", ".htm",
    ".xlsx", ".xls", ".pptx", ".ppt", ".csv", ".rtf",
)


def _strip_known_ext(name: str) -> str:
    low = name.lower()
    for ext in _FILE_EXT_TAILS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return name


def _normalize_year_short(raw: str) -> int:
    """Expand 2-digit year using the standard pivot: <70 → 20xx, ≥70 → 19xx."""
    n = int(raw)
    if len(raw) == 2:
        return 1900 + n if n >= 70 else 2000 + n
    return n


def _month_token_to_num(tok: str) -> int | None:
    """Map any month spelling (full or 3-letter prefix) to its 1–12 number."""
    key = tok.strip().lower()[:3]
    short = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    return short.get(key)


def _date_to_shard_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def extract_shard_key(
    filename: str, body: str,
) -> Tuple[str, ShardKeyOrigin]:
    """Derive ``(shard_key, origin)`` from a document.

    Pure function — no I/O. Always returns a valid shard_key string.
    """
    # 1. Filename — strip known extensions first so a "23.docx" tail
    # doesn't confuse the month-year regex (the dot+digits look like a
    # day-of-month).
    if filename:
        stem = _strip_known_ext(filename)

        # 1a. Strict day-month-year (most specific) via the shared parser.
        tup = extract_date_tuple(stem)
        if tup is not None:
            day, month_str, year = tup
            return (
                _date_to_shard_key(year, _MONTH_NUM[month_str]),
                ShardKeyOrigin.FILENAME,
            )

        # 1b. Filename-only "MONTH YEAR" (no day), e.g. "Jan 23",
        # "feB 23", "March 2023 report". 2-digit year pivots at 70.
        m = _FILENAME_MONTH_YEAR_RE.search(stem)
        if m:
            mn = _month_token_to_num(m.group("month"))
            if mn is not None:
                year = _normalize_year_short(m.group("year"))
                return (
                    _date_to_shard_key(year, mn),
                    ShardKeyOrigin.FILENAME,
                )

    # 2. YAML frontmatter
    if body:
        m = _FRONTMATTER_DATE_RE.search(body[:2000])
        if m:
            return (
                _date_to_shard_key(int(m.group(1)), int(m.group(2))),
                ShardKeyOrigin.FRONTMATTER,
            )

    # 3. First date in body (first 1000 chars)
    if body:
        head = body[:1000]
        tup = extract_date_tuple(head)
        if tup is not None:
            day, month_str, year = tup
            return (
                _date_to_shard_key(year, _MONTH_NUM[month_str]),
                ShardKeyOrigin.BODY,
            )

    # 4. Fallback
    today = _dt.date.today()
    return (
        _date_to_shard_key(today.year, today.month),
        ShardKeyOrigin.INGEST_DEFAULT,
    )


_SHARD_KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def parse_shard_key(sk: str) -> Tuple[int, int]:
    """Parse a shard_key into ``(year, month)``. Raises ValueError on malformed."""
    m = _SHARD_KEY_RE.match(sk)
    if not m:
        raise ValueError(
            f"invalid shard_key {sk!r}; expected 'YYYY-MM' (zero-padded month)"
        )
    return int(m.group(1)), int(m.group(2))


def iter_shard_keys(start: str, end: str) -> Iterable[str]:
    """Yield consecutive shard_keys from ``start`` to ``end`` inclusive.

    Both endpoints are 'YYYY-MM' format. Crosses year boundaries.
    """
    sy, sm = parse_shard_key(start)
    ey, em = parse_shard_key(end)
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield _date_to_shard_key(y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


__all__ = [
    "ShardKeyOrigin",
    "extract_shard_key",
    "parse_shard_key",
    "iter_shard_keys",
]
