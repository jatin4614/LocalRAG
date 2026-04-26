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


def _date_to_shard_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def extract_shard_key(
    filename: str, body: str,
) -> Tuple[str, ShardKeyOrigin]:
    """Derive ``(shard_key, origin)`` from a document.

    Pure function — no I/O. Always returns a valid shard_key string.
    """
    # 1. Filename
    if filename:
        tup = extract_date_tuple(filename)
        if tup is not None:
            day, month_str, year = tup
            return (
                _date_to_shard_key(year, _MONTH_NUM[month_str]),
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
