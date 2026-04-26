"""Intent-conditional time-decay multiplier.

Plan B Phase 5.7. ``score' = score * exp(-lambda * delta_days)``, applied
ONLY for current-state intent (``specific`` + present-tense + no temporal
constraint). Evolution / aggregation queries should NOT down-weight old
documents — that defeats the question.

``lambda`` derived from half-life: ``lambda = ln(2) / half_life_days``.
Default ``RAG_TIME_DECAY_LAMBDA_DAYS=90`` → 3-month half-life.

Summary-level RAPTOR nodes (level >= 2) are NOT decayed — they're
aggregates by design.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
from typing import Optional


# Present-tense / "now"-signal verbs and adverbs.
_PRESENT_TENSE_RE = re.compile(
    r"\b(is|are|am|has|have|currently|now|today|present|status|exists)\b",
    re.IGNORECASE,
)


def should_apply_time_decay(
    *,
    query: str,
    intent: str,
    temporal_constraint: Optional[dict],
) -> bool:
    """Decide whether to apply time-decay for this query.

    Conservative: returns True only for ``specific`` intent + present-tense
    signal + no explicit temporal constraint. All other paths return False.
    """
    if intent != "specific":
        return False
    if temporal_constraint:
        return False
    if not query:
        return False
    return bool(_PRESENT_TENSE_RE.search(query))


def time_decay_multiplier(*, age_days: float, lambda_days: float) -> float:
    """Return ``exp(-lambda * age_days)``.

    Negative or zero ages clamp to 1 (= no decay). Negative or zero
    ``lambda_days`` also returns 1 (decay is disabled).
    """
    if age_days <= 0 or lambda_days <= 0:
        return 1.0
    lam = math.log(2) / lambda_days
    return math.exp(-lam * age_days)


def _shard_key_age_days(shard_key: str) -> float:
    """Return age in days from today to the END of the shard's month.

    Anchoring at end-of-month means a hit that lives in the current
    month always has age=0 (today < end-of-month → clamped) and so
    receives no decay. Older months count from their last day, which
    is the most-recent timestamp anything in that shard could carry.
    Same intuition as "treat the shard as if everything in it was
    written on the last day of its month".
    """
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(shard_key)
    # End-of-month: first day of next month minus one day.
    if m == 12:
        next_month = _dt.date(y + 1, 1, 1)
    else:
        next_month = _dt.date(y, m + 1, 1)
    eom = next_month - _dt.timedelta(days=1)
    delta = (_dt.date.today() - eom).days
    return float(max(0, delta))


def apply_time_decay_to_hits(
    hits: list[dict], *, lambda_days: Optional[float] = None,
) -> list[dict]:
    """Multiply each hit's ``score`` by the time-decay factor in place.

    Hits without a ``shard_key`` payload are passed through unchanged.
    Hits at level >= 2 (summaries) are passed through unchanged.
    Returns the same list (mutates each dict's ``score``).
    """
    if lambda_days is None:
        try:
            lambda_days = float(
                os.environ.get("RAG_TIME_DECAY_LAMBDA_DAYS", "90")
            )
        except (ValueError, TypeError):
            lambda_days = 90.0
    for hit in hits:
        payload = hit.get("payload") or {}
        if payload.get("level", 0) >= 2:
            continue
        sk = payload.get("shard_key")
        if not sk:
            continue
        try:
            age_days = _shard_key_age_days(sk)
        except ValueError:
            # malformed shard_key in payload — skip silently
            continue
        mul = time_decay_multiplier(age_days=age_days, lambda_days=lambda_days)
        hit["score"] = hit["score"] * mul
    return hits


__all__ = [
    "should_apply_time_decay",
    "time_decay_multiplier",
    "apply_time_decay_to_hits",
]
