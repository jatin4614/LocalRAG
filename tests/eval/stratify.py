"""Stratification helpers for eval golden set.

The eval harness calls `stratify(rows)` to produce per-stratum subsets that
metrics get aggregated over. Strata: by intent, by year, by difficulty, by
language, and intent×year (for regression attribution across time buckets).
"""
from __future__ import annotations
from collections import defaultdict
from typing import Iterable


def _bucket(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        out[str(v)].append(r)
    return dict(out)


def stratify(rows: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """Return {dimension: {stratum_value: [rows]}}."""
    return {
        "intent": _bucket(rows, "intent_label"),
        "year": _bucket(rows, "year_bucket"),
        "difficulty": _bucket(rows, "difficulty"),
        "language": _bucket(rows, "language"),
        "adversarial_category": _bucket(
            [r for r in rows if r.get("adversarial_category")],
            "adversarial_category",
        ),
    }


def intent_year_strata(rows: list[dict]) -> dict[str, list[dict]]:
    """Cross-product strata: 'specific__2024', 'global__2025', ..."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        intent = r.get("intent_label")
        year = r.get("year_bucket")
        if intent and year:
            out[f"{intent}__{year}"].append(r)
    return dict(out)
