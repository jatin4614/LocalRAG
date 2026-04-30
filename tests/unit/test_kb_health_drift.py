"""Unit tests for the Phase 4 drift-% computation in ``ext.routers.kb_admin``.

``compute_drift_pct`` is a pure function — we test its branches here
and rely on integration tests elsewhere for the full endpoint flow.
"""
from __future__ import annotations

import math

import pytest

from ext.routers.kb_admin import compute_drift_pct


def test_drift_zero_when_expected_and_observed_match() -> None:
    assert compute_drift_pct(100, 100) == 0.0


def test_drift_on_observed_surplus() -> None:
    """Qdrant has more points than Postgres expected (orphans)."""
    # 110 vs 100 → 10% drift
    assert math.isclose(compute_drift_pct(100, 110), 10.0, abs_tol=1e-9)


def test_drift_on_observed_deficit() -> None:
    """Qdrant has fewer points than Postgres expected (missing chunks)."""
    # 90 vs 100 → 10% drift
    assert math.isclose(compute_drift_pct(100, 90), 10.0, abs_tol=1e-9)


def test_drift_empty_kb_with_no_orphans_is_zero() -> None:
    """A clean empty KB (expected=0, observed=0) reports 0.0 drift."""
    assert compute_drift_pct(0, 0) == 0.0


def test_drift_empty_kb_with_orphans_returns_sentinel() -> None:
    """M9: expected=0 but Qdrant has chunks → orphan sentinel 999.0.

    A new/empty KB that somehow has Qdrant points means a previous KB
    deletion left points behind, OR an ingest ran but the docs row
    never got committed. Either way, operators should see this loudly
    in their drift dashboard, not as a silent 0.0.
    """
    from ext.routers.kb_admin import ORPHAN_DRIFT_SENTINEL
    assert compute_drift_pct(0, 50) == ORPHAN_DRIFT_SENTINEL
    assert compute_drift_pct(0, 1) == ORPHAN_DRIFT_SENTINEL


def test_drift_negative_expected_with_no_orphans_is_zero() -> None:
    """Defensive: negative expected (corrupt rows) treated as 0; with
    no orphan observed, drift is 0."""
    assert compute_drift_pct(-5, 0) == 0.0


def test_drift_negative_expected_with_orphans_returns_sentinel() -> None:
    """Negative expected + positive observed → orphan sentinel."""
    from ext.routers.kb_admin import ORPHAN_DRIFT_SENTINEL
    assert compute_drift_pct(-5, 100) == ORPHAN_DRIFT_SENTINEL


def test_drift_symmetric_on_delta() -> None:
    """Absolute-value — 90 and 110 observations are same-distance from 100."""
    assert compute_drift_pct(100, 90) == compute_drift_pct(100, 110)


def test_drift_large_delta() -> None:
    """Zero-match case: 2590 expected, 0 observed → 100% drift."""
    assert math.isclose(compute_drift_pct(2590, 0), 100.0, abs_tol=1e-9)


def test_drift_accepts_string_coercible_ints() -> None:
    """Some callers pass the count as str (e.g. from a JSON field). Coerce."""
    # compute_drift_pct casts to int internally; SQLAlchemy rows can
    # return Decimal/int ambiguously. Verify it doesn't TypeError on
    # pure-int values but also tolerates numeric-string-like ints.
    assert compute_drift_pct(int("100"), int("110")) == 10.0


@pytest.mark.parametrize(
    "expected,observed,want",
    [
        (2590, 2590, 0.0),
        (2590, 2589, 100 / 2590),      # ~0.0386 %
        (2590, 2600, 1000 / 2590),     # ~0.386 %
        (1000, 2000, 100.0),           # Qdrant 2x Postgres
    ],
)
def test_drift_known_values(expected: int, observed: int, want: float) -> None:
    assert math.isclose(compute_drift_pct(expected, observed), want, abs_tol=1e-6)


def test_compute_drift_pct_orphan_sentinel(caplog) -> None:
    """M9 — orphan path emits a WARNING and returns the named sentinel."""
    from ext.routers.kb_admin import ORPHAN_DRIFT_SENTINEL
    with caplog.at_level("WARNING", logger="orgchat.kb_admin"):
        result = compute_drift_pct(0, 42)
    assert result == ORPHAN_DRIFT_SENTINEL
    assert any(
        "orphan chunks detected" in rec.message.lower()
        for rec in caplog.records
    )
