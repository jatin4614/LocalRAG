"""Unit tests for ``_record_silent_failure`` auth-error escalation.

Background — 2026-05-03 incident:
    When Qdrant search itself failed inside ``_search_one`` (in
    ``ext/services/retriever.py``), the exception was swallowed by
    ``_record_silent_failure("retrieve.per_kb_search", exc)`` with only a
    WARNING-level log and a generic counter bump. During the brief auth
    misconfiguration window every per-KB search returned 401 → empty hit
    list → the LLM "answered" with hallucinated content for hours, because
    nothing escalated the noise to a level an operator would notice.

Contract under test:
    For ``stage == "retrieve.per_kb_search"`` AND the underlying error is
    an ``UnexpectedResponse`` with ``status_code in (401, 403)`` (or, as a
    string-match fallback, contains "Unauthorized" / "Forbidden"):
        * Log at ERROR level (not WARNING) so the default operator log
          filter catches it.
        * Bump a NEW counter label
          ``RAG_SILENT_FAILURE.labels(stage="retrieve.per_kb_search.auth_error")``
          so a Prometheus alert can fire on the auth-error label
          specifically.
        * KEEP the original ``stage="retrieve.per_kb_search"`` counter bump
          so existing dashboards keep working.
        * Do NOT raise — fail-open semantics are preserved (flipping to
          fail-closed is a future call).

    For ANY OTHER error (e.g. ConnectionError) on the same stage:
        * Behaviour is unchanged: WARNING-level log + only the original
          counter.
"""
from __future__ import annotations

import logging

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from ext.services.retriever import _record_silent_failure


def _unexpected_response(status_code: int, message: str = "HTTP error") -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code,
        reason_phrase=message,
        content=b"",
        headers=None,
    )


def _counter_value(stage: str) -> float:
    """Read the current rag_silent_failure_total{stage=...} value."""
    from ext.services.metrics import RAG_SILENT_FAILURE
    metric = RAG_SILENT_FAILURE.labels(stage=stage)
    # prometheus_client Counter exposes ``_value.get()`` for the underlying value.
    return metric._value.get()


# --------- 401 / 403 on retrieve.per_kb_search → ERROR + dedicated counter --


def test_auth_401_logs_error_and_bumps_dedicated_counter(caplog) -> None:
    """A 401 must escalate to ERROR + a dedicated ``.auth_error`` counter."""
    before_orig = _counter_value("retrieve.per_kb_search")
    before_auth = _counter_value("retrieve.per_kb_search.auth_error")
    err = _unexpected_response(401, "Unauthorized")

    with caplog.at_level(logging.DEBUG, logger="ext.services.retriever"):
        _record_silent_failure("retrieve.per_kb_search", err)

    # ERROR-level log emitted (not just WARNING).
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, (
        f"expected an ERROR-level log for 401 auth failure, "
        f"got records: {[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
    )

    # Original counter still bumped (don't break dashboards).
    assert _counter_value("retrieve.per_kb_search") == pytest.approx(before_orig + 1)
    # New dedicated counter bumped.
    assert _counter_value("retrieve.per_kb_search.auth_error") == pytest.approx(
        before_auth + 1
    )


def test_auth_403_logs_error_and_bumps_dedicated_counter(caplog) -> None:
    """403 forbidden behaves the same as 401 — operator-actionable."""
    before_auth = _counter_value("retrieve.per_kb_search.auth_error")
    err = _unexpected_response(403, "Forbidden")

    with caplog.at_level(logging.DEBUG, logger="ext.services.retriever"):
        _record_silent_failure("retrieve.per_kb_search", err)

    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert _counter_value("retrieve.per_kb_search.auth_error") == pytest.approx(
        before_auth + 1
    )


# --------- non-auth error on the same stage → unchanged behaviour ----------


def test_non_auth_error_keeps_warning_only(caplog) -> None:
    """A plain ConnectionError on retrieve.per_kb_search must NOT escalate.

    Most silent failures are legitimate fail-open paths; only auth-shaped
    errors are operator-actionable enough to deserve ERROR + a dedicated
    counter.
    """
    before_orig = _counter_value("retrieve.per_kb_search")
    before_auth = _counter_value("retrieve.per_kb_search.auth_error")
    err = ConnectionError("qdrant down")

    with caplog.at_level(logging.DEBUG, logger="ext.services.retriever"):
        _record_silent_failure("retrieve.per_kb_search", err)

    # No ERROR-level log emitted (only WARNING).
    assert not any(r.levelno >= logging.ERROR for r in caplog.records), (
        f"unexpected ERROR log for non-auth failure: "
        f"{[(r.levelname, r.getMessage()) for r in caplog.records]!r}"
    )
    # Some warning was logged (existing behaviour).
    assert any(r.levelno == logging.WARNING for r in caplog.records)

    # Original counter bumped, dedicated counter UNCHANGED.
    assert _counter_value("retrieve.per_kb_search") == pytest.approx(before_orig + 1)
    assert _counter_value("retrieve.per_kb_search.auth_error") == pytest.approx(before_auth)


def test_auth_401_on_unrelated_stage_does_not_escalate(caplog) -> None:
    """The escalation is scoped to the ``retrieve.per_kb_search`` stage —
    a 401 on, say, the rerank stage doesn't bump the dedicated counter
    (different blast radius, different alert).
    """
    before_orig = _counter_value("rerank.score")
    before_auth = _counter_value("retrieve.per_kb_search.auth_error")
    err = _unexpected_response(401, "Unauthorized")

    with caplog.at_level(logging.DEBUG, logger="ext.services.retriever"):
        _record_silent_failure("rerank.score", err)

    assert _counter_value("rerank.score") == pytest.approx(before_orig + 1)
    # Dedicated retrieve auth counter NOT bumped — wrong stage.
    assert _counter_value("retrieve.per_kb_search.auth_error") == pytest.approx(before_auth)


def test_auth_error_does_not_raise() -> None:
    """Fail-open is preserved — _record_silent_failure must never raise."""
    err = _unexpected_response(401, "Unauthorized")
    # Should not raise.
    _record_silent_failure("retrieve.per_kb_search", err)
