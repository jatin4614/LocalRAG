import asyncio
import pytest

from ext.services.circuit_breaker import CircuitBreaker, CircuitOpenError


def test_closed_breaker_passes_calls_through():
    cb = CircuitBreaker(name="test", fail_threshold=3, window_sec=5, cooldown_sec=1)
    for _ in range(5):
        cb.record_success()
    assert cb.state == "closed"


def test_breaker_opens_after_threshold_failures():
    cb = CircuitBreaker(name="test", fail_threshold=3, window_sec=5, cooldown_sec=1)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open"


@pytest.mark.asyncio
async def test_open_breaker_raises_until_cooldown():
    cb = CircuitBreaker(name="test", fail_threshold=2, window_sec=5, cooldown_sec=0.1)
    cb.record_failure(); cb.record_failure()
    assert cb.state == "open"
    with pytest.raises(CircuitOpenError):
        cb.raise_if_open()
    await asyncio.sleep(0.15)
    cb.raise_if_open()  # transitions to half_open, must not raise
    assert cb.state == "half_open"


def test_half_open_closes_on_success_opens_on_failure():
    cb = CircuitBreaker(name="test", fail_threshold=2, window_sec=5, cooldown_sec=0.01)
    cb.record_failure(); cb.record_failure()
    cb._state = "half_open"
    cb.record_success()
    assert cb.state == "closed"

    cb2 = CircuitBreaker(name="test2", fail_threshold=2, window_sec=5, cooldown_sec=0.01)
    cb2.record_failure(); cb2.record_failure()
    cb2._state = "half_open"
    cb2.record_failure()
    assert cb2.state == "open"


def test_failures_outside_window_do_not_accumulate():
    import time
    cb = CircuitBreaker(name="test", fail_threshold=3, window_sec=0.05, cooldown_sec=1)
    cb.record_failure(); cb.record_failure()
    time.sleep(0.08)
    cb.record_failure()
    assert cb.state == "closed"


def test_breaker_for_kill_switch_returns_noop(monkeypatch):
    """RAG_CIRCUIT_BREAKER_ENABLED=0 must short-circuit to a no-op breaker
    that always reports closed and never raises, regardless of failures."""
    from ext.services.circuit_breaker import _NoopBreaker, breaker_for

    monkeypatch.setenv("RAG_CIRCUIT_BREAKER_ENABLED", "0")
    cb = breaker_for("qdrant:any")
    assert isinstance(cb, _NoopBreaker)
    # No-op semantics: failures never accumulate, never opens, never raises.
    for _ in range(100):
        cb.record_failure()
    assert cb.state == "closed"
    cb.raise_if_open()  # must not raise
