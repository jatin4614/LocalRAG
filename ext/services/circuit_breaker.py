"""Lightweight async-safe circuit breaker with sliding-window failure counting.

State machine:
    closed → (fail_threshold in window_sec) → open
    open → (cooldown_sec elapsed) → half_open
    half_open → (one success) → closed
    half_open → (one failure) → open

Deliberately simple — we don't pull in pybreaker for the core path because
our needs (per-KB keys, sliding window, async) are specific enough that a
~100-LOC local impl is clearer and easier to test than configuring pybreaker.
"""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from threading import Lock
from typing import Deque

logger = logging.getLogger(__name__)


class CircuitOpenError(RuntimeError):
    """Raised when a call is attempted against an open breaker."""


class CircuitBreaker:
    def __init__(
        self,
        *,
        name: str,
        fail_threshold: int = 3,
        window_sec: float = 300.0,
        cooldown_sec: float = 30.0,
    ) -> None:
        self.name = name
        self.fail_threshold = fail_threshold
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec
        self._failures: Deque[float] = deque()
        self._state = "closed"
        self._opened_at: float = 0.0
        self._lock = Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_transition()
            return self._state

    def _maybe_transition(self) -> None:
        now = time.monotonic()
        if self._state == "open" and now - self._opened_at >= self.cooldown_sec:
            self._state = "half_open"
            logger.info(
                "breaker %s: open → half_open after %.1fs cooldown",
                self.name, self.cooldown_sec,
            )

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._state = "closed"
                self._failures.clear()
                logger.info("breaker %s: half_open → closed (success)", self.name)
            elif self._state == "closed":
                self._failures.clear()

    def record_failure(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = now
                logger.warning("breaker %s: half_open → open (probe failed)", self.name)
                return
            self._failures.append(now)
            cutoff = now - self.window_sec
            while self._failures and self._failures[0] < cutoff:
                self._failures.popleft()
            if len(self._failures) >= self.fail_threshold and self._state == "closed":
                self._state = "open"
                self._opened_at = now
                logger.warning(
                    "breaker %s: closed → open (%d failures in %.0fs)",
                    self.name, len(self._failures), self.window_sec,
                )

    def raise_if_open(self) -> None:
        with self._lock:
            self._maybe_transition()
            if self._state == "open":
                raise CircuitOpenError(f"circuit {self.name!r} is open")


# Module-level registry — one breaker per KB (or 'global' for non-KB-scoped ops)
_BREAKERS: dict[str, CircuitBreaker] = {}
_REGISTRY_LOCK = Lock()


def breaker_for(key: str) -> "CircuitBreaker | _NoopBreaker":
    """Return (creating if needed) the breaker for the given key.

    Honors ``RAG_CIRCUIT_BREAKER_ENABLED=0`` as a kill switch — when off,
    returns a ``_NoopBreaker`` that never opens, so the wrapping code stays
    identical regardless of whether breakers are active.
    """
    if os.environ.get("RAG_CIRCUIT_BREAKER_ENABLED", "1") != "1":
        return _NoopBreaker()
    with _REGISTRY_LOCK:
        cb = _BREAKERS.get(key)
        if cb is None:
            cb = CircuitBreaker(
                name=key,
                fail_threshold=int(os.environ.get("RAG_CB_FAIL_THRESHOLD", "3")),
                window_sec=float(os.environ.get("RAG_CB_WINDOW_SEC", "300")),
                cooldown_sec=float(os.environ.get("RAG_CB_COOLDOWN_SEC", "30")),
            )
            _BREAKERS[key] = cb
        return cb


class _NoopBreaker:
    """Feature-flag-off sentinel — always closed."""
    state = "closed"

    def record_success(self) -> None:
        pass

    def record_failure(self) -> None:
        pass

    def raise_if_open(self) -> None:
        pass
