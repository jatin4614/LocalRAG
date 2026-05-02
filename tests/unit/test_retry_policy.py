"""Unit tests for ``ext.services.retry_policy`` (review §3.4).

Focus: HTTP 429 must be classified as transient and retried. Without this
tests would only cover the existing behaviour. Pre-existing 5xx + connection
error coverage lives in ``test_retry_wrappers.py``.
"""
from __future__ import annotations

import httpx
import pytest

from ext.services.retry_policy import is_transient, with_transient_retry


def _status_error(code: int) -> httpx.HTTPStatusError:
    """Build an ``HTTPStatusError`` for a given HTTP status code."""
    request = httpx.Request("GET", "http://x")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


def test_is_transient_429_classified_as_transient():
    """HTTP 429 (Too Many Requests) is by definition retriable.

    Before the §3.4 fix, ``is_transient`` only matched 5xx; 429 fell into
    the non-transient bucket and starved out under upstream pressure.
    """
    assert is_transient(_status_error(429))


def test_is_transient_400_still_not_transient():
    """4xx other than 429 must remain non-transient — retrying a 400 wastes
    cycles and worsens user-visible latency on a permanent error."""
    assert not is_transient(_status_error(400))
    assert not is_transient(_status_error(401))
    assert not is_transient(_status_error(403))
    assert not is_transient(_status_error(404))


@pytest.mark.asyncio
async def test_retry_wrapper_retries_429_then_succeeds():
    """Mock a function that returns 429 then 200 — wrapper must retry."""
    calls = {"n": 0}

    @with_transient_retry(attempts=3, base_sec=0.01)
    async def rate_limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(429)
        return "ok"

    assert await rate_limited() == "ok"
    assert calls["n"] == 2
