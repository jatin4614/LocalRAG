"""Shared transient-error retry decorator for outbound HTTP calls.

Wraps async functions that make HTTP calls to TEI, vllm-chat, reranker, etc.
Retries only on transient errors (timeouts, connection drops, 5xx); does NOT
retry on 4xx (bad request → retrying won't help) or other exceptions.

Feature-flagged by RAG_TENACITY_RETRY (default 1 = on). Set to 0 to disable
retry globally (useful for debugging retry storms).
"""
from __future__ import annotations

import logging
import os
from functools import wraps
from typing import Any, Callable, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def is_transient(exc: BaseException) -> bool:
    """True if retrying the call may succeed.

    Retriable: connection-level errors, 5xx server errors, and HTTP 429
    (Too Many Requests). 429 was added per review §3.4 — vLLM and TEI both
    emit it under load, and treating it as permanent caused user-visible
    failures the moment a single concurrent burst clipped the rate limit.
    All other 4xx codes remain non-transient (a 400 / 401 / 403 / 404 won't
    fix itself by waiting and retrying just amplifies pressure).
    """
    if isinstance(
        exc,
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            return True
        return 500 <= status < 600
    return False


def with_transient_retry(
    *,
    attempts: int = 3,
    base_sec: float = 0.5,
    max_sec: float = 5.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator. Retries on transient errors with exp backoff + jitter.

    Feature-gated by ``RAG_TENACITY_RETRY`` env var (default ``"1"`` = on).
    When set to ``"0"`` (or anything other than ``"1"``), the decorator
    becomes a no-op pass-through — single shot, original exception bubbles.
    Useful for debugging retry storms, or for forcing fail-open behaviour
    in tests / synthetic load.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if os.environ.get("RAG_TENACITY_RETRY", "1") != "1":
                return await fn(*args, **kwargs)
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(attempts),
                    wait=wait_exponential_jitter(initial=base_sec, max=max_sec),
                    retry=retry_if_exception(is_transient),
                    reraise=True,
                ):
                    with attempt:
                        return await fn(*args, **kwargs)
            except Exception:
                raise

        return wrapper

    return decorator


__all__ = ["is_transient", "with_transient_retry"]
