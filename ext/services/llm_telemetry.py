"""LLM call telemetry — async context manager for chat/completions calls.

Emits the WIP-declared LLM metrics (``rag_tokens_prompt_total``,
``rag_tokens_completion_total``, ``rag_llm_ttft_seconds``,
``rag_llm_tpot_seconds``) at the call site. Counters fire in ``finally``
so an HTTP failure mid-call still records the prompt-token spend
upstream — operators need that visibility regardless of outcome.

The ``stage`` argument (``contextualizer`` | ``hyde`` | ``rewriter`` …)
is carried for future label expansion + observability tracing, but the
WIP counters themselves are labelled only by ``model`` (and ``kb`` for
the prompt counter). Don't rename those — call sites and dashboards
already depend on the names. If you want per-stage breakdowns, add a
new metric instead of widening the existing one.

Usage:

    async with record_llm_call(stage="contextualizer", model="gemma-4") as rec:
        response = await client.post(...)
        usage = response.json().get("usage", {})
        rec.set_tokens(
            prompt=usage.get("prompt_tokens", 0),
            completion=usage.get("completion_tokens", 0),
        )

For streaming calls, additionally call ``rec.set_first_token_at(t)``
with ``time.perf_counter()`` at the moment the first token arrives —
that's the only way TTFT/TPOT histograms get observations. Non-streaming
JSON POSTs (the default in our chat call sites) intentionally leave the
histograms alone; recording a synthetic TTFT == total-latency would
poison those dashboards.

Fail-open: every metric write is wrapped in ``try/except`` so a broken
exporter cannot bubble up and break the wrapped LLM call. Metrics are
best-effort; the LLM call is not.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from ext.services.metrics import (
    llm_tpot_seconds,
    llm_ttft_seconds,
    tokens_completion_total,
    tokens_prompt_total,
)


@dataclass
class LlmCallRecorder:
    """Mutable state passed back to the caller from ``record_llm_call``.

    The caller writes token counts (and optionally a first-token wall time)
    into this object during the wrapped block; the context manager reads
    them in ``finally`` and emits the corresponding metric observations.
    """

    stage: str
    model: str
    kb: Optional[str] = None
    _prompt_tokens: int = 0
    _completion_tokens: int = 0
    _first_token_at: Optional[float] = None

    def set_tokens(self, *, prompt: int, completion: int) -> None:
        """Record token usage from the LLM response.

        Pass 0 for either side if the endpoint didn't return a usage block.
        Counters skip emission when the value is 0 (no point inflating a
        counter with no-op increments).
        """
        self._prompt_tokens = int(prompt or 0)
        self._completion_tokens = int(completion or 0)

    def set_first_token_at(self, t: float) -> None:
        """Record the wall-clock time (``time.perf_counter()``) when the
        first streamed token arrived.

        Must be called inside the ``async with`` block. When unset, the
        TTFT/TPOT histograms are left alone — see module docstring.
        """
        self._first_token_at = t

    def set_kb(self, kb: str) -> None:
        """Override the KB label for ``rag_tokens_prompt_total`` after entry.

        Useful when the KB selection isn't known until after the body
        is built (e.g. a router that resolves KB IDs from the request).
        """
        self.kb = kb


@asynccontextmanager
async def record_llm_call(
    *,
    stage: str,
    model: str,
    kb: Optional[str] = None,
) -> AsyncIterator[LlmCallRecorder]:
    """Async context manager that emits LLM telemetry on exit.

    Args:
        stage: Logical pipeline stage — ``contextualizer``, ``hyde``,
            ``rewriter``, etc. Carried on the recorder for tracing /
            future label expansion. Does NOT label the WIP counters
            (those are model-scoped only).
        model: The chat model name (e.g. ``orgchat-chat``,
            ``gemma-4``). Read by the caller from ``CHAT_MODEL`` env or
            request config — keep it stable across calls so dashboards
            aggregate cleanly.
        kb: Comma-joined KB identifier(s) used by the request, or
            ``None``. Becomes the ``kb`` label on
            ``rag_tokens_prompt_total``; defaults to ``"none"`` when
            unset.
    """
    rec = LlmCallRecorder(stage=stage, model=model, kb=kb)
    t0 = time.perf_counter()
    try:
        yield rec
    finally:
        dur = time.perf_counter() - t0
        # Token counters. Skip emission on 0 to keep dashboards readable
        # when an endpoint omits the usage block. ``or "none"`` guards
        # against a None ``kb`` blowing up the labels() call — Prometheus
        # rejects None as a label value at scrape time.
        try:
            if rec._prompt_tokens > 0:
                tokens_prompt_total.labels(
                    model=rec.model, kb=(rec.kb or "none")
                ).inc(rec._prompt_tokens)
        except Exception:
            pass
        try:
            if rec._completion_tokens > 0:
                tokens_completion_total.labels(model=rec.model).inc(
                    rec._completion_tokens
                )
        except Exception:
            pass
        # Streaming-only histograms. Only fire when the caller threaded
        # set_first_token_at(...) — see module docstring rationale.
        if rec._first_token_at is not None:
            ttft = rec._first_token_at - t0
            try:
                llm_ttft_seconds.labels(model=rec.model).observe(ttft)
            except Exception:
                pass
            if rec._completion_tokens > 0:
                tpot = (dur - ttft) / rec._completion_tokens
                try:
                    llm_tpot_seconds.labels(model=rec.model).observe(tpot)
                except Exception:
                    pass


__all__ = ["LlmCallRecorder", "record_llm_call"]
