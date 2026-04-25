"""Unit tests for ``ext.services.llm_telemetry`` (P1.6).

The recorder must:
  1. Emit token counters (prompt + completion) on success, with the
     WIP-declared label set (``model``, ``kb`` for prompt; ``model`` for
     completion).
  2. Still emit token counters even when the wrapped block raises — the
     in-flight LLM call already burned tokens upstream and operators need
     to see that spend regardless of HTTP outcome.

We don't assert on TTFT/TPOT here: those are histograms that only fire
when the caller threads ``set_first_token_at()`` through a streaming
response. The current chat call sites (contextualizer, hyde,
query_rewriter) are non-streaming JSON POSTs, so they intentionally
leave the histograms alone. A follow-up that wraps the streaming
chat path would test those.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_record_llm_call_emits_token_counters_on_success(monkeypatch):
    from ext.services import llm_telemetry, metrics

    fake_prompt = MagicMock()
    fake_completion = MagicMock()
    monkeypatch.setattr(metrics, "tokens_prompt_total", fake_prompt)
    monkeypatch.setattr(metrics, "tokens_completion_total", fake_completion)
    monkeypatch.setattr(llm_telemetry, "tokens_prompt_total", fake_prompt)
    monkeypatch.setattr(llm_telemetry, "tokens_completion_total", fake_completion)

    async with llm_telemetry.record_llm_call(stage="contextualizer", model="gemma-4") as rec:
        rec.set_tokens(prompt=850, completion=50)

    fake_prompt.labels.assert_called_with(model="gemma-4", kb="none")
    fake_prompt.labels.return_value.inc.assert_called_with(850)
    fake_completion.labels.assert_called_with(model="gemma-4")
    fake_completion.labels.return_value.inc.assert_called_with(50)


@pytest.mark.asyncio
async def test_record_llm_call_emits_metrics_even_on_exception(monkeypatch):
    from ext.services import llm_telemetry, metrics

    fake_prompt = MagicMock()
    monkeypatch.setattr(metrics, "tokens_prompt_total", fake_prompt)
    monkeypatch.setattr(llm_telemetry, "tokens_prompt_total", fake_prompt)

    with pytest.raises(RuntimeError):
        async with llm_telemetry.record_llm_call(stage="hyde", model="gemma-4") as rec:
            rec.set_tokens(prompt=200, completion=0)
            raise RuntimeError("boom")

    # Counter still incremented in finally
    fake_prompt.labels.assert_called_with(model="gemma-4", kb="none")
    fake_prompt.labels.return_value.inc.assert_called_with(200)
