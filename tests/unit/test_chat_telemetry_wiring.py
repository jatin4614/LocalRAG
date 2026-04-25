"""Verify the chat streaming path emits LLM telemetry through the
canonical ``record_llm_call`` helper.

Phase 1.6 follow-up. ``record_llm_call`` is the single source of truth
for ``rag_llm_ttft_seconds`` / ``rag_llm_tpot_seconds`` /
``rag_tokens_*`` observations. ``ext/routers/rag_stream.py`` previously
called ``llm_ttft_seconds.labels(...).observe(...)`` directly, drifting
from the contextualizer / hyde / rewriter paths and making it harder
for operators to add cross-cutting telemetry (per-stage labels, retries,
etc.).

This test asserts:
1. ``rag_stream.py`` imports ``record_llm_call`` from
   ``ext.services.llm_telemetry``.
2. The streaming response handler uses the recorder context manager
   and threads a ``set_first_token_at`` call through it.
"""
from __future__ import annotations

from pathlib import Path


def _read_rag_stream_source() -> str:
    p = Path(__file__).resolve().parents[2] / "ext" / "routers" / "rag_stream.py"
    return p.read_text(encoding="utf-8")


def test_rag_stream_imports_record_llm_call():
    src = _read_rag_stream_source()
    # The exact import line; case-sensitive. We check for the symbol
    # because the import line might be aliased (`as _rec`).
    assert (
        "from ..services.llm_telemetry import record_llm_call" in src
        or "from ext.services.llm_telemetry import record_llm_call" in src
    ), (
        "rag_stream.py must import record_llm_call from llm_telemetry "
        "so chat telemetry uses the canonical helper "
        "(parity with contextualizer/hyde/rewriter)"
    )


def test_rag_stream_uses_recorder_in_streaming_handler():
    src = _read_rag_stream_source()
    # The recorder must be entered as an async context manager — that's
    # the only way ``record_llm_call`` emits its histograms in finally.
    assert "async with _rec(" in src or "async with record_llm_call(" in src, (
        "rag_stream.py must wrap the SSE streaming loop in "
        "record_llm_call(...) so TTFT is observed via the canonical "
        "context manager"
    )
    # And the recorder's first-token hook must be threaded through —
    # without this the TTFT histogram never gets an observation.
    assert "set_first_token_at" in src, (
        "rag_stream.py must call recorder.set_first_token_at(...) on "
        "the first SSE event so the TTFT histogram is populated"
    )
    # And the ``stage="chat"`` label must be set so dashboards can
    # distinguish chat from contextualizer / hyde / rewriter.
    assert 'stage="chat"' in src, (
        "rag_stream.py must label the recorder with stage=\"chat\" "
        "so per-stage breakdowns work"
    )
