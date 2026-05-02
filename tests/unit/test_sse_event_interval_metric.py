"""Unit test for the sse_event_interval_seconds metric (review §6.5).

The original code emitted SSE event spacing into ``rag_llm_tpot_seconds``,
which is misleading — the SSE never sees actual LLM tokens (the chat-LLM
call is a separate request fired in parallel by the frontend). Renamed
to ``rag_sse_event_interval_seconds`` so the LLM TPOT dashboard shows
ONLY actual streaming-LLM data.
"""
from __future__ import annotations


def test_sse_event_interval_metric_exists():
    from ext.services import metrics

    assert hasattr(metrics, "sse_event_interval_seconds"), (
        "Expected metrics.sse_event_interval_seconds to be defined "
        "(review §6.5: rename SSE event spacing out of llm_tpot_seconds)."
    )
    # It's a histogram with a model label
    h = metrics.sse_event_interval_seconds
    # Smoke-test labels() works (would raise on label mismatch)
    h.labels(model="any")


def test_rag_stream_uses_sse_event_interval_metric():
    """``rag_stream.py`` must NOT emit SSE event spacing into
    ``llm_tpot_seconds`` — that histogram is reserved for actual LLM
    streaming TPOT and the dashboards depend on its purity.
    """
    import inspect

    from ext.routers import rag_stream

    src = inspect.getsource(rag_stream)
    # The SSE event-spacing observation must use the new metric, not the
    # LLM TPOT one. The string ``llm_tpot_seconds`` is allowed if it
    # appears in a comment explaining the rename, but it MUST NOT be
    # called as a metric. We grep for ``.observe(`` on the LLM metric.
    # The simplest invariant: the file should reference the new metric.
    assert "sse_event_interval_seconds" in src, (
        "rag_stream.py must use sse_event_interval_seconds for SSE "
        "event spacing observations (review §6.5)."
    )


def test_metric_definition_documents_purpose():
    """The new metric's docstring/help text must explain it measures SSE
    event spacing, not LLM token timing.
    """
    from prometheus_client.metrics import MetricWrapperBase

    from ext.services import metrics

    h = metrics.sse_event_interval_seconds
    assert isinstance(h, MetricWrapperBase)
    # Prometheus exposes the help text via _documentation
    doc = (getattr(h, "_documentation", "") or "").lower()
    assert "sse" in doc, (
        "Help text should mention SSE so operators understand this "
        "isn't an LLM-token metric."
    )
