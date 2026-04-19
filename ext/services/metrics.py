"""Prometheus metrics for the RAG pipeline.

Fail-open philosophy: every exported metric accessor is wrapped so that
``prometheus_client`` being absent at runtime does not break retrieval.
When the optional dependency is installed (it is listed in
``[project].dependencies`` in ``pyproject.toml``) all metrics register
themselves on the default registry and a FastAPI app can expose them
via ``prometheus_client.make_asgi_app`` mounted at ``/metrics``.

Metric naming follows Prometheus conventions: ``rag_<name>_<unit>``
for histograms/counters, snake_case labels.

Exported names:
    rag_stage_latency_seconds  — Histogram(stage)
    rag_retrieval_hits_total   — Counter(kb, path)
    rag_rerank_cache_total     — Counter(outcome)
    rag_flag_enabled           — Gauge(flag)
    rag_ingest_chunks_total    — Counter(collection, path)

All metrics are best-effort. Callers should not rely on side-effects —
if you need an "always incremented" counter, do not use this module.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

# Sentinel: the set of exported names we want to guarantee even when
# prometheus_client is missing. Each is replaced by a no-op object so
# callers can do ``metric.labels(...).inc()`` without guarding.

_NS = "rag"

try:  # pragma: no cover - import-guard branch
    from prometheus_client import Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when dep missing
    _PROM_AVAILABLE = False

    class _NoopMetric:
        """Minimal stand-in for Counter/Gauge/Histogram when dep is absent."""

        def labels(self, *_args, **_kwargs) -> "_NoopMetric":
            return self

        def inc(self, *_args, **_kwargs) -> None:
            return None

        def observe(self, *_args, **_kwargs) -> None:
            return None

        def set(self, *_args, **_kwargs) -> None:
            return None

    def Counter(*_a, **_kw):  # type: ignore[no-redef]
        return _NoopMetric()

    def Gauge(*_a, **_kw):  # type: ignore[no-redef]
        return _NoopMetric()

    def Histogram(*_a, **_kw):  # type: ignore[no-redef]
        return _NoopMetric()


# Per-stage latency histograms (seconds).
# stage label: retrieve | rerank | mmr | expand | budget | total
stage_latency = Histogram(
    f"{_NS}_stage_latency_seconds",
    "Time spent in a RAG pipeline stage",
    labelnames=("stage",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Retrieval hit counters.
# kb label: <kb_id> | "chat" | "eval" | "unknown"
# path label: "dense" | "hybrid"
retrieval_hits_total = Counter(
    f"{_NS}_retrieval_hits_total",
    "Count of retrieval hits by KB and search path",
    labelnames=("kb", "path"),
)

# Reranker score-cache outcomes.
# outcome label: "hit" | "miss"
rerank_cache_total = Counter(
    f"{_NS}_rerank_cache_total",
    "Reranker score cache hits/misses",
    labelnames=("outcome",),
)

# Flag state snapshot — set on each retrieval so operators can see the
# current process's effective flag configuration.
# flag label: e.g. "hybrid", "rerank", "mmr", "context_expand", "spotlight"
flag_state = Gauge(
    f"{_NS}_flag_enabled",
    "Whether a RAG_* feature flag is currently enabled (per process)",
    labelnames=("flag",),
)

# Ingest metrics.
# collection label: qdrant collection name (e.g. "kb_1", "chat_42")
# path label: "sync" | "celery"
ingest_chunks_total = Counter(
    f"{_NS}_ingest_chunks_total",
    "Chunks upserted into Qdrant",
    labelnames=("collection", "path"),
)


@contextmanager
def time_stage(stage: str) -> Iterator[None]:
    """Wrap a block to record its duration into ``stage_latency{stage="..."}``.

    Fail-open: any exception inside the observe() path is swallowed so
    the surrounding retrieval logic is never disturbed by metrics. The
    wrapped code still raises normally via ``finally``.
    """
    t0 = perf_counter()
    try:
        yield
    finally:
        try:
            stage_latency.labels(stage=stage).observe(perf_counter() - t0)
        except Exception:
            pass


def prom_available() -> bool:
    """Return True if prometheus_client was importable at module load."""
    return _PROM_AVAILABLE
