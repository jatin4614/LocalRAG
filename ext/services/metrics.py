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

        def dec(self, *_args, **_kwargs) -> None:
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

# -----------------------------------------------------------------------
# OTel-companion metrics (also scraped by Prometheus). Declared here so
# both the chat/retrieval code path and the ingest code path can import
# the same exporter instances (no duplicate registrations).
# -----------------------------------------------------------------------

# Token accounting. Prompt tokens carry a ``kb`` label (comma-joined
# kb_ids used for the request, "none" if none) so operators can
# attribute spend across KB selections; completion tokens are per-model.
tokens_prompt_total = Counter(
    f"{_NS}_tokens_prompt_total",
    "Prompt tokens consumed",
    labelnames=("model", "kb"),
)

tokens_completion_total = Counter(
    f"{_NS}_tokens_completion_total",
    "Completion tokens generated",
    labelnames=("model",),
)

# LLM streaming timing. TTFT: wall time between send and first streamed
# token; TPOT: per-token time across the streamed tail.
llm_ttft_seconds = Histogram(
    f"{_NS}_llm_ttft_seconds",
    "Time to first token",
    labelnames=("model",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

llm_tpot_seconds = Histogram(
    f"{_NS}_llm_tpot_seconds",
    "Time per output token",
    labelnames=("model",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# RBAC denials (403). Label by route so dashboards can attribute which
# endpoint is rejecting access (e.g. /api/rag/retrieve vs KB admin).
rbac_denied_total = Counter(
    f"{_NS}_rbac_denied_total",
    "RBAC access denials",
    labelnames=("route",),
)

# Active sessions gauge — incremented on chat-request entry, decremented
# on exit (try/finally in ``chat_rag_bridge``). Useful for capacity
# planning and spotting leaks.
active_sessions = Gauge(
    f"{_NS}_active_sessions",
    "Current active chat sessions",
)

# Qdrant call latencies. Shared between chat/retrieval and ingest paths.
qdrant_search_latency_seconds = Histogram(
    f"{_NS}_qdrant_search_latency_seconds",
    "Qdrant search latency",
    labelnames=("collection",),
)

qdrant_upsert_latency_seconds = Histogram(
    f"{_NS}_qdrant_upsert_latency_seconds",
    "Qdrant upsert latency",
)

# -----------------------------------------------------------------------
# Ingest-path metrics (appended). Owned by the ingest/upload code path.
# -----------------------------------------------------------------------
upload_bytes_total = Counter(
    f"{_NS}_upload_bytes_total",
    "Bytes uploaded for ingest",
    labelnames=("kb",),
)

ingest_duration_seconds = Histogram(
    f"{_NS}_ingest_duration_seconds",
    "Per-stage ingest duration",
    labelnames=("stage",),
    buckets=(0.05, 0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 300),
)

ingest_failures_total = Counter(
    f"{_NS}_ingest_failures_total",
    "Ingest failures by stage",
    labelnames=("stage", "reason"),
)

ingest_queue_depth = Gauge(
    f"{_NS}_ingest_queue_depth",
    "Celery ingest queue depth (observed)",
)

ingest_document_bytes = Histogram(
    f"{_NS}_ingest_document_bytes",
    "Size of documents ingested",
    labelnames=("kb", "format"),
    buckets=(1024, 10 * 1024, 100 * 1024, 1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024),
)

chunk_count = Histogram(
    f"{_NS}_ingest_chunks_per_doc",
    "Chunks produced per document",
    labelnames=("kb",),
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

# -----------------------------------------------------------------------
# Phase 1.1 — tokenizer fallback counter.
#
# Incremented when ``ext.services.budget`` falls back to cl100k from a
# non-cl100k alias. Should be 0 in steady state after the startup
# preflight passes — any non-zero value means either (a) the preflight
# was skipped, or (b) the HF cache went away after startup. Either way
# the alert should fire because ~10-15%% token-budget drift evicts
# relevant chunks silently.
#
# from_alias label: the alias the operator asked for (e.g. "gemma-4")
# to label: "cl100k" (runtime fallback) | "cl100k_forced_crash" (preflight raised)
# -----------------------------------------------------------------------
tokenizer_fallback_total = Counter(
    f"{_NS}_tokenizer_fallback_total",
    "Times the budget tokenizer fell back to cl100k from another alias. "
    "Should be 0 in steady state after preflight passes at startup.",
    labelnames=("from_alias", "to"),
)

# -----------------------------------------------------------------------
# Phase 2.1 — Spotlighting wrap counter.
#
# Incremented once per call to ``ext.services.spotlight.wrap_chunks`` (or
# ``wrap_context``) that actually produces a wrapped payload. Lets
# operators verify the flag is on in production (a flat zero line means
# Spotlight isn't running on the hot path) and trend per-request volume.
#
# No labels: the wrap call site is intentionally minimal — KB-level
# attribution can be derived from ``rag_retrieval_hits_total`` joined on
# request timestamp. Adding labels here would explode cardinality without
# materially helping debugging.
# -----------------------------------------------------------------------
spotlight_wrapped_total = Counter(
    f"{_NS}_spotlight_wrapped_total",
    "Number of retrieved chunks (or context blobs) wrapped with Spotlight "
    "untrusted-content tags. Increments once per wrap call when Spotlight "
    "is enabled.",
)

# -----------------------------------------------------------------------
# Phase 4 observability: KB health drift + scheduled eval gauges.
#
# ``rag_kb_drift_pct`` — per-KB percentage divergence between the
# expected chunk count (sum of ``kb_documents.chunk_count`` for live rows)
# and the observed Qdrant point count. Emitted by the
# ``GET /api/kb/{kb_id}/health`` handler as a side-effect, NOT a
# background task — keeps the metric lazy and avoids threading surprises.
#
# ``rag_eval_*`` — top-line scores from the weekly scheduled eval task
# (``ext.workers.scheduled_eval``). Last-value-wins gauges; each Monday
# (or whenever the beat fires) they're overwritten with the fresh run.
# -----------------------------------------------------------------------
rag_kb_drift_pct = Gauge(
    f"{_NS}_kb_drift_pct",
    "Percentage difference between expected chunks and Qdrant points for a KB",
    labelnames=("kb_id",),
)

rag_eval_chunk_recall = Gauge(
    f"{_NS}_eval_chunk_recall",
    "chunk_recall@10 from the most recent scheduled eval run",
)

rag_eval_faithfulness = Gauge(
    f"{_NS}_eval_faithfulness",
    "Faithfulness score from the most recent scheduled eval run",
)

rag_eval_p95_latency = Gauge(
    f"{_NS}_eval_p95_latency_ms",
    "p95 retrieval latency (ms) from the most recent scheduled eval run",
)


# -----------------------------------------------------------------------
# Plan B Phase 4.7 — Query Understanding LLM metrics
# -----------------------------------------------------------------------
rag_qu_invocations = Counter(
    f"{_NS}_qu_invocations_total",
    "QU classifier invocations by source (regex / llm / cached)",
    labelnames=("source",),
)

rag_qu_escalations = Counter(
    f"{_NS}_qu_escalations_total",
    "QU escalations from regex to LLM by predicate reason",
    labelnames=("reason",),
)

rag_qu_latency = Histogram(
    f"{_NS}_qu_latency_seconds",
    "QU LLM call latency",
    buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0),
)

rag_qu_schema_violations = Counter(
    f"{_NS}_qu_schema_violations_total",
    "QU LLM responses that failed schema validation",
)

rag_qu_cache_hits = Counter(
    f"{_NS}_qu_cache_hits_total",
    "QU cache hits",
)

rag_qu_cache_misses = Counter(
    f"{_NS}_qu_cache_misses_total",
    "QU cache misses",
)

rag_qu_cache_hit_ratio = Gauge(
    f"{_NS}_qu_cache_hit_ratio",
    "QU cache hit ratio (derived; recomputed every 30s by background task)",
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


# -----------------------------------------------------------------------
# Plan B Phase 5.9 — per-shard metrics for temporally-sharded collections.
#
# The four metrics let operators answer:
#   * how big is each shard?           -> rag_shard_point_count
#   * is search fast on each shard?    -> rag_shard_search_latency_seconds
#   * is upsert fast on each shard?    -> rag_shard_upsert_latency_seconds
#   * which tier is each shard in?     -> rag_shard_tier (label-as-state)
#
# ``rag_shard_tier`` uses the label-as-state pattern: at any time exactly
# one of {hot, warm, cold} is set to 1 for a (collection, shard_key)
# pair, the others are 0. ``set_shard_tier`` enforces that invariant.
# This makes Prometheus ``rate(rag_shard_tier[1d])`` a useful flap
# detector (Phase 5.10 alert).
# -----------------------------------------------------------------------
RAG_SHARD_POINT_COUNT = Gauge(
    f"{_NS}_shard_point_count",
    "Number of points per shard, refreshed by tier cron",
    labelnames=("collection", "shard_key"),
)

RAG_SHARD_SEARCH_LATENCY = Histogram(
    f"{_NS}_shard_search_latency_seconds",
    "Per-shard search latency (only emitted for shard-filtered searches)",
    labelnames=("collection", "shard_key"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RAG_SHARD_UPSERT_LATENCY = Histogram(
    f"{_NS}_shard_upsert_latency_seconds",
    "Per-shard upsert latency",
    labelnames=("collection", "shard_key"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RAG_SHARD_TIER = Gauge(
    f"{_NS}_shard_tier",
    "Current tier for each shard (1=current; 0=not). "
    "Exactly one tier label is 1 per (collection, shard_key) at any time.",
    labelnames=("collection", "shard_key", "tier"),
)


def set_shard_tier(*, collection: str, shard_key: str, tier: str) -> None:
    """Update the tier gauge: set requested tier to 1, others to 0.

    Fail-open — any prometheus_client error is swallowed so a metrics
    misconfiguration cannot break the cron path.
    """
    for t in ("hot", "warm", "cold"):
        try:
            RAG_SHARD_TIER.labels(
                collection=collection, shard_key=shard_key, tier=t,
            ).set(1.0 if t == tier else 0.0)
        except Exception:
            pass


def prom_available() -> bool:
    """Return True if prometheus_client was importable at module load."""
    return _PROM_AVAILABLE


# ---------------------------------------------------------------------------
# Plan B Phase 6.7 — image caption ingest skip counter.
#
# Incremented when ``ext.services.ingest.extract_images_as_chunks``
# captions an image but the vision service raises (most often because
# vllm-vision is on-demand and not yet warm). Operators monitor this
# alongside ``rag_image_skip_total`` in Grafana to spot regressions in
# image recall.
# ---------------------------------------------------------------------------
RAG_IMAGE_SKIP = Counter(
    f"{_NS}_image_skip_total",
    "Images that could not be captioned (e.g. vision unreachable).",
)
