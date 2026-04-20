# RAG Pipeline Metrics (Prometheus)

P2.5 wires Prometheus metrics into the RAG pipeline so operators can
see per-stage latency, per-KB hit counts, cache hit/miss ratios, and
current flag state.

## Endpoint

- **URL**: `/metrics` on the ext FastAPI app (mounted by
  `ext.app._mount_metrics` inside `build_app()`).
- **Format**: Prometheus text exposition (default `make_asgi_app`).
- **Scrape**: any Prometheus server can point at
  `http://<ext-host>:<port>/metrics` with a 15s scrape interval.

Fail-open: if `prometheus_client` is missing at runtime, the mount is
skipped and a warning logged. Metric calls from the pipeline collapse
to no-ops (see `ext/services/metrics.py`).

## Metric Reference

| Name | Type | Labels | Meaning |
|---|---|---|---|
| `rag_stage_latency_seconds` | Histogram | `stage` | Time spent in a pipeline stage. `stage` ∈ `retrieve`, `rerank`, `mmr`, `expand`, `budget`, `total`. |
| `rag_retrieval_hits_total` | Counter | `kb`, `path` | Raw hits returned by `retrieve()` before reranking. `kb` is the KB id (as string), `"chat"` for private docs, or `"unknown"`. `path` is `"hybrid"` or `"dense"`. |
| `rag_rerank_cache_total` | Counter | `outcome` | Cross-encoder score cache probes. `outcome` is `"hit"` or `"miss"`. |
| `rag_flag_enabled` | Gauge | `flag` | Whether a RAG_* feature flag is on (1) or off (0). `flag` ∈ `hybrid`, `rerank`, `mmr`, `context_expand`, `spotlight`. Re-set on every retrieval. |
| `rag_ingest_chunks_total` | Counter | `collection`, `path` | Chunks upserted into Qdrant. `collection` is the Qdrant collection name (e.g. `kb_1`, `chat_42`). `path` is `"sync"` or `"celery"`. |

## Suggested Grafana Queries (PromQL)

- **p95 total RAG latency** (seconds):
  ```
  histogram_quantile(0.95, sum by (le) (rate(rag_stage_latency_seconds_bucket{stage="total"}[5m])))
  ```
- **p95 retrieval vs rerank vs MMR vs expand** (split):
  ```
  histogram_quantile(0.95, sum by (le, stage) (rate(rag_stage_latency_seconds_bucket[5m])))
  ```
- **Retrieval hits per KB per minute** (top 10):
  ```
  topk(10, sum by (kb) (rate(rag_retrieval_hits_total[1m])))
  ```
- **Rerank cache hit ratio** (higher = more cache efficiency):
  ```
  sum(rate(rag_rerank_cache_total{outcome="hit"}[5m]))
    / sum(rate(rag_rerank_cache_total[5m]))
  ```
- **Dense vs hybrid split**:
  ```
  sum by (path) (rate(rag_retrieval_hits_total[5m]))
  ```
- **Current flag state** (instantaneous):
  ```
  rag_flag_enabled
  ```
- **Chunks ingested by collection**:
  ```
  sum by (collection) (rate(rag_ingest_chunks_total[5m]))
  ```

## Troubleshooting

- **Metric missing from `/metrics`?** The counter/histogram is lazy —
  it only appears once it has been observed at least once. A
  flag-gated stage (MMR, expand) won't show up until `RAG_MMR=1` or
  `RAG_CONTEXT_EXPAND=1` runs a request. The `total` stage is always
  observed when `retrieve_kb_sources()` runs.
- **`/metrics` returns 404?** `prometheus_client` isn't installed in
  this env. Check logs for `"/metrics disabled"`. Install with
  `pip install 'prometheus-client>=0.20'` (it's in
  `[project].dependencies`, so a normal `pip install .` installs it).
- **Cache ratio is 0?** `RAG_RERANK` must be on, and Redis must be
  reachable at `RAG_REDIS_URL`. See `ext/services/rerank_cache.py`
  for connection settings. A low hit ratio immediately after rollout
  is normal until queries recur.
- **Per-KB hit counter shows `kb="unknown"`?** The hit's payload is
  missing `kb_id`. Likely a legacy doc or a chat private doc without
  kb_id. Chat-private hits should show `kb="chat"`.

## Implementation Notes

All metric calls in the hot path are wrapped in `try: ... except: pass`
— metrics must NEVER break retrieval. The `time_stage` context
manager swallows observation errors but always propagates exceptions
from the wrapped block. See the fail-open tests in
`tests/unit/test_metrics.py::test_time_stage_preserves_exceptions`.
