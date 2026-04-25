# OrgChat Observability — Design Spec

**Date:** 2026-04-21
**Status:** Approved, executing
**Owner:** Jatin

## 1. Goal

Provide complete real-time operational visibility into the OrgChat RAG stack:

1. **System overview** — host CPU/RAM/disk/net, GPU util + VRAM per-process, per-container resource usage.
2. **Application overview** — per-request waterfall for the full pipeline (query → embed → retrieve → rerank → MMR → expand → budget → LLM → stream), per-KB health, ingest/upload pipeline, LLM token accounting, RBAC denials. Jaeger-UI-style span tree per request.

## 2. Hard Constraints

- **No disruption of any currently running container** — orgchat services (`orgchat-*`) stay up; FRAMS stack (`frams-*`) is off-limits entirely.
- Air-gapped — no external data egress.
- Single-box resource discipline — observability stack capped so it cannot starve RAG inference.

## 3. Deployment Model

- Entirely separate Docker Compose project: `/home/vogic/LocalRAG/observability/docker-compose.yml`.
- Dedicated network `orgchat-obs-net`; orgchat containers are **live-attached** to it via `docker network connect` (no restart).
- All UIs bound to `127.0.0.1` only (SSH tunnel for remote access).
- Instrumentation code gated by `OBS_ENABLED` env var (default `false`) — merged dormant, activated only during an explicit maintenance window.

## 4. Port Plan (conflict-free with FRAMS)

| Component | Host port | Notes |
|---|---|---|
| Prometheus | 9091 | 9090 taken by frams-prometheus |
| Grafana | 3002 | 3001 taken by frams-grafana |
| Loki | 3101 | 3100 taken by frams-loki |
| Jaeger UI | 16687 | |
| OTel Collector OTLP gRPC | 4319 | 4317 taken internally |
| OTel Collector OTLP HTTP | 4320 | 4318 taken internally |
| node-exporter | 9101 | own instance to avoid scraping frams |
| cAdvisor | 8081 | |
| DCGM | 9401 | |
| postgres-exporter | 9187 | new |
| redis-exporter | 9122 | own instance |

## 5. Resource Caps & Retention

| Service | mem_limit | cpus | Retention |
|---|---|---|---|
| Prometheus | 1 GB | 1.0 | 7d metrics |
| Loki | 1 GB | 0.5 | 3d logs |
| Jaeger (Badger) | 1 GB | 1.0 | 3d traces |
| Grafana | 512 MB | 0.5 | — |
| OTel Collector | 256 MB | 0.5 | — |
| exporters (each) | 128 MB | 0.25 | — |

Disk budget: ~25 GB under `volumes/observability/`.

## 6. Stack Components

**Collectors / Backends (all new):**
- Prometheus (metrics TSDB)
- Loki + Promtail (logs; Promtail tails Docker JSON log driver via mounted socket, read-only)
- Jaeger all-in-one with Badger storage (traces)
- OpenTelemetry Collector — single OTLP ingress that fans out to Jaeger, Loki, Prom (via remote-write)

**Exporters (all new, per-orgchat):**
- node-exporter (host)
- cAdvisor (containers)
- nvidia-dcgm-exporter (GPU — requires `--gpus all`, uses NVIDIA runtime)
- postgres-exporter (pg_stat_*)
- redis-exporter

**Scrape targets (existing /metrics endpoints):**
- `orgchat-open-webui:8080/metrics` (extension's FastAPI)
- `orgchat-vllm-chat:8000/metrics` (vLLM native)
- `orgchat-tei:80/metrics` (TEI native)
- `orgchat-qdrant:6333/metrics` (Qdrant native)
- `orgchat-model-manager:8080/metrics`
- celery worker `/metrics` (once instrumented)

## 7. Instrumentation Scope (C — Full)

**Trace root spans per user interaction:**
- Chat message — full waterfall through `chat_rag_bridge`
- Document upload — through extract/chunk/embed/upsert
- KB CRUD, auth, vision, whisper, TTS

**Auto-instrumented via OTel:** FastAPI, SQLAlchemy, Redis, httpx, requests, psycopg2, Celery (send + consume with trace-id propagation through task headers).

**Manual spans in:**
- `ext/routers/rag.py`, `rag_stream.py`, `upload.py`, `kb_admin.py`, `kb_retrieval.py`
- `ext/services/chat_rag_bridge.py` — `rbac.check`, `embed.query`, `retrieve.parallel` (child per KB), `rerank.cross_encoder`, `mmr.dedupe`, `context.expand`, `budget.truncate`, `prompt.inject`, `llm.call`, `stream.token_first`, `stream.token_last`
- `ext/services/retriever.py`, `reranker.py`, `cross_encoder_reranker.py`, `mmr.py`, `context_expand.py`, `budget.py`, `hyde.py`, `query_rewriter.py`, `spotlight.py`, `raptor.py`
- `ext/services/extractor.py`, `chunker.py`, `embedder.py`, `ingest.py`, `vector_store.py`
- `ext/workers/ingest_worker.py`
- `ext/services/auth.py`, `jwt_verifier.py`, `rbac.py`, `vision.py`
- `whisper_service/`, `tts_service/`

**New Prometheus metrics:**
- `rag_tokens_prompt_total{model,kb}`, `rag_tokens_completion_total{model}`
- `rag_llm_ttft_seconds{model}`, `rag_llm_tpot_seconds{model}`
- `rag_upload_bytes_total{kb}`, `rag_ingest_duration_seconds{stage}`
- `rag_qdrant_search_latency_seconds{collection}`, `rag_qdrant_upsert_latency_seconds`
- `rag_rbac_denied_total{route}`, `rag_active_sessions`

**Logs:** Migrate FastAPI + Celery loggers to structured JSON (via `python-json-logger`), injecting `trace_id`, `span_id`, `user_id`, `request_id`. No transport change — Promtail tails Docker JSON driver.

## 8. Grafana Dashboards (7)

All provisioned into folder `OrgChat`:

1. **System Overview** — node-exporter + DCGM + cAdvisor filtered to `orgchat-*`
2. **Pipeline E2E** — rag_stage_latency_seconds p50/p95/p99 stacked, request rate, error rate
3. **KB Health** — per-KB QPS, hit rate, avg chunks, avg score, chunk/doc counts, ingest lag
4. **Ingest/Upload** — upload rate, extraction time by format, chunking throughput, Qdrant upsert latency, Celery queue depth
5. **LLM** — TTFT, TPOT, tokens/sec, prompt/completion counts, vLLM queue depth, model-manager load/unload events
6. **Errors & RBAC** — 4xx/5xx by route, RBAC denials, rerank-cache hit rate, ingest failures
7. **Traces Landing** — embedded Jaeger frame + exemplar links

Histogram panels emit OTel exemplars → click-through to Jaeger traces.

## 9. Bring-Up Order

1. Deploy exporters + Prometheus → verify scrape targets green
2. Deploy Loki + Promtail → verify orgchat logs flow
3. Deploy Jaeger + OTel Collector
4. Deploy Grafana, provision datasources + dashboards
5. Merge instrumentation PR with `OBS_ENABLED=false` (code dormant)
6. During explicit maintenance window: live-attach `orgchat-obs-net` to orgchat containers, flip `OBS_ENABLED=true`, restart ONLY extension process

## 10. Rollback

- `docker compose -f observability/docker-compose.yml down` — removes entire obs stack
- `docker network disconnect orgchat-obs-net <container>` — detaches without restart
- `OBS_ENABLED=false` — disables instrumentation even if stack is up
- Existing `/metrics` endpoint continues to function regardless

## 11. Non-Goals

- Alertmanager / paging (future)
- SSO integration (Grafana uses local admin auth)
- Long-term metrics archival (Thanos/Mimir) — not needed at current scale
- Editing any FRAMS container or config

## 12. Success Criteria

- Grafana `OrgChat` folder shows all 7 dashboards with live data within 5 min of a real user chat
- A single chat message produces a Jaeger trace with ≥12 spans covering the full pipeline
- GPU utilization, VRAM per-process, and container CPU/RAM visible per orgchat container
- Zero restart of any orgchat or FRAMS container during bring-up
- `OBS_ENABLED=false` keeps all instrumentation dormant with no measurable latency impact
