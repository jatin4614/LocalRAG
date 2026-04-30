# LocalRAG — Org Chat Assistant with Hierarchical KB RAG

**Project type:** Self-hosted, air-gapped, multi-user ChatGPT-style assistant for organizations
**Status:** Plan A + Plan B both merged to `main` (2026-04-26 commit `d1aa862` no-ff). Production live; soak fixes ongoing.
**Repo strategy:** Thin fork of Open WebUI; org-specific code lives under `ext/` and `compose/`, upstream tracked via the `upstream/` git submodule and small patches in `patches/`.
**Hardware (deploy host):** Single box with **GPU 0: RTX 6000 Ada 48 GB** (vllm-chat) + **GPU 1: RTX PRO 4000 Blackwell 24 GB** (TEI, vllm-qu, reranker, fastembed). 125 GB RAM, 2.7 TB disk, Linux 6.8, CUDA 13.0.
**Deploy mode:** `docker-compose` on a single host. Air-gapped after first model download (`HF_HUB_OFFLINE=1` everywhere).

---

## 1. What this project is

A self-hosted ChatGPT-like assistant for orgs (20–200 users) with:
- Local LLM inference (no cloud API calls, no telemetry leaving the host)
- **Hierarchical Knowledge Bases** (KB → optional subtags) with per-session selection
- **Strict per-user data isolation** — three-layer enforcement (DB constraints → API RBAC → Qdrant payload filter)
- **Shared RBAC-gated KBs** assigned to users or groups
- **Document, audio, and image inputs** (Gemma-4-31B is multimodal; Whisper for STT; Piper for TTS)
- **Hybrid + ColBERT + cross-encoder rerank retrieval** with intent-conditional MMR / context expansion
- **Temporal sharding** (Qdrant 1.16 cluster mode) and **temporal RAPTOR summary tree** for multi-year inter-related corpora
- **Async ingest** via Celery, **OCR fallback**, **structure-aware chunking** (Plan B Phase 6)
- **Query Understanding LLM** (Qwen3-4B AWQ on GPU 1) escalating ambiguous queries beyond regex (Plan B Phase 4)

**Core problem solved:** Plain RAG retrieves indiscriminately across all docs. This system lets users explicitly select which KBs to query per chat, scopes retrieval there + to the chat's private upload namespace, and enforces RBAC at every layer.

---

## 2. Quick reference

### One-shot commands

```bash
# Bring up the full stack (postgres, redis, qdrant, vllm-chat, vllm-qu, tei,
# whisper, tts, model-manager, open-webui, celery-worker, celery-beat, caddy)
make up

# Smoke test
make smoke

# Apply Postgres migrations (idempotent, asyncpg)
python scripts/apply_migrations.py

# Run pytest suite (use the venv binary explicitly; host pytest may not be on PATH)
.venv/bin/pytest tests/ -q

# Run eval gate (golden_starter + golden_evolution)
make eval-baseline
make eval-evolution
```

### Where to find what

| Concern | Path |
|---|---|
| RAG orchestrator (entry from chat middleware) | `ext/services/chat_rag_bridge.py:568` `retrieve_kb_sources` → `_run_pipeline:754` |
| Ingest pipeline | `ext/services/ingest.py` `ingest_bytes` |
| Parallel retrieve fan-out | `ext/services/retriever.py:132` `retrieve` |
| Rerank | `ext/services/reranker.py` (heuristic) + `cross_encoder_reranker.py` (cross-encoder) |
| Vector store wrapper | `ext/services/vector_store.py` (httpx pool, sharding-aware upsert) |
| Embedder (TEI + sparse + colbert) | `ext/services/embedder.py`, `sparse_embedder.py` |
| KB admin endpoints | `ext/routers/kb_admin.py` |
| Upload (sync + async) | `ext/routers/upload.py` |
| Per-KB config | `ext/services/kb_config.py` (UNION/MAX merge) |
| Feature flags | `ext/services/flags.py` (env + per-request overlay) |
| Postgres migrations | `ext/db/migrations/001…012_*.sql` |
| SQLAlchemy models | `ext/db/models/{kb,chat_ext,compat}.py` |
| Qdrant schema constants | `ext/db/qdrant_schema.py` |
| Celery worker | `ext/workers/{celery_app,ingest_worker,scheduled_eval,blob_gc_task}.py` |
| Compose | `compose/docker-compose.yml` (+ `docker-compose.observability.yml` overlay) |
| Frontend patches | `patches/0001…0004` applied over `upstream/` submodule |
| Operator scripts | `scripts/*.py` (reshard, reingest, backfill, soak, eval) |

---

## 3. Architecture overview

### 3.1 Service inventory (`compose/docker-compose.yml`)

| Service | Image / build | GPU | Role |
|---|---|---|---|
| `postgres` | `postgres:15-alpine` | — | KB metadata, users, RBAC grants, chats. Auto-runs migrations from `/docker-entrypoint-initdb.d` on first init. |
| `redis` | `redis:7-alpine` | — | App cache (DB 0), Celery broker (DB 1), Celery results (DB 2), RBAC cache (DB 3), QU cache (DB 4). AOF on. |
| `qdrant` | `qdrant/qdrant:latest` | — | Vector DB. **Cluster mode enabled** since 2026-04-26 (Plan B Phase 5). Port 6333; `--uri http://qdrant:6335` for P2P consensus. Collections: `kb_{id}`, `kb_{id}_v4` (custom-sharded), `chat_private`. |
| `vllm-chat` | `vllm/vllm-openai:nightly` | **0** | Multimodal chat LLM. **Gemma-4-31B-it-AWQ** (`QuantTrio/gemma-4-31B-it-AWQ`). max-model-len 32768, fp8 KV cache, prefix caching, gemma4 reasoning + tool-call parsers, vision_tower fp16 patch mounted from `patches/vllm/gemma4_mm.py`. ~89% of GPU 0 VRAM. |
| `vllm-qu` | `vllm/vllm-openai:latest` | **1** | Query Understanding LLM. `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`, served as `qwen3-4b-qu`. max-model-len 8192, gpu-memory-util 0.45, prefix caching. ~10 GB VRAM. **Soft-fails**: bridge falls back to regex if unreachable. |
| `tei` | `ghcr.io/huggingface/text-embeddings-inference:120-1.9` | **1** | Dense embeddings via `BAAI/bge-m3` (1024-dim). Blackwell sm_120 build. Does NOT serve sparse — `/embed_sparse` returns 424. ~1.6 GB VRAM. |
| `open-webui` | `orgchat-open-webui:cu128-test` (built from `Dockerfile.openwebui.cu128`) | **1** | Frontend + chat backend + RAG host. Runs cross-encoder reranker (`BAAI/bge-reranker-v2-m3`, ~3 GB), fastembed BM25 + ColBERT (CUDA via fastembed-gpu). cu128 + torch ≥ 2.7 for sm_120 kernels. |
| `whisper` | local build (`whisper_service/`) | — | faster-whisper STT, currently CPU int8 (base.en). To move to GPU 1: rebuild on `nvidia/cuda:12.1-cudnn8-runtime` and flip `WHISPER_DEVICE=cuda`. |
| `tts` | local build (`tts_service/`) | — | Piper TTS (default voice `en_US-lessac-medium`). |
| `model-manager` | local build (`model_manager/`) | — | Polls vllm-chat / whisper, exposes `/healthz`. Vision was consolidated into vllm-chat 2026-04-20 — `VISION_URL` points at vllm-chat. |
| `celery-worker` | `compose/Dockerfile.celery` | **1** | Async ingest queue worker. fastembed-gpu for ColBERT + sparse on GPU 1. Concurrency 2. Queue `ingest`. **Note:** image is intentionally minimal — no torch / transformers / sentence-transformers (those live in open-webui). |
| `celery-beat` | `compose/Dockerfile.celery` | — | Cron scheduler. Fires daily 03:00 tier-storage cron over `RAG_TIER_COLLECTIONS` (default `kb_1_v4`). Schedule file persisted to `celery_beat_state` volume. **Single instance only.** |
| `caddy` | `caddy:2-alpine` | — | Reverse proxy + TLS. Ports 8880/8443 → open-webui. Caddyfile in `compose/caddy/`. |

**Network:** `orgchat-net` bridge. **Volumes:** `postgres_data`, `redis_data`, `qdrant_data`, `ingest_blobs` (open-webui producer ↔ celery-worker consumer), `celery_beat_state`. Bind mounts: `volumes/models` (HF cache, shared by vllm-chat / vllm-qu / tei), `volumes/uploads`, `volumes/hf-cache`, `volumes/certs`.

### 3.2 Observability stack

Separate compose at `observability/docker-compose.yml` (LGTM + Jaeger, bound to 127.0.0.1). Enabled via overlay `compose/docker-compose.observability.yml`:

- **prometheus** v3.5.0 (port 9091:9090) — 7d retention, remote-write-receiver on, scrapes `/metrics` on open-webui port 9464 (`PROM_METRICS_PORT`)
- **loki** 2.9.3 + **promtail** 3.2.0 — Docker log tailing
- **jaeger** all-in-one (Badger backend, 72h TTL) — UI on 16686
- **grafana** + **otelcol**
- **DCGM exporter** for GPU metrics; **node-exporter, redis-exporter, postgres-exporter, cadvisor** for system + DB metrics

Open WebUI gets `OBS_ENABLED=true`, `OTEL_EXPORTER_OTLP_ENDPOINT=http://orgchat-obs-otelcol:4317`, `OTEL_SERVICE_NAME=orgchat-ext`, attached to `orgchat-obs-net` external network.

### 3.3 Hardware reality (verified 2026-04-26)

| GPU | Model | VRAM | Tenants | Notes |
|---|---|---|---|---|
| 0 | RTX 6000 Ada | 48 GB | vllm-chat (Gemma-4-31B-AWQ ~36 GB) + external `frams-recognition-worker-*` (~7 GB) | ~89% used — **do not add load** until external worker moves off |
| 1 | RTX PRO 4000 Blackwell | 24 GB | tei (~1.6 GB) + vllm-qu (~10 GB) + reranker (~3 GB) + celery-worker fastembed (~2 GB) | ~17 GB resident, ~7 GB headroom |

Driver 580.126.09, CUDA 13.0. Sparse embeddings are client-side (fastembed 0.8.0 inside open-webui + celery-worker) — TEI does not serve sparse on this build.

---

## 4. RAG pipeline

### 4.1 Ingest pipeline

```
Upload                                 ext/routers/upload.py
  ├─ sync mode  (RAG_SYNC_INGEST=1)    → ingest_bytes inline
  └─ async mode (RAG_SYNC_INGEST=0)    → write blob to /var/ingest/{sha}, enqueue Celery
                                         (celery-worker picks up, runs ingest_bytes)
                                         → ext/workers/ingest_worker.py
ingest_bytes                           ext/services/ingest.py
  Extract                              ext/services/extractor.py
    │ PDF (pypdf + pdfplumber + pymupdf), DOCX (python-docx), XLSX (openpyxl),
    │ TXT, MD, HTML. Returns list[ExtractedBlock] with page / heading_path / sheet
  OCR fallback [RAG_OCR_ENABLED=1, per-KB ocr_policy]      ext/services/ocr.py
    │ Triggered when extracted text < RAG_OCR_TRIGGER_CHARS (50). Backends:
    │ tesseract (default, baked into image), AWS Textract (opt-in, env creds),
    │ Google Document AI (opt-in, env creds).
  Chunk                                ext/services/chunker.py / chunker_structured.py
    │ Default "window" strategy: 800 tokens, 100 overlap. Honors
    │ RAG_BUDGET_TOKENIZER so chunk sizes match the chat model's tokenizer
    │ (Gemma-4 vs cl100k differ ~10-15%). Structured strategy
    │ [RAG_STRUCTURED_CHUNKER=1] respects heading boundaries; per-KB
    │ rag_config.chunking_strategy="structured".  *Note: structured chunker
    │ exists but is not yet wired into ingest_bytes — chunking_strategy is
    │ a no-op until that lands (Phase 6 follow-up).*
  Contextualize [RAG_CONTEXTUALIZE_KBS=1, per-KB]    ext/services/contextualizer.py
    │ LLM call (concurrency 8, timeout 30s) prepends ~50-token context prefix
    │ per chunk. Cache-friendly prompt: doc-level system message stable across
    │ chunks, chunk-specific text is the last user message → vllm-chat prefix
    │ caching kicks in. Bakes "Document: X; Date: Y; KB: Z; Relationships: …"
    │ into chunk text.
  RAPTOR [RAG_RAPTOR=1]                ext/services/raptor.py
    │ Hierarchical KMeans → LLM-summarize cluster → recursive. Default 3 levels,
    │ min cluster 5, concurrency 4. Stamps level=1..N points alongside leaves.
    │ Plan B Phase 5 deprecated this in favor of temporal_raptor.py for
    │ time-aware corpora.
  Temporal RAPTOR (Plan B Phase 5)     ext/services/temporal_raptor.py
    │ L0 raw chunks → L1 per-month → L2 per-quarter (change-vs-prior prompt) →
    │ L3 per-year → L4 multi-year. Driver: scripts/build_temporal_tree.py.
    │ L1 capped at 30 sampled chunks per prompt to fit Gemma 32K ctx.
  Doc summary [RAG_DOC_SUMMARIES=1, default ON]   ext/services/doc_summarizer.py
    │ One level=doc point per document, 3-line synopsis. Mirrored to
    │ kb_documents.doc_summary column. Required for "global" intent path to
    │ return one chunk per doc on summarize-all queries. Timeout 90s.
  Embed                                ext/services/embedder.py
    │ Dense:    TEI bge-m3 (1024d), batches of 32
    │ Sparse:   fastembed Qdrant/bm25 (in-process, no TEI)
    │ ColBERT:  fastembed colbert-ir/colbertv2.0 (128d multi-vector, MaxSim)
  Upsert                               ext/services/vector_store.py:upsert / upsert_temporal
    │ Deterministic UUIDv5 point IDs (namespace 6ba7b810…, doc_id+chunk_index)
    │ Named vectors: "dense", "bm25" (sparse), "colbert" (multi-vector)
    │ Custom sharding [RAG_SHARDING_ENABLED=1]: shard_key derived from filename
    │   (date pattern) or body fallback; passed via shard_key_selector.
    │ kb_documents row: pending → queued → chunking → embedding → done | failed
    │ Idempotent: same (doc_id, chunk_index) → same point ID → blind overwrite is safe.
```

**Blob lifecycle (async path):** open-webui writes the upload to `/var/ingest/{sha256}` on the shared `ingest_blobs` volume; Celery worker reads + processes; blob GC (`ext/services/blob_gc.py`, `ext/workers/blob_gc_task.py`) sweeps after `RAG_BLOB_RETENTION_DAYS`. Failed tasks DLQ after 3 retries with exponential backoff.

**Migration ladder for `kb_documents.ingest_status`** (CHECK constraint, see migration 012):
`pending` → `queued` → `chunking` → `embedding` → `done` | `failed`

### 4.2 Retrieval pipeline

Entry: `ext/services/chat_rag_bridge.py:568 retrieve_kb_sources(kb_config, query, user_id, chat_id, history, progress_cb)`

```
Stage  Component                                           Notes / flag gating
-----  -------------------------------------------------   -----------------------------------
1      Active sessions gauge inc                           active_sessions metric
2      Query rewrite (multi-turn)    query_rewriter.py     RAG_DISABLE_REWRITE != "1" AND history.
                                                            Default disabled (RAG_DISABLE_REWRITE=1).
                                                            Uses REWRITE_MODEL or CHAT_MODEL.
3      Request context set           request_ctx.py        request_id (8-char uuid) + user_id
                                                            in ContextVars; flows to logs + spans.
4      RBAC check                    rbac.py:resolved_     Cache-first via redis://…/3,
                                     allowed_kb_ids        TTL=RAG_RBAC_CACHE_TTL_SECS (30s)
                                                            + pubsub channel `rbac:invalidate`.
                                                            DB miss is source of truth.
                                                            Fail-closed: on lookup error, skip KB.
5      Per-KB rag_config merge       kb_config.py          Strictest-wins UNION/MAX merge of each
                                                            selected KB's JSONB rag_config.
6      Intent classification         classify_intent       Sub-ms regex over metadata / global /
                                                            specific patterns. Plan B 4.6 added
                                                            async hybrid _classify_with_qu using
                                                            QU LLM, with Redis cache (DB 4).
                                                            B11/B12 fixes (commit 7a89d25)
                                                            tightened metadata vs specific_date.
7      Intent flag overlay           resolve_intent_flags  Maps intent → MMR/expand defaults
                                                            via _INTENT_FLAG_POLICY. Per-KB
                                                            rag_config wins. Mode controlled by
                                                            RAG_INTENT_OVERLAY_MODE
                                                            (intent | env, default intent).
8      Embedding (query)             embedder.py           TEI bge-m3 (1024d). HyDE optional
                                                            [RAG_HYDE=1, N expansions averaged].
                                                            SemCache check optional [RAG_SEMCACHE=1].
9      Parallel retrieve fan-out     retriever.py:retrieve asyncio.gather across each selected KB
                                                            + chat_private. Per-KB concurrency
                                                            via shared httpx pool (size 32).
                                                            Hybrid [RAG_HYBRID=1, default ON]:
                                                              dense + sparse RRF k=60.
                                                            Tri-fusion [RAG_COLBERT=1, default ON
                                                              for kb_1_v4]: + colbert head, RRF
                                                              all three.
                                                            Filters: level (chunk|doc), shard_key
                                                              [RAG_TEMPORAL_LEVELS=1], doc_ids
                                                              (specific_date intent).
                                                            Limits: 10/KB, 30 total
                                                              (global: 50/100, specific_date: 30/60).
10     Cross-KB merge                retriever.py:         RRF k=60 if rerank off; raw score sort if on.
                                     merge_kb_results
11     Rerank                        reranker.py /         Heuristic (default): max-normalize per
                                     cross_encoder_        KB + global sort. Cross-encoder
                                     reranker.py             [RAG_RERANK=1, default ON]: bge-
                                                            reranker-v2-m3 on GPU 1, batched.
                                                            Cache: rerank_cache (300s TTL).
                                                            Global intent skips rerank.
                                                            B6 fix: silent failures counted via
                                                            RAG_SILENT_FAILURE{stage}.
12     MMR diversify [RAG_MMR=1]     mmr.py                λ=RAG_MMR_LAMBDA (default 0.7).
                                                            Disabled for global intent.
13     Context expand                context_expand.py     ±N siblings via doc_id + chunk_index.
       [RAG_CONTEXT_EXPAND=1]                              Window = RAG_CONTEXT_EXPAND_WINDOW.
                                                            Disabled for global intent.
14     Time decay                    time_decay.py         score' = score · exp(-λ·Δt_days).
       [RAG_TIME_DECAY=1]                                  Half-life RAG_TIME_DECAY_LAMBDA_DAYS=90.
                                                            Intent-conditional in plan; not
                                                            currently default-on.
15     Token budget                  budget.py:truncate_   Truncate from low-rank end to fit
                                     to_max_tokens         max_tokens (default 5000). Tokenizer
                                                            via RAG_BUDGET_TOKENIZER (default
                                                            gemma-4 in deployed env).
16     Spotlight wrap                spotlight.py          Wrap each chunk in
       [RAG_SPOTLIGHT=1, default ON]                       <UNTRUSTED_RETRIEVED_CONTENT> tags;
                                                            defang nested closing tags.
                                                            Defense against indirect prompt-
                                                            injection in retrieved text.
17     Format upstream sources +     chat_rag_bridge.py    Group by source doc. Inject KB Catalog
       preambles                       (~790-1063)         preamble (authoritative for
                                                            metadata/global). Inject current
                                                            datetime preamble [RAG_INJECT_DATETIME=1,
                                                            default ON; RAG_TZ=UTC default].
18     SSE progress emit             rag_stream.py         If progress_cb set: stage events
                                                            (embed, retrieve, rerank, mmr, expand,
                                                            hits, done, error). 15s keepalive.
19     Active sessions gauge dec
```

**Per-KB `rag_config` keys** (JSONB on `knowledge_bases`, merged at request time):
- `rerank` (bool), `mmr` (bool), `context_expand` (bool), `context_expand_window` (int)
- `colbert` (bool), `hybrid` (bool), `spotlight` (bool), `hyde` (bool), `raptor` (bool)
- `chunking_strategy` (`"window"` | `"structured"`), `contextualize` (bool)
- `intent_overlay_mode` (`"intent"` | `"env"`)

### 4.3 Cross-cutting

- **Metrics** (`ext/services/metrics.py`): `rag_stage_latency_seconds{stage}`, `rag_retrieval_hits_total{kb,status}`, `rag_flag_enabled{flag}`, `active_sessions`, `upload_bytes_total`, `RAG_SILENT_FAILURE{stage}`, `llm_tokens_total{stage,model,direction}`, `llm_requests_total{stage,model,status}`, `llm_latency_seconds{stage,model}`, `rag_qu_disagree_total`, `rag_qu_latency_seconds`. Fail-open everywhere — counter import / inc must never break retrieval.
- **OTel spans** (`ext/services/obs.py`): `rag.rbac_check`, `embed.query`, `retrieve.parallel`, `retrieve` (per KB), `rerank.score`, `mmr_rerank`, `budget.truncate`. `OBS_ENABLED=false` → no-op.
- **Structured logs**: JSON to stderr; file handlers configurable. Shadow log JSONL at `RAG_QU_SHADOW_LOG_PATH=/var/log/orgchat/qu_shadow.jsonl`, best-effort install at startup.
- **Circuit breaker** (`circuit_breaker.py`): per-tenant fast-fail after `RAG_CB_FAIL_THRESHOLD=3` failures in `RAG_CB_WINDOW_SEC=300`s; cooldown `RAG_CB_COOLDOWN_SEC=30`s.
- **Retry policy** (`retry_policy.py`): tenacity exponential backoff + jitter for TEI / reranker / HyDE; `RAG_TENACITY_RETRY=1` default.

### 4.4 Isolation invariant (sacred — three layers)

Every chunk in Qdrant carries `kb_id` + `doc_id` (+ optional `subtag_id`) for shared KB content, OR `chat_id` + `owner_user_id` for private chat docs. Never both.

- **DB layer** (`kb_access` table): CHECK `(user_id IS NOT NULL) + (group_id IS NOT NULL) = 1` — exactly one. Cascade-delete on KB deletion. Soft-delete on KBs (`deleted_at IS NOT NULL` filter on every read).
- **API layer** (`rbac.py`): `get_allowed_kb_ids(session, user_id)` → direct grants + group grants. Admin role = all. Cached in Redis DB 3 with TTL 30s + pubsub invalidation on grant mutation. Cache miss → DB hit (DB is source of truth, cache is accelerator only).
- **Vector layer** (`vector_store.py`): every search injects `kb_id ∈ allowed_kb_ids` filter; private namespace queries inject `chat_id == session_chat AND owner_user_id == session_user`.

Six explicit isolation tests in `tests/integration/test_kb_isolation.py` are the mandatory CI gate.

---

## 5. Database schema

### 5.1 Postgres migrations (`ext/db/migrations/`)

| # | File | Purpose |
|---|---|---|
| 001 | `001_create_kb_schema.sql` | `knowledge_bases`, `kb_subtags`, `kb_documents`, `kb_access` + indexes; `chats.selected_kb_config` JSONB (later dropped in 007) |
| 002 | `002_soft_delete_kb.sql` | `knowledge_bases.deleted_at` + partial index |
| 003 | `003_add_chunk_count.sql` | `kb_documents.chunk_count` int |
| 004 | `004_add_pipeline_version.sql` | `kb_documents.pipeline_version` text — composite version stamped at ingest; enables re-index gating without full collection rebuild |
| 005 | `005_add_kb_document_blob_sha.sql` | `kb_documents.blob_sha` text + partial index — sha256 for blob GC dedup |
| 006 | `006_add_kb_rag_config.sql` | `knowledge_bases.rag_config` JSONB default `{}` + GIN index — per-KB retrieval overrides |
| 007 | `007_drop_orphan_selected_kb_config.sql` | Drop `chats.selected_kb_config` (canonical state moved to upstream `chats.meta.kb_config`) |
| 008 | `008_add_doc_summary.sql` | `kb_documents.doc_summary` text — Tier 1 per-doc synopsis (mirrors level=doc Qdrant point) |
| 009 | `009_rbac_pubsub_channel.sql` | Comment-only — documents `rbac:invalidate` Redis pubsub contract |
| 010 | `010_add_kb_chunking_strategy.sql` | Default `chunking_strategy: "window"` in `rag_config` JSONB (Plan B Phase 6.6) |
| 011 | `011_add_kb_ocr_policy.sql` | `knowledge_bases.ocr_policy` JSONB default tesseract config (Plan B Phase 6.3) |
| 012 | `012_add_kb_documents_queued_status.sql` | Add `'queued'` to `kb_documents.ingest_status` CHECK (Plan B Phase 6.2 async ingest) |

Runner: `scripts/apply_migrations.py` runs all `*.sql` sorted by name; idempotent (`IF NOT EXISTS` guards everywhere); async via asyncpg.

### 5.2 SQLAlchemy models (`ext/db/models/`)

**kb.py** — extension-owned:

| Model | Table | Key columns | Constraints |
|---|---|---|---|
| `KnowledgeBase` | `knowledge_bases` | id (PK), name (UNIQUE), admin_id (UUID), created_at, deleted_at, rag_config JSONB, ocr_policy JSONB | partial idx WHERE deleted_at IS NULL; GIN on rag_config |
| `KBSubtag` | `kb_subtags` | id (PK), kb_id (FK CASCADE), name | UNIQUE(kb_id, name); CHECK length(name) > 0 |
| `KBDocument` | `kb_documents` | id (PK), kb_id (FK), subtag_id (FK), filename, mime_type, bytes, ingest_status, error_message, uploaded_at, uploaded_by (UUID), deleted_at, chunk_count, pipeline_version, blob_sha, doc_summary | partial indexes WHERE deleted_at IS NULL; CHECK on ingest_status enum |
| `KBAccess` | `kb_access` | id (PK), kb_id (FK CASCADE), user_id OR group_id (XOR), access_type (`"read"` \| `"write"`), granted_at | CHECK exactly-one user XOR group |

**compat.py** — read-only mirrors of upstream Open WebUI tables:

`User` (users), `Group` (groups), `UserGroup` (user_groups composite PK), `Chat` (chats; canonical kb_config lives in `chat.meta.kb_config` JSON since migration 007).

**chat_ext.py** — `SelectedKBConfig` dataclass (frozen): `[{kb_id: int, subtag_ids: [int]}]` schema for in-memory passing.

### 5.3 Qdrant collection schema (`ext/db/qdrant_schema.py`, `ext/services/vector_store.py`)

**Collection naming:**
- Per-KB: `kb_{kb_id}` (legacy), `kb_{kb_id}_v2`/`v3` (Plan A reconciliation), `kb_{kb_id}_v4` (Plan B custom-sharded by month). Aliases swap to point at the live target.
- Shared private: `chat_private` (tenant-partitioned on `chat_id` + `owner_user_id`)
- Eval: `kb_eval`, `kb_1_rebuild`

**Named vectors:**

| Name | Dim | Distance | When |
|---|---|---|---|
| `dense` | 1024 | Cosine | Always present; bge-m3 |
| `bm25` (sparse) | — | IDF modifier | When created with sparse support; fastembed in-process |
| `colbert` | 128 | Cosine MaxSim (multi-vector) | When `RAG_COLBERT=1` at ingest AND collection has the slot |

**Payload schema** (canonical, per `coerce_to_canonical()`):

| Field | Type | Indexed | Tenant | Notes |
|---|---|---|---|---|
| `kb_id` | int | yes | no | KB owner |
| `doc_id` | int | yes | no | from `kb_documents.id` |
| `subtag_id` | int? | yes | no | optional |
| `chat_id` | str? | yes | yes | for `chat_private` only |
| `owner_user_id` | str | yes | yes | UUID; private docs |
| `filename` | str | yes | no | original upload name |
| `text` | str | no | no | chunk content |
| `page` | int? | no | no | PDF |
| `heading_path` | list? | no | no | breadcrumbs |
| `sheet` | str? | no | no | XLSX |
| `chunk_index` | int | yes | no | ordinal within doc |
| `level` | str | yes | no | `"chunk"` \| `"doc"` \| 1..N (RAPTOR) |
| `chunk_level` | int | no | no | RAPTOR tree level |
| `source_chunk_ids` | list | no | no | leaves a summary covers |
| `kind` | str | no | no | producer tag (e.g. `"doc_summary"`) |
| `context_prefix` | str? | no | no | Plan A 3.2 contextual prefix |
| `shard_key` | str | no | no | Plan B Phase 5 (e.g. `"2026-04"`) |
| `shard_key_origin` | str | no | no | `"filename"` \| `"body"` \| `"default"` |

**HNSW config** (defaults via env):
`m=16` (`RAG_QDRANT_M`), `ef_construct=200` (`RAG_QDRANT_EF_CONSTRUCT`), `ef=128` (`RAG_QDRANT_EF`), `full_scan_threshold=10000` (`RAG_QDRANT_FULL_SCAN_THRESHOLD`).

**Quantization** (opt-in via `RAG_QDRANT_QUANTIZE=1`): scalar INT8, 4× RAM reduction, <2% recall loss; rescoring with `RAG_QDRANT_OVERSAMPLING=2.0`.

**Live collection state** (post-Phase-5 reshard, 2026-04-26):

| Collection | Schema | Points | Role |
|---|---|---|---|
| `kb_1_v4` | dense + sparse(bm25) + colbert; custom-sharded (4 month keys × 4 shards = 16 shards); RAPTOR L1-L4 baked in | 2705 (2698 chunks + 7 RAPTOR) | LIVE — `kb_1` alias points here |
| `kb_1_v3` | Plan A canonical (dense+sparse+colbert+context_prefix, no sharding) | 2698 | Phase 5 14d rollback target until 2026-05-10 |
| `kb_1_v2` | Plan A 1.7 (dense+sparse, no colbert/context) | 2698 | Plan A 14d rollback (cron-deletes 2026-05-09) |
| `kb_1_rebuild` | legacy schema | 2698 | pre-Plan-A rollback (cron-deletes 2026-05-09) |
| `kb_eval` | dense + sparse | 130 | eval corpus |
| `open-webui_files` | upstream | 398 | not used by orgchat retrieval |

---

## 6. API surface

All extension routes mounted under upstream's FastAPI app via `ext/app.py:build_ext_routers()`. Auth via Open WebUI's existing JWT session machinery (`ext/services/auth.py`, `jwt_verifier.py`). Standalone mode (`build_app()`) for local dev / smoke tests.

### 6.1 KB administration (`ext/routers/kb_admin.py`, admin-only)

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/kb` | Create KB |
| GET | `/api/kb` | List KBs |
| GET | `/api/kb/{kb_id}` | Get KB |
| PATCH | `/api/kb/{kb_id}` | Update name/description |
| DELETE | `/api/kb/{kb_id}` | Soft-delete KB (cascades to docs + access grants) |
| GET | `/api/kb/{kb_id}/config` | Read per-KB `rag_config` JSONB |
| PATCH | `/api/kb/{kb_id}/config` | Merge keys into `rag_config` |
| POST | `/api/kb/{kb_id}/subtags` | Create subtag |
| GET | `/api/kb/{kb_id}/subtags` | List subtags |
| DELETE | `/api/kb/{kb_id}/subtags/{subtag_id}` | Delete subtag |
| POST | `/api/kb/{kb_id}/access` | Grant access (user XOR group; invalidates cache) |
| GET | `/api/kb/{kb_id}/access` | List grants |
| DELETE | `/api/kb/{kb_id}/access/{grant_id}` | Revoke grant (invalidates cache) |
| GET | `/api/kb/{kb_id}/documents` | List documents (excludes soft-deleted) |
| DELETE | `/api/kb/{kb_id}/documents/{doc_id}` | Hard-delete doc end-to-end (Qdrant + DB) |
| POST | `/api/kb/{kb_id}/documents/{doc_id}/reembed` | 501 — placeholder (originals not stored) |
| GET | `/api/kb/{kb_id}/health` | KB drift snapshot (postgres count, Qdrant count, drift_pct, oldest/newest, failed_docs) |

### 6.2 User-facing (`ext/routers/kb_retrieval.py`, authenticated)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/kb/available` | List KBs the current user has access to |
| GET | `/api/kb/{kb_id}/subtags` | List subtags (404 if no access) |
| PUT | `/api/chats/{chat_id}/kb_config` | Persist KB selection on a chat (locked once chat has ≥1 user message) |
| GET | `/api/chats/{chat_id}/kb_config` | Read KB selection from `chat.meta` |

### 6.3 Upload (`ext/routers/upload.py`)

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/kb/{kb_id}/subtag/{subtag_id}/upload` | Admin | Upload KB document (max `RAG_MAX_UPLOAD_BYTES`=50MB; sync or async per `RAG_SYNC_INGEST`) |
| POST | `/api/chats/{chat_id}/private_docs/upload` | Chat owner | Upload private doc into shared `chat_private` collection |

Returns `{status: "queued"|"done", chunks, doc_id, task_id, sha}`.

### 6.4 Retrieval & streaming

| Method | Path | Router | Purpose |
|---|---|---|---|
| POST | `/api/rag/retrieve` | `rag.py` | Low-level: retrieve from selected KBs + chat-private, rerank, budget |
| GET | `/api/rag/stream/{chat_id}?q=…` | `rag_stream.py` | SSE — `stage`, `hits`, `done`, `error` events; 15s keepalive |
| GET | `/api/kb/{kb_id}/ingest-stream?token=…` | `ingest_stream.py` | SSE — per-doc `processing` → `done`/`failed` events for admin UI; admin OR KB-grant; JWT in query (browser EventSource limitation) |

### 6.5 Health & admin UI

| Path | Purpose |
|---|---|
| `GET /healthz` | Liveness (returns `{status: "ok"}`) |
| `GET /api/kb/admin-ui` | Standalone HTML KB admin (no Svelte; `ext/static/kb-admin.html`) |
| `GET /metrics` (port 9464 in upstream mode) | Prometheus scrape |

---

## 7. Configuration

Three sources, merged at request time:
1. **Process env** (`os.environ`) — set in compose
2. **Per-KB `rag_config`** JSONB — strictest-wins UNION/MAX merge across all KBs in the request
3. **Intent overlay** — `_INTENT_FLAG_POLICY` stamps MMR/expand based on intent class; mode controlled by `RAG_INTENT_OVERLAY_MODE`

**Effective lookup:** `flags.get(key, default)` reads the per-request overlay first, then env, then the hardcoded default. Use `flags.with_overrides(dict)` to layer additional values for the duration of a `with` block.

### 7.1 Always-on infrastructure

| Var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | (required) | `postgresql+asyncpg://...` |
| `REDIS_URL` | `redis://redis:6379/0` | App cache |
| `QDRANT_URL` | `http://qdrant:6333` | Vector DB |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Celery |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Celery |
| `OPENAI_API_BASE_URL` | `http://vllm-chat:8000/v1` | Chat LLM endpoint |
| `OPENAI_API_KEY` | dummy | Chat LLM key |
| `INGEST_BLOB_ROOT` | `/var/ingest` | Async upload buffer |

### 7.2 Models & served names

| Var | Default | Purpose |
|---|---|---|
| `CHAT_MODEL` | deploy: `QuantTrio/gemma-4-31B-it-AWQ`; example: `Qwen/Qwen2.5-14B-Instruct-AWQ` | vllm-chat model |
| `EMBED_MODEL` | `BAAI/bge-m3` | TEI |
| `TEI_URL` | `http://tei:80` | TEI base |
| `RAG_QU_URL` | `http://vllm-qu:8000/v1` | Query Understanding LLM |
| `RAG_QU_MODEL` | `qwen3-4b-qu` | served name |
| `WHISPER_MODEL` | `base.en` | STT |
| `WHISPER_DEVICE` | `cpu` | STT device |
| `VISION_API_BASE_URL` | `http://vllm-chat:8000/v1` (consolidated) | vision endpoint |

### 7.3 RAG quality flags (per-KB overridable)

| Var | Default | Purpose |
|---|---|---|
| `RAG_HYBRID` | `1` | Dense + BM25 RRF fusion |
| `RAG_COLBERT` | `1` | Late-interaction multi-vector head (kb_1_v4) |
| `RAG_RERANK` | `1` | bge-reranker-v2-m3 cross-encoder on GPU 1 |
| `RAG_MMR` | `0` | Diversity rerank — intent-conditional via overlay |
| `RAG_MMR_LAMBDA` | `0.7` | MMR weight |
| `RAG_CONTEXT_EXPAND` | `0` | ±N siblings — intent-conditional |
| `RAG_CONTEXT_EXPAND_WINDOW` | `1` | siblings each side |
| `RAG_SPOTLIGHT` | `1` | Wrap chunks in `<UNTRUSTED_RETRIEVED_CONTENT>` |
| `RAG_RAPTOR` | `0` | Hierarchical summary tree (legacy; superseded by temporal) |
| `RAG_HYDE` | `0` | Hypothetical-doc query expansion |
| `RAG_HYDE_N` | `1` | HyDE expansions to average |
| `RAG_SEMCACHE` | `0` | Quantized query-vector cache |
| `RAG_DOC_SUMMARIES` | `1` | Per-doc level=doc summary point |
| `RAG_DOC_SUMMARY_TIMEOUT` | `90` | seconds |
| `RAG_CONTEXTUALIZE_KBS` | `0` | LLM context-prefix per chunk at ingest (Plan A) |
| `RAG_CONTEXTUALIZE_CONCURRENCY` | `8` | parallel LLM calls |
| `RAG_CONTEXTUALIZE_TIMEOUT` | `30` | per-call seconds |

### 7.4 Pipeline behavior

| Var | Default | Purpose |
|---|---|---|
| `RAG_SYNC_INGEST` | `0` | `1` reverts to legacy in-process sync ingest |
| `RAG_DISABLE_REWRITE` | `1` | `0` enables multi-turn query rewrite |
| `RAG_INTENT_ROUTING` | `1` | Plan B Phase 4 — route by intent label |
| `RAG_INTENT_OVERLAY_MODE` | `intent` | `env` lets env vars escape the intent overlay |
| `RAG_INJECT_DATETIME` | `1` | Inject current datetime as pseudo-source |
| `RAG_TZ` | `UTC` | timezone |
| `RAG_TEMPORAL_LEVELS` | `1` | shard_key + level injection at retrieval |
| `RAG_SHARDING_ENABLED` | `0` | New ingests use `upsert_temporal` with derived shard_keys |
| `RAG_BUDGET_TOKENIZER` | deploy: `gemma-4`; default: `cl100k` | budget + chunker tokenizer |
| `RAG_BUDGET_TOKENIZER_MODEL` | deploy: `QuantTrio/gemma-4-31B-it-AWQ` | HF model id for the tokenizer |

### 7.5 Query Understanding (Plan B Phase 4)

| Var | Default | Purpose |
|---|---|---|
| `RAG_QU_ENABLED` | `0` | Enable LLM intent classification (currently in 7-day shadow soak; flip target ~2026-05-03) |
| `RAG_QU_SHADOW_MODE` | deploy: `1` | Logging-only classification to JSONL |
| `RAG_QU_SHADOW_LOG_PATH` | `/var/log/orgchat/qu_shadow.jsonl` | path |
| `RAG_QU_INTENT_MIN_CONF` | `0.80` | confidence threshold for routing override |
| `RAG_QU_LATENCY_BUDGET_MS` | `1500` | timeout |
| `RAG_QU_CACHE_ENABLED` | `1` | Redis DB 4 |
| `RAG_QU_CACHE_TTL_SECS` | `300` | TTL |
| `RAG_QU_REDIS_DB` | `4` | DB number |

### 7.6 Plan B Phase 6 (OCR / async / structured)

| Var | Default | Purpose |
|---|---|---|
| `RAG_OCR_ENABLED` | `0` | global OCR fallback (per-KB `ocr_policy` is recommended path) |
| `RAG_OCR_TRIGGER_CHARS` | `50` | char threshold below which OCR fires |
| `RAG_STRUCTURED_CHUNKER` | `0` | Use heading-aware chunker |
| `RAG_IMAGE_CAPTIONS` | `0` | Caption embedded images via vision LLM |
| `RAG_VISION_URL` | `http://vllm-chat:8000/v1` (consolidated) | vision endpoint |
| `RAG_VISION_MODEL` | `orgchat-chat` | vision served name |

### 7.7 Reranker tuning

| Var | Default | Purpose |
|---|---|---|
| `RAG_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | cross-encoder |
| `RAG_RERANK_DEVICE` | `auto` | `cpu`/`cuda`/`auto` |
| `RAG_RERANK_MAX_LEN` | `512` | seq len |
| `RAG_RERANK_TOP_K` | `10` | candidates after retrieval, before rerank |
| `RAG_RERANK_LOAD_RETRIES` | `3` | mitigates @lru_cache load-failure poison (P1 fix) |
| `RAG_RERANK_LOAD_RETRY_BASE_SEC` | `1.0` | backoff |
| `RAG_RERANK_CACHE_DISABLED` | `0` | escape hatch |
| `RAG_RERANK_CACHE_TTL` | `300` | cache TTL seconds |

### 7.8 Qdrant tuning

| Var | Default | Purpose |
|---|---|---|
| `RAG_VECTOR_SIZE` | `1024` | dense dim |
| `RAG_QDRANT_M` | `16` | HNSW connectivity |
| `RAG_QDRANT_EF_CONSTRUCT` | `200` | build-time |
| `RAG_QDRANT_EF` | `128` | search-time |
| `RAG_QDRANT_FULL_SCAN_THRESHOLD` | `10000` | linear scan cutoff |
| `RAG_QDRANT_MAX_CONNS` | `32` | shared httpx pool |
| `RAG_QDRANT_QUANTIZE` | `0` | INT8 scalar quantization |
| `RAG_QDRANT_OVERSAMPLING` | `2.0` | quantization rescore oversampling |
| `RAG_QDRANT_ON_DISK_PAYLOAD` | `0` | spill payload to disk |

### 7.9 RBAC cache & circuit breaker

| Var | Default | Purpose |
|---|---|---|
| `RAG_RBAC_CACHE_REDIS_URL` | `redis://redis:6379/3` | dedicated DB |
| `RAG_RBAC_CACHE_TTL_SECS` | `30` | safety net (`0` disables) |
| `RAG_CIRCUIT_BREAKER_ENABLED` | `1` | per-tenant fast-fail |
| `RAG_CB_FAIL_THRESHOLD` | `3` | failures before tripping |
| `RAG_CB_WINDOW_SEC` | `300` | rolling window |
| `RAG_CB_COOLDOWN_SEC` | `30` | reset after |
| `RAG_TENACITY_RETRY` | `1` | backoff + jitter on TEI/reranker/HyDE |

### 7.10 Tier storage (Plan B Phase 5.8)

| Var | Default | Purpose |
|---|---|---|
| `RAG_TIER_HOT_MONTHS` | `3` | hot (RAM HNSW) horizon |
| `RAG_TIER_WARM_MONTHS` | `12` | warm (mmap) horizon; older = cold (on-disk + INT8) |
| `RAG_TIER_COLLECTIONS` | `kb_1_v4` | comma-separated collections fed to celery-beat cron |
| `RAG_TIER_CRON_TIMEOUT` | `1800` | seconds |

### 7.11 Observability & auth

| Var | Default | Purpose |
|---|---|---|
| `OBS_ENABLED` | `0` (overlay sets `1`) | OTel + Prometheus |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://orgchat-obs-otelcol:4317` | collector |
| `OTEL_SERVICE_NAME` | `orgchat-ext` | identity |
| `LOG_LEVEL` | `INFO` | logging |
| `PROM_METRICS_PORT` | `9464` | upstream metrics port |
| `AUTH_MODE` | deploy: `jwt`; default: `stub` | auth backend |
| `WEBUI_SECRET_KEY` | (required) | session signing |
| `RAG_ADMIN_TOKEN` | (optional) | shared secret for ops scripts (`reingest_kb.py`, `celery_soak_test.py`) |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD` | (required for seed) | bootstrap admin via `scripts/seed_admin.py` |

### 7.12 Air-gap / offline

| Var | Default | Purpose |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | block HF Hub network calls |
| `TRANSFORMERS_OFFLINE` | `1` | same for transformers |
| `HF_DATASETS_OFFLINE` | `1` | same for datasets |
| `HF_TOKEN` | (operator's) | for first-time downloads only |
| `FASTEMBED_CACHE_PATH` | `/opt/fastembed_cache` | baked into image |
| `RAG_FASTEMBED_PROVIDERS` | `CUDAExecutionProvider,CPUExecutionProvider` | colbert + sparse on GPU 1 |

### 7.13 OCR backends (when `RAG_OCR_ENABLED=1` or per-KB policy)

| Var | Purpose |
|---|---|
| `TEXTRACT_REGION` / `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS Textract |
| `DOCUMENT_AI_PROJECT` / `DOCUMENT_AI_LOCATION` / `DOCUMENT_AI_PROCESSOR` | Google Document AI |

Default backend `tesseract` — baked into open-webui + celery-worker images, no creds.

---

## 8. Frontend (Open WebUI patches)

Upstream Open WebUI is tracked as a git submodule under `upstream/`; org-specific changes are 4 surgical patches in `patches/` applied at image build (`scripts/apply_patches.sh`):

| Patch | Purpose |
|---|---|
| `0001-mount-ext-routers.patch` | Wire `ext/app.py:build_ext_routers()` into upstream's FastAPI app at startup |
| `0002-kb-selector-frontend.patch` | KBAdmin.svelte (~720 lines): full KB / subtag / document / grant / user admin UI |
| `0003-navbar-kb-picker-rewire.patch` | Wire KBPickerModal into Navbar; pending state for new chats; persist via PUT `/api/chats/{id}/kb_config` on first user message |
| `0004-admin-kb-nav-link.patch` | "Knowledge Base" entry in admin layout (route `/admin/kb`) |

Plus `patches/vllm/gemma4_mm.py` — Gemma 4 multimodal model executor with vision_tower fp16 fix (vLLM PR #40347, still open as of 2026-04-27); mounted into vllm-chat container at `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4_mm.py:ro`.

KB selection flow (frontend):
1. New chat → KBPickerModal (multi-select, hierarchical with subtags)
2. User confirms → state stored client-side (pending)
3. First user message → `PUT /api/chats/{id}/kb_config` persists to `chat.meta.kb_config`
4. Subsequent messages locked to that selection (design §2.4)
5. Private chat docs uploaded ad-hoc go into the shared `chat_private` Qdrant collection scoped on `(chat_id, owner_user_id)`

Standalone HTML admin available at `/api/kb/admin-ui` for environments where the Svelte build isn't desired.

---

## 9. Operator runbook

### 9.1 Cold start

```bash
cd /home/vogic/LocalRAG/compose
cp .env.example .env
# edit: WEBUI_NAME, DOMAIN, CHAT_MODEL, ADMIN_EMAIL, ADMIN_PASSWORD,
#       SESSION_SECRET (32 random bytes), WEBUI_SECRET_KEY (32 random bytes),
#       HF_TOKEN (only if first-time model download)

# First-time model download — needs internet; subsequent runs offline
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 docker compose -p orgchat up -d vllm-chat tei vllm-qu
# wait for healthchecks, then:
docker compose -p orgchat up -d

# Apply migrations (idempotent)
.venv/bin/python scripts/apply_migrations.py

# Seed admin
ADMIN_EMAIL=admin@local ADMIN_PASSWORD=changeme .venv/bin/python scripts/seed_admin.py

# Seed analyst RAG_TEMPLATE + orgchat-chat system prompt (military briefing
# format + visual-first habit). Idempotent — safe to re-run any time the
# .txt files in scripts/ are updated. Required for the depth/structure
# behavior; without it, Open WebUI's bundled DEFAULT_RAG_TEMPLATE wins and
# answers ship short. See scripts/rag_template_military.txt + scripts/system_prompt_analyst.txt.
.venv/bin/python scripts/apply_analyst_config.py

# Browse to https://${DOMAIN}:8443 (or http://localhost:8880 → 6100)
```

### 9.2 Daily operations

- **Logs:** `docker compose -p orgchat logs -f open-webui celery-worker vllm-chat vllm-qu`
- **Metrics:** Prometheus at `http://localhost:9091`, Grafana via observability overlay
- **Traces:** Jaeger at `http://localhost:16686`
- **Qdrant cluster status:** `curl http://localhost:6333/cluster`
- **Soak test (non-prod KB only):** `RAG_ADMIN_TOKEN=… .venv/bin/python scripts/celery_soak_test.py --target-kb <kb_id> --doc-count 1000 --concurrency 8`
- **Rollback alias** (Phase 5): see `scripts/kb_cleanup_phase5_rollback.sh`
- **Daily eval cron** (`scripts/daily_eval_cron.sh`): writes textfile collector → Prometheus gauge → alert on 5pp nDCG drop from 7d median

### 9.3 Backups

`scripts/backup.sh` snapshots Qdrant collections + dumps Postgres + tars `volumes/uploads`. `scripts/restore.sh` for the inverse. Cron-driven; backups in `/home/vogic/LocalRAG/backups/`.

### 9.4 Cleanup calendar

| Date | Action |
|---|---|
| 2026-05-03 | End of QU 7-day shadow soak; review `rag_qu_*` metrics; if clean, flip `RAG_QU_ENABLED=1` |
| 2026-05-09 | cron deletes `kb_1_v2` + `kb_1_rebuild` (Plan A 14d window expired) |
| 2026-05-10 | operator may drop `kb_1_v3` (Phase 5 rollback expired); extend `scripts/kb_cleanup_post_plan_a.sh` |

### 9.5 Known soak fixes (post-Plan-B-merge, all on `main`)

- `2b5129b` `vector_store.upsert` accepts `shard_key_selector` + auto-derives from payload
- `c81f003` `Dockerfile.openwebui.cu128` adds `celery[redis]` + `redis` deps
- `26bde3c` `vector_store.upsert` forces named-vector path when target has any named slot
- `94798ee` + migration `012` `upsert_temporal` delegates to `upsert`; `queued` added to status enum
- `4a32141` `ingest_worker._update_doc_status` transitions `queued → done` / `failed`
- `7a89d25` B11+B12 regex precedence — list-in-KB metadata + summary-of-date specific_date
- `2802b89` patch 0003 KBPickerModal → Navbar
- `1dcb08b` patch 0004 admin nav link

### 9.6 Image build gotchas

- **`Dockerfile.celery` is intentionally minimal.** No torch / transformers / sentence-transformers. Heavy reranker + dense embed live in open-webui. Worker uses TEI over HTTP for dense + fastembed for sparse + colbert.
- **First-time celery image build is slow** (~30 min for fastembed onnxruntime). Subsequent builds cache the pip layer.
- **BuildKit cache invalidation:** when only the COPY layer changes, sometimes BuildKit's content-hash cache doesn't invalidate cleanly. If a fresh image still shows old code, force rebuild with `--no-cache`.
- **Build is non-tty.** `docker compose build` emits no progress in non-interactive shells. Errors are tail-only — check `docker history` to see what layers landed.
- **`celery-worker` and `celery-beat` use SEPARATE images** even though both build from `Dockerfile.celery`. After rebuilding worker, also tag for beat:
  ```bash
  docker tag orgchat-celery-worker:latest orgchat-celery-beat:latest
  docker compose -p orgchat up -d --force-recreate celery-beat
  ```

---

## 10. Testing strategy

`tests/` layout:
- `tests/unit/` — fast, no I/O. Pipeline shape, classifier, flag overlay, intent overlay mode, RRF fusion, MMR, budget, spotlight, contextualizer prompt construction, etc. ~1040 unit tests as of Phase 6 baseline.
- `tests/integration/` — Postgres + Qdrant + Redis live (testcontainers or compose-managed). Six mandatory **isolation tests** in `test_kb_isolation.py`. RBAC pubsub propagation, RAG retrieval, model loading on-demand.
- `tests/e2e/` — multi-user concurrent flows, image/audio trigger paths, KB-with-private-doc.
- `tests/eval/` — golden sets (`golden_starter.jsonl` 60+ queries, `golden_evolution.jsonl` 30 queries) + RAGAS / TruLens harness scripts. Per-intent stratified (point-in-time, evolution, aggregation, current-state).

Mandatory CI gates:
1. All six isolation tests pass
2. `make eval-baseline` does not regress nDCG@10 by >5pp from the committed baseline (Phase 0 baseline JSON in `tests/eval/results/`)
3. RBAC tests pass (admin grant/revoke, group membership, cache invalidation)
4. Pytest suite green

Run: `.venv/bin/pytest -q` (host `pytest` may not be on PATH in plan-b worktree contexts — use the venv binary explicitly).

---

## 11. Roadmap & open work

### 11.1 Phase status

| Phase | Status |
|---|---|
| Plan A Phases 0–3 (eval harness + robustness + cheap quality + contextualization + canonical schema) | **MERGED** to main 2026-04-25 |
| Plan B Phase 4 — QU LLM | **MERGED**; in 7-day shadow soak. Flip `RAG_QU_ENABLED=1` ~2026-05-03 after clean metrics |
| Plan B Phase 5 — Qdrant temporal sharding + temporal RAPTOR | **MERGED**; live on `kb_1 → kb_1_v4`. Pending: hand-label `golden_evolution.jsonl`, run `make eval-evolution`, commit `phase-5-baseline.json` |
| Plan B Phase 6 — async ingest + OCR + structured chunking | **MERGED**; default `RAG_SYNC_INGEST=0`; per-KB `chunking_strategy` + `ocr_policy` plumbing done; **structured chunker not yet wired into `ingest_bytes`** (`chunking_strategy="structured"` is a no-op until then) |

### 11.2 Operator follow-ups (deferred from merge)

- **QU LLM:** finish 7-day shadow soak (started 2026-04-26), flip `RAG_QU_ENABLED=1`, run eval baseline with QU on
- **Phase 5 eval:** label `tests/eval/golden_evolution.jsonl`, run `make eval-evolution`, commit baseline
- **Phase 5 cleanup cron:** extend `scripts/kb_cleanup_post_plan_a.sh` to drop `kb_1_v3` after 2026-05-10
- **Phase 6 OCR:** rebuild open-webui image (already has tesseract), enable `RAG_OCR_ENABLED=1` per-KB via `ocr_policy` PATCH for non-English KBs; re-ingest scanned PDFs via `scripts/reingest_for_ocr.py`
- **Phase 6 structured chunker:** wire `chunk_text_for_kb` + `extract_images_as_chunks` into `ingest_bytes` (currently calls legacy `chunk_text`)
- **Intent overlay A/B:** per the recorded A/B plan — when real production data lands, run baseline with `RAG_INTENT_OVERLAY_MODE=intent` vs `=env`, commit results, decide default

### 11.3 Known robustness gaps (from review 2026-04-24)

P0 / P1 risks identified before scale-out (some addressed, some remaining):

- **P0** Per-KB circuit breaker / bulkhead: shared httpx pool of 32 — one slow Qdrant collection stalls all parallel KB searches. *Status: partial — `circuit_breaker.py` exists but per-tenant isolation not yet in `vector_store.py`.*
- **P0** HF_TOKEN silent fallback: budget tokenizer fails to cl100k → 10–15% drift. *Status: addressed by Phase 1.1 preflight in `app.py` — startup crashes loudly if `RAG_BUDGET_TOKENIZER` is set explicitly to a non-cl100k alias and can't load.*
- **P1** Reranker `@lru_cache` failure poisoning. *Status: addressed via retry singleton (`RAG_RERANK_LOAD_RETRIES`).*
- **P1** Stale RBAC during long sessions. *Status: addressed by Redis pubsub `rbac:invalidate` + 30s TTL.*
- **P1** No backoff/jitter on TEI / reranker / HyDE. *Status: addressed via tenacity in `retry_policy.py`.*
- **P2** Ingest summary failure doesn't flip status. *Status: addressed by `_update_doc_status` (commit `4a32141`).*
- **P2** Spotlight wrapping at retrieval not ingest — rerank-cache hits could bypass. *Status: still at retrieval.*
- **P2** Contextualizer concurrency unbounded across workers. *Status: per-process cap at 8; cross-worker not gated.*

### 11.4 Future / out of current scope

- Phase 2 hardware: 4×48 GB GPUs + Kubernetes; 70B models; higher concurrency
- Document versioning + rollback
- Custom fine-tuning pipelines
- SOC2 / HIPAA compliance overlay
- Microsoft GraphRAG / LightRAG / HippoRAG (deferred unless eval shows residual gaps after Plan B settles)
- Self-RAG (deferred — needs fine-tune; poor ROI self-hosted)
- Proposition chunking (rejected — actively wrong for narrative/evolution corpora)
- One-collection-per-month sharding (rejected — kills RBAC)

---

## 12. Repo layout (current)

```
LocalRAG/
├── ext/                                Extensions (org-owned code)
│   ├── app.py                          FastAPI app builder; standalone + upstream-mount modes
│   ├── config.py                       Settings (pydantic), settings cache
│   ├── db/
│   │   ├── session.py                  AsyncEngine + async_sessionmaker
│   │   ├── base.py                     DeclarativeBase
│   │   ├── qdrant_schema.py            Canonical payload schema, indexes, HNSW config
│   │   ├── migrations/                 001…012 SQL files (auto-applied by postgres init)
│   │   └── models/{kb,chat_ext,compat}.py
│   ├── routers/                        FastAPI routers (kb_admin, kb_retrieval, upload,
│   │                                   rag, rag_stream, ingest_stream)
│   ├── services/                       Business logic — see §2 for canonical map
│   ├── workers/                        Celery (celery_app, ingest_worker, scheduled_eval, blob_gc_task)
│   └── static/kb-admin.html            Standalone admin UI fallback
├── compose/
│   ├── docker-compose.yml              Main stack
│   ├── docker-compose.observability.yml LGTM/Jaeger overlay
│   ├── Dockerfile.celery               Minimal worker (no torch)
│   ├── Dockerfile.vllm-chat            Deprecated shim (use nightly image)
│   ├── vllm-qu/                        QU-specific files
│   ├── caddy/Caddyfile                 Reverse proxy
│   └── .env.example                    Template
├── observability/                      LGTM stack docker-compose + configs
├── Dockerfile.openwebui.cu128          Open WebUI image, cu128 + torch ≥ 2.7
├── upstream/                           Open WebUI git submodule
├── patches/                            0001…0004 + vllm/gemma4_mm.py
├── scripts/                            Operator + maintenance scripts
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── eval/                           golden_*.jsonl, RAGAS/TruLens harnesses
│   └── conftest.py
├── docs/
│   └── superpowers/
│       ├── plans/                      Implementation plans (Plan A, Plan B)
│       └── specs/                      Original design specs
├── volumes/                            Bind-mount targets (models, uploads, hf-cache, certs, observability)
├── backups/                            Snapshot output
├── brand/                              Logos, favicons (baked into open-webui image)
├── model_manager/                      Health probe + on-demand orchestrator
├── whisper_service/                    faster-whisper FastAPI wrapper
├── tts_service/                        Piper TTS FastAPI wrapper
├── k8s/                                Future Phase 2 manifests (placeholder)
├── Makefile                            up, smoke, eval-baseline, eval-evolution, etc.
├── pyproject.toml                      Python project config (ruff, mypy, deps)
└── CLAUDE.md                           This file
```

---

## 13. Operating principles (preserved across rewrites)

These invariants override surface-level optimizations. Break them only with an explicit decision in the relevant plan doc.

1. **Isolation invariant** — three-layer enforcement (DB / API / Vector). The DB miss MUST always run on cache absence so isolation is never weakened by cache outage. RBAC cache is an accelerator, not a source of truth.
2. **Fail-open everywhere** — silent fallthrough must `_record_silent_failure(stage, err)` (logged + counter incremented). Never re-raise from the helper. Counter or label cardinality must never break retrieval.
3. **Default-off feature flags** — quality boosts (`RAG_RERANK`, `RAG_MMR`, `RAG_CONTEXT_EXPAND`, `RAG_HYDE`, `RAG_SPOTLIGHT`, `RAG_CONTEXTUALIZE_KBS`, `RAG_RAPTOR`, `RAG_SEMCACHE`) ship default-off. The exceptions deliberately turned on for kb_1_v4 (`RAG_HYBRID`, `RAG_COLBERT`, `RAG_RERANK`, `RAG_DOC_SUMMARIES`, `RAG_INTENT_ROUTING`, `RAG_SPOTLIGHT`, `RAG_INJECT_DATETIME`, `RAG_TEMPORAL_LEVELS`) are documented in compose.
4. **Air-gapped deployment** — all model weights cached under `volumes/models` + `/opt/fastembed_cache` + `/opt/tiktoken-cache`. `HF_HUB_OFFLINE=1` everywhere except first-time download. Validate before deploy: `python scripts/preflight_models.py`.
5. **Idempotent operations** — UUIDv5 deterministic point IDs (doc_id + chunk_index); `task_acks_late=True`; blob SHA dedup. Re-running ingest is always safe.
6. **Schema reconciliation before quality** — Plan A 1.7 enforced canonical Qdrant payload (integer doc_id, on_disk_payload=True, indexed fields) before Phase 3 added `context_prefix` and ColBERT. Same pattern applies for any future field addition.
7. **Eval gate before flag flip** — every quality flag has a baseline JSON in `tests/eval/results/`. No flag flips production-on without `make eval-gate` passing on the relevant golden set.
8. **Re-ingest is operator work, not a deploy** — Plan A Phase 3.7 and Plan B Phase 5 reshards run in dedicated off-hours windows with dual-collection alias cutover and 14-day rollback retention. Code lands in normal deploy; data migration is a separate runbook step.
9. **Per-KB rag_config wins over env** — admins can override quality flags per collection. Env is the global default; per-KB JSONB is the customization layer; intent overlay is the per-request shape.
10. **Three-tier separation of compose / env / per-KB config** — compose pins service shape (images, volumes, GPUs). `.env` pins per-deploy values (secrets, model names, defaults). `rag_config` pins per-collection behavior. Don't mix.

---

## 14. Plan & spec docs

Detailed implementation plans (TDD-driven, ~6000–8000 lines each):

- `docs/superpowers/specs/2026-04-12-org-chat-assistant-design.md` — original design (sections 1–14)
- `docs/superpowers/specs/2026-04-16-kb-rag-pipeline-workflow.md` — 8-stage RAG, 10 mitigations
- `docs/superpowers/specs/2026-04-16-kb-rag-pipeline-implementation.md` — 7-phase initial implementation
- `docs/superpowers/plans/2026-04-24-rag-robustness-and-quality.md` — Plan A (Phases 0–3, 29 tasks, 254 TDD steps)
- `docs/superpowers/plans/2026-04-25-rag-plan-b-llm-shard-async.md` — Plan B (Phases 4–6: QU LLM + temporal sharding/RAPTOR + async + OCR)
- `docs/runbook/temporal-reshard-procedure.md` — Phase 5 cluster-mode enablement runbook

Long-form context (older drafts, kept for archaeology — not necessarily current):
- `RAG.md`, `Ragupdate.md`, `recommendation.md`, `debugger.md`, `upgrade_issue.md`

---

## 15. Quick troubleshooting

| Symptom | Likely cause | First check |
|---|---|---|
| Upload returns 200 `{status: "queued"}` but doc never appears | Celery worker stuck OR blob path mismatch | `docker compose logs celery-worker`; `ls /var/lib/docker/volumes/orgchat_ingest_blobs/_data/` |
| `Wrong input: Not existing vector name error` on upsert | Mixed-schema points in collection — vector_store force-named-vector path missing | Verify commit `26bde3c` in deployed image; rebuild |
| `Shard key not specified` 400 on async ingest | `RAG_SHARDING_ENABLED=1` but `shard_key_selector` not derived from payload | Verify commit `2b5129b`; rebuild |
| `kb_documents.ingest_status` stuck at `queued` | `_update_doc_status` not transitioning | Verify commit `4a32141`; rebuild |
| Image upload returns `<pad>` tokens | Gemma 4 vision_tower fp16 overflow | Verify `patches/vllm/gemma4_mm.py` mounted into vllm-chat |
| `<reasoning_token>` markers in chat output | gemma4 reasoning parser missing | Verify `--reasoning-parser gemma4` + `--tool-call-parser gemma4` in vllm-chat command |
| Token budget evicting too much | tokenizer mismatch (cl100k vs gemma-4) | Set `RAG_BUDGET_TOKENIZER=gemma-4` + `RAG_BUDGET_TOKENIZER_MODEL=QuantTrio/gemma-4-31B-it-AWQ`; verify HF cache mount |
| RBAC revocation not propagating | pubsub down | Logs for `rbac subscriber: started`; falls back to 30s TTL |
| QU LLM unreachable; chat still works | bridge soft-fails to regex | `docker compose logs vllm-qu`; metrics `rag_qu_*` |
| Qdrant cluster mode regression after compose change | mismatched compose files between worktrees | Verify `QDRANT__CLUSTER__ENABLED=true` + `--uri http://qdrant:6335` in BOTH compose files |

---

**Document maintained by:** Claude Code (sub-agent-driven analysis)
**Last fully synthesized:** 2026-04-28 from current code + memory snapshots + parallel exploration agents
**Source of truth:** `ext/`, `compose/`, `scripts/` over this document. When this file disagrees with code, code wins.
