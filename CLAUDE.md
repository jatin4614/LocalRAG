# LocalRAG — Org Chat Assistant with Hierarchical KB RAG

Self-hosted, air-gapped, multi-user ChatGPT-style assistant for orgs (20–200 users). Thin fork of Open WebUI; org code lives under `ext/`; upstream is **vendored** under `upstream/` (was a submodule pre-2026-04-30). Plan A + Plan B merged to `main` (commit `d1aa862`, 2026-04-26).

**Hardware:** GPU 0 RTX 6000 Ada 48 GB (vllm-chat), GPU 1 RTX PRO 4000 Blackwell 24 GB (TEI, vllm-qu, reranker, fastembed). 125 GB RAM. **Deploy:** docker-compose, single host, `HF_HUB_OFFLINE=1` after first download.

## 1. Core invariants (don't break without an explicit decision)

1. **Isolation, three layers.** Every Qdrant chunk has either `kb_id`/`doc_id` (+optional `subtag_id`) for shared KBs, OR `chat_id`/`owner_user_id` for chat-private. Never both. Enforced at DB (`kb_access` XOR check), API (`rbac.get_allowed_kb_ids` + Redis cache w/ pubsub invalidation, DB is source of truth — cache miss MUST fall through to DB), and vector layer (every search injects `kb_id ∈ allowed` filter). The mandatory CI gate is the **isolation suite (11 tests across three files):**
   - `tests/integration/test_kb_isolation.py` — 2 tests: cross-user KB visibility + admin-only routes (end-to-end via FastAPI test client).
   - `tests/integration/test_rag_isolation.py` — 2 tests: chat-private retrieval + chat-scoped query containment (end-to-end RAG path).
   - `tests/integration/test_rbac_cache_invalidation.py` — 6 tests: the RBAC cache contract (group grant, pubsub revocation, TTL safety net, dropped-message recovery, concurrent-revocation consistency, per-user cache key isolation).
   - `tests/unit/test_kb_models.py::test_kb_access_check_enforced_in_model` — 1 test: SQL CHECK XOR constraint at the model layer (`KBAccess(user_id=None, group_id=None)` and `KBAccess(user_id=1, group_id=1)` both raise).
2. **Fail-open.** Silent fallthrough must `_record_silent_failure(stage, err)` (logs + counter). Never re-raise from helper. Counter / label cardinality must never break retrieval.
3. **Default-off quality flags.** `RAG_RERANK`, `RAG_MMR`, `RAG_CONTEXT_EXPAND`, `RAG_HYDE`, `RAG_SPOTLIGHT`, `RAG_CONTEXTUALIZE_KBS`, `RAG_RAPTOR`, `RAG_SEMCACHE` ship default-off. Production deviations live in `compose/.env`.
4. **Air-gapped.** All weights cached under `volumes/models` + `/opt/fastembed_cache` + `/opt/tiktoken-cache`. Validate before deploy with `python scripts/preflight_models.py`.
5. **Idempotent.** UUIDv5 deterministic point IDs (`doc_id + chunk_index`), `task_acks_late=True`, blob SHA dedup. Re-running ingest is always safe.
6. **Per-KB `rag_config` JSONB > env > intent overlay.** Strictest-wins UNION/MAX merge across selected KBs at request time. Env is global default; per-KB is collection customization; intent overlay (`_INTENT_FLAG_POLICY` in `chat_rag_bridge.py`) shapes per-request. Mode controlled by `RAG_INTENT_OVERLAY_MODE` (`intent` default | `env`).
7. **Eval gate before flag flip.** Baselines under `tests/eval/results/`; `make eval-gate` must pass before flipping a flag production-on.
8. **Re-ingest is operator work, not deploy.** Dual-collection alias cutover with 14d rollback; runbook step, not code path.

## 2. Where to find what

| Concern | Path |
|---|---|
| RAG entry point | `ext/services/chat_rag_bridge.py` `retrieve_kb_sources` → `_run_pipeline` |
| Ingest pipeline | `ext/services/ingest.py:ingest_bytes` |
| Parallel retrieve | `ext/services/retriever.py:retrieve` |
| Rerank | `ext/services/reranker.py` + `cross_encoder_reranker.py` |
| Vector store | `ext/services/vector_store.py` (httpx pool, sharding-aware upsert) |
| Embedder | `ext/services/embedder.py`, `sparse_embedder.py` |
| Routers | `ext/routers/{kb_admin,kb_retrieval,upload,rag,rag_stream,ingest_stream}.py` |
| Per-KB config / flags | `ext/services/{kb_config,flags}.py` |
| Postgres migrations | `ext/db/migrations/001…012_*.sql` (idempotent, asyncpg) |
| Models | `ext/db/models/{kb,chat_ext,compat}.py` |
| Qdrant schema | `ext/db/qdrant_schema.py` |
| Celery | `ext/workers/{celery_app,ingest_worker,scheduled_eval,blob_gc_task}.py` |
| Compose | `compose/docker-compose.yml` (+ `docker-compose.observability.yml` overlay) |
| Frontend patches | `patches/0001…0004` pre-applied in `upstream/` (kept for re-derive) |
| Operator scripts | `scripts/*.py` |

## 3. Services (compose)

postgres (KB metadata, RBAC), redis (DB 0 cache / 1 broker / 2 results / 3 RBAC / 4 QU), qdrant (cluster mode since 2026-04-26; collections `kb_{id}`, `kb_{id}_v4`, `chat_private`), **vllm-chat** (GPU 0, Gemma-4-31B-it-AWQ, 32K ctx, fp8 KV, `gemma4` reasoning + tool-call parsers, `patches/vllm/gemma4_mm.py` mounted), **vllm-qu** (GPU 1, Qwen3-4B-AWQ as `qwen3-4b-qu`, soft-fails to regex), **tei** (GPU 1, bge-m3 1024d; sparse is client-side via fastembed), **open-webui** (GPU 1, runs cross-encoder `bge-reranker-v2-m3` + fastembed BM25/ColBERT, cu128 + torch ≥ 2.7), whisper (CPU int8 base.en), tts (Piper), model-manager, celery-worker (GPU 1, `Dockerfile.celery` is **intentionally minimal — no torch/transformers**; uses TEI + fastembed), celery-beat (single instance; daily 03:00 tier-storage cron), caddy.

Network `orgchat-net`. Volumes: `postgres_data`, `redis_data`, `qdrant_data`, `ingest_blobs` (open-webui ↔ celery-worker), `celery_beat_state`. Bind mounts: `volumes/{models,uploads,hf-cache,certs}`.

Observability overlay (`docker-compose.observability.yml` + `observability/`): prometheus 3.5 (9091:9090), loki+promtail, jaeger (16686), grafana, otelcol, dcgm/node/redis/postgres/cadvisor exporters.

## 4. RAG pipeline

### Ingest (`ingest.py:ingest_bytes`)
Extract (pypdf+pdfplumber+pymupdf / docx / openpyxl / md / html → `ExtractedBlock`) → optional OCR (`RAG_OCR_ENABLED=1` or per-KB `ocr_policy`; tesseract default, Textract/DocumentAI opt-in) → Chunk (default `window` 800/100; `structured` chunker in `chunker_structured.py` is wired via `ingest.chunk_text_for_kb` (`ext/services/ingest.py:1074-1106`) and fires when per-KB `rag_config.chunking_strategy="structured"` AND env `RAG_STRUCTURED_CHUNKER=1` are both on — table/code preserved as atomic units) → optional Contextualize (LLM ~50-tok prefix, concurrency 8, prefix-cache friendly) → optional RAPTOR / Temporal RAPTOR (L0 chunks → L1 month → L2 quarter → L3 year → L4 multi-year; `temporal_raptor.py` superseded legacy `raptor.py` for time-aware corpora) → Doc summary (`RAG_DOC_SUMMARIES=1` default; mirrored to `kb_documents.doc_summary`; required for global intent) → Embed (TEI dense, fastembed sparse + colbert) → Upsert (deterministic UUIDv5; named vectors `dense`/`bm25`/`colbert`; custom shard_key derived from filename date or body fallback when `RAG_SHARDING_ENABLED=1`).

`kb_documents.ingest_status` ladder (CHECK constraint, migration 012): `pending → queued → chunking → embedding → done | failed`. Async path: open-webui writes blob to `/var/ingest/{sha}` → Celery picks up → `ingest_bytes`. DLQ after 3 retries w/ exponential backoff. Blob GC after `RAG_BLOB_RETENTION_DAYS`.

### Retrieve (`chat_rag_bridge.py:retrieve_kb_sources`)
1. active_sessions inc → 2. optional query rewrite (`RAG_DISABLE_REWRITE=1` default) → 3. request_ctx (request_id + user_id) → 4. RBAC (`rbac.get_allowed_kb_ids`, Redis DB 3, TTL `RAG_RBAC_CACHE_TTL_SECS=30`, pubsub `rbac:invalidate`, fail-closed) → 5. per-KB `rag_config` merge → 6. intent classify (regex `metadata`/`global`/`specific`/`specific_date`; B11/B12 fix `7a89d25` tightened metadata vs specific_date; QU LLM hybrid via `_classify_with_qu` w/ Redis DB 4 cache) → 7. intent flag overlay → 8. embed query (HyDE optional, SemCache optional) → 9. **parallel retrieve** (`asyncio.gather` per selected KB + `chat_private`; shared httpx pool 32; hybrid dense+sparse RRF k=60; tri-fusion +colbert when `RAG_COLBERT=1`; filters: level/shard_key/doc_ids; limits 10/KB 30 total, global 50/100, specific_date 30/60) → 10. cross-KB merge (RRF k=60, or raw sort if rerank on) → 11. rerank (heuristic max-norm default; `bge-reranker-v2-m3` GPU 1 when `RAG_RERANK=1`; cache 300s; **global intent skips rerank**; B6 silent-failure counters) → 12. MMR (`RAG_MMR=1`, λ=`RAG_MMR_LAMBDA=0.7`, off for global) → 13. context expand (±N siblings, off for global) → 14. time decay (intent-conditional, not currently default-on) → 15. token budget (`budget.truncate_to_max_tokens`, tokenizer via `RAG_BUDGET_TOKENIZER`; gemma-4 in deployed env) → 16. spotlight wrap (`<UNTRUSTED_RETRIEVED_CONTENT>` tags, defang nested closes) → 17. format sources + KB Catalog preamble + datetime preamble (`RAG_INJECT_DATETIME=1`, `RAG_TZ=UTC`) → 18. SSE progress emit → 19. active_sessions dec.

Per-KB `rag_config` JSONB keys: `rerank`, `mmr`, `context_expand`, `context_expand_window`, `colbert`, `hybrid`, `spotlight`, `hyde`, `raptor`, `chunking_strategy`, `contextualize`, `intent_overlay_mode`.

Cross-cutting: metrics (`rag_stage_latency_seconds`, `rag_retrieval_hits_total`, `rag_flag_enabled`, `RAG_SILENT_FAILURE`, `llm_*`, `rag_qu_*`); OTel spans (`rag.rbac_check`, `embed.query`, `retrieve.parallel`, `retrieve` per-KB, `rerank.score`, `mmr_rerank`, `budget.truncate`); per-tenant circuit breaker (`RAG_CB_FAIL_THRESHOLD=3` / `RAG_CB_WINDOW_SEC=300` / cooldown 30s); tenacity retry on TEI/reranker/HyDE.

## 5. Schema (high-level — read the migration / model file for detail)

**Postgres** (12 migrations; `scripts/apply_migrations.py` runs sorted, idempotent):
- `knowledge_bases` (id, name UNIQUE, admin_id, deleted_at, **rag_config JSONB GIN-indexed**, **ocr_policy JSONB**)
- `kb_subtags` (CASCADE)
- `kb_documents` (ingest_status enum w/ CHECK, chunk_count, pipeline_version, blob_sha, **doc_summary**)
- `kb_access` (CHECK exactly-one user XOR group; access_type `read`|`write`)
- Upstream tables (`users`, `groups`, `user_groups`, `chats`) read via `ext/db/models/compat.py`. Canonical KB selection lives in `chat.meta.kb_config` JSON (migration 007 dropped the orphan column).

**Qdrant** — collections `kb_{id}` (legacy), `kb_{id}_v4` (custom-sharded by month, live for kb_1 since 2026-04-26 Phase 5 reshard); shared `chat_private`. Named vectors `dense`(1024 cosine) / `bm25`(sparse IDF) / `colbert`(128 multi-vector MaxSim). Indexed payload: `kb_id`, `doc_id`, `subtag_id`, `chat_id`, `owner_user_id`, `filename`, `chunk_index`, `level`. HNSW `m=16 / ef_construct=200 / ef=128`. Optional INT8 quant w/ 2x oversample rescore. `level` ∈ `chunk` | `doc` | `1..N` (RAPTOR).

Live state post-Phase-5: `kb_1_v4` (2705 pts, alias target) | `kb_1_v3` (Phase 5 rollback until 2026-05-10) | `kb_1_v2` + `kb_1_rebuild` (cron-deletes 2026-05-09).

## 6. API (mounted via `ext/app.py:build_ext_routers()`; auth via Open WebUI JWT)

**Admin** (`/api/kb`): CRUD on KBs, subtags, access grants; `GET /api/kb/{id}/health` returns drift snapshot; `DELETE /api/kb/{id}/documents/{doc_id}` hard-deletes end-to-end; `POST /api/kb/{id}/access` invalidates RBAC cache via pubsub.

**User**: `GET /api/kb/available`, `GET/PUT /api/chats/{id}/kb_config` (selection locks once chat has ≥1 user message).

**Upload**: `POST /api/kb/{id}/subtag/{sid}/upload` (admin), `POST /api/chats/{id}/private_docs/upload`. Max `RAG_MAX_UPLOAD_BYTES=50MB`; sync vs async per `RAG_SYNC_INGEST` (default 0). Returns `{status, chunks, doc_id, task_id, sha}`.

**Retrieval/SSE**: `POST /api/rag/retrieve`, `GET /api/rag/stream/{chat_id}`, `GET /api/kb/{id}/ingest-stream?token=…` (SSE; JWT in query for EventSource limitation; admin OR KB-grant).

**Health**: `GET /healthz`, `GET /api/kb/admin-ui` (standalone HTML at `ext/static/kb-admin.html`), `GET /metrics` (port 9464 in upstream mode).

## 7. Configuration

`compose/.env.example` is the source of truth for vars + defaults. The list below is what you most often need to know about.

**Models** — `CHAT_MODEL=QuantTrio/gemma-4-31B-it-AWQ` (deploy), `EMBED_MODEL=BAAI/bge-m3`, `RAG_QU_MODEL=qwen3-4b-qu`, `RAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3`, `RAG_BUDGET_TOKENIZER=gemma-4` + `RAG_BUDGET_TOKENIZER_MODEL=QuantTrio/gemma-4-31B-it-AWQ`.

**Production-on flags** (deviating from default-off): `RAG_HYBRID=1`, `RAG_COLBERT=1`, `RAG_RERANK=1`, `RAG_DOC_SUMMARIES=1`, `RAG_INTENT_ROUTING=1`, `RAG_SPOTLIGHT=1`, `RAG_INJECT_DATETIME=1`, `RAG_TEMPORAL_LEVELS=1`, `RAG_CONTEXT_EXPAND=1`, `RAG_MMR=1`, `RAG_GLOBAL_DRILLDOWN=1` (`K=5`, budget `RAG_GLOBAL_BUDGET_TOKENS=22000` for global vs `RAG_BUDGET_TOKENS=10000` otherwise), `RAG_TOP_K=12` / `RAG_RERANK_TOP_K=12`.

**Async path**: `RAG_SYNC_INGEST=0`, `INGEST_BLOB_ROOT=/var/ingest`, `RAG_BLOB_RETENTION_DAYS`.

**QU LLM**: `RAG_QU_ENABLED=0` (in 7-day shadow soak; flip target ~2026-05-03). Shadow log at `RAG_QU_SHADOW_LOG_PATH=/var/log/orgchat/qu_shadow.jsonl`. Cache: Redis DB 4, TTL 300s.

**Air-gap**: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `FASTEMBED_CACHE_PATH=/opt/fastembed_cache`, `RAG_FASTEMBED_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`.

**Auth/secrets**: `AUTH_MODE=jwt`, `WEBUI_SECRET_KEY`, `RAG_ADMIN_TOKEN` (ops scripts), `ADMIN_EMAIL`/`ADMIN_PASSWORD` (`scripts/seed_admin.py`).

**Trap**: docker-compose passes vars by **explicit** mapping — to flip a behaviour from `.env`, the var must also be listed under the service's `environment:` block in `compose/docker-compose.yml`. We've hit this for `RAG_GLOBAL_DRILLDOWN_K` / `RAG_GLOBAL_BUDGET_TOKENS` / `RAG_MMR_LAMBDA` etc.

## 8. Operator runbook

**Cold start** (long-form: `docs/runbook/cold-start.md`):
```bash
cd compose && cp .env.example .env  # edit DOMAIN, ADMIN_*, *_SECRET_KEY, HF_TOKEN
HF_HUB_OFFLINE=0 docker compose -p orgchat up -d vllm-chat tei vllm-qu  # first-time download only
docker compose -p orgchat up -d
# Wave 1b (review §11.10): one-shot UID alignment for the shared /var/ingest
# volume between open-webui (producer) and celery-worker (consumer). After
# the USER 1000:1000 baked into the images (review §10.1), the named volume
# may still be root-owned from its first creation; chown once.
docker compose -p orgchat run --rm --user root open-webui \
    chown -R 1000:1000 /var/ingest /app/backend/data/uploads /root/.cache/huggingface
.venv/bin/python scripts/apply_migrations.py
ADMIN_EMAIL=… ADMIN_PASSWORD=… .venv/bin/python scripts/seed_admin.py
.venv/bin/python scripts/apply_analyst_config.py  # idempotent — seeds RAG_TEMPLATE + system prompt
```

**Daily**: `docker compose logs -f open-webui celery-worker vllm-chat`; Prometheus 9091; Jaeger 16686; `curl localhost:6333/cluster`.

**Backup / restore**: see `docs/runbook/backup-restore.md`. Daily Qdrant snapshots fire automatically via `ext/workers/snapshot_task.py` (Celery beat, 02:30 UTC). Before risky changes (schema migrations, Qdrant collection changes), the operator MUST run `scripts/backup_postgres.sh` + `scripts/backup_qdrant.sh` and verify via `scripts/restore_drill.sh`.

**Tests**: `.venv/bin/pytest -q` (host `pytest` may not be on PATH). Mandatory CI: 6 isolation tests + RBAC tests + `make eval-baseline` (no >5pp nDCG@10 regression vs committed baseline).

**Image gotchas**: `Dockerfile.celery` is intentionally minimal — heavy reranker/embed live in open-webui. First-time celery image build ~30 min (fastembed onnxruntime). After rebuilding worker, also `docker tag orgchat-celery-worker:latest orgchat-celery-beat:latest` + recreate beat.

## 9. Quick troubleshooting

| Symptom | Cause | Check |
|---|---|---|
| Upload `{queued}` but doc never appears | Celery stuck OR blob path mismatch | `docker compose logs celery-worker`; `ls /var/lib/docker/volumes/orgchat_ingest_blobs/_data/` |
| `Wrong input: Not existing vector name` on upsert | Mixed-schema points; force-named-vector path missing | Verify commit `26bde3c` in deployed image |
| `Shard key not specified` 400 on async ingest | `RAG_SHARDING_ENABLED=1` but selector not derived | Verify commit `2b5129b` |
| `ingest_status` stuck at `queued` | `_update_doc_status` not transitioning | Verify commit `4a32141` |
| Image upload returns `<pad>` tokens | Gemma 4 vision_tower fp16 overflow | Verify `patches/vllm/gemma4_mm.py` mounted into vllm-chat |
| `<reasoning_token>` markers in chat | gemma4 parser missing | Verify `--reasoning-parser gemma4` + `--tool-call-parser gemma4` in vllm-chat |
| Token budget evicting too much | tokenizer mismatch (cl100k vs gemma-4) | Set `RAG_BUDGET_TOKENIZER=gemma-4` + model; verify HF cache mount |
| Env var set in `.env` but missing in container | not in `environment:` block | Add explicit mapping in `compose/docker-compose.yml` |
| RBAC revocation not propagating | pubsub down | Logs for `rbac subscriber: started`; falls back to 30s TTL |
| QU LLM unreachable; chat still works | bridge soft-fails to regex | `docker compose logs vllm-qu`; metrics `rag_qu_*` |

## 10. Plans / specs

Detailed plans under `docs/superpowers/plans/`:
- `2026-04-24-rag-robustness-and-quality.md` — Plan A (Phases 0–3, 29 tasks, 254 TDD steps)
- `2026-04-25-rag-plan-b-llm-shard-async.md` — Plan B (Phases 4–6: QU LLM + temporal sharding/RAPTOR + async + OCR)
- `docs/runbook/temporal-reshard-procedure.md` — Phase 5 cluster-mode runbook

When this file disagrees with code, **code wins**. Source of truth: `ext/`, `compose/`, `scripts/`.
