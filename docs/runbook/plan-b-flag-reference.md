# Plan B Flag Reference

Per-flag guidance for everything Plan B introduces. Lives alongside
`docs/runbook/flag-reference.md` (the master Plan A reference) — Plan B's
flags are appended there in summary form.

## Phase 4 — Query Understanding LLM

| Flag | Phase | Default | Description | Safe to toggle at runtime? |
|---|---|---|---|---|
| `RAG_QU_ENABLED` | 4.6 | `0` → `1` after Phase 4 gate | Master switch for the hybrid LLM router | Yes (restart open-webui) |
| `RAG_QU_URL` | 4.1 | `http://vllm-qu:8000/v1` | vLLM base URL | Yes (restart) |
| `RAG_QU_MODEL` | 4.1 | `qwen3-4b-qu` | served-model-name | Yes (restart) |
| `RAG_QU_LATENCY_BUDGET_MS` | 4.3 | `600` | Soft deadline; on miss the bridge falls back to regex | Yes (restart) |
| `RAG_QU_CACHE_ENABLED` | 4.5 | `1` | Redis cache for QU results | Yes (restart) |
| `RAG_QU_CACHE_TTL_SECS` | 4.5 | `300` | Cache TTL (s) | Yes (restart) |
| `RAG_QU_REDIS_DB` | 4.5 | `4` | Redis DB number — must be unique vs DB 3 (RBAC) | No (set once) |
| `RAG_QU_SHADOW_MODE` | 4.8 | `0` | Always run LLM, log both, route regex | Yes (restart) |

## Phase 5 — Temporal sharding (NOT YET SHIPPED)

The following flags are reserved for Phase 5. They are listed here for
completeness so operators reviewing this document don't add conflicting
custom env vars.

| Flag | Phase | Default | Description |
|---|---|---|---|
| `RAG_SHARDING_ENABLED` | 5.2 | `0` → `1` for new collections | Derive `shard_key="YYYY-MM"` at ingest |
| `RAG_TEMPORAL_LEVELS` | 5.6 | `0` → `1` for sharded collections | Inject L3 / L2 levels for global / evolution intents |
| `RAG_TIME_DECAY` | 5.7 | `0` | Apply `exp(-λΔt)` multiplier on current-state intent |
| `RAG_TIME_DECAY_LAMBDA_DAYS` | 5.7 | `90` | Half-life for time-decay (days) |
| `RAG_TIER_HOT_MONTHS` | 5.3 | `3` | Months kept in HNSW RAM tier |
| `RAG_TIER_WARM_MONTHS` | 5.3 | `12` | Months kept in mmap SSD tier |
| `RAG_TIER_COLD_QUANTIZATION` | 5.3 | `int8` | Cold tier scalar quantization |

## Phase 6 — Async ingest + OCR (NOT YET SHIPPED)

| Flag | Phase | Default | Description |
|---|---|---|---|
| `RAG_SYNC_INGEST` | 6.2 | `1` → `0` after soak | Sync ingest path; `0` = celery worker |
| `RAG_OCR_ENABLED` | 6.3 | `0` → `1` after verification | OCR fallback for scanned PDFs |
| `RAG_OCR_BACKEND` | 6.3 | `tesseract` | `tesseract` or `cloud:textract` or `cloud:document_ai` |
| `RAG_OCR_TRIGGER_CHARS` | 6.4 | `50` | <N chars per page → rasterize+OCR |
| `RAG_STRUCTURED_CHUNKER` | 6.5 | `0` → `1` after KB strategy | Tables/code as atomic units |
| `RAG_IMAGE_CAPTIONS` | 6.7 | `0` | Emit chunks with `chunk_type="image_caption"` |

## Plan B retires these Plan A flags

- **`RAG_INTENT_LLM`** — replaced by `RAG_QU_ENABLED`. The Plan A stub
  (`_llm_classify`, which raised `NotImplementedError` to fail loudly)
  is removed in Phase 4.4 along with the synchronous tiebreaker branch
  in `classify_with_reason`. The `intent_llm` key in
  `ext/services/kb_config.py:_KEY_TO_ENV` is deleted in Phase 4.10 —
  per-KB `rag_config` no longer accepts it.

- **`RAG_INTENT_ROUTING`** — replaced by the hybrid router. The legacy
  Tier-2 conditional in `_run_pipeline` stays in place during Phase 4 so
  the existing routing test contract holds; the cleanup commit lands in
  Plan C (or sooner if the Phase 4 shadow A/B shows the LLM intent label
  agrees with the regex Tier-2 label > 95% of the time, in which case
  the conditional becomes pure dead code).

## Carryover Plan A flags Plan B does NOT touch

- `RAG_HYDE`, `RAG_SEMCACHE`, `RAG_DISABLE_REWRITE` — still default-OFF;
  defer to Plan C.
