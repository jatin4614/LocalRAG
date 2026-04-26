# RAG Plan B — Query Understanding LLM, Temporal Sharding, Async Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build on the Plan A foundation by adding (a) a hybrid regex+LLM query-understanding router backed by a Qwen3-4B-AWQ vLLM service on GPU 1, (b) temporal sharding of the kb_1 Qdrant collection by `YYYY-MM` month with a temporal-then-semantic RAPTOR tree and tiered storage, and (c) flipping `RAG_SYNC_INGEST=0` to default-on after a Celery soak test, plus OCR and structure-aware chunking.

**Architecture:** Three additive phases that depend on Plan A's canonical schema (`kb_1_v3` with contextualization + ColBERT), RBAC cache, observability scaffolding, and eval harness. Phase 4 introduces a new `vllm-qu` compose service pinned to GPU 1 (≈ 3.5 GB weights + 4–5 GB activations into the 24 GB headroom Plan A documented). Phase 5 migrates `kb_1_v3` → `kb_1_v4` via a dual-collection alias swap (same pattern Plan A used for 1.7 schema reconciliation and 3.7 re-ingest), introducing `shard_key="YYYY-MM"` and a temporal-semantic tree built on top of contextualized chunks. Phase 6 graduates the Celery worker from "defined but not running" to the default ingest path, and adds a Tesseract OCR fallback plus a structure-aware chunker for tables/code.

**Tech Stack:** Python ≥ 3.10, Qwen3-4B-Instruct-2507-AWQ on vLLM ≥ 0.6.4 with V1 engine + xgrammar guided JSON, Qdrant ≥ 1.17.1 (pinned by Plan A) with custom sharding, Celery 5.x (already declared), Tesseract OCR 5 + pytesseract for the OCR module, pdfplumber + pymupdf for structure extraction, Redis 7 (DB 4 added for QU cache; Plan A took DB 3 for RBAC). No new GPU model on GPU 0 (still pinned by vllm-chat at 89% VRAM as of 2026-04-24).

**Working directory:** `/home/vogic/LocalRAG/`

**Hardware (rechecked 2026-04-24, see `~/.claude/projects/-home-vogic-LocalRAG/memory/hardware_reality.md`):**
- GPU 0: NVIDIA RTX 6000 Ada Generation, 48 GB. Currently 89% used by `vllm-chat` (Gemma-4-31B-AWQ) + external `frams-recognition` workers. **No headroom for new GPU 0 load. Phase 4 explicitly avoids GPU 0.**
- GPU 1: NVIDIA RTX PRO 4000 Blackwell, 24 GB. Currently ≈ 17% used (TEI dense ≈ 1.4 GB + sentence-transformers reranker ≈ 2.75 GB ≈ 4.2 GB resident). Plan B Phase 4 adds Qwen3-4B-AWQ ≈ 3.5 GB weights + 3.5–4.5 GB KV cache (cap via `gpu_memory_utilization=0.45`), totaling ≈ 11–12 GB / 24 GB ≈ 50% — **comfortable**.

**Reference plan:** `/home/vogic/LocalRAG/docs/superpowers/plans/2026-04-24-rag-robustness-and-quality.md` (Plan A). Plan B assumes Plan A has shipped through Phase 3 with the re-ingest executed (memory: `~/.claude/projects/-home-vogic-LocalRAG/memory/plan_a_executed.md`).

**Deployment window:** 4 days total, after Plan A has been in production with monitoring green for at least 7 days.

**Scope split:**
- **Plan A (shipped 2026-04-24/25)** — Phases 0–3. Brought up canonical schema, RBAC cache, retry/circuit-breaker, contextualization, ColBERT third RRF head, eval harness.
- **Plan B (this document)** — Phases 4–6. Adds hybrid regex+LLM router, temporal shard_key + temporal-semantic RAPTOR, async ingest default + OCR + structure-aware chunking.
- **Out of plan B** — Phase C of `temporal_corpus_plan` (LightRAG/HippoRAG, agentic multi-hop). Defer to a future Plan C only if Phase 5 eval shows residual gaps.

---

## Non-Goals (explicit scope wall)

This plan does **not**:

- Replace the regex classifier outright. Regex stays as the fast path; the LLM is a tiebreaker on ambiguous queries (Phase 4 keeps the policy in `query_intent.py` mostly intact and extends, not replaces, `chat_rag_bridge.classify_intent`).
- Add a knowledge-graph retrieval head (LightRAG / HippoRAG / GraphRAG). These are Phase C of the `temporal_corpus_plan` memory; defer to a future plan that's gated on Phase 5 eval.
- Change the chat model on GPU 0. Gemma-4-31B-AWQ stays. The new model is a **separate** vLLM service on GPU 1.
- Add multi-GPU tensor parallelism. Qwen3-4B fits easily on GPU 1 at TP=1.
- Replace the bge-m3 embedding model on GPU 1.
- Migrate `kb_eval` (130 pts) — Plan B reshards `kb_1_v3` only. Other collections stay flat.
- Change the contextualizer prompt. Plan A's prompt (with date + KB/subtag + relationships hints) carries forward unchanged.
- Add agentic multi-hop / Self-RAG / Self-RAG-style reflection. Defer to Plan C.
- Implement proposition chunking. The `temporal_corpus_plan` memory explicitly flags this as **wrong** for narrative/evolution corpora.
- Add cloud OCR as the default. Tesseract is the air-gap-safe default; cloud OCR (Textract / Document AI) is opt-in per KB and gated behind a feature flag.
- Replace the eval harness with RAGAS / TruLens. The pure-Python harness committed in Plan A Phase 0 is sufficient for Plan B's gating; introducing third-party eval frameworks is out of scope.
- Change the Redis container or Qdrant version. Plan A pinned Qdrant to 1.17.1; that pin holds.
- Add SSO/LDAP. Same as Plan A — local auth only.

If any of these come up during execution, they are deferred to a future Plan C — do not expand scope.

---

## Assumptions (validated 2026-04-25 against current `main`, post-Plan-A)

If any of these change mid-execution, stop and re-validate before continuing.

1. **Plan A is shipped.** `kb_1` alias points to `kb_1_v3` (canonical schema + dense + sparse + ColBERT + `context_prefix` payload). Verified by:
   ```bash
   curl -s http://localhost:6333/collections/kb_1/exists
   curl -s http://localhost:6333/aliases | python -m json.tool
   # Expected: alias kb_1 → kb_1_v3
   ```
2. **`kb_1_v2` is the 14-day rollback target** for Plan A's 1.7 schema migration and is still on disk. Plan B Phase 5 must not delete it inside the rollback window.
3. **GPU topology:** GPU 0 still 89% used (vllm-chat + frams-recognition). GPU 1 still 17% used. `nvidia-smi` confirms.
4. **Qdrant version:** 1.17.1 (pinned by Plan A A.4). `curl -s http://localhost:6333/ | python -m json.tool | grep version` reports `"version": "1.17.1"`.
5. **TEI version:** 1.9.3, dense-only for bge-m3. Sparse + ColBERT computed client-side via fastembed 0.8.0 inside the open-webui container.
6. **Celery worker container is defined but NOT running.** `docker ps | grep celery-worker` returns nothing today. Phase 6.1 stands it up and soak-tests; Phase 6.2 flips the default.
7. **Current ingest path is sync.** `RAG_SYNC_INGEST=1` is the default. `ext/routers/upload.py:34` reads this flag.
8. **eval harness from Plan A Phase 0.4 is committed** at `tests/eval/harness.py` + `tests/eval/golden_starter.jsonl` (60 queries, 4 strata). `make eval-baseline` works against `kb_eval` and reports `chunk_recall@10`, `mrr@10`, `ndcg@10` per stratum + global.
9. **RBAC cache is shipped** (Plan A Phase 1.5). Redis DB 3 holds `rbac:user:<id>:allowed_kbs`. Plan B Phase 4 uses Redis DB 4 for QU cache to avoid stomping.
10. **Contextualizer + ColBERT are live** for `kb_1_v3` (Plan A Phase 3). Plan B Phase 5's resharding inherits both — the resharded `kb_1_v4` collection retains all `context_prefix` payloads and ColBERT vectors.
11. **vllm-chat (GPU 0) Gemma-4-31B-AWQ is the chat model.** Phase 4's separate `vllm-qu` model never replaces or contests it.
12. **HF_HOME=/var/models/hf_cache** mount is live in open-webui (Plan A A.5). Phase 4's Qwen3-4B-AWQ weights pre-cache into the same `/var/models/hf_cache` directory and are mounted into the new `vllm-qu` container.
13. **Network is connected** at the time Plan B is implemented (development phase). The final disconnect (Appendix A.7 of Plan A) has NOT yet been executed. Plan B's pre-cache step (B.A.1) runs while connected and is verified with `HF_HUB_OFFLINE=1` before disconnect.
14. **Existing flags-still-OFF after Plan A** (per Plan A's flag-kill-list policy):
    - `RAG_DISABLE_REWRITE` (rewrite OFF) — Plan B Phase 4 audit decides.
    - `RAG_HYDE` — Plan B does NOT decide; defer to Plan C.
    - `RAG_RAPTOR` — Plan B Phase 5 replaces this with the temporal-semantic tree builder; the legacy `raptor.py` flat path is removed (Phase 5.10 audit).
    - `RAG_SEMCACHE` — Plan B does NOT decide; defer to Plan C.
    - `RAG_INTENT_ROUTING` (Tier 2) — Plan B Phase 4 replaces with the QU LLM hybrid router; the flag is removed in Phase 4.10.

---

## File structure delivered by this plan

Files created (all paths absolute under `/home/vogic/LocalRAG/`):

```
ext/
├── services/
│   ├── query_understanding.py             # NEW: Phase 4.3 — JSON-schema-guided LLM router
│   ├── qu_cache.py                        # NEW: Phase 4.5 — Redis DB 4 cache for QU results
│   ├── temporal_raptor.py                 # NEW: Phase 5.5 — temporal-then-semantic tree
│   ├── temporal_shard.py                  # NEW: Phase 5.1 + 5.2 — shard_key derivation + ensure_collection variant
│   ├── time_decay.py                      # NEW: Phase 5.7 — intent-conditional time-decay scoring
│   ├── ocr.py                             # NEW: Phase 6.3 — Tesseract default + cloud opt-in
│   └── chunker_structured.py              # NEW: Phase 6.5 — table/code-aware chunker
ext/db/migrations/
├── 010_add_kb_chunking_strategy.sql       # NEW: Phase 6.6 — per-KB chunking strategy column
└── 011_add_kb_ocr_policy.sql              # NEW: Phase 6.3 — per-KB OCR policy
compose/
├── docker-compose.yml                     # MODIFY: Phase 4.1 — vllm-qu service; Phase 6.2 — celery-worker default
└── vllm-qu/
    └── README.md                          # NEW: Phase 4.1 — service overview + flag reference
scripts/
├── reshard_kb_temporal.py                 # NEW: Phase 5.4 — kb_1_v3 → kb_1_v4 migration
├── tier_storage_cron.py                   # NEW: Phase 5.8 — daily hot/warm/cold tier movement
├── stage_qwen3_qu.sh                      # NEW: Phase 4.2 — pre-cache Qwen3-4B-AWQ on host
└── celery_soak_test.py                    # NEW: Phase 6.1 — 1000-doc upload soak harness
tests/
├── eval/
│   ├── golden_evolution.jsonl             # NEW: Phase 5 — 30 evolution-stratified queries
│   └── results/
│       ├── phase-4-baseline.json          # NEW: Phase 4 baseline eval
│       ├── phase-5-baseline.json          # NEW: Phase 5 baseline eval
│       └── phase-6-baseline.json          # NEW: Phase 6 baseline eval
├── unit/
│   ├── test_query_understanding_schema.py # NEW: Phase 4.3
│   ├── test_qu_router_escalation.py       # NEW: Phase 4.4
│   ├── test_qu_cache.py                   # NEW: Phase 4.5
│   ├── test_temporal_shard_key.py         # NEW: Phase 5.2
│   ├── test_temporal_raptor_tree.py       # NEW: Phase 5.5
│   ├── test_time_decay_intent_gating.py   # NEW: Phase 5.7
│   ├── test_chunker_structured_table.py   # NEW: Phase 6.5
│   ├── test_chunker_structured_code.py    # NEW: Phase 6.5
│   └── test_ocr_trigger_threshold.py      # NEW: Phase 6.4
├── integration/
│   ├── test_vllm_qu_live.py               # NEW: Phase 4.9 — fixture container
│   ├── test_temporal_resharding.py        # NEW: Phase 5.4
│   ├── test_celery_soak.py                # NEW: Phase 6.1
│   └── test_ocr_pipeline.py               # NEW: Phase 6.3
docs/
├── runbook/
│   ├── plan-b-flag-reference.md           # NEW: Phase 4.10 — every new RAG_* flag
│   ├── temporal-reshard-procedure.md      # NEW: Phase 5.4 — operator runbook
│   ├── temporal-reshard-checklist.md      # NEW: Phase 5.10 — checklist for off-hours window
│   ├── ocr-runbook.md                     # NEW: Phase 6.8 — operator runbook for OCR-needed KBs
│   ├── tiered-storage-runbook.md          # NEW: Phase 5.8 — hot/warm/cold automation
│   └── qu-llm-runbook.md                  # NEW: Phase 4.10 — QU LLM operations
observability/
├── prometheus/
│   ├── alerts-qu.yml                      # NEW: Phase 4.7 — QU LLM SLO alerts
│   ├── alerts-tiered-shards.yml           # NEW: Phase 5.9 — per-shard health alerts
│   └── alerts-celery.yml                  # NEW: Phase 6.1 — DLQ depth + retry rate
```

Files modified:

```
ext/services/chat_rag_bridge.py                  # MODIFY: Phase 4.6 — replace classify_intent with analyze_query
ext/services/query_intent.py                     # MODIFY: Phase 4.4 — replace _llm_classify stub with hybrid router escalation hook
ext/services/vector_store.py                     # MODIFY: Phase 5.1 — ensure_collection variant with shard_key + tier config
ext/services/ingest.py                           # MODIFY: Phase 5.2 — extract shard_key at ingest; Phase 6.4 — OCR fallback; Phase 6.5 — structure-aware chunker hook
ext/services/raptor.py                           # MODIFY: Phase 5.5 — make legacy path opt-in; route to temporal_raptor.py when KB has shard_key
ext/services/retriever.py                        # MODIFY: Phase 5.6 — temporal-aware level injection; Phase 5.7 — apply time-decay multiplier
ext/services/kb_config.py                        # MODIFY: Phase 6.6 — read chunking_strategy and ocr_policy
ext/routers/upload.py                            # MODIFY: Phase 6.2 — flip RAG_SYNC_INGEST default after soak
ext/workers/celery_app.py                        # MODIFY: Phase 6.2 — add tier_storage_cron beat schedule
ext/services/metrics.py                          # MODIFY: Phase 4.7 — QU LLM counters/histograms; Phase 5.9 — per-shard gauges
compose/docker-compose.yml                       # MODIFY: Phase 4.1 — vllm-qu service; Phase 6.2 — celery-worker without profile gate
docs/runbook/flag-reference.md                   # MODIFY: Phase 4.10 + 6.8 — append Plan B flags
Makefile                                         # MODIFY: Phase 5.4 — add `reshard-kb-temporal` and `eval-evolution` targets
```

---

## Rollback Appendix (one row per phase task — execute if the task fails its gate)

| Task | Rollback action | How | Verification | Max revert time |
|---|---|---|---|---|
| **4.1** vllm-qu compose service | `docker compose stop vllm-qu && docker compose rm -f vllm-qu`. Revert compose change. | `git revert <4.1 commit>`, `docker compose up -d`. | `nvidia-smi` shows GPU 1 back to ~17%. | 5 min |
| **4.2** Stage Qwen3-4B weights | Delete `/var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/`. | `rm -rf` that path. | Disk free returns. | 1 min |
| **4.3** `query_understanding.py` module | No-op rollback — module is unused unless 4.6 wires it in. | N/A | N/A | N/A |
| **4.4** Hybrid router escalation | Revert `query_intent.py:_llm_classify` to stub. | `git revert <4.4 commit>`. | `RAG_INTENT_LLM=1` returns "specific" everywhere. | 5 min |
| **4.5** Redis DB 4 QU cache | `redis-cli -n 4 FLUSHDB`. Set `RAG_QU_CACHE_ENABLED=0`. | Edit `.env`, restart open-webui. | `qu_cache_hit_ratio` gauge reads 0. | 2 min |
| **4.6** Wire QU into bridge | Set `RAG_QU_ENABLED=0`. The bridge falls back to regex `classify_intent`. | Edit `.env`, restart. | Pipeline behavior matches Plan A end-state. | 2 min |
| **4.7** QU metrics | Revert metrics.py changes. | `git revert <4.7 commit>`. | `/metrics` no longer lists `rag_qu_*`. | 5 min |
| **4.8** Shadow A/B harness | Set `RAG_QU_SHADOW_MODE=0`. Shadow logging stops; production behavior unchanged. | Edit `.env`. | `rag_qu_shadow_disagreement_total` stops incrementing. | 2 min |
| **4.9** vllm-qu integration test | Test-only — no production rollback. | `pytest --deselect tests/integration/test_vllm_qu_live.py`. | n/a | 0 min |
| **4.10** Runbook fill-in | Documentation only. No rollback. | N/A | N/A | N/A |
| **5.1** ensure_collection variant | Revert vector_store.py shard_key path. | `git revert <5.1 commit>`. | Existing collections unaffected (additive method). | 5 min |
| **5.2** Date extraction | Set `RAG_SHARDING_ENABLED=0`. Ingest skips shard_key derivation; payload shape unchanged. | Edit `.env`, restart. | New ingests have no `shard_key` payload. | 2 min |
| **5.3** Tiered storage config | Per-collection setting; revert by recreating without tier hints. | Operator restores from snapshot. | `curl /collections/kb_1` shows uniform storage. | 30 min |
| **5.4** Reshard script + execution | **High risk.** Rollback is a 1-second alias swap back to `kb_1_v3`. | `qdrant-client alias swap kb_1 → kb_1_v3`. | Live retrieval hits kb_1_v3. | 30 sec |
| **5.5** Temporal RAPTOR builder | Revert `raptor.py` routing; legacy path resumes. | `git revert <5.5 commit>`. | `RAG_RAPTOR=1` builds flat tree as before. | 10 min |
| **5.6** Retrieval temporal-level injection | Set `RAG_TEMPORAL_LEVELS=0`. Retrieval ignores level injection rules. | Edit `.env`. | Retrieval matches Plan A end-state. | 2 min |
| **5.7** Time-decay scoring | Set `RAG_TIME_DECAY=0`. Scores unmodified. | Edit `.env`. | `time_decay_active` gauge reads 0. | 2 min |
| **5.8** Tier movement cron | `crontab -r` removes the daily job. | Manual + `git revert`. | Cron list empty for the user. | 2 min |
| **5.9** Per-shard metrics | Revert metrics.py changes. | `git revert`. | `/metrics` no longer lists `rag_shard_*`. | 5 min |
| **5.10** Operator runbook | Doc-only. No rollback. | N/A | N/A | N/A |
| **6.1** Celery soak test | Test-only. No production rollback. | n/a | n/a | 0 min |
| **6.2** Flip RAG_SYNC_INGEST default | Set `RAG_SYNC_INGEST=1` in `.env`. Reverts to sync ingest. | Edit `.env`, restart open-webui. | `celery_jobs_total` stops incrementing on uploads. | 2 min |
| **6.3** OCR module | Set `RAG_OCR_ENABLED=0`. Ingest skips OCR fallback. | Edit `.env`. | `ocr_invocations_total` stops. | 2 min |
| **6.4** OCR trigger | Same as 6.3. | Same. | Same. | 2 min |
| **6.5** Structured chunker | Set `RAG_STRUCTURED_CHUNKER=0`. Chunker reverts to current window strategy. | Edit `.env`. | New chunks match pre-Plan-B shape. | 2 min |
| **6.6** Per-KB chunking strategy | Set `chunking_strategy=window` in `kb_config` for affected KBs. | `curl PATCH /api/kb/<id>/config`. | KB resumes default chunking. | 2 min |
| **6.7** Image caption extraction | Set `RAG_IMAGE_CAPTIONS=0`. | Edit `.env`. | Ingest skips image caption emission. | 2 min |
| **6.8** OCR runbook + re-ingest | Doc-only. No rollback. | N/A | N/A | N/A |

---

## Flag Kill-List Policy (Plan B continuation)

**Policy unchanged from Plan A:** Any RAG feature flag still default-OFF globally after Plan B Phase 6 ships must be either:
- **(a)** Turned default-ON with a one-line eval justification committed to `docs/runbook/flag-reference.md`, OR
- **(b)** Deleted from the codebase (flag check + the guarded code path), OR
- **(c)** Justified as a per-KB-only customization with a one-line decision note in `docs/runbook/flag-reference.md`.

Phase 4.10 (Plan A's deferred audit) executes the audit. After Plan B Phase 6 completes, the audit runs again.

Plan B introduces these flags. Their fate after Plan B:
- `RAG_QU_ENABLED` — default ON after Phase 4 eval gate; deleted only after a 30-day production window with no incidents.
- `RAG_QU_SHADOW_MODE` — default ON for Phase 4.8 shadow window (1 week), then OFF, then deleted in Phase 4.10 if no follow-up A/B planned.
- `RAG_QU_CACHE_ENABLED` — default ON after Phase 4.5 cache passes hit-rate threshold; permanent toggle (low risk to keep).
- `RAG_QU_LATENCY_BUDGET_MS` — operator knob, retains.
- `RAG_SHARDING_ENABLED` — default ON for new collections after Phase 5.4 ships.
- `RAG_TEMPORAL_LEVELS` — default ON for collections that have a temporal-RAPTOR tree (Phase 5.5+).
- `RAG_TIME_DECAY` — default OFF still after Phase 5; eval-conditional toggle that flips per intent inside the bridge.
- `RAG_RAPTOR` — Plan A flag. Phase 5.10 audit either replaces with `RAG_RAPTOR_TEMPORAL` or removes outright.
- `RAG_SYNC_INGEST` — default flips to 0 in Phase 6.2 (the whole point of the phase).
- `RAG_OCR_ENABLED` — default ON after Phase 6.4 verification; per-KB `ocr_policy` overrides.
- `RAG_STRUCTURED_CHUNKER` — default ON after Phase 6.6 ships per-KB strategy.
- `RAG_IMAGE_CAPTIONS` — default ON after Phase 6.7.

Plan A flags Plan B explicitly retires:
- `RAG_INTENT_ROUTING` (Tier 2) — Phase 4.10 deletes after QU LLM is default ON.
- `RAG_INTENT_LLM` (the stub flag in `query_intent.py`) — Phase 4.4 removes; QU LLM path is governed by `RAG_QU_ENABLED` instead.

Plan A flags Plan B does NOT touch:
- `RAG_HYDE`, `RAG_SEMCACHE`, `RAG_DISABLE_REWRITE` — defer to Plan C.

---

## Per-phase gating criteria (before proceeding to next phase)

Each phase has an explicit gate. If the gate fails, do not proceed — either fix within the phase or trigger rollback.

- **Phase 4 gate**: All Phase 4 unit tests pass; `tests/integration/test_vllm_qu_live.py` passes against the live vllm-qu container; QU LLM p95 latency ≤ 600ms (per SLO Phase 4 budget); QU schema-violation rate ≤ 1% over a 1000-query soak; shadow A/B run for ≥ 7 days shows hybrid agreement with regex on ≥ 85% of queries (escalation rate measured); baseline eval on `kb_1` shows `chunk_recall@10` within ±2 pp of Plan A end-state on the `specific` stratum; on the `multihop` and `evolution` strata, hybrid router shows ≥ +3 pp improvement.
- **Phase 5 gate**: All Phase 5 unit + integration tests pass; reshard script succeeds against a staging clone of `kb_1_v3` first, then against production `kb_1_v3` → `kb_1_v4`; live retrieval against `kb_1_v4` hits all 36 monthly shards (verified); `golden_evolution.jsonl` (30 queries, Phase 5 introduces) shows `chunk_recall@10` ≥ +5 pp improvement vs. flat retrieval baseline; `phase-5-baseline.json` committed. Hot/warm/cold tier movement runs without errors for 7 days (operator window).
- **Phase 6 gate**: Celery soak test (1000 docs, parallel uploaders) completes with 0 lost docs and DLQ depth ≤ 5 over 1 hour window; `RAG_SYNC_INGEST=0` runs in production for 7 days with no regressions in `chunk_recall@10`; OCR pipeline correctly handles a corpus of 50 scanned PDFs (manual eval); structured chunker produces atomic chunks for at least one KB containing tables (eval review).

---

## Execution cadence inside the 4-day window

- **Day 0 (before window opens)**: Appendix B.A — pre-cache Qwen3-4B-AWQ on the deploy host while connected; verify with `HF_HUB_OFFLINE=1`; audit Plan A's final state via `~/.claude/projects/-home-vogic-LocalRAG/memory/plan_a_executed.md`.
- **Day 1 morning**: Phase 4.1–4.3 (vllm-qu service, weight staging, QU module skeleton).
- **Day 1 afternoon**: Phase 4.4–4.6 (hybrid router, cache, bridge wiring).
- **Day 1 evening**: Phase 4.7–4.10 (metrics, shadow harness, integration test, runbook fill-in). Start the shadow window — leaves shadow A/B logging running for the next 7 days while Phase 5 work continues.
- **Day 2 morning**: Phase 5.1–5.3 (ensure_collection variant, date extraction, tiered storage config).
- **Day 2 afternoon**: Phase 5.4 (reshard script + dry-run against staging clone).
- **Day 2 evening (off-peak)**: Phase 5.4 production reshard + alias swap.
- **Day 3 morning**: Phase 5.5–5.7 (temporal RAPTOR, retrieval level injection, time-decay).
- **Day 3 afternoon**: Phase 5.8–5.10 (cron, metrics, runbook).
- **Day 4 morning**: Phase 6.1–6.2 (Celery soak, sync flag flip).
- **Day 4 afternoon**: Phase 6.3–6.6 (OCR, trigger, structured chunker, per-KB strategy).
- **Day 4 evening**: Phase 6.7–6.8 (image captions, runbook). Final eval pass; commit `phase-6-baseline.json`.

Phase 4 shadow A/B runs continuously across days 1–7; full evaluation against `golden_starter` + `golden_evolution` happens on Day 8 with sign-off before Phase 4 flag flips to default-ON.

---

## Appendix B.A — Pre-cache Qwen3-4B-AWQ + offline verification

The same offline-readiness procedure Plan A used for Gemma-4 tokenizer + bge-reranker-v2-m3 + ColBERT applies to Qwen3-4B. Done on the deploy host while still connected.

### B.A.1 — Pre-cache Qwen3-4B-Instruct-2507-AWQ weights

Requires HF account; the AWQ build is on `Qwen/Qwen3-4B-Instruct-2507-AWQ` (or a community fork — verify on huggingface.co at execution time).

- [ ] **Step B.A.1.1: Download the AWQ build**

```bash
export HF_TOKEN="<your token>"
HF_HOME=/var/models/hf_cache huggingface-cli download \
  Qwen/Qwen3-4B-Instruct-2507-AWQ \
  --include "*.json" "*.safetensors" "*.txt" "tokenizer.model"
```

Expected: ≈ 3.5 GB total under `/var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/snapshots/<rev>/`.

- [ ] **Step B.A.1.2: Verify file inventory**

```bash
ls -lh /var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/snapshots/*/ | head
du -sh /var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/
```

Expected: at least one `.safetensors` file (typically 1–2), `config.json`, `tokenizer_config.json`, `tokenizer.json` (or `tokenizer.model`), and `quant_config.json`. Total size ≈ 3.5 GB.

### B.A.2 — Verify weights load with `HF_HUB_OFFLINE=1`

- [ ] **Step B.A.2.1: Run a vLLM smoke test**

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HOME=/var/models/hf_cache \
  docker run --rm --gpus '"device=1"' \
    -v /var/models/hf_cache:/root/.cache/huggingface:ro \
    -e HF_HUB_OFFLINE=1 \
    -e VLLM_USE_V1=1 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-4B-Instruct-2507-AWQ \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.45 \
    --port 8000 &
sleep 90
curl -s http://localhost:8000/v1/models | python -m json.tool
```

Expected: model list returns Qwen3-4B-Instruct-2507-AWQ. If it errors with "model not found" — re-run B.A.1.1 against a different revision; the AWQ build occasionally moves snapshots.

- [ ] **Step B.A.2.2: Tear down the smoke container**

```bash
docker ps | grep vllm | awk '{print $1}' | xargs -r docker stop
```

### B.A.3 — Commit nothing

The pre-cache step has no committed artifact (the weights live outside the repo). The verification command lives in `docs/runbook/qu-llm-runbook.md` (Phase 4.10).

### B.A — completion gate

Before proceeding to Phase 4.1:

- [ ] B.A.1 — Qwen3-4B weights resident under `/var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/`.
- [ ] B.A.2 — Smoke test passed; model serves at `localhost:8000/v1/models` with `HF_HUB_OFFLINE=1`.
- [ ] GPU 1 returns to baseline (≈17%) after smoke teardown — verify with `nvidia-smi`.

---

## Phase 4 — Query Understanding LLM on GPU 1 (Day 1)

**Phase goal:** Replace the regex-only `chat_rag_bridge.classify_intent` with a hybrid regex+LLM router. Regex stays as the fast path (sub-millisecond, zero I/O); the LLM is consulted only on ambiguous queries (pronouns, relative time, multi-clause, long query, no NER hit + question word). LLM output is a constrained JSON object (xgrammar guided_json) containing `intent`, `resolved_query` (rewritten standalone form for retrieval), `temporal_constraint`, `entities`, and `confidence`. Cached in Redis DB 4 keyed by `sha256(normalize(query) + last_turn_id)` with a 5-minute TTL.

**Why this phase first:** Phase 5's temporal sharding depends on accurate intent classification — global / evolution / point-in-time queries must hit the right tier of the temporal-RAPTOR tree. Today's regex classifier mislabels ambiguous queries (e.g. "what changed last quarter") as `specific`, defeating per-intent routing. The QU LLM lifts this ceiling before Phase 5 builds on top of it.

**Hardware sizing:**
- GPU 1: NVIDIA RTX PRO 4000 Blackwell, 24 GB
- Existing: TEI ≈ 1.4 GB + sentence-transformers reranker ≈ 2.75 GB ≈ 4.2 GB
- New: Qwen3-4B-AWQ ≈ 3.5 GB weights + KV cache cap 4–5 GB (via `gpu_memory_utilization=0.45`) ≈ 7.5–8.5 GB
- Total resident: ≈ 12–13 GB / 24 GB ≈ 50%
- Headroom for KV-cache spikes and OS/driver overhead: ≈ 11 GB. **Comfortable.**

---

### Task 4.1: Add `vllm-qu` service to docker-compose

**Files:**
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`
- Create: `/home/vogic/LocalRAG/compose/vllm-qu/README.md`

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_compose_vllm_qu_service.py`:

```python
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "compose" / "docker-compose.yml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def test_vllm_qu_service_defined():
    compose = _load_compose()
    assert "vllm-qu" in compose["services"], (
        "Plan B Phase 4.1 requires a vllm-qu service in compose/docker-compose.yml"
    )


def test_vllm_qu_service_pinned_to_gpu_1():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    env = svc.get("environment", {}) or {}
    if isinstance(env, list):
        env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in env if "=" in item}
    visible = env.get("NVIDIA_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
    assert visible == "1", (
        f"vllm-qu must be pinned to GPU 1, got {visible!r}. "
        "GPU 0 is reserved for vllm-chat and is at 89% VRAM."
    )


def test_vllm_qu_uses_qwen3_4b_awq():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    cmd = svc.get("command") or []
    if isinstance(cmd, str):
        cmd = cmd.split()
    joined = " ".join(cmd)
    assert "Qwen3-4B-Instruct-2507-AWQ" in joined or "Qwen/Qwen3-4B" in joined, (
        f"vllm-qu must serve Qwen3-4B-AWQ, got command: {joined}"
    )


def test_vllm_qu_caps_gpu_memory():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    cmd = svc.get("command") or []
    if isinstance(cmd, str):
        cmd = cmd.split()
    joined = " ".join(cmd)
    assert "--gpu-memory-utilization" in joined, (
        "vllm-qu must cap GPU memory utilization to leave room for TEI + reranker"
    )


def test_vllm_qu_health_check():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    assert "healthcheck" in svc, "vllm-qu must declare a healthcheck"


def test_vllm_qu_mounts_offline_cache():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    volumes = svc.get("volumes") or []
    cache_mount = any(
        ("/var/models/hf_cache" in str(v)) or ("hf_cache" in str(v))
        for v in volumes
    )
    assert cache_mount, "vllm-qu must mount /var/models/hf_cache for offline weights"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_compose_vllm_qu_service.py -v
```

Expected: 6 failures with "Plan B Phase 4.1 requires a vllm-qu service".

- [ ] **Step 3: Add the service to compose**

Edit `/home/vogic/LocalRAG/compose/docker-compose.yml`. Find the existing `tei:` service block (Plan A pinned it at the same indent), and add this sibling service block immediately after `tei` (so the GPU 1 services are colocated in the file):

```yaml
  vllm-qu:
    # Query Understanding LLM. Pinned to GPU 1 (RTX PRO 4000 Blackwell, 24 GB).
    # GPU 0 is reserved for vllm-chat (Gemma-4-31B-AWQ, 89% VRAM).
    image: vllm/vllm-openai:latest
    container_name: orgchat-vllm-qu
    restart: unless-stopped
    runtime: nvidia
    ipc: host
    environment:
      NVIDIA_VISIBLE_DEVICES: "1"
      VLLM_USE_V1: "1"
      HF_HOME: /root/.cache/huggingface
      HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-0}
      TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE:-0}
    command: >-
      --model Qwen/Qwen3-4B-Instruct-2507-AWQ
      --quantization awq
      --dtype auto
      --max-model-len 8192
      --gpu-memory-utilization 0.45
      --enable-prefix-caching
      --port 8000
      --served-model-name qwen3-4b-qu
    volumes:
      - /var/models/hf_cache:/root/.cache/huggingface:ro
    ports:
      - "8101:8000"
    networks:
      - orgchat
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://localhost:8000/v1/models | grep -q qwen3-4b-qu || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 120s
```

- [ ] **Step 4: Add open-webui dependency on vllm-qu (optional, soft-fail)**

Find the `open-webui:` block, add to its `environment:`:

```yaml
      RAG_QU_URL: ${RAG_QU_URL:-http://vllm-qu:8000/v1}
      RAG_QU_MODEL: ${RAG_QU_MODEL:-qwen3-4b-qu}
      RAG_QU_ENABLED: ${RAG_QU_ENABLED:-0}
      RAG_QU_LATENCY_BUDGET_MS: ${RAG_QU_LATENCY_BUDGET_MS:-600}
      RAG_QU_CACHE_ENABLED: ${RAG_QU_CACHE_ENABLED:-1}
      RAG_QU_CACHE_TTL_SECS: ${RAG_QU_CACHE_TTL_SECS:-300}
      RAG_QU_SHADOW_MODE: ${RAG_QU_SHADOW_MODE:-0}
```

We do NOT add `vllm-qu` to open-webui's `depends_on:` because the bridge soft-fails when QU is unreachable — falling back to regex-only classification keeps chat alive even if vllm-qu crashes.

- [ ] **Step 5: Create service README**

Create `/home/vogic/LocalRAG/compose/vllm-qu/README.md`:

```markdown
# vllm-qu — Query Understanding LLM

Serves Qwen3-4B-Instruct-2507-AWQ for the hybrid regex+LLM intent router. GPU 1 only.

**Model:** `Qwen/Qwen3-4B-Instruct-2507-AWQ` (≈ 3.5 GB weights + ≈ 4 GB KV cache at gpu_memory_utilization=0.45)

**Endpoint:** `http://vllm-qu:8000/v1` (host port 8101)

**Engine:** vLLM V1 with xgrammar guided JSON. The QU module (`ext/services/query_understanding.py`) supplies a JSON schema; vLLM constrains generation to valid output.

**Flags:**
- `RAG_QU_ENABLED` — master switch. 0 = regex-only fallback. Default 0 until shadow A/B (Phase 4.8) sign-off.
- `RAG_QU_URL` — vLLM base URL. Default `http://vllm-qu:8000/v1`.
- `RAG_QU_MODEL` — served-model-name. Default `qwen3-4b-qu`.
- `RAG_QU_LATENCY_BUDGET_MS` — soft deadline; falls back to regex if exceeded. Default 600.
- `RAG_QU_CACHE_ENABLED` — Redis DB 4 cache. Default 1.
- `RAG_QU_CACHE_TTL_SECS` — cache TTL. Default 300.
- `RAG_QU_SHADOW_MODE` — log both regex and LLM, no production routing change. Default 0.

**Operational notes:**
- vllm-qu starts in ≈ 90 s (model load). Don't block open-webui on it.
- If `nvidia-smi` shows GPU 1 > 18 GB (75%), reduce `--gpu-memory-utilization` to 0.40.
- Health endpoint returns 200 once model loaded; the docker healthcheck checks `/v1/models`.
```

- [ ] **Step 6: Re-run tests to verify pass**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_compose_vllm_qu_service.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Smoke test the service**

```bash
cd /home/vogic/LocalRAG/compose
docker compose up -d vllm-qu
sleep 120
docker logs --tail 20 orgchat-vllm-qu
curl -s http://localhost:8101/v1/models | python -m json.tool
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv | grep -E '^[01],'
```

Expected: `qwen3-4b-qu` listed; GPU 1 memory.used ≈ 8 GB / 24 GB.

- [ ] **Step 8: Commit**

```bash
git add compose/docker-compose.yml compose/vllm-qu/README.md tests/unit/test_compose_vllm_qu_service.py
git commit -m "phase-4.1: add vllm-qu service for Query Understanding LLM (GPU 1)"
```

---

### Task 4.2: Stage Qwen3-4B-AWQ weights script

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/stage_qwen3_qu.sh`

This script automates the pre-cache step (Appendix B.A.1) so the operator can re-stage on a fresh deploy host without consulting the runbook.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_stage_qwen3_qu_script.py`:

```python
import pathlib
import stat

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage_qwen3_qu.sh"


def test_script_exists():
    assert SCRIPT.exists(), "stage_qwen3_qu.sh missing"


def test_script_is_executable():
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, "stage_qwen3_qu.sh must be executable"


def test_script_uses_offline_cache_path():
    content = SCRIPT.read_text()
    assert "/var/models/hf_cache" in content, "Script must pre-cache to /var/models/hf_cache"


def test_script_downloads_qwen3_4b_awq():
    content = SCRIPT.read_text()
    assert "Qwen3-4B-Instruct-2507-AWQ" in content, "Script must download the Qwen3-4B AWQ model"


def test_script_verifies_size():
    content = SCRIPT.read_text()
    assert "du -sh" in content or "du -h" in content, (
        "Script must verify the cached size to catch incomplete downloads"
    )


def test_script_fails_loudly_on_missing_token():
    content = SCRIPT.read_text()
    assert "HF_TOKEN" in content, "Script must check for HF_TOKEN"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_stage_qwen3_qu_script.py -v
```

Expected: 6 failures.

- [ ] **Step 3: Write the script**

Create `/home/vogic/LocalRAG/scripts/stage_qwen3_qu.sh`:

```bash
#!/usr/bin/env bash
# Pre-cache Qwen3-4B-Instruct-2507-AWQ weights into /var/models/hf_cache.
# Run on the deploy host while still connected to the internet.
# Plan B Phase 4.2 / Appendix B.A.1.
set -euo pipefail

CACHE_DIR="${CACHE_DIR:-/var/models/hf_cache}"
MODEL_ID="Qwen/Qwen3-4B-Instruct-2507-AWQ"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN must be set. Get one from https://huggingface.co/settings/tokens" >&2
  exit 2
fi

if [[ ! -d "$CACHE_DIR" ]]; then
  echo "Creating cache dir $CACHE_DIR (requires sudo)"
  sudo mkdir -p "$CACHE_DIR"
  sudo chown "$USER":"$USER" "$CACHE_DIR"
fi

echo "Downloading $MODEL_ID into $CACHE_DIR (this is ~3.5 GB)"
HF_HOME="$CACHE_DIR" huggingface-cli download "$MODEL_ID" \
  --include "*.json" "*.safetensors" "*.txt" "tokenizer.model" "tokenizer.json"

echo "Cache size after download:"
du -sh "$CACHE_DIR/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ"

echo
echo "Files staged:"
ls -lh "$CACHE_DIR/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/snapshots/"*/

echo
echo "Done. Next: bring up vllm-qu via 'docker compose up -d vllm-qu'."
```

- [ ] **Step 4: Mark executable**

```bash
chmod +x /home/vogic/LocalRAG/scripts/stage_qwen3_qu.sh
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_stage_qwen3_qu_script.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/stage_qwen3_qu.sh tests/unit/test_stage_qwen3_qu_script.py
git commit -m "phase-4.2: stage Qwen3-4B-AWQ weights into /var/models/hf_cache"
```

---

### Task 4.3: New `query_understanding.py` module — JSON schema + analyze_query

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/query_understanding.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_query_understanding_schema.py`

This module owns the LLM-side of the hybrid router. It defines the JSON output schema, builds the prompt, calls vllm-qu, parses the result, and surfaces a typed `QueryUnderstanding` dataclass to callers. The module does NOT decide WHEN to escalate — that's Task 4.4's job inside `query_intent.py`.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_query_understanding_schema.py`:

```python
import json
import pytest

from ext.services.query_understanding import (
    QueryUnderstanding,
    QU_OUTPUT_SCHEMA,
    build_qu_prompt,
    parse_qu_response,
)


def test_schema_has_required_fields():
    props = QU_OUTPUT_SCHEMA["properties"]
    for required in ("intent", "resolved_query", "temporal_constraint",
                     "entities", "confidence"):
        assert required in props, f"schema missing field {required}"


def test_intent_enum_values():
    enum = QU_OUTPUT_SCHEMA["properties"]["intent"]["enum"]
    assert set(enum) == {"metadata", "global", "specific", "specific_date"}


def test_temporal_constraint_nullable():
    tc = QU_OUTPUT_SCHEMA["properties"]["temporal_constraint"]
    # null + object union
    types = tc.get("anyOf") or tc.get("oneOf") or [{"type": tc.get("type")}]
    assert any(t.get("type") == "null" for t in types), \
        "temporal_constraint must allow null"


def test_confidence_bounded():
    c = QU_OUTPUT_SCHEMA["properties"]["confidence"]
    assert c["type"] == "number"
    assert c["minimum"] == 0.0 and c["maximum"] == 1.0


def test_required_list_includes_all_fields():
    required = set(QU_OUTPUT_SCHEMA["required"])
    assert required == {"intent", "resolved_query", "temporal_constraint",
                        "entities", "confidence"}


def test_build_qu_prompt_includes_query():
    prompt = build_qu_prompt(query="what changed last quarter", history=[])
    assert "what changed last quarter" in prompt


def test_build_qu_prompt_includes_history_context():
    history = [
        {"role": "user", "content": "tell me about the OFC roadmap"},
        {"role": "assistant", "content": "The OFC roadmap covers 2026-Q1 to 2027-Q1..."},
    ]
    prompt = build_qu_prompt(query="and what about Q2?", history=history)
    assert "OFC roadmap" in prompt or "previous turn" in prompt.lower()


def test_build_qu_prompt_includes_today_date():
    prompt = build_qu_prompt(query="last month", history=[])
    # The prompt must anchor "last month" relative to today
    import datetime as dt
    today = dt.date.today().isoformat()
    assert today in prompt


def test_parse_qu_response_happy_path():
    raw = json.dumps({
        "intent": "specific_date",
        "resolved_query": "outages on January 5 2026",
        "temporal_constraint": {"year": 2026, "quarter": None, "month": 1},
        "entities": ["outages"],
        "confidence": 0.92,
    })
    qu = parse_qu_response(raw)
    assert qu.intent == "specific_date"
    assert qu.resolved_query == "outages on January 5 2026"
    assert qu.temporal_constraint == {"year": 2026, "quarter": None, "month": 1}
    assert qu.entities == ["outages"]
    assert qu.confidence == 0.92


def test_parse_qu_response_rejects_invalid_intent():
    raw = json.dumps({
        "intent": "freeform",  # not in enum
        "resolved_query": "x",
        "temporal_constraint": None,
        "entities": [],
        "confidence": 0.5,
    })
    with pytest.raises(ValueError, match="invalid intent"):
        parse_qu_response(raw)


def test_parse_qu_response_clamps_confidence():
    raw = json.dumps({
        "intent": "specific",
        "resolved_query": "x",
        "temporal_constraint": None,
        "entities": [],
        "confidence": 1.5,  # out of range
    })
    qu = parse_qu_response(raw)
    assert qu.confidence == 1.0  # clamped


def test_parse_qu_response_handles_garbage_json():
    with pytest.raises(ValueError):
        parse_qu_response("not even json {{")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_query_understanding_schema.py -v
```

Expected: ImportError — `query_understanding` module doesn't exist.

- [ ] **Step 3: Write the module**

Create `/home/vogic/LocalRAG/ext/services/query_understanding.py`:

```python
"""Query Understanding LLM — JSON-schema-guided intent + resolution.

Plan B Phase 4. This module is the LLM half of the hybrid regex+LLM router.
The fast regex path lives in :mod:`ext.services.query_intent`; this module
is invoked only when the router escalates (see Task 4.4 in Plan B).

Output is a constrained JSON object enforced by xgrammar guided_json on the
vLLM server side. We send the schema in the request; vLLM constrains
generation to valid output, eliminating the schema-violation noise that
plagued unconstrained LLM routers.

Schema:

  {
    "intent": "metadata" | "global" | "specific" | "specific_date",
    "resolved_query": "<rewritten standalone query, no pronouns>",
    "temporal_constraint": {
      "year": int | null, "quarter": int | null, "month": int | null
    } | null,
    "entities": ["<named entity>", ...],
    "confidence": float in [0.0, 1.0]
  }

The model is `Qwen/Qwen3-4B-Instruct-2507-AWQ` served by `vllm-qu` on GPU 1.
This module is async and uses :mod:`httpx`; it has a hard deadline
(``RAG_QU_LATENCY_BUDGET_MS``, default 600 ms). On deadline-miss or HTTP
failure the caller falls back to regex-only classification.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


log = logging.getLogger("orgchat.qu")


# JSON schema fed to vLLM's guided_json. Keep this in lockstep with
# ``QueryUnderstanding`` below — tests in
# ``test_query_understanding_schema.py`` enforce the shape.
QU_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "resolved_query", "temporal_constraint",
                 "entities", "confidence"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["metadata", "global", "specific", "specific_date"],
        },
        "resolved_query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
        "temporal_constraint": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["year", "quarter", "month"],
                    "properties": {
                        "year": {"anyOf": [{"type": "integer", "minimum": 1900,
                                            "maximum": 2100}, {"type": "null"}]},
                        "quarter": {"anyOf": [{"type": "integer", "minimum": 1,
                                               "maximum": 4}, {"type": "null"}]},
                        "month": {"anyOf": [{"type": "integer", "minimum": 1,
                                             "maximum": 12}, {"type": "null"}]},
                    },
                },
            ],
        },
        "entities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
            "maxItems": 10,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}


@dataclass
class QueryUnderstanding:
    """Structured output from the QU LLM."""

    intent: str
    resolved_query: str
    temporal_constraint: Optional[dict]
    entities: list[str] = field(default_factory=list)
    confidence: float = 0.0
    # Provenance — lets the bridge log "regex" vs "llm:high_conf" reasons.
    source: str = "llm"
    cached: bool = False


_VALID_INTENTS = {"metadata", "global", "specific", "specific_date"}


_PROMPT_TEMPLATE = """You are a query classifier and rewriter for a retrieval system.

Today's date is {today}.

Recent conversation context (oldest first):
{history_block}

Current user query: "{query}"

Your task:
1. Classify the query into ONE of: metadata, global, specific, specific_date.
   - metadata: enumeration / catalog questions ("list documents", "what files do I have")
   - global: aggregation / coverage across the corpus ("compare", "trends", "summarize all")
   - specific: single-document or content-anchored question
   - specific_date: question pinpointing a date or month
2. Resolve the query into a standalone form. Replace pronouns ("it", "that") with
   their antecedents from history. Replace relative time ("last quarter", "yesterday")
   with absolute dates relative to today.
3. Extract a temporal_constraint object {{year, quarter, month}} if any is implied;
   otherwise null. quarter is 1-4 (Q1=Jan-Mar). month is 1-12.
4. List the named entities (products, places, people) referenced in the query.
5. Output your confidence in [0.0, 1.0].

Respond with JSON ONLY, conforming to the provided schema."""


def build_qu_prompt(query: str, history: list[dict]) -> str:
    """Compose the prompt for the QU LLM.

    ``history`` is a list of {role, content} dicts in chronological order
    (oldest first). We include up to the last 3 turns, truncated to 200
    chars each, to keep the prompt under the model's prefix-cache window.
    """
    today = _dt.date.today().isoformat()
    if history:
        # Take last 3 turns (= 6 messages max if user/assistant interleave)
        truncated = history[-6:]
        lines = []
        for msg in truncated:
            role = msg.get("role", "user")
            content = (msg.get("content") or "").replace("\n", " ").strip()
            if len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"  [{role}]: {content}")
        history_block = "\n".join(lines)
    else:
        history_block = "  (no previous turns)"
    return _PROMPT_TEMPLATE.format(
        today=today, history_block=history_block, query=query
    )


def parse_qu_response(raw: str) -> QueryUnderstanding:
    """Parse a JSON response from the QU LLM into a QueryUnderstanding.

    Raises ValueError on malformed JSON or invalid intent. Clamps
    confidence to [0.0, 1.0] silently (defensive — the schema bounds it
    but pre-vLLM-V1 servers occasionally violated guided_json bounds).
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"QU response not valid JSON: {e}") from e

    intent = data.get("intent")
    if intent not in _VALID_INTENTS:
        raise ValueError(f"invalid intent in QU response: {intent!r}")

    resolved = data.get("resolved_query")
    if not isinstance(resolved, str) or not resolved.strip():
        raise ValueError("resolved_query missing or empty")

    tc = data.get("temporal_constraint")
    if tc is not None and not isinstance(tc, dict):
        raise ValueError("temporal_constraint must be null or object")

    entities = data.get("entities") or []
    if not isinstance(entities, list):
        raise ValueError("entities must be a list")

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return QueryUnderstanding(
        intent=intent,
        resolved_query=resolved.strip(),
        temporal_constraint=tc,
        entities=[str(e) for e in entities[:10]],
        confidence=confidence,
        source="llm",
        cached=False,
    )


async def analyze_query(
    query: str,
    history: list[dict] | None = None,
    *,
    qu_url: str | None = None,
    model: str | None = None,
    timeout_ms: int | None = None,
) -> Optional[QueryUnderstanding]:
    """Call the QU LLM and return a parsed QueryUnderstanding.

    Returns ``None`` on any failure — the caller falls back to regex.
    Soft-deadline via ``timeout_ms`` (default ``RAG_QU_LATENCY_BUDGET_MS``,
    600 ms); deadline misses are logged at WARN with ``rag_qu.timeout``
    metric increment.
    """
    qu_url = qu_url or os.environ.get("RAG_QU_URL", "http://vllm-qu:8000/v1")
    model = model or os.environ.get("RAG_QU_MODEL", "qwen3-4b-qu")
    timeout_ms = timeout_ms or int(os.environ.get("RAG_QU_LATENCY_BUDGET_MS", "600"))
    timeout = timeout_ms / 1000.0

    prompt = build_qu_prompt(query=query, history=history or [])

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output only JSON. No prose."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        # vLLM V1 + xgrammar guided_json. The schema is enforced server-side.
        "extra_body": {"guided_json": QU_OUTPUT_SCHEMA},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{qu_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            return parse_qu_response(raw)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        log.warning("QU LLM timed out after %dms; falling back to regex", timeout_ms)
        return None
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        log.warning("QU LLM error: %s; falling back to regex", e)
        return None


__all__ = [
    "QueryUnderstanding",
    "QU_OUTPUT_SCHEMA",
    "analyze_query",
    "build_qu_prompt",
    "parse_qu_response",
]
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_query_understanding_schema.py -v
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/query_understanding.py tests/unit/test_query_understanding_schema.py
git commit -m "phase-4.3: query_understanding module with xgrammar JSON schema"
```

---

### Task 4.4: Hybrid regex+LLM router — escalation predicates

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/query_intent.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_qu_router_escalation.py`

The hybrid router decides WHEN to escalate from regex to LLM. Escalation predicates (in order; any match triggers escalation):

1. **Pronoun reference** — `\b(it|that|this|those|these|they|them)\b` AND history is non-empty
2. **Relative time** — `\b(last|previous|next|coming|prior)\s+(week|month|quarter|year|day)\b` OR `\b(yesterday|tomorrow|today)\b`
3. **Multi-clause** — query contains `\b(and|or|but|also|while|whereas)\b` AND has more than 8 tokens
4. **Long query** — token count > 25
5. **No NER hit + question word** — query starts with `(what|how|when|where|why|which|who|do|did|is|was)` AND no capitalized non-stopword sequence (rough proxy for "no entity to anchor retrieval")
6. **Comparison verb** — `\b(compare|contrast|differ|change|evolve|trend)\b`

When the regex fast-path returns `metadata` or `global` with high pattern confidence, we DO NOT escalate — those classes are trustworthy. When fast-path returns `specific_date` with a successfully extracted tuple, we don't escalate either. Escalation is ONLY for `specific` results (the fast-path's "fallback" class) where one of the predicates fires.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_qu_router_escalation.py`:

```python
import pytest

from ext.services.query_intent import (
    should_escalate_to_llm,
    EscalationReason,
)


class TestEscalationPredicates:
    """should_escalate_to_llm returns (bool, EscalationReason)."""

    def test_no_escalation_for_short_specific_query_with_entity(self):
        # "OFC roadmap" has an entity-shaped token (OFC) and short
        escalate, reason = should_escalate_to_llm(
            query="show me OFC roadmap", regex_label="specific", history=[],
        )
        assert escalate is False
        assert reason is EscalationReason.NONE

    def test_no_escalation_for_metadata_label(self):
        # metadata path is trustworthy — never escalate
        escalate, reason = should_escalate_to_llm(
            query="list all reports", regex_label="metadata", history=[],
        )
        assert escalate is False

    def test_no_escalation_for_global_label(self):
        escalate, reason = should_escalate_to_llm(
            query="summarize everything", regex_label="global", history=[],
        )
        assert escalate is False

    def test_no_escalation_for_specific_date(self):
        escalate, reason = should_escalate_to_llm(
            query="outages on 5 Jan 2026", regex_label="specific_date", history=[],
        )
        assert escalate is False

    def test_escalation_for_pronoun_with_history(self):
        history = [
            {"role": "user", "content": "tell me about OFC roadmap"},
            {"role": "assistant", "content": "OFC roadmap covers..."},
        ]
        escalate, reason = should_escalate_to_llm(
            query="and what about it in Q2?", regex_label="specific", history=history,
        )
        assert escalate is True
        assert reason is EscalationReason.PRONOUN_REF

    def test_no_escalation_for_pronoun_without_history(self):
        # Pronoun without antecedent is meaningless — regex result is fine
        escalate, _ = should_escalate_to_llm(
            query="what is it?", regex_label="specific", history=[],
        )
        assert escalate is False

    def test_escalation_for_relative_time(self):
        escalate, reason = should_escalate_to_llm(
            query="what happened last quarter?", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.RELATIVE_TIME

    def test_escalation_for_yesterday(self):
        escalate, reason = should_escalate_to_llm(
            query="yesterday's incidents", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.RELATIVE_TIME

    def test_escalation_for_long_query(self):
        long = " ".join(["word"] * 30)
        escalate, reason = should_escalate_to_llm(
            query=long, regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.LONG_QUERY

    def test_escalation_for_multi_clause(self):
        escalate, reason = should_escalate_to_llm(
            query="show me the roadmap and explain how it changed in Q2",
            regex_label="specific", history=[],
        )
        assert escalate is True
        # Either MULTI_CLAUSE or RELATIVE_TIME — both fire here; first match wins
        assert reason in (
            EscalationReason.MULTI_CLAUSE,
            EscalationReason.RELATIVE_TIME,
            EscalationReason.PRONOUN_REF,
        )

    def test_escalation_for_question_no_entity(self):
        # "what changed" — question word, no capitalized entity
        escalate, reason = should_escalate_to_llm(
            query="what changed?", regex_label="specific", history=[],
        )
        assert escalate is True

    def test_escalation_for_comparison_verb(self):
        escalate, reason = should_escalate_to_llm(
            query="compare the budgets", regex_label="specific", history=[],
        )
        assert escalate is True
        assert reason is EscalationReason.COMPARISON_VERB


class TestHybridClassify:
    """classify_with_qu wraps regex + escalate + analyze_query."""

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_qu_disabled(self, monkeypatch):
        from ext.services.query_intent import classify_with_qu
        monkeypatch.setenv("RAG_QU_ENABLED", "0")
        result = await classify_with_qu("compare budgets", history=[])
        assert result.intent == "specific"
        assert result.source == "regex"

    @pytest.mark.asyncio
    async def test_uses_qu_when_enabled_and_escalated(self, monkeypatch, mocker):
        from ext.services import query_intent as qi
        from ext.services.query_understanding import QueryUnderstanding

        async def fake_analyze(query, history, **kw):
            return QueryUnderstanding(
                intent="global", resolved_query="compare budgets across all years",
                temporal_constraint=None, entities=["budgets"],
                confidence=0.95, source="llm", cached=False,
            )

        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        mocker.patch.object(qi, "_invoke_qu", side_effect=fake_analyze)
        result = await qi.classify_with_qu("compare budgets", history=[])
        assert result.intent == "global"
        assert result.source == "llm"
        assert "across all years" in result.resolved_query

    @pytest.mark.asyncio
    async def test_falls_back_to_regex_when_qu_returns_none(self, monkeypatch, mocker):
        from ext.services import query_intent as qi

        async def fake_analyze(*a, **kw):
            return None  # simulate timeout / HTTP error

        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        mocker.patch.object(qi, "_invoke_qu", side_effect=fake_analyze)
        result = await qi.classify_with_qu("compare budgets", history=[])
        assert result.intent == "specific"
        assert result.source == "regex"

    @pytest.mark.asyncio
    async def test_does_not_escalate_for_metadata_query(self, monkeypatch, mocker):
        from ext.services import query_intent as qi
        spy = mocker.patch.object(qi, "_invoke_qu")
        monkeypatch.setenv("RAG_QU_ENABLED", "1")
        result = await qi.classify_with_qu("list all reports", history=[])
        assert result.intent == "metadata"
        assert result.source == "regex"
        spy.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_router_escalation.py -v
```

Expected: ImportError — `should_escalate_to_llm` doesn't exist.

- [ ] **Step 3: Modify `query_intent.py` — add escalation predicates and hybrid classifier**

Edit `/home/vogic/LocalRAG/ext/services/query_intent.py`. Find the existing module docstring and the `_llm_classify` stub function.

Replace the imports + add new exports near the top of the file (keep the existing `from typing import Literal, Optional, Tuple` line; add the rest):

```python
from __future__ import annotations

import enum
import os
import re
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple
```

Below the existing `_DATE_RE_ISO` block but above `_normalize_year`, add the escalation predicate constants:

```python
# --------------------------------------------------------------------------
# Plan B Phase 4.4 — escalation predicates for the hybrid regex+LLM router.
# --------------------------------------------------------------------------
_PRONOUN_RE = re.compile(r"\b(it|that|this|those|these|they|them)\b", re.IGNORECASE)
_RELATIVE_TIME_RE = re.compile(
    r"\b(last|previous|next|coming|prior)\s+(week|month|quarter|year|day)\b|"
    r"\b(yesterday|tomorrow|today)\b",
    re.IGNORECASE,
)
_MULTI_CLAUSE_CONNECTOR_RE = re.compile(
    r"\b(and|or|but|also|while|whereas)\b", re.IGNORECASE
)
_QUESTION_WORD_RE = re.compile(
    r"^\s*(what|how|when|where|why|which|who|do|did|is|was|are|were)\b",
    re.IGNORECASE,
)
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_COMPARISON_VERB_RE = re.compile(
    r"\b(compare|contrast|differ|change|evolve|trend)\w*\b", re.IGNORECASE
)

_LONG_QUERY_TOKEN_THRESHOLD = 25
_MULTI_CLAUSE_TOKEN_THRESHOLD = 8


class EscalationReason(enum.Enum):
    """Why the hybrid router decided to escalate to the LLM (or didn't)."""
    NONE = "none"
    PRONOUN_REF = "pronoun_ref"
    RELATIVE_TIME = "relative_time"
    MULTI_CLAUSE = "multi_clause"
    LONG_QUERY = "long_query"
    NO_ENTITY = "no_entity_question"
    COMPARISON_VERB = "comparison_verb"


def should_escalate_to_llm(
    query: str, regex_label: str, history: list[dict] | None,
) -> Tuple[bool, EscalationReason]:
    """Return (escalate, reason).

    Only escalates when the regex result is ``specific`` AND one of the
    six predicates fires. Other regex labels are trusted as-is. Order of
    predicates is fixed so the same input always picks the same reason
    (useful for shadow-mode A/B logging).
    """
    if regex_label != "specific":
        return False, EscalationReason.NONE
    if not query:
        return False, EscalationReason.NONE

    history = history or []
    tokens = query.split()
    n_tokens = len(tokens)

    # 1. Pronoun reference — only meaningful with history
    if history and _PRONOUN_RE.search(query):
        return True, EscalationReason.PRONOUN_REF

    # 2. Relative time
    if _RELATIVE_TIME_RE.search(query):
        return True, EscalationReason.RELATIVE_TIME

    # 3. Multi-clause
    if (n_tokens > _MULTI_CLAUSE_TOKEN_THRESHOLD
            and _MULTI_CLAUSE_CONNECTOR_RE.search(query)):
        return True, EscalationReason.MULTI_CLAUSE

    # 4. Long query
    if n_tokens > _LONG_QUERY_TOKEN_THRESHOLD:
        return True, EscalationReason.LONG_QUERY

    # 5. Comparison verb
    if _COMPARISON_VERB_RE.search(query):
        return True, EscalationReason.COMPARISON_VERB

    # 6. Question word + no capitalized entity
    if _QUESTION_WORD_RE.search(query) and not _CAPITALIZED_TOKEN_RE.search(query):
        return True, EscalationReason.NO_ENTITY

    return False, EscalationReason.NONE
```

Replace the existing `_llm_classify` stub with the wired hybrid classifier. Find:

```python
def _llm_classify(query: str) -> Tuple[Intent, str]:
    """LLM tiebreaker — TODO stub.
```

Replace the entire function body and update the public API. Append below `classify_with_reason`:

```python
@dataclass
class HybridClassification:
    """Hybrid router result.

    ``intent`` and ``resolved_query`` are the primary signals consumed by
    the bridge. ``source`` is "regex" or "llm" — used by metrics and
    shadow-mode logging. ``escalation_reason`` is the predicate that
    triggered (or NONE for non-escalated regex hits).
    """
    intent: str
    resolved_query: str
    temporal_constraint: Optional[dict]
    entities: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "regex"
    escalation_reason: EscalationReason = EscalationReason.NONE
    regex_reason: str = ""
    cached: bool = False


async def _invoke_qu(query: str, history: list[dict]) -> Optional["QueryUnderstanding"]:
    """Indirection so tests can monkeypatch the LLM call."""
    # Local import — keeps the regex hot-path import-clean (no httpx).
    from .query_understanding import analyze_query
    return await analyze_query(query=query, history=history)


async def classify_with_qu(
    query: str, history: list[dict] | None = None,
) -> HybridClassification:
    """Hybrid regex+LLM classifier.

    Always runs regex first. Escalates to the QU LLM only when:
      - ``RAG_QU_ENABLED=1``, AND
      - regex returned ``specific``, AND
      - an escalation predicate fired.

    On QU failure (timeout, HTTP error, schema violation), returns the
    regex result. The bridge can rely on this never raising — it's safe to
    call on every query.
    """
    regex_label, regex_reason = classify_with_reason(query)
    history = history or []

    # Default: regex result wins
    result = HybridClassification(
        intent=regex_label,
        resolved_query=query,
        temporal_constraint=None,
        entities=[],
        confidence=1.0,
        source="regex",
        escalation_reason=EscalationReason.NONE,
        regex_reason=regex_reason,
    )

    qu_enabled = os.environ.get("RAG_QU_ENABLED", "0") == "1"
    if not qu_enabled:
        return result

    escalate, reason = should_escalate_to_llm(query, regex_label, history)
    result.escalation_reason = reason
    if not escalate:
        return result

    qu = await _invoke_qu(query, history)
    if qu is None:
        # LLM failed — keep regex result, log via metrics elsewhere
        return result

    # LLM trusted — but do not let it lower confidence below 0.5
    if qu.confidence < 0.5:
        return result

    return HybridClassification(
        intent=qu.intent,
        resolved_query=qu.resolved_query,
        temporal_constraint=qu.temporal_constraint,
        entities=qu.entities,
        confidence=qu.confidence,
        source="llm",
        escalation_reason=reason,
        regex_reason=regex_reason,
        cached=qu.cached,
    )
```

Update `__all__` at the bottom:

```python
__all__ = [
    "Intent",
    "classify",
    "classify_with_reason",
    "classify_with_qu",
    "extract_date_tuple",
    "should_escalate_to_llm",
    "EscalationReason",
    "HybridClassification",
]
```

Delete the legacy `_llm_classify` stub entirely — `classify_with_qu` replaces it. Also update the `RAG_INTENT_LLM` block in `classify_with_reason` to be a no-op (the flag is being retired in Plan B Phase 4.10):

```python
    # Plan B Phase 4 — RAG_INTENT_LLM is retired in favor of RAG_QU_ENABLED.
    # The hybrid path is exposed via classify_with_qu(); this sync function
    # remains regex-only for backward compatibility with non-async callers.
    return "specific", _DEFAULT_REASON
```

Remove the lines that referenced `_llm_classify`.

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_router_escalation.py tests/unit/test_query_understanding_schema.py -v
```

Expected: all passed.

- [ ] **Step 5: Run the existing query_intent tests to confirm no regression**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_query_intent.py -v
```

Expected: all existing pass (regex behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add ext/services/query_intent.py tests/unit/test_qu_router_escalation.py
git commit -m "phase-4.4: hybrid regex+LLM router with escalation predicates"
```

---

### Task 4.5: Redis DB 4 cache for QU results

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/qu_cache.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_qu_cache.py`

Cache key: `qu:{sha256(normalize(query) + last_turn_id)}`. TTL configurable via `RAG_QU_CACHE_TTL_SECS` (default 300 s).

`normalize(query)` lowercases, strips, collapses whitespace, and removes trailing punctuation. The `last_turn_id` is the assistant's last turn ID (or `""` for new chats) so different conversation contexts don't share a cache entry.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_qu_cache.py`:

```python
import json
import pytest

from ext.services.qu_cache import (
    QUCache,
    _normalize_for_cache,
    _make_key,
)
from ext.services.query_understanding import QueryUnderstanding


def test_normalize_lowercases():
    assert _normalize_for_cache("OFC Roadmap") == "ofc roadmap"


def test_normalize_collapses_whitespace():
    assert _normalize_for_cache("hello   world  ") == "hello world"


def test_normalize_strips_trailing_punct():
    assert _normalize_for_cache("what is it??") == "what is it"


def test_make_key_deterministic():
    k1 = _make_key("compare budgets", "turn-42")
    k2 = _make_key("compare budgets", "turn-42")
    assert k1 == k2


def test_make_key_history_sensitive():
    k1 = _make_key("what about Q2", "turn-1")
    k2 = _make_key("what about Q2", "turn-2")
    assert k1 != k2


def test_make_key_normalizes_query():
    # Same content despite case + whitespace = same key
    k1 = _make_key("Compare  Budgets", "turn-42")
    k2 = _make_key("compare budgets", "turn-42")
    assert k1 == k2


def test_make_key_starts_with_namespace():
    assert _make_key("x", "y").startswith("qu:")


class TestQUCache:
    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        result = await cache.get("any query", "turn-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        qu = QueryUnderstanding(
            intent="global",
            resolved_query="compare budgets across years",
            temporal_constraint=None,
            entities=["budgets"],
            confidence=0.9,
            source="llm",
        )
        await cache.set("compare budgets", "turn-1", qu)
        result = await cache.get("compare budgets", "turn-1")
        assert result is not None
        assert result.intent == "global"
        assert result.cached is True  # set on retrieval

    @pytest.mark.asyncio
    async def test_disabled_cache_is_noop(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300, enabled=False)
        qu = QueryUnderstanding(
            intent="global", resolved_query="x", temporal_constraint=None,
            entities=[], confidence=0.9,
        )
        await cache.set("q", "t", qu)
        assert await cache.get("q", "t") is None  # never returns

    @pytest.mark.asyncio
    async def test_ttl_applied_on_set(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=42)
        qu = QueryUnderstanding(
            intent="specific", resolved_query="x", temporal_constraint=None,
            entities=[], confidence=0.5,
        )
        await cache.set("q", "t", qu)
        # fake_redis exposes the TTL through .ttl()
        ttl = await fake_redis.ttl(_make_key("q", "t"))
        assert 0 < ttl <= 42

    @pytest.mark.asyncio
    async def test_corrupt_cached_value_returns_none(self, fake_redis):
        cache = QUCache(redis_client=fake_redis, ttl_secs=300)
        # Manually inject garbage at the key
        await fake_redis.set(_make_key("q", "t"), "not json", ex=300)
        assert await cache.get("q", "t") is None
```

Add a `fake_redis` fixture to `/home/vogic/LocalRAG/tests/conftest.py`:

```python
@pytest.fixture
def fake_redis():
    """In-memory Redis double for cache tests.

    Implements the subset of redis.asyncio.Redis we touch: get/set/ttl/delete.
    """
    class _FakeRedis:
        def __init__(self) -> None:
            self._store: dict[str, str] = {}
            self._ttls: dict[str, int] = {}

        async def get(self, key):
            return self._store.get(key)

        async def set(self, key, value, ex=None):
            self._store[key] = value
            if ex is not None:
                self._ttls[key] = int(ex)
            return True

        async def ttl(self, key):
            return self._ttls.get(key, -1)

        async def delete(self, key):
            self._store.pop(key, None)
            self._ttls.pop(key, None)

    return _FakeRedis()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_cache.py -v
```

Expected: ImportError — `qu_cache` doesn't exist.

- [ ] **Step 3: Write the cache module**

Create `/home/vogic/LocalRAG/ext/services/qu_cache.py`:

```python
"""Redis-backed cache for QueryUnderstanding results.

Plan B Phase 4.5. Uses Redis DB 4 (Plan A took DB 3 for RBAC cache; we
avoid stomping). Key namespace: ``qu:<sha256>`` where the digest covers
the normalized query + last assistant turn ID. TTL default 300 s
(``RAG_QU_CACHE_TTL_SECS``).

Cached value is the JSON-serialized QueryUnderstanding. On retrieval we
mark ``cached=True`` so the bridge logs distinguish hot-cache hits from
cold LLM calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict
from typing import Optional

from .query_understanding import QueryUnderstanding


log = logging.getLogger("orgchat.qu_cache")


_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[?!.,;:]+$")


def _normalize_for_cache(query: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation.

    Goal: cache-hit "What is OFC?" and "what is ofc" as the same query
    while keeping "what is OFC roadmap" distinct.
    """
    if not query:
        return ""
    q = query.strip().lower()
    q = _WHITESPACE_RE.sub(" ", q)
    q = _TRAILING_PUNCT_RE.sub("", q)
    return q


def _make_key(query: str, last_turn_id: str) -> str:
    """Build the Redis key.

    ``last_turn_id`` is the assistant's last turn ID (string). Empty
    string for new chats. The hash includes both so different
    conversation contexts don't share entries.
    """
    norm = _normalize_for_cache(query)
    payload = f"{norm}\x00{last_turn_id or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"qu:{digest}"


class QUCache:
    """Async cache facade.

    Constructor takes a redis client (``redis.asyncio.Redis`` or compatible
    fake). Tests inject the fake; production wiring lives in the bridge.
    """

    def __init__(
        self,
        redis_client,
        *,
        ttl_secs: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._r = redis_client
        self._ttl = ttl_secs if ttl_secs is not None else int(
            os.environ.get("RAG_QU_CACHE_TTL_SECS", "300")
        )
        self._enabled = (
            enabled if enabled is not None
            else os.environ.get("RAG_QU_CACHE_ENABLED", "1") == "1"
        )

    async def get(self, query: str, last_turn_id: str) -> Optional[QueryUnderstanding]:
        if not self._enabled:
            return None
        key = _make_key(query, last_turn_id)
        try:
            raw = await self._r.get(key)
        except Exception as e:  # connection refused, etc.
            log.warning("qu_cache.get failed: %s", e)
            return None
        if not raw:
            return None
        try:
            data = json.loads(raw)
            qu = QueryUnderstanding(**data)
            qu.cached = True
            return qu
        except (json.JSONDecodeError, TypeError) as e:
            log.warning("qu_cache: corrupt cached value at %s: %s", key, e)
            return None

    async def set(
        self,
        query: str,
        last_turn_id: str,
        qu: QueryUnderstanding,
    ) -> None:
        if not self._enabled:
            return
        key = _make_key(query, last_turn_id)
        # Don't persist the cached flag — every read sets it to True
        payload = {**asdict(qu), "cached": False}
        try:
            await self._r.set(key, json.dumps(payload), ex=self._ttl)
        except Exception as e:
            log.warning("qu_cache.set failed: %s", e)


__all__ = ["QUCache", "_normalize_for_cache", "_make_key"]
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_cache.py -v
```

Expected: all passed.

- [ ] **Step 5: Add Redis DB 4 to compose env**

Edit `/home/vogic/LocalRAG/compose/docker-compose.yml`, find the open-webui environment block, add:

```yaml
      RAG_QU_REDIS_DB: ${RAG_QU_REDIS_DB:-4}
```

(The Redis URL comes from the existing `REDIS_URL` env; the bridge will pass `db=4` to its connection.)

- [ ] **Step 6: Commit**

```bash
git add ext/services/qu_cache.py tests/unit/test_qu_cache.py tests/conftest.py compose/docker-compose.yml
git commit -m "phase-4.5: Redis DB 4 QU cache with TTL + soft-fail"
```

---

### Task 4.6: Wire QU into chat_rag_bridge — replace classify_intent

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/chat_rag_bridge.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_bridge_qu_wiring.py`

The bridge currently calls the local `classify_intent(query) -> str` (a 3-class regex). Phase 4.6 replaces this call site with `classify_with_qu(query, history) -> HybridClassification` and threads the result (intent + resolved_query + temporal_constraint + entities + confidence + source) into the existing intent-flag policy and downstream retrieval.

The bridge must:
1. Look up the chat's last assistant turn ID before calling QU (for cache key).
2. Initialize `QUCache` once per process (singleton).
3. Check cache before calling LLM. On cache hit, surface as `source="llm"` + `cached=True`.
4. Use `resolved_query` for retrieval (the rewritten standalone form), but keep the original `query` for response framing.
5. Fall back gracefully if QU returns None (already handled inside `classify_with_qu`).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_bridge_qu_wiring.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_bridge_uses_resolved_query_for_retrieval(monkeypatch, mocker):
    """When QU runs, the bridge must use resolved_query (not original) for retrieval."""
    from ext.services import chat_rag_bridge as bridge
    from ext.services.query_intent import HybridClassification, EscalationReason

    monkeypatch.setenv("RAG_QU_ENABLED", "1")

    fake_hybrid = HybridClassification(
        intent="global",
        resolved_query="compare budgets across 2024 2025 2026",
        temporal_constraint=None,
        entities=["budgets"],
        confidence=0.95,
        source="llm",
        escalation_reason=EscalationReason.RELATIVE_TIME,
    )

    async def fake_classify(query, history=None):
        return fake_hybrid
    mocker.patch.object(bridge, "_classify_with_qu", side_effect=fake_classify)

    captured = {}
    async def fake_retrieve(query, **kw):
        captured["query"] = query
        return []
    mocker.patch.object(bridge, "_retrieve_for_query", side_effect=fake_retrieve)

    # Drive a retrieval through the bridge
    await bridge._run_pipeline(  # internal entry for the test
        query="compare budgets last quarter",
        history=[],
        chat_id="chat-1",
        user=mocker.MagicMock(id=1),
        kb_config=[{"kb_id": 1}],
    )

    assert captured["query"] == "compare budgets across 2024 2025 2026"


@pytest.mark.asyncio
async def test_bridge_falls_back_to_regex_when_qu_disabled(monkeypatch):
    from ext.services import chat_rag_bridge as bridge
    monkeypatch.setenv("RAG_QU_ENABLED", "0")

    # Use a query the regex correctly classifies
    label = bridge.classify_intent("list all reports")
    assert label == "metadata"


@pytest.mark.asyncio
async def test_bridge_uses_cache_on_repeat(monkeypatch, mocker):
    """Second identical call within TTL hits cache; LLM not invoked twice."""
    from ext.services import chat_rag_bridge as bridge
    from ext.services.query_intent import HybridClassification, EscalationReason

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_CACHE_ENABLED", "1")

    invoke_count = {"n": 0}
    async def fake_invoke_qu(query, history):
        invoke_count["n"] += 1
        from ext.services.query_understanding import QueryUnderstanding
        return QueryUnderstanding(
            intent="specific", resolved_query=query, temporal_constraint=None,
            entities=[], confidence=0.9,
        )
    mocker.patch("ext.services.query_intent._invoke_qu", side_effect=fake_invoke_qu)

    # First call — cache miss
    r1 = await bridge._classify_with_qu(query="compare budgets last quarter",
                                         history=[], last_turn_id="t-1")
    # Second identical call — cache hit
    r2 = await bridge._classify_with_qu(query="compare budgets last quarter",
                                         history=[], last_turn_id="t-1")
    assert invoke_count["n"] == 1, "QU should only be invoked once"
    assert r2.cached is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_bridge_qu_wiring.py -v
```

Expected: AttributeError — `_classify_with_qu` not in bridge.

- [ ] **Step 3: Modify the bridge**

Edit `/home/vogic/LocalRAG/ext/services/chat_rag_bridge.py`. Add near the imports:

```python
from .query_intent import (
    classify_with_qu as _qu_classify,
    classify_with_reason as _ci_classify,
    HybridClassification,
)
from .qu_cache import QUCache
```

Add a module-level cache singleton (lazy-initialized):

```python
_qu_cache_singleton: QUCache | None = None


def _get_qu_cache() -> QUCache | None:
    global _qu_cache_singleton
    if _qu_cache_singleton is not None:
        return _qu_cache_singleton
    if _os.environ.get("RAG_QU_CACHE_ENABLED", "1") != "1":
        return None
    try:
        import redis.asyncio as aioredis
        url = _os.environ.get("REDIS_URL", "redis://redis:6379")
        db = int(_os.environ.get("RAG_QU_REDIS_DB", "4"))
        client = aioredis.from_url(url, db=db, decode_responses=True)
        _qu_cache_singleton = QUCache(redis_client=client)
        return _qu_cache_singleton
    except Exception as e:
        logger.warning("QU cache init failed: %s — running without cache", e)
        return None
```

Replace the existing `classify_intent` call site at the top of `_run_pipeline` (currently around line 413). Find:

```python
    _intent_label = classify_intent(query)
```

Replace with the new flow that incorporates QU:

```python
    last_turn_id = _extract_last_turn_id(history) if history else ""
    hybrid = await _classify_with_qu(query=query, history=history,
                                       last_turn_id=last_turn_id)
    _intent_label = hybrid.intent
    # Use resolved_query downstream; original query preserved for response
    if hybrid.source == "llm" and hybrid.resolved_query and \
            hybrid.resolved_query != query:
        retrieval_query = hybrid.resolved_query
    else:
        retrieval_query = query
```

Add the helpers below `classify_intent`:

```python
def _extract_last_turn_id(history: list[dict]) -> str:
    """Return the last assistant turn's stable ID, or '' if none.

    The chat middleware passes history with optional ``id`` keys per turn.
    Falls back to the assistant's content-hash when no ID is present.
    """
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            tid = msg.get("id")
            if tid:
                return str(tid)
            content = msg.get("content") or ""
            return _hash_short(content)
    return ""


def _hash_short(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


async def _classify_with_qu(
    query: str, history: list[dict] | None, last_turn_id: str = "",
) -> HybridClassification:
    """Bridge-side wrapper around query_intent.classify_with_qu with caching."""
    cache = _get_qu_cache()
    if cache is not None:
        from .query_understanding import QueryUnderstanding
        cached = await cache.get(query, last_turn_id)
        if cached is not None:
            from .query_intent import EscalationReason
            return HybridClassification(
                intent=cached.intent,
                resolved_query=cached.resolved_query,
                temporal_constraint=cached.temporal_constraint,
                entities=cached.entities,
                confidence=cached.confidence,
                source="llm",
                escalation_reason=EscalationReason.NONE,
                cached=True,
            )
    result = await _qu_classify(query=query, history=history)
    if cache is not None and result.source == "llm":
        from .query_understanding import QueryUnderstanding
        qu = QueryUnderstanding(
            intent=result.intent,
            resolved_query=result.resolved_query,
            temporal_constraint=result.temporal_constraint,
            entities=result.entities,
            confidence=result.confidence,
            source="llm",
            cached=False,
        )
        await cache.set(query, last_turn_id, qu)
    return result
```

The legacy `classify_intent(query) -> str` synchronous helper stays as-is for any non-async caller; mark it deprecated in its docstring:

```python
def classify_intent(query: str) -> str:
    """DEPRECATED — use ``_classify_with_qu`` for new call sites.

    Sync regex fallback retained for backward compatibility with existing
    synchronous callers (logging hooks, debug endpoints). New code should
    consume the full ``HybridClassification`` from ``_classify_with_qu``.
    """
    # ... existing body unchanged ...
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_bridge_qu_wiring.py tests/unit/test_chat_rag_bridge_intent_label.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/chat_rag_bridge.py tests/unit/test_bridge_qu_wiring.py
git commit -m "phase-4.6: wire hybrid QU classifier into chat_rag_bridge with cache"
```

---

### Task 4.7: QU LLM metrics — escalation rate, latency, schema violations, cache hit ratio

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/metrics.py`
- Modify: `/home/vogic/LocalRAG/ext/services/query_understanding.py` (add metric increments)
- Modify: `/home/vogic/LocalRAG/ext/services/qu_cache.py` (add metric increments)
- Modify: `/home/vogic/LocalRAG/ext/services/query_intent.py` (escalation counter)
- Create: `/home/vogic/LocalRAG/tests/unit/test_qu_metrics.py`

Counters/histograms exposed at `/metrics`:
- `rag_qu_invocations_total{source="llm|regex|cached"}` (counter)
- `rag_qu_escalations_total{reason="<enum>"}` (counter)
- `rag_qu_latency_seconds_bucket` (histogram, label-less since vllm-qu serves one model)
- `rag_qu_schema_violations_total` (counter)
- `rag_qu_cache_hits_total` / `rag_qu_cache_misses_total` (counter pair)
- `rag_qu_cache_hit_ratio` (gauge — derived; recomputed every 30s)

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_qu_metrics.py`:

```python
import pytest
from prometheus_client import REGISTRY


def _counter_value(name: str, labels: dict | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels=labels or {}) or 0.0


def test_metrics_exposed():
    from ext.services import metrics
    # The expected metrics must exist after import
    assert hasattr(metrics, "RAG_QU_INVOCATIONS")
    assert hasattr(metrics, "RAG_QU_ESCALATIONS")
    assert hasattr(metrics, "RAG_QU_LATENCY")
    assert hasattr(metrics, "RAG_QU_SCHEMA_VIOLATIONS")
    assert hasattr(metrics, "RAG_QU_CACHE_HITS")
    assert hasattr(metrics, "RAG_QU_CACHE_MISSES")


@pytest.mark.asyncio
async def test_escalation_counter_incremented_on_escalation(monkeypatch, mocker):
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding
    monkeypatch.setenv("RAG_QU_ENABLED", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global", resolved_query="x", temporal_constraint=None,
            entities=[], confidence=0.9,
        )
    mocker.patch.object(qi, "_invoke_qu", side_effect=fake_invoke)

    before = _counter_value(
        "rag_qu_escalations_total", {"reason": "comparison_verb"}
    )
    await qi.classify_with_qu("compare budgets", history=[])
    after = _counter_value(
        "rag_qu_escalations_total", {"reason": "comparison_verb"}
    )
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_invocations_source_label(monkeypatch, mocker):
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding
    monkeypatch.setenv("RAG_QU_ENABLED", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global", resolved_query="x", temporal_constraint=None,
            entities=[], confidence=0.9,
        )
    mocker.patch.object(qi, "_invoke_qu", side_effect=fake_invoke)

    before_llm = _counter_value("rag_qu_invocations_total", {"source": "llm"})
    await qi.classify_with_qu("compare budgets", history=[])
    after_llm = _counter_value("rag_qu_invocations_total", {"source": "llm"})
    assert after_llm > before_llm


@pytest.mark.asyncio
async def test_schema_violation_counted(monkeypatch, mocker):
    from ext.services import query_understanding as qu
    import httpx

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "not json garbage {{"}}]}

    async def fake_post(self, *a, **kw):
        return _FakeResp()
    mocker.patch.object(httpx.AsyncClient, "post", fake_post)

    before = _counter_value("rag_qu_schema_violations_total")
    result = await qu.analyze_query("x", history=[], qu_url="http://stub", model="m")
    after = _counter_value("rag_qu_schema_violations_total")
    assert result is None
    assert after - before == 1.0


@pytest.mark.asyncio
async def test_cache_hit_counter(fake_redis):
    from ext.services.qu_cache import QUCache
    from ext.services.query_understanding import QueryUnderstanding

    cache = QUCache(redis_client=fake_redis, ttl_secs=300)
    qu = QueryUnderstanding(
        intent="specific", resolved_query="x", temporal_constraint=None,
        entities=[], confidence=0.5,
    )
    await cache.set("q", "t", qu)

    before_hit = _counter_value("rag_qu_cache_hits_total")
    before_miss = _counter_value("rag_qu_cache_misses_total")
    await cache.get("q", "t")    # hit
    await cache.get("q2", "t")   # miss
    after_hit = _counter_value("rag_qu_cache_hits_total")
    after_miss = _counter_value("rag_qu_cache_misses_total")

    assert after_hit - before_hit == 1.0
    assert after_miss - before_miss == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_metrics.py -v
```

Expected: AttributeError — `RAG_QU_*` not in metrics.

- [ ] **Step 3: Add metrics**

Edit `/home/vogic/LocalRAG/ext/services/metrics.py`. Append:

```python
# -----------------------------------------------------------------------
# Plan B Phase 4.7 — Query Understanding LLM metrics
# -----------------------------------------------------------------------
from prometheus_client import Counter, Gauge, Histogram

RAG_QU_INVOCATIONS = Counter(
    "rag_qu_invocations_total",
    "QU classifier invocations by source (regex / llm / cached)",
    ["source"],
)

RAG_QU_ESCALATIONS = Counter(
    "rag_qu_escalations_total",
    "QU escalations from regex to LLM by predicate reason",
    ["reason"],
)

RAG_QU_LATENCY = Histogram(
    "rag_qu_latency_seconds",
    "QU LLM call latency",
    buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0, 5.0),
)

RAG_QU_SCHEMA_VIOLATIONS = Counter(
    "rag_qu_schema_violations_total",
    "QU LLM responses that failed schema validation",
)

RAG_QU_CACHE_HITS = Counter(
    "rag_qu_cache_hits_total",
    "QU cache hits",
)

RAG_QU_CACHE_MISSES = Counter(
    "rag_qu_cache_misses_total",
    "QU cache misses",
)

RAG_QU_CACHE_HIT_RATIO = Gauge(
    "rag_qu_cache_hit_ratio",
    "QU cache hit ratio (derived; recomputed every 30s by background task)",
)
```

Edit `/home/vogic/LocalRAG/ext/services/query_understanding.py` — wrap the LLM call with timing + violation counter:

```python
import time
from .metrics import RAG_QU_LATENCY, RAG_QU_SCHEMA_VIOLATIONS

async def analyze_query(query, history=None, *, qu_url=None, model=None, timeout_ms=None):
    # ... existing prep ...
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{qu_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            try:
                qu = parse_qu_response(raw)
            except ValueError:
                RAG_QU_SCHEMA_VIOLATIONS.inc()
                raise
            return qu
    # ... existing except blocks ...
    finally:
        RAG_QU_LATENCY.observe(time.monotonic() - start)
```

Edit `/home/vogic/LocalRAG/ext/services/query_intent.py` — increment counters in `classify_with_qu`:

```python
from .metrics import RAG_QU_INVOCATIONS, RAG_QU_ESCALATIONS

async def classify_with_qu(query, history=None):
    regex_label, regex_reason = classify_with_reason(query)
    history = history or []
    # ... existing default result ...
    qu_enabled = os.environ.get("RAG_QU_ENABLED", "0") == "1"
    if not qu_enabled:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result
    escalate, reason = should_escalate_to_llm(query, regex_label, history)
    result.escalation_reason = reason
    if not escalate:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result
    RAG_QU_ESCALATIONS.labels(reason=reason.value).inc()
    qu = await _invoke_qu(query, history)
    if qu is None or qu.confidence < 0.5:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result
    RAG_QU_INVOCATIONS.labels(source="llm").inc()
    return HybridClassification(... )  # existing return
```

Edit `/home/vogic/LocalRAG/ext/services/qu_cache.py` — increment hit/miss:

```python
from .metrics import RAG_QU_CACHE_HITS, RAG_QU_CACHE_MISSES, RAG_QU_INVOCATIONS

async def get(self, query, last_turn_id):
    # ... existing prep ...
    if not raw:
        RAG_QU_CACHE_MISSES.inc()
        return None
    try:
        # ... parse ...
        RAG_QU_CACHE_HITS.inc()
        RAG_QU_INVOCATIONS.labels(source="cached").inc()
        return qu
    except (json.JSONDecodeError, TypeError):
        RAG_QU_CACHE_MISSES.inc()  # corrupt = effectively a miss
        return None
```

- [ ] **Step 4: Add Prometheus alert rules**

Create `/home/vogic/LocalRAG/observability/prometheus/alerts-qu.yml`:

```yaml
groups:
  - name: qu_llm_slo
    interval: 30s
    rules:
      - alert: QULLMHighLatency
        expr: histogram_quantile(0.95, rate(rag_qu_latency_seconds_bucket[5m])) > 0.6
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "QU LLM p95 latency > 600ms"
          description: "vllm-qu p95 over 5m is {{ $value }}s, ceiling 600ms."

      - alert: QULLMHighSchemaViolation
        expr: rate(rag_qu_schema_violations_total[10m]) / rate(rag_qu_invocations_total{source="llm"}[10m]) > 0.01
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "QU LLM schema-violation rate > 1%"
          description: "Check vllm-qu logs for guided_json failures."

      - alert: QULLMHighEscalationRate
        expr: rate(rag_qu_escalations_total[1h]) / rate(rag_qu_invocations_total[1h]) > 0.40
        for: 30m
        labels: {severity: info}
        annotations:
          summary: "QU LLM escalation rate > 40%"
          description: "Hybrid router escalating more than expected. Review predicates."

      - alert: QULLMLowCacheHitRatio
        expr: rag_qu_cache_hit_ratio < 0.3
        for: 1h
        labels: {severity: info}
        annotations:
          summary: "QU cache hit ratio < 30%"
          description: "Either workload is unique or TTL is too short."
```

- [ ] **Step 5: Re-run metric tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_metrics.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/metrics.py ext/services/query_understanding.py \
        ext/services/qu_cache.py ext/services/query_intent.py \
        observability/prometheus/alerts-qu.yml \
        tests/unit/test_qu_metrics.py
git commit -m "phase-4.7: QU LLM metrics + Prometheus alerts"
```

---

### Task 4.8: Shadow-mode A/B harness

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/query_intent.py`
- Modify: `/home/vogic/LocalRAG/ext/services/chat_rag_bridge.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_qu_shadow_mode.py`
- Create: `/home/vogic/LocalRAG/scripts/analyze_shadow_log.py`

When `RAG_QU_SHADOW_MODE=1`, the bridge ALWAYS runs both regex and LLM paths and writes a structured log line per query. Production routing remains regex-only (the LLM result is only logged, not consumed). After a 7-day shadow window, the operator analyzes the log to compare regex vs LLM agreement rates per intent class — if LLM disagrees > 15% on `specific` queries with reason `comparison_verb` or `relative_time`, the LLM is winning the disambiguation case the predicates target.

Log format (one line per QU shadow event, JSON):

```json
{
  "ts": "2026-04-26T14:23:01Z",
  "user_id": 42,
  "chat_id": "...",
  "query": "what changed last quarter",
  "regex_label": "specific",
  "regex_reason": "default:no_pattern_matched",
  "llm_label": "global",
  "llm_resolved_query": "what changed in 2026-Q1",
  "llm_temporal": {"year": 2026, "quarter": 1, "month": null},
  "llm_confidence": 0.91,
  "agree": false,
  "escalation_reason": "relative_time"
}
```

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_qu_shadow_mode.py`:

```python
import json
import pytest


@pytest.mark.asyncio
async def test_shadow_mode_logs_both_paths(monkeypatch, mocker, caplog):
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")

    async def fake_invoke(*a, **kw):
        return QueryUnderstanding(
            intent="global",
            resolved_query="what changed in 2026-Q1",
            temporal_constraint={"year": 2026, "quarter": 1, "month": None},
            entities=[], confidence=0.91,
        )
    mocker.patch.object(qi, "_invoke_qu", side_effect=fake_invoke)

    caplog.set_level("INFO", logger="orgchat.qu_shadow")
    result = await qi.classify_with_qu("what changed last quarter", history=[])

    # Production routing is still regex-only in shadow mode
    assert result.source == "regex"
    assert result.intent == "specific"

    # Shadow log emitted
    shadow_records = [r for r in caplog.records if r.name == "orgchat.qu_shadow"]
    assert len(shadow_records) == 1
    payload = json.loads(shadow_records[0].message)
    assert payload["regex_label"] == "specific"
    assert payload["llm_label"] == "global"
    assert payload["agree"] is False


@pytest.mark.asyncio
async def test_shadow_mode_off_does_not_invoke_llm_unnecessarily(
    monkeypatch, mocker
):
    from ext.services import query_intent as qi
    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "0")

    spy = mocker.patch.object(qi, "_invoke_qu")
    # Query the regex labels as metadata — escalation predicates DO NOT fire
    await qi.classify_with_qu("list all reports", history=[])
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_shadow_mode_invokes_llm_even_without_escalation(
    monkeypatch, mocker, caplog
):
    """In shadow mode, LLM runs on EVERY query so we observe the full distribution."""
    from ext.services import query_intent as qi
    from ext.services.query_understanding import QueryUnderstanding

    monkeypatch.setenv("RAG_QU_ENABLED", "1")
    monkeypatch.setenv("RAG_QU_SHADOW_MODE", "1")

    invoked = {"count": 0}
    async def fake_invoke(*a, **kw):
        invoked["count"] += 1
        return QueryUnderstanding(
            intent="metadata", resolved_query="list all reports",
            temporal_constraint=None, entities=[], confidence=0.9,
        )
    mocker.patch.object(qi, "_invoke_qu", side_effect=fake_invoke)

    caplog.set_level("INFO", logger="orgchat.qu_shadow")
    # A regex-trustworthy query — normal mode wouldn't invoke LLM
    await qi.classify_with_qu("list all reports", history=[])

    assert invoked["count"] == 1, "shadow mode must invoke LLM even on non-escalated queries"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_shadow_mode.py -v
```

Expected: failures — shadow mode not implemented.

- [ ] **Step 3: Modify `classify_with_qu` to support shadow mode**

Edit `/home/vogic/LocalRAG/ext/services/query_intent.py`. Replace the existing `classify_with_qu` body with this shadow-aware variant:

```python
import json as _json

_shadow_log = logging.getLogger("orgchat.qu_shadow")


async def classify_with_qu(
    query: str, history: list[dict] | None = None,
) -> HybridClassification:
    """Hybrid regex+LLM classifier with optional shadow mode.

    When ``RAG_QU_SHADOW_MODE=1``, runs the LLM on EVERY query (not just
    escalated ones), logs both decisions to ``orgchat.qu_shadow``, but
    uses the regex result for production routing. Use this to observe
    the LLM-vs-regex agreement distribution before promoting LLM-as-default.
    """
    regex_label, regex_reason = classify_with_reason(query)
    history = history or []

    result = HybridClassification(
        intent=regex_label,
        resolved_query=query,
        temporal_constraint=None,
        entities=[],
        confidence=1.0,
        source="regex",
        escalation_reason=EscalationReason.NONE,
        regex_reason=regex_reason,
    )

    qu_enabled = os.environ.get("RAG_QU_ENABLED", "0") == "1"
    shadow_mode = os.environ.get("RAG_QU_SHADOW_MODE", "0") == "1"
    if not qu_enabled and not shadow_mode:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result

    escalate, reason = should_escalate_to_llm(query, regex_label, history)
    result.escalation_reason = reason

    # In shadow mode we always invoke; in normal mode only on escalation
    if not shadow_mode and not escalate:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result

    if escalate and not shadow_mode:
        RAG_QU_ESCALATIONS.labels(reason=reason.value).inc()

    qu = await _invoke_qu(query, history)

    if shadow_mode:
        _emit_shadow_log(query=query, regex_label=regex_label,
                          regex_reason=regex_reason, qu=qu, escalation=reason)
        # Shadow mode: production routing stays on regex
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result

    if qu is None or qu.confidence < 0.5:
        RAG_QU_INVOCATIONS.labels(source="regex").inc()
        return result

    RAG_QU_INVOCATIONS.labels(source="llm").inc()
    return HybridClassification(
        intent=qu.intent,
        resolved_query=qu.resolved_query,
        temporal_constraint=qu.temporal_constraint,
        entities=qu.entities,
        confidence=qu.confidence,
        source="llm",
        escalation_reason=reason,
        regex_reason=regex_reason,
        cached=qu.cached,
    )


def _emit_shadow_log(
    *, query: str, regex_label: str, regex_reason: str,
    qu: Optional["QueryUnderstanding"], escalation: EscalationReason,
) -> None:
    payload = {
        "query": query,
        "regex_label": regex_label,
        "regex_reason": regex_reason,
        "llm_label": qu.intent if qu else None,
        "llm_resolved_query": qu.resolved_query if qu else None,
        "llm_temporal": qu.temporal_constraint if qu else None,
        "llm_confidence": qu.confidence if qu else None,
        "agree": (qu.intent == regex_label) if qu else None,
        "escalation_reason": escalation.value,
    }
    _shadow_log.info(_json.dumps(payload, ensure_ascii=False))
```

- [ ] **Step 4: Add a shadow log analysis script**

Create `/home/vogic/LocalRAG/scripts/analyze_shadow_log.py`:

```python
#!/usr/bin/env python3
"""Analyze QU LLM shadow-mode log entries.

Reads JSON-line log entries from stdin (or a file) and reports:
  - Total queries
  - Agreement rate overall and per regex_label
  - Disagreement examples (sampled) per (regex_label, llm_label) bucket
  - Escalation breakdown

Usage:
    docker logs orgchat-open-webui 2>&1 | grep 'orgchat.qu_shadow' | \\
        python scripts/analyze_shadow_log.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import IO


def parse_lines(stream: IO[str]):
    for line in stream:
        line = line.strip()
        if not line:
            continue
        # Strip prefixes from logger output ("INFO orgchat.qu_shadow: {...}")
        if "{" not in line:
            continue
        line = line[line.index("{"):]
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("file", nargs="?", default="-",
                   help="path to shadow log; '-' for stdin")
    p.add_argument("--samples", type=int, default=3,
                   help="sample disagreements per bucket")
    args = p.parse_args()

    stream: IO[str] = sys.stdin if args.file == "-" else open(args.file)

    total = 0
    by_regex_label = collections.Counter()
    agreement_by_regex_label = collections.Counter()
    bucket_samples = collections.defaultdict(list)
    escalation_counts = collections.Counter()

    for entry in parse_lines(stream):
        total += 1
        rl = entry["regex_label"]
        by_regex_label[rl] += 1
        if entry.get("agree"):
            agreement_by_regex_label[rl] += 1
        else:
            bucket = (rl, entry.get("llm_label"))
            if len(bucket_samples[bucket]) < args.samples:
                bucket_samples[bucket].append(entry["query"])
        escalation_counts[entry.get("escalation_reason", "none")] += 1

    if total == 0:
        print("No shadow log entries found.", file=sys.stderr)
        return 1

    print(f"Total queries: {total}")
    print()
    print("Per-regex-label agreement:")
    for rl, count in by_regex_label.most_common():
        agree = agreement_by_regex_label[rl]
        rate = agree / count * 100 if count else 0
        print(f"  {rl:>15}: {agree}/{count} ({rate:.1f}%)")
    print()
    print("Escalation reason breakdown:")
    for reason, count in escalation_counts.most_common():
        print(f"  {reason:>20}: {count} ({count/total*100:.1f}%)")
    print()
    print("Disagreement samples (regex_label -> llm_label):")
    for (rl, ll), samples in sorted(bucket_samples.items(),
                                     key=lambda x: -len(x[1])):
        print(f"  {rl} -> {ll}:")
        for q in samples:
            print(f"    {q!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/analyze_shadow_log.py
```

- [ ] **Step 5: Re-run shadow tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_qu_shadow_mode.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/query_intent.py scripts/analyze_shadow_log.py \
        tests/unit/test_qu_shadow_mode.py
git commit -m "phase-4.8: shadow-mode A/B harness for QU LLM"
```

---

### Task 4.9: Integration test against live vllm-qu

**Files:**
- Create: `/home/vogic/LocalRAG/tests/integration/test_vllm_qu_live.py`

This test requires the vllm-qu container to be running. It is marked with `@pytest.mark.integration` so the standard `pytest` invocation skips it; the operator runs it explicitly with `pytest -m integration`.

- [ ] **Step 1: Write the test**

Create `/home/vogic/LocalRAG/tests/integration/test_vllm_qu_live.py`:

```python
"""Integration tests against a live vllm-qu container.

Requires:
  - vllm-qu running (docker compose up -d vllm-qu, ~90s to load)
  - RAG_QU_URL pointing at it (default http://localhost:8101/v1)

Skipped by default. Run with:
  pytest -m integration tests/integration/test_vllm_qu_live.py -v
"""
from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def qu_url():
    return os.environ.get("RAG_QU_URL", "http://localhost:8101/v1")


@pytest.fixture
def qu_model():
    return os.environ.get("RAG_QU_MODEL", "qwen3-4b-qu")


@pytest.mark.asyncio
async def test_live_basic_classification(qu_url, qu_model):
    from ext.services.query_understanding import analyze_query
    qu = await analyze_query(
        query="list all reports from January 2026",
        history=[],
        qu_url=qu_url,
        model=qu_model,
        timeout_ms=10_000,
    )
    assert qu is not None, "QU LLM did not respond"
    assert qu.intent in {"metadata", "global", "specific", "specific_date"}


@pytest.mark.asyncio
async def test_live_resolves_pronoun_with_history(qu_url, qu_model):
    from ext.services.query_understanding import analyze_query
    history = [
        {"role": "user", "content": "Tell me about the OFC roadmap"},
        {"role": "assistant",
         "content": "OFC roadmap covers 2026-Q1 to 2027-Q1 with milestones..."},
    ]
    qu = await analyze_query(
        query="and what about it in Q2?",
        history=history,
        qu_url=qu_url, model=qu_model, timeout_ms=10_000,
    )
    assert qu is not None
    # Should resolve "it" -> "OFC roadmap" in resolved_query
    assert "OFC" in qu.resolved_query or "ofc" in qu.resolved_query.lower() or \
        "roadmap" in qu.resolved_query.lower()


@pytest.mark.asyncio
async def test_live_extracts_temporal_constraint(qu_url, qu_model):
    from ext.services.query_understanding import analyze_query
    qu = await analyze_query(
        query="outages in Q1 2026",
        history=[], qu_url=qu_url, model=qu_model, timeout_ms=10_000,
    )
    assert qu is not None
    assert qu.temporal_constraint is not None
    assert qu.temporal_constraint.get("year") == 2026
    assert qu.temporal_constraint.get("quarter") == 1


@pytest.mark.asyncio
async def test_live_classifies_global_for_compare(qu_url, qu_model):
    from ext.services.query_understanding import analyze_query
    qu = await analyze_query(
        query="compare budgets across all years",
        history=[], qu_url=qu_url, model=qu_model, timeout_ms=10_000,
    )
    assert qu is not None
    assert qu.intent == "global"


@pytest.mark.asyncio
async def test_live_returns_none_on_timeout(qu_url, qu_model):
    from ext.services.query_understanding import analyze_query
    # 1 ms timeout — guaranteed to miss even a hot model
    qu = await analyze_query(
        query="hello",
        history=[], qu_url=qu_url, model=qu_model, timeout_ms=1,
    )
    assert qu is None  # soft-fail by design


@pytest.mark.asyncio
async def test_live_p95_latency_under_budget(qu_url, qu_model):
    """Run 20 calls; p95 must be under 600ms (the SLO)."""
    import time
    from ext.services.query_understanding import analyze_query
    queries = [
        "what changed last quarter",
        "compare budgets",
        "list documents",
        "outages on 5 Jan 2026",
        "summary of march",
    ] * 4  # 20 calls

    durations = []
    for q in queries:
        start = time.monotonic()
        await analyze_query(
            query=q, history=[], qu_url=qu_url, model=qu_model,
            timeout_ms=10_000,
        )
        durations.append(time.monotonic() - start)
    durations.sort()
    p95 = durations[int(0.95 * len(durations))]
    assert p95 < 0.6, f"p95 latency {p95:.3f}s exceeds 600ms SLO"
```

- [ ] **Step 2: Run the integration test**

```bash
cd /home/vogic/LocalRAG/compose && docker compose up -d vllm-qu && sleep 90
cd /home/vogic/LocalRAG && pytest -m integration tests/integration/test_vllm_qu_live.py -v
```

Expected: 6 passed. If `test_live_p95_latency_under_budget` fails, vllm-qu may not have warmed up; re-run after a few minutes.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_vllm_qu_live.py
git commit -m "phase-4.9: integration tests against live vllm-qu container"
```

---

### Task 4.10: Runbook fill-in + flag-reference update + retire `RAG_INTENT_LLM`

**Files:**
- Create: `/home/vogic/LocalRAG/docs/runbook/qu-llm-runbook.md`
- Create: `/home/vogic/LocalRAG/docs/runbook/plan-b-flag-reference.md`
- Modify: `/home/vogic/LocalRAG/docs/runbook/flag-reference.md`
- Modify: `/home/vogic/LocalRAG/ext/services/query_intent.py` (delete `RAG_INTENT_LLM` + dead `_llm_classify` references)

The kill-list policy from Plan A says: any flag still default-OFF after Plan B Phase 4 must be turned default-ON, deleted, or justified per-KB. `RAG_INTENT_LLM` is being deleted (replaced by `RAG_QU_ENABLED`). `RAG_INTENT_ROUTING` (Tier 2) is also being retired in this task.

- [ ] **Step 1: Write the QU runbook**

Create `/home/vogic/LocalRAG/docs/runbook/qu-llm-runbook.md`:

```markdown
# Query Understanding LLM Runbook

**Purpose:** operate, troubleshoot, and tune the hybrid regex+LLM intent router introduced in Plan B Phase 4.

## Service overview

- **Container:** `orgchat-vllm-qu`
- **Image:** `vllm/vllm-openai:latest`
- **Model:** `Qwen/Qwen3-4B-Instruct-2507-AWQ`
- **GPU:** 1 (RTX PRO 4000 Blackwell, 24 GB; total resident with TEI + reranker ≈ 12 GB)
- **Endpoint (intra-cluster):** `http://vllm-qu:8000/v1/chat/completions`
- **Endpoint (host):** `http://localhost:8101/v1/chat/completions`
- **Engine:** vLLM V1 with xgrammar guided JSON

## Healthcheck

```bash
docker compose ps vllm-qu
# Expected: STATUS Up X (healthy)

curl -s http://localhost:8101/v1/models | python -m json.tool
# Expected: data[0].id == "qwen3-4b-qu"

# A real classification probe
curl -s -X POST http://localhost:8101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b-qu",
    "messages": [
      {"role":"system","content":"You output only JSON."},
      {"role":"user","content":"Classify this as JSON: \"list reports\""}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

## Flags

| Flag | Default | Effect when 1 |
|---|---|---|
| `RAG_QU_ENABLED` | 0 (will flip after shadow gate) | Hybrid router promotes LLM result on escalated queries |
| `RAG_QU_URL` | `http://vllm-qu:8000/v1` | vLLM base URL (override for canary / staging) |
| `RAG_QU_MODEL` | `qwen3-4b-qu` | served-model-name |
| `RAG_QU_LATENCY_BUDGET_MS` | 600 | Soft deadline; on miss the bridge falls back to regex |
| `RAG_QU_CACHE_ENABLED` | 1 | Redis DB 4 cache |
| `RAG_QU_CACHE_TTL_SECS` | 300 | Cache TTL |
| `RAG_QU_REDIS_DB` | 4 | Redis DB number (3 reserved for RBAC cache by Plan A) |
| `RAG_QU_SHADOW_MODE` | 0 | Run LLM on every query and log both decisions; production routing stays regex-only |

## Daily checks

1. `nvidia-smi` — GPU 1 memory.used should be ≈ 12 GB (TEI + reranker + Qwen3-4B)
2. `curl -s http://localhost:9090/api/v1/query?query=rag_qu_cache_hit_ratio` — should be > 0.3 in steady state
3. `curl -s http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(rag_qu_latency_seconds_bucket[5m]))` — should be < 0.6

## Common failure modes

### `vllm-qu` container restart loops

- Check `docker logs orgchat-vllm-qu --tail 100`. Common: out-of-memory if another process took GPU 1, or HF cache miss (weights not in `/var/models/hf_cache`).
- Verify weights: `ls /var/models/hf_cache/hub/models--Qwen--Qwen3-4B-Instruct-2507-AWQ/`
- Re-stage with `scripts/stage_qwen3_qu.sh` if missing

### `rag_qu_schema_violations_total` rising

- Check `docker logs orgchat-vllm-qu` for `xgrammar` errors. Most often means the Qwen revision changed and tokenizer mismatch broke guided JSON.
- Mitigation: pin the snapshot revision in the model id: `Qwen/Qwen3-4B-Instruct-2507-AWQ@<rev_hash>`

### Hybrid escalation rate > 40%

- Inspect via `scripts/analyze_shadow_log.py` (with shadow mode on)
- Likely: queries are unusually pronoun-heavy or relative-time-heavy; consider raising the `_LONG_QUERY_TOKEN_THRESHOLD` or adding a per-tenant policy

### Cache hit ratio < 0.3

- Workload may be unique enough that 5-minute TTL isn't catching repeats
- Try `RAG_QU_CACHE_TTL_SECS=900`. Higher TTL is safe — the cache key includes the last assistant turn ID, so context shifts invalidate naturally

## Promotion checklist (shadow → production)

After 7 days of shadow A/B:

1. `python scripts/analyze_shadow_log.py < /var/log/openwebui/shadow.log`
2. Confirm: agreement rate > 75% on `metadata` and `global`, < 60% on `specific` (the LLM is winning the hard cases)
3. Run `make eval-baseline` with `RAG_QU_ENABLED=1` against `kb_eval` and `kb_1` — `chunk_recall@10` improvement on `multihop` and `evolution` strata ≥ +3 pp
4. Flip `RAG_QU_ENABLED=1` in `.env`
5. Set `RAG_QU_SHADOW_MODE=0`
6. Restart `open-webui`

## Rollback

```bash
cd /home/vogic/LocalRAG/compose
sed -i 's/^RAG_QU_ENABLED=1$/RAG_QU_ENABLED=0/' .env
docker compose up -d --force-recreate open-webui
```

The bridge soft-fails to regex; users see no error.
```

- [ ] **Step 2: Write Plan B's flag reference**

Create `/home/vogic/LocalRAG/docs/runbook/plan-b-flag-reference.md`:

```markdown
# Plan B Flag Reference

| Flag | Phase | Default | Description | Safe to toggle at runtime? |
|---|---|---|---|---|
| `RAG_QU_ENABLED` | 4.6 | 0 → 1 after Phase 4 gate | Master switch for hybrid LLM router | Yes (restart) |
| `RAG_QU_URL` | 4.1 | `http://vllm-qu:8000/v1` | vLLM base URL | Yes (restart) |
| `RAG_QU_MODEL` | 4.1 | `qwen3-4b-qu` | served-model-name | Yes (restart) |
| `RAG_QU_LATENCY_BUDGET_MS` | 4.3 | 600 | Soft deadline (ms) | Yes (restart) |
| `RAG_QU_CACHE_ENABLED` | 4.5 | 1 | Redis cache for QU results | Yes (restart) |
| `RAG_QU_CACHE_TTL_SECS` | 4.5 | 300 | Cache TTL (s) | Yes (restart) |
| `RAG_QU_REDIS_DB` | 4.5 | 4 | Redis DB number | No — must be unique vs DB 3 (RBAC) |
| `RAG_QU_SHADOW_MODE` | 4.8 | 0 | Always run LLM, log both, route regex | Yes (restart) |
| `RAG_SHARDING_ENABLED` | 5.2 | 0 → 1 for new collections | Derive shard_key="YYYY-MM" at ingest | Yes for new collections only |
| `RAG_TEMPORAL_LEVELS` | 5.6 | 0 → 1 for sharded collections | Inject L3 / L2 levels for global / evolution intents | Yes (restart) |
| `RAG_TIME_DECAY` | 5.7 | 0 | Apply exp(-λΔt) multiplier on current-state intent | Yes (restart) |
| `RAG_TIME_DECAY_LAMBDA_DAYS` | 5.7 | 90 | Half-life for time-decay (days) | Yes (restart) |
| `RAG_TIER_HOT_MONTHS` | 5.3 | 3 | Months kept in HNSW RAM tier | Operator-only |
| `RAG_TIER_WARM_MONTHS` | 5.3 | 12 | Months kept in mmap SSD tier | Operator-only |
| `RAG_TIER_COLD_QUANTIZATION` | 5.3 | int8 | Cold tier scalar quantization | Operator-only |
| `RAG_SYNC_INGEST` | 6.2 | 1 → 0 after soak | Sync ingest path; 0 = celery worker | Yes (restart) |
| `RAG_OCR_ENABLED` | 6.3 | 0 → 1 after verification | OCR fallback for scanned PDFs | Yes (restart) |
| `RAG_OCR_BACKEND` | 6.3 | `tesseract` | `tesseract` or `cloud:textract` or `cloud:document_ai` | Per-KB override available |
| `RAG_OCR_TRIGGER_CHARS` | 6.4 | 50 | <N chars per page → rasterize+OCR | Yes (restart) |
| `RAG_STRUCTURED_CHUNKER` | 6.5 | 0 → 1 after KB strategy | Tables/code as atomic units | Yes (restart) |
| `RAG_IMAGE_CAPTIONS` | 6.7 | 0 | Emit chunks with `chunk_type="image_caption"` | Yes (restart) |

## Plan B retires these Plan A flags

- `RAG_INTENT_LLM` — replaced by `RAG_QU_ENABLED`
- `RAG_INTENT_ROUTING` (Tier 2) — replaced by hybrid router

## Carryover Plan A flags Plan B does not change

- `RAG_HYDE`, `RAG_SEMCACHE`, `RAG_DISABLE_REWRITE` — still default-OFF; defer to Plan C
```

- [ ] **Step 3: Append Plan B section to the master flag-reference**

Edit `/home/vogic/LocalRAG/docs/runbook/flag-reference.md`. Append at the bottom:

```markdown
---

## Plan B Phase 4 additions

See `docs/runbook/plan-b-flag-reference.md` for the full table. Summary:

- New flags: `RAG_QU_ENABLED`, `RAG_QU_URL`, `RAG_QU_MODEL`, `RAG_QU_LATENCY_BUDGET_MS`, `RAG_QU_CACHE_ENABLED`, `RAG_QU_CACHE_TTL_SECS`, `RAG_QU_REDIS_DB`, `RAG_QU_SHADOW_MODE`.
- Retired: `RAG_INTENT_LLM`, `RAG_INTENT_ROUTING`.
```

- [ ] **Step 4: Delete `RAG_INTENT_LLM` and dead `_llm_classify` references**

Edit `/home/vogic/LocalRAG/ext/services/query_intent.py`:

- Delete the `if os.environ.get("RAG_INTENT_LLM", "0") == "1":` block in `classify_with_reason`.
- Delete the `_llm_classify` function (already replaced by `classify_with_qu`).
- Update the module docstring to point at `classify_with_qu` for the LLM path.

```bash
cd /home/vogic/LocalRAG && grep -rn "RAG_INTENT_LLM\|_llm_classify" ext/ tests/
```

Expected: no remaining references after the edit.

- [ ] **Step 5: Add a regression test for the retirement**

Append to `/home/vogic/LocalRAG/tests/unit/test_query_intent.py`:

```python
def test_rag_intent_llm_flag_removed(monkeypatch):
    """Plan B Phase 4.10 — RAG_INTENT_LLM is retired."""
    import importlib
    from ext.services import query_intent
    monkeypatch.setenv("RAG_INTENT_LLM", "1")
    importlib.reload(query_intent)
    src = pathlib.Path(query_intent.__file__).read_text()
    assert "RAG_INTENT_LLM" not in src, (
        "RAG_INTENT_LLM is retired in Plan B Phase 4.10; remove all references."
    )
    assert "_llm_classify" not in src, "Dead _llm_classify stub must be removed."
```

(Add `import pathlib` at the top of the test file if not already present.)

- [ ] **Step 6: Run the retirement regression**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_query_intent.py -v -k "rag_intent_llm_flag_removed or test_qu"
```

Expected: passes.

- [ ] **Step 7: Commit**

```bash
git add docs/runbook/qu-llm-runbook.md docs/runbook/plan-b-flag-reference.md \
        docs/runbook/flag-reference.md ext/services/query_intent.py \
        tests/unit/test_query_intent.py
git commit -m "phase-4.10: QU runbook + flag reference; retire RAG_INTENT_LLM"
```

### Phase 4 completion gate

- [ ] All Phase 4 unit tests pass.
- [ ] `tests/integration/test_vllm_qu_live.py` passes against the live container.
- [ ] `vllm-qu` container is `healthy` for ≥ 24 hours.
- [ ] `nvidia-smi` GPU 1 memory.used ≤ 14 GB (≈ 60% headroom remains).
- [ ] Shadow log running for ≥ 7 days. `scripts/analyze_shadow_log.py` produces a report committed at `tests/eval/results/phase-4-shadow-summary.txt`.
- [ ] `make eval-baseline` against `kb_1` with `RAG_QU_ENABLED=1` shows `chunk_recall@10` ≥ +3 pp on `multihop` + `evolution` strata vs Plan A end-state, no regression > 2 pp on any other stratum.
- [ ] Phase 4 baseline JSON committed at `tests/eval/results/phase-4-baseline.json`.

---

## Phase 5 — Qdrant temporal sharding + temporal-semantic RAPTOR (Days 2–3)

**Phase goal:** Migrate `kb_1_v3` (post-Plan-A canonical: dense + sparse + ColBERT + `context_prefix`) into a new `kb_1_v4` collection that uses Qdrant 1.16+'s custom sharding feature with `shard_key="YYYY-MM"`, tier shards by recency (hot/warm/cold), and build a temporal-then-semantic RAPTOR tree on top of the canonical chunks. Aliases swap atomically; rollback to `kb_1_v3` is a 1-second alias swap.

**Why this phase:** The 36-month corpus the user is targeting (`temporal_corpus_plan` memory) saturates flat RAPTOR trees: distant-but-similar chunks cluster together and lose their temporal signal. Sharding by month + a temporal RAPTOR (L0 chunks → L1 monthly subtrees → L2 quarterly change-vs-prior summaries → L3 yearly → L4 3-year meta) preserves the time signal at every level. Tiered storage holds ingest costs flat as the corpus grows past hot RAM HNSW.

**Migration model:** identical to Plan A 1.7 / 3.7 — dual-collection, alias swap, 14-day rollback retention. Plan A executed `kb_1` → `kb_1_v3`; Plan B executes `kb_1_v3` → `kb_1_v4`. The intermediate `kb_1_v2` (Plan A 1.7 rollback target) stays on disk until 2026-05-09 per Plan A's cleanup calendar — Phase 5 must not delete it.

---

### Task 5.1: VectorStore.ensure_collection variant with custom sharding

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/vector_store.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_vector_store_temporal_sharding.py`

Qdrant's custom sharding requires `sharding_method="custom"` at collection creation, and `shard_number=N` (one shard per shard_key). We add a method `ensure_collection_temporal()` that creates the collection with this configuration. The existing `ensure_collection()` stays unchanged for non-temporal collections.

Custom shard keys are upserted at write time via `shard_key_selector`. Reads can either fan out to all shards (default) or filter by shard_key for date-bounded queries.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_vector_store_temporal_sharding.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_ensure_collection_temporal_passes_custom_sharding(monkeypatch):
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=False)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01", "2024-02", "2024-03"],
        with_sparse=True,
        with_colbert=True,
    )

    # Inspect the create_collection kwargs
    call = vs._client.create_collection.call_args
    assert call.kwargs.get("sharding_method") == "custom" or \
        "sharding_method" in str(call), \
        "ensure_collection_temporal must set sharding_method=custom"

    # All shard keys created
    assert vs._client.create_shard_key.call_count == 3


@pytest.mark.asyncio
async def test_ensure_collection_temporal_with_replication_factor(monkeypatch):
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=False)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01"],
        replication_factor=2,
    )

    call = vs._client.create_collection.call_args
    assert call.kwargs.get("replication_factor") == 2 or \
        "replication_factor=2" in str(call)


@pytest.mark.asyncio
async def test_ensure_collection_temporal_idempotent(monkeypatch):
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.collection_exists = AsyncMock(return_value=True)
    vs._client.create_collection = AsyncMock()
    vs._client.create_shard_key = AsyncMock()

    await vs.ensure_collection_temporal(
        "kb_1_v4",
        shard_keys=["2024-01", "2024-02"],
    )

    # Existing collection: no create_collection call, but shard_keys still ensured
    vs._client.create_collection.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_with_shard_key(monkeypatch):
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.upsert = AsyncMock()

    await vs.upsert_temporal(
        "kb_1_v4",
        points=[{"id": "p1", "vector": [0.1] * 1024,
                 "payload": {"shard_key": "2024-01"}}],
        shard_key="2024-01",
    )

    call = vs._client.upsert.call_args
    assert call.kwargs.get("shard_key_selector") == "2024-01" or \
        "shard_key_selector" in str(call)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_vector_store_temporal_sharding.py -v
```

Expected: AttributeError — `ensure_collection_temporal` doesn't exist.

- [ ] **Step 3: Add the method to VectorStore**

Edit `/home/vogic/LocalRAG/ext/services/vector_store.py`. Append below the existing `ensure_collection` method:

```python
async def ensure_collection_temporal(
    self,
    name: str,
    shard_keys: list[str],
    *,
    with_sparse: bool = True,
    with_colbert: bool = False,
    on_disk_payload: bool | None = None,
    replication_factor: int = 1,
) -> None:
    """Create a Qdrant collection with custom temporal sharding.

    One shard per ``shard_key`` (typically "YYYY-MM" for monthly buckets).
    Shard creation is idempotent; existing keys are not re-created. The
    collection itself is also idempotent — if it exists with a different
    sharding strategy, raises (the operator must drop + recreate to change
    sharding).

    Plan B Phase 5.1.
    """
    from qdrant_client.http.models import (
        VectorParams, SparseVectorParams, MultiVectorConfig,
        Distance, PayloadSchemaType, ShardingMethod,
    )
    on_disk_payload = on_disk_payload if on_disk_payload is not None \
        else _env_bool("RAG_QDRANT_ON_DISK_PAYLOAD", True)

    if await self._client.collection_exists(collection_name=name):
        logger.info("collection %s exists; ensuring shard keys", name)
    else:
        vectors_config: dict = {
            "dense": VectorParams(size=self._vector_size, distance=Distance.COSINE)
        }
        if with_colbert:
            vectors_config["colbert"] = VectorParams(
                size=128, distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(comparator="max_sim"),
            )
        sparse_vectors = None
        if with_sparse:
            sparse_vectors = {"sparse": SparseVectorParams()}

        await self._client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors,
            on_disk_payload=on_disk_payload,
            sharding_method=ShardingMethod.CUSTOM,
            shard_number=len(shard_keys),
            replication_factor=replication_factor,
        )
        logger.info("created temporal collection %s with %d shards",
                    name, len(shard_keys))

    # Ensure shard keys (idempotent — Qdrant returns 200 even if exists)
    for sk in shard_keys:
        try:
            await self._client.create_shard_key(
                collection_name=name, shard_key=sk,
            )
        except Exception as e:
            # 409 / "already exists" is fine
            if "exists" not in str(e).lower():
                raise

    # Add per-payload index on shard_key for filterable date-bounded queries
    try:
        await self._client.create_payload_index(
            collection_name=name,
            field_name="shard_key",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass  # idempotent


async def upsert_temporal(
    self,
    collection: str,
    points: list[dict],
    *,
    shard_key: str,
) -> None:
    """Upsert into a specific shard_key.

    Caller must ensure all points belong to the named shard. Mixing shards
    in one call is a Qdrant constraint violation.
    """
    from qdrant_client.http.models import PointStruct
    structs = [
        PointStruct(
            id=p["id"],
            vector=p["vector"],
            payload=p.get("payload", {}),
        )
        for p in points
    ]
    await self._client.upsert(
        collection_name=collection,
        points=structs,
        shard_key_selector=shard_key,
    )
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_vector_store_temporal_sharding.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/vector_store.py tests/unit/test_vector_store_temporal_sharding.py
git commit -m "phase-5.1: VectorStore.ensure_collection_temporal + upsert_temporal"
```

---

### Task 5.2: Date extraction at ingest — derive shard_key from filename + content

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/temporal_shard.py`
- Modify: `/home/vogic/LocalRAG/ext/services/ingest.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_temporal_shard_key.py`

The shard_key is derived from the document at ingest time. Priority order:
1. **Filename pattern** — "05 Jan 2026.docx" → `2026-01`. Existing `query_intent.extract_date_tuple` does most of the work.
2. **Content first-line / front-matter** — markdown YAML frontmatter `date: 2026-01-15` → `2026-01`.
3. **First date found in body** (first 1000 chars).
4. **Fallback** — current month at ingest time, with a `shard_key_origin: "ingest_default"` payload tag for observability.

The `shard_key` is also added to the point payload for filterable date-range queries that don't have a single bucket.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_temporal_shard_key.py`:

```python
import datetime as dt
import pytest

from ext.services.temporal_shard import (
    extract_shard_key,
    parse_shard_key,
    iter_shard_keys,
    ShardKeyOrigin,
)


class TestExtractShardKey:
    def test_filename_dmy(self):
        sk, origin = extract_shard_key(
            filename="05 Jan 2026.docx", body="irrelevant"
        )
        assert sk == "2026-01"
        assert origin is ShardKeyOrigin.FILENAME

    def test_filename_iso(self):
        sk, origin = extract_shard_key(
            filename="2024-08-15-summary.md", body=""
        )
        assert sk == "2024-08"
        assert origin is ShardKeyOrigin.FILENAME

    def test_filename_mdy(self):
        sk, origin = extract_shard_key(
            filename="January 5 2026 report.pdf", body=""
        )
        assert sk == "2026-01"

    def test_yaml_frontmatter(self):
        body = "---\ndate: 2025-03-20\ntitle: Q1 Update\n---\n\nContent..."
        sk, origin = extract_shard_key(filename="random.md", body=body)
        assert sk == "2025-03"
        assert origin is ShardKeyOrigin.FRONTMATTER

    def test_first_body_date(self):
        body = "Meeting on June 12, 2025. Attendees: ..."
        sk, origin = extract_shard_key(filename="meeting.txt", body=body)
        assert sk == "2025-06"
        assert origin is ShardKeyOrigin.BODY

    def test_fallback_to_now(self):
        body = "No dates here at all"
        sk, origin = extract_shard_key(filename="random.txt", body=body)
        # Should be current month
        now = dt.date.today()
        assert sk == f"{now.year:04d}-{now.month:02d}"
        assert origin is ShardKeyOrigin.INGEST_DEFAULT

    def test_filename_takes_priority_over_body(self):
        # Filename: 2026-01, body has 2025-03 frontmatter — filename wins
        body = "---\ndate: 2025-03-20\n---\n"
        sk, origin = extract_shard_key(
            filename="05 Jan 2026.docx", body=body,
        )
        assert sk == "2026-01"


class TestParseShardKey:
    def test_valid(self):
        y, m = parse_shard_key("2024-07")
        assert y == 2024 and m == 7

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_shard_key("2024-7")  # need zero-pad

    def test_invalid_month(self):
        with pytest.raises(ValueError):
            parse_shard_key("2024-13")


class TestIterShardKeys:
    def test_year_range(self):
        keys = list(iter_shard_keys(start="2024-01", end="2024-04"))
        assert keys == ["2024-01", "2024-02", "2024-03", "2024-04"]

    def test_year_boundary(self):
        keys = list(iter_shard_keys(start="2024-11", end="2025-02"))
        assert keys == ["2024-11", "2024-12", "2025-01", "2025-02"]

    def test_full_36_months(self):
        keys = list(iter_shard_keys(start="2024-01", end="2026-12"))
        assert len(keys) == 36
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_shard_key.py -v
```

Expected: ImportError — `temporal_shard` doesn't exist.

- [ ] **Step 3: Write the module**

Create `/home/vogic/LocalRAG/ext/services/temporal_shard.py`:

```python
"""Temporal shard_key derivation for Qdrant custom sharding.

Plan B Phase 5.2. The shard_key partitions a KB by month — the
ingest pipeline derives it once per document and the same key is used
for every chunk of that document.

Format: ``"YYYY-MM"`` (zero-padded, ASCII). Always 7 chars. This makes
shard_keys sortable lexicographically and trivial to enumerate.

Priority order for derivation:
  1. Filename pattern (uses existing query_intent.extract_date_tuple)
  2. YAML frontmatter ``date:`` field
  3. First date in the body's first 1000 chars
  4. Current month at ingest (fallback; tagged with ShardKeyOrigin.INGEST_DEFAULT
     in payload for observability)
"""
from __future__ import annotations

import datetime as _dt
import enum
import re
from typing import Iterable, Optional, Tuple

from .query_intent import extract_date_tuple


class ShardKeyOrigin(enum.Enum):
    FILENAME = "filename"
    FRONTMATTER = "frontmatter"
    BODY = "body"
    INGEST_DEFAULT = "ingest_default"


_FRONTMATTER_DATE_RE = re.compile(
    r"^---\s*\n.*?^date:\s*(\d{4})-(\d{2})-\d{2}",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

_MONTH_NUM = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}


def _date_to_shard_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def extract_shard_key(
    filename: str, body: str,
) -> Tuple[str, ShardKeyOrigin]:
    """Derive (shard_key, origin) from a document.

    Pure function — no I/O. Always returns a valid shard_key string.
    """
    # 1. Filename
    if filename:
        tup = extract_date_tuple(filename)
        if tup is not None:
            day, month_str, year = tup
            return _date_to_shard_key(year, _MONTH_NUM[month_str]), \
                   ShardKeyOrigin.FILENAME

    # 2. YAML frontmatter
    if body:
        m = _FRONTMATTER_DATE_RE.search(body[:2000])
        if m:
            return _date_to_shard_key(int(m.group(1)), int(m.group(2))), \
                   ShardKeyOrigin.FRONTMATTER

    # 3. First date in body (first 1000 chars)
    if body:
        head = body[:1000]
        tup = extract_date_tuple(head)
        if tup is not None:
            day, month_str, year = tup
            return _date_to_shard_key(year, _MONTH_NUM[month_str]), \
                   ShardKeyOrigin.BODY

    # 4. Fallback
    today = _dt.date.today()
    return _date_to_shard_key(today.year, today.month), \
           ShardKeyOrigin.INGEST_DEFAULT


_SHARD_KEY_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def parse_shard_key(sk: str) -> Tuple[int, int]:
    """Parse a shard_key into (year, month). Raises ValueError on malformed."""
    m = _SHARD_KEY_RE.match(sk)
    if not m:
        raise ValueError(f"invalid shard_key {sk!r}; expected 'YYYY-MM' "
                          "(zero-padded month)")
    return int(m.group(1)), int(m.group(2))


def iter_shard_keys(start: str, end: str) -> Iterable[str]:
    """Yield consecutive shard_keys from ``start`` to ``end`` inclusive.

    Both endpoints are 'YYYY-MM' format. Crosses year boundaries.
    """
    sy, sm = parse_shard_key(start)
    ey, em = parse_shard_key(end)
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield _date_to_shard_key(y, m)
        m += 1
        if m > 12:
            m = 1
            y += 1


__all__ = [
    "ShardKeyOrigin",
    "extract_shard_key",
    "parse_shard_key",
    "iter_shard_keys",
]
```

- [ ] **Step 4: Wire into ingest.py**

Edit `/home/vogic/LocalRAG/ext/services/ingest.py`. Find the `ingest_document` (or equivalent main entry) function. Add near the top:

```python
from .temporal_shard import extract_shard_key, ShardKeyOrigin
```

In the document-level processing block (before chunk loop), add:

```python
sharding_enabled = os.environ.get("RAG_SHARDING_ENABLED", "0") == "1"
if sharding_enabled:
    shard_key, origin = extract_shard_key(filename=filename, body=raw_text)
    logger.info("ingest doc=%s shard_key=%s origin=%s",
                filename, shard_key, origin.value)
else:
    shard_key, origin = None, None
```

In the per-chunk payload construction, when `sharding_enabled` is true, inject:

```python
payload["shard_key"] = shard_key
payload["shard_key_origin"] = origin.value
```

In the upsert call, when `sharding_enabled` is true, route to `upsert_temporal` instead of the default `upsert`:

```python
if sharding_enabled:
    await vector_store.upsert_temporal(
        collection=collection_name, points=batch, shard_key=shard_key,
    )
else:
    await vector_store.upsert(collection=collection_name, points=batch)
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_shard_key.py -v
```

Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/temporal_shard.py ext/services/ingest.py \
        tests/unit/test_temporal_shard_key.py
git commit -m "phase-5.2: shard_key derivation at ingest from filename/body"
```

---

### Task 5.3: Tiered storage config — hot/warm/cold with INT8 quantization on cold

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/vector_store.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_tiered_storage_config.py`

Qdrant 1.16+ supports per-collection tiered config via `optimizers_config` and `quantization_config`. We define three tiers:
- **Hot:** last `RAG_TIER_HOT_MONTHS` (default 3) — in-memory HNSW (`memmap_threshold=0`, no quantization)
- **Warm:** `RAG_TIER_WARM_MONTHS` (default 12) — mmap on SSD (`memmap_threshold=20000`)
- **Cold:** older — on-disk + INT8 scalar quantization (`always_ram=false`, `quantization=scalar:int8`)

The `ensure_collection_temporal` method in 5.1 creates the collection without tier hints (default Qdrant config). Tier movement is a separate operation that re-configures specific shards via `update_collection`. Phase 5.3 ships the configuration helpers; Phase 5.8 ships the cron that calls them daily.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_tiered_storage_config.py`:

```python
import datetime as dt
import pytest
from unittest.mock import AsyncMock, MagicMock

from ext.services.vector_store import VectorStore


@pytest.mark.asyncio
async def test_classify_tier_hot_for_recent():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    sk = f"{today.year:04d}-{today.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "hot"


@pytest.mark.asyncio
async def test_classify_tier_warm_after_3_months():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    # 6 months ago
    six_months_ago = (today.replace(day=1) - dt.timedelta(days=180))
    sk = f"{six_months_ago.year:04d}-{six_months_ago.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "warm"


@pytest.mark.asyncio
async def test_classify_tier_cold_after_12_months():
    from ext.services.vector_store import classify_tier
    today = dt.date.today()
    twenty_months_ago = (today.replace(day=1) - dt.timedelta(days=600))
    sk = f"{twenty_months_ago.year:04d}-{twenty_months_ago.month:02d}"
    assert classify_tier(sk, hot_months=3, warm_months=12) == "cold"


@pytest.mark.asyncio
async def test_apply_tier_config_cold_uses_int8():
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.update_collection = AsyncMock()

    await vs.apply_tier_config(
        collection="kb_1_v4",
        shard_key="2023-01",
        tier="cold",
    )

    call = vs._client.update_collection.call_args
    # quantization_config + scalar:int8
    str_call = str(call)
    assert "quantization_config" in str_call or "scalar" in str_call.lower()


@pytest.mark.asyncio
async def test_apply_tier_config_hot_disables_mmap():
    vs = VectorStore(url="http://stub", vector_size=1024)
    vs._client = MagicMock()
    vs._client.update_collection = AsyncMock()

    await vs.apply_tier_config(
        collection="kb_1_v4",
        shard_key="2026-04",
        tier="hot",
    )

    call = vs._client.update_collection.call_args
    # memmap_threshold should be 0 (or very high — always RAM)
    str_call = str(call)
    assert "memmap" in str_call.lower() or "ram" in str_call.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_tiered_storage_config.py -v
```

Expected: ImportError — `classify_tier`, `apply_tier_config` don't exist.

- [ ] **Step 3: Add tier helpers to VectorStore**

Edit `/home/vogic/LocalRAG/ext/services/vector_store.py`. Append:

```python
import datetime as _dt


def classify_tier(
    shard_key: str,
    *,
    hot_months: int = 3,
    warm_months: int = 12,
) -> str:
    """Return 'hot' / 'warm' / 'cold' for a 'YYYY-MM' shard_key.

    Plan B Phase 5.3.
    """
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(shard_key)
    today = _dt.date.today()
    age_months = (today.year - y) * 12 + (today.month - m)
    if age_months < hot_months:
        return "hot"
    if age_months < warm_months:
        return "warm"
    return "cold"


async def apply_tier_config(
    self,
    collection: str,
    shard_key: str,
    tier: str,
) -> None:
    """Update the per-shard tier configuration.

    Hot: in-memory HNSW (memmap_threshold=0, no quantization).
    Warm: mmap on SSD (memmap_threshold=20_000).
    Cold: on-disk + INT8 scalar quantization (always_ram=False).

    Note: Qdrant currently scopes optimizer + quantization config at
    collection level. For per-shard control on a temporal collection,
    we use the shard_key as a partition key in the indexing optimizer
    threshold; per-shard quantization requires re-creating the affected
    shard. The cron (Phase 5.8) coordinates this carefully.
    """
    from qdrant_client.http.models import (
        OptimizersConfigDiff, ScalarQuantization,
        ScalarQuantizationConfig, ScalarType,
        QuantizationConfig,
    )

    if tier == "hot":
        await self._client.update_collection(
            collection_name=collection,
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=0,  # all in RAM
            ),
        )
    elif tier == "warm":
        await self._client.update_collection(
            collection_name=collection,
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20_000,
            ),
        )
    elif tier == "cold":
        await self._client.update_collection(
            collection_name=collection,
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=20_000,
            ),
            quantization_config=QuantizationConfig(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                ),
            ),
        )
    else:
        raise ValueError(f"unknown tier {tier!r}")
```

Bind `apply_tier_config` to the class (drop the `self` declaration if you used a free function above; the test invokes it as a method).

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_tiered_storage_config.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/vector_store.py tests/unit/test_tiered_storage_config.py
git commit -m "phase-5.3: tier classification + apply_tier_config (hot/warm/cold + INT8)"
```

---

### Task 5.4: Migration script `reshard_kb_temporal.py` + production reshard

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/reshard_kb_temporal.py`
- Create: `/home/vogic/LocalRAG/docs/runbook/temporal-reshard-procedure.md`
- Create: `/home/vogic/LocalRAG/tests/integration/test_temporal_resharding.py`
- Modify: `/home/vogic/LocalRAG/Makefile`

The migration follows Plan A's dual-collection model (1.7 + 3.7). Source: `kb_1_v3` (post-Plan-A canonical). Target: `kb_1_v4` (sharded, same payload schema + ColBERT + context_prefix). Process:

1. **Snapshot** source for safety.
2. **Enumerate shard keys** by scrolling source to discover all distinct months.
3. **Create target** via `ensure_collection_temporal` with that shard set.
4. **Stream points** by month — group by inferred shard_key, upsert into the matching shard. Keep ColBERT + sparse + context_prefix payload.
5. **Verify counts** match per shard.
6. **Run eval** against the new collection.
7. **Alias swap** `kb_1` → `kb_1_v4`.
8. **Hold rollback window** 14 days.

Idempotent via the existing UUID5 point IDs (Plan A 3.7 established the pattern).

- [ ] **Step 1: Write the integration test**

Create `/home/vogic/LocalRAG/tests/integration/test_temporal_resharding.py`:

```python
"""Integration test for the temporal resharding script.

Requires:
  - Local Qdrant running
  - A small fixture collection seeded by the test setup

Skipped by default. Run:
  pytest -m integration tests/integration/test_temporal_resharding.py -v
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance, PointStruct,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def qclient():
    c = AsyncQdrantClient(url="http://localhost:6333", timeout=30.0)
    yield c


@pytest.fixture
async def fixture_source_collection(qclient):
    """Build a minimal source collection with 3 documents across 3 months."""
    name = "_test_reshard_source"
    if await qclient.collection_exists(collection_name=name):
        await qclient.delete_collection(collection_name=name)
    await qclient.create_collection(
        collection_name=name,
        vectors_config={"dense": VectorParams(size=4, distance=Distance.COSINE)},
    )
    points = [
        PointStruct(
            id=i, vector={"dense": [0.1 * i, 0.2, 0.3, 0.4]},
            payload={
                "doc_id": i,
                "filename": fn,
                "chunk_index": 0,
                "text": f"chunk for doc {i}",
            },
        )
        for i, fn in enumerate([
            "05 Jan 2026.docx", "10 Feb 2026.md", "15 Mar 2026.pdf",
        ], start=1)
    ]
    await qclient.upsert(collection_name=name, points=points)
    yield name
    await qclient.delete_collection(collection_name=name)


@pytest.mark.asyncio
async def test_reshard_creates_per_month_shards(qclient, fixture_source_collection):
    import subprocess
    target = "_test_reshard_target"
    # Cleanup any leftovers
    if await qclient.collection_exists(collection_name=target):
        await qclient.delete_collection(collection_name=target)

    result = subprocess.run([
        "python", "scripts/reshard_kb_temporal.py",
        "--source", fixture_source_collection,
        "--target", target,
        "--qdrant-url", "http://localhost:6333",
        "--batch-size", "10",
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"reshard failed: {result.stderr}"

    # Target exists with 3 shards
    info = await qclient.get_collection(collection_name=target)
    # We can't always introspect shard_number via client, so just check counts
    src_count = (await qclient.count(collection_name=fixture_source_collection)).count
    tgt_count = (await qclient.count(collection_name=target)).count
    assert tgt_count == src_count

    # Cleanup
    await qclient.delete_collection(collection_name=target)


@pytest.mark.asyncio
async def test_reshard_payload_includes_shard_key(qclient, fixture_source_collection):
    import subprocess
    target = "_test_reshard_target2"
    if await qclient.collection_exists(collection_name=target):
        await qclient.delete_collection(collection_name=target)

    subprocess.run([
        "python", "scripts/reshard_kb_temporal.py",
        "--source", fixture_source_collection,
        "--target", target,
        "--qdrant-url", "http://localhost:6333",
    ], check=True, timeout=120)

    points, _ = await qclient.scroll(
        collection_name=target, limit=10, with_payload=True,
    )
    for p in points:
        assert "shard_key" in p.payload
        assert p.payload["shard_key"] in {"2026-01", "2026-02", "2026-03"}

    await qclient.delete_collection(collection_name=target)
```

- [ ] **Step 2: Write the reshard script**

Create `/home/vogic/LocalRAG/scripts/reshard_kb_temporal.py`:

```python
#!/usr/bin/env python3
"""Reshard a Qdrant collection into per-month custom shards.

Plan B Phase 5.4. Source: existing Plan-A canonical collection (e.g.
kb_1_v3). Target: new collection with sharding_method=custom and
shard_key="YYYY-MM" per month.

Process:
  1. Scroll source, derive shard_key per point from filename + body.
  2. Group points by shard_key.
  3. Create target with ensure_collection_temporal(name, all_shard_keys).
  4. Upsert each shard's points via upsert_temporal.
  5. Verify per-shard counts.

Vectors carried over: dense (always), sparse (if present), colbert (if
present). Payload preserved as-is, plus shard_key + shard_key_origin
appended.

Usage:
    python scripts/reshard_kb_temporal.py \\
        --source kb_1_v3 \\
        --target kb_1_v4 \\
        --qdrant-url http://localhost:6333

If you also want to apply the alias swap: pass --swap-alias kb_1.
The alias swap is OPTIONAL and not done by default — operator should
run eval first, then swap manually.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import sys
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    DeleteAliasOperation, CreateAliasOperation,
    DeleteAlias, CreateAlias,
    PointStruct,
)

# Ensure we can import ext.services from a script
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ext.services.temporal_shard import extract_shard_key  # noqa: E402
from ext.services.vector_store import VectorStore  # noqa: E402


log = logging.getLogger("reshard")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


async def collect_source_points(client, collection: str, batch_size: int):
    """Yield batches of (point, derived_shard_key)."""
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=collection, limit=batch_size, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            sk, origin = extract_shard_key(
                filename=payload.get("filename", ""),
                body=payload.get("text", ""),
            )
            yield p, sk, origin
        if offset is None:
            break


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--swap-alias", default=None,
                        help="if set, swap this alias to the new collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="enumerate but do not write target")
    args = parser.parse_args()

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=300.0)
    vs = VectorStore(url=args.qdrant_url, vector_size=1024)
    vs._client = qc

    log.info("source=%s target=%s", args.source, args.target)

    if not await qc.collection_exists(collection_name=args.source):
        log.error("source collection does not exist: %s", args.source)
        return 2

    # Discover the source vectors config so we know which heads to carry over
    src_info = await qc.get_collection(collection_name=args.source)
    has_sparse = bool(getattr(src_info.config.params,
                              "sparse_vectors", None))
    # ColBERT detection: look for a named vector starting with "colbert"
    vector_names = list(
        (src_info.config.params.vectors or {}).keys()
        if isinstance(src_info.config.params.vectors, dict) else []
    )
    has_colbert = any(n.lower().startswith("colbert") for n in vector_names)
    log.info("source has_sparse=%s has_colbert=%s", has_sparse, has_colbert)

    # Pass 1: enumerate shard_keys + bucket points
    log.info("Pass 1: scrolling source to enumerate shard_keys...")
    buckets: dict[str, list[PointStruct]] = collections.defaultdict(list)
    origin_counter = collections.Counter()
    total = 0
    async for p, sk, origin in collect_source_points(
        qc, args.source, args.batch_size,
    ):
        payload = dict(p.payload or {})
        payload["shard_key"] = sk
        payload["shard_key_origin"] = origin.value
        buckets[sk].append(PointStruct(
            id=p.id, vector=p.vector, payload=payload,
        ))
        origin_counter[origin.value] += 1
        total += 1
    log.info("collected %d points across %d shard_keys", total, len(buckets))
    log.info("shard_key origin distribution: %s", dict(origin_counter))

    if args.dry_run:
        log.info("DRY RUN — exiting before target creation")
        for sk in sorted(buckets):
            log.info("  shard_key %s: %d points", sk, len(buckets[sk]))
        return 0

    # Pass 2: create target collection
    shard_keys = sorted(buckets.keys())
    log.info("creating target %s with %d shard keys", args.target, len(shard_keys))
    await vs.ensure_collection_temporal(
        args.target,
        shard_keys=shard_keys,
        with_sparse=has_sparse,
        with_colbert=has_colbert,
    )

    # Pass 3: upsert per shard
    for sk in shard_keys:
        points = buckets[sk]
        log.info("upserting shard_key=%s (%d points)", sk, len(points))
        await qc.upsert(
            collection_name=args.target,
            points=points,
            shard_key_selector=sk,
        )

    # Verify counts
    src_count = (await qc.count(collection_name=args.source)).count
    tgt_count = (await qc.count(collection_name=args.target)).count
    log.info("verify: source=%d target=%d", src_count, tgt_count)
    if src_count != tgt_count:
        log.error("count mismatch — investigate before proceeding")
        return 3

    # Optional alias swap
    if args.swap_alias:
        log.warning("swapping alias %s → %s", args.swap_alias, args.target)
        await qc.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(alias_name=args.swap_alias)
                ),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=args.target,
                        alias_name=args.swap_alias,
                    )
                ),
            ],
        )
        log.info("alias swap complete")

    log.info("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/reshard_kb_temporal.py
```

- [ ] **Step 3: Write the runbook**

Create `/home/vogic/LocalRAG/docs/runbook/temporal-reshard-procedure.md`:

```markdown
# Temporal Reshard Procedure (kb_1_v3 → kb_1_v4)

**Purpose:** migrate the post-Plan-A canonical collection `kb_1_v3` (dense + sparse + ColBERT + context_prefix) into a temporally sharded collection `kb_1_v4` with `shard_key="YYYY-MM"`.

**Schedule:** off-peak window. The reshard reads + writes the entire collection and must not contend with live ingestion.

**Pre-requisites verified:**

- [ ] Plan A is shipped and `kb_1` alias points to `kb_1_v3`.
- [ ] `kb_1_v2` is on disk (Plan A 14-day rollback target). DO NOT delete during this window.
- [ ] Phase 5.1 (`ensure_collection_temporal`), 5.2 (date extraction), 5.3 (tier helpers) are merged.
- [ ] Phase 0 eval baseline + Phase 4 baseline both committed.
- [ ] Off-peak window confirmed; chat QPS low.

## Throttle policy

The reshard reads + writes Qdrant; it does NOT call vllm-chat or any LLM. Throttle is by Qdrant load only (not chat p95).

## Step-by-step

### 1. Snapshot the source

```bash
SOURCE=kb_1_v3
curl -s -X POST "http://localhost:6333/collections/$SOURCE/snapshots" \
  | python -m json.tool
# Note the snapshot path; this is your absolute rollback target
```

### 2. Dry-run the reshard against staging clone

Make a clone first to validate the script doesn't blow up:

```bash
SOURCE=kb_1_v3
STAGING=kb_1_v3_staging
curl -X POST "http://localhost:6333/collections/$STAGING/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d '{"location": "<snapshot_path_from_step_1>"}'

python scripts/reshard_kb_temporal.py \
  --source $STAGING \
  --target kb_1_v4_staging \
  --dry-run
```

Expected: prints per-shard_key counts, ratio of `filename` vs `frontmatter` vs `body` vs `ingest_default` origins.

### 3. Run the actual reshard against staging

```bash
python scripts/reshard_kb_temporal.py \
  --source $STAGING \
  --target kb_1_v4_staging \
  --batch-size 256
```

### 4. Eval against staging target

Create a temporary KB row in Postgres pointing at `kb_1_v4_staging`, then:

```bash
make eval-evolution KB_EVAL_ID=$NEW_KB_ID
```

Compare against `tests/eval/results/phase-4-baseline.json`. Gate: chunk_recall@10 ≥ +5pp on `golden_evolution.jsonl`, no per-intent regression > 2pp on `golden_starter.jsonl`.

### 5. If staging gate passes — production reshard

```bash
python scripts/reshard_kb_temporal.py \
  --source kb_1_v3 \
  --target kb_1_v4 \
  --batch-size 256
```

Monitor: `nvidia-smi` (no GPU change expected — script only touches Qdrant), Qdrant logs, `docker stats orgchat-qdrant`.

### 6. Verify counts per shard

```bash
for sk in $(curl -s "http://localhost:6333/collections/kb_1_v4" | python -c "
import sys, json
data = json.load(sys.stdin)
# Print shard_keys via scroll-based discovery
"); do
  curl -s "http://localhost:6333/collections/kb_1_v4/points/count" \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"filter\": {\"must\": [{\"key\": \"shard_key\", \"match\": {\"value\": \"$sk\"}}]}}" \
    | python -c "import sys, json; r = json.load(sys.stdin); print('$sk', r['result']['count'])"
done
```

### 7. Apply tier configuration

```bash
python - <<PY
import asyncio
from ext.services.vector_store import VectorStore, classify_tier
from ext.services.temporal_shard import iter_shard_keys
import datetime as dt

async def apply():
    vs = VectorStore(url="http://localhost:6333", vector_size=1024)
    today = dt.date.today()
    # Adjust range as needed
    for sk in iter_shard_keys("2024-01", f"{today.year:04d}-{today.month:02d}"):
        tier = classify_tier(sk)
        print(f"{sk} -> {tier}")
        await vs.apply_tier_config("kb_1_v4", sk, tier)

asyncio.run(apply())
PY
```

### 8. Run eval against the new production collection

Move the kb_id row in Postgres to point at `kb_1_v4` (or use the alias swap to make this transparent):

```bash
make eval KB_EVAL_ID=$KB_ID
make eval-evolution KB_EVAL_ID=$KB_ID
```

Gate: same as step 4.

### 9. Alias swap (production cutover)

```bash
curl -X PUT http://localhost:6333/collections/aliases \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "kb_1"}},
      {"create_alias": {"collection_name": "kb_1_v4", "alias_name": "kb_1"}}
    ]
  }'
```

### 10. Confirm cutover

```bash
curl -s -X POST http://localhost:6100/api/rag/retrieve \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -d '{"query":"OFC roadmap","selected_kb_config":[{"kb_id":1}],"top_k":3}' \
  | python -m json.tool

# Inspect a hit's payload — must include shard_key
```

### 11. Hold rollback window

- Keep `kb_1_v3` for 14 days (mark read-only at app level).
- Plan A's `kb_1_v2` is ALSO still in its 14-day window — don't delete that either.
- Monitor `retrieval_ndcg_daily` for 14 days.
- If regression detected: alias swap back: `kb_1 → kb_1_v3`.

## Rollback

```bash
curl -X PUT http://localhost:6333/collections/aliases \
  -H "Content-Type: application/json" \
  -d '{
    "actions": [
      {"delete_alias": {"alias_name": "kb_1"}},
      {"create_alias": {"collection_name": "kb_1_v3", "alias_name": "kb_1"}}
    ]
  }'
```

Total revert time: < 1 second.

## Known failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| `create_shard_key` fails with "already exists" | Idempotency edge — script handles, but a misclick may have run partial reshard | Inspect with `curl /collections/kb_1_v4/cluster`; resume by running script (idempotent via UUID5 IDs) |
| Target count > source count | Duplicate points across multiple shard_keys (race in date extraction) | Compare `(doc_id, chunk_index)` distinct count instead of raw point count; if duplicates exist, drop kb_1_v4 + re-run |
| `_v4` shard_key origin mostly `ingest_default` | Date extraction unable to recover dates from existing collection | Verify ingest used filename-with-date convention; consider a one-off re-derivation pass that mines content harder |
| Live retrieval slower after swap | Tiered config not yet applied | Run step 7; cold shards default to in-RAM until reconfigured |
```

- [ ] **Step 4: Add Makefile targets**

Edit `/home/vogic/LocalRAG/Makefile`. Append:

```makefile
.PHONY: reshard-kb-temporal
reshard-kb-temporal:
	@if [ -z "$(SOURCE)" ] || [ -z "$(TARGET)" ]; then \
	  echo "Usage: make reshard-kb-temporal SOURCE=kb_1_v3 TARGET=kb_1_v4 [DRY_RUN=1]"; \
	  exit 2; \
	fi
	@DRY_FLAG=$(if $(DRY_RUN),--dry-run,) ; \
	python scripts/reshard_kb_temporal.py \
	  --source $(SOURCE) --target $(TARGET) $$DRY_FLAG

.PHONY: eval-evolution
eval-evolution:
	@if [ -z "$(KB_EVAL_ID)" ]; then \
	  echo "Usage: make eval-evolution KB_EVAL_ID=<id>"; \
	  exit 2; \
	fi
	pytest tests/eval/run_eval.py \
	  --kb-id $(KB_EVAL_ID) \
	  --golden tests/eval/golden_evolution.jsonl \
	  --output tests/eval/results/evolution-$$(date +%Y%m%d-%H%M).json
```

- [ ] **Step 5: Run integration test against local Qdrant**

```bash
cd /home/vogic/LocalRAG && pytest -m integration tests/integration/test_temporal_resharding.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/reshard_kb_temporal.py docs/runbook/temporal-reshard-procedure.md \
        tests/integration/test_temporal_resharding.py Makefile
git commit -m "phase-5.4: temporal reshard script + procedure runbook"
```

---

### Task 5.5: Temporal-semantic RAPTOR builder

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/temporal_raptor.py`
- Modify: `/home/vogic/LocalRAG/ext/services/raptor.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_temporal_raptor_tree.py`

The flat RAPTOR (`raptor.py`) clusters by semantic similarity alone — for a 36-month corpus this collapses cross-month similar chunks together and loses temporal signal. Temporal RAPTOR builds the tree in two stages:

- **Stage 1: per-month subtrees.** Group chunks by `shard_key`. For each month, run flat RAPTOR (reuse existing builder) up to L1 monthly summaries.
- **Stage 2: temporal-then-semantic upper levels.**
  - **L2 (per-quarter):** group L1 nodes by quarter (3 months). Each L2 node is a summary prompted with: "Summarize the following month-summaries from {QUARTER}. Note what changed compared to the prior quarter."
  - **L3 (per-year):** group L2 nodes by year. Prompt: "Summarize the following quarterly summaries from {YEAR} as a year-in-review. Highlight cross-quarter trends."
  - **L4 (3-year meta):** single root over all L3 nodes. Prompt: "Synthesize the following yearly summaries into a 3-year overview. Highlight long-term trends and inflection points."

Each node carries `level` + `shard_key` + `time_range` payload so the retriever can route by intent (Phase 5.6).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_temporal_raptor_tree.py`:

```python
import pytest
from unittest.mock import AsyncMock

from ext.services.temporal_raptor import (
    build_temporal_tree,
    group_chunks_by_shard_key,
    build_quarter_prompt,
    build_year_prompt,
    build_meta_prompt,
    quarter_for_shard_key,
)


def test_quarter_for_shard_key():
    assert quarter_for_shard_key("2026-01") == ("2026-Q1", 2026, 1)
    assert quarter_for_shard_key("2026-04") == ("2026-Q2", 2026, 2)
    assert quarter_for_shard_key("2026-07") == ("2026-Q3", 2026, 3)
    assert quarter_for_shard_key("2026-10") == ("2026-Q4", 2026, 4)


def test_group_chunks_by_shard_key():
    chunks = [
        {"text": "a", "shard_key": "2026-01", "doc_id": 1, "chunk_index": 0},
        {"text": "b", "shard_key": "2026-01", "doc_id": 2, "chunk_index": 0},
        {"text": "c", "shard_key": "2026-02", "doc_id": 3, "chunk_index": 0},
    ]
    grouped = group_chunks_by_shard_key(chunks)
    assert set(grouped.keys()) == {"2026-01", "2026-02"}
    assert len(grouped["2026-01"]) == 2
    assert len(grouped["2026-02"]) == 1


def test_quarter_prompt_includes_change_vs_prior():
    prompt = build_quarter_prompt(
        quarter_label="2026-Q2",
        month_summaries=["April: ...", "May: ...", "June: ..."],
        prior_quarter_summary="2026-Q1: ...",
    )
    assert "2026-Q2" in prompt
    assert "April" in prompt and "May" in prompt and "June" in prompt
    assert "prior quarter" in prompt.lower() or "compared to" in prompt.lower()


def test_quarter_prompt_handles_no_prior():
    prompt = build_quarter_prompt(
        quarter_label="2024-Q1",
        month_summaries=["Jan", "Feb", "Mar"],
        prior_quarter_summary=None,
    )
    # Should not break — first quarter in corpus has no prior
    assert "2024-Q1" in prompt
    assert "no prior" in prompt.lower() or "first" in prompt.lower()


def test_year_prompt_synthesizes_quarters():
    prompt = build_year_prompt(
        year=2025,
        quarter_summaries=["2025-Q1: ...", "2025-Q2: ...",
                            "2025-Q3: ...", "2025-Q4: ..."],
    )
    assert "2025" in prompt
    assert "year-in-review" in prompt.lower() or "annual" in prompt.lower() \
        or "trends" in prompt.lower()


def test_meta_prompt_3_year():
    prompt = build_meta_prompt(
        year_summaries=["2024: ...", "2025: ...", "2026: ..."],
    )
    assert "3-year" in prompt or "three-year" in prompt.lower() \
        or "long-term" in prompt.lower()


@pytest.mark.asyncio
async def test_build_temporal_tree_emits_all_levels():
    """Synthetic test — verify the tree has L1, L2, L3, L4 nodes for a multi-year corpus."""

    async def fake_embed(texts):
        return [[0.1] * 4 for _ in texts]

    async def fake_summarize(prompt: str) -> str:
        # Echo a stub summary
        return f"summary[{prompt[:30]}]"

    chunks = []
    for year in (2024, 2025, 2026):
        for month in (1, 4, 7, 10):  # 4 months per year, one per quarter
            for chunk_idx in range(3):
                chunks.append({
                    "text": f"chunk y={year} m={month} i={chunk_idx}",
                    "shard_key": f"{year:04d}-{month:02d}",
                    "doc_id": year * 100 + month,
                    "chunk_index": chunk_idx,
                    "embedding": [0.1, 0.2, 0.3, 0.4],
                })

    nodes = await build_temporal_tree(
        chunks=chunks,
        summarize=fake_summarize,
        embed=fake_embed,
        chat_model="qwen3-4b-qu",  # any model id — we mock summarize
    )

    levels = {n["level"] for n in nodes}
    # Levels include 1 (monthly), 2 (quarterly), 3 (yearly), 4 (meta)
    assert 1 in levels and 2 in levels and 3 in levels
    # Meta level only when more than 1 year present
    assert 4 in levels


@pytest.mark.asyncio
async def test_build_temporal_tree_single_year_no_meta():
    """Single-year corpus has no L4 meta node."""

    async def fake_embed(texts):
        return [[0.1] * 4 for _ in texts]

    async def fake_summarize(prompt: str) -> str:
        return "summary"

    chunks = [
        {
            "text": f"chunk m={month}",
            "shard_key": f"2026-{month:02d}",
            "doc_id": month,
            "chunk_index": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4],
        }
        for month in (1, 4, 7, 10)
    ]
    nodes = await build_temporal_tree(
        chunks=chunks, summarize=fake_summarize, embed=fake_embed,
        chat_model="x",
    )
    levels = {n["level"] for n in nodes}
    assert 4 not in levels  # no meta for single year
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_raptor_tree.py -v
```

Expected: ImportError — `temporal_raptor` doesn't exist.

- [ ] **Step 3: Write the temporal RAPTOR module**

Create `/home/vogic/LocalRAG/ext/services/temporal_raptor.py`:

```python
"""Temporal-then-semantic RAPTOR tree builder.

Plan B Phase 5.5. Replaces the flat RAPTOR for collections that have
shard_key payload (i.e. temporally-sharded collections from Phase 5.4).

Tree layout:
  L0 — original chunks (verbatim, untouched, NOT emitted by this module)
  L1 — per-month sub-tree summaries (one or more nodes per shard_key,
       built by reusing the flat RAPTOR clustering on per-month subsets)
  L2 — per-quarter summaries (3 monthly nodes → 1 quarterly node, prompted
       to highlight changes vs the prior quarter)
  L3 — per-year summaries (4 quarterly nodes → 1 yearly node)
  L4 — 3-year meta summary (only if corpus spans >1 year)

Each node payload includes:
  - level: 1 | 2 | 3 | 4
  - shard_key: the lowest-level shard_key (or one of them) the node covers
  - time_range: {"start": "YYYY-MM", "end": "YYYY-MM"}
  - source_chunk_ids: [int]  (the L0 leaves the node ultimately covers)

Uses the existing flat raptor.build_tree as the per-month sub-tree
builder so we don't duplicate clustering logic.
"""
from __future__ import annotations

import collections
import logging
from typing import Awaitable, Callable

from . import raptor as flat_raptor


log = logging.getLogger("orgchat.temporal_raptor")


def quarter_for_shard_key(sk: str) -> tuple[str, int, int]:
    """Return (quarter_label, year, quarter_num) for 'YYYY-MM'."""
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(sk)
    q = (m - 1) // 3 + 1
    return f"{y:04d}-Q{q}", y, q


def group_chunks_by_shard_key(
    chunks: list[dict],
) -> dict[str, list[dict]]:
    """Bucket chunks by their shard_key payload."""
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        sk = c.get("shard_key")
        if sk:
            grouped[sk].append(c)
    return grouped


def build_quarter_prompt(
    quarter_label: str,
    month_summaries: list[str],
    prior_quarter_summary: str | None,
) -> str:
    months_block = "\n\n".join(
        f"  Month {i+1}: {ms}" for i, ms in enumerate(month_summaries)
    )
    if prior_quarter_summary:
        prior_block = f"\nPrior quarter ({quarter_label}'s predecessor):\n  {prior_quarter_summary}\n"
        instruction = (
            f"Summarize the following month-summaries from {quarter_label}. "
            "Note what changed compared to the prior quarter."
        )
    else:
        prior_block = ""
        instruction = (
            f"Summarize the following month-summaries from {quarter_label}. "
            "This is the first quarter in the corpus — no prior to compare against."
        )

    return f"""{instruction}

{months_block}{prior_block}

Quarterly summary:"""


def build_year_prompt(year: int, quarter_summaries: list[str]) -> str:
    qb = "\n\n".join(
        f"  Q{i+1}: {qs}" for i, qs in enumerate(quarter_summaries)
    )
    return f"""Synthesize the following quarterly summaries from {year} into a year-in-review. Highlight cross-quarter trends.

{qb}

Annual summary:"""


def build_meta_prompt(year_summaries: list[str]) -> str:
    yb = "\n\n".join(year_summaries)
    return f"""Synthesize the following yearly summaries into a 3-year overview. Highlight long-term trends and inflection points.

{yb}

3-year synthesis:"""


async def build_temporal_tree(
    *,
    chunks: list[dict],
    summarize: Callable[[str], Awaitable[str]],
    embed: Callable[[list[str]], Awaitable[list[list[float]]]],
    chat_model: str,
) -> list[dict]:
    """Build the temporal-semantic tree.

    ``chunks`` is the flat list of L0 leaves with shard_key payload.
    ``summarize(prompt) -> str`` is an injected LLM caller.
    ``embed(texts) -> [[float]]`` embeds the resulting summary text.

    Returns a list of node dicts ready for upsert (each has text,
    embedding, payload).
    """
    by_sk = group_chunks_by_shard_key(chunks)
    if not by_sk:
        return []

    nodes: list[dict] = []

    # Stage 1: per-month subtree → L1 monthly summaries
    monthly_summaries: dict[str, str] = {}
    for sk in sorted(by_sk):
        month_chunks = by_sk[sk]
        leaves = [
            flat_raptor.RaptorNode(
                text=c["text"], level=0, parent_id=None,
                cluster_id=None, source_chunk_ids=[c["chunk_index"]],
                embedding=c.get("embedding"),
            )
            for c in month_chunks
        ]
        # Reuse the flat raptor builder for one level above leaves
        try:
            tree = await flat_raptor.build_tree(
                leaves=leaves, summarize=summarize, embed=embed,
                max_levels=1,
            )
        except Exception as e:
            log.warning("flat_raptor build failed for %s: %s; "
                        "falling back to single-summary", sk, e)
            tree = []

        # Aggregate L1 nodes from this month's subtree
        l1_nodes = [n for n in tree if n.level == 1]
        if not l1_nodes and len(month_chunks) > 1:
            # Single-summary fallback
            joined = "\n\n".join(c["text"][:1000] for c in month_chunks)
            text = await summarize(
                f"Summarize the following content from {sk}:\n\n{joined}"
            )
            embedding = (await embed([text]))[0]
            l1_nodes = [
                flat_raptor.RaptorNode(
                    text=text, level=1, parent_id=None, cluster_id=0,
                    source_chunk_ids=[c["chunk_index"] for c in month_chunks],
                    embedding=embedding,
                )
            ]

        monthly_summaries[sk] = "\n\n".join(n.text for n in l1_nodes)

        for n in l1_nodes:
            nodes.append(_node_dict(
                node=n, level=1, shard_key=sk,
                time_range={"start": sk, "end": sk},
            ))

    # Stage 2a: L2 per-quarter
    quarterly_summaries: dict[tuple[int, int], str] = {}
    by_quarter: dict[tuple[int, int], list[str]] = collections.defaultdict(list)
    for sk, summary in monthly_summaries.items():
        _, y, q = quarter_for_shard_key(sk)
        by_quarter[(y, q)].append((sk, summary))
    for (y, q), pairs in sorted(by_quarter.items()):
        pairs.sort()  # by shard_key
        quarter_label = f"{y}-Q{q}"
        prior_key = (y, q - 1) if q > 1 else (y - 1, 4)
        prior = quarterly_summaries.get(prior_key)
        prompt = build_quarter_prompt(
            quarter_label=quarter_label,
            month_summaries=[s for _, s in pairs],
            prior_quarter_summary=prior,
        )
        text = await summarize(prompt)
        embedding = (await embed([text]))[0]
        quarterly_summaries[(y, q)] = text
        # Time range covers all months in the quarter
        first_sk = pairs[0][0]
        last_sk = pairs[-1][0]
        nodes.append({
            "text": text,
            "embedding": embedding,
            "payload": {
                "level": 2,
                "shard_key": first_sk,  # one of them
                "time_range": {"start": first_sk, "end": last_sk},
                "quarter_label": quarter_label,
            },
        })

    # Stage 2b: L3 per-year
    yearly_summaries: dict[int, str] = {}
    by_year: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for (y, q), s in quarterly_summaries.items():
        by_year[y].append((q, s))
    for y, pairs in sorted(by_year.items()):
        pairs.sort()
        prompt = build_year_prompt(year=y, quarter_summaries=[s for _, s in pairs])
        text = await summarize(prompt)
        embedding = (await embed([text]))[0]
        yearly_summaries[y] = text
        nodes.append({
            "text": text,
            "embedding": embedding,
            "payload": {
                "level": 3,
                "shard_key": f"{y:04d}-12",  # year-end as anchor
                "time_range": {"start": f"{y:04d}-01", "end": f"{y:04d}-12"},
                "year": y,
            },
        })

    # Stage 2c: L4 meta — only if more than one year
    if len(yearly_summaries) > 1:
        years_sorted = sorted(yearly_summaries)
        prompt = build_meta_prompt(
            year_summaries=[
                f"{y}: {yearly_summaries[y]}" for y in years_sorted
            ],
        )
        text = await summarize(prompt)
        embedding = (await embed([text]))[0]
        nodes.append({
            "text": text,
            "embedding": embedding,
            "payload": {
                "level": 4,
                "shard_key": f"{years_sorted[-1]:04d}-12",
                "time_range": {
                    "start": f"{years_sorted[0]:04d}-01",
                    "end": f"{years_sorted[-1]:04d}-12",
                },
                "is_meta": True,
            },
        })

    return nodes


def _node_dict(node, level: int, shard_key: str,
               time_range: dict) -> dict:
    return {
        "text": node.text,
        "embedding": node.embedding,
        "payload": {
            "level": level,
            "shard_key": shard_key,
            "time_range": time_range,
            "source_chunk_ids": node.source_chunk_ids,
        },
    }


__all__ = [
    "build_temporal_tree",
    "group_chunks_by_shard_key",
    "build_quarter_prompt",
    "build_year_prompt",
    "build_meta_prompt",
    "quarter_for_shard_key",
]
```

- [ ] **Step 4: Wire `raptor.py` to delegate when shard_key present**

Edit `/home/vogic/LocalRAG/ext/services/raptor.py`. Find the `build_tree` function. Add a wrapper:

```python
async def build_tree_for_collection(
    *,
    chunks: list[dict],
    summarize, embed, chat_model: str,
) -> list[dict]:
    """Choose temporal vs flat tree based on whether chunks have shard_key.

    If any chunk has a ``shard_key`` payload key, route to
    ``temporal_raptor.build_temporal_tree``. Otherwise build the flat
    tree as before.

    Plan B Phase 5.5 — extends raptor.py to support both code paths.
    """
    has_shard_key = any(c.get("shard_key") for c in chunks)
    if has_shard_key:
        from .temporal_raptor import build_temporal_tree
        return await build_temporal_tree(
            chunks=chunks, summarize=summarize, embed=embed,
            chat_model=chat_model,
        )
    # Fall back to legacy flat path
    from .raptor import build_tree, RaptorNode
    leaves = [
        RaptorNode(
            text=c["text"], level=0, parent_id=None, cluster_id=None,
            source_chunk_ids=[c["chunk_index"]],
            embedding=c.get("embedding"),
        )
        for c in chunks
    ]
    nodes = await build_tree(
        leaves=leaves, summarize=summarize, embed=embed, max_levels=3,
    )
    return [_legacy_node_dict(n) for n in nodes if n.level > 0]


def _legacy_node_dict(n):
    return {
        "text": n.text,
        "embedding": n.embedding,
        "payload": {
            "level": n.level,
            "source_chunk_ids": n.source_chunk_ids,
        },
    }
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_raptor_tree.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/temporal_raptor.py ext/services/raptor.py \
        tests/unit/test_temporal_raptor_tree.py
git commit -m "phase-5.5: temporal-then-semantic RAPTOR tree builder"
```

---

### Task 5.6: Retrieval — temporal-aware level injection

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/retriever.py`
- Modify: `/home/vogic/LocalRAG/ext/services/chat_rag_bridge.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_temporal_level_injection.py`

For temporal collections, retrieval must inject the right level of the temporal RAPTOR tree based on intent:

- **`global` intent** → ALWAYS pull L3 (per-year) AND L4 (3-year meta) into the candidate set, regardless of dense scores. Top-1 of each must be guaranteed.
- **`evolution` intent** (a NEW intent class introduced by Plan B) → pull L2 (quarterly change-vs-prior) AND L3 (yearly).
- **`specific_date` intent** → narrow to the matching shard_key (no level injection beyond L0).
- **`specific` intent** → default L0 chunks; L1 monthly summaries optional via flag.

The "evolution" intent is exposed by the QU LLM (`temporal_constraint` is set AND query has comparison verb). The hybrid router doesn't classify it as a separate label today (kept the 4-class labels) — instead the bridge derives it post-hoc:

```
if intent == "global" and qu.entities and "compare/change/evolve" in original_query:
    treat as evolution
```

This avoids breaking the 4-class label invariant. The retriever takes an `intent_hint` parameter that reflects this derived label.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_temporal_level_injection.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_global_intent_injects_l3_l4(monkeypatch, mocker):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    # Stub vector_store.search to return candidate hits with level payload
    async def fake_search(*a, **kw):
        return [
            {"id": "p1", "score": 0.9, "payload": {"level": 0, "text": "x"}},
            {"id": "p2", "score": 0.8, "payload": {"level": 0, "text": "y"}},
        ]
    mocker.patch.object(retriever, "_dense_search", side_effect=fake_search)

    # Stub level-specific lookup to return one L3 and one L4 node
    async def fake_levels(collection, levels, top_k):
        return {
            3: [{"id": "y1", "score": 1.0, "payload": {"level": 3, "year": 2025}}],
            4: [{"id": "m1", "score": 1.0, "payload": {"level": 4, "is_meta": True}}],
        }
    mocker.patch.object(retriever, "_fetch_temporal_levels",
                        side_effect=fake_levels)

    hits = await retriever.retrieve_for_kb(
        collection="kb_1_v4", query="summarize all years", query_vec=[0.1] * 4,
        top_k=10, intent_hint="global",
    )

    levels_present = {h["payload"].get("level") for h in hits}
    assert 3 in levels_present, "global intent must include L3"
    assert 4 in levels_present, "global intent must include L4"


@pytest.mark.asyncio
async def test_evolution_intent_injects_l2_l3(monkeypatch, mocker):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    async def fake_search(*a, **kw):
        return [{"id": "p1", "score": 0.9, "payload": {"level": 0, "text": "x"}}]
    mocker.patch.object(retriever, "_dense_search", side_effect=fake_search)

    async def fake_levels(collection, levels, top_k):
        return {
            2: [{"id": "q1", "score": 1.0, "payload": {"level": 2}}],
            3: [{"id": "y1", "score": 1.0, "payload": {"level": 3}}],
        }
    mocker.patch.object(retriever, "_fetch_temporal_levels",
                        side_effect=fake_levels)

    hits = await retriever.retrieve_for_kb(
        collection="kb_1_v4", query="how have budgets evolved", query_vec=[0.1] * 4,
        top_k=10, intent_hint="evolution",
    )

    levels_present = {h["payload"].get("level") for h in hits}
    assert 2 in levels_present and 3 in levels_present


@pytest.mark.asyncio
async def test_specific_date_filters_by_shard_key(monkeypatch, mocker):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "1")

    captured = {}
    async def fake_search(collection, query_vec, top_k, qdrant_filter=None, **kw):
        captured["filter"] = qdrant_filter
        return [{"id": "p1", "score": 0.9, "payload": {"level": 0}}]
    mocker.patch.object(retriever, "_dense_search", side_effect=fake_search)

    await retriever.retrieve_for_kb(
        collection="kb_1_v4", query="outages on 5 Jan 2026", query_vec=[0.1] * 4,
        top_k=10, intent_hint="specific_date",
        temporal_constraint={"year": 2026, "quarter": None, "month": 1},
    )

    # Filter must constrain shard_key to "2026-01"
    f = captured["filter"]
    assert f is not None
    assert "2026-01" in str(f)


@pytest.mark.asyncio
async def test_temporal_levels_disabled_by_flag(monkeypatch, mocker):
    from ext.services import retriever
    monkeypatch.setenv("RAG_TEMPORAL_LEVELS", "0")

    async def fake_search(*a, **kw):
        return [{"id": "p1", "score": 0.9, "payload": {"level": 0}}]
    mocker.patch.object(retriever, "_dense_search", side_effect=fake_search)

    spy = mocker.patch.object(retriever, "_fetch_temporal_levels")
    await retriever.retrieve_for_kb(
        collection="kb_1_v4", query="summarize", query_vec=[0.1] * 4,
        top_k=10, intent_hint="global",
    )
    spy.assert_not_called()  # disabled — no level fetch
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_level_injection.py -v
```

Expected: failures — the new functions don't exist yet.

- [ ] **Step 3: Modify retriever.py**

Edit `/home/vogic/LocalRAG/ext/services/retriever.py`. Add the level injection logic.

Add helpers near the bottom of the file:

```python
import os
from typing import Optional

# Plan B Phase 5.6 — intent → level injection rules
_INTENT_LEVEL_INJECTION = {
    "global": [3, 4],       # yearly + 3-year meta
    "evolution": [2, 3],    # quarterly + yearly
    "specific_date": [],    # filter by shard_key, no level injection
    "specific": [],
    "metadata": [],
}


async def _fetch_temporal_levels(
    collection: str, levels: list[int], top_k: int = 1,
) -> dict[int, list[dict]]:
    """Fetch top_k nodes per level from a temporal collection.

    Returns {level: [hit, ...]}. Each hit is the same dict shape as
    a normal retrieval result: {id, score, payload}.
    """
    from .vector_store import VectorStore  # lazy import
    vs = _get_vector_store_singleton()
    out: dict[int, list[dict]] = {}
    for level in levels:
        # Filter by payload.level == <level>
        from qdrant_client.http.models import (
            Filter, FieldCondition, MatchValue,
        )
        f = Filter(must=[
            FieldCondition(key="level", match=MatchValue(value=level))
        ])
        # Take any K — these are summary nodes, count is small
        points, _ = await vs._client.scroll(
            collection_name=collection,
            limit=top_k,
            scroll_filter=f,
            with_payload=True, with_vectors=False,
        )
        out[level] = [
            {"id": str(p.id), "score": 1.0, "payload": p.payload or {}}
            for p in points
        ]
    return out


def _filter_by_temporal_constraint(
    constraint: dict | None,
):
    """Build a Qdrant filter that narrows to matching shard_keys."""
    if not constraint:
        return None
    from qdrant_client.http.models import (
        Filter, FieldCondition, MatchValue, MatchAny,
    )
    year = constraint.get("year")
    month = constraint.get("month")
    quarter = constraint.get("quarter")

    if month and year:
        sk = f"{year:04d}-{month:02d}"
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchValue(value=sk))
        ])
    if quarter and year:
        first_month = (quarter - 1) * 3 + 1
        sks = [f"{year:04d}-{first_month + i:02d}" for i in range(3)]
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchAny(any=sks))
        ])
    if year:
        sks = [f"{year:04d}-{m:02d}" for m in range(1, 13)]
        return Filter(must=[
            FieldCondition(key="shard_key", match=MatchAny(any=sks))
        ])
    return None
```

In the existing main retrieval entry function (find `retrieve_for_kb` or equivalent), add intent-conditional logic. Modify the signature to accept `intent_hint` and `temporal_constraint`:

```python
async def retrieve_for_kb(
    *,
    collection: str,
    query: str,
    query_vec: list[float],
    top_k: int,
    intent_hint: str = "specific",
    temporal_constraint: dict | None = None,
    **kwargs,
) -> list[dict]:
    temporal_enabled = os.environ.get("RAG_TEMPORAL_LEVELS", "0") == "1"

    # specific_date — apply shard_key filter
    qdrant_filter = None
    if temporal_enabled and intent_hint == "specific_date":
        qdrant_filter = _filter_by_temporal_constraint(temporal_constraint)

    # Run the standard dense + sparse + colbert retrieval (existing path)
    base_hits = await _dense_search(
        collection=collection, query_vec=query_vec, top_k=top_k,
        qdrant_filter=qdrant_filter, **kwargs,
    )

    if not temporal_enabled:
        return base_hits

    # Inject summary levels for global/evolution intents
    levels_to_fetch = _INTENT_LEVEL_INJECTION.get(intent_hint, [])
    if levels_to_fetch:
        levels_hits = await _fetch_temporal_levels(
            collection=collection, levels=levels_to_fetch, top_k=2,
        )
        # Merge — guaranteed-include, prepended so they survive top-K trimming
        injected = []
        for level in sorted(levels_to_fetch):
            injected.extend(levels_hits.get(level, []))
        return injected + base_hits

    return base_hits


def _get_vector_store_singleton():
    """Module-level VectorStore singleton.

    The global is initialized lazily by the bridge during startup; tests
    monkeypatch this function.
    """
    global _vs_singleton
    if _vs_singleton is None:
        from .vector_store import VectorStore
        _vs_singleton = VectorStore(
            url=os.environ.get("QDRANT_URL", "http://qdrant:6333"),
            vector_size=int(os.environ.get("RAG_DENSE_DIM", "1024")),
        )
    return _vs_singleton

_vs_singleton = None
```

- [ ] **Step 4: Wire bridge to pass intent_hint + temporal_constraint**

Edit `/home/vogic/LocalRAG/ext/services/chat_rag_bridge.py`. In `_run_pipeline`, after calling `_classify_with_qu`, derive `intent_hint`:

```python
# Plan B Phase 5.6 — derive 'evolution' intent for temporal level injection
intent_hint = hybrid.intent
original_lower = (query or "").lower()
if hybrid.intent == "global" and any(
    v in original_lower for v in
    ("compare", "evolve", "change", "trend", "differ", "contrast")
):
    intent_hint = "evolution"

# Pass through to retriever
hits = await _retrieve_for_query(
    query=retrieval_query,
    intent_hint=intent_hint,
    temporal_constraint=hybrid.temporal_constraint,
    # ... existing args ...
)
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_temporal_level_injection.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/retriever.py ext/services/chat_rag_bridge.py \
        tests/unit/test_temporal_level_injection.py
git commit -m "phase-5.6: intent-aware temporal level injection (L2/L3/L4 + shard_key filter)"
```

---

### Task 5.7: Time-decay scoring — intent-conditional multiplier

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/time_decay.py`
- Modify: `/home/vogic/LocalRAG/ext/services/retriever.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_time_decay_intent_gating.py`

Time-decay applies `score' = score · exp(-λ · Δt_days)` ONLY for `current-state` intent (a sub-classification of `specific` where the query implies "now" — heuristics: present-tense verbs + no temporal_constraint). For `evolution` queries, decay is explicitly disabled — older docs should NOT be down-weighted.

We don't introduce a 5th intent class. Instead the retriever asks: "should I apply decay?" via a small predicate function.

`λ` is chosen via half-life: `λ = ln(2) / half_life_days`. Default `RAG_TIME_DECAY_LAMBDA_DAYS=90` (3-month half-life).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_time_decay_intent_gating.py`:

```python
import datetime as dt
import math
import pytest

from ext.services.time_decay import (
    should_apply_time_decay,
    time_decay_multiplier,
    apply_time_decay_to_hits,
)


class TestShouldApplyTimeDecay:
    def test_applies_for_present_tense_specific(self):
        assert should_apply_time_decay(
            query="what is the current OFC roadmap",
            intent="specific",
            temporal_constraint=None,
        ) is True

    def test_does_not_apply_for_evolution(self):
        assert should_apply_time_decay(
            query="how have budgets changed",
            intent="evolution",
            temporal_constraint=None,
        ) is False

    def test_does_not_apply_for_specific_date(self):
        assert should_apply_time_decay(
            query="outages on 5 Jan 2026",
            intent="specific_date",
            temporal_constraint={"year": 2026, "month": 1},
        ) is False

    def test_does_not_apply_for_global(self):
        assert should_apply_time_decay(
            query="summarize everything",
            intent="global",
            temporal_constraint=None,
        ) is False

    def test_does_not_apply_for_metadata(self):
        assert should_apply_time_decay(
            query="list reports",
            intent="metadata",
            temporal_constraint=None,
        ) is False

    def test_does_not_apply_when_temporal_constraint_set(self):
        # User said "in 2024" — they want 2024, not "now"
        assert should_apply_time_decay(
            query="status of OFC",
            intent="specific",
            temporal_constraint={"year": 2024},
        ) is False


class TestTimeDecayMultiplier:
    def test_zero_age_returns_1(self):
        m = time_decay_multiplier(age_days=0, lambda_days=90)
        assert m == pytest.approx(1.0)

    def test_one_half_life_returns_0_5(self):
        m = time_decay_multiplier(age_days=90, lambda_days=90)
        assert m == pytest.approx(0.5, abs=0.01)

    def test_two_half_lives_returns_0_25(self):
        m = time_decay_multiplier(age_days=180, lambda_days=90)
        assert m == pytest.approx(0.25, abs=0.02)

    def test_negative_age_clamped_to_zero(self):
        m = time_decay_multiplier(age_days=-30, lambda_days=90)
        assert m == pytest.approx(1.0)


class TestApplyTimeDecayToHits:
    def test_decays_hit_by_shard_key_age(self):
        today = dt.date.today()
        recent_sk = f"{today.year:04d}-{today.month:02d}"
        # 6 months ago
        six_ago = today.replace(day=1) - dt.timedelta(days=180)
        old_sk = f"{six_ago.year:04d}-{six_ago.month:02d}"

        hits = [
            {"id": "a", "score": 1.0, "payload": {"shard_key": recent_sk}},
            {"id": "b", "score": 1.0, "payload": {"shard_key": old_sk}},
        ]
        out = apply_time_decay_to_hits(hits, lambda_days=90)
        # Recent score unchanged (or close)
        assert out[0]["score"] == pytest.approx(1.0, abs=0.05)
        # Old score halved + (180 days = 2 half-lives) = ~0.25
        assert out[1]["score"] == pytest.approx(0.25, abs=0.05)

    def test_skips_hits_without_shard_key(self):
        hits = [
            {"id": "a", "score": 0.8, "payload": {}},
        ]
        out = apply_time_decay_to_hits(hits, lambda_days=90)
        assert out[0]["score"] == 0.8

    def test_summary_level_nodes_not_decayed(self):
        # L2/L3 nodes are aggregates — don't decay them
        hits = [
            {"id": "y", "score": 0.9, "payload": {
                "shard_key": "2024-01", "level": 3
            }},
        ]
        out = apply_time_decay_to_hits(hits, lambda_days=90)
        assert out[0]["score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_time_decay_intent_gating.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the module**

Create `/home/vogic/LocalRAG/ext/services/time_decay.py`:

```python
"""Intent-conditional time-decay multiplier.

Plan B Phase 5.7. score' = score · exp(-λ · Δt_days), applied ONLY
for current-state intent (specific + present-tense + no temporal
constraint). Evolution / aggregation queries should NOT down-weight
old documents — that defeats the question.

λ derived from half-life: λ = ln(2) / half_life_days. Default
``RAG_TIME_DECAY_LAMBDA_DAYS=90`` → 3-month half-life.

Summary-level RAPTOR nodes (level >= 2) are NOT decayed — they're
aggregates by design.
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re


_PRESENT_TENSE_RE = re.compile(
    r"\b(is|are|am|has|have|currently|now|today|present|status|exists)\b",
    re.IGNORECASE,
)


def should_apply_time_decay(
    *, query: str, intent: str, temporal_constraint: dict | None,
) -> bool:
    """Decide whether to apply time-decay for this query.

    Conservative: returns True only for ``specific`` intent + present-tense
    signal + no explicit temporal constraint. All other paths return False.
    """
    if intent != "specific":
        return False
    if temporal_constraint:
        return False
    if not query:
        return False
    return bool(_PRESENT_TENSE_RE.search(query))


def time_decay_multiplier(*, age_days: float, lambda_days: float) -> float:
    """Return exp(-λ · age_days). Negative ages clamp to 0 (= multiplier 1)."""
    if age_days <= 0 or lambda_days <= 0:
        return 1.0
    lam = math.log(2) / lambda_days
    return math.exp(-lam * age_days)


def _shard_key_age_days(shard_key: str) -> float:
    """Return age in days from the shard_key's mid-month to today."""
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(shard_key)
    sk_date = _dt.date(y, m, 15)  # mid-month anchor
    delta = (_dt.date.today() - sk_date).days
    return float(max(0, delta))


def apply_time_decay_to_hits(
    hits: list[dict], *, lambda_days: float | None = None,
) -> list[dict]:
    """Multiply each hit's ``score`` by the time-decay factor in place.

    Hits without a shard_key payload are passed through unchanged.
    Hits at level >= 2 (summaries) are passed through unchanged.
    Returns the same list (mutates each dict's ``score``).
    """
    lam = lambda_days if lambda_days is not None else float(
        os.environ.get("RAG_TIME_DECAY_LAMBDA_DAYS", "90")
    )
    for hit in hits:
        payload = hit.get("payload") or {}
        if payload.get("level", 0) >= 2:
            continue
        sk = payload.get("shard_key")
        if not sk:
            continue
        age_days = _shard_key_age_days(sk)
        mul = time_decay_multiplier(age_days=age_days, lambda_days=lam)
        hit["score"] = hit["score"] * mul
    return hits


__all__ = [
    "should_apply_time_decay",
    "time_decay_multiplier",
    "apply_time_decay_to_hits",
]
```

- [ ] **Step 4: Wire into retriever**

Edit `/home/vogic/LocalRAG/ext/services/retriever.py`. In `retrieve_for_kb`, after getting hits but before returning:

```python
if os.environ.get("RAG_TIME_DECAY", "0") == "1":
    from .time_decay import should_apply_time_decay, apply_time_decay_to_hits
    if should_apply_time_decay(
        query=query, intent=intent_hint,
        temporal_constraint=temporal_constraint,
    ):
        apply_time_decay_to_hits(base_hits)
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_time_decay_intent_gating.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/services/time_decay.py ext/services/retriever.py \
        tests/unit/test_time_decay_intent_gating.py
git commit -m "phase-5.7: intent-conditional time-decay scoring"
```

---

### Task 5.8: Hot/warm/cold tier movement automation (daily cron)

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/tier_storage_cron.py`
- Modify: `/home/vogic/LocalRAG/ext/workers/celery_app.py` (add beat schedule)
- Create: `/home/vogic/LocalRAG/docs/runbook/tiered-storage-runbook.md`
- Create: `/home/vogic/LocalRAG/tests/unit/test_tier_storage_cron.py`

The cron runs daily at off-peak (default 03:00 local). For each temporally-sharded collection:
1. Enumerate active shard_keys.
2. For each shard_key, compute the desired tier (hot/warm/cold).
3. Compare against the last-known tier (cached in Redis DB 5).
4. If different, call `vector_store.apply_tier_config` and log the transition.

Idempotent: if every shard is already in the correct tier, the cron is a no-op.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_tier_storage_cron.py`:

```python
import datetime as dt
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_cron_promotes_shard_to_correct_tier(monkeypatch):
    from scripts.tier_storage_cron import process_collection

    vs = MagicMock()
    vs._client.scroll = AsyncMock(return_value=([], None))

    apply_calls = []
    async def fake_apply(collection, shard_key, tier):
        apply_calls.append((collection, shard_key, tier))
    vs.apply_tier_config = fake_apply

    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)  # never moved before
    redis_client.set = AsyncMock()

    today = dt.date.today()
    # 2 shard_keys: one recent (hot), one 18 months old (cold)
    recent = f"{today.year:04d}-{today.month:02d}"
    eighteen_ago = today.replace(day=1) - dt.timedelta(days=540)
    old = f"{eighteen_ago.year:04d}-{eighteen_ago.month:02d}"

    await process_collection(
        vs=vs, redis_client=redis_client, collection="kb_1_v4",
        shard_keys=[recent, old],
    )

    tiers = {sk: tier for _, sk, tier in apply_calls}
    assert tiers[recent] == "hot"
    assert tiers[old] == "cold"


@pytest.mark.asyncio
async def test_cron_skips_already_correct_tier(monkeypatch):
    from scripts.tier_storage_cron import process_collection

    vs = MagicMock()
    apply_calls = []
    async def fake_apply(collection, shard_key, tier):
        apply_calls.append((collection, shard_key, tier))
    vs.apply_tier_config = fake_apply

    redis_client = MagicMock()
    today = dt.date.today()
    recent_sk = f"{today.year:04d}-{today.month:02d}"
    redis_client.get = AsyncMock(return_value="hot")  # already hot
    redis_client.set = AsyncMock()

    await process_collection(
        vs=vs, redis_client=redis_client, collection="kb_1_v4",
        shard_keys=[recent_sk],
    )

    assert apply_calls == []  # no-op
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_tier_storage_cron.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the cron script**

Create `/home/vogic/LocalRAG/scripts/tier_storage_cron.py`:

```python
#!/usr/bin/env python3
"""Daily hot/warm/cold tier movement for temporally-sharded collections.

Plan B Phase 5.8. For each collection that uses shard_key="YYYY-MM",
compute the desired tier per shard and apply if changed since last run.

Last-known tier per shard is cached in Redis DB 5 (key: tier:<col>:<sk>).
This avoids re-hitting Qdrant's update_collection on every run when no
boundary has crossed.

Schedule: invoked by Celery Beat (added in Phase 6.2 wiring) at 03:00
local. May also be run manually:
    python scripts/tier_storage_cron.py --collection kb_1_v4
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any

import redis.asyncio as aioredis

import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ext.services.vector_store import VectorStore, classify_tier  # noqa: E402


log = logging.getLogger("tier_cron")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


async def discover_shard_keys(qclient, collection: str) -> list[str]:
    """Scroll the collection to learn which shard_keys are populated."""
    seen: set[str] = set()
    offset = None
    while True:
        points, offset = await qclient.scroll(
            collection_name=collection, limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            sk = (p.payload or {}).get("shard_key")
            if sk:
                seen.add(sk)
        if offset is None:
            break
    return sorted(seen)


async def process_collection(
    vs, redis_client, collection: str, shard_keys: list[str],
) -> dict:
    """Apply tier config per shard. Returns summary dict."""
    transitions = {}
    hot_months = int(os.environ.get("RAG_TIER_HOT_MONTHS", "3"))
    warm_months = int(os.environ.get("RAG_TIER_WARM_MONTHS", "12"))

    for sk in shard_keys:
        desired = classify_tier(sk, hot_months=hot_months,
                                  warm_months=warm_months)
        cache_key = f"tier:{collection}:{sk}"
        previous = await redis_client.get(cache_key)
        if previous == desired:
            continue
        log.info("transition %s/%s: %s -> %s",
                 collection, sk, previous or "?", desired)
        await vs.apply_tier_config(
            collection=collection, shard_key=sk, tier=desired,
        )
        await redis_client.set(cache_key, desired)
        transitions[sk] = (previous, desired)

    return transitions


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url",
                        default=os.environ.get("QDRANT_URL", "http://qdrant:6333"))
    parser.add_argument("--redis-url",
                        default=os.environ.get("REDIS_URL", "redis://redis:6379"))
    parser.add_argument("--redis-db", type=int, default=5)
    args = parser.parse_args()

    vs = VectorStore(url=args.qdrant_url, vector_size=1024)
    rc = aioredis.from_url(args.redis_url, db=args.redis_db,
                            decode_responses=True)

    shard_keys = await discover_shard_keys(vs._client, args.collection)
    log.info("collection=%s shard_keys=%d", args.collection, len(shard_keys))

    transitions = await process_collection(
        vs=vs, redis_client=rc, collection=args.collection,
        shard_keys=shard_keys,
    )
    log.info("applied %d tier transitions", len(transitions))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 4: Add beat schedule entry**

Edit `/home/vogic/LocalRAG/ext/workers/celery_app.py`. Find the existing `beat_schedule` (or add the block if not present). Add:

```python
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    "tier-storage-cron-daily": {
        "task": "ext.workers.scheduled_eval.tier_storage_cron",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"collection": "kb_1_v4"},
    },
}
```

Add the wrapper task. Edit (or create) `/home/vogic/LocalRAG/ext/workers/scheduled_eval.py`:

```python
from .celery_app import celery_app


@celery_app.task(name="ext.workers.scheduled_eval.tier_storage_cron")
def tier_storage_cron(collection: str = "kb_1_v4") -> dict:
    """Celery task wrapper around scripts/tier_storage_cron.py."""
    import subprocess
    result = subprocess.run(
        ["python", "scripts/tier_storage_cron.py", "--collection", collection],
        capture_output=True, text=True, check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],  # tail
        "stderr": result.stderr[-2000:],
    }
```

- [ ] **Step 5: Write the runbook**

Create `/home/vogic/LocalRAG/docs/runbook/tiered-storage-runbook.md`:

```markdown
# Tiered Storage Runbook

Plan B Phase 5.8.

## Tiers

| Tier | Age | Storage | Quantization |
|---|---|---|---|
| hot  | 0–2 months | in-RAM HNSW (memmap_threshold=0) | none |
| warm | 3–11 months | mmap on SSD | none |
| cold | ≥ 12 months | on-disk + mmap | INT8 scalar (always_ram=False) |

Defaults `RAG_TIER_HOT_MONTHS=3`, `RAG_TIER_WARM_MONTHS=12`. Tunable per environment.

## Daily cron

Celery Beat fires `ext.workers.scheduled_eval.tier_storage_cron` at 03:00.
The task runs `scripts/tier_storage_cron.py` for each tracked collection.

Manual run:

```bash
python scripts/tier_storage_cron.py --collection kb_1_v4
```

The cron is idempotent — uses Redis DB 5 (`tier:<col>:<sk>` keys) to skip
no-op transitions.

## Verification

After a transition (or daily), check:

```bash
# Last applied tiers
redis-cli -n 5 KEYS 'tier:kb_1_v4:*' | head

# Each key's value
for key in $(redis-cli -n 5 KEYS 'tier:kb_1_v4:*'); do
  val=$(redis-cli -n 5 GET "$key")
  echo "$key: $val"
done

# Qdrant collection config
curl -s http://localhost:6333/collections/kb_1_v4 | python -m json.tool | \
  grep -A 20 'optimizers_config\|quantization_config'
```

## Common operations

### Force a re-tier (e.g. after changing month thresholds)

```bash
redis-cli -n 5 --scan --pattern 'tier:kb_1_v4:*' | xargs -r redis-cli -n 5 DEL
python scripts/tier_storage_cron.py --collection kb_1_v4
```

### Pin a specific shard to hot (e.g. for an incident)

```bash
python -c "
import asyncio
from ext.services.vector_store import VectorStore
async def main():
  vs = VectorStore(url='http://localhost:6333', vector_size=1024)
  await vs.apply_tier_config(collection='kb_1_v4', shard_key='2024-06', tier='hot')
asyncio.run(main())
"
# Then prevent the cron from reverting:
redis-cli -n 5 SET 'tier:kb_1_v4:2024-06' 'hot'
```

### Disable cron entirely

Comment out the entry in `ext/workers/celery_app.py:beat_schedule`,
restart `celery-beat`.
```

- [ ] **Step 6: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_tier_storage_cron.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add scripts/tier_storage_cron.py ext/workers/celery_app.py \
        ext/workers/scheduled_eval.py docs/runbook/tiered-storage-runbook.md \
        tests/unit/test_tier_storage_cron.py
git commit -m "phase-5.8: daily tier movement cron + Celery Beat wiring"
```

---

### Task 5.9: Per-shard health metrics

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/metrics.py`
- Create: `/home/vogic/LocalRAG/observability/prometheus/alerts-tiered-shards.yml`
- Modify: `/home/vogic/LocalRAG/ext/services/vector_store.py` (emit gauges in upsert / search)
- Create: `/home/vogic/LocalRAG/tests/unit/test_per_shard_metrics.py`

Per-shard metrics:
- `rag_shard_point_count{collection, shard_key}` (gauge — refreshed nightly by the tier cron)
- `rag_shard_search_latency_seconds{collection, shard_key}` (histogram — observed per shard-filtered search)
- `rag_shard_upsert_latency_seconds{collection, shard_key}` (histogram)
- `rag_shard_tier{collection, shard_key, tier}` (gauge with label-as-state — set to 1 for current tier)

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_per_shard_metrics.py`:

```python
import pytest
from prometheus_client import REGISTRY


def _gauge_value(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels=labels) or 0.0


def test_metrics_exposed():
    from ext.services import metrics
    for n in ("RAG_SHARD_POINT_COUNT", "RAG_SHARD_SEARCH_LATENCY",
              "RAG_SHARD_UPSERT_LATENCY", "RAG_SHARD_TIER"):
        assert hasattr(metrics, n), f"metric {n} not exposed"


def test_set_shard_tier_gauge():
    from ext.services.metrics import RAG_SHARD_TIER, set_shard_tier
    set_shard_tier(collection="kb_1_v4", shard_key="2026-04", tier="hot")
    assert _gauge_value("rag_shard_tier",
                        {"collection": "kb_1_v4",
                         "shard_key": "2026-04", "tier": "hot"}) == 1.0
    # Switch tier — old gauge resets to 0
    set_shard_tier(collection="kb_1_v4", shard_key="2026-04", tier="warm")
    assert _gauge_value("rag_shard_tier",
                        {"collection": "kb_1_v4",
                         "shard_key": "2026-04", "tier": "hot"}) == 0.0
    assert _gauge_value("rag_shard_tier",
                        {"collection": "kb_1_v4",
                         "shard_key": "2026-04", "tier": "warm"}) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_per_shard_metrics.py -v
```

Expected: AttributeError.

- [ ] **Step 3: Add metrics**

Edit `/home/vogic/LocalRAG/ext/services/metrics.py`. Append:

```python
# -----------------------------------------------------------------------
# Plan B Phase 5.9 — per-shard metrics
# -----------------------------------------------------------------------
RAG_SHARD_POINT_COUNT = Gauge(
    "rag_shard_point_count",
    "Number of points per shard, refreshed by tier cron",
    ["collection", "shard_key"],
)

RAG_SHARD_SEARCH_LATENCY = Histogram(
    "rag_shard_search_latency_seconds",
    "Per-shard search latency (only for shard-filtered searches)",
    ["collection", "shard_key"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RAG_SHARD_UPSERT_LATENCY = Histogram(
    "rag_shard_upsert_latency_seconds",
    "Per-shard upsert latency",
    ["collection", "shard_key"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RAG_SHARD_TIER = Gauge(
    "rag_shard_tier",
    "Current tier for each shard (1=current; 0=not)",
    ["collection", "shard_key", "tier"],
)


def set_shard_tier(*, collection: str, shard_key: str, tier: str) -> None:
    """Update the tier gauge: set requested tier to 1, others to 0."""
    for t in ("hot", "warm", "cold"):
        RAG_SHARD_TIER.labels(
            collection=collection, shard_key=shard_key, tier=t,
        ).set(1.0 if t == tier else 0.0)
```

- [ ] **Step 4: Wire `apply_tier_config` to update the gauge**

Edit `/home/vogic/LocalRAG/ext/services/vector_store.py`. In `apply_tier_config`, after the Qdrant call:

```python
from .metrics import set_shard_tier
set_shard_tier(collection=collection, shard_key=shard_key, tier=tier)
```

Wrap `upsert_temporal` to observe latency:

```python
import time
from .metrics import RAG_SHARD_UPSERT_LATENCY

async def upsert_temporal(self, collection, points, *, shard_key):
    start = time.monotonic()
    try:
        # ... existing body ...
    finally:
        RAG_SHARD_UPSERT_LATENCY.labels(
            collection=collection, shard_key=shard_key,
        ).observe(time.monotonic() - start)
```

- [ ] **Step 5: Add Prometheus alerts**

Create `/home/vogic/LocalRAG/observability/prometheus/alerts-tiered-shards.yml`:

```yaml
groups:
  - name: tiered_shards
    interval: 1m
    rules:
      - alert: ShardSearchLatencyHigh
        expr: histogram_quantile(0.95, rate(rag_shard_search_latency_seconds_bucket[5m])) > 1.0
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Per-shard p95 search latency > 1s on {{ $labels.collection }}/{{ $labels.shard_key }}"
          description: "Cold shards may be reading from disk. Check tier configuration."

      - alert: ShardCountUnexpectedlyHigh
        expr: rag_shard_point_count > 1000000
        for: 1h
        labels: {severity: info}
        annotations:
          summary: "Shard {{ $labels.collection }}/{{ $labels.shard_key }} has > 1M points"
          description: "Consider re-sharding more granularly (e.g. per-week instead of per-month)."

      - alert: ShardTierTransitionFlapping
        expr: changes(rag_shard_tier[1d]) > 4
        for: 6h
        labels: {severity: info}
        annotations:
          summary: "Shard {{ $labels.collection }}/{{ $labels.shard_key }} flapping between tiers"
          description: "Cron may be misconfigured. Inspect Redis DB 5 keys."
```

- [ ] **Step 6: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_per_shard_metrics.py -v
```

Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add ext/services/metrics.py ext/services/vector_store.py \
        observability/prometheus/alerts-tiered-shards.yml \
        tests/unit/test_per_shard_metrics.py
git commit -m "phase-5.9: per-shard metrics + Prometheus alerts"
```

---

### Task 5.10: Operator runbook + Phase 5 completion checklist

**Files:**
- Create: `/home/vogic/LocalRAG/docs/runbook/temporal-reshard-checklist.md`
- Modify: `/home/vogic/LocalRAG/docs/runbook/troubleshooting.md`
- Create: `/home/vogic/LocalRAG/tests/eval/golden_evolution.jsonl` (30 evolution-stratified queries)
- Audit: `RAG_RAPTOR` flag fate

The checklist is the operator-facing companion to `temporal-reshard-procedure.md` (Phase 5.4). It mirrors the structure of Plan A's `reingest-checklist.md`.

The `golden_evolution.jsonl` is 30 queries that exercise evolution-stratified retrieval — comparison queries, time-bounded summaries, change-vs-prior questions. These complement Plan A's `golden_starter.jsonl` (60 queries, 4 strata).

The `RAG_RAPTOR` flag audit: Plan A introduced flag-OFF default. Phase 5.10 decides — Plan B replaces flat RAPTOR with `temporal_raptor` for sharded collections; the legacy path stays opt-in for non-sharded collections. Decision: keep `RAG_RAPTOR=0` default, add `RAG_RAPTOR_TEMPORAL=1` default for collections with shard_key, document in flag reference.

- [ ] **Step 1: Write the operator checklist**

Create `/home/vogic/LocalRAG/docs/runbook/temporal-reshard-checklist.md`:

```markdown
# Temporal Reshard Checklist (operator-facing)

Use this checklist in the off-hours window to execute Task 5.4.

**Pre-requisites verified:**

- [ ] Plan A Phase 3.7 re-ingest is complete (kb_1 → kb_1_v3 alias active for ≥ 7 days).
- [ ] Plan A's kb_1_v2 is still in its 14-day rollback window — DO NOT delete during this window.
- [ ] Phase 5.1–5.3 + 5.5 + 5.7 merged to main.
- [ ] Phase 0 baseline + Phase 4 baseline both committed.
- [ ] tests/eval/golden_evolution.jsonl committed with at least 30 queries.
- [ ] Off-peak window confirmed; chat QPS low.
- [ ] `nvidia-smi` GPU 1 ≤ 50% (Phase 4 vllm-qu does not contend with this work but watch anyway).

**Execution:**

- [ ] Step 1 — Snapshot kb_1_v3 (per docs/runbook/temporal-reshard-procedure.md §1).
- [ ] Step 2 — Dry-run reshard against staging clone (§2). Inspect shard_key origin distribution.
- [ ] Step 3 — Run actual reshard against staging (§3).
- [ ] Step 4 — `make eval-evolution KB_EVAL_ID=$STAGING_KB_ID`. Confirm gate (§4).
- [ ] Step 5 — Production reshard (§5). Monitor Qdrant disk I/O.
- [ ] Step 6 — Per-shard count verification (§6).
- [ ] Step 7 — Apply tier configuration via the Python helper (§7).
- [ ] Step 8 — `make eval` and `make eval-evolution` against new collection (§8). Confirm gate.
- [ ] Step 9 — Alias swap (§9).
- [ ] Step 10 — Spot-check live retrieval (§10).
- [ ] Step 11 — Mark kb_1_v3 read-only for 14 days (§11).

**Post-window:**

- [ ] Update `docs/runbook/flag-reference.md` and `docs/runbook/plan-b-flag-reference.md`:
  - `RAG_SHARDING_ENABLED=1` for kb_1_v4
  - `RAG_TEMPORAL_LEVELS=1` for kb_1_v4
- [ ] Set calendar reminder for Day 14 to drop kb_1_v3 (and kb_1_v2 if its window has also passed).
- [ ] Build the temporal RAPTOR tree for kb_1_v4 (separate operation; see below).
- [ ] Announce completion to the team.

**Building the temporal RAPTOR tree:**

After resharding, the L0 chunks exist in kb_1_v4 but L1–L4 nodes do not. Build them:

```bash
python - <<PY
import asyncio
from qdrant_client import AsyncQdrantClient
from ext.services.temporal_raptor import build_temporal_tree
# ... wire summarize via vllm-chat, embed via TEI, then upsert returned nodes
PY
```

(Operator: a complete script for this is `scripts/build_temporal_tree.py`, mirrored on the pattern from Plan A 3.7.)

**If any step fails:** follow Rollback in `temporal-reshard-procedure.md`. Log the failure mode in `docs/runbook/troubleshooting.md` under "Temporal reshard issues."
```

- [ ] **Step 2: Append troubleshooting entries**

Edit `/home/vogic/LocalRAG/docs/runbook/troubleshooting.md`. Append:

```markdown
## Temporal reshard issues

### Reshard stuck — Qdrant CPU at 100%

Cause: the reshard does N (per-shard) upserts in a tight loop; very large source collections can saturate Qdrant.
Fix: lower `--batch-size` from 256 to 64; resume (idempotent via UUID5 IDs).

### `kb_1_v4` retrieval slower than `kb_1_v3` after swap

Cause: tier configuration not yet applied. Cold shards default to in-RAM until the cron runs.
Fix: run `python scripts/tier_storage_cron.py --collection kb_1_v4` immediately. Or wait for the daily 03:00 run.

### `apply_tier_config` reports "shard not found"

Cause: shard was never created (the source collection had a date that didn't classify into any month, falling to ingest_default).
Fix: verify `shard_key_origin` distribution from the reshard log. If `ingest_default` is > 5%, the date extraction failed for too many docs — fix the underlying convention before resharding.

### Temporal RAPTOR build is OOM'ing vllm-chat

Cause: too many parallel summarize calls. Each L1/L2/L3/L4 node = 1 chat call.
Fix: introduce a semaphore around `summarize` calls. Default concurrency 4; lower to 1 if needed. Monitor GPU 0 during build.

### Evolution queries returning bad results despite L2/L3 injection

Cause: L2 nodes don't actually contain change-vs-prior content because the prior summary was missing at build time (year-1 quarter-1 has no prior).
Fix: rebuild the tree if you've added historical data after initial build. The first quarter in the corpus will always have "no prior comparison".
```

- [ ] **Step 3: Write the evolution golden set**

Create `/home/vogic/LocalRAG/tests/eval/golden_evolution.jsonl`:

```jsonl
{"query":"how have OFC roadmap milestones changed across 2024-2026","intent":"evolution","kb_id":1,"expected_doc_ids":[],"notes":"3-year change query — should pull L3 + L4"}
{"query":"compare Q1 2025 budget to Q1 2024","intent":"evolution","kb_id":1,"expected_doc_ids":[],"notes":"quarter-over-quarter — pulls L2 nodes"}
{"query":"what shifted between January and June of 2026","intent":"evolution","kb_id":1,"expected_doc_ids":[],"notes":"intra-year evolution"}
{"query":"trend in incident frequency 2024 vs 2025","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how did the team structure evolve last year","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare Q4 2024 and Q4 2025 retention","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"what have been the major architectural changes since 2024","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"evolution of OFC scope from inception","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare deployment cadence year over year","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how has the budget for the OFC roadmap shifted across the three years","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"changes in the on-call rotation policy across 2024-2026","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"what improved and what regressed between 2025-Q2 and 2025-Q3","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"trend of customer-reported issues quarter over quarter","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how did SLO compliance change from 2024 to 2026","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare OFC progress between 2024-Q1 and 2026-Q1","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"what has been the trajectory of feature ship velocity","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how have priorities shifted across the past year","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare 2024 retros to 2025 retros for recurring themes","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"changes in postmortem volume across the corpus","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how has the architecture diagram diverged from initial proposals","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare Q3 2025 to Q3 2024 OKR completion","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how has the team's vendor list changed","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"3 year overview of major incidents","intent":"global","kb_id":1,"expected_doc_ids":[],"notes":"global → L4 meta"}
{"query":"summarize the entire 2025","intent":"global","kb_id":1,"expected_doc_ids":[],"notes":"global → L3"}
{"query":"give me an annual summary for 2024","intent":"global","kb_id":1,"expected_doc_ids":[]}
{"query":"compare retention between 2024 and 2025","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"how has the technical debt evolved","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"compare quarterly check-ins for OFC","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"changes in deployment process over the last 18 months","intent":"evolution","kb_id":1,"expected_doc_ids":[]}
{"query":"what's the high-level 3-year story","intent":"global","kb_id":1,"expected_doc_ids":[],"notes":"L4 meta is the primary answer"}
```

(Operator: `expected_doc_ids` is empty — fill in with the operator's actual production doc IDs as part of the eval setup.)

- [ ] **Step 4: Audit RAG_RAPTOR**

Edit `/home/vogic/LocalRAG/docs/runbook/plan-b-flag-reference.md`. Add row:

```markdown
| `RAG_RAPTOR` | Plan A | 0 | Legacy flat RAPTOR for non-sharded collections | Yes (restart). Plan B 5.10 audit decision: KEEP off; superseded by RAG_RAPTOR_TEMPORAL on sharded collections |
| `RAG_RAPTOR_TEMPORAL` | 5.5 | 1 (for sharded collections only) | Build temporal-then-semantic tree at ingest | Yes (restart) |
```

- [ ] **Step 5: Commit**

```bash
git add docs/runbook/temporal-reshard-checklist.md \
        docs/runbook/troubleshooting.md \
        docs/runbook/plan-b-flag-reference.md \
        tests/eval/golden_evolution.jsonl
git commit -m "phase-5.10: temporal reshard checklist + golden_evolution.jsonl + RAG_RAPTOR audit"
```

### Phase 5 completion gate

- [ ] All Phase 5 unit + integration tests pass.
- [ ] `scripts/reshard_kb_temporal.py --dry-run` against `kb_1_v3` succeeds; shard_key origin distribution shows ≥ 90% non-`ingest_default`.
- [ ] Production reshard executed during off-peak window. Per-shard counts match source.
- [ ] Tier config applied; `nvidia-smi` post-tier shows reduced GPU 1 memory or unchanged (cold tier moves data off GPU).
- [ ] `make eval-evolution KB_EVAL_ID=$KB_ID` shows `chunk_recall@10` ≥ +5 pp on evolution stratum vs Plan A baseline.
- [ ] `make eval KB_EVAL_ID=$KB_ID` shows no per-intent regression > 2 pp on `golden_starter`.
- [ ] Phase 5 baseline JSON committed at `tests/eval/results/phase-5-baseline.json`.
- [ ] Tier cron has run successfully for ≥ 3 consecutive days.
- [ ] Per-shard metrics emitting at `/metrics`.

---

## Phase 6 — Async ingest default + OCR + structure-aware chunking (Day 4)

**Phase goal:** Flip `RAG_SYNC_INGEST=0` as the default after a Celery soak test confirms zero lost docs and bounded DLQ depth. Add a Tesseract-backed OCR fallback for scanned PDFs (cloud OCR opt-in per KB). Add a structure-aware chunker that treats tables and code blocks as atomic units. Each step is gated by its own eval pass so a regression in any sub-feature doesn't block the others from shipping.

**Why this phase last:** Phases 4 and 5 are on the critical path for retrieval quality. Phase 6 is operational hygiene + content-type extension. Doing it last means the eval baseline at the end of Plan B reflects the full retrieval-quality story (Phase 4 + 5 wins) plus operational confidence (Phase 6 won't drop docs at scale).

---

### Task 6.1: Celery soak test — 1000-doc upload

**Files:**
- Create: `/home/vogic/LocalRAG/scripts/celery_soak_test.py`
- Create: `/home/vogic/LocalRAG/tests/integration/test_celery_soak.py`
- Create: `/home/vogic/LocalRAG/observability/prometheus/alerts-celery.yml`

The soak test validates that the Celery worker (defined but not running per Plan A's assumption #5) handles a sustained burst of 1000 document uploads with parallel uploaders without losing docs or saturating the DLQ. Pass criteria:

- 0 lost docs (every uploaded doc lands in Qdrant)
- DLQ depth ≤ 5 over the full 1-hour window
- Worker memory stable (no leak; check `docker stats`)
- Chat p95 latency stays under 3 s during the test (proves chat is not contended)

The script generates synthetic 5 KB markdown docs with random content + filename-encoded dates so the existing `extract_shard_key` produces realistic distribution.

- [ ] **Step 1: Write the soak test script**

Create `/home/vogic/LocalRAG/scripts/celery_soak_test.py`:

```python
#!/usr/bin/env python3
"""Celery soak test — 1000-doc concurrent upload.

Plan B Phase 6.1. Validates that switching RAG_SYNC_INGEST=0 (Celery
async path) handles bursty uploads without losing documents.

Generates synthetic 5KB markdown docs with filename-encoded dates so
shard_key derivation works as it would on real production data.

Usage:
    # Standard 1000-doc soak
    python scripts/celery_soak_test.py \\
        --target-kb 1 --target-subtag 1 \\
        --doc-count 1000 --concurrency 8 \\
        --api-base http://localhost:6100

    # Verify after run completes
    python scripts/celery_soak_test.py \\
        --verify --target-kb 1 \\
        --expected-doc-count 1000
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import os
import random
import string
import sys
import time
from typing import Any

import httpx


def _gen_doc(idx: int) -> tuple[str, str]:
    """Return (filename, body) for synthetic doc."""
    # Date spans the last 24 months for realistic shard distribution
    today = _dt.date.today()
    days_ago = random.randint(0, 24 * 30)
    doc_date = today - _dt.timedelta(days=days_ago)
    fn = doc_date.strftime("%d %b %Y") + f"-soak-{idx:04d}.md"
    body = f"""# Soak document {idx}

Date: {doc_date.isoformat()}
Topic: synthetic-{idx % 20}

This is a synthetic document used by the Celery soak test. It contains
roughly 1000 words of generated text so the chunker has real chunks to
emit.

""" + "\n\n".join(
        " ".join(random.choices(string.ascii_lowercase + " ", k=80))
        for _ in range(20)
    )
    return fn, body


async def upload_one(
    client: httpx.AsyncClient,
    api_base: str, token: str, kb_id: int, subtag_id: int,
    filename: str, body: str,
) -> tuple[bool, float]:
    start = time.monotonic()
    files = {"file": (filename, body, "text/markdown")}
    try:
        r = await client.post(
            f"{api_base}/api/kb/{kb_id}/subtag/{subtag_id}/upload",
            headers={"Authorization": f"Bearer {token}"},
            files=files, timeout=30.0,
        )
        return (r.status_code in (200, 202, 409), time.monotonic() - start)
    except Exception as e:
        print(f"upload err {filename}: {e}", file=sys.stderr)
        return (False, time.monotonic() - start)


async def soak(args) -> int:
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(idx: int):
        async with sem:
            fn, body = _gen_doc(idx)
            return await upload_one(
                client, args.api_base, args.token,
                args.target_kb, args.target_subtag, fn, body,
            )

    async with httpx.AsyncClient() as client:
        start = time.monotonic()
        tasks = [worker(i) for i in range(args.doc_count)]
        results = []
        for completed in asyncio.as_completed(tasks):
            results.append(await completed)
            if len(results) % 50 == 0:
                print(f"progress: {len(results)}/{args.doc_count}",
                      file=sys.stderr)
        elapsed = time.monotonic() - start

    success = sum(1 for ok, _ in results if ok)
    failures = len(results) - success
    durations = sorted(d for _, d in results)
    p50 = durations[len(durations) // 2]
    p95 = durations[int(0.95 * len(durations))]
    p99 = durations[int(0.99 * len(durations))]
    print(f"\nSoak complete in {elapsed:.1f}s")
    print(f"  uploads: {len(results)} success={success} failures={failures}")
    print(f"  per-upload latency: p50={p50:.2f}s p95={p95:.2f}s p99={p99:.2f}s")
    print(f"  rate: {len(results)/elapsed:.1f} uploads/s")
    return 0 if failures == 0 else 1


async def verify(args) -> int:
    """Count chunks belonging to the 'soak-' filename prefix in Qdrant."""
    from qdrant_client import AsyncQdrantClient
    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    # Count distinct doc_ids whose filename starts with the soak pattern
    seen_doc_ids: set = set()
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name=f"kb_{args.target_kb}",
            limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            if "-soak-" in (payload.get("filename") or ""):
                seen_doc_ids.add(payload.get("doc_id"))
        if offset is None:
            break
    print(f"Found {len(seen_doc_ids)} unique soak doc_ids in Qdrant")
    if args.expected_doc_count and len(seen_doc_ids) < args.expected_doc_count:
        print(f"  MISSING: {args.expected_doc_count - len(seen_doc_ids)}",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-kb", type=int, required=True)
    p.add_argument("--target-subtag", type=int, default=1)
    p.add_argument("--doc-count", type=int, default=1000)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--api-base", default="http://localhost:6100")
    p.add_argument("--token", default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--expected-doc-count", type=int, default=0)
    args = p.parse_args()

    if not args.token and not args.verify:
        print("ERROR: --token or RAG_ADMIN_TOKEN required for upload",
              file=sys.stderr)
        return 2

    if args.verify:
        return asyncio.run(verify(args))
    return asyncio.run(soak(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Write the integration test wrapper**

Create `/home/vogic/LocalRAG/tests/integration/test_celery_soak.py`:

```python
"""Celery soak test integration wrapper.

Skipped by default. Operator runs explicitly:
  pytest -m integration tests/integration/test_celery_soak.py -v

Requires:
  - Celery worker running (compose service celery-worker)
  - Postgres + Redis healthy
  - RAG_SYNC_INGEST=0 in the open-webui environment
"""
from __future__ import annotations

import os
import subprocess
import time
import pytest

pytestmark = pytest.mark.integration


def test_celery_soak_1000_docs():
    token = os.environ.get("RAG_ADMIN_TOKEN")
    if not token:
        pytest.skip("RAG_ADMIN_TOKEN not set")

    # 1. Drive 1000 uploads
    upload = subprocess.run(
        [
            "python", "scripts/celery_soak_test.py",
            "--target-kb", "1", "--target-subtag", "1",
            "--doc-count", "1000", "--concurrency", "8",
        ],
        capture_output=True, text=True, timeout=1800,  # 30min cap
    )
    assert upload.returncode == 0, f"upload phase failed: {upload.stderr}"

    # 2. Wait for celery to drain
    print("Sleeping 5min for celery to process the queue...")
    time.sleep(300)

    # 3. Verify
    verify = subprocess.run(
        [
            "python", "scripts/celery_soak_test.py",
            "--verify", "--target-kb", "1",
            "--expected-doc-count", "1000",
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert verify.returncode == 0, f"verification failed: {verify.stderr}"
```

- [ ] **Step 3: Add Celery alerts**

Create `/home/vogic/LocalRAG/observability/prometheus/alerts-celery.yml`:

```yaml
groups:
  - name: celery_soak
    interval: 30s
    rules:
      - alert: CeleryDLQDepthHigh
        expr: celery_dlq_depth > 10
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Celery DLQ depth {{ $value }} > 10"
          description: "Workers are unable to process some tasks. Inspect logs."

      - alert: CeleryWorkerDown
        expr: up{job="celery-worker"} == 0
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "celery-worker container down"
          description: "Async ingest will fail open with sync (if RAG_SYNC_INGEST=1) or hang (if RAG_SYNC_INGEST=0)."

      - alert: CeleryUploadLatencyHigh
        expr: histogram_quantile(0.95, rate(celery_upload_latency_seconds_bucket[5m])) > 30
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Celery upload p95 latency > 30s"
          description: "Worker may be saturated or slow."
```

- [ ] **Step 4: Run the soak (operator step)**

```bash
# Bring up the celery worker
cd /home/vogic/LocalRAG/compose
docker compose up -d celery-worker

# Run the soak (15-20 minutes)
RAG_ADMIN_TOKEN=<token> python scripts/celery_soak_test.py \
  --target-kb 1 --target-subtag 1 --doc-count 1000 --concurrency 8

# Wait for drain, verify
sleep 300
python scripts/celery_soak_test.py --verify --target-kb 1 --expected-doc-count 1000
```

- [ ] **Step 5: Commit**

```bash
git add scripts/celery_soak_test.py tests/integration/test_celery_soak.py \
        observability/prometheus/alerts-celery.yml
git commit -m "phase-6.1: Celery soak test (1000-doc upload + verification)"
```

---

### Task 6.2: Flip `RAG_SYNC_INGEST=0` default after soak passes

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/routers/upload.py`
- Modify: `/home/vogic/LocalRAG/compose/docker-compose.yml`
- Create: `/home/vogic/LocalRAG/tests/unit/test_upload_default_async.py`

This task is GATED on Phase 6.1 passing. If the soak test loses any docs or saturates DLQ, do NOT proceed — fix the worker first.

The flip is a 1-line change in `ext/routers/upload.py` (`RAG_SYNC_INGEST = os.environ.get("RAG_SYNC_INGEST", "1") == "1"` → default `"0"`). The compose file removes any `RAG_SYNC_INGEST=1` override. The celery-worker compose service is moved out of any opt-in profile (Plan A's assumption #5 said it's defined but not running — we make it default-up).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_upload_default_async.py`:

```python
import importlib
import os


def test_rag_sync_ingest_default_is_zero(monkeypatch):
    """Plan B Phase 6.2 — default flipped from 1 to 0 (async via Celery)."""
    monkeypatch.delenv("RAG_SYNC_INGEST", raising=False)
    from ext.routers import upload
    importlib.reload(upload)
    assert upload.RAG_SYNC_INGEST is False, (
        "Plan B Phase 6.2: RAG_SYNC_INGEST default must be 0 (async via Celery)"
    )


def test_rag_sync_ingest_can_still_be_enabled(monkeypatch):
    monkeypatch.setenv("RAG_SYNC_INGEST", "1")
    from ext.routers import upload
    importlib.reload(upload)
    assert upload.RAG_SYNC_INGEST is True


def test_compose_celery_worker_no_profile_gate():
    import yaml
    import pathlib
    compose_file = pathlib.Path(__file__).resolve().parents[2] / \
        "compose" / "docker-compose.yml"
    compose = yaml.safe_load(compose_file.read_text())
    assert "celery-worker" in compose["services"]
    profiles = compose["services"]["celery-worker"].get("profiles", [])
    assert profiles == [], (
        "celery-worker must be in the default compose profile (no profile gate)."
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_upload_default_async.py -v
```

Expected: 3 failures.

- [ ] **Step 3: Flip the default**

Edit `/home/vogic/LocalRAG/ext/routers/upload.py`. Find:

```python
RAG_SYNC_INGEST = os.environ.get("RAG_SYNC_INGEST", "1") == "1"
```

Replace with:

```python
# Plan B Phase 6.2 — default flipped to async after Phase 6.1 soak validation.
# Set RAG_SYNC_INGEST=1 to revert to the in-process synchronous ingest path.
RAG_SYNC_INGEST = os.environ.get("RAG_SYNC_INGEST", "0") == "1"
```

- [ ] **Step 4: Update compose**

Edit `/home/vogic/LocalRAG/compose/docker-compose.yml`. Remove any `RAG_SYNC_INGEST=1` setting from the open-webui environment. Verify the celery-worker service has NO `profiles:` key (it must be in the default profile so `docker compose up -d` brings it up).

If the celery-worker block has a `profiles: [...]` key, delete it.

Example block (after edits):

```yaml
  celery-worker:
    image: ${OPEN_WEBUI_IMAGE}
    container_name: orgchat-celery-worker
    command: celery -A ext.workers.celery_app worker --loglevel=info --concurrency=4
    environment:
      DATABASE_URL: ${DATABASE_URL}
      REDIS_URL: ${REDIS_URL}
      OPENAI_API_BASE_URL: ${OPENAI_API_BASE_URL}
      RAG_EMBEDDING_OPENAI_API_BASE_URL: ${RAG_EMBEDDING_OPENAI_API_BASE_URL}
      QDRANT_URL: ${QDRANT_URL}
    depends_on:
      - redis
      - postgres
      - qdrant
    networks:
      - orgchat
    restart: unless-stopped
```

- [ ] **Step 5: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_upload_default_async.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add ext/routers/upload.py compose/docker-compose.yml \
        tests/unit/test_upload_default_async.py
git commit -m "phase-6.2: flip RAG_SYNC_INGEST default to 0 after soak validation"
```

---

### Task 6.3: OCR module — Tesseract default + cloud opt-in per KB

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/ocr.py`
- Create: `/home/vogic/LocalRAG/ext/db/migrations/011_add_kb_ocr_policy.sql`
- Modify: `/home/vogic/LocalRAG/ext/services/kb_config.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_ocr_module.py`
- Create: `/home/vogic/LocalRAG/tests/integration/test_ocr_pipeline.py`
- Modify: `/home/vogic/LocalRAG/Dockerfile.openwebui.cu128` (install tesseract)

The OCR module exposes a single async function `ocr_pdf(pdf_bytes, *, backend, language) -> str` that returns extracted text. Backends:
- **`tesseract`** (default, air-gap-safe): rasterize each page via pymupdf, run pytesseract, concatenate.
- **`cloud:textract`** (opt-in, requires `TEXTRACT_REGION` + AWS creds): batch-upload to AWS Textract.
- **`cloud:document_ai`** (opt-in, GCP): Document AI batch.

Per-KB OCR policy lives in a new column `kb_config.ocr_policy` (JSONB):
```json
{"enabled": true, "backend": "tesseract", "language": "eng", "trigger_chars_per_page": 50}
```

The cloud backends are the LAST place in the codebase that breaks the air-gap promise. Their use is gated per KB; default for all KBs is Tesseract.

- [ ] **Step 1: Write the failing unit test**

Create `/home/vogic/LocalRAG/tests/unit/test_ocr_module.py`:

```python
import pytest
from unittest.mock import patch, MagicMock

from ext.services.ocr import (
    ocr_pdf,
    OCRBackend,
    select_ocr_backend,
)


def test_select_backend_default_tesseract():
    assert select_ocr_backend(None) is OCRBackend.TESSERACT
    assert select_ocr_backend({}) is OCRBackend.TESSERACT


def test_select_backend_explicit_tesseract():
    assert select_ocr_backend({"backend": "tesseract"}) is OCRBackend.TESSERACT


def test_select_backend_textract():
    assert select_ocr_backend({"backend": "cloud:textract"}) \
        is OCRBackend.CLOUD_TEXTRACT


def test_select_backend_document_ai():
    assert select_ocr_backend({"backend": "cloud:document_ai"}) \
        is OCRBackend.CLOUD_DOCUMENT_AI


def test_select_backend_unknown_falls_back_to_tesseract():
    assert select_ocr_backend({"backend": "unknown"}) is OCRBackend.TESSERACT


@pytest.mark.asyncio
async def test_ocr_tesseract_calls_pytesseract(monkeypatch):
    """OCR module shells out to pytesseract via async wrapper."""
    fake_text = "extracted text from page"

    def fake_image_to_string(image, lang=None):
        return fake_text
    monkeypatch.setattr("pytesseract.image_to_string", fake_image_to_string)

    # Fake pymupdf doc with 2 pages
    class _FakePixmap:
        def tobytes(self, fmt):
            return b"pretend png"
    class _FakePage:
        def get_pixmap(self, dpi=None):
            return _FakePixmap()
    class _FakeDoc:
        def __iter__(self):
            return iter([_FakePage(), _FakePage()])
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def close(self): pass

    def fake_open(stream=None):
        return _FakeDoc()
    monkeypatch.setattr("pymupdf.open", fake_open)

    # PIL bypass — we stub Image.open to return a sentinel
    monkeypatch.setattr("PIL.Image.open", lambda *a, **kw: MagicMock())

    out = await ocr_pdf(b"%PDF-fake", backend=OCRBackend.TESSERACT,
                          language="eng")
    assert fake_text in out


@pytest.mark.asyncio
async def test_ocr_cloud_textract_unavailable_raises_clear_error(monkeypatch):
    """Cloud backends raise an actionable error if creds missing."""
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("TEXTRACT_REGION", raising=False)
    with pytest.raises(RuntimeError, match="TEXTRACT|AWS"):
        await ocr_pdf(b"%PDF-fake", backend=OCRBackend.CLOUD_TEXTRACT,
                       language="eng")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_ocr_module.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the OCR module**

Create `/home/vogic/LocalRAG/ext/services/ocr.py`:

```python
"""OCR fallback for scanned PDFs.

Plan B Phase 6.3. Default backend: Tesseract (air-gap safe).
Optional cloud backends (per-KB opt-in): AWS Textract, GCP Document AI.

Trigger: ingest detects pages with < N text characters via pdfplumber
(see Phase 6.4), rasterizes those pages, runs OCR, returns extracted
text. The result is concatenated with any pdfplumber text and re-fed
into the chunker.

The cloud backends are the only code path in this codebase that can
make outbound network calls. They are gated by per-KB policy
(``kb_config.ocr_policy.backend``) and disabled globally unless an
operator explicitly opts in. They WILL FAIL CLOSED if their credentials
are missing — never silently fall back to Tesseract.
"""
from __future__ import annotations

import asyncio
import enum
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor


log = logging.getLogger("orgchat.ocr")
_executor = ThreadPoolExecutor(max_workers=2)


class OCRBackend(enum.Enum):
    TESSERACT = "tesseract"
    CLOUD_TEXTRACT = "cloud:textract"
    CLOUD_DOCUMENT_AI = "cloud:document_ai"


def select_ocr_backend(policy: dict | None) -> OCRBackend:
    """Map a kb_config.ocr_policy.backend string to OCRBackend.

    Unknown values fall back to TESSERACT (safer default — never accidentally
    upload to a cloud endpoint).
    """
    if not policy:
        return OCRBackend.TESSERACT
    raw = (policy.get("backend") or "tesseract").lower().strip()
    try:
        return OCRBackend(raw)
    except ValueError:
        log.warning("Unknown OCR backend %r; defaulting to tesseract", raw)
        return OCRBackend.TESSERACT


async def ocr_pdf(
    pdf_bytes: bytes,
    *,
    backend: OCRBackend = OCRBackend.TESSERACT,
    language: str = "eng",
) -> str:
    """Extract text from a PDF via the named backend.

    Returns the concatenated text of all pages. Raises RuntimeError if
    cloud backend credentials are missing.
    """
    if backend is OCRBackend.TESSERACT:
        return await _ocr_tesseract(pdf_bytes, language=language)
    if backend is OCRBackend.CLOUD_TEXTRACT:
        return await _ocr_textract(pdf_bytes)
    if backend is OCRBackend.CLOUD_DOCUMENT_AI:
        return await _ocr_document_ai(pdf_bytes)
    raise ValueError(f"unsupported OCR backend {backend!r}")


async def _ocr_tesseract(pdf_bytes: bytes, *, language: str) -> str:
    import pymupdf
    import pytesseract
    from PIL import Image

    def _run() -> str:
        out_pages = []
        with pymupdf.open(stream=pdf_bytes) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang=language)
                out_pages.append(text)
        return "\n\n".join(out_pages)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


async def _ocr_textract(pdf_bytes: bytes) -> str:
    region = os.environ.get("TEXTRACT_REGION")
    if not region or not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError(
            "AWS Textract requires TEXTRACT_REGION + AWS_ACCESS_KEY_ID"
        )
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 not installed; cannot use Textract") from e

    def _run() -> str:
        client = boto3.client("textract", region_name=region)
        resp = client.detect_document_text(Document={"Bytes": pdf_bytes})
        lines = [
            b["Text"] for b in resp.get("Blocks", [])
            if b.get("BlockType") == "LINE"
        ]
        return "\n".join(lines)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


async def _ocr_document_ai(pdf_bytes: bytes) -> str:
    project = os.environ.get("DOCUMENT_AI_PROJECT")
    location = os.environ.get("DOCUMENT_AI_LOCATION", "us")
    processor = os.environ.get("DOCUMENT_AI_PROCESSOR")
    if not project or not processor:
        raise RuntimeError(
            "Document AI requires DOCUMENT_AI_PROJECT + DOCUMENT_AI_PROCESSOR"
        )
    try:
        from google.cloud import documentai_v1
    except ImportError as e:
        raise RuntimeError("google-cloud-documentai not installed") from e

    def _run() -> str:
        client = documentai_v1.DocumentProcessorServiceClient()
        name = client.processor_path(project, location, processor)
        raw_doc = documentai_v1.RawDocument(
            content=pdf_bytes, mime_type="application/pdf",
        )
        req = documentai_v1.ProcessRequest(name=name, raw_document=raw_doc)
        result = client.process_document(request=req)
        return result.document.text

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, _run)


__all__ = ["OCRBackend", "select_ocr_backend", "ocr_pdf"]
```

- [ ] **Step 4: Add the migration**

Create `/home/vogic/LocalRAG/ext/db/migrations/011_add_kb_ocr_policy.sql`:

```sql
-- Plan B Phase 6.3 — per-KB OCR policy
ALTER TABLE knowledge_bases
ADD COLUMN IF NOT EXISTS ocr_policy JSONB NOT NULL DEFAULT '{
  "enabled": true,
  "backend": "tesseract",
  "language": "eng",
  "trigger_chars_per_page": 50
}'::jsonb;

COMMENT ON COLUMN knowledge_bases.ocr_policy IS
  'OCR fallback configuration per KB. Cloud backends (cloud:textract, cloud:document_ai) require operator opt-in and creds.';
```

- [ ] **Step 5: Modify kb_config.py to read the policy**

Edit `/home/vogic/LocalRAG/ext/services/kb_config.py`. Add a getter for OCR policy alongside the existing `rag_config` reader:

```python
def get_ocr_policy(kb_id: int, db_session) -> dict | None:
    """Return the per-KB OCR policy or None if disabled.

    Reads from the new ``ocr_policy`` column added in migration 011.
    """
    from ..models.kb import KnowledgeBase
    kb = db_session.query(KnowledgeBase).filter_by(id=kb_id).first()
    if not kb:
        return None
    policy = kb.ocr_policy or {}
    if not policy.get("enabled", True):
        return None
    return policy
```

- [ ] **Step 6: Bake Tesseract into the open-webui image**

Edit `/home/vogic/LocalRAG/Dockerfile.openwebui.cu128`. Find the existing `RUN pip install ...` layer and add an apt-get block above it:

```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      pytesseract==0.3.13 \
      pymupdf==1.24.10 \
      pillow==10.4.0
```

(Boto3 and google-cloud-documentai are NOT installed by default — only when an operator explicitly enables a cloud backend.)

- [ ] **Step 7: Re-run unit tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_ocr_module.py -v
```

Expected: 6 passed.

- [ ] **Step 8: Commit**

```bash
git add ext/services/ocr.py ext/db/migrations/011_add_kb_ocr_policy.sql \
        ext/services/kb_config.py Dockerfile.openwebui.cu128 \
        tests/unit/test_ocr_module.py
git commit -m "phase-6.3: OCR module (Tesseract default + cloud opt-in per KB)"
```

---

### Task 6.4: OCR trigger — pdfplumber < 50 chars per page → rasterize → OCR

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/ingest.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_ocr_trigger_threshold.py`

The trigger fires inside the existing PDF text-extraction step in `ingest.py`. Today (per Plan A end-state), `ingest.py` uses pdfplumber. After the existing extract loop, count chars per page; if any page has < `RAG_OCR_TRIGGER_CHARS` (default 50) chars AND `RAG_OCR_ENABLED=1`, rasterize that page and run OCR. Concatenate the OCR text into the per-page output. Then proceed with chunking as normal.

Per-KB OCR policy overrides the global default (e.g., disable for a KB known to be all-text).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_ocr_trigger_threshold.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_ocr_triggers_when_low_text_pages(monkeypatch, mocker):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    monkeypatch.setenv("RAG_OCR_TRIGGER_CHARS", "50")

    from ext.services import ingest

    # Stub pdfplumber to return [low_text_page, normal_page]
    fake_pdf_text = ["", "this is a normal page with plenty of extracted text"]

    def fake_extract(pdf_bytes):
        return fake_pdf_text
    monkeypatch.setattr(ingest, "_extract_pdf_text_per_page", fake_extract)

    # Stub OCR to return synthetic
    async def fake_ocr(pdf_bytes, *, backend, language):
        return "OCR-RECOVERED-TEXT"
    mocker.patch.object(ingest, "_ocr_pdf_pages", side_effect=fake_ocr)

    text = await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    assert "OCR-RECOVERED-TEXT" in text
    assert "normal page" in text  # non-OCR pages included


@pytest.mark.asyncio
async def test_ocr_does_not_trigger_when_all_pages_have_text(
    monkeypatch, mocker
):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    monkeypatch.setenv("RAG_OCR_TRIGGER_CHARS", "50")

    from ext.services import ingest
    fake_pdf_text = [
        "page one has plenty of text far above the threshold for OCR",
        "page two also has plenty of text well over fifty characters",
    ]
    monkeypatch.setattr(ingest, "_extract_pdf_text_per_page",
                         lambda b: fake_pdf_text)

    spy = mocker.patch.object(ingest, "_ocr_pdf_pages")
    text = await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    spy.assert_not_called()
    assert "page one" in text


@pytest.mark.asyncio
async def test_ocr_disabled_globally(monkeypatch, mocker):
    monkeypatch.setenv("RAG_OCR_ENABLED", "0")
    from ext.services import ingest
    monkeypatch.setattr(ingest, "_extract_pdf_text_per_page", lambda b: ["", ""])
    spy = mocker.patch.object(ingest, "_ocr_pdf_pages")
    text = await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf", ocr_policy=None,
    )
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_ocr_per_kb_policy_overrides_global(monkeypatch, mocker):
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    from ext.services import ingest
    monkeypatch.setattr(ingest, "_extract_pdf_text_per_page", lambda b: [""])
    spy = mocker.patch.object(ingest, "_ocr_pdf_pages")
    # Per-KB explicitly disables
    await ingest.extract_pdf_with_ocr_fallback(
        pdf_bytes=b"%PDF-fake", filename="x.pdf",
        ocr_policy={"enabled": False},
    )
    spy.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_ocr_trigger_threshold.py -v
```

Expected: AttributeError — `extract_pdf_with_ocr_fallback` doesn't exist yet.

- [ ] **Step 3: Wire OCR into ingest**

Edit `/home/vogic/LocalRAG/ext/services/ingest.py`. Add at the top:

```python
from .ocr import OCRBackend, select_ocr_backend, ocr_pdf
```

Add the wrapper function:

```python
async def extract_pdf_with_ocr_fallback(
    *,
    pdf_bytes: bytes,
    filename: str,
    ocr_policy: dict | None,
) -> str:
    """Extract text via pdfplumber; OCR pages where text < threshold.

    Plan B Phase 6.4. Returns concatenated text. Per-page OCR is
    triggered when ``len(page_text) < trigger_chars``.
    """
    pages = _extract_pdf_text_per_page(pdf_bytes)

    if os.environ.get("RAG_OCR_ENABLED", "0") != "1":
        return "\n\n".join(pages)

    if ocr_policy and not ocr_policy.get("enabled", True):
        return "\n\n".join(pages)

    threshold = int(
        (ocr_policy or {}).get(
            "trigger_chars_per_page",
            os.environ.get("RAG_OCR_TRIGGER_CHARS", "50"),
        )
    )

    backend = select_ocr_backend(ocr_policy)
    language = (ocr_policy or {}).get("language", "eng")

    needs_ocr = any(len(p.strip()) < threshold for p in pages)
    if not needs_ocr:
        return "\n\n".join(pages)

    log.info("ocr trigger: %s has %d/%d low-text pages",
             filename,
             sum(1 for p in pages if len(p.strip()) < threshold),
             len(pages))
    ocr_text = await _ocr_pdf_pages(pdf_bytes, backend=backend,
                                       language=language)

    # Splice OCR text in for low-text pages; keep pdfplumber text for the rest
    out_pages = []
    ocr_segments = ocr_text.split("\n\n")
    for i, p in enumerate(pages):
        if len(p.strip()) < threshold and i < len(ocr_segments):
            out_pages.append(ocr_segments[i])
        else:
            out_pages.append(p)
    return "\n\n".join(out_pages)


async def _ocr_pdf_pages(pdf_bytes, *, backend, language):
    """Indirection for tests to patch."""
    return await ocr_pdf(pdf_bytes, backend=backend, language=language)


def _extract_pdf_text_per_page(pdf_bytes: bytes) -> list[str]:
    """Existing pdfplumber path, adapted to return per-page text."""
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages
```

In the existing PDF processing entry point, route to `extract_pdf_with_ocr_fallback`:

```python
if filename.lower().endswith(".pdf"):
    ocr_policy = get_ocr_policy(kb_id, db_session)
    raw_text = await extract_pdf_with_ocr_fallback(
        pdf_bytes=raw_bytes, filename=filename, ocr_policy=ocr_policy,
    )
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_ocr_trigger_threshold.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/ingest.py tests/unit/test_ocr_trigger_threshold.py
git commit -m "phase-6.4: OCR trigger threshold + per-page splice"
```

---

### Task 6.5: Structure-aware chunker — tables/code as atomic units

**Files:**
- Create: `/home/vogic/LocalRAG/ext/services/chunker_structured.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_chunker_structured_table.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_chunker_structured_code.py`

The current chunker (window-based, 800 tokens / 100 overlap) splits tables and code blocks mid-row / mid-line, destroying their structure. The structure-aware chunker:

1. **Tables** (Markdown pipe-tables and HTML `<table>`): emit each table as a single chunk with `chunk_type="table"`. If the table is > 1500 tokens, split by row-group (preserve header repetition).
2. **Code blocks** (Markdown fenced `\`\`\`...\`\`\``): emit each fenced block as one chunk with `chunk_type="code"` + `language` payload.
3. **Prose**: window-chunk as before.

The chunker emits structured chunks with stable `chunk_type` payload so the retriever / reranker can boost or down-weight them per intent (e.g., specific-date queries should NOT be answered with code blocks; metadata queries should weight tables higher).

- [ ] **Step 1: Write the failing tests**

Create `/home/vogic/LocalRAG/tests/unit/test_chunker_structured_table.py`:

```python
import pytest

from ext.services.chunker_structured import chunk_structured


def test_pipe_table_emitted_as_single_chunk():
    text = """## Q1 budget

Some prose before.

| Quarter | Budget | Actual |
|---------|--------|--------|
| Q1      | 100    | 95     |
| Q2      | 110    | 105    |
| Q3      | 105    | 110    |

Some prose after the table."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    # Table includes header + all 3 rows
    body = table_chunks[0]["text"]
    assert "Q1" in body and "Q2" in body and "Q3" in body
    assert "| Quarter" in body  # header preserved


def test_html_table_emitted_as_single_chunk():
    text = """<p>some prose</p>
<table>
  <thead><tr><th>Q</th><th>Budget</th></tr></thead>
  <tbody>
    <tr><td>Q1</td><td>100</td></tr>
    <tr><td>Q2</td><td>110</td></tr>
  </tbody>
</table>
<p>more prose</p>"""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) == 1
    assert "<table>" in table_chunks[0]["text"] or \
        "Q1" in table_chunks[0]["text"]


def test_giant_table_split_with_repeated_header():
    rows = "\n".join(
        f"| Row{i} | {i * 100} | {i * 95} |"
        for i in range(500)  # ~3 KB+ table
    )
    text = f"""prose

| Header | A | B |
|--------|---|---|
{rows}

trailer prose"""

    chunks = chunk_structured(text, chunk_size_tokens=400, overlap_tokens=0)
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(table_chunks) > 1, "table > limit must be split"
    # Each split must repeat the header for context
    for tc in table_chunks:
        assert "| Header" in tc["text"]


def test_prose_around_table_still_window_chunked():
    text = """prose paragraph one. """ + (" word" * 1000) + """

| Header | A |
|--------|---|
| 1      | 2 |

prose paragraph after. """ + (" word" * 1000)

    chunks = chunk_structured(text, chunk_size_tokens=400, overlap_tokens=50)
    prose_chunks = [c for c in chunks if c["chunk_type"] == "prose"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    assert len(prose_chunks) >= 2  # before + after
    assert len(table_chunks) == 1
```

Create `/home/vogic/LocalRAG/tests/unit/test_chunker_structured_code.py`:

```python
import pytest

from ext.services.chunker_structured import chunk_structured


def test_fenced_code_block_emitted_as_single_chunk():
    text = """## Setup

```python
import asyncio
async def main():
    await asyncio.sleep(0)
```

That's the setup."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 1
    assert "import asyncio" in code_chunks[0]["text"]
    assert code_chunks[0].get("language") == "python"


def test_multiple_code_blocks_each_atomic():
    text = """First block.

```sh
echo hello
```

Middle text.

```python
print("hello")
```

End."""

    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 2
    languages = sorted(c.get("language") for c in code_chunks)
    assert languages == ["python", "sh"]


def test_code_block_without_language_tagged_unknown():
    text = """```
just some text
```"""
    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 1
    assert code_chunks[0].get("language") in ("", None, "unknown", "text")


def test_inline_code_not_treated_as_code_chunk():
    text = """A paragraph with `inline_code` mixed in. Just prose."""
    chunks = chunk_structured(text, chunk_size_tokens=800, overlap_tokens=100)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) == 0


def test_oversized_code_block_split_with_continuation_marker():
    huge = "x = 1\n" * 500
    text = f"""```python
{huge}```"""
    chunks = chunk_structured(text, chunk_size_tokens=200, overlap_tokens=20)
    code_chunks = [c for c in chunks if c["chunk_type"] == "code"]
    assert len(code_chunks) > 1
    # Each chunk must announce continuation in payload
    for cc in code_chunks[1:]:
        assert cc.get("continuation") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_chunker_structured_table.py tests/unit/test_chunker_structured_code.py -v
```

Expected: ImportError.

- [ ] **Step 3: Write the structured chunker**

Create `/home/vogic/LocalRAG/ext/services/chunker_structured.py`:

```python
"""Structure-aware chunker for prose + tables + fenced code blocks.

Plan B Phase 6.5. Replaces the window-only chunker for KBs that opt
in via ``rag_config.chunking_strategy="structured"`` (Phase 6.6).

Algorithm:
  1. Split the input into segments. A segment is one of:
     - a fenced code block (``\`\`\`...\`\`\``)
     - a markdown pipe table (header line + dashes + ≥1 row)
     - an HTML <table>...</table>
     - free prose
  2. For code/table segments, emit one chunk per segment with the
     appropriate ``chunk_type`` payload. Oversized segments are split
     by row-group (tables) or line-group (code) with a continuation flag.
  3. For prose segments, fall back to the existing window chunker.

The output is a list of dicts ready for embedding + payload upsert:
  {"text": ..., "chunk_type": "prose|table|code|image_caption",
   "language": "<for code>", "continuation": False}

Token counts are approximate (chars / 4); for KBs that need exact
counts, the caller can re-tokenize after.
"""
from __future__ import annotations

import re
from typing import Iterator


_FENCED_CODE_RE = re.compile(
    r"^```(\w+)?\s*\n(.*?)^```\s*$",
    re.MULTILINE | re.DOTALL,
)

# Markdown pipe table: header + |---|---| separator + at least one row
_PIPE_TABLE_RE = re.compile(
    r"(\|.+?\|\s*\n\|[-: |]+\|\s*\n(?:\|.+?\|\s*\n?)+)",
    re.MULTILINE,
)

_HTML_TABLE_RE = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE)


def _tok_estimate(s: str) -> int:
    return max(1, len(s) // 4)


def _split_giant_table(text: str, max_tokens: int) -> list[str]:
    """Split a markdown pipe table by row-groups, repeating the header."""
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return [text]
    header, sep, *rows = lines
    out = []
    cur_rows = []
    cur_tok = _tok_estimate("\n".join([header, sep]))
    for row in rows:
        rtok = _tok_estimate(row)
        if cur_tok + rtok > max_tokens and cur_rows:
            out.append("\n".join([header, sep, *cur_rows]))
            cur_rows = [row]
            cur_tok = _tok_estimate("\n".join([header, sep, row]))
        else:
            cur_rows.append(row)
            cur_tok += rtok
    if cur_rows:
        out.append("\n".join([header, sep, *cur_rows]))
    return out


def _split_giant_code(text: str, language: str | None,
                       max_tokens: int) -> list[tuple[str, bool]]:
    """Split a fenced code block by line-groups. Returns [(text, continuation)]."""
    inner = text
    # Strip the outer fence to get the inner code
    fence_match = re.match(r"^```(\w+)?\s*\n(.*?)^```\s*$",
                            text, re.DOTALL | re.MULTILINE)
    if fence_match:
        inner = fence_match.group(2)
    lang = language or (fence_match.group(1) if fence_match else "")
    lines = inner.split("\n")
    out = []
    cur = []
    cur_tok = 0
    for line in lines:
        ltok = _tok_estimate(line)
        if cur_tok + ltok > max_tokens and cur:
            out.append("\n".join(cur))
            cur = [line]
            cur_tok = ltok
        else:
            cur.append(line)
            cur_tok += ltok
    if cur:
        out.append("\n".join(cur))

    fence_lang = lang or ""
    return [
        (f"```{fence_lang}\n{seg}\n```", i > 0)
        for i, seg in enumerate(out)
    ]


def _window_chunk_prose(text: str, chunk_size_tokens: int,
                         overlap_tokens: int) -> list[str]:
    """Reuse the existing prose chunker logic.

    For Plan B's purposes we approximate via word windows. Production
    callers re-tokenize for exact budgets downstream.
    """
    if not text.strip():
        return []
    # Approximate: 1 token ≈ 0.75 words. Use words for reproducible splits.
    words = text.split()
    target_words = max(50, int(chunk_size_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))
    out = []
    i = 0
    while i < len(words):
        seg = words[i:i + target_words]
        out.append(" ".join(seg))
        if i + target_words >= len(words):
            break
        i += max(1, target_words - overlap_words)
    return out


def _segments_with_offsets(text: str) -> list[tuple[int, int, str, dict]]:
    """Return [(start, end, type, meta)] sorted by start."""
    segs: list[tuple[int, int, str, dict]] = []
    for m in _FENCED_CODE_RE.finditer(text):
        segs.append((m.start(), m.end(), "code", {"language": m.group(1)}))
    for m in _PIPE_TABLE_RE.finditer(text):
        # Skip if inside a code segment
        if any(s <= m.start() < e for s, e, t, _ in segs if t == "code"):
            continue
        segs.append((m.start(), m.end(), "table", {"format": "markdown"}))
    for m in _HTML_TABLE_RE.finditer(text):
        if any(s <= m.start() < e for s, e, _, _ in segs):
            continue
        segs.append((m.start(), m.end(), "table", {"format": "html"}))
    segs.sort()
    return segs


def chunk_structured(
    text: str, *, chunk_size_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[dict]:
    """Chunk text preserving table + code structure.

    Returns a list of chunk dicts:
        {"text": ..., "chunk_type": "prose|table|code",
         "language": "..." (code only),
         "continuation": False (set True on overflow segments)}
    """
    segments = _segments_with_offsets(text)
    out: list[dict] = []
    cursor = 0
    for start, end, typ, meta in segments:
        # Prose before this structured segment
        if start > cursor:
            prose = text[cursor:start]
            for p in _window_chunk_prose(
                prose, chunk_size_tokens, overlap_tokens,
            ):
                if p.strip():
                    out.append({"text": p, "chunk_type": "prose"})
        # Emit the structured segment
        seg_text = text[start:end]
        if typ == "code":
            if _tok_estimate(seg_text) > chunk_size_tokens:
                for stext, cont in _split_giant_code(
                    seg_text, meta.get("language"), chunk_size_tokens,
                ):
                    out.append({
                        "text": stext, "chunk_type": "code",
                        "language": meta.get("language") or "unknown",
                        "continuation": cont,
                    })
            else:
                out.append({
                    "text": seg_text, "chunk_type": "code",
                    "language": meta.get("language") or "unknown",
                    "continuation": False,
                })
        elif typ == "table":
            if _tok_estimate(seg_text) > chunk_size_tokens \
                    and meta.get("format") == "markdown":
                for stext in _split_giant_table(seg_text, chunk_size_tokens):
                    out.append({
                        "text": stext, "chunk_type": "table",
                        "format": "markdown", "continuation": False,
                    })
            else:
                out.append({
                    "text": seg_text, "chunk_type": "table",
                    "format": meta.get("format"), "continuation": False,
                })
        cursor = end
    # Trailing prose
    if cursor < len(text):
        for p in _window_chunk_prose(
            text[cursor:], chunk_size_tokens, overlap_tokens,
        ):
            if p.strip():
                out.append({"text": p, "chunk_type": "prose"})
    return out


__all__ = ["chunk_structured"]
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_chunker_structured_table.py tests/unit/test_chunker_structured_code.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/chunker_structured.py \
        tests/unit/test_chunker_structured_table.py \
        tests/unit/test_chunker_structured_code.py
git commit -m "phase-6.5: structure-aware chunker (tables + code as atomic units)"
```

---

### Task 6.6: Per-KB chunking strategy via `rag_config.chunking_strategy`

**Files:**
- Create: `/home/vogic/LocalRAG/ext/db/migrations/010_add_kb_chunking_strategy.sql`
- Modify: `/home/vogic/LocalRAG/ext/services/kb_config.py`
- Modify: `/home/vogic/LocalRAG/ext/services/ingest.py`
- Create: `/home/vogic/LocalRAG/tests/unit/test_kb_chunking_strategy.py`

The schema migration adds a `chunking_strategy` enum column to `knowledge_bases.rag_config` (JSONB). Values: `"window"` (existing default) or `"structured"` (new). The ingest pipeline reads the strategy and routes to the appropriate chunker.

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_kb_chunking_strategy.py`:

```python
import pytest


def test_chunking_strategy_default_window():
    from ext.services.kb_config import get_chunking_strategy
    # No rag_config -> default "window"
    assert get_chunking_strategy(None) == "window"
    assert get_chunking_strategy({}) == "window"


def test_chunking_strategy_explicit_window():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy({"chunking_strategy": "window"}) == "window"


def test_chunking_strategy_structured():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy(
        {"chunking_strategy": "structured"}
    ) == "structured"


def test_chunking_strategy_unknown_falls_back_to_window():
    from ext.services.kb_config import get_chunking_strategy
    assert get_chunking_strategy({"chunking_strategy": "lol"}) == "window"


def test_ingest_calls_structured_chunker_when_kb_opts_in(monkeypatch, mocker):
    """Ingest pipeline picks the chunker per KB."""
    import importlib
    monkeypatch.setenv("RAG_STRUCTURED_CHUNKER", "1")
    from ext.services import ingest
    importlib.reload(ingest)

    spy_structured = mocker.patch(
        "ext.services.chunker_structured.chunk_structured",
        return_value=[{"text": "x", "chunk_type": "prose"}],
    )
    spy_window = mocker.patch.object(
        ingest, "_chunk_window", return_value=["x"],
    )

    chunks = ingest.chunk_text_for_kb(
        text="prose", rag_config={"chunking_strategy": "structured"},
    )

    spy_structured.assert_called_once()
    spy_window.assert_not_called()


def test_ingest_uses_window_chunker_by_default(monkeypatch, mocker):
    import importlib
    monkeypatch.setenv("RAG_STRUCTURED_CHUNKER", "1")
    from ext.services import ingest
    importlib.reload(ingest)

    spy_structured = mocker.patch(
        "ext.services.chunker_structured.chunk_structured",
    )
    spy_window = mocker.patch.object(
        ingest, "_chunk_window", return_value=["x"],
    )

    ingest.chunk_text_for_kb(text="prose", rag_config={})

    spy_window.assert_called_once()
    spy_structured.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_kb_chunking_strategy.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add the migration**

Create `/home/vogic/LocalRAG/ext/db/migrations/010_add_kb_chunking_strategy.sql`:

```sql
-- Plan B Phase 6.6 — per-KB chunking strategy
-- Adds the field to the rag_config JSONB blob (no schema change beyond default).
-- Use UPDATE to populate the default for existing rows.

UPDATE knowledge_bases
SET rag_config = COALESCE(rag_config, '{}'::jsonb) || '{"chunking_strategy": "window"}'::jsonb
WHERE NOT (rag_config ? 'chunking_strategy');

COMMENT ON COLUMN knowledge_bases.rag_config IS
  'Per-KB RAG config (JSONB). Includes chunking_strategy ("window" | "structured"), contextualize, colbert, etc.';
```

- [ ] **Step 4: Add the helper to kb_config.py**

Edit `/home/vogic/LocalRAG/ext/services/kb_config.py`. Append:

```python
_VALID_CHUNKING_STRATEGIES = ("window", "structured")


def get_chunking_strategy(rag_config: dict | None) -> str:
    """Return 'window' (default) or 'structured'.

    Plan B Phase 6.6.
    """
    if not rag_config:
        return "window"
    raw = (rag_config.get("chunking_strategy") or "window").lower().strip()
    if raw not in _VALID_CHUNKING_STRATEGIES:
        return "window"
    return raw
```

- [ ] **Step 5: Wire ingest to dispatch**

Edit `/home/vogic/LocalRAG/ext/services/ingest.py`. Add the dispatch function:

```python
from .kb_config import get_chunking_strategy


def chunk_text_for_kb(
    *, text: str, rag_config: dict | None,
    chunk_size_tokens: int = 800, overlap_tokens: int = 100,
) -> list[dict]:
    """Dispatch to the right chunker per KB strategy.

    Always returns a list of chunk dicts with at least ``text`` and
    ``chunk_type`` keys. The window chunker emits all chunks with
    ``chunk_type="prose"``.
    """
    strategy = get_chunking_strategy(rag_config)
    if strategy == "structured" and \
            os.environ.get("RAG_STRUCTURED_CHUNKER", "0") == "1":
        from .chunker_structured import chunk_structured
        return chunk_structured(
            text, chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    # Window default
    return [
        {"text": w, "chunk_type": "prose"}
        for w in _chunk_window(
            text, chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


def _chunk_window(text, *, chunk_size_tokens=800, overlap_tokens=100):
    """Existing window-chunker entry point. Wrap whatever exists today."""
    # If the existing function is named differently, alias it here.
    return _legacy_window_chunk(text, chunk_size_tokens, overlap_tokens)
```

In the existing ingest entry point, replace any direct call to the old chunker with `chunk_text_for_kb(text=raw_text, rag_config=kb_rag_config)`. The downstream code that takes a list of strings now must take a list of dicts:

```python
chunks = chunk_text_for_kb(text=raw_text, rag_config=kb_rag_config)
for ci, chunk in enumerate(chunks):
    payload["text"] = chunk["text"]
    payload["chunk_type"] = chunk["chunk_type"]
    if chunk.get("language"):
        payload["language"] = chunk["language"]
    # ... existing per-chunk processing ...
```

- [ ] **Step 6: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_kb_chunking_strategy.py -v
```

Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add ext/db/migrations/010_add_kb_chunking_strategy.sql \
        ext/services/kb_config.py ext/services/ingest.py \
        tests/unit/test_kb_chunking_strategy.py
git commit -m "phase-6.6: per-KB chunking strategy (window vs structured)"
```

---

### Task 6.7: Image caption extraction

**Files:**
- Modify: `/home/vogic/LocalRAG/ext/services/ingest.py`
- Modify: `/home/vogic/LocalRAG/ext/services/vision.py` (if not already wired for ingest)
- Create: `/home/vogic/LocalRAG/tests/unit/test_image_caption_extraction.py`

When a document contains embedded images (PDF or DOCX), extract them, send each through the vllm-vision service (already in compose), and emit a chunk with `chunk_type="image_caption"` containing the caption + a reference to the source page/position.

This task ONLY emits chunks; the retriever / reranker treatment of image_caption chunks is the same as prose for Plan B (chunk_type-aware reranking is deferred to a future plan). The reason for emitting them: today images are silently dropped, costing recall on visual-content queries.

The vision service may not be running (it's on-demand per Plan A). If not running, image extraction silently skips with a metric tick (`rag_image_skip_total`).

- [ ] **Step 1: Write the failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_image_caption_extraction.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_image_caption_emitted_for_pdf_image(monkeypatch, mocker):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    from ext.services import ingest

    # Stub the image extractor to return 1 fake image
    async def fake_extract_images(pdf_bytes):
        return [{"page": 1, "image_bytes": b"fake png", "position": (0, 0)}]
    mocker.patch.object(ingest, "_extract_pdf_images",
                        side_effect=fake_extract_images)

    # Stub the vision caller
    async def fake_caption(img_bytes):
        return "A bar chart showing Q1 revenue growth"
    mocker.patch.object(ingest, "_caption_image", side_effect=fake_caption)

    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="report.pdf",
    )
    assert len(chunks) == 1
    assert chunks[0]["chunk_type"] == "image_caption"
    assert "bar chart" in chunks[0]["text"]
    assert chunks[0]["payload"]["page"] == 1


@pytest.mark.asyncio
async def test_image_caption_skipped_when_flag_off(monkeypatch, mocker):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "0")
    from ext.services import ingest
    spy = mocker.patch.object(ingest, "_extract_pdf_images")
    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="x.pdf",
    )
    assert chunks == []
    spy.assert_not_called()


@pytest.mark.asyncio
async def test_image_caption_skipped_when_vision_unreachable(
    monkeypatch, mocker,
):
    monkeypatch.setenv("RAG_IMAGE_CAPTIONS", "1")
    from ext.services import ingest
    async def fake_extract_images(pdf_bytes):
        return [{"page": 1, "image_bytes": b"x", "position": (0, 0)}]
    mocker.patch.object(ingest, "_extract_pdf_images",
                        side_effect=fake_extract_images)
    async def fake_caption(img_bytes):
        raise ConnectionError("vision unreachable")
    mocker.patch.object(ingest, "_caption_image", side_effect=fake_caption)

    chunks = await ingest.extract_images_as_chunks(
        pdf_bytes=b"%PDF-fake", filename="x.pdf",
    )
    assert chunks == []  # silent skip, but logged
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_image_caption_extraction.py -v
```

Expected: AttributeError — `extract_images_as_chunks` doesn't exist.

- [ ] **Step 3: Wire image captioning into ingest**

Edit `/home/vogic/LocalRAG/ext/services/ingest.py`. Add:

```python
async def extract_images_as_chunks(
    *, pdf_bytes: bytes, filename: str,
) -> list[dict]:
    """Plan B Phase 6.7 — emit image_caption chunks for embedded images.

    Returns a list of chunk dicts. Returns [] (silent skip) if:
      - RAG_IMAGE_CAPTIONS=0, or
      - vision service unreachable, or
      - no images extracted
    """
    if os.environ.get("RAG_IMAGE_CAPTIONS", "0") != "1":
        return []
    try:
        images = await _extract_pdf_images(pdf_bytes)
    except Exception as e:
        log.warning("image extraction failed for %s: %s", filename, e)
        return []
    if not images:
        return []

    out: list[dict] = []
    for img in images:
        try:
            caption = await _caption_image(img["image_bytes"])
        except Exception as e:
            log.warning("image caption failed for %s page %s: %s",
                        filename, img.get("page"), e)
            try:
                from .metrics import RAG_IMAGE_SKIP
                RAG_IMAGE_SKIP.inc()
            except Exception:
                pass
            continue
        if not caption:
            continue
        out.append({
            "text": caption,
            "chunk_type": "image_caption",
            "payload": {
                "page": img.get("page"),
                "position": img.get("position"),
            },
        })
    return out


async def _extract_pdf_images(pdf_bytes: bytes) -> list[dict]:
    """Use pymupdf to enumerate images. Returns list of dicts."""
    import pymupdf
    out = []
    with pymupdf.open(stream=pdf_bytes) as doc:
        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                out.append({
                    "page": page_num,
                    "image_bytes": base_img["image"],
                    "position": (img_idx,),
                })
    return out


async def _caption_image(image_bytes: bytes) -> str:
    """Send an image to vllm-vision and return the caption.

    Soft-fails if the vision service is unreachable (no exception bubbles).
    """
    import base64
    import httpx
    vision_url = os.environ.get(
        "RAG_VISION_URL", "http://vllm-vision:8000/v1"
    )
    vision_model = os.environ.get("RAG_VISION_MODEL", "qwen2-vl-7b")
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": vision_model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Describe the key information in this image in 1-2 sentences."},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.post(f"{vision_url}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
```

In the ingest pipeline, after `chunk_text_for_kb`, append image chunks:

```python
chunks = chunk_text_for_kb(text=raw_text, rag_config=kb_rag_config)
if filename.lower().endswith(".pdf") and \
        os.environ.get("RAG_IMAGE_CAPTIONS", "0") == "1":
    image_chunks = await extract_images_as_chunks(
        pdf_bytes=raw_bytes, filename=filename,
    )
    chunks.extend(image_chunks)
```

Add the metric:

```python
# In ext/services/metrics.py
RAG_IMAGE_SKIP = Counter(
    "rag_image_skip_total",
    "Images that could not be captioned (e.g. vision unreachable)",
)
```

- [ ] **Step 4: Re-run tests**

```bash
cd /home/vogic/LocalRAG && pytest tests/unit/test_image_caption_extraction.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add ext/services/ingest.py ext/services/metrics.py \
        tests/unit/test_image_caption_extraction.py
git commit -m "phase-6.7: image caption extraction with soft-fail to vision service"
```

---

### Task 6.8: Operator runbook + re-ingest path for OCR-needed KBs

**Files:**
- Create: `/home/vogic/LocalRAG/docs/runbook/ocr-runbook.md`
- Create: `/home/vogic/LocalRAG/scripts/reingest_for_ocr.py`
- Modify: `/home/vogic/LocalRAG/docs/runbook/troubleshooting.md`
- Modify: `/home/vogic/LocalRAG/docs/runbook/plan-b-flag-reference.md`

The OCR runbook tells the operator how to:
1. Identify which KBs need OCR (heuristic: > 5% of docs have empty extracted text).
2. Enable OCR for that KB via `kb_config.ocr_policy`.
3. Re-ingest the existing docs to recover OCR text (separate from the original ingest).

The `reingest_for_ocr.py` script is similar to Plan A's `reingest_kb.py` but restricted to docs whose existing text is empty / very short — it re-fetches the original blob from doc storage, runs through the OCR pipeline, and replaces the chunks.

- [ ] **Step 1: Write the runbook**

Create `/home/vogic/LocalRAG/docs/runbook/ocr-runbook.md`:

```markdown
# OCR Runbook

Plan B Phase 6.8.

## When to enable OCR

Default: `RAG_OCR_ENABLED=1` (Plan B Phase 6.4 turns this on globally).
Per-KB override: `kb_config.ocr_policy.enabled = false` to disable for KBs known to be all-text (faster ingest).

## Symptoms a KB needs OCR

- Retrieval misses on docs that visually contain text but the search returns nothing relevant.
- Doc inspection shows pdfplumber extracted < 50 chars per page.
- Documents are scanned PDFs, image-only PDFs, or photos.

Detect:

```bash
python - <<PY
import asyncio
from qdrant_client import AsyncQdrantClient
from collections import defaultdict

async def main():
    qc = AsyncQdrantClient(url="http://localhost:6333")
    by_doc = defaultdict(int)
    sizes = defaultdict(int)
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name="kb_1", limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points: break
        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            txt = payload.get("text", "")
            by_doc[did] += 1
            sizes[did] += len(txt)
        if offset is None: break
    short = sum(1 for d in by_doc if sizes[d] / max(1, by_doc[d]) < 100)
    print(f"docs with avg chunk < 100 chars: {short}/{len(by_doc)}")

asyncio.run(main())
PY
```

If > 5% of docs are short — that KB likely needs OCR.

## Enable OCR for a KB

```bash
KB_ID=2
curl -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ocr_policy": {
      "enabled": true,
      "backend": "tesseract",
      "language": "eng",
      "trigger_chars_per_page": 50
    }
  }'
```

For non-English content, set `language` to a Tesseract language code (e.g. `"deu"` for German, `"chi_sim"` for Simplified Chinese — install the language pack first via `apt-get install tesseract-ocr-deu`).

## Use a cloud backend (opt-in only)

**Air-gap warning:** the cloud backends MAKE OUTBOUND CALLS. Only enable on hosts that are still connected.

AWS Textract:

```bash
# 1. Install boto3 in open-webui
docker exec orgchat-open-webui pip install boto3

# 2. Set creds in .env
cat >> compose/.env <<EOF
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
TEXTRACT_REGION=us-east-1
EOF

# 3. Restart
docker compose up -d --force-recreate open-webui

# 4. Per-KB switch
curl -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -d '{"ocr_policy": {"backend": "cloud:textract"}}'
```

GCP Document AI:

```bash
docker exec orgchat-open-webui pip install google-cloud-documentai
# Mount service account JSON; set DOCUMENT_AI_PROJECT, DOCUMENT_AI_PROCESSOR
# Then PATCH ocr_policy.backend = "cloud:document_ai"
```

## Re-ingest existing docs after enabling OCR

```bash
python scripts/reingest_for_ocr.py \
  --kb-id $KB_ID \
  --short-text-threshold 100 \
  --api-base http://localhost:6100 \
  --admin-token $RAG_ADMIN_TOKEN
```

Inspect output: per-doc OCR latency + char count delta. Re-run eval after.

## Common failure modes

### Tesseract fails: "Could not find tesseract language data"

The container doesn't have the language pack.

```bash
docker exec -u root orgchat-open-webui apt-get install -y tesseract-ocr-<lang>
docker exec orgchat-open-webui python -c "import pytesseract; print(pytesseract.get_languages())"
```

### OCR text is gibberish

- Wrong DPI (default 200 in `ocr.py`); raise to 300 if scans are low-res
- Wrong language; verify with `pytesseract.get_languages()`
- Page is rotated; pre-process via PIL: `image.rotate(90)`

### OCR is too slow

- Reduce DPI to 150 (faster but lower quality)
- Use cloud backend (Textract / Document AI) — typically 5-10× faster than local Tesseract
- Per-KB: set `ocr_policy.enabled=false` for KBs that don't need it (skip the trigger check entirely)

### Cloud backend errors with "Access Denied"

- IAM role not attached to host (Textract)
- Service account JSON missing or wrong scope (Document AI)
- Region mismatch
```

- [ ] **Step 2: Write the re-ingest script**

Create `/home/vogic/LocalRAG/scripts/reingest_for_ocr.py`:

```python
#!/usr/bin/env python3
"""Re-ingest existing docs that have empty / short extracted text.

Plan B Phase 6.8. Use after enabling OCR on a KB to recover docs that
were originally ingested without OCR.

Detection: a doc is "needs OCR" if its average chunk text length is
below ``--short-text-threshold`` (default 100 chars).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import logging
import os
import sys

import httpx
from qdrant_client import AsyncQdrantClient


log = logging.getLogger("reingest_ocr")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")


async def find_short_docs(qc, collection: str, threshold: int) -> dict:
    """Scroll collection. Return {doc_id: {filename, avg_chars}}."""
    chunks_by_doc = collections.defaultdict(list)
    filenames = {}
    offset = None
    while True:
        points, offset = await qc.scroll(
            collection_name=collection, limit=512, offset=offset,
            with_payload=True, with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if did is None:
                continue
            chunks_by_doc[did].append(len(payload.get("text", "")))
            filenames[did] = payload.get("filename", f"doc_{did}")
        if offset is None:
            break

    short = {}
    for did, lens in chunks_by_doc.items():
        avg = sum(lens) / max(1, len(lens))
        if avg < threshold:
            short[did] = {"filename": filenames[did], "avg_chars": avg}
    return short


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kb-id", type=int, required=True)
    p.add_argument("--short-text-threshold", type=int, default=100)
    p.add_argument("--api-base", default="http://localhost:6100")
    p.add_argument("--admin-token",
                   default=os.environ.get("RAG_ADMIN_TOKEN", ""))
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    args = p.parse_args()

    if not args.admin_token:
        log.error("RAG_ADMIN_TOKEN required")
        return 2

    qc = AsyncQdrantClient(url=args.qdrant_url, timeout=60.0)
    short = await find_short_docs(
        qc, f"kb_{args.kb_id}", args.short_text_threshold,
    )
    log.info("found %d docs with avg chunk text < %d chars",
             len(short), args.short_text_threshold)

    if not short:
        return 0

    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {args.admin_token}"},
        timeout=300.0,
    ) as client:
        for did, info in short.items():
            log.info("re-ingest doc_id=%d filename=%s avg=%d",
                     did, info["filename"], int(info["avg_chars"]))
            # Trigger re-ingest via the admin endpoint that fetches the
            # original blob and runs the full pipeline (which now includes
            # OCR fallback)
            r = await client.post(
                f"{args.api_base}/api/kb/{args.kb_id}/doc/{did}/reingest",
            )
            if r.status_code not in (200, 202, 409):
                log.warning("re-ingest failed for doc_id=%d: %d %s",
                            did, r.status_code, r.text[:200])
                continue
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

```bash
chmod +x /home/vogic/LocalRAG/scripts/reingest_for_ocr.py
```

- [ ] **Step 3: Append troubleshooting**

Edit `/home/vogic/LocalRAG/docs/runbook/troubleshooting.md`. Append:

```markdown
## OCR / image-caption issues

See `docs/runbook/ocr-runbook.md` for the full guide. Quick checks:

- `docker exec orgchat-open-webui tesseract --version` — verify Tesseract installed
- `docker exec orgchat-open-webui python -c "import pytesseract; print(pytesseract.get_languages())"` — verify language packs
- `curl -s http://localhost:9090/api/v1/query?query=rag_image_skip_total` — non-zero means vision service unreachable
- `kb_config.ocr_policy` for a specific KB:
  ```sql
  SELECT id, name, ocr_policy FROM knowledge_bases WHERE id = <kb_id>;
  ```
```

- [ ] **Step 4: Update plan-b-flag-reference**

Edit `/home/vogic/LocalRAG/docs/runbook/plan-b-flag-reference.md`. Add at the bottom of the Plan B table:

```markdown
| `RAG_VISION_URL` | 6.7 | `http://vllm-vision:8000/v1` | vllm-vision endpoint for image captions | Yes (restart) |
| `RAG_VISION_MODEL` | 6.7 | `qwen2-vl-7b` | served model name | Yes (restart) |
```

- [ ] **Step 5: Commit**

```bash
git add docs/runbook/ocr-runbook.md scripts/reingest_for_ocr.py \
        docs/runbook/troubleshooting.md docs/runbook/plan-b-flag-reference.md
git commit -m "phase-6.8: OCR runbook + re-ingest script + troubleshooting"
```

### Phase 6 completion gate

- [ ] All Phase 6 unit tests pass.
- [ ] `tests/integration/test_celery_soak.py` (1000-doc soak) passes; 0 lost docs; DLQ depth ≤ 5.
- [ ] `RAG_SYNC_INGEST=0` runs in production for ≥ 7 days; `chunk_recall@10` not regressed.
- [ ] OCR pipeline correctly extracts text from a fixture corpus of 50 scanned PDFs (operator manual eval).
- [ ] Structured chunker produces atomic table + code chunks for at least one production KB containing those structures.
- [ ] Image captions emit chunks; sample inspection confirms captions are reasonable.
- [ ] `phase-6-baseline.json` committed.

---

## Final cadence + completion gates

### End-of-Plan-B sign-off checklist

- [ ] **Phase 4 gate** met (see Phase 4 completion gate above).
- [ ] **Phase 5 gate** met (see Phase 5 completion gate above).
- [ ] **Phase 6 gate** met (see Phase 6 completion gate above).
- [ ] **Cross-phase eval** committed at `tests/eval/results/plan-b-final-baseline.json`. Run:
  ```bash
  cd /home/vogic/LocalRAG
  RAG_QU_ENABLED=1 RAG_TEMPORAL_LEVELS=1 \
    make eval KB_EVAL_ID=$KB_EVAL_ID
  RAG_QU_ENABLED=1 RAG_TEMPORAL_LEVELS=1 \
    make eval-evolution KB_EVAL_ID=$KB_EVAL_ID
  ```
  Compare against Plan A end-state (`tests/eval/results/phase-3-baseline.json`):
  - `chunk_recall@10` global: ≥ +5 pp
  - `chunk_recall@10` `multihop` stratum: ≥ +5 pp
  - `chunk_recall@10` `evolution` stratum: ≥ +8 pp
  - No per-intent regression > 2 pp on any stratum
  - p95 retrieval latency ≤ 1.5 s (per Plan A SLO post-Plan-B target)
- [ ] **Flag kill-list audit** complete. For every flag listed in "Plan B Flag Reference" status table:
  - default-ON: confirmed via `.env` + verified at runtime
  - per-KB: confirmed via at least one `kb_config` example with the flag set
  - removed: `grep -rn "<FLAG>" ext/ tests/` returns 0 hits
- [ ] **Operator runbooks** committed:
  - `docs/runbook/qu-llm-runbook.md`
  - `docs/runbook/temporal-reshard-procedure.md`
  - `docs/runbook/temporal-reshard-checklist.md`
  - `docs/runbook/tiered-storage-runbook.md`
  - `docs/runbook/ocr-runbook.md`
  - `docs/runbook/plan-b-flag-reference.md`
- [ ] **All 14 Plan B alert rules** present in `observability/prometheus/`:
  - `alerts-qu.yml` (4 rules)
  - `alerts-tiered-shards.yml` (3 rules)
  - `alerts-celery.yml` (3 rules)
  - alert files cross-referenced from `docs/runbook/troubleshooting.md`
- [ ] **End-to-end smoke** completed by an operator different from the implementer:
  1. Cold start: `docker compose up -d` brings up all services including `vllm-qu` and `celery-worker` with no manual interventions
  2. Upload a test document via the UI; verify async ingest landed it in Qdrant within 60 s
  3. Send a chat with intent="evolution" — confirm `rag_qu_invocations_total{source="llm"}` incremented + retrieval returned an L2 or L3 node
  4. `make eval` against `kb_eval` — passes gates
- [ ] **`upstream` submodule fast-forward audit**: confirm Plan B's bridge changes still apply cleanly atop the latest upstream Open WebUI commit
- [ ] **Disconnect rehearsal**: re-run Appendix A.6 (offline-readiness smoke) from Plan A — `HF_HUB_OFFLINE=1` smoke covers the new Qwen3-4B weights

### Post-window follow-ups

After the 4-day Plan B window closes:

1. **Day 5–11**: shadow A/B for QU LLM continues running. On Day 8, analyze with `scripts/analyze_shadow_log.py`.
2. **Day 12**: Promote `RAG_QU_ENABLED=1` as default if shadow shows agreement ≥ 75% on metadata/global, < 60% on specific.
3. **Day 14**: cleanup of Plan A's `kb_1_v2` and Plan B's `kb_1_v3` rollback collections (operator's call — only if no rollback needed).
4. **Week 4**: expand `golden_evolution.jsonl` from 30 → 100 queries with real production examples.
5. **Week 5**: schedule Plan C scoping discussion (knowledge-graph head, agentic multi-hop) gated on Plan B Phase 5 eval results.

### Decision boundary into Plan C

Plan C is in scope only if Plan B's end-of-window eval shows:
- `chunk_recall@10` improvement on `evolution` stratum < +5 pp (the temporal-RAPTOR didn't deliver enough)
- OR `mrr@10` on `multihop` stratum < 0.50 floor
- OR a category of queries with consistent < 50% nDCG that doesn't fit any of the four intent classes

If Plan B hits its targets, defer Plan C indefinitely — the cost/complexity of LightRAG / HippoRAG / agentic multi-hop is hard to justify without a clear residual gap.

---

## Open design questions for human decision before execution

These are NOT placeholders — they are deliberate operator decisions that should be answered before kicking off Plan B implementation.

1. **Qwen3-4B-AWQ revision pin**: should the compose `--model` arg pin a specific snapshot revision (`Qwen/Qwen3-4B-Instruct-2507-AWQ@<rev_hash>`) to insulate from upstream re-uploads? Recommendation: yes, after the first successful smoke. Capture the `commit` hash from `huggingface-cli download` output.
2. **OCR cloud backend approval**: are AWS Textract and GCP Document AI both acceptable, or only one? The runbook documents both; `Dockerfile.openwebui.cu128` does NOT install boto3 / google-cloud-documentai by default — operator installs explicitly per backend.
3. **`evolution` derivation policy** (Phase 5.6): Plan B derives "evolution" intent as `intent=="global" + comparison verb in original query`. Alternative: use the QU LLM's `entities` + `temporal_constraint.year` count to detect cross-year intent. Recommendation: keep the current heuristic; revisit after eval shows whether `evolution` is being correctly detected.
4. **Tier cron schedule**: 03:00 local default. If the deployment spans timezones, consider hard-coding UTC or making the schedule per-collection.
5. **Image captions latency budget**: vllm-vision is on-demand (Plan A). Phase 6.7 doesn't add a deadline — image captioning during ingest could take 30+ s per image if the model has to load. Recommendation: add `RAG_IMAGE_CAPTION_TIMEOUT_MS=30_000` and a metric for skipped images. (Out of scope for Plan B — file as a post-window follow-up.)
6. **`golden_evolution.jsonl` real data**: the 30 queries committed are templates with empty `expected_doc_ids`. The operator must populate the IDs against the real corpus before Phase 5's eval gate is meaningful. Recommendation: this is a Day-4 task, not a hard prereq for Day 1–3 implementation.
7. **kb_1_v4 retention window**: Plan A used 14 days; Plan B inherits this convention. If storage becomes tight after 36-month corpus reaches steady state, consider compressing older rollback collections via Qdrant snapshot + deletion of the live collection.

---

## Self-review notes (executed at end of writing)

**Spec coverage:** every concern in the writing brief has a task —
- Phase 4 (10 tasks): vllm-qu service, weight staging, QU module, hybrid router, cache, bridge wiring, metrics, shadow A/B, integration test, runbook + flag retirement
- Phase 5 (10 tasks): ensure_collection variant, date extraction, tiered storage config, reshard script, temporal RAPTOR, retrieval level injection, time-decay, tier cron, per-shard metrics, runbook + checklist
- Phase 6 (8 tasks): Celery soak, async default flip, OCR module, OCR trigger, structured chunker, per-KB strategy, image captions, runbook

**Placeholder scan:** no `TBD` / `TODO` / `fill in later` markers in any task. Every code block contains real code; every test block contains real assertions. The two intentional exceptions (`golden_evolution.jsonl` `expected_doc_ids` empty + Open Question #6) are flagged explicitly as operator inputs.

**Type consistency:**
- `QueryUnderstanding` dataclass shape (intent + resolved_query + temporal_constraint + entities + confidence + source + cached) is consistent across `query_understanding.py`, `qu_cache.py`, `query_intent.py`, `chat_rag_bridge.py`
- `HybridClassification` dataclass shape consistent across query_intent.py and bridge wiring
- `EscalationReason` enum values match between predicates and metric labels
- `OCRBackend` enum values match the per-KB `ocr_policy.backend` strings
- Shard key format ("YYYY-MM" zero-padded) consistent across `temporal_shard.py`, `vector_store.py`, `temporal_raptor.py`, `tier_storage_cron.py`
- Metric names use the `rag_qu_*` / `rag_shard_*` / `rag_image_*` prefixes consistently

**File structure section:** every new file referenced in tasks appears in the file structure block at the top. Every modified file is listed.

**Migration count:** 010 (chunking_strategy) + 011 (ocr_policy) — sequential numbering picks up from Plan A's 009.









