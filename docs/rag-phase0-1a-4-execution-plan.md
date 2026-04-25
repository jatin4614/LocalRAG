# RAG upgrade — Phase 0 + 1a + 4 execution plan

**Date:** 2026-04-21
**Scope:** Phase 0 (measurement), Phase 1a (chunking fixes only, no embedder swap), Phase 4 (observability). **Explicitly excludes** Harrier swap (1b), intent router / doc summary index (2), and default-flip / simplification (3) — those decisions are data-gated.
**Companion:** `docs/rag-phase0-1a-4-rollback.md`
**Supersedes for this cycle:** `docs/rag-upgrade-execution-plan.md` (historical, Apr-19 P0 work already merged)
**Baseline SHA:** `b5fe768` (at backup time)
**Backup location:** `backups/20260421-182222/` (verified, checksums OK)

---

## 1. Current state snapshot

- **Branch:** `main` at `b5fe768`
- **KBs live:** one (`kb_1`, 110 docs, 2590 chunks → avg 23.5 chunks/doc)
- **Chunker:** `ext/services/chunker.py` — O(N) sentence-walker, 800/100 tokens, **cl100k_base**, operates on flat `block.text`.
- **Ingest:** `ext/services/ingest.py` — chunks **per block** independently (line 134-137). No adjacent-block coalescence. `heading_path` is stamped into Qdrant payload (line 287) but **never prepended to the embedded text**.
- **Embedder:** `BAAI/bge-m3` via TEI, 1024-d.
- **Pipeline version:** `chunker=v2|extractor=v2|embedder=bge-m3|ctx=none`.
- **Defaults:** RAG_HYBRID=1, RAG_RERANK=0, RAG_MMR=0, RAG_CONTEXTUALIZE_KBS=0, RAG_RAPTOR=0, RAG_HYDE=0.

---

## 2. Objectives

| Phase | Goal | Quality gate |
|---|---|---|
| **0** | Make every downstream decision evidence-driven. Hand-labeled golden set + unified eval runner + captured baseline against current running system. | Baseline JSON exists. |
| **1a** | Fix chunking: coalesce blocks, prepend heading path at embed time, treat tables atomically. Bump pipeline to `chunker=v3`. Re-ingest `kb_1` once. | +3pp chunk_recall@10 AND no faithfulness regression > 2pp vs baseline. |
| **4** | KB health endpoint + drift gauge + structured query logging + scheduled eval + OOD signal + runbook update. | Health endpoint responds, drift metric scraped by Prometheus. |

Deferred phases (1b, 2, 3) **do not start** until Phase 0 post-1a baseline lands and is compared.

---

## 3. Phase 0 — measurement foundation

### 3.1 Deliverables

| # | File / artifact | Purpose |
|---|---|---|
| 0.1 | `tests/eval/golden_human.jsonl` | 50 hand-labeled queries over kb_1 |
| 0.2 | `tests/eval/run_all.py` (new) | Single-command wrapper: retrieval eval + faithfulness + per-intent breakdown |
| 0.3 | `tests/eval/results/baseline-pre-phase1a.json` | Baseline measurement against current running app |
| 0.4 | `tests/eval/chunk_size_histogram.py` (new) | Scrolls kb_1 payloads, reports chunk-size distribution — confirms/refutes fragmentation hypothesis |
| 0.5 | `tests/eval/query_mix_classifier.py` (new) | Offline pattern classifier over sample prod queries → feeds Phase 2 go/no-go |

### 3.2 Golden set composition (50 rows)

```
intent_label ∈ {specific, global, metadata, multihop, adversarial}
```

- **specific (15):** content-anchored, single-doc questions. ("What did 15 Mar report say about supply chain?")
- **global (10):** cross-doc aggregation. ("Compare March and April risk trends.")
- **metadata (10):** document enumeration. ("List April reports.", "How many reports from Q1?")
- **multihop (10):** multi-doc synthesis. ("Top 3 recurring risks across March.")
- **adversarial (5):** typos, vague, OOD. ("asdf", "tell me something interesting")

Row schema:
```json
{
  "query": "...",
  "intent_label": "specific",
  "expected_doc_ids": [17, 42],
  "expected_chunk_indices": [3, 7],
  "expected_answer_snippet": "supply chain disruption in Southeast Asia",
  "notes": "optional hint for grader"
}
```

### 3.3 Unified runner output

```json
{
  "timestamp": "2026-04-21T...",
  "pipeline_version": "chunker=v2|extractor=v2|embedder=bge-m3|ctx=none",
  "aggregate": {
    "chunk_recall@10": 0.xx,
    "mrr@10": 0.xx,
    "faithfulness": 0.xx,
    "context_precision": 0.xx,
    "context_recall": 0.xx,
    "answer_relevance": 0.xx
  },
  "per_intent": {
    "specific":   {"chunk_recall@10": ..., "faithfulness": ...},
    "global":     {...},
    "metadata":   {...},
    "multihop":   {...},
    "adversarial":{...}
  },
  "p10": {"chunk_recall@10": ..., "faithfulness": ...},
  "latency_ms": {"p50": ..., "p95": ...}
}
```

### 3.4 Constraints

- **Must not modify** `ext/services/*` or `ext/routers/*`.
- Only touches `tests/eval/` and `scripts/` (read-only analysis scripts).
- Baseline run executes against the **currently running** open-webui container (no rebuild).

### 3.5 Exit criteria

- `baseline-pre-phase1a.json` written, aggregate non-empty, per-intent breakdown covers all 5 labels.
- Chunk-size histogram output committed to `tests/eval/results/chunk-size-histogram-pre-1a.json` for later comparison.

---

## 4. Phase 1a — chunking fixes

### 4.1 File-level changes

| File | Change |
|---|---|
| `ext/services/extractor.py` | Add `block_type: Literal["prose","table","code"] = "prose"` field to `ExtractedBlock`. `_blocks_docx` tags table-rows blocks as `"table"`; `_blocks_xlsx` tags sheet blocks as `"table"`. Prose and markdown stay `"prose"`. |
| `ext/services/chunker.py` | New `coalesce_blocks(blocks, *, target_tokens, overlap_tokens)` function that packs adjacent blocks up to `target_tokens`, flushing on heading-path change when buffer ≥ `0.4 × target_tokens` OR buffer ≥ `target_tokens`. Emits tagged `CoalescedChunk(text, heading_path, block_type, page, sheet, source_block_indices)`. Chunks of type `"table"` ≤ `1.5 × target_tokens` bypass coalescence (atomic). |
| `ext/services/ingest.py` | Replace `for b in blocks: for c in chunk_text(b.text, ...)` with `for cc in coalesce_blocks(blocks, ...)`. Pass `cc.text` to `chunk_text` for long-form coalesced buffers; skip `chunk_text` for table blocks (already bounded). At embed time, build `embed_text = f"[{' > '.join(cc.heading_path)}]\n{chunk.text}"` when `heading_path` non-empty; else raw. Store `embed_text` in `vectors = embedder.embed(embed_texts)`. Store raw `chunk.text` in payload unchanged. Add `payload["block_type"] = cc.block_type`. |
| `ext/services/vector_store.py` | Add `"block_type"` to `_PAYLOAD_FIELDS` allowlist (line 127-138). |
| `ext/services/pipeline_version.py` | Bump `CHUNKER_VERSION = "v3"`. |
| `scripts/reingest_all.py` (new) | Idempotent re-ingest runner: `--kb <id>` (or `--all`), `--dry-run` (compute chunk delta only), `--batch-size N`, resumable via pipeline_version filter. Reads `kb_documents` rows, re-extracts from `volumes/uploads/`, re-chunks, re-embeds, upserts. |
| `scripts/delete_orphan_chunks.py` (new) | Post-ingest cleanup: scrolls each KB collection, groups by `doc_id`, identifies points whose `chunk_index >= max_chunk_index_for_that_doc_in_pg` or whose pipeline_version is stale, deletes them. |
| `tests/unit/test_chunker_coalesce.py` (new) | Unit tests for `coalesce_blocks`: short-block packing, heading-boundary flush, table atomicity, long-block fallthrough. |
| `tests/unit/test_ingest_heading_prepend.py` (new) | Unit tests that `ingest_bytes` embeds heading-prepended text while storing raw chunk in payload. |

### 4.2 What Phase 1a does NOT touch

- Embedder (stays bge-m3). No new containers.
- Retrieval code (`retriever.py`, `chat_rag_bridge.py`, `vector_store.py` queries).
- Tokenizer choice (chunker stays cl100k_base). Budget stays gemma-4. Tokenizer consolidation deferred to 1b.
- Reranker / MMR / HyDE / RAPTOR.
- Any flag defaults.

### 4.3 Re-ingest procedure

1. `docker compose build open-webui && docker compose up -d --force-recreate open-webui` — deploy new code.
2. `python scripts/reingest_all.py --kb 1 --dry-run` — prints expected chunk-count delta; no writes.
3. If delta is sane (within 50% of current 2590, not 100× smaller), proceed:
4. `python scripts/reingest_all.py --kb 1 --batch-size 16` — real reingest. Expect ~20-40 min wall clock on 110 docs at TEI batch 32.
5. `python scripts/delete_orphan_chunks.py --kb 1` — removes stale v2 chunks.
6. Verify: `curl http://localhost:8080/api/kb/1/health` (added in Phase 4) reports `drift_pct < 2` and `pipeline_version_distribution == {chunker=v3|...: N}`.

### 4.4 Exit criteria

- All unit tests green (existing + new).
- kb_1 reingested at v3. Drift < 2%. Zero v2 chunks remaining.
- `baseline-post-phase1a.json` captured. Aggregate chunk_recall@10 improves by ≥ +3pp AND faithfulness regression ≤ 2pp.
- If the delta fails the gate → initiate rollback per `rag-phase0-1a-4-rollback.md` §3.

---

## 5. Phase 4 — observability

### 5.1 File-level changes

| File | Change |
|---|---|
| `ext/routers/kb_admin.py` | Add `GET /api/kb/{kb_id}/health` returning `{postgres_doc_count, qdrant_point_count, expected_chunks_from_rows, drift_pct, pipeline_version_distribution, oldest_chunk_uploaded_at, newest_chunk_uploaded_at, failed_docs: [{doc_id, error_message}]}`. Admin-guarded. |
| `ext/services/metrics.py` | Add `rag_kb_drift_pct` Prometheus gauge with `kb_id` label. Emitted by a background task every 60 s. |
| `ext/services/chat_rag_bridge.py` | Add structured log line per retrieval: `logger.info("rag_query req=%s intent=%s kbs=%s hits=%d total_ms=%d", ...)`. Intent is a lightweight pattern-match at `_run_pipeline` entry. No routing yet — logging only. |
| `ext/services/ood_signal.py` (new) | Computes cosine(query_vec, kb_centroid) against cached per-KB centroids. Emits a log WARN if below threshold; no UI change, no blocking. Centroid cache invalidated every 1 hour. |
| `ext/workers/scheduled_eval.py` (new) | Celery beat task running `tests/eval/run_all.py` weekly, writing `tests/eval/results/weekly-YYYY-MM-DD.json`, pushing top-line scores as Prometheus gauges `rag_eval_*`. |
| `ext/workers/celery_app.py` | Register the scheduled-eval beat entry. |
| `RAG.md` §10 | Append rollback procedure for v3 → v2 (reference `rag-phase0-1a-4-rollback.md`). |
| `scripts/reembed_all.py` (new) | Generalization of `scripts/reingest_all.py` — takes any pipeline version, re-embeds without re-extracting. First-class operational tool for future embedder swaps. |

### 5.2 What Phase 4 does NOT touch

- Does not add UI warnings for OOD — log-only for now.
- Does not change retrieval routing based on intent classification — log-only.
- Does not change any flag defaults.

### 5.3 Exit criteria

- `GET /api/kb/1/health` returns JSON with all fields populated.
- `curl http://open-webui:8080/metrics | grep rag_kb_drift_pct` shows metric.
- `docker logs orgchat-celery-worker | grep scheduled_eval` shows beat registration.
- One weekly eval run completes on first boot with a fresh `results/weekly-*.json` written.

---

## 6. Sequencing and gates

```
Wave 1 (DONE): Backup ✅
Wave 2 (DONE): Plan docs (this file + rollback) ✅
Wave 3 (PARALLEL, 3 agents):
  - Agent A: Phase 0 (golden set, runner, baseline-PRE against running app)
  - Agent B: Phase 4 (health, drift, logging, scheduled eval, OOD, runbook)
  - Agent C: Phase 1a code only (NO reingest)
Wave 4 (SEQUENTIAL, supervised):
  - Rebuild + deploy
  - Dry-run reingest (inspect delta)
  - Real reingest (kb_1)
  - Orphan cleanup
  - Capture baseline-POST
  - Compare → gate for 1b/2/3 decisions
```

### Gate between Wave 3 and Wave 4

- All three agents complete without error.
- `pytest` on new + existing unit tests passes no worse than baseline.
- `baseline-pre-phase1a.json` exists (Agent A).
- `docker compose build open-webui` succeeds.

### Gate between Wave 4 staged-reingest and full commit

- Dry-run chunk-count delta is **not** > 50% smaller OR > 500% larger than current 2590.
- If delta is pathological, stop and investigate before the real reingest.

### Gate to declare the cycle complete

- Post-1a baseline shows chunk_recall@10 delta ≥ +3pp.
- Faithfulness delta ≥ -2pp (no meaningful regression).
- Drift endpoint reports < 2%.
- No unexpected errors in `docker logs orgchat-open-webui` for 1 hour after deployment.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Coalescence produces too few chunks** (over-coalesces, loses granularity) | Phase 0 chunk-size histogram captures pre-state. Dry-run reports post-state. Gate on 50% threshold in §4.3 step 3. |
| **Coalescence produces too many chunks** (wider coverage but slower retrieval) | Same dry-run gate. Also: per-KB limit at retrieval is 10 hits pre-rerank, so count growth doesn't directly hurt latency. |
| **Heading-path prefix confuses embedder** | Fail-open: if `heading_path` is empty or garbage, raw text is embedded. Dense-side only; sparse BM25 unaffected (receives only raw query text). |
| **Reingest crashes mid-way** | Migration script is resumable — re-running skips docs already stamped `pipeline_version=v3`. |
| **Orphan chunks pollute retrieval post-reingest** | `delete_orphan_chunks.py` runs as mandatory post-step. Health endpoint's `drift_pct` catches any leftovers. |
| **Rebuild breaks running chat** | Reranker container already persists HF cache; re-recreate preserves volumes. If open-webui fails, restart with prior image tag (see rollback §2). |
| **Eval infrastructure gives noisy signal** | Auto-generated goldens are known-weak (tests/eval/README.md admits this). That's precisely why Phase 0 adds the **hand-labeled** set first. |
| **Celery beat duplicate eval runs** | Beat task uses a Redis lock; second scheduler instance no-ops. |

---

## 8. What this plan explicitly does NOT do

- Does **not** deploy Harrier-0.6B. No embedder swap, no new vLLM container, no VRAM budget renegotiation.
- Does **not** add intent routing at retrieve time. Only logs intent labels.
- Does **not** flip any flag defaults.
- Does **not** delete the bge-m3 TEI container. Stays up.
- Does **not** modify `upstream/` or any upstream Open WebUI file.
- Does **not** force re-ingest on `kb_eval` or `open-webui_files` — only `kb_1`.
- Does **not** touch RBAC, authentication, or network topology.

---

## 9. Rollback authority

Any of these trigger automatic rollback per `rag-phase0-1a-4-rollback.md`:

1. Reingest dry-run shows chunk delta outside ±50% of current 2590.
2. Post-1a chunk_recall@10 < baseline − 2pp.
3. Post-1a faithfulness < baseline − 5pp.
4. `GET /api/kb/1/health` returns `drift_pct > 20` after `delete_orphan_chunks.py`.
5. `docker logs orgchat-open-webui` shows retrieval errors at > 1% rate for 10 minutes.
6. Any integration test regression in `tests/` compared to pre-deploy baseline.

---

## 10. References

- `RAG.md` — current pipeline reference
- `backups/20260421-182222/` — verified rollback source
- `docs/rag-phase0-1a-4-rollback.md` — companion rollback procedures
- `tests/eval/README.md` — eval harness current state + known caveats
