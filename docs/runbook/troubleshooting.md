# Troubleshooting Guide

Each phase fills in its own diagnosis section. Start here when something is wrong.

## Retrieval is slow (p95 > SLO)

1. Check `retrieval_ndcg_daily` gauge — if dropping, quality regression is stealing time (over-retrieval).
2. Check `qdrant_search_latency_seconds` histogram by collection — one slow collection?
3. Check `reranker_batch_latency_seconds` — reranker on cold GPU?
4. Check `llm_latency_seconds{stage="contextualizer"}` — ingest is competing with chat?
5. Last resort: `RAG_CIRCUIT_BREAKER_ENABLED=1` + restart, see if stalled collection fast-fails.

## Retrieval returns empty

1. Check RBAC cache hits: `rbac_cache_hit_ratio` — if 0, cache broken, revert to TTL=0.
2. Check `allowed_kb_ids_total{user_id=...}` — user has any KB access?
3. Qdrant point count: `curl :6333/collections/kb_X` — zero points?
4. Spotlight-wrapping issue: `RAG_SPOTLIGHT=0` temp disable, re-query.

## RBAC denial unexpected

1. `psql -c "select * from kb_access where user_id = '...'"` — row present?
2. Redis cache: `redis-cli -n 3 keys 'rbac:*'` — cached wrong value? `redis-cli -n 3 flushdb` to clear.
3. Pub/sub delivery: `redis-cli PSUBSCRIBE 'rbac:*'` and mutate access — event arrives?

## Reranker returns raw scores (not reranked)

1. Check `reranker_loaded` gauge — model actually loaded?
2. `docker logs orgchat-open-webui 2>&1 | grep -i "reranker"` — startup preload error?
3. GPU 1 OOM? `nvidia-smi` — reranker needs ~2.75 GB.
4. Temporary: `RAG_RERANK=0` to fall back to heuristic.

## Budget tokenizer fell back to cl100k

Phase 1.1 crashes on startup if this happens with an explicit tokenizer. If you see this at runtime:
1. `docker exec orgchat-open-webui ls /models/hf_cache/hub/` — Gemma tokenizer present?
2. `docker exec orgchat-open-webui env | grep HF_HOME` — cache path correct?
3. `docker exec orgchat-open-webui env | grep RAG_BUDGET_TOKENIZER` — flag set as expected?

## Circuit breaker opened for a KB

Symptom: retrieval for one KB suddenly returns empty / 503 while other KBs work, and logs contain
`circuit 'qdrant:kb_X' is open`.

1. Confirm the breaker fired. `docker logs orgchat-open-webui 2>&1 | grep -E "breaker.*qdrant|CircuitOpenError"`
   shows `closed → open` transitions and the failure count.
2. Check Qdrant collection health for the affected KB:
   `curl -s http://qdrant:6333/collections/kb_X | jq '.result.status'`
   — `green` is healthy, `yellow`/`red` means the underlying issue is the collection itself
   (segment corruption, replica lag, etc.). Restore from backup if red.
3. After 30s of quiet, the breaker auto half-opens and probes. To force-close immediately
   (e.g. you've fixed the upstream and don't want to wait): `docker compose restart open-webui`
   — the in-process registry is wiped on restart.
4. Tune cooldown: shorter `RAG_CB_COOLDOWN_SEC` (e.g. `10`) recovers faster when transient
   issues are common; longer (`120`) avoids retry-storms during sustained outages. Default 30s.
5. Kill switch (debug only): `RAG_CIRCUIT_BREAKER_ENABLED=0` falls through to the raw client.
   Never leave this off in production — it removes blast-radius protection.

## RBAC pubsub not delivering

Symptom: admin grants/revokes a KB to a user but the user keeps seeing the old result for up to
the full TTL (default 30s). Pub/sub should invalidate within ~100ms.

1. Subscribe to the channel from inside the network: `redis-cli -n 3 PSUBSCRIBE 'rbac:*'`.
2. Mutate a grant via the admin UI (or `POST /api/kb/{kb_id}/access`). Expect a message of the
   form `rbac:user:{user_id}` within ~100ms.
3. If nothing arrives, check open-webui publish path: `docker logs orgchat-open-webui 2>&1 | grep -i "rbac"`.
   Common causes: `RAG_RBAC_CACHE_REDIS_URL` mismatch between writer and reader, or the cache
   was disabled via `RAG_RBAC_CACHE_TTL_SECS=0` (no cache → no pubsub channel).
4. Safety net: even with broken pubsub, the cache TTL (`RAG_RBAC_CACHE_TTL_SECS`, default 30s)
   guarantees eventual consistency — the user sees the new grant after at most 30s. To accelerate
   while debugging, drop TTL to `5` or `0`.
5. Cross-check the Redis DB: pubsub uses the same DB as the cache (`/3` by default). If you
   point them at different DBs you'll get silent delivery failures.

## LLM metrics all zero

Symptom: Grafana panels for token usage / TTFT / TPOT show no data, or `tokenizer_fallback_total`
never increments despite known fallbacks.

1. Confirm the metrics endpoint is reachable. From inside the network:
   `docker exec orgchat-open-webui curl -s localhost:9464/metrics | grep -E "^rag_(tokens_prompt|tokens_completion|llm_ttft|llm_tpot|tokenizer_fallback)_"`
   should return populated counters/histograms. (Port 9464 is the in-container scrape port; not
   exposed on the host — Prometheus scrapes via the docker network.)
2. If the names are missing entirely, the Prometheus client server didn't start. Check
   `docker logs orgchat-open-webui 2>&1 | grep "prometheus_client metrics server"` — should log
   `listening on :9464` at boot. If absent, prometheus-client failed to import.
3. If the names exist but values are stuck at 0, the call sites aren't wrapped. The Phase 1.6
   wrappers live in `ext/services/contextualizer.py`, `ext/services/hyde.py`, and
   `ext/services/query_rewriter.py` — confirm they import `record_llm_call` from
   `ext/services/llm_telemetry.py` and that those code paths are actually being hit (check
   feature flags: `RAG_HYDE`, `RAG_DISABLE_REWRITE`, `RAG_CONTEXTUALIZE_KBS`).
4. Verify the Prometheus scrape: `curl -s http://prometheus:9090/api/v1/targets | jq '.data.activeTargets[] | select(.labels.job=="open-webui")'`
   — `health` should be `up`.

## Tokenizer preflight crashed at startup

Symptom: `open-webui` container exits with `RuntimeError: tokenizer preflight failed` (or
similar) on boot. Phase 1.1 deliberately crashes here rather than silently fall back to cl100k.

1. Identify the requested alias: `docker logs orgchat-open-webui 2>&1 | grep RAG_BUDGET_TOKENIZER`.
   Default is `gemma-4`.
2. Check the HF cache for the matching tokenizer. Container path is
   `/root/.cache/huggingface` (mounted from `volumes/hf-cache/` on the host). For the default:
   `docker exec orgchat-open-webui ls /root/.cache/huggingface/hub/models--google--gemma-4-31B-it/`
   — must contain a `snapshots/<commit>/tokenizer.json` (or `tokenizer.model`).
3. If missing, either pre-populate the cache (`HF_HUB_OFFLINE=0` + first boot will download,
   assuming the user is logged in to HF for gated repos) or set
   `RAG_BUDGET_TOKENIZER=cl100k` in `compose/.env` and restart — this accepts the drift
   and unblocks startup.
4. Gated-repo gotcha: `google/gemma-*` requires a `huggingface-cli login` token in the cache
   even when offline, because the tokenizer config validates the access. If the cache was
   populated by a different user, copy the token file too.

## Reranker preload failed at startup

Non-fatal — the system falls open to the heuristic ranker, but quality drops noticeably.
Symptom: `reranker_loaded` gauge is `0` and `docker logs orgchat-open-webui 2>&1 | grep -i "reranker"`
shows a load error at boot.

1. Confirm GPU 1 has VRAM. `nvidia-smi` — the bge-reranker-v2-m3 needs ~2.75 GB. TEI
   (~3 GB) and (later) the Qwen3-4B query understanding model also live on GPU 1.
2. Confirm the model is in the cache:
   `docker exec orgchat-open-webui ls /root/.cache/huggingface/hub/ | grep -i bge-reranker`.
   If absent, set `HF_HUB_OFFLINE=0` for one boot to download, then re-enable offline mode.
3. Check `RAG_RERANK_MODEL` — defaults to `BAAI/bge-reranker-v2-m3`. If overridden to a
   different repo, that repo must also be in the cache.
4. Workaround while diagnosing: `RAG_RERANK=0` disables the cross-encoder entirely
   (heuristic-only) and silences the load attempt. Don't leave this off in production —
   recall@5 typically drops by 5-10 points without the reranker.

## GPU alert firing — VRAM > 95%

1. `nvidia-smi` to identify which GPU and which process. Two GPUs in the rig:
   - **GPU 0 (RTX 6000 Ada, 48 GB):** `vllm-chat` (~33 GB at `gpu-memory-utilization=0.70`)
     plus the external `frams-recognition-worker-*` processes (~7.3 GB). Headroom is tight.
   - **GPU 1 (Blackwell):** TEI (~3 GB) + open-webui's in-process reranker (~2.75 GB) +
     (post Plan B Phase 4) Qwen3-4B query understanding (~8 GB).
2. Transient spike on GPU 0: usually a long generation under burst load. Wait — vllm will
   evict KV blocks. If it persists, check `rag_llm_tpot_seconds` for stalls.
3. Permanent capacity issue on GPU 0: lower `--gpu-memory-utilization` on `vllm-chat` in
   `compose/docker-compose.yml` (current default `0.70`). Each 0.05 step is ~2.4 GB freed
   but proportionally fewer concurrent sequences.
4. GPU 1 contention: temporarily set `RAG_RERANK=0` to free ~2.75 GB while you investigate
   TEI memory growth or the Qwen3-4B startup. Reranker fallback is heuristic — quality
   drops but service stays up.
5. Long-term: if GPU 1 is saturated, the next escalation is to move TEI to a sidecar GPU
   or downsize to `BAAI/bge-small-en-v1.5` (1024 → 384 dims, requires re-embed of all KBs
   via `scripts/reembed_all.py`).

## Daily eval cron showing zero recall

Symptom: `daily_eval_cron.sh` runs but `retrieval_ndcg_daily` reports 0 / `recall@5 = 0`.
Almost always a missing/placeholder dataset, not a regression.

1. Inspect the daily subset: `wc -l /home/vogic/LocalRAG-plan-a/tests/eval/golden_daily_subset.jsonl`
   and `head -3 /home/vogic/LocalRAG-plan-a/tests/eval/golden_daily_subset.jsonl`. If the file
   is empty, has 1-2 placeholder rows, or every record's `relevant_doc_ids` is empty,
   the cron has nothing to score against.
2. Check whether the human-labeled golden exists: `wc -l tests/eval/golden_human.jsonl`.
   The Task 0.3 follow-up calls for the operator to hand-label this; if it's still a
   placeholder, that's the root cause. (Note: `golden_starter.jsonl` from the plan was
   renamed/folded into `golden_human.jsonl` in this repo.)
3. Confirm the cron is scoring the right collection by reading the latest JSON in
   `tests/eval/results/` — `kb_id` and `total_queries` should match expectations.
4. Until the operator delivers a real labeled subset, treat the alert as "data
   pending" rather than a quality regression. Document the date the labels are due.

## Schema reconciliation: how to migrate kb_*

Use this when a KB collection's payload schema is drifting (added a new field, changed
a tokenizer, switched embedder dimensions). The reconcile script in Phase 1.7 builds a
shadow `_v2` collection, alias-swaps it in, and keeps the source for rollback.

1. Run the migration in shadow mode (writes to `kb_X_v2`, leaves `kb_X` serving):
   `cd /home/vogic/LocalRAG-plan-a && python scripts/reconcile_qdrant_schema.py --collection kb_X --target kb_X_v2`
2. Verify counts after the script reports done:
   `curl -s http://qdrant:6333/collections/kb_X | jq '.result.points_count'` and
   `curl -s http://qdrant:6333/collections/kb_X_v2 | jq '.result.points_count'`. They
   should match within ~1 (allow for in-flight writes).
3. Spot-check payload shape on a sample point from each collection:
   `curl -s -X POST http://qdrant:6333/collections/kb_X_v2/points/scroll -H 'Content-Type: application/json' -d '{"limit":1,"with_payload":true}' | jq '.result.points[0].payload | keys'`.
4. Alias swap: point the read alias at `kb_X_v2`. The script supports `--commit` to do this
   atomically — re-run with `--commit` to flip the alias once verified.
5. Keep `kb_X` (the source) in place for **14 days** as a rollback target. After the
   bake period, drop it: `curl -X DELETE http://qdrant:6333/collections/kb_X`.
6. If anything goes wrong post-swap, alias-swap back to the source and investigate before
   re-running the reconciler.

## Retrieval returns weirdly similar chunks (near-duplicates)

1. Intent classified as `specific`? MMR is disabled for specific by design.
2. If the query is actually about comparison ("compare X vs Y"), it should classify as `global`. Fix: the regex in `classify_intent` may miss the pattern. Check `ext/services/chat_rag_bridge.py:41-64` for pattern gaps.
3. Override per-KB via `rag_config.RAG_MMR = "1"` to force MMR on for that KB only.

## Retrieval loses local context (chunk mentions "as discussed above")

1. Intent classified as `global`? Context-expand is disabled for global.
2. Per-KB override: `rag_config.RAG_CONTEXT_EXPAND = "1"` forces on.
3. Check `RAG_CONTEXT_EXPAND_WINDOW` — default N=1, raise to 2 for long-narrative KBs.

## One KB is dominating results (chatty-KB problem)

1. This was mitigated in Phase 2.3 via RRF fallback — but only when cross-encoder rerank is OFF. If you see this with rerank ON, the cross-encoder itself is biased toward one collection's content.
2. Temporary: `RAG_RERANK=0` — retrieval falls back to RRF.
3. Long-term: confirm both collections have the canonical schema (Phase 1.7 migration).

## Contextualized chunks look wrong (prefix is irrelevant or hallucinated)

1. Reproduce the prompt the model actually sees:
   `python -c "from ext.services.contextualizer import build_contextualize_prompt; print(build_contextualize_prompt(document_text='X', chunk_text='Y', document_metadata={'filename':'a.md','kb_name':'K','subtag_name':None,'document_date':None,'related_doc_titles':[]}))"`
   Optional metadata fields collapse to empty strings when missing — if you see literal `None` / `[]` in the body, the metadata dict isn't matching the contract (see `build_contextualize_prompt` docstring).
2. Inspect `rag_tokens_prompt_total` for the contextualizer model — is the prompt_tokens per chunk ~stable (vllm-chat prefix cache is reusing the doc-level header) or growing across sibling chunks (cache broken — request body shape changed)? Note: the WIP counter is labelled `(model, kb)` only, NOT by stage. Filter by the model that the contextualizer is calling (env `CHAT_MODEL`, default `orgchat-chat`); other stages (HyDE, rewriter) share that label, so isolate by tailing the timestamp window during a known re-ingest.
3. For a specific chunk, fetch its payload from Qdrant and confirm `context_prefix` is set:
   `curl -s http://localhost:6333/collections/kb_1/points/<point_id> | python -m json.tool | grep context_prefix`
4. If the prefix references the wrong date / wrong KB / wrong related docs: the `document_metadata` threading is broken upstream of the contextualizer. Check `ext/services/ingest.py` at the point it constructs the metadata dict that feeds `contextualize_chunks_with_prefix` — a missing or stale field there propagates to every chunk in that document.

## ColBERT search returns worse results than dense

1. Per-KB `rag_config.colbert` = true but the collection lacks the `colbert` named-vector slot → the retriever silently falls back to 2-head RRF (dense + sparse). Inspect the collection's vector schema:
   `curl -s http://localhost:6333/collections/kb_1 | grep -E 'colbert|dense'`
   If only `dense` (and optionally `sparse`) is present, the ColBERT write path never landed — re-ingest the KB after Phase 3.4 to populate the slot.
2. Evaluate the ColBERT head in isolation. On a dev instance: `RAG_COLBERT=1` and `RAG_HYBRID=0`, then run a few representative queries. If ColBERT alone is materially worse than dense alone, the model cache is likely the wrong checkpoint — confirm the late-interaction model in `compose/.env` matches what was used at ingest time. Mismatched checkpoints (e.g. ingested with one tokenizer, querying with another) silently produce noise.

## Re-ingest stuck — throttle is permanent

Symptom: the `scripts/reingest_kb.py` driver pauses repeatedly because chat p95 keeps tripping the throttle ceiling, and the run never completes within its scheduled window.

1. Chat p95 legitimately > 3000ms all the time? Then base load is too high to absorb the contextualization cost. Three options, in order of preference:
   (a) Schedule the re-ingest during off-peak hours (defer the run; this is by far the cleanest outcome — see `docs/runbook/reingest-procedure.md` for the off-hours runbook).
   (b) Raise the throttle ceiling, accepting that user-facing chat will degrade for the duration. Document the degraded window for the on-call.
   (c) Run contextualization in the background outside the re-ingest path — generate prefixes asynchronously, persist them, then have the re-ingest step skip the LLM call and read the cached prefix. Higher implementation cost but isolates re-ingest cost from chat latency.
