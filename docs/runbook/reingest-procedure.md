# KB Re-ingest Procedure (Phase 3 contextualization + ColBERT)

**Use when:** enabling contextualization and/or ColBERT multi-vector on a KB that already has ingested documents. The existing embeddings do not carry the contextual prefix and the existing points do not have the `colbert` named vector slot.

**Do NOT use for:** ingesting brand-new docs (that happens automatically via the upload endpoint).

## Preconditions

- [ ] Phase 1.7 schema reconciliation complete for the target KB's collection.
- [ ] Phase 3.1–3.5 code merged and deployed.
- [ ] Appendix A staging complete: `/var/models/fastembed_cache/colbert-ir--colbertv2.0/` exists.
- [ ] Operator has admin token; KB id and current collection name known.
- [ ] Eval baseline from Phase 0 is current and committed.
- [ ] Off-peak window scheduled (ideally ≥ 2 hours with low chat traffic).

## Strategy: dual-collection with alias cutover

We never write contextualized/ColBERT-enriched points on top of existing points. Instead:

1. Create a fresh collection `kb_{id}_v3` with canonical schema + `colbert` slot.
2. Stream all documents from the source KB back through the ingest pipeline (which now reads contextualize + colbert flags and writes both).
3. Verify point counts match.
4. Run eval on the new collection — must be within +5 pp global of baseline.
5. Swap the Qdrant alias from the old collection to the new.
6. Keep the old collection read-only for 14 days as rollback target.

Rationale: writing in-place during re-ingest creates a window where some points have prefixes and some don't — retrieval during that window is inconsistent. Dual-collection avoids the window entirely.

## Throttle policy during re-ingest

Contextualization calls vllm-chat on GPU 0 (at 89% VRAM steady-state). Uncontrolled, it will spike chat p95 above 3 s. Policy:

- Celery worker reads `RAG_INGEST_CHAT_P95_CEILING_MS` (default 3000).
- Before each batch, worker queries Prometheus for 5-min chat p95.
- If > ceiling, worker sleeps 30 s and re-checks.
- Alert `ChatLatencyDuringIngest` (Phase 1.9) fires if this throttle isn't effective.

## Step-by-step

Everything below assumes running commands as the operator on the deployment host.

### 1. Snapshot the source

```bash
SOURCE=kb_1_rebuild    # or current canonical name after 1.7 migration
curl -s -X POST "http://localhost:6333/collections/$SOURCE/snapshots" \
  | python -m json.tool
# Note the snapshot name; this is your rollback target.
```

### 2. Create the target collection

Run a one-shot init command (you can add this as a Makefile target later):

```bash
TARGET=kb_1_v3
python - <<PY
import asyncio
from ext.services.vector_store import VectorStore
vs = VectorStore(url="http://localhost:6333", vector_size=1024)
asyncio.run(vs.ensure_collection("$TARGET", with_sparse=True, with_colbert=True))
print("created $TARGET with dense + sparse + colbert")
PY
```

### 3. Enable per-KB contextualize in `rag_config`

```bash
KB_ID=1
curl -s -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"contextualize": true, "colbert": true}'
```

### 4. Set environment for this ingest session

```bash
export RAG_COLBERT=1
export RAG_SYNC_INGEST=1   # keep sync for visibility; async via Celery is Plan B
export RAG_INGEST_CHAT_P95_CEILING_MS=3000
```

### 5. Stream source docs through the pipeline

```bash
# Use the supplied re-ingest script (create in next step if not present)
python scripts/reingest_kb.py \
  --source-collection $SOURCE \
  --target-collection $TARGET \
  --kb-id $KB_ID \
  --api-base-url http://localhost:6100 \
  --admin-token "$RAG_ADMIN_TOKEN" \
  --throttle-ceiling-ms 3000
```

Expected output: progress log every 50 docs; total time ~15–60 min depending on chat contention.

### 6. Verify counts

```bash
src_count=$(curl -s http://localhost:6333/collections/$SOURCE | python -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])")
tgt_count=$(curl -s http://localhost:6333/collections/$TARGET | python -c "import sys,json;print(json.load(sys.stdin)['result']['points_count'])")
echo "source=$src_count target=$tgt_count"
test "$src_count" = "$tgt_count"
```

### 7. Run eval against the new collection

Create a temporary KB row in Postgres pointing at the new collection, then:

```bash
make eval KB_EVAL_ID=$NEW_KB_ID
```

Compare against `tests/eval/results/phase-0-baseline.json`. Gate: chunk_recall@10 ≥ +5 pp global, no per-intent regression > 2 pp.

### 8. Swap the alias

Only after eval gate passes:

```bash
curl -X PUT http://localhost:6333/collections/aliases -d '{
  "actions": [
    {"delete_alias": {"alias_name": "kb_1"}},
    {"create_alias": {"collection_name": "'$TARGET'", "alias_name": "kb_1"}}
  ]
}'
```

### 9. Confirm cutover

```bash
# Live retrieval now hits the new collection
curl -s -X POST http://localhost:6100/api/rag/retrieve \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -d '{"query":"OFC roadmap","selected_kb_config":[{"kb_id":1}],"top_k":3}' \
  | python -m json.tool
```

Spot-check that hits look right (prefix present in payload, ColBERT used if `RAG_COLBERT=1`).

### 10. Hold rollback window

- Mark source collection read-only at app level — do not delete for 14 days.
- Monitor the `retrieval_ndcg_daily` alert for 14 days.
- If regression detected: `curl -X PUT .../aliases` to swap back.

## Rollback

```bash
# Swap alias back to the original collection
curl -X PUT http://localhost:6333/collections/aliases -d '{
  "actions": [
    {"delete_alias": {"alias_name": "kb_1"}},
    {"create_alias": {"collection_name": "'$SOURCE'", "alias_name": "kb_1"}}
  ]
}'
# Revert rag_config
curl -s -X PATCH "http://localhost:6100/api/kb/$KB_ID/config" \
  -H "Authorization: Bearer $RAG_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"contextualize": false, "colbert": false}'
```

Total revert time: <1 minute.

## Known failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| Target point count < source | Ingest retry exhausted on some docs | Grep logs for `ingest failed`, fix root cause (usually transient TEI), re-run for remaining docs (idempotent via UUID5 point IDs) |
| Chat p95 stuck > 3s during ingest | Throttle policy not honoring ceiling | Kill the reingest script, reduce concurrency, restart |
| ColBERT embeddings unreasonably slow | Model not in fastembed cache | Check `/var/models/fastembed_cache/`; re-stage per Appendix A |
| Eval worse, not better | Contextualizer prompt produces noisy prefixes | Review `llm_tokens_total{stage="contextualizer"}` — is prompt caching working? Check prompt template in `contextualizer.py:build_contextualize_prompt` |
