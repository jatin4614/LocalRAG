# RAG Pipeline E2E Testing Guide

Run these in order. Each step verifies one layer of the new pipeline and shows
how it improves over the old one.

---

## 0. Pre-flight: verify the stack is live

```bash
# All 10 containers healthy?
docker ps --format "{{.Names}}\t{{.Status}}" | grep orgchat

# Open WebUI responsive?
curl -sf http://localhost:6100/health

# Our ext routers mounted?
curl -sI http://localhost:6100/api/kb/4/config   # 200 (unauth) means route exists

# Qdrant collections — expect kb_1_v2, kb_3_v2, kb_4_v2, kb_5_v2 + chat_private + kb_eval + open-webui_files
curl -s http://localhost:6333/collections | python3 -c "import json,sys; print('\n'.join(c['name'] for c in json.load(sys.stdin)['result']['collections']))"

# Per-KB rag_config — all 4 KBs should have the best-quality preset
docker exec orgchat-postgres psql -U orgchat -d orgchat \
  -c "SELECT id, name, jsonb_pretty(rag_config) FROM knowledge_bases ORDER BY id;"

# Qdrant has tenant indexes?
curl -s http://localhost:6333/collections/kb_4 | python3 -c "
import json,sys; d=json.load(sys.stdin)['result']
print('sparse:', list((d['config']['params'].get('sparse_vectors') or {}).keys()))
print('payload_schema:', list(d.get('payload_schema',{}).keys()))
"
```

**Expected:** all 10 containers `healthy`, /api/kb/4/config returns `200`,
12 Qdrant collections, all 4 KBs have `rag_config` with `{rerank, context_expand, spotlight, semcache, hyde}`,
payload_schema includes `kb_id`, `chat_id`, `owner_user_id`, `subtag_id`, `doc_id`, `deleted`.

---

## 1. The golden test query — use on Comn KB

Comn KB contains military communications outage logs (brigades, OFC failures, alternate paths, dates).

**Test query:**
> "What alternate communication paths are configured for 109 Inf Bde when Army OFC fails, and which locations experienced outages in March 2026?"

This query is deliberately compound:
- **Keywords** `109 Inf Bde`, `Army OFC`, `March 2026` — BM25 catches exact matches
- **Semantic phrasing** `alternate communication paths`, `configured`, `experienced outages` — dense embedding catches meaning
- **Multi-chunk answer** — requires seeing several outage records
- **Temporal filter** `March 2026` — tests pipeline's handling of dates
- **Aggregation** `which locations experienced` — needs multiple hits

---

## 2. Baseline path — RAG_HYBRID=0 (old dense-only)

Force dense-only for comparison. Open a new chat in Open WebUI, **do NOT select any KB yet**, and set the request-level override by using the API directly:

```bash
# From inside orgchat-net (via docker exec) for direct retrieval probe:
docker exec orgchat-open-webui bash -c '
cd /app && RAG_HYBRID=0 python3 -c "
import asyncio, os
os.environ[\"OPENAI_API_BASE_URL\"] = \"http://vllm-chat:8000/v1\"
from ext.services.vector_store import VectorStore
from ext.services.embedder import TEIEmbedder
from ext.services.retriever import retrieve

async def main():
    vs = VectorStore(url=\"http://qdrant:6333\", vector_size=1024)
    emb = TEIEmbedder(base_url=\"http://tei:80\")
    hits = await retrieve(
        query=\"What alternate communication paths are configured for 109 Inf Bde when Army OFC fails, and which locations experienced outages in March 2026?\",
        selected_kbs=[{\"kb_id\": 4, \"subtag_ids\": []}],
        chat_id=None,
        vector_store=vs, embedder=emb,
        per_kb_limit=10, total_limit=30,
    )
    print(f\"dense-only: {len(hits)} hits\")
    for i,h in enumerate(hits[:5]):
        print(f\"  [{i+1}] score={h.score:.3f}  doc_id={h.payload.get(\\\"doc_id\\\")}  file={h.payload.get(\\\"filename\\\",\\\"\\\")[:30]}\")
        print(f\"       text: {str(h.payload.get(\\\"text\\\",\\\"\\\"))[:120]}...\")
    await vs.close()

asyncio.run(main())
"
'
```

**Record:** top-5 doc_ids, their scores, snippet of text. These are the "before" numbers.

---

## 3. Hybrid path — RAG_HYBRID=1 (new default)

Same query, hybrid on (the runtime default). Same command but without `RAG_HYBRID=0`:

```bash
docker exec orgchat-open-webui bash -c '
cd /app && python3 -c "
import asyncio, os
os.environ[\"OPENAI_API_BASE_URL\"] = \"http://vllm-chat:8000/v1\"
from ext.services.vector_store import VectorStore
from ext.services.embedder import TEIEmbedder
from ext.services.retriever import retrieve

async def main():
    vs = VectorStore(url=\"http://qdrant:6333\", vector_size=1024)
    emb = TEIEmbedder(base_url=\"http://tei:80\")
    hits = await retrieve(
        query=\"What alternate communication paths are configured for 109 Inf Bde when Army OFC fails, and which locations experienced outages in March 2026?\",
        selected_kbs=[{\"kb_id\": 4, \"subtag_ids\": []}],
        chat_id=None,
        vector_store=vs, embedder=emb,
        per_kb_limit=10, total_limit=30,
    )
    print(f\"hybrid: {len(hits)} hits\")
    for i,h in enumerate(hits[:5]):
        print(f\"  [{i+1}] score={h.score:.3f}  doc_id={h.payload.get(\\\"doc_id\\\")}  file={h.payload.get(\\\"filename\\\",\\\"\\\")[:30]}\")
        print(f\"       text: {str(h.payload.get(\\\"text\\\",\\\"\\\"))[:120]}...\")
    await vs.close()

asyncio.run(main())
"
'
```

**Compare:** hybrid should surface chunks containing `109 Inf Bde` and `Army OFC` higher than dense-only did — BM25 gives those keyword hits a boost that RRF fusion preserves.

---

## 4. Full quality stack — end-to-end through chat UI

This is the real test — a user chat with Comn selected. The Comn KB already has
`rag_config = {rerank, context_expand, spotlight, semcache, hyde}` stamped, so
when you select Comn in the chat UI, **all those flags flip on for that request**.

### 4a. Via the Open WebUI UI
1. Open `http://localhost:6100` in browser.
2. Log in with `admin@orgchat.lan` / `OrgChatAdmin2026!`.
3. New chat → KB selector → pick `Comn`.
4. Paste the golden query.
5. Wait for response — expect:
   - Brief pause (~200-500 ms) while HyDE generates a hypothetical answer
   - Brief pause (~50 ms) while context_expand fetches sibling chunks
   - Brief pause (~40-300 ms) while cross-encoder rerank runs on GPU
   - LLM generates answer citing specific dates + brigades + locations

### 4b. Observe progress via SSE (optional)
In a second terminal, before sending the message:
```bash
# Get an auth token (admin session cookie via curl login — skip if already in browser)
# Then:
curl -N -H "Accept: text/event-stream" \
  "http://localhost:6100/api/rag/stream/<chat_id>?q=<URL-encoded query>"
```
You'll see events:
```
event: stage
data: {"stage":"embed","status":"running"}
event: stage
data: {"stage":"retrieve","status":"done","ms":12,"hits":30}
event: stage
data: {"stage":"rerank","status":"done","ms":267,"top_k":20}
event: stage
data: {"stage":"mmr","status":"skipped","reason":"flag_off"}
event: stage
data: {"stage":"expand","status":"done","ms":18,"siblings_fetched":12}
event: stage
data: {"stage":"budget","status":"done","ms":4,"chunks":14}
event: hits
data: {...}
event: done
data: {"total_ms":320}
```

### 4c. Metrics endpoint — watch the pipeline breathe
```bash
# After sending 2-3 queries, scrape /metrics:
curl -s http://localhost:6100/metrics | grep -E "^rag_" | head -40
```
Look for:
- `rag_stage_latency_seconds_bucket{stage="retrieve"}` — histogram of retrieve latency
- `rag_stage_latency_seconds_bucket{stage="rerank"}` — distribution shows cache effect (cold vs warm)
- `rag_retrieval_hits_total{kb="4",path="hybrid"}` — count of hits per KB per path
- `rag_rerank_cache_total{outcome="hit"}` vs `{outcome="miss"}` — cache hit ratio
- `rag_flag_enabled{flag="hybrid"}` should be 1; others 0 at process level (per-KB overrides don't show here)

---

## 5. Quality comparison — what you should observe

| Metric | Dense-only (old) | Full stack (new) |
|---|---|---|
| Top hit relevance | Semantic match but may miss exact `109 Inf Bde` | Exact match boosted by BM25 arm |
| Answer grounding | 3-5 supporting chunks, some off-topic | 10-14 chunks including sibling context, all on-topic |
| Entity recall | Picks up the **general** outage pattern | Recovers **specific** brigade + date + location combos |
| Follow-up "and what about March 12?" | Coldstart retrieval every turn | Semcache hit on similar embedding → ~5 ms |
| Abstract query ("what backup paths work best?") | Low recall — no exact match for "backup paths" | HyDE generates a hypothetical "backup communication paths include DMR, SAM Phone, STARS V..." and embeds that → hits all the right chunks |

---

## 6. Per-KB config tuning

Every KB starts with the same best-quality preset. Adjust per-KB:

```bash
# Get current config
docker exec orgchat-postgres psql -U orgchat -d orgchat \
  -c "SELECT jsonb_pretty(rag_config) FROM knowledge_bases WHERE id = 4;"

# Lower the config for a small/cheap KB (e.g. turn off HyDE + rerank):
curl -X PATCH http://localhost:6100/api/kb/4/config \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your-token>' \
  -d '{"rerank": false, "hyde": false, "context_expand": true}'

# Heaviest preset for a year-long KB (everything on, wider expand window):
curl -X PATCH http://localhost:6100/api/kb/1/config \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your-token>' \
  -d '{"rerank": true, "rerank_top_k": 30, "mmr": true, "mmr_lambda": 0.8, "context_expand": true, "context_expand_window": 3, "spotlight": true, "semcache": true, "hyde": true, "hyde_n": 3, "contextualize_on_ingest": true}'
```

---

## 7. Regression checks

```bash
# Full unit test suite — expect 5 pre-existing auth failures only
cd /home/vogic/LocalRAG
/home/vogic/LocalRAG/.venv/bin/pytest tests/unit/ -q | tail -5

# Eval matrix on kb_eval (takes ~1 min with rerank on GPU, ~5 s without)
cd /home/vogic/LocalRAG
/home/vogic/LocalRAG/.venv/bin/python tests/eval/run_eval_kb_eval.py \
  --label post-merge-smoke --out tests/eval/results/post-merge.json
```

Expected metrics on kb_eval (50 queries, k=10):
- `chunk_recall@10` ≥ 0.96 (hybrid)
- `MRR@10` ≥ 0.92
- `p50_latency_ms` ≤ 15 (dense) or ≤ 300 (full stack with rerank warm cache)

---

## 8. Rollback if anything breaks

Backup is at `~/rag-upgrade-backups/20260420-194343-pre-merge/`. To restore:
```bash
BACKUP=~/rag-upgrade-backups/20260420-194343-pre-merge
cd /home/vogic/LocalRAG
docker compose stop open-webui
gunzip -c "$BACKUP/postgres.sql.gz" | docker exec -i orgchat-postgres psql -U orgchat
git reset --hard ba28dd519bb37d5c133b163a369d897eb31d84ed
docker compose build open-webui && docker compose up -d open-webui
```
