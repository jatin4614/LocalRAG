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
