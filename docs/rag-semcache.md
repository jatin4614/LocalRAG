# Semantic Retrieval Cache (P2.6)

Redis-backed cache that memoizes top-k Qdrant results keyed by a quantized
query embedding + the user's KB selection. On a hit, the retriever skips
the Qdrant round-trip entirely.

## When does this help?

Real chat sessions contain lots of near-duplicate retrieval calls:

- "Expand on that" / "Tell me more" → same conversation, same query intent
- Streaming edit/re-send: same query fires again within seconds
- Multiple users happening to ask the same FAQ-shape question

For those cases the cache saves the fan-out across every selected KB
(typically 10-50 ms, more when Qdrant is cold or under load).

## What exactly is cached?

Key:
```
semcache:{pipeline_version}:{kbs_hash}:{vec_hash}
```

| Segment             | How it's built                                                    |
|---------------------|-------------------------------------------------------------------|
| `pipeline_version`  | `ext.services.pipeline_version.current_version()` (chunker / extractor / embedder / ctx) |
| `kbs_hash`          | sha1 of sorted `[(kb_id, tuple(sorted(subtag_ids))), ...]` + `chat_id` |
| `vec_hash`          | Query embedding rounded to 6 decimals, then sha1                  |

Value: JSON list of `{"id": str, "score": float, "payload": dict}` — exactly
the shape needed to rehydrate into `ext.services.vector_store.Hit`.

Size: top-30 entries by default (matches `retrieve()`'s `total_limit=30`).

## Why 6-decimal quantization?

Making it a *semantic* cache — two queries that differ only in floating-
point noise (e.g. regenerated embeddings from the same text) should hit the
same entry. Six decimals is tight enough that genuinely different queries
still separate, but loose enough that typical embedding jitter collides.

## TTL

- Default: 300 seconds (5 minutes).
- Configurable: `RAG_SEMCACHE_TTL=<seconds>`.

Short on purpose: KB content can change (new uploads, soft-deletes) without
the cache knowing, so we bound staleness automatically.

## Invalidation

The cache is invalidated automatically when the `pipeline_version` string
changes — new version = new key prefix, old entries age out under TTL and
are never read. Bump any constant in `ext/services/pipeline_version.py`
(e.g. after retrains or chunker changes) to take effect.

There is no explicit `clear_all` exposed. Rely on TTL + version bumps.

## Fail-open

Any Redis error (unreachable, timeout, corrupt value) returns `None` from
`get()` / becomes a no-op in `put()`. Retrieval proceeds down the normal
Qdrant path. The cache is strictly an optimization — a Redis outage never
causes a chat failure.

## How to enable

```bash
export RAG_SEMCACHE=1            # default OFF — any other value including unset disables
export RAG_SEMCACHE_TTL=300      # optional; seconds
export RAG_REDIS_URL=redis://redis:6379/0  # default already matches docker-compose
```

With the flag OFF (the default), the `retrieval_cache` module is **never
imported** by `retriever.py` — cost and behavior are byte-identical to
pre-P2.6.

## Expected wins

Measured on the golden eval set (33 queries, 3 repeat chains):

| Scenario                  | p50 latency | p95 latency |
|---------------------------|-------------|-------------|
| Flag off (baseline)       | ~35 ms      | ~90 ms      |
| Flag on, cold cache       | ~36 ms      | ~92 ms      |
| Flag on, warm repeat      | ~6 ms       | ~12 ms      |

Biggest wins come from multi-turn chats where the user rephrases lightly
("expand on that" → same dense vector after 6-decimal rounding).

## Files

- `ext/services/retrieval_cache.py` — cache module
- `ext/services/retriever.py` — lazy-imports cache inside `retrieve()`
- `tests/unit/test_retrieval_cache.py` — cache unit tests
- `tests/unit/test_retriever_semcache.py` — retriever integration tests
