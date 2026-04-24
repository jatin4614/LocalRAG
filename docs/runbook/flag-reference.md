# RAG Flag Reference

All env flags read by the retrieval pipeline. Filled in by each phase of Plan A/B.

| Flag | Default | Owner phase | Runtime-safe toggle? | What it does | When to disable |
|---|---|---|---|---|---|
| `RAG_HYBRID` | `1` | Pre-Plan A | Yes, via per-KB `rag_config` override | Enables dense + sparse RRF fusion. bge-m3 dense + fastembed BM25. | Only if sparse vectors are suspected to be corrupting results. |
| `RAG_RERANK` | `0` | Pre-Plan A | Yes | Enables cross-encoder rerank (BAAI/bge-reranker-v2-m3). ~60-120 ms on GPU 1 batched. | If reranker model fails to load or GPU 1 is OOM. |
| `RAG_MMR` | `0` → `intent-conditional` after Phase 2.2 | Phase 2.2 | Yes | Maximal Marginal Relevance diversification. λ=0.7. | If eval shows it hurts specific-intent recall. |
| `RAG_CONTEXT_EXPAND` | `0` → `intent-conditional` after Phase 2.2 | Phase 2.2 | Yes | Fetches ±N sibling chunks for each top hit. Default N=1. | If token budget overruns. |
| `RAG_SPOTLIGHT` | `0` → `1` after Phase 2.1 | Phase 2.1 | Yes | Wraps retrieved content in `<UNTRUSTED_RETRIEVED_CONTENT>` tags, adds system-prompt rule. | Never in multi-tenant production. |
| `RAG_SEMCACHE` | `0` | Deferred | Yes | Redis-backed semantic cache keyed on quantized query vec. | Known stale-result risk. |
| `RAG_HYDE` | `0` | Deferred | Yes | Hypothetical Document Embeddings. One chat call per query. | Latency-sensitive workloads. |
| `RAG_RAPTOR` | `0` | Deferred (Plan B Phase 5 replaces) | No (ingest-time) | Builds hierarchical tree at ingest. Flat tree collapses temporal signal. | Plan B replaces with temporal-semantic variant. |
| `RAG_CONTEXTUALIZE_KBS` | `0` → per-KB opt-in after Phase 3.3 | Phase 3.3 | No (ingest-time) | Anthropic Contextual Retrieval — LLM prepends context per chunk. | High-ingest-volume KBs where cost matters. |
| `RAG_INTENT_ROUTING` | `0` | Deferred (Plan B Phase 4 replaces) | Yes | Tier 2 regex-based intent routing. | Plan B replaces with Query Understanding LLM. |
| `RAG_DISABLE_REWRITE` | `1` (rewrite OFF) | Deferred (Plan B Phase 4 replaces) | Yes | Inverted flag — `1` disables the LLM query rewriter. | Plan B replaces with Query Understanding LLM. |
| `RAG_SYNC_INGEST` | `1` | Plan B Phase 6 | **No** (restart required) | Inline vs Celery async ingest. | Plan B flips to 0 after soak test. |
| `RAG_BUDGET_TOKENIZER` | `gemma-4` | Phase 1.1 | **No** (startup validated) | Tokenizer family for budget counting. Phase 1.1 adds a preflight that crashes on fallback. | Set to `cl100k` to accept drift. |
| `RAG_BUDGET_TOKENIZER_MODEL` | (unset) | Phase 1.1 | **No** | Override specific HF model id. | — |
| `RAG_RBAC_CACHE_TTL_SECS` | `30` | Phase 1.5 | Yes | TTL on Redis-backed allowed_kb_ids cache. `0` disables the cache (DB lookup every request). | Debugging permission lag. |
| `RAG_CIRCUIT_BREAKER_ENABLED` | `1` | Phase 1.3 | Yes | Per-KB circuit breaker. `0` falls through to raw client. | Never in production; enable for debug only. |
| `RAG_TENACITY_RETRY` | `1` | Phase 1.4 | Yes | Exponential backoff retry on TEI / reranker / HyDE. `0` reverts to single-shot fail-open. | Debug retry-storm only. |
| `RAG_COLBERT` | `0` → `1` after Phase 3.5 | Phase 3.5 | Yes | Enables ColBERT third RRF head. | If ColBERT model fails to load. |

## Kill-list status

Flags remaining default-OFF globally at end of Plan A (subject to Plan B Phase 4 audit):
- `RAG_SEMCACHE`, `RAG_HYDE`, `RAG_RAPTOR`, `RAG_INTENT_ROUTING`, `RAG_DISABLE_REWRITE` (→ replaced by Query Understanding LLM).
