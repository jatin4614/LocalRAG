# RAG Pipeline SLO Document

**Owner:** RAG team
**Last updated:** 2026-04-24 (Plan A Phase 0.1)
**Review cadence:** monthly, or after any phase ships

## Retrieval latency budget

Measured end-to-end from `retrieve_kb_sources()` entry to return, excluding LLM generation.

| Phase state | p50 | p95 | p99 | Breach action |
|---|---|---|---|---|
| Current (pre-Plan A) | 320 ms | 850 ms | 1400 ms | — baseline |
| After Plan A Phase 2 | 450 ms | 1100 ms | 1800 ms | block Phase 3 if breached |
| After Plan A Phase 3 | 550 ms | 1300 ms | 2000 ms | block Plan B if breached |
| After Plan B Phase 4 | 700 ms | 1500 ms | 2400 ms | block Plan B Phase 5 if breached |
| Hard ceiling — never exceed | — | — | **3000 ms** | any phase breaching this must rollback |

## Cost budget

Cost is measured in vllm-chat GPU-seconds (proxy for $ since inference is self-hosted).

| Call site | Per-request budget | Plan A baseline | Plan A target |
|---|---|---|---|
| Query rewrite (LLM) | 500 tokens in, 200 tokens out | OFF (default) | OFF (remains gated) |
| Contextualizer (LLM, per chunk at ingest) | 800 in, 50 out, prompt-cached | OFF | per-KB opt-in only |
| HyDE generation (LLM) | 200 in, 250 out | OFF | OFF (remains gated) |
| Reranker (local GPU) | 15 pairs × 50ms | OFF | ON (Phase 2 cheap wins, Task 2.2 doesn't enable globally; only intent-conditional) |

Ingest one-time budget: re-ingesting kb_1_rebuild (2,698 chunks) with contextualization enabled is roughly 2,698 × 850 tokens × 1 chat call. At vllm-chat's measured throughput (≈ 3000 tokens/s for Gemma-4 AWQ on RTX 6000 Ada), that's ≈ 13 minutes of pure LLM time if undisturbed. With prompt caching (document-level prefix shared across chunks of the same doc), 3–5 minutes. With throttle policy (pause if chat p95 > 3s), assume 2–4× that = 12–20 minutes on a lightly-loaded system, 60+ minutes under heavy load.

## Error rate ceiling

Measured as `(5xx + timeout + explicit-failure) / total_retrieval_requests` over 5-min windows.

| Metric | Ceiling | Alert threshold |
|---|---|---|
| Retrieval 5xx/timeout | 0.5% | 0.2% (5-min) |
| Reranker failures | 1% | 0.5% |
| Embedder (TEI) failures | 0.1% | 0.05% |
| RBAC cache miss rate | n/a (performance only) | > 50% means cache broken |
| LLM (contextualizer / rewriter / HyDE) failures | 2% (fail-open is OK) | 5% |

## Quality floor per intent

Measured via Phase 0.7 eval harness against the golden starter set.

| Intent | chunk_recall@10 floor | MRR@10 floor | nDCG@10 floor |
|---|---|---|---|
| specific | 0.80 | 0.70 | 0.65 |
| global | 0.75 | — (doc-level) | 0.60 |
| metadata | 0.70 | 0.65 | 0.55 |
| multihop | 0.60 | 0.50 | 0.45 |
| adversarial | n/a (pass/fail — injection blocked, cross-user denied, empty-retrieval returns empty) |
| non-English (tag) | within 5pp of same-intent English baseline |

**Breach action:** any phase dropping an intent below its floor rolls back before merge.

## Post-plan projection

Plan A end (after Phase 3 ships) should hit:
- p95 retrieval: 1300 ms
- Global `chunk_recall@10`: +5 pp vs baseline (Phase 2 + Phase 3 combined)
- Error rates unchanged

Plan B end (after Phase 5 ships): the same or better on all intents, with `evolution`-tagged queries showing measurable improvement (stratum added in Plan B).
