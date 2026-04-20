# MMR Diversification (P1.3)

Maximal Marginal Relevance (MMR) post-processes the top-k reranker output
to maximise both relevance AND diversity. It penalises near-duplicate
chunks so the k passages presented to the LLM span multiple concepts or
documents rather than N copies of the same passage.

This feature is **off by default** — controlled by the `RAG_MMR`
environment variable. When the flag is unset or `0`, the `ext.services.mmr`
module is never imported and behaviour is byte-identical to the pre-P1.3
release.

## Why MMR

A user asks: "What did Jatin say about deployment?" Retrieval surfaces
ten chunks — but the top eight are near-identical paragraphs from the
same meeting-notes doc, and two are from a different doc with a materially
different angle. Pure-relevance reranking keeps the eight duplicates at
the top; MMR demotes most of them, lifting the two diverse chunks into
the budget so the LLM actually sees multiple perspectives.

In the textbook form:

```
S = []
while |S| < k:
    argmax_{d in C - S} [ lambda * sim(q, d) - (1 - lambda) * max_{s in S} sim(d, s) ]
    add to S
return S
```

* `lambda = 1.0` — pure relevance (MMR is a no-op against its input).
* `lambda = 0.0` — pure diversity: first pick is the most relevant, then
  each subsequent pick is the most-dissimilar-from-selected.
* `lambda = 0.7` — the default; 70% relevance / 30% diversity.

## Enabling MMR

1. **No new dependencies.** MMR is pure stdlib (`math` + list ops).
2. **Set the flag.**
   ```bash
   export RAG_MMR=1
   # optional: override the lambda
   export RAG_MMR_LAMBDA=0.7
   ```
3. **Restart the service.** The flag is read at call time, so a hot toggle
   within a running process works, but production flips should be a
   restart.

## How it interacts with the rest of the pipeline

MMR runs *after* the reranker and *before* budget truncation. When MMR
is on the bridge widens the reranker's candidate pool so MMR has actual
surplus to diversify over — otherwise `rerank(top=k) -> MMR(top=k)` is a
pass-through, which is what the pre-P2 pipeline accidentally did.

```
retrieve (top-30)
  -> rerank_with_flag (legacy OR cross-encoder, top-rerank_k)
       rerank_k = max(2 * final_k, 20) when RAG_MMR=1 (default: 20)
       rerank_k = final_k             when RAG_MMR=0 (default: 10)
       rerank_k = RAG_RERANK_TOP_K    when the env var is set (clamped to >= final_k)
  -> mmr_rerank_from_hits (flag-gated; trims rerank_k -> final_k with diversity)
  -> budget_chunks (token-fit the final list)
```

### Why the widening was needed

Before P2 the bridge called `rerank_with_flag(..., top_k=10)` followed by
`mmr_rerank_from_hits(..., top_k=len(reranked))`. Since the reranker had
already narrowed to ten candidates, MMR's "pick the best k from C" was
just "reorder the ten already selected". That is why the eval matrix
showed identical metrics with `RAG_MMR=0` vs `RAG_MMR=1`. With the P2
widening, MMR actually has a surplus of ten extra candidates (20 -> 10)
to trade relevance for diversity.

### Operator override (`RAG_RERANK_TOP_K`)

If reranker cost is a concern — for instance on a cold cross-encoder
cache or during an outage where reranking is on the hot path — pin the
candidate pool:

```bash
# Cheap mode: only score 12 pairs, MMR still trims to 10.
export RAG_MMR=1
export RAG_RERANK_TOP_K=12

# Expensive / high-recall mode: score 30, MMR picks a diverse 10.
export RAG_MMR=1
export RAG_RERANK_TOP_K=30
```

`RAG_RERANK_TOP_K` is clamped *up* to `final_k` (pointless to rerank
fewer candidates than the final budget). When MMR is off and the
override yields more than `final_k` candidates, the bridge trims the
tail back to `final_k` before the budget stage so downstream sees the
same count as the pre-P2 pipeline.

### Cost of widening

The reranker scores `rerank_k` query/passage pairs instead of `final_k`.
For the cross-encoder path that is at most 2x more pairs on the default
(`final_k=10` -> `rerank_k=20`); with a GPU and the P2 rerank cache this
is typically single-digit milliseconds. MMR itself still embeds only the
top `rerank_k` passages (one batched TEI call) — same pattern as before,
just 20 passages instead of 10.

## Cost

MMR re-embeds the top-k passages plus the query in a **single batched
TEI call** (`embed(list[str])`). TEIEmbedder's existing `embed` signature
already accepts a list, so this is one additional HTTP round-trip per
query. On a warm TEI server with BAAI/bge-m3 this is ~50-200 ms for
top-30 passages. The greedy MMR loop itself runs in O(k^2 * d) Python
without numpy and completes in <1 ms for k=10, d=1024.

Why re-embed rather than thread vectors through the retriever? The
retriever's `Hit` does not carry the dense vector today, and making
`vector_store.search` optionally return vectors is more invasive than a
single extra embed call. The tradeoff is documented in
`ext/services/mmr.py`.

## Expected quality gain

MMR is most valuable when the retriever surfaces **many near-duplicate
chunks**. For organisations with:

* Repetitive docs (boilerplate in SOPs, copied-then-edited meeting notes).
* Same source material split across subtags (e.g., the same retrospective
  note pasted into three different KBs).
* Many small chunks from one long doc (chunker-induced duplicates).

expect moderate recall@k gains (diverse chunks surface sooner) and
stronger user-visible answer diversity (the LLM cites distinct sources,
not copies of one). For KBs without duplication, MMR is near-identity and
the flag-on cost is just the extra TEI call.

Gate the default flip on pre/post eval:

```bash
RAG_MMR=0 python tests/eval/run_eval.py --out baseline.json
RAG_MMR=1 python tests/eval/run_eval.py --out mmr.json
```

## Fail-open design

`chat_rag_bridge` wraps the MMR call in a bare `except Exception`. If
the embedder is unavailable, TEI returns an error, or the MMR reorder
raises for any reason, retrieval silently falls back to the reranker's
output. This keeps retrieval working in degraded environments at the
cost of silently skipping diversification. To surface failures during
debugging, run with `PYTHONWARNINGS=error` or add logging at the except
block in `ext/services/chat_rag_bridge.py`.

## Flags

| Flag                | Default                 | Meaning                                              |
|---------------------|-------------------------|------------------------------------------------------|
| `RAG_MMR`           | `0`                     | `1` to enable; anything else is off                  |
| `RAG_MMR_LAMBDA`    | `0.7`                   | Float in `[0, 1]`; higher = more relevance, less diversity |
| `RAG_RERANK_TOP_K`  | `max(2*final_k, 20)` when `RAG_MMR=1`, else `final_k` | Integer; how many candidates the reranker emits before MMR trims. Clamped to `>= final_k`. |

## Troubleshooting

### Results identical to legacy path

Check the flag is actually engaged. Only the exact string `"1"` enables
MMR (not `"true"`, not `"yes"`). If the flag is set but results look
unchanged, three common causes:

1. **`RAG_RERANK_TOP_K` pinned to `final_k`.** The operator override
   collapses the candidate pool back to the final budget, which
   degenerates MMR into a reorder-in-place. Unset the variable or raise
   it above `final_k`.
2. **The reranker returned fewer hits than `rerank_k`.** If the
   retriever surfaced `<= final_k` candidates, MMR short-circuits to a
   passthrough (nothing to subset). Raise `per_kb_limit`/`total_limit`
   in retrieval.
3. **Retrieval returned no near-duplicates.** MMR only reorders when
   pairwise passage similarity is high enough to trigger the diversity
   penalty. A healthy KB with diverse documents will see minimal
   reordering — this is the desired behaviour, not a bug.

### Very high lambda still lets duplicates through

At `RAG_MMR_LAMBDA >= 0.9` the relevance term dominates and duplicates
often win. Lower the value to 0.5-0.7 for meaningful diversification.
