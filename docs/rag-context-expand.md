# Parent-Document Context Expansion (P1.4)

Parent-document (a.k.a. "context expansion") retrieval **retrieves at fine
granularity and presents at coarse granularity**. The retriever still
searches over the chunker's 800-token chunks for maximum recall, but at
response time each top hit is expanded with its immediate sibling chunks
from the same document. The LLM sees coherent prose — paragraphs flowing
together — instead of isolated fragments.

This feature is **off by default** and controlled by the
`RAG_CONTEXT_EXPAND` environment variable. When the flag is unset or `0`,
the `ext.services.context_expand` module is never imported and behaviour
is byte-identical to the pre-P1.4 release.

## Why expand

A single retrieved 800-token chunk often starts mid-paragraph and ends
mid-thought. The cross-encoder reranker happily scores that chunk high
because it contains the matching keywords, but when the LLM reads it in
isolation the context is clipped and answers turn hedgy or hallucinatory.
Expanding with `+/- window` siblings solves this cleanly:

* **Retrieval** is fine-grained (800-token chunks) → higher recall, more
  precise reranker scoring.
* **Presentation** is coarse-grained (center chunk + neighbors) → the LLM
  sees a coherent region of the document.

## Enabling

1. **No new dependencies.** Context expansion uses Qdrant's existing
   `scroll` endpoint to fetch siblings by payload filter — no model loads,
   no extra network services.
2. **Set the flag.**
   ```bash
   export RAG_CONTEXT_EXPAND=1
   # optional: widen the window (default 1 -> +/-1 chunk; try 2 or 3)
   export RAG_CONTEXT_EXPAND_WINDOW=1
   ```
3. **Restart the service.** The flag is read at call time, so a hot
   toggle works, but production flips should be a restart.

## How it interacts with the rest of the pipeline

Expansion runs **after** reranking and MMR, **before** budget truncation:

```
retrieve (top-30)
  -> rerank_with_flag (legacy OR cross-encoder, top-10)
  -> mmr_rerank_from_hits (flag-gated; reorders within top-10)
  -> expand_context (flag-gated; each hit -> center + siblings)
  -> budget_chunks (token-fit the final list)
```

Because `budget_chunks` is a longest-prefix-that-fits truncation, it
**naturally trims expanded context that overflows the token budget**.
Over-emitting is safe — the budget step prunes excess while preserving
the highest-ranked material.

## Cost

For each of the top K reranker hits we issue **one `scroll` call** to
Qdrant with a payload filter (`doc_id + chunk_index range + deleted=False`).
`asyncio.gather` runs these in parallel.

* **Network**: up to K scroll round-trips (typical K=10, window=1 →
  10 calls, each pulling up to 3 records).
* **Latency**: ~10-30 ms added on a warm Qdrant (scrolls are served from
  the HNSW graph's payload index, no vector work).
* **Memory**: negligible — each sibling is one payload (~1-2 KB).

Widening the window to 2 or 3 multiplies the returned records per call
but not the number of calls, so total latency scales slowly.

## Deduplication

Siblings that fall inside multiple hits' windows are **emitted exactly
once**. The dedupe key is `(scope_kind, scope_val, chunk_index)` where
`scope_kind` is `"doc"` for KB hits and `"chat"` for private-chat hits.

Example: two adjacent top hits at chunk indices 5 and 6 in the same doc,
window=2. The union of windows is `[3..7] U [4..8] = [3..8]` → six unique
chunks emitted in rank order (rank-1's expansion first, then rank-2's
unique additions).

Rank ordering is preserved across doc boundaries — if rank 1 is doc 10
and rank 2 is doc 20, all of doc 10's siblings appear before any of
doc 20's siblings.

## Fail-open design

`chat_rag_bridge` wraps the expansion call in a bare `except Exception`.
If Qdrant is unreachable, the scroll filter schema changes, or any hit
raises during expansion, retrieval silently falls back to the reranker
(or MMR) output. To surface failures for debugging, run with
`PYTHONWARNINGS=error` or add logging at the `except` block in
`ext/services/chat_rag_bridge.py`.

Per-hit failures in `expand_context` also fail open — if scroll raises
for one hit, that hit is kept as-is while the remaining hits still
expand normally.

## Legacy payloads

Hits whose payload lacks `chunk_index` (documents ingested before the
field was stamped) pass through untouched. Expansion is skipped for
those hits without crashing; the LLM sees the raw hit just as it did
pre-P1.4.

## Expected quality gain

Parent-document retrieval is one of the highest-leverage RAG
improvements on long-document corpora:

* **Faithfulness**: LLM sees paragraph boundaries instead of clipped
  mid-sentence chunks → fewer hallucinations at boundaries.
* **Answer coherence**: the model can reason over a ~2400-token
  window (3 chunks) rather than one 800-token fragment.
* **Citation quality**: sibling chunks help the model anchor its
  answer to the specific passage the reranker pointed at.

Recall itself isn't improved (the retriever already found the relevant
chunk); what improves is **answer quality on the same retrieval set**.

Gate the default flip on pre/post eval:

```bash
RAG_CONTEXT_EXPAND=0 python tests/eval/run_eval.py --out baseline.json
RAG_CONTEXT_EXPAND=1 python tests/eval/run_eval.py --out expanded.json
```

## Flags

| Flag                        | Default | Meaning                                                        |
|-----------------------------|---------|----------------------------------------------------------------|
| `RAG_CONTEXT_EXPAND`        | `0`     | `1` to enable; anything else is off                            |
| `RAG_CONTEXT_EXPAND_WINDOW` | `1`     | Positive int; fetch +/- N siblings per hit. 1-3 is reasonable. |

## Troubleshooting

### Results identical to legacy path

* **Flag not set to exact `"1"`.** Only the string `"1"` enables
  expansion — not `"true"`, `"yes"`, or `"on"`.
* **All hits lack `chunk_index`.** Documents ingested before the
  `chunk_index` field was stamped pass through unchanged. Re-ingest
  those documents (or live with truncated context for legacy data).
* **Retriever returned zero hits.** Expansion is a no-op on an empty
  list.

### Latency spike on flip

Widening `RAG_CONTEXT_EXPAND_WINDOW` adds more records per scroll call
but does not add more calls. If latency spikes, the usual culprit is
cold Qdrant payload indexes — warm them with a handful of scrolls after
the flag flip, or lower `RAG_CONTEXT_EXPAND_WINDOW` temporarily.

### Budget always discards expansions

With `budget_chunks(max_tokens=4000)` and a 10-hit reranker output,
window=1 expansion emits up to 30 chunks ~= 24,000 tokens — well over
budget. Budget keeps the first prefix that fits, so the tail of your
expanded list is silently dropped. This is desired behaviour; if you
want more expanded context to survive, lift the budget:

```bash
export RAG_MAX_CONTEXT_TOKENS=8000  # if exposed in your config
```

Or narrow the retriever's top_k so the expanded total is smaller.
