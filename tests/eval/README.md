# RAG eval harness (P0.1)

Numerically gate later RAG changes (hybrid search, reranker, MMR, chunker, etc.)
so we don't ship "feels better" without proof.

## Prereqs
Live Docker stack up: `orgchat-qdrant` on `:6333`, `orgchat-tei` reachable, and
`orgchat-vllm-chat` reachable for golden-set generation (eval run itself only
needs Qdrant + TEI).

## Seeding the eval corpus

For meaningful baseline numbers, the live Qdrant must have a `kb_eval` collection
with at least ~100 chunks of diverse text. Live KB collections are too sparse
(kb_1=1, kb_3=6, kb_4=146, kb_5=0 chunks) to gate quality changes.

Seed once (idempotent — skips if `kb_eval` is already populated):

    python scripts/seed_eval_corpus.py --qdrant-url http://localhost:6333 \
                                       --tei-url http://localhost:8080

Then generate golden + run eval as documented below. The seed corpus is
the worktree's own documentation plus any `.md`/`.txt` files committed under
`tests/eval/seed_corpus/` — drop more public-domain files there to grow the set.

Why a dedicated collection:
- **reproducibility** — running eval shouldn't depend on whatever happens to be
  in `kb_1`/`kb_4` today
- **density** — we need ≥ 50 docs, ≥ 100 chunks, ≥ 100 back-generated queries
- **isolation** — `kb_eval` is tagged with `kb_id="eval"` so it doesn't pollute
  user-facing analytics or leak into retrieval for real chats

## Generate a golden set
```
python -m tests.eval.generate_golden \
    --qdrant-url http://localhost:6333 \
    --chat-url   http://localhost:8000/v1 \
    --chat-model Qwen/Qwen2.5-14B-Instruct-AWQ \
    --chat-api-key sk-internal-dummy \
    --collections kb_eval \
    --samples-per-collection 100 \
    --out tests/eval/golden.jsonl
```
`--collections` accepts a CSV list (e.g. `kb_1,kb_3,kb_4,kb_5`) if you want
to eval against live KBs instead. When the default `kb_eval` is missing the
script falls back to scanning the live `kb_*` collections automatically.
It scrolls random points from each collection, asks the chat model
"what question would this excerpt answer?", and writes one JSONL row per
accepted chunk. Idempotent: refuses to overwrite unless `--force` is set.

## Run the eval
```
python -m tests.eval.run_eval \
    --golden tests/eval/golden.jsonl \
    --qdrant-url http://localhost:6333 \
    --tei-url http://localhost:8080 \
    --k 10
```
Writes `tests/eval/results-<YYYY-MM-DD>.json` and prints a one-line summary.

## Metrics
- **chunk_recall@K** — fraction of gold chunks present in the top-K retrieved.
  Primary signal for "did we pull the right chunk?" Ranges 0..1.
- **doc_recall@K** — fraction of gold docs present in top-K. Softer signal.
- **MRR@K** — reciprocal rank of the first gold doc. Sensitive to ordering.
- **unique_docs@K** — diversity signal, used by MMR-style changes.
- **p50/p95 latency_ms** — wall-clock for the `retrieve()` call per query.

## Interpreting results
A flag flip (e.g. enabling hybrid search) must produce at minimum:
- chunk_recall@10: **+3 pp absolute** improvement
- mrr@10: **no regression** (± noise within stddev over 3 runs)

If a change regresses chunk_recall@10 by more than 2 pp it must revert unless
there's a documented reason (e.g. intentional diversity trade).

## Important caveat — this is a plumbing test, not quality
The golden set is **auto-generated**: we take a chunk, ask an LLM to invent a
question it could answer, and then check whether retrieval finds that same
chunk. This is near-trivial for any working retriever; the retriever gets to
cheat because the chunk's own semantics are baked into the query. It will
catch:
- broken filtering (wrong kb_id / deleted = true leak)
- broken embedder (wrong dim, all-zero vectors)
- broken chunk_id plumbing
- catastrophic regressions (recall falls off a cliff)

It will **not** catch:
- real relevance losses on messy human queries
- multi-hop / aggregation failures
- ranking quality at the margin

Human-labelled golden queries are still required before the CI regression
gate is treated as meaningful. See the plan's P0.1 Step 1 for the handwritten
schema (`expected_doc_ids` + `expected_chunk_indices` per user question).

## Faithfulness (P3.5)

The retrieval harness above grades **did we pull the right chunk?** It does
**not** grade **did the model's answer stay grounded in that chunk?**
Features like `RAG_CONTEXT_EXPAND` and `RAG_CONTEXTUALIZE_KBS` primarily
improve answer quality on the same retrieval set, so they are invisible to
the retrieval metrics.

RAGAS-style **faithfulness** plugs that gap: fraction of an answer's factual
claims that are supported by the retrieved context. Two-pass LLM judge:

1. **Extract claims** — ask the chat model to break the generated answer into
   a flat list of standalone claims.
2. **Grade each claim** — for each claim, ask the judge *"is this claim
   supported by the context? YES / NO"*.
3. **Score** = supported / total claims, range `[0.0, 1.0]`.

### Run it

```
python tests/eval/run_faithfulness_eval.py \
    --golden tests/eval/golden.jsonl \
    --qdrant-url http://localhost:6333 \
    --tei-url http://172.19.0.6:80 \
    --chat-url http://172.19.0.7:8000/v1 \
    --chat-model orgchat-chat \
    --label faithfulness_baseline \
    --out tests/eval/results/faithfulness_baseline.json
```

The same `RAG_HYBRID`/`RAG_RERANK`/`RAG_MMR`/`RAG_CONTEXT_EXPAND` env flags
control the retrieval stage, so flip a flag and re-run to compare faithfulness
before and after the change.

### Cost per run

Per query: 1 retrieve + 1 answer-gen + 1 claim-extract + up to 10 grade calls.
On Qwen2.5-14B-AWQ that is roughly **2-3 seconds per query, ~2-3 minutes for
a 50-row golden** at the default `--concurrency 4`.

### Interpretation

| Score     | Read-out                                                              |
|-----------|-----------------------------------------------------------------------|
| `>= 0.90` | Very grounded. Answers are mostly supported by the retrieved chunks.  |
| `0.70-0.90` | Reasonable. Expect occasional unsupported sentences.                |
| `< 0.70`  | Frequent hallucination. Retrieval may be under-covering the question, |
|           | or the model is rambling past what the context supports.              |

A flag flip should **lift mean faithfulness by at least +3 pp** with no
regression in retrieval metrics to justify a default flip.

### Bias note

We use the **same** local Qwen2.5-14B-AWQ as both the answer generator
**and** the judge. That is self-grading and systematically optimistic: the
model tends to grade its own claims generously. Ideal setup uses a distinct
stronger judge (GPT-4, Claude) but our deployment is air-gapped. Treat
absolute scores as directional; **relative** deltas between two runs against
the same judge are what matter for gating decisions.

Sample-size caveat: the auto-generated 50-row golden is tight. Treat a
`+3 pp` lift with fewer than 3 runs (std-dev unknown) as suggestive, not
conclusive.
