# RAG Faithfulness Eval (P3.5)

## What it is

**Faithfulness** (as defined by the [RAGAS](https://arxiv.org/abs/2309.15217)
paper) measures the fraction of factual claims in a generated answer that are
supported by the retrieved context. It is an **answer-quality** metric, not a
retrieval metric: retrieval can be perfect and the model can still hallucinate
past the supplied evidence, and retrieval can be mediocre while the model
stays faithfully hedged. `chunk_recall@K` and `MRR@K` are blind to that
distinction.

Our implementation is a local, two-pass LLM-as-judge:

1. **Extract claims** — chat model breaks the answer into standalone factual
   claims.
2. **Grade each claim** — same chat model decides whether each claim is
   supported by the context (YES/NO).
3. **Score = supported / total** ∈ `[0.0, 1.0]`.

Ambiguous judge responses ("maybe", "probably") are deliberately counted as
NO — unfaithful by default is safer than over-crediting the answer.

## How to run

```bash
python tests/eval/run_faithfulness_eval.py \
    --golden tests/eval/golden.jsonl \
    --qdrant-url http://localhost:6333 \
    --tei-url http://172.19.0.6:80 \
    --chat-url http://172.19.0.7:8000/v1 \
    --chat-model orgchat-chat \
    --label faithfulness_baseline \
    --out tests/eval/results/faithfulness_baseline.json
```

Cost: ~2-3 seconds per query on Qwen2.5-14B-AWQ → ~2-3 minutes for a 50-row
golden set at the default `--concurrency 4`.

Retrieval-stage env flags (`RAG_HYBRID`, `RAG_RERANK`, `RAG_MMR`,
`RAG_CONTEXT_EXPAND`, `RAG_CONTEXTUALIZE_KBS`) are read at call time, so a
single process can compare flagged-on vs. flagged-off by varying the env
between runs.

## When to use

**Before flipping a default on** for any feature whose purpose is to improve
answer quality rather than retrieval recall:

* `RAG_CONTEXT_EXPAND=1` — sibling-chunk expansion gives the model paragraph
  context instead of clipped fragments. Retrieval metrics are unchanged;
  faithfulness should improve.
* `RAG_CONTEXTUALIZE_KBS=1` — contextual retrieval prepends doc-level
  summaries to each chunk. Again: retrieval unchanged, answer grounding
  should improve.

Suggested workflow:

```bash
# Baseline: feature off
RAG_CONTEXT_EXPAND=0 python tests/eval/run_faithfulness_eval.py \
    --label faithfulness_ctx_expand_off \
    --out tests/eval/results/faithfulness_ctx_expand_off.json

# Treatment: feature on
RAG_CONTEXT_EXPAND=1 python tests/eval/run_faithfulness_eval.py \
    --label faithfulness_ctx_expand_on \
    --out tests/eval/results/faithfulness_ctx_expand_on.json
```

A **+3 pp mean-faithfulness lift** with no retrieval regression is a
reasonable bar to justify flipping the default. Anything less is
probably within noise on a 50-row golden.

## Caveats

* **Local-judge bias.** We use the same Qwen2.5-14B-AWQ as both answer
  generator and judge. Models tend to grade their own outputs generously, so
  absolute scores skew high. Treat deltas as the signal, not absolute
  numbers. An air-gapped deployment cannot use external judges (Claude,
  GPT-4) which would be more reliable.
* **Sample size.** 50 auto-generated golden queries is tight. Run 3x and
  average before declaring a feature flip — std-dev across runs can be a
  couple pp given the temperature-0 but still non-deterministic chat model.
* **Synthetic queries.** The golden set is back-generated (given a chunk,
  ask the LLM what question this would answer). That biases toward
  "chunk-answerable" queries, which means faithfulness here is a **lower
  bound** on the real-world score — messy human queries typically yield
  lower faithfulness because the context is less on-the-nose.
* **Vacuous cases.** Empty answers and answers where zero claims are
  extractable score 1.0 (vacuously faithful). Watch the `n_claims` stats in
  the output JSON — if the model refuses to produce claims, the score is
  meaningless.

## See also

* [RAGAS paper](https://arxiv.org/abs/2309.15217) — the framework we're
  mimicking. Our version drops their *context_precision* and
  *answer_relevance* metrics to keep the local-judge cost budget tight.
* [`docs/rag-metrics.md`](./rag-metrics.md) — the retrieval-side harness.
* [`docs/rag-context-expand.md`](./rag-context-expand.md),
  [`docs/rag-contextual-retrieval.md`](./rag-contextual-retrieval.md) — the
  features whose default flip this eval gates.
