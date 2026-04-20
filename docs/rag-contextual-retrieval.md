# Contextual Retrieval (P2.7)

Optional per-chunk LLM-generated context prepended to each chunk **before**
embedding. Adapted from Anthropic's
[Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
recipe.

## Idea

Rather than embedding a raw chunk:

```
The monthly plan costs $99.
```

we ask the chat model to write a one-sentence context that situates the
chunk in its document, then embed + index the concatenation:

```
This chunk is about pricing in the intro section of pricing-sheet.md.

The monthly plan costs $99.
```

Retrieval for a vague query like "how much is the subscription?" now sees
"pricing" alongside the raw `$99`, which the raw chunk alone would miss
with a bag-of-words or dense-only match.

## Expected wins (the caveat)

Anthropic reports a **35% reduction in retrieval failure** on their
evaluation using **Claude** to generate the per-chunk context. This repo
uses **Qwen2.5-14B-Instruct-AWQ** locally, and the effectiveness on the
smaller local model is **unproven**. Treat the flag as an eval-gated
experiment, not a default.

**Run `tests/eval/run.py` before and after enabling**; only enable for a
given KB if the contextualized build shows a measurable gain on that
KB's query distribution.

## Cost

Per-chunk cost on Qwen2.5-14B-AWQ (single 4090 / RTX 6000 Ada):

| Measure                        | Value (approx.) |
|--------------------------------|-----------------|
| Tokens / call (prompt + output)| ~300-400 in, ~60 out |
| Latency / call                 | ~2-3 seconds    |
| Cost / call                    | 0 (self-hosted) |
| Concurrency (default)          | 8               |
| Real ingest time, 1000 chunks  | ~40 minutes     |

vLLM's prefix cache amortizes the shared document-title + preamble tokens
across sibling chunks, so sustained throughput is better than the
per-call number suggests — but it is still a fundamentally slow pass.
Use only on **high-value KBs** where the eval win justifies the cost.

## Flags

```bash
# Required to enable
export RAG_CONTEXTUALIZE_KBS=1

# Optional tuning
export RAG_CONTEXTUALIZE_CONCURRENCY=8   # parallel chat calls per ingest
```

Both flags are read inside `ingest_bytes()` at call time — no restart
needed for changes between ingests. Other values of `RAG_CONTEXTUALIZE_KBS`
(`"true"`, `"yes"`, etc.) are treated as OFF; only the literal `"1"`
enables.

## Default-off guarantee

When the flag is unset, `ingest.py` does **not** import
`ext.services.contextualizer` at all. No httpx client is constructed,
no chat-model call is made, no extra bytes are added to the payload.
A unit test (`test_flag_off_does_not_import_contextualizer`) guards
against future refactors that accidentally pull the module into the
default path.

## Pipeline version impact

- **Flag off** → `model_version` ends in `ctx=none` (unchanged).
- **Flag on + augmentation ran** → `model_version` ends in `ctx=contextual-v1`.
- **Flag on + augmentation fell open** → `ctx=none` (honest stamp; these
  chunks really are un-augmented).

Because `pipeline_version` is stamped on every Qdrant point, mixing
`ctx=none` and `ctx=contextual-v1` chunks in the same collection is
safe — retrieval treats them equally for ranking but the provenance
tag lets eval and reindex jobs tell them apart.

**To realize the win on an existing KB: re-run ingest.** Old chunks
keep `ctx=none` until their documents are re-ingested. A targeted
reindex script (not in this PR) would filter
`pipeline_version != current_version(context_augmented=True)` and
re-embed just those rows.

## Fail-open behaviour

Two levels of fail-open so a chat-endpoint hiccup never crashes ingest:

1. **Per chunk** — `contextualize_chunk` catches timeout / 5xx / empty
   response / oversized response / malformed JSON and returns the raw
   chunk. Other chunks in the same batch are unaffected.
2. **Per batch** — `_maybe_contextualize_chunks` in `ingest.py` wraps
   the whole `contextualize_batch` call in a try/except. If the entire
   call errors (import failure, missing env var, network totally down)
   ingest continues with raw chunks and stamps `ctx=none`.

## Prompt

```
<document>
{doc_title}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short, succinct context to situate this chunk within the
overall document, so that retrieval of this chunk works better. Answer
ONLY with the succinct context - no preamble, no quotes, no apologies.
Under 50 words.
```

The chunk is clipped to 4000 chars in the prompt (safety valve, not a
tuning knob — real chunks top out at ~3200 chars at 800 tokens).

Echo prefixes the model sometimes emits despite the "no preamble"
instruction (`Context:`, `Situated context:`, `Here is the context:`,
`Here's the context:`, `Succinct context:`) are stripped case-insensitively.

## Files

- `ext/services/contextualizer.py` — the augmentation module
- `ext/services/pipeline_version.py` — `current_version(context_augmented=bool)`
- `ext/services/ingest.py` — gated call between chunk and embed
- `tests/unit/test_contextualizer.py` — module unit tests
- `tests/unit/test_ingest_contextualize_flag.py` — ingest-wiring tests
