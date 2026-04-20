# Per-KB RAG Quality Settings (P3.0)

## Why

A single process-level `RAG_RERANK=1` flag is too coarse. A 50-document
FAQ knowledge base does not need cross-encoder reranking — hybrid
retrieval already finds the right chunk and the reranker adds 200+ms
for no quality gain. A year-long engineering-docs KB with 50,000 chunks
absolutely does. One switch can't satisfy both.

P3.0 ships per-KB `rag_config`. An admin stamps the quality preferences
on each KB; when a chat selects multiple KBs the strictest setting
wins (union for booleans, max for numeric thresholds). The chat sees
the effective configuration via an SSE progress stream so the UI can
show "retrieving... reranking... thinking..." in real time.

## The config keys

Each entry is optional. Missing keys inherit the process-level default.

| Key | Type | Maps to env | Meaning |
|-----|------|-------------|---------|
| `rerank` | bool | `RAG_RERANK` | Cross-encoder reranker (BAAI/bge-reranker-v2-m3) |
| `rerank_top_k` | int | `RAG_RERANK_TOP_K` | Candidates fed to the reranker (default 10, wider when MMR is on) |
| `mmr` | bool | `RAG_MMR` | Maximal Marginal Relevance diversification |
| `mmr_lambda` | float | `RAG_MMR_LAMBDA` | Relevance vs. diversity tradeoff (0.0-1.0, default 0.7) |
| `context_expand` | bool | `RAG_CONTEXT_EXPAND` | Fetch +/-N sibling chunks so the LLM sees coherent context |
| `context_expand_window` | int | `RAG_CONTEXT_EXPAND_WINDOW` | How many siblings each side (default 1) |
| `spotlight` | bool | `RAG_SPOTLIGHT` | Wrap retrieved text with anti-prompt-injection markers |
| `semcache` | bool | `RAG_SEMCACHE` | Redis-backed semantic retrieval cache |
| `contextualize_on_ingest` | bool | `RAG_CONTEXTUALIZE_KBS` | Ingest-side — chat-model rewrites each chunk with summary context (applied on next re-ingest) |

## Setting a KB's config (admin only)

```bash
# Turn on rerank + context expansion for the "engineering-docs" KB.
curl -X PATCH http://orgchat.local/api/kb/42/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"rerank": true, "context_expand": true, "context_expand_window": 2}'

# Response:
# {"kb_id": 42, "rag_config": {"rerank": true, "context_expand": true,
#                              "context_expand_window": 2}}
```

The PATCH is partial — only keys you include are changed. Setting
`rerank: false` explicitly disables rerank for this KB even when the
process default is on.

Read the current config:

```bash
curl http://orgchat.local/api/kb/42/config -H "Authorization: Bearer $ADMIN_TOKEN"
```

Unknown keys return `400 Bad Request`. Bad value types (e.g.
`"context_expand_window": "abc"`) also return 400 with the offending
key in the `detail` field.

## Multi-KB merge policy

When a user selects 2+ KBs for a chat, the configs are merged with
**UNION (for booleans) / MAX (for numeric)** — the strictest setting
wins. Reasoning:

- Rerank is a per-chunk decision; we can't rerank half the candidate
  pool. If any selected KB wants rerank, the whole request gets it.
- A wider context-expand window is never wrong (just slower); the larger
  window satisfies the more demanding KB.
- This policy is conservative on latency (you pay the peak) but correct
  on quality — no silent downgrade.

Example: small FAQ KB has `{"rerank": false}`, big docs KB has
`{"rerank": true, "context_expand_window": 3}`. The effective config
for a chat that selects both: `{"rerank": true, "context_expand_window": 3}`.

## How the bridge applies the config

Per request, the bridge:

1. Reads the chat's `kb_config` (list of selected KB IDs).
2. Fetches each KB's `rag_config` JSONB column.
3. Calls `kb_config.merge_configs(...)` → merged dict.
4. Calls `kb_config.config_to_env_overrides(...)` → `{RAG_*: "value"}` dict.
5. Enters `flags.with_overrides(...)` scope for the entire pipeline.
6. Every pipeline stage reads via `flags.get("RAG_*", default)` —
   contextvar overlay wins, falling back to `os.environ` when the
   current KB has no opinion.

Crucially, the overlay is stored in a `contextvars.ContextVar` so:

- Concurrent requests don't pollute each other's flags.
- `asyncio.create_task` and `asyncio.gather` correctly propagate the
  overlay into child tasks (parallel KB searches inherit the config).
- Exit the `with` block → overlay reverts atomically.

## Progress SSE endpoint

The UX half of P3.0: users should see that work is happening between
"submit" and "first LLM token". Subscribe to:

```
GET /api/rag/stream/{chat_id}?q=<urlencoded-query>
Accept: text/event-stream
```

Each stage emits one or two events. Shape:

```
event: stage
data: {"stage": "embed", "status": "running"}

event: stage
data: {"stage": "retrieve", "status": "done", "ms": 9, "hits": 30}

event: stage
data: {"stage": "rerank", "status": "done", "ms": 267, "top_k": 10}

event: stage
data: {"stage": "mmr", "status": "skipped", "reason": "flag_off"}

event: hits
data: {"hits": [{"doc_id": 42, "filename": "policy.md", "kb_id": 7}, ...]}

event: done
data: {"total_ms": 580}
```

The LLM generation is NOT part of this stream — the frontend should
fire the existing `/api/chat/completions` request in parallel. The SSE
stream exists so the UI can render a progress indicator during the
400-700ms the retrieval pipeline is running.

Curl test:

```bash
curl -N \
  -H "X-User-Id: 42" -H "X-User-Role: user" \
  "http://localhost:8080/api/rag/stream/my-chat-id?q=how%20do%20I%20set%20vacation"
```

## Recommended settings per KB size

Rough starting points — measure with RAGAS before committing:

| KB shape | Recommended config |
|----------|-------------------|
| Small FAQ (< 500 chunks) | `{}` (all defaults — hybrid dense+sparse is enough) |
| Product-docs (1k-10k chunks) | `{"rerank": true}` |
| Year-long engineering docs (10k+ chunks) | `{"rerank": true, "context_expand": true, "context_expand_window": 2}` |
| Legal / compliance (must cite exactly) | `{"rerank": true, "context_expand": true, "spotlight": true}` |
| High-traffic public KB | Add `{"semcache": true}` to any of the above |

## Observability

Prometheus gauges at `/metrics` snapshot the effective flags per request:

- `orgchat_rag_flag_state{flag="rerank"}` — 0/1 gauge
- `orgchat_rag_flag_state{flag="mmr"}` — 0/1 gauge
- `orgchat_rag_flag_state{flag="context_expand"}` — 0/1 gauge
- `orgchat_rag_stage_latency_seconds{stage="..."}` — histograms

Grep logs for `rag: request started req=xxxxx user=... kbs=... chat=...`
to trace a single request's KB selection and subsequent stage timings.
