# Query Understanding LLM Runbook

**Purpose:** operate, troubleshoot, and tune the hybrid regex+LLM intent
router introduced in Plan B Phase 4.

## Service overview

| Field | Value |
|---|---|
| Container | `orgchat-vllm-qu` |
| Image | `vllm/vllm-openai:latest` (V1 engine) |
| Model | `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit` |
| Served-model-name | `qwen3-4b-qu` |
| GPU | 1 (RTX PRO 4000 Blackwell, 24 GB) |
| Endpoint (intra-cluster) | `http://vllm-qu:8000/v1/chat/completions` |
| Endpoint (host) | `http://localhost:8101/v1/chat/completions` |
| Schema enforcement | OpenAI `response_format` + `json_schema` (xgrammar) |

## Healthcheck

```bash
docker compose ps vllm-qu
# Expected: STATUS Up X (healthy)

curl -s http://localhost:8101/v1/models | python -m json.tool
# Expected: data[0].id == "qwen3-4b-qu"

# A real classification probe
curl -s -X POST http://localhost:8101/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-4b-qu",
    "messages": [
      {"role":"system","content":"You output only JSON."},
      {"role":"user","content":"Classify the query as one of metadata, global, specific, specific_date. Query: list reports"}
    ],
    "temperature": 0,
    "max_tokens": 64,
    "response_format": {
      "type":"json_schema",
      "json_schema": {
        "name":"intent_only",
        "schema":{"type":"object","required":["intent"],"properties":{"intent":{"type":"string","enum":["metadata","global","specific","specific_date"]}}}
      }
    }
  }'
```

## Flags

| Flag | Default | Effect when 1 |
|---|---|---|
| `RAG_QU_ENABLED` | `0` (flips after shadow gate) | Hybrid router promotes the LLM result on escalated queries |
| `RAG_QU_URL` | `http://vllm-qu:8000/v1` | vLLM base URL (override for canary / staging) |
| `RAG_QU_MODEL` | `qwen3-4b-qu` | served-model-name |
| `RAG_QU_LATENCY_BUDGET_MS` | `600` | Soft deadline; on miss the bridge falls back to regex |
| `RAG_QU_CACHE_ENABLED` | `1` | Redis DB 4 cache |
| `RAG_QU_CACHE_TTL_SECS` | `300` | Cache TTL (s) |
| `RAG_QU_REDIS_DB` | `4` | Redis DB number (3 reserved for RBAC by Plan A) |
| `RAG_QU_SHADOW_MODE` | `0` | Run LLM on every query and log both decisions |

## Pre-cache the model weights

Before bringing up vllm-qu on a fresh deploy host, stage the weights into
the cache directory the container mounts:

```bash
HF_HOME=/home/vogic/LocalRAG/volumes \
  hf download cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit \
  --cache-dir /home/vogic/LocalRAG/volumes/models
```

or run the bundled script:

```bash
./scripts/stage_qwen3_qu.sh
```

Verify the cache is warm:

```bash
ls /home/vogic/LocalRAG/volumes/models/models--cpatonn--Qwen3-4B-Instruct-2507-AWQ-4bit/snapshots/*/*.safetensors
du -sh /home/vogic/LocalRAG/volumes/models/models--cpatonn--Qwen3-4B-Instruct-2507-AWQ-4bit/
```

Expected: ≈ 3.3 GB total, one `model.safetensors` symlinked to the
content-addressed blob.

## Daily checks

1. `nvidia-smi` — GPU 1 memory.used should be ≈ 12-13 GB (TEI + reranker
   + Qwen3-4B); >18 GB suggests another tenant moved in and you should
   reduce `--gpu-memory-utilization` to 0.40.
2. `curl -s http://localhost:9090/api/v1/query?query=rag_qu_cache_hit_ratio`
   — should be > 0.3 in steady state.
3. `curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(rag_qu_latency_seconds_bucket[5m]))'`
   — should be < 0.6.

## Common failure modes

### `vllm-qu` container restart loops

- Check `docker logs orgchat-vllm-qu --tail 100`. Common causes:
  - Out-of-memory: another process took GPU 1. Run `nvidia-smi` and
    consider lowering `--gpu-memory-utilization` to 0.40.
  - HF cache miss: weights not in `volumes/models`. Re-run
    `./scripts/stage_qwen3_qu.sh`.
  - Quantization mismatch: never pass `--quantization awq` for the
    cpatonn build — its `config.json` declares `compressed-tensors` and
    vLLM auto-detects.

### `rag_qu_schema_violations_total` rising

- Check `docker logs orgchat-vllm-qu` for guided JSON errors. Most often
  this means the model revision rotated and the tokenizer mismatch broke
  schema enforcement.
- Mitigation: pin the snapshot revision in the model id. The cache dir
  records the snapshot under `snapshots/<rev_hash>/`.

### Hybrid escalation rate > 40%

- Inspect via `scripts/analyze_shadow_log.py` (with shadow mode on for a
  representative window).
- Likely: queries are unusually pronoun-heavy or relative-time-heavy.
  Consider raising `_LONG_QUERY_TOKEN_THRESHOLD` in
  `ext/services/query_intent.py` or adding a per-tenant policy.

### Cache hit ratio < 0.3

- The workload may be unique enough that the 5-minute TTL isn't catching
  repeats.
- Try `RAG_QU_CACHE_TTL_SECS=900`. Higher TTL is safe — the cache key
  includes the last assistant turn ID, so context shifts invalidate
  naturally.

## Promotion checklist (shadow → production)

After 7 days of `RAG_QU_SHADOW_MODE=1`:

1. Pipe the shadow log through the analyzer:
   ```bash
   docker logs orgchat-open-webui 2>&1 \
     | grep 'orgchat.qu_shadow' \
     | python scripts/analyze_shadow_log.py
   ```
2. Confirm: agreement rate > 75% on `metadata` / `global`, < 60% on
   `specific` (the LLM is winning the hard cases).
3. Run `make eval-baseline` with `RAG_QU_ENABLED=1` against `kb_eval`
   and `kb_1` — `chunk_recall@10` improvement on `multihop` and
   `evolution` strata ≥ +3 pp vs Plan A end-state.
4. Flip `RAG_QU_ENABLED=1` in `compose/.env`.
5. Set `RAG_QU_SHADOW_MODE=0`.
6. `docker compose up -d --force-recreate open-webui`.

## Rollback

```bash
cd /home/vogic/LocalRAG/compose
sed -i 's/^RAG_QU_ENABLED=1$/RAG_QU_ENABLED=0/' .env
docker compose up -d --force-recreate open-webui
```

The bridge soft-fails to regex; users see no error.

To stop the vllm-qu service entirely:

```bash
docker compose stop vllm-qu
docker compose rm -f vllm-qu
# Verify GPU 1 returned to baseline
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```
