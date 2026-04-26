# vllm-qu — Query Understanding LLM

Serves **Qwen3-4B-Instruct-2507-AWQ-4bit** for the hybrid regex+LLM intent
router introduced in Plan B Phase 4. Pinned to GPU 1 (RTX PRO 4000
Blackwell, 24 GB) alongside TEI and the open-webui reranker — GPU 0 is
reserved for `vllm-chat` (Gemma-4-31B-AWQ at ~89% VRAM).

## Model

- **Repo:** `cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit`
- **Footprint:** ≈ 3.5 GB weights + ≈ 4 GB KV cache at
  `--gpu-memory-utilization 0.45`
- **Total resident on GPU 1 with TEI + reranker:** ≈ 12 GB / 24 GB ≈ 50%

## Endpoints

| Surface | URL |
|---|---|
| Inside the docker network | `http://vllm-qu:8000/v1` |
| Host machine | `http://localhost:8101/v1` |

## Engine

vLLM V1 with **xgrammar guided JSON**. The QU module
(`ext/services/query_understanding.py`) supplies a JSON schema; vLLM
constrains generation to valid output, eliminating the schema-violation
noise that plagued unconstrained LLM routers.

## Flags

The bridge consumes these via the `open-webui` container env. Defaults
shown live in `compose/docker-compose.yml`.

| Flag | Default | Effect when 1 |
|---|---|---|
| `RAG_QU_ENABLED` | `0` (flips after Phase 4 shadow gate) | Hybrid router promotes LLM result on escalated queries |
| `RAG_QU_URL` | `http://vllm-qu:8000/v1` | vLLM base URL (override for canary / staging) |
| `RAG_QU_MODEL` | `qwen3-4b-qu` | served-model-name |
| `RAG_QU_LATENCY_BUDGET_MS` | `600` | Soft deadline; on miss the bridge falls back to regex |
| `RAG_QU_CACHE_ENABLED` | `1` | Redis DB 4 cache |
| `RAG_QU_CACHE_TTL_SECS` | `300` | Cache TTL (s) |
| `RAG_QU_REDIS_DB` | `4` | Redis DB number (3 reserved for RBAC by Plan A) |
| `RAG_QU_SHADOW_MODE` | `0` | Run LLM on every query and log both decisions |

## Operational notes

- **Cold start:** ≈ 90 s to load weights + warm KV cache. open-webui
  intentionally does NOT `depends_on` vllm-qu — the bridge soft-fails to
  regex when vllm-qu is unreachable, so chat stays alive even if the
  service crashes.
- **GPU pressure check:** if `nvidia-smi` shows GPU 1 > 18 GB
  (75% saturated), reduce `--gpu-memory-utilization` to 0.40.
- **Healthcheck:** the docker `healthcheck` shells `curl /v1/models` and
  greps for the served-model-name. Returns 200 once the model is loaded.

## Pre-cache the weights (offline-air-gap setup)

Before bringing the service up on a fresh deploy host, stage the model
weights into the cache directory the container mounts:

```bash
HF_HOME=/home/vogic/LocalRAG/volumes \
  hf download cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit \
  --cache-dir /home/vogic/LocalRAG/volumes/models
```

Verify the cache:

```bash
du -sh /home/vogic/LocalRAG/volumes/models/models--cpatonn--Qwen3-4B-Instruct-2507-AWQ-4bit
ls /home/vogic/LocalRAG/volumes/models/models--cpatonn--Qwen3-4B-Instruct-2507-AWQ-4bit/snapshots/*/*.safetensors
```

Expected: ≈ 3.3 GB total, one `model.safetensors` symlinked to the
content-addressed blob.

## Bring up the service

```bash
cd /home/vogic/LocalRAG/compose
docker compose up -d vllm-qu

# Wait for the model to load (~90 s)
docker compose ps vllm-qu
docker logs --tail 50 orgchat-vllm-qu

# Smoke test
curl -s http://localhost:8101/v1/models | python -m json.tool
```
