# Swapping the chat model

This stack is designed so the chat LLM can be swapped with a single env-var
change and a preflight re-run. Embeddings, vision, and whisper are
independent and unaffected.

## Pre-check

1. Verify the target model's quantized size fits in VRAM alongside TEI
   (~3 GB) and any optional reranker (~1.5 GB). On a 32 GB GPU, keep the
   chat model at ~25 GB or less to leave headroom for KV-cache / batching.
2. Check if your target has an AWQ / int4 / gptq quant on Hugging Face.
   Unquantized bf16 27B+ models won't fit on 32 GB.
3. Gated repos (e.g. `google/gemma-3-*`, `meta-llama/Llama-3.*`) require
   accepting the license on Hugging Face first, then setting `HF_TOKEN`
   in the environment you run `preflight_models.py` from.

## Steps

1. Stop the chat services (leave embeddings, qdrant, etc. up):
   ```
   docker compose stop vllm-chat model-manager
   ```
2. Edit `compose/.env`:
   ```
   CHAT_MODEL=<new-hf-repo>
   ```
3. Update the tokenizer alias if the model family changed. This keeps the
   RAG token budget accurate for the new tokenizer:
   ```
   RAG_BUDGET_TOKENIZER=gemma   # or: qwen, llama, cl100k
   ```
   Optional: pin a specific tokenizer repo via
   `RAG_BUDGET_TOKENIZER_MODEL=<hf-repo>` when the default for the alias
   doesn't match your exact model build.
4. Populate the model cache:
   ```
   HF_HUB_OFFLINE=0 python scripts/preflight_models.py
   ```
   The script auto-reads `CHAT_MODEL` from env. For gated repos, set
   `HF_TOKEN=...` in the same shell before running.
5. Start chat:
   ```
   docker compose up -d vllm-chat model-manager
   ```
6. Verify vllm is serving the new model:
   ```
   curl http://localhost:8000/v1/models
   ```

### One-shot CLI

The steps above are wrapped by `scripts/swap_chat_model.py`:
```
python scripts/swap_chat_model.py --model google/gemma-3-12b-it --tokenizer gemma
```
It backs up `compose/.env`, patches `CHAT_MODEL` (and optionally
`RAG_BUDGET_TOKENIZER`), then runs preflight with `HF_HUB_OFFLINE=0`. It
does NOT restart containers — do that yourself after verifying preflight
succeeded. `--dry-run` prints the plan without changes.

## Rollback

1. Edit `compose/.env` back to the previous `CHAT_MODEL` (a `.env.bak`
   written by the swap script is next to it).
2. Restart:
   ```
   docker compose restart vllm-chat model-manager
   ```
   Old weights are still in `volumes/models/` — no re-download needed.

## VRAM cheat sheet

Values are approximate, on RTX 6000 Ada 32 GB, chat-only (no vision /
whisper loaded concurrently):

| Model                                   | VRAM    | Notes                                   |
| --------------------------------------- | ------- | --------------------------------------- |
| `Qwen/Qwen2.5-14B-Instruct-AWQ`         | ~8 GB   | current default                         |
| `Qwen/Qwen2.5-32B-Instruct-AWQ`         | ~18 GB  | good 32B quant                          |
| `google/gemma-3-12b-it`                 | ~24 GB  | tight; disables GPU reranker            |
| `google/gemma-3-27b-it`                 | ~54 GB  | **does not fit** on 32 GB               |
| `solidrust/gemma-3-27b-it-AWQ`          | ~14 GB  | recommended Gemma target                |
| `meta-llama/Llama-3.3-70B-Instruct-AWQ` | ~18 GB  | AWQ 4-bit 70B                           |

## Model-family tokenizer mapping

| Chat model family | `RAG_BUDGET_TOKENIZER` |
| ----------------- | ---------------------- |
| `Qwen/*`          | `qwen`                 |
| `google/gemma*`   | `gemma`                |
| `meta-llama/*`    | `llama`                |
| (fallback)        | `cl100k`               |

Exact-version aliases are also registered: `qwen2.5`, `gemma-3`,
`gemma-3-12b`. Unknown aliases fall back to `cl100k` with a warning in
the server log — safe but less accurate for non-English text.

## Gated repos — Hugging Face tokens

`google/gemma-*` and `meta-llama/*` are gated models. To download:
1. Visit the repo page on huggingface.co and accept the license.
2. Create a read token at https://huggingface.co/settings/tokens.
3. Export it before running preflight:
   ```
   export HF_TOKEN=hf_xxxxxxxxxxxx
   HF_HUB_OFFLINE=0 python scripts/preflight_models.py
   ```
   Or pass it through `scripts/swap_chat_model.py` the same way (it
   inherits the current shell environment).

## What doesn't change

- Embedding model (`EMBED_MODEL`, default `BAAI/bge-m3`) is independent
  and unaffected by chat-model swaps. Changing the embed model requires
  re-indexing the entire RAG corpus — that is a separate operation.
- Vision (`VISION_MODEL`) and whisper (`WHISPER_MODEL`) are independent.
- RAG retrieval, rerank, and budget logic are tokenizer-aware but
  model-agnostic.
