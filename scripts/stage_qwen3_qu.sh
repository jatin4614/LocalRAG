#!/usr/bin/env bash
# Pre-cache Qwen3-4B-Instruct-2507-AWQ-4bit weights into volumes/models/.
#
# Plan B Phase 4.2 / Appendix B.A.1.
#
# Run on the deploy host while still connected to the internet. The cached
# directory is mounted into orgchat-vllm-qu read-only at runtime, so the
# container can run with HF_HUB_OFFLINE=1 thereafter.
#
# Usage:
#     ./scripts/stage_qwen3_qu.sh
#
# Environment overrides:
#     CACHE_DIR  — host cache root (default: <repo>/volumes/models)
#     MODEL_ID   — HF repo id (default: cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit)
#     HF_TOKEN   — required only if pulling from a gated mirror; the cpatonn
#                  build is public, so this is optional in the default flow.
set -euo pipefail

# Resolve repo root from this script's location so the script works regardless
# of the user's PWD (CI, cron, operator shell, etc.).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd -P)"

CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/volumes/models}"
MODEL_ID="${MODEL_ID:-cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit}"

# The cpatonn AWQ build is a public repo, so HF_TOKEN is optional. We surface
# it explicitly so users on a gated mirror don't get a confusing 401 and can
# provide a token without re-reading the runbook.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[stage_qwen3_qu] HF_TOKEN not set — assuming public repo. " \
       "Set HF_TOKEN if your mirror is gated." >&2
fi

if [[ ! -d "$CACHE_DIR" ]]; then
  echo "[stage_qwen3_qu] Creating cache dir $CACHE_DIR"
  mkdir -p "$CACHE_DIR"
fi

# Prefer the new `hf` CLI (huggingface_hub >= 1.0); fall back to the legacy
# huggingface-cli alias for older environments.
if command -v hf &>/dev/null; then
  CLI=(hf download "$MODEL_ID" --cache-dir "$CACHE_DIR")
elif command -v huggingface-cli &>/dev/null; then
  CLI=(huggingface-cli download "$MODEL_ID" --cache-dir "$CACHE_DIR")
else
  echo "[stage_qwen3_qu] ERROR: neither 'hf' nor 'huggingface-cli' found on PATH." >&2
  echo "  Install with: pip install --user 'huggingface_hub>=1.0'" >&2
  exit 2
fi

echo "[stage_qwen3_qu] Downloading $MODEL_ID into $CACHE_DIR (~3.5 GB)"
HF_HOME="$CACHE_DIR" "${CLI[@]}"

# The cache dir name HF uses is models--<org>--<name> with org/name double-dashed.
SAFE_ID="${MODEL_ID/\//--}"
CACHED_REPO_DIR="$CACHE_DIR/models--$SAFE_ID"

echo
echo "[stage_qwen3_qu] Cache size after download:"
du -sh "$CACHED_REPO_DIR"

echo
echo "[stage_qwen3_qu] Files staged:"
ls -lh "$CACHED_REPO_DIR/snapshots/"*/ 2>/dev/null || ls -lh "$CACHED_REPO_DIR"

echo
echo "[stage_qwen3_qu] Done. Next: bring up vllm-qu via:"
echo "    cd compose && docker compose up -d vllm-qu"
