#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/upstream"

if ! git -C "$UP" rev-parse --git-dir &>/dev/null; then
  echo "!! upstream submodule not initialized — run: git submodule update --init upstream" >&2
  exit 1
fi

shopt -s nullglob
for p in "$ROOT"/patches/*.patch; do
  echo "applying $(basename "$p")"
  if git -C "$UP" apply --check "$p" 2>/dev/null; then
    git -C "$UP" apply "$p"
  else
    if git -C "$UP" apply --check --reverse "$p" 2>/dev/null; then
      echo "  already applied (skipping)"
    else
      echo "!! patch $p does not apply cleanly; upstream changed — regenerate" >&2
      exit 2
    fi
  fi
done
echo "all patches applied."
