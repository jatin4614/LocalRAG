#!/usr/bin/env bash
# DEPRECATED (2026-04-30): upstream/ is now vendored, not a submodule. The
# patches in patches/*.patch are already applied in the vendored source
# tree; bootstrap.sh no longer calls this script. It is kept only for
# re-deriving deltas if/when Open WebUI is upgraded — initialize a temporary
# git repo inside upstream/ in that case.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/upstream"

if ! git -C "$UP" rev-parse --git-dir &>/dev/null; then
  echo "!! upstream/ is no longer a git repo (vendored 2026-04-30)." >&2
  echo "   Patches are pre-applied. To re-derive patches against an upgraded" >&2
  echo "   upstream, init a temporary git repo inside upstream/, commit the" >&2
  echo "   pristine v0.X.Y, then apply your changes and 'git format-patch'." >&2
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
