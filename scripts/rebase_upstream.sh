#!/usr/bin/env bash
# Rebase upstream to a new Open WebUI tag, reapply patches.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-tag>    e.g. $0 v0.7.0" >&2
  exit 2
fi
NEW_TAG="$1"
cd "$(dirname "$0")/.."

echo ">> fetching upstream"
(cd upstream && git fetch --tags origin && git checkout "$NEW_TAG")

echo ">> staging new gitlink in parent repo"
git add upstream

echo ">> updating UPSTREAM_VERSION"
echo "$NEW_TAG" > UPSTREAM_VERSION

echo ">> reapplying patches"
for p in patches/*.patch; do
  [[ -e "$p" ]] || continue
  echo ">>   applying $p"
  git -C upstream apply --3way "$(realpath "$p")" || {
    echo "!! patch $p failed to apply cleanly; resolve manually then rerun"
    exit 1
  }
done

echo ">> done. Review, test, commit."
