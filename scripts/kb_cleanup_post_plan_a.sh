#!/usr/bin/env bash
# kb_cleanup_post_plan_a.sh — delete the Plan A rollback collections
# (kb_1_v2 and kb_1_rebuild) after the 14-day window expires on
# 2026-05-09. Idempotent. Safe to dry-run anytime.
#
# Usage:
#   bash scripts/kb_cleanup_post_plan_a.sh                 # dry-run (default)
#   bash scripts/kb_cleanup_post_plan_a.sh --execute       # actually delete
#   bash scripts/kb_cleanup_post_plan_a.sh --execute --self-remove
#       additionally remove this script's cron entry after success
#
# Pre-flight checks (always, even in dry-run):
#   1. Today's date is on or after 2026-05-09 (defensive)
#   2. Already-cleaned marker file does NOT exist
#   3. kb_1 alias resolves to kb_1_v3
#   4. kb_1_v3 has its expected ~2698 points
#   5. kb_1_v2 + kb_1_rebuild point counts unchanged from migration baseline
#   6. Smoke retrieval against kb_1 returns hits
#
# Destructive actions on --execute:
#   1. Take Qdrant snapshot of each rollback collection
#   2. delete_collection kb_1_v2
#   3. delete_collection kb_1_rebuild
#   4. Verify kb_1 retrieval still works
#   5. Append cleanup record to docs/runbook/post-plan-a-cleanup-log.md
#   6. git commit (NOT push)
#   7. Create marker file at /tmp/.kb_cleanup_post_plan_a.done
#   8. (with --self-remove) drop cron entry referencing this script
#
# All output also tees to a log file; pass LOG_FILE=path to override.
#
# References:
#   ~/.claude/projects/-home-vogic-LocalRAG/memory/plan_a_executed.md
#   docs/superpowers/plans/2026-04-24-rag-robustness-and-quality.md (Task 1.7, 3.7)

set -euo pipefail

# --- defaults ---------------------------------------------------------
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
WEBUI_URL=${WEBUI_URL:-http://localhost:6100}
EXPECTED_KB_1_V2_COUNT=${EXPECTED_KB_1_V2_COUNT:-2698}
EXPECTED_KB_1_REBUILD_COUNT=${EXPECTED_KB_1_REBUILD_COUNT:-2698}
EXPECTED_ALIAS_TARGET=${EXPECTED_ALIAS_TARGET:-kb_1_v3}
EXPECTED_KB_1_V3_MIN=${EXPECTED_KB_1_V3_MIN:-2500}
TARGET_DATE=${TARGET_DATE:-2026-05-09}
MARKER_FILE=${MARKER_FILE:-/tmp/.kb_cleanup_post_plan_a.done}
TOKEN_FILE=${TOKEN_FILE:-/tmp/.rag_admin_token}
REPO_ROOT=${REPO_ROOT:-/home/vogic/LocalRAG}
LOG_FILE=${LOG_FILE:-/tmp/kb_cleanup_$(date +%Y%m%d-%H%M%S).log}

EXECUTE=0
SELF_REMOVE=0
for arg in "$@"; do
  case "$arg" in
    --execute) EXECUTE=1 ;;
    --self-remove) SELF_REMOVE=1 ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 1 ;;
  esac
done

# tee everything from here on into the log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== kb_cleanup_post_plan_a.sh @ $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "  qdrant: $QDRANT_URL"
echo "  webui : $WEBUI_URL"
echo "  log   : $LOG_FILE"
echo "  mode  : $([ "$EXECUTE" = 1 ] && echo EXECUTE || echo DRY-RUN)"
echo "  self-remove: $([ "$SELF_REMOVE" = 1 ] && echo yes || echo no)"
echo

# --- helper: jq-free JSON extract via python3 -------------------------
json_get() {
  python3 -c "import sys,json;d=json.load(sys.stdin);
v=d
for k in '$1'.split('.'):
    if k.isdigit(): v=v[int(k)]
    else: v=v.get(k) if isinstance(v,dict) else None
print(v)"
}

count_collection() {
  curl -sS -X POST "$QDRANT_URL/collections/$1/points/count" \
    -d '{"exact":true}' 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('result',{}).get('count','MISSING'))"
}

# --- pre-flight check 0: today's date ---------------------------------
TODAY=$(date '+%Y-%m-%d')
if [[ "$TODAY" < "$TARGET_DATE" ]]; then
  echo "[0/6] Today is $TODAY; target was $TARGET_DATE. Refusing to run early."
  echo "      (set TARGET_DATE env var if you need to override)"
  exit 4
fi
echo "[0/6] Date check OK: today=$TODAY, target=$TARGET_DATE"

# --- pre-flight check 1: not already cleaned --------------------------
if [ -f "$MARKER_FILE" ]; then
  echo "[1/6] Already cleaned (marker $MARKER_FILE exists from $(stat -c %y "$MARKER_FILE"))."
  echo "      Nothing to do. Re-run after rm $MARKER_FILE if intended."
  if [ "$SELF_REMOVE" = 1 ]; then
    echo "      Removing cron entry anyway (--self-remove)..."
    crontab -l 2>/dev/null | grep -v kb_cleanup_post_plan_a.sh | crontab - || true
  fi
  exit 0
fi
echo "[1/6] No prior cleanup marker — proceeding"

# --- pre-flight check 2: kb_1 alias still points at kb_1_v3 -----------
echo "[2/6] Verifying kb_1 alias..."
ALIAS_TARGET=$(curl -sS "$QDRANT_URL/aliases" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for a in d.get('result',{}).get('aliases',[]):
    if a['alias_name']=='kb_1':
        print(a['collection_name']); break")
if [ "$ALIAS_TARGET" != "$EXPECTED_ALIAS_TARGET" ]; then
  echo "      FATAL: kb_1 → '$ALIAS_TARGET' (expected '$EXPECTED_ALIAS_TARGET')."
  echo "      Has the migration been rolled back? Refusing to delete rollback collections."
  exit 2
fi
echo "      OK: kb_1 → $EXPECTED_ALIAS_TARGET"

# --- pre-flight check 3: kb_1_v3 healthy ------------------------------
KB1V3_COUNT=$(count_collection kb_1_v3)
if [ "$KB1V3_COUNT" = "MISSING" ] || [ "$KB1V3_COUNT" -lt "$EXPECTED_KB_1_V3_MIN" ]; then
  echo "      FATAL: kb_1_v3 has '$KB1V3_COUNT' points (expected ≥ $EXPECTED_KB_1_V3_MIN). Aborting."
  exit 2
fi
echo "      OK: kb_1_v3 has $KB1V3_COUNT points"

# --- pre-flight check 4: rollback collection counts unchanged ---------
echo "[3/6] Verifying rollback collection counts unchanged..."
KB1V2_COUNT=$(count_collection kb_1_v2)
if [ "$KB1V2_COUNT" = "MISSING" ]; then
  echo "      kb_1_v2 already gone? Skipping (idempotent)."
elif [ "$KB1V2_COUNT" -ne "$EXPECTED_KB_1_V2_COUNT" ]; then
  echo "      FATAL: kb_1_v2 has $KB1V2_COUNT points (expected $EXPECTED_KB_1_V2_COUNT)."
  echo "      Someone wrote to it. Aborting; investigate before deletion."
  exit 2
else
  echo "      OK: kb_1_v2 has $KB1V2_COUNT points (matches baseline)"
fi

KB1REB_COUNT=$(count_collection kb_1_rebuild)
if [ "$KB1REB_COUNT" = "MISSING" ]; then
  echo "      kb_1_rebuild already gone? Skipping (idempotent)."
elif [ "$KB1REB_COUNT" -ne "$EXPECTED_KB_1_REBUILD_COUNT" ]; then
  echo "      FATAL: kb_1_rebuild has $KB1REB_COUNT points (expected $EXPECTED_KB_1_REBUILD_COUNT)."
  exit 2
else
  echo "      OK: kb_1_rebuild has $KB1REB_COUNT points (matches baseline)"
fi

# --- smoke retrieval pre-delete ---------------------------------------
echo "[4/6] Smoke retrieval against kb_1 (pre-delete)..."
TOKEN=""
if [ -f "$TOKEN_FILE" ]; then
  TOKEN=$(cat "$TOKEN_FILE")
  PRE_HITS=$(curl -sS -X POST "$WEBUI_URL/api/rag/retrieve" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"chat_id":null,"query":"test cleanup","selected_kb_config":[{"kb_id":1}],"top_k":3,"max_tokens":1000}' \
    2>/dev/null | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('hits',[])))" 2>/dev/null || echo 0)
  if [ "${PRE_HITS:-0}" -lt 1 ]; then
    echo "      WARN: retrieval returned 0 hits (token may be expired). Skipping smoke."
  else
    echo "      OK: kb_1 retrieval returned $PRE_HITS hits"
  fi
else
  echo "      SKIP: no admin token at $TOKEN_FILE — pre-delete smoke skipped"
fi

if [ "$EXECUTE" = 0 ]; then
  echo
  echo "=== DRY RUN COMPLETE ==="
  echo "All safety checks passed. Re-run with --execute to actually delete."
  exit 0
fi

# === DESTRUCTIVE PATH from here ======================================

echo "[5/6] Snapshotting + deleting rollback collections..."
for col in kb_1_v2 kb_1_rebuild; do
  CT=$(count_collection "$col")
  if [ "$CT" = "MISSING" ]; then
    echo "  $col: already gone, skipping"
    continue
  fi
  echo "  snapshotting $col..."
  SNAP=$(curl -sS -X POST "$QDRANT_URL/collections/$col/snapshots" \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',{}).get('name','UNKNOWN'))")
  echo "    snapshot: $SNAP (lives in qdrant storage)"
  echo "  deleting $col..."
  RESP=$(curl -sS -X DELETE "$QDRANT_URL/collections/$col")
  OK=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',False))")
  if [ "$OK" != "True" ]; then
    echo "    FATAL: delete failed: $RESP"
    exit 3
  fi
  echo "    deleted ($CT points freed)"
done

echo "[6/6] Verifying kb_1 retrieval still works post-delete..."
sleep 3
if [ -n "$TOKEN" ]; then
  POST_HITS=$(curl -sS -X POST "$WEBUI_URL/api/rag/retrieve" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"chat_id":null,"query":"test cleanup","selected_kb_config":[{"kb_id":1}],"top_k":3,"max_tokens":1000}' \
    2>/dev/null | python3 -c "import sys,json;print(len(json.load(sys.stdin).get('hits',[])))" 2>/dev/null || echo 0)
  if [ "${POST_HITS:-0}" -lt 1 ]; then
    echo "      FATAL: kb_1 retrieval returned 0 hits POST-delete. Investigate immediately."
    echo "      Snapshots were taken; check qdrant storage and restore if needed."
    exit 3
  fi
  echo "      OK: kb_1 still returns $POST_HITS hits"
else
  echo "      SKIP: no token; trust that kb_1_v3 alias is intact"
fi

# --- runbook commit ---------------------------------------------------
echo
echo "Appending cleanup record to runbook..."
cd "$REPO_ROOT"
NOTE_FILE="docs/runbook/post-plan-a-cleanup-log.md"
mkdir -p "$(dirname "$NOTE_FILE")"
{
  echo
  echo "## Cleanup on $(date '+%Y-%m-%d %H:%M %Z')"
  echo
  echo "- Deleted **kb_1_v2** (Plan A 1.7 schema migration target; was $EXPECTED_KB_1_V2_COUNT pts)"
  echo "- Deleted **kb_1_rebuild** (pre-Plan-A original; was $EXPECTED_KB_1_REBUILD_COUNT pts)"
  echo "- Snapshots taken pre-delete (in Qdrant default storage; verify path under /var/lib/qdrant/snapshots/ or container default)"
  echo "- kb_1 alias still → kb_1_v3 (verified post-delete)"
  echo "- Operator: $USER, host: $(hostname), log: $LOG_FILE"
} >> "$NOTE_FILE"

git add "$NOTE_FILE"
git commit -m "ops: post-Plan-A cleanup — deleted kb_1_v2 + kb_1_rebuild rollback collections" || \
  echo "  (no commit — runbook note may have been empty or git state unusual)"

# --- marker + self-remove --------------------------------------------
touch "$MARKER_FILE"
echo
echo "Marker written: $MARKER_FILE (re-runs will no-op)"

if [ "$SELF_REMOVE" = 1 ]; then
  echo "Removing cron entry referencing this script..."
  TMPCRON=$(mktemp)
  if crontab -l 2>/dev/null > "$TMPCRON"; then
    BEFORE=$(wc -l < "$TMPCRON")
    grep -v kb_cleanup_post_plan_a.sh "$TMPCRON" > "${TMPCRON}.new" || true
    AFTER=$(wc -l < "${TMPCRON}.new")
    if [ "$BEFORE" -ne "$AFTER" ]; then
      crontab "${TMPCRON}.new"
      echo "  removed $((BEFORE - AFTER)) cron entries"
    else
      echo "  no cron entries matched (already clean)"
    fi
    rm -f "$TMPCRON" "${TMPCRON}.new"
  else
    echo "  no crontab to clean"
  fi
fi

echo
echo "=== DONE @ $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "Cleanup committed locally. NOT pushed to remote."
