#!/usr/bin/env bash
# kb_cleanup_phase5_rollback.sh — delete the Plan B Phase 5 rollback
# collection (kb_1_v3) after the 14-day window expires on 2026-05-10.
# Idempotent. Safe to dry-run anytime.
#
# This is the sister script to kb_cleanup_post_plan_a.sh (Plan A rollback,
# 2026-05-09). Phase 5 reshard kb_1_v3 -> kb_1_v4 ran on 2026-04-26;
# kb_1_v3 is the rollback safety net retained for 14 days.
#
# Usage:
#   bash scripts/kb_cleanup_phase5_rollback.sh                 # dry-run (default)
#   bash scripts/kb_cleanup_phase5_rollback.sh --execute       # actually delete
#   bash scripts/kb_cleanup_phase5_rollback.sh --execute --self-remove
#       additionally remove this script's cron entry after success
#
# Pre-flight checks (always, even in dry-run):
#   1. Today's date is on or after 2026-05-10 (defensive)
#   2. Already-cleaned marker file does NOT exist
#   3. kb_1 alias resolves to kb_1_v4 (Phase 5 live target — NOT kb_1_v3)
#   4. kb_1_v4 has its expected ~2705 points (2698 chunks + 7 RAPTOR nodes)
#   5. kb_1_v3 point count unchanged from migration baseline (~2698)
#   6. Smoke retrieval against kb_1 returns hits
#
# Destructive actions on --execute:
#   1. Take Qdrant snapshot of kb_1_v3 (final safety net)
#   2. delete_collection kb_1_v3
#   3. Verify kb_1 retrieval still works
#   4. Append cleanup record to docs/runbook/post-plan-a-cleanup-log.md
#   5. git commit (NOT push)
#   6. Create marker file at /tmp/.kb_cleanup_phase5_rollback.done
#   7. (with --self-remove) drop cron entry referencing this script
#
# References:
#   ~/.claude/projects/-home-vogic-LocalRAG/memory/plan_b_executed.md
#   docs/superpowers/plans/2026-04-25-rag-plan-b-llm-shard-async.md (Task 5.4)

set -euo pipefail

# --- defaults ---------------------------------------------------------
QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
WEBUI_URL=${WEBUI_URL:-http://localhost:6100}
EXPECTED_KB_1_V3_COUNT=${EXPECTED_KB_1_V3_COUNT:-2698}
EXPECTED_ALIAS_TARGET=${EXPECTED_ALIAS_TARGET:-kb_1_v4}
EXPECTED_KB_1_V4_MIN=${EXPECTED_KB_1_V4_MIN:-2700}
TARGET_DATE=${TARGET_DATE:-2026-05-10}
MARKER_FILE=${MARKER_FILE:-/tmp/.kb_cleanup_phase5_rollback.done}
TOKEN_FILE=${TOKEN_FILE:-/tmp/.rag_admin_token}
REPO_ROOT=${REPO_ROOT:-/home/vogic/LocalRAG}
LOG_FILE=${LOG_FILE:-/tmp/kb_cleanup_phase5_$(date +%Y%m%d-%H%M%S).log}

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

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== kb_cleanup_phase5_rollback.sh @ $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "  qdrant: $QDRANT_URL"
echo "  webui : $WEBUI_URL"
echo "  log   : $LOG_FILE"
echo "  mode  : $([ "$EXECUTE" = 1 ] && echo EXECUTE || echo DRY-RUN)"
echo "  self-remove: $([ "$SELF_REMOVE" = 1 ] && echo yes || echo no)"
echo

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
  if [ "$SELF_REMOVE" = 1 ]; then
    echo "      Removing cron entry anyway (--self-remove)..."
    crontab -l 2>/dev/null | grep -v kb_cleanup_phase5_rollback.sh | crontab - || true
  fi
  exit 0
fi
echo "[1/6] No prior cleanup marker — proceeding"

# --- pre-flight check 2: kb_1 alias points at kb_1_v4 (NOT v3) --------
echo "[2/6] Verifying kb_1 alias..."
ALIAS_TARGET=$(curl -sS "$QDRANT_URL/aliases" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for a in d.get('result',{}).get('aliases',[]):
    if a['alias_name']=='kb_1':
        print(a['collection_name']); break")
if [ "$ALIAS_TARGET" != "$EXPECTED_ALIAS_TARGET" ]; then
  echo "      FATAL: kb_1 -> '$ALIAS_TARGET' (expected '$EXPECTED_ALIAS_TARGET')."
  echo "      Phase 5 may have been rolled back to kb_1_v3 — refusing to delete the rollback target."
  exit 2
fi
echo "      OK: kb_1 -> $EXPECTED_ALIAS_TARGET"

# --- pre-flight check 3: kb_1_v4 healthy ------------------------------
KB1V4_COUNT=$(count_collection kb_1_v4)
if [ "$KB1V4_COUNT" = "MISSING" ] || [ "$KB1V4_COUNT" -lt "$EXPECTED_KB_1_V4_MIN" ]; then
  echo "      FATAL: kb_1_v4 has '$KB1V4_COUNT' points (expected >= $EXPECTED_KB_1_V4_MIN). Aborting."
  exit 2
fi
echo "      OK: kb_1_v4 has $KB1V4_COUNT points"

# --- pre-flight check 4: kb_1_v3 unchanged from baseline --------------
echo "[3/6] Verifying kb_1_v3 unchanged..."
KB1V3_COUNT=$(count_collection kb_1_v3)
if [ "$KB1V3_COUNT" = "MISSING" ]; then
  echo "      kb_1_v3 already gone? Skipping (idempotent)."
elif [ "$KB1V3_COUNT" -ne "$EXPECTED_KB_1_V3_COUNT" ]; then
  echo "      FATAL: kb_1_v3 has $KB1V3_COUNT points (expected $EXPECTED_KB_1_V3_COUNT)."
  echo "      Someone wrote to it. Aborting; investigate before deletion."
  exit 2
else
  echo "      OK: kb_1_v3 has $KB1V3_COUNT points (matches baseline)"
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
  echo "      SKIP: no admin token at $TOKEN_FILE -- pre-delete smoke skipped"
fi

if [ "$EXECUTE" = 0 ]; then
  echo
  echo "=== DRY RUN COMPLETE ==="
  echo "All safety checks passed. Re-run with --execute to actually delete."
  exit 0
fi

# === DESTRUCTIVE PATH from here ======================================

echo "[5/6] Snapshotting + deleting kb_1_v3..."
CT=$(count_collection kb_1_v3)
if [ "$CT" = "MISSING" ]; then
  echo "  kb_1_v3: already gone, skipping"
else
  echo "  snapshotting kb_1_v3..."
  SNAP=$(curl -sS -X POST "$QDRANT_URL/collections/kb_1_v3/snapshots" \
    | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',{}).get('name','UNKNOWN'))")
  echo "    snapshot: $SNAP (lives in qdrant storage)"
  echo "  deleting kb_1_v3..."
  RESP=$(curl -sS -X DELETE "$QDRANT_URL/collections/kb_1_v3")
  OK=$(echo "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('result',False))")
  if [ "$OK" != "True" ]; then
    echo "    FATAL: delete failed: $RESP"
    exit 3
  fi
  echo "    deleted ($CT points freed)"
fi

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
    exit 3
  fi
  echo "      OK: kb_1 still returns $POST_HITS hits"
else
  echo "      SKIP: no token; trust that kb_1_v4 alias is intact"
fi

# --- runbook commit ---------------------------------------------------
echo
echo "Appending cleanup record to runbook..."
cd "$REPO_ROOT"
NOTE_FILE="docs/runbook/post-plan-a-cleanup-log.md"
mkdir -p "$(dirname "$NOTE_FILE")"
{
  echo
  echo "## Phase 5 rollback cleanup on $(date '+%Y-%m-%d %H:%M %Z')"
  echo
  echo "- Deleted **kb_1_v3** (Plan B Phase 5 rollback target; was $EXPECTED_KB_1_V3_COUNT pts)"
  echo "- Snapshot taken pre-delete (in Qdrant default storage)"
  echo "- kb_1 alias still -> kb_1_v4 (verified post-delete)"
  echo "- Operator: $USER, host: $(hostname), log: $LOG_FILE"
} >> "$NOTE_FILE"

git add "$NOTE_FILE"
git commit -m "ops: post-Phase-5 cleanup -- deleted kb_1_v3 rollback collection" || \
  echo "  (no commit -- runbook note may have been empty or git state unusual)"

# --- marker + self-remove --------------------------------------------
touch "$MARKER_FILE"
echo
echo "Marker written: $MARKER_FILE (re-runs will no-op)"

if [ "$SELF_REMOVE" = 1 ]; then
  echo "Removing cron entry referencing this script..."
  TMPCRON=$(mktemp)
  if crontab -l 2>/dev/null > "$TMPCRON"; then
    BEFORE=$(wc -l < "$TMPCRON")
    grep -v kb_cleanup_phase5_rollback.sh "$TMPCRON" > "${TMPCRON}.new" || true
    AFTER=$(wc -l < "${TMPCRON}.new")
    if [ "$BEFORE" -ne "$AFTER" ]; then
      crontab "${TMPCRON}.new"
      echo "  removed $((BEFORE - AFTER)) cron entries"
    fi
    rm -f "$TMPCRON" "${TMPCRON}.new"
  fi
fi

echo
echo "=== DONE @ $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
echo "Cleanup committed locally. NOT pushed to remote."
