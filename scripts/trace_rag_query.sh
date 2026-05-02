#!/bin/bash
# Watch open-webui logs for the next "rag: request started" through
# the "rag_query" event line and emit the full per-stage diagnostic.
#
# Usage: ./scripts/trace_rag_query.sh [<label>]
#
# Tail starts at "now" so the next user query is captured. Stops when
# the matching ``rag_query`` log line appears (one full request).

set -e
LABEL="${1:-trial}"
echo "[trace] watching for next request — label=${LABEL}"
echo "[trace] (now: $(date -u +%H:%M:%S) UTC). Issue your query in the UI now."
echo "----------------------------------------------------------------"

docker compose -p orgchat logs --since 0s -f open-webui 2>&1 \
  | awk -v LABEL="$LABEL" '
    /rag: request started/ && !started {
      started = 1
      print "[" LABEL "] STARTED:  " $0
      next
    }
    started == 1 && /ext\.services\.chat_rag_bridge/ {
      print "[" LABEL "] BRIDGE:   " $0
    }
    started == 1 && /qu_shadow|llm_label|llm_resolved/ {
      print "[" LABEL "] QU:       " $0
    }
    started == 1 && /multi-entity decompose/ {
      print "[" LABEL "] DECOMP:   " $0
    }
    started == 1 && /mmr_rerank failed|mmr fail-trim|context_expand cap/ {
      print "[" LABEL "] EVENT:    " $0
    }
    started == 1 && /tei\/embed.*424|out of memory|OOM/ {
      print "[" LABEL "] OOM:      " $0
    }
    started == 1 && /rag_query/ {
      print "[" LABEL "] DONE:     " $0
      exit 0
    }
  '
