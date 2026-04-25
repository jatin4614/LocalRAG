#!/usr/bin/env bash
# Daily eval cron — run via crontab on the deploy host:
#   5 3 * * * /home/vogic/LocalRAG/scripts/daily_eval_cron.sh
# Emits retrieval_ndcg_daily gauge via node-exporter textfile collector.

set -euo pipefail

REPO="/home/vogic/LocalRAG"
OUT_DIR="/var/lib/node_exporter/textfile_collector"
TMP_JSON="/tmp/daily_eval.json"
mkdir -p "$OUT_DIR"

cd "$REPO"
python -m tests.eval.harness \
  --golden tests/eval/golden_daily_subset.jsonl \
  --kb-id "${KB_EVAL_ID:-1}" \
  --api-base-url "${API_BASE:-http://localhost:6100}" \
  --out "$TMP_JSON"

# Convert to prom textfile format (one metric per line)
python - <<PY > "$OUT_DIR/retrieval_daily.prom"
import json
r = json.load(open("$TMP_JSON"))
ts = int(__import__("time").time())
print(f'# HELP retrieval_ndcg_daily Daily eval nDCG@10 per intent stratum')
print(f'# TYPE retrieval_ndcg_daily gauge')
for intent, agg in r.get("by_intent", {}).items():
    if agg.get("n", 0) == 0: continue
    v = agg.get("ndcg@10") or 0.0
    print(f'retrieval_ndcg_daily{{intent="{intent}"}} {v:.4f} {ts}000')
g = r["global"]
print(f'retrieval_ndcg_daily{{intent="__global__"}} {g.get("ndcg@10") or 0.0:.4f} {ts}000')
print(f'# HELP retrieval_daily_latency_p95_ms Daily eval p95 retrieval latency')
print(f'# TYPE retrieval_daily_latency_p95_ms gauge')
print(f'retrieval_daily_latency_p95_ms {g.get("p95_latency_ms") or 0.0:.2f} {ts}000')
PY

echo "daily eval emitted $(date)"
