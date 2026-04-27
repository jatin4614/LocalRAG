#!/usr/bin/env python3
"""Resume an interrupted E2E run.

Loads /tmp/e2e_results.json from a prior run, identifies queries with
no telemetry (None final_intent + None regex_label → server was likely
unresponsive), and re-runs ONLY those against the same chat IDs.

The merged results overwrite /tmp/e2e_results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from run_e2e import QUERIES, hit_query, prom_snapshot


def main() -> None:
    state = json.loads(Path("/tmp/e2e_results.json").read_text())
    chats = state["chats"]
    results = state["results"]

    needs_redo = [
        i for i, r in enumerate(results)
        if r.get("final_intent") is None
        and r.get("regex_label") is None
        and r.get("intent_from_hits") is None
    ]
    print(f"Re-running {len(needs_redo)}/{len(results)} queries that had no telemetry")
    if not needs_redo:
        return

    for idx in needs_redo:
        expected, query = QUERIES[idx]
        chat_id = chats[expected]
        print(f"  [{idx + 1:02d}/40] {expected:14} | {query[:62]}", flush=True)
        try:
            r = hit_query(chat_id, query)
            r["expected_intent"] = expected
            actual = r.get("final_intent") or r.get("intent_from_hits")
            r["actual_intent"] = actual
            r["correct_routing"] = (actual == expected)
            results[idx] = r
            print(
                f"           regex={r.get('regex_label')!s:14} "
                f"llm={r.get('llm_label')!s:14} "
                f"actual={actual!s:14} "
                f"hits={r.get('real_hits_count'):2d} "
                f"ms={r.get('final_total_ms') or r.get('sse_total_ms')!s:>5} "
                f"corr={'Y' if r['correct_routing'] else 'N'}"
            )
        except Exception as e:
            print(f"           ERR: {e}")
            results[idx] = {
                "query": query, "expected_intent": expected,
                "harness_error": str(e),
            }
        time.sleep(0.5)

    state["results"] = results
    state["prom_after"] = prom_snapshot()
    state["resumed_at"] = "2026-04-27T05:43:00Z"
    Path("/tmp/e2e_results.json").write_text(json.dumps(state, indent=2, default=str))
    print(f"Updated /tmp/e2e_results.json — {len(results)} results")


if __name__ == "__main__":
    main()
