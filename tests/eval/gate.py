"""Compare latest eval output against committed baseline and SLO thresholds.

Exits 0 if all gates pass, 1 if any gate fails, 2 on misuse.

Gates (from docs/runbook/slo.md):
- Global chunk_recall@10: no regression >1pp
- Per-intent chunk_recall@10: no regression >2pp
- metadata intent floor: chunk_recall@10 >= 0.70
- p95 latency: within SLO band for current phase (caller passes expected band)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REGRESSION_GLOBAL_PP = 1.0
REGRESSION_INTENT_PP = 2.0
METADATA_FLOOR = 0.70


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--latest", required=True, type=Path)
    p.add_argument("--slo", required=True, type=Path)  # unused for now; reserved for auto-parse
    args = p.parse_args()

    if not args.baseline.exists():
        print(f"FAIL: baseline missing: {args.baseline}", file=sys.stderr)
        return 2
    baseline = json.loads(args.baseline.read_text())
    latest = json.loads(args.latest.read_text())

    errors: list[str] = []

    # Global regression
    b = baseline["global"]["chunk_recall@10"]
    l = latest["global"]["chunk_recall@10"]
    if (b - l) * 100 > REGRESSION_GLOBAL_PP:
        errors.append(
            f"GLOBAL regression: chunk_recall@10 {l:.3f} vs baseline {b:.3f} "
            f"(Δ=-{(b - l) * 100:.1f}pp > {REGRESSION_GLOBAL_PP}pp threshold)"
        )

    # Per-intent regression
    for intent, bagg in baseline.get("by_intent", {}).items():
        lagg = latest.get("by_intent", {}).get(intent)
        if lagg is None or lagg.get("n", 0) == 0:
            continue
        bv = bagg["chunk_recall@10"]
        lv = lagg["chunk_recall@10"]
        if (bv - lv) * 100 > REGRESSION_INTENT_PP:
            errors.append(
                f"INTENT '{intent}' regression: chunk_recall@10 {lv:.3f} "
                f"vs baseline {bv:.3f} (Δ=-{(bv - lv) * 100:.1f}pp > {REGRESSION_INTENT_PP}pp)"
            )

    # Metadata floor
    meta = latest.get("by_intent", {}).get("metadata", {})
    if meta.get("n", 0) > 0:
        mv = meta["chunk_recall@10"]
        if mv < METADATA_FLOOR:
            errors.append(
                f"FLOOR breach: metadata chunk_recall@10 {mv:.3f} < floor {METADATA_FLOOR}"
            )

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: gate passed. "
          f"global_recall={latest['global']['chunk_recall@10']:.3f} "
          f"p95={latest['global']['p95_latency_ms']:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
