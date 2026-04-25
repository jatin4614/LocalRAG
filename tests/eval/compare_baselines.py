#!/usr/bin/env python3
"""Compare pre-1a vs post-1a baseline JSONs and print a delta report.

Usage:
    python tests/eval/compare_baselines.py \\
        --pre tests/eval/results/baseline-pre-phase1a.json \\
        --post tests/eval/results/baseline-post-phase1a.json

Output is plain text (stdout) formatted as a decision table that feeds the
gate criteria in ``docs/rag-phase0-1a-4-execution-plan.md §6``.

Gate for "Phase 1a passes":
  - chunk_recall@10 delta >= +3 pp  AND
  - faithfulness delta >= -2 pp     AND
  - no per-intent bucket regresses chunk_recall by more than 5 pp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pct(x: float) -> str:
    return f"{x*100:+6.2f}pp" if x is not None else "   n/a"


def _val(x: float) -> str:
    return f"{x:7.4f}" if x is not None else "    n/a"


def _load(p: Path) -> dict:
    with p.open() as f:
        return json.load(f)


def _delta(pre: dict, post: dict, key_path: list[str]):
    def dig(d, keys):
        for k in keys:
            if d is None or k not in d:
                return None
            d = d[k]
        return d
    pre_v = dig(pre, key_path)
    post_v = dig(post, key_path)
    if pre_v is None or post_v is None:
        return pre_v, post_v, None
    try:
        return pre_v, post_v, post_v - pre_v
    except TypeError:
        return pre_v, post_v, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True)
    args = ap.parse_args()

    pre = _load(args.pre)
    post = _load(args.post)

    print(f"Pre :  {args.pre.name}  label={pre.get('label','-')}")
    print(f"Post:  {args.post.name}  label={post.get('label','-')}")
    print()

    # Aggregate metrics
    print("=" * 72)
    print(f"{'METRIC':<26} {'PRE':>10} {'POST':>10} {'DELTA':>14}")
    print("-" * 72)
    for key in ("chunk_recall@10", "mrr@10", "faithfulness",
                "answer_relevance", "context_precision", "context_recall"):
        a, b, d = _delta(pre, post, ["aggregate", key])
        print(f"{key:<26} {_val(a):>10} {_val(b):>10} {_pct(d):>14}")
    print()

    # Latency
    for key in ("p50", "p95"):
        a, b, d = _delta(pre, post, ["latency_ms", key])
        if a is not None:
            d_str = f"{b-a:+8.1f}ms"
        else:
            d_str = "   n/a"
        print(f"latency_{key:<18} {a or '-':>10} {b or '-':>10} {d_str:>14}")
    print()

    # Per-intent chunk_recall@10
    print("=" * 72)
    print(f"{'PER-INTENT chunk_recall@10':<26} {'PRE':>10} {'POST':>10} {'DELTA':>14}")
    print("-" * 72)
    intents = sorted(
        set((pre.get("per_intent") or {}).keys()) |
        set((post.get("per_intent") or {}).keys())
    )
    per_intent_issues = []
    for intent in intents:
        a, b, d = _delta(pre, post, ["per_intent", intent, "chunk_recall@10"])
        print(f"{intent:<26} {_val(a):>10} {_val(b):>10} {_pct(d):>14}")
        if d is not None and d < -0.05:
            per_intent_issues.append((intent, d))
    print()

    # Per-intent faithfulness
    print("=" * 72)
    print(f"{'PER-INTENT faithfulness':<26} {'PRE':>10} {'POST':>10} {'DELTA':>14}")
    print("-" * 72)
    for intent in intents:
        a, b, d = _delta(pre, post, ["per_intent", intent, "faithfulness"])
        print(f"{intent:<26} {_val(a):>10} {_val(b):>10} {_pct(d):>14}")
    print()

    # Gate verdict
    print("=" * 72)
    print("GATE VERDICT (plan §6 decision criteria)")
    print("-" * 72)
    _, _, d_recall = _delta(pre, post, ["aggregate", "chunk_recall@10"])
    _, _, d_faith = _delta(pre, post, ["aggregate", "faithfulness"])

    recall_ok = d_recall is not None and d_recall >= 0.03
    faith_ok = d_faith is not None and d_faith >= -0.02
    per_intent_ok = not per_intent_issues

    print(f"  chunk_recall@10 delta >= +3pp     : {'PASS' if recall_ok else 'FAIL'}  ({_pct(d_recall)})")
    print(f"  faithfulness delta >= -2pp        : {'PASS' if faith_ok else 'FAIL'}  ({_pct(d_faith)})")
    if per_intent_ok:
        print(f"  no per-intent recall -5pp regress : PASS")
    else:
        print(f"  no per-intent recall -5pp regress : FAIL")
        for intent, d in per_intent_issues:
            print(f"    - {intent}: {_pct(d)}")

    all_pass = recall_ok and faith_ok and per_intent_ok
    print()
    print(f"OVERALL: {'GATES PASSED — Phase 1a validated' if all_pass else 'GATES FAILED — investigate before proceeding'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
