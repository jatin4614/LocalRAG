#!/usr/bin/env python3
"""Analyze QU LLM shadow-mode log entries (Plan B Phase 4.8).

Reads JSON-line log entries from a file (or stdin) and prints:

  - Total queries observed
  - Agreement rate overall and per regex_label
  - Disagreement examples (sampled) per (regex_label, llm_label) bucket
  - Escalation reason breakdown

Each input line is expected to be either:
  * a bare JSON object emitted by :mod:`ext.services.query_intent.classify_with_qu`
    via the ``orgchat.qu_shadow`` logger, OR
  * a logger-prefixed line such as ``"INFO orgchat.qu_shadow: {...}"`` —
    we strip everything before the first ``{`` so docker logs grep output
    works directly.

Usage:
    docker logs orgchat-open-webui 2>&1 \\
        | grep 'orgchat.qu_shadow' \\
        | python scripts/analyze_shadow_log.py

    # or against a file
    python scripts/analyze_shadow_log.py /var/log/openwebui/shadow.log
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from typing import IO, Iterator


def parse_lines(stream: IO[str]) -> Iterator[dict]:
    """Yield one parsed JSON object per non-empty line.

    Skips lines without a JSON object and lines with parse errors so the
    analyzer can be run against mixed (logger-prefixed + bare) input.
    """
    for line in stream:
        line = line.strip()
        if not line or "{" not in line:
            continue
        # Strip any logger prefix before the first { so docker logs output
        # ("INFO orgchat.qu_shadow: {...}") parses cleanly.
        line = line[line.index("{"):]
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "file",
        nargs="?",
        default="-",
        help="path to shadow log; '-' for stdin (default)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=3,
        help="sample this many disagreement queries per (regex_label, llm_label) bucket",
    )
    args = p.parse_args()

    stream: IO[str] = sys.stdin if args.file == "-" else open(args.file)

    total = 0
    by_regex_label: collections.Counter[str] = collections.Counter()
    agreement_by_regex_label: collections.Counter[str] = collections.Counter()
    bucket_samples: dict[tuple, list[str]] = collections.defaultdict(list)
    escalation_counts: collections.Counter[str] = collections.Counter()

    for entry in parse_lines(stream):
        total += 1
        rl = entry.get("regex_label", "unknown")
        by_regex_label[rl] += 1
        if entry.get("agree"):
            agreement_by_regex_label[rl] += 1
        else:
            bucket = (rl, entry.get("llm_label"))
            if len(bucket_samples[bucket]) < args.samples:
                bucket_samples[bucket].append(entry.get("query", ""))
        escalation_counts[entry.get("escalation_reason", "none")] += 1

    if total == 0:
        print("No shadow log entries found.", file=sys.stderr)
        return 1

    print(f"Total queries: {total}")
    print()
    print("Per-regex-label agreement:")
    for rl, count in by_regex_label.most_common():
        agree = agreement_by_regex_label[rl]
        rate = agree / count * 100 if count else 0
        print(f"  {rl:>15}: {agree}/{count} ({rate:.1f}%)")
    print()
    print("Escalation reason breakdown:")
    for reason, count in escalation_counts.most_common():
        pct = count / total * 100
        print(f"  {reason:>20}: {count} ({pct:.1f}%)")
    print()
    print("Disagreement samples (regex_label -> llm_label):")
    for (rl, ll), samples in sorted(
        bucket_samples.items(), key=lambda x: -len(x[1])
    ):
        print(f"  {rl} -> {ll}:")
        for q in samples:
            print(f"    {q!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
