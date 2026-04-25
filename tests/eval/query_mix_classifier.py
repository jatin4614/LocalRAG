#!/usr/bin/env python3
"""Pattern-based query-intent classifier (Phase 0 infrastructure).

Buckets a query string into one of five labels used by the golden set and
the execution plan:

    metadata     — "list all reports", "how many docs", enumerations over filenames.
    global       — "summarize trends", "compare X to Y", cross-doc aggregations.
    specific     — content-anchored questions; the default.
    multihop     — "reviewing both A and B", "combine X with Y", multi-doc synthesis.
    adversarial  — OOD, typo-heavy, or empty prompts.

This is cheap string + regex matching — no embeddings, no LLM. It is wired
up so Phase 2 can feed a cron-collected sample of production queries through
it, and compute a mix ratio that gates the decision to build intent routing.
No prod query data is wired yet; the classifier plus unit tests ship first.

Usage::

    python -m tests.eval.query_mix_classifier --stdin     # classify one query per line
    python -m tests.eval.query_mix_classifier --golden tests/eval/golden_human.jsonl

Library use::

    from tests.eval.query_mix_classifier import classify
    classify("list all April reports")  # -> "metadata"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# Order matters: checks are tried top-to-bottom; first match wins.
# Keep patterns small and auditable — this is a proxy, not ground truth.
_ADVERSARIAL_PATTERNS = [
    # Garbage-looking very-short strings dominated by a single run of letters.
    re.compile(r"^\s*[a-zA-Z]{1,6}\s*$"),
    # "tell me something interesting" / "something interesting" / "anything"
    re.compile(r"^\s*(?:tell me\s+)?(?:something|anything)\s+(?:interesting|cool|weird|fun)\s*$", re.I),
    # All-punctuation or all-digits
    re.compile(r"^\s*[\W\d_]+\s*$"),
]

# Token-level distinguisher for adversarial queries that are longer.
_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "at", "and", "or",
    "with", "from", "by", "about", "what", "when", "where", "who", "how",
    "why", "is", "are", "was", "were", "be", "been", "do", "does", "did",
}


_METADATA_TRIGGERS = [
    # Enumeration / listing intent over filenames, dates, docs.
    re.compile(r"\blist (?:all|every|the)\b", re.I),
    re.compile(r"\bwhich (?:reports?|documents?|files?|dates?)\b", re.I),
    re.compile(r"\bwhat (?:reports?|documents?|files?) (?:are|do you have|exist)\b", re.I),
    re.compile(r"\bhow many (?:reports?|documents?|files?|pages?|chunks?)\b", re.I),
    re.compile(r"\bavailable (?:reports?|documents?|files?|dates?)\b", re.I),
    re.compile(r"\b(?:earliest|latest|most recent|oldest)\b.*(?:report|document|file|date)\b", re.I),
    re.compile(r"\benumer(?:ate|ation)\b", re.I),
    re.compile(r"\bfilename\b", re.I),
    re.compile(r"\bdoes (?:a|the|an) (?:report|document|file) exist\b", re.I),
    re.compile(r"\breports? from\b", re.I),
    re.compile(r"\breports? in (?:January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I),
]

_GLOBAL_TRIGGERS = [
    # Aggregation / summary / compare across many docs.
    re.compile(r"\bsummari[sz]e (?:the\s+)?trends?\b", re.I),
    re.compile(r"\bcompare\b.*\b(?:and|vs|versus|between|with)\b", re.I),
    re.compile(r"\boverall (?:status|trend|summary)\b", re.I),
    re.compile(r"\bacross (?:all |the |every )?(?:reports?|documents?|days|months|weeks|files?)\b", re.I),
    re.compile(r"\bover (?:the\s+)?(?:past|last)\s+\d+\s+(?:days|weeks|months)\b", re.I),
    re.compile(r"\btop\s+\d+\b", re.I),
    re.compile(r"\bwhat are (?:the\s+)?(?:common|recurring|main|key)\b", re.I),
    re.compile(r"\bfrequen(?:cy|tly)\b", re.I),
    re.compile(r"\bmost (?:common|frequent|reported)\b", re.I),
    re.compile(r"\btrends? (?:in|for|of)\b", re.I),
]

_MULTIHOP_TRIGGERS = [
    # Multi-doc synthesis signals.
    re.compile(r"\breview(?:ing)? both\b", re.I),
    re.compile(r"\bcombin(?:e|ing)\b.*\b(?:with|and)\b", re.I),
    re.compile(r"\bacross (?:the\s+)?reports?\b.*\b(?:first|last|evolved|evolution|progression)\b", re.I),
    re.compile(r"\bevolv(?:e|ed|ing)\b.*\bbetween\b", re.I),
    re.compile(r"\bhow (?:did|does)\b.*\b(?:evolve|change|progress)\b", re.I),
    re.compile(r"\bbetween\b.*\band\b.*\b(?:reports?|documents?|dates?)\b", re.I),
    re.compile(r"\bprogress(?:ion)?\b.*\b(?:across|between|from)\b", re.I),
    re.compile(r"\bcompile a list of\b", re.I),
]


def _looks_like_typo_soup(query: str) -> bool:
    """Heuristic: lots of non-dictionary-looking tokens and few English stopwords.

    Catches things like 'Cmn update sprk fir incdnt NC Pas' where consonant
    clusters and missing vowels dominate. A token counts as "consonant-heavy"
    if length >= 4 and vowels/length < 0.25 — tight enough to skip English
    words like "list" (len 4, 1 vowel = 0.25) and "reports" (len 7, 2 vowels
    = 0.29) but catches "sprk" (0.0) and "incdnt" (0.17). Short 3-letter
    abbreviations like "Cmn", "Pas" also count regardless of vowel count
    when they lack any vowels.
    """
    q = query.strip()
    if not q:
        return True
    tokens = re.findall(r"[A-Za-z]+", q.lower())
    if len(tokens) < 2:
        return False
    stopword_hits = sum(1 for t in tokens if t in _STOP_WORDS)

    def is_consonant_heavy(t: str) -> bool:
        if t in _STOP_WORDS:
            return False
        vowels = len(re.findall(r"[aeiou]", t))
        if len(t) >= 3 and vowels == 0:
            return True
        if len(t) >= 4 and vowels / len(t) < 0.25:
            return True
        return False

    consonant_heavy = sum(1 for t in tokens if is_consonant_heavy(t))
    short_ratio = consonant_heavy / max(1, len(tokens))
    # No stopwords AND ≥ 40% consonant-heavy, or mostly consonant-heavy with
    # at most one stopword. Deliberately strict — false positives here mean
    # we refuse to classify a real query, which is worse than missing some
    # true typo-soup cases.
    if stopword_hits == 0 and short_ratio >= 0.4:
        return True
    if stopword_hits <= 1 and short_ratio >= 0.6:
        return True
    return False


def classify(query: str) -> str:
    """Return one of ``{metadata, global, specific, multihop, adversarial}``.

    Pure string-pattern logic — no embeddings, no LLM. Order of checks:
      1. ``adversarial`` — empty / single-token / typo-soup / all-punct.
      2. ``metadata`` — enumeration, counts, filename lookup.
      3. ``global`` — cross-doc aggregation, summarization.
      4. ``multihop`` — multi-doc synthesis.
      5. ``specific`` — default (content-anchored).
    """
    if query is None:
        return "adversarial"
    q = query.strip()
    if not q:
        return "adversarial"

    for pat in _ADVERSARIAL_PATTERNS:
        if pat.match(q):
            return "adversarial"
    if _looks_like_typo_soup(q):
        return "adversarial"

    for pat in _METADATA_TRIGGERS:
        if pat.search(q):
            return "metadata"

    for pat in _MULTIHOP_TRIGGERS:
        if pat.search(q):
            return "multihop"

    for pat in _GLOBAL_TRIGGERS:
        if pat.search(q):
            return "global"

    return "specific"


def classify_batch(queries):
    """Classify an iterable of queries — returns list[tuple(query, label)]."""
    return [(q, classify(q)) for q in queries]


def _unit_tests() -> int:
    """Compact self-test. Returns 0 on pass, 1 on any failure.

    Exercised via ``python -m tests.eval.query_mix_classifier --test`` and by
    the CI caller in scheduled_eval (once wired in Phase 4).
    """
    cases = [
        # --- adversarial ---
        ("asdf", "adversarial"),
        ("tell me something interesting", "adversarial"),
        ("", "adversarial"),
        ("???", "adversarial"),
        ("Cmn update sprk fir incdnt NC Pas", "adversarial"),
        # --- metadata ---
        ("list all April reports", "metadata"),
        ("list every report filename that contains '18 Apr'", "metadata"),
        ("how many reports are in the knowledge base?", "metadata"),
        ("which reports are from March 2026?", "metadata"),
        ("does a report exist for 15 January 2026?", "metadata"),
        ("what is the most recent report available?", "metadata"),
        # --- global ---
        ("summarize the trends in BDE TO BN OUTAGE across February", "global"),
        ("compare March and April risk trends", "global"),
        ("describe trends in SAMBHAV outage counts from early April through mid April", "global"),
        ("what are the most recurring issues across the reports", "global"),
        # --- multihop ---
        ("reviewing both the 02 Jan and 03 Jan reports, what STM issues were reported", "multihop"),
        ("combine Ex GAGAN BHUMIKA with Ex GAGAN BHED", "multihop"),
        ("how did the Tangdhar to Shararat OFC break evolve between January and April 2026", "multihop"),
        ("compile a list of all BBR trials conducted between 02 Jan and 09 Jan 2026", "multihop"),
        # --- specific (default) ---
        ("what phone number was configured for the SIP phone at P Gali on 05 Jan 2026", "specific"),
        ("how many PCs were audited on 01 Jan 2026", "specific"),
        ("what is the docket number for the MKBT Terminal 068 complaint", "specific"),
    ]
    fails = []
    for q, expected in cases:
        got = classify(q)
        if got != expected:
            fails.append((q, expected, got))
    if fails:
        print("FAIL — query_mix_classifier self-test", file=sys.stderr)
        for q, exp, got in fails:
            print(f"  query={q!r}  expected={exp!r}  got={got!r}", file=sys.stderr)
        return 1
    print(f"pass {len(cases)} cases", file=sys.stderr)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stdin", action="store_true", help="read queries from stdin, one per line")
    p.add_argument("--golden", default=None, help="classify queries from a JSONL golden file")
    p.add_argument("--test", action="store_true", help="run compact self-test")
    p.add_argument("--out", default=None, help="optional JSON summary path")
    return p.parse_args(argv)


def main(argv=None) -> int:
    ns = parse_args(argv)
    if ns.test:
        return _unit_tests()

    queries: list[str] = []
    if ns.stdin:
        queries.extend(line.rstrip("\n") for line in sys.stdin if line.strip())
    if ns.golden:
        gp = Path(ns.golden)
        if not gp.exists():
            print(f"golden file not found: {gp}", file=sys.stderr)
            return 2
        for line in gp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"WARN: bad JSONL line: {e}", file=sys.stderr)
                continue
            q = row.get("query") if isinstance(row, dict) else None
            if q:
                queries.append(q)

    if not queries:
        print("no queries to classify — pass --stdin or --golden", file=sys.stderr)
        return 2

    results = classify_batch(queries)
    counts = Counter(lbl for _, lbl in results)
    for q, lbl in results:
        print(f"{lbl:<11}  {q}")
    print(
        "\nsummary: "
        + " ".join(f"{lbl}={counts.get(lbl, 0)}" for lbl in ("specific", "global", "metadata", "multihop", "adversarial")),
        file=sys.stderr,
    )
    if ns.out:
        out_path = Path(ns.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "n_queries": len(queries),
                    "mix": dict(counts),
                    "labels": [{"query": q, "label": lbl} for q, lbl in results],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
