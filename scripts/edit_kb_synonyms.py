"""Edit per-KB synonym table.

Usage:
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --list
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --add '["A","B","C"]'
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --remove '["A","B","C"]'
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --load FILE.json
    .venv/bin/python scripts/edit_kb_synonyms.py --kb 2 --load -    # stdin

Env: DATABASE_URL (required)

Spec: docs/superpowers/specs/2026-05-03-retrieval-quality-fix-design.md §5.2.4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg


async def _conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _list(kb_id: int) -> list:
    conn = await _conn()
    try:
        row = await conn.fetchrow(
            "SELECT synonyms FROM knowledge_bases WHERE id = $1", kb_id,
        )
        if not row:
            print(f"error: no KB with id={kb_id}", file=sys.stderr)
            sys.exit(2)
        raw = row["synonyms"] or []
        # asyncpg may return JSONB as string in some configurations
        if isinstance(raw, str):
            raw = json.loads(raw)
        return raw
    finally:
        await conn.close()


async def _set(kb_id: int, synonyms: list) -> None:
    conn = await _conn()
    try:
        result = await conn.execute(
            "UPDATE knowledge_bases SET synonyms = $1::jsonb WHERE id = $2",
            json.dumps(synonyms), kb_id,
        )
        # asyncpg execute returns "UPDATE N" string for UPDATE
        rowcount = int(result.split()[-1]) if result.startswith("UPDATE ") else 0
        if rowcount == 0:
            print(f"error: no KB with id={kb_id}", file=sys.stderr)
            sys.exit(2)
    finally:
        await conn.close()


async def _add(kb_id: int, new_class: list) -> None:
    if not isinstance(new_class, list) or not all(isinstance(s, str) for s in new_class):
        print("error: --add expects a JSON array of strings", file=sys.stderr)
        sys.exit(2)
    conn = await _conn()
    try:
        # Atomic append-if-not-present. The @> check is a JSONB containment
        # operator: returns true only when the array on the left contains
        # the array on the right as an element. Combining the WHERE NOT @>
        # with the || append makes the operation idempotent without a
        # client-side read-modify-write race.
        new_class_json = json.dumps([new_class])  # outer array so || appends as one element
        target_json = json.dumps([new_class])
        result = await conn.execute(
            """
            UPDATE knowledge_bases
            SET synonyms = synonyms || $1::jsonb
            WHERE id = $2 AND NOT (synonyms @> $3::jsonb)
            """,
            new_class_json, kb_id, target_json,
        )
        # asyncpg returns "UPDATE N" string; parse rowcount
        rowcount = int(result.split()[-1]) if result.startswith("UPDATE ") else 0
        if rowcount == 0:
            # Either KB doesn't exist OR class already present — distinguish:
            exists = await conn.fetchval(
                "SELECT 1 FROM knowledge_bases WHERE id = $1", kb_id,
            )
            if not exists:
                print(f"error: no KB with id={kb_id}", file=sys.stderr)
                sys.exit(2)
            print("info: class already present, no change", file=sys.stderr)
    finally:
        await conn.close()


async def _remove(kb_id: int, target: list) -> None:
    current = await _list(kb_id)
    new = [c for c in current if c != target]
    if len(new) == len(current):
        print("info: class not found, no change", file=sys.stderr)
        return
    await _set(kb_id, new)


async def _load(kb_id: int, path: str) -> None:
    if path == "-":
        data = json.load(sys.stdin)
    else:
        with open(path) as fh:
            data = json.load(fh)
    if not isinstance(data, list):
        print("error: --load expects a JSON array of arrays of strings",
              file=sys.stderr)
        sys.exit(2)
    if not all(
        isinstance(cls, list) and all(isinstance(s, str) for s in cls)
        for cls in data
    ):
        print("error: --load expects a JSON array of arrays of STRINGS — "
              "each inner element must itself be a JSON array of strings",
              file=sys.stderr)
        sys.exit(2)
    await _set(kb_id, data)


async def main_async(args: argparse.Namespace) -> int:
    if args.list:
        out = await _list(args.kb)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if args.add:
        await _add(args.kb, json.loads(args.add))
        return 0
    if args.remove:
        await _remove(args.kb, json.loads(args.remove))
        return 0
    if args.load:
        await _load(args.kb, args.load)
        return 0
    print("error: pick one of --list, --add, --remove, --load",
          file=sys.stderr)
    return 2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--add", help="JSON array of strings to add as one equivalence class")
    g.add_argument("--remove", help="JSON array of strings — remove the class that exactly matches")
    g.add_argument("--load", help="path to JSON file (or - for stdin) replacing the entire table")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
