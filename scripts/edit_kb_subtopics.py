"""Edit per-KB subtopic-keywords table (rag_config.subtopic_keywords).

Usage:
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --list
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --add '{"visits":["v","vis"]}'
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --remove '["visits"]'
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --load FILE.json
    .venv/bin/python scripts/edit_kb_subtopics.py --kb 2 --load -

Env: DATABASE_URL (required)

Spec: docs/superpowers/specs/2026-05-04-multi-entity-elaborate-answers-design.md §6.1.6
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import asyncpg


async def _conn() -> asyncpg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)
    url = url.replace("+asyncpg", "")
    return await asyncpg.connect(url)


async def _get_table(kb_id: int) -> dict[str, list[str]]:
    conn = await _conn()
    try:
        row = await conn.fetchrow(
            "SELECT rag_config FROM knowledge_bases WHERE id = $1", kb_id,
        )
    finally:
        await conn.close()
    if row is None:
        print(f"error: kb_id={kb_id} not found", file=sys.stderr)
        sys.exit(3)
    rc = row["rag_config"] or {}
    if isinstance(rc, str):
        rc = json.loads(rc)
    table = rc.get("subtopic_keywords") or {}
    if not isinstance(table, dict):
        return {}
    return table


async def _set_table(kb_id: int, table: dict[str, list[str]]) -> None:
    """Atomic JSONB merge — replace the subtopic_keywords key only."""
    conn = await _conn()
    try:
        await conn.execute(
            "UPDATE knowledge_bases "
            "SET rag_config = jsonb_set(rag_config, '{subtopic_keywords}', $1::jsonb) "
            "WHERE id = $2",
            json.dumps(table), kb_id,
        )
    finally:
        await conn.close()


def _validate_payload(payload) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        print("error: payload must be a dict", file=sys.stderr)
        sys.exit(4)
    out: dict[str, list[str]] = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            print(f"error: key must be string: {k!r}", file=sys.stderr)
            sys.exit(4)
        if not isinstance(v, list):
            print(f"error: value must be list: {k!r} -> {v!r}", file=sys.stderr)
            sys.exit(4)
        items = [s for s in v if isinstance(s, str) and s.strip()]
        out[k] = items
    return out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kb", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--load", help="path to JSON file or '-' for stdin (REPLACES table)")
    g.add_argument("--add", help="JSON object — merge new keys into table")
    g.add_argument("--remove", help="JSON list of keys to drop")
    args = p.parse_args()

    if args.list:
        table = await _get_table(args.kb)
        print(json.dumps(table, indent=2))
        return

    if args.load:
        if args.load == "-":
            data = sys.stdin.read()
        else:
            with open(args.load) as f:
                data = f.read()
        payload = _validate_payload(json.loads(data))
        await _set_table(args.kb, payload)
        print(f"replaced subtopic_keywords for kb={args.kb}: {len(payload)} entries")
        return

    if args.add:
        payload = _validate_payload(json.loads(args.add))
        current = await _get_table(args.kb)
        current.update(payload)
        await _set_table(args.kb, current)
        print(f"added/updated keys for kb={args.kb}: {sorted(payload.keys())}")
        return

    if args.remove:
        keys_to_drop = json.loads(args.remove)
        if not isinstance(keys_to_drop, list):
            print("error: --remove takes a JSON list of key names", file=sys.stderr)
            sys.exit(4)
        current = await _get_table(args.kb)
        before = set(current.keys())
        for k in keys_to_drop:
            current.pop(k, None)
        after = set(current.keys())
        await _set_table(args.kb, current)
        print(f"removed keys for kb={args.kb}: {sorted(before - after)}")
        return


if __name__ == "__main__":
    asyncio.run(main())
