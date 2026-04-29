#!/usr/bin/env python3
"""Seed Kairos analyst-style RAG configuration into Postgres.

Writes:
- ``config.data->'rag'->'template'`` — the military-format RAG_TEMPLATE
  (BLUF + SMEAC + visualization discipline). Read at startup by Open WebUI's
  PersistentConfig and used to wrap every RAG-using turn's context.
- ``config.data->'rag'->'top_k'`` and ``top_k_reranker`` — bumped to 12 so
  the template has enough material to brief deeply.
- ``model.params.system`` for the ``orgchat-chat`` model — analyst persona
  applied to non-RAG turns too (briefing format + visual-first habit).

Idempotent: re-running overwrites the same keys. Safe to invoke during cold
start AND any time the operator updates the .txt files in this directory.

Source files (read relative to this script):
- ``rag_template_military.txt``
- ``system_prompt_analyst.txt``

Postgres access goes through ``docker exec`` because the postgres container
does not publish 5432 to the host. Override the container name via env
``KAIROS_POSTGRES_CONTAINER`` if your compose project name differs.

Usage:
    .venv/bin/python scripts/apply_analyst_config.py             # apply
    .venv/bin/python scripts/apply_analyst_config.py --dry-run   # print SQL only

Exit codes:
    0 — applied (or dry-run successful)
    1 — psql failed / docker exec failed
    2 — bad input (missing template files, tag collision)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = SCRIPT_DIR / "rag_template_military.txt"
SYSPROMPT_PATH = SCRIPT_DIR / "system_prompt_analyst.txt"

DEFAULT_CONTAINER = os.environ.get("KAIROS_POSTGRES_CONTAINER", "orgchat-postgres")
DEFAULT_DB = os.environ.get("KAIROS_POSTGRES_DB", "orgchat")
DEFAULT_USER = os.environ.get("KAIROS_POSTGRES_USER", "orgchat")
DEFAULT_MODEL_ID = os.environ.get("KAIROS_MODEL_ID", "orgchat-chat")

# Dollar-quote tags. Picked unlikely-to-collide; assert at runtime to be safe.
TAG_T = "_KAIROSTEMPL_"
TAG_S = "_KAIROSSYS_"


def build_sql(template: str, sysprompt: str, model_id: str) -> str:
    if f"${TAG_T}$" in template:
        raise SystemExit(f"!! tag collision: {TAG_T} appears in template text")
    if f"${TAG_S}$" in sysprompt:
        raise SystemExit(f"!! tag collision: {TAG_S} appears in sysprompt text")
    return f"""
\\set ON_ERROR_STOP on
BEGIN;

-- Build the rag block whole (jsonb_set can't create nested keys under a
-- missing parent), then assign it as the value of the 'rag' key on
-- config.data. config.data is JSON (not JSONB); cast both ways.
UPDATE config
SET data = jsonb_set(
    COALESCE(data::jsonb, '{{}}'::jsonb),
    '{{rag}}',
    COALESCE(data::jsonb -> 'rag', '{{}}'::jsonb) || jsonb_build_object(
      'template',           ${TAG_T}${template}${TAG_T}$::text,
      'top_k',              12,
      'top_k_reranker',     12,
      'relevance_threshold', 0.0
    ),
    true
)::json
WHERE id = 1;

-- Merge the analyst system prompt into the orgchat-chat model row's
-- ``params`` (text-encoded JSON object). Other params keys are preserved.
UPDATE model
SET
  params = (
    COALESCE(NULLIF(params, '')::jsonb, '{{}}'::jsonb)
    || jsonb_build_object('system', ${TAG_S}${sysprompt}${TAG_S}$::text)
  )::text,
  updated_at = extract(epoch from now())::bigint
WHERE id = '{model_id}';

COMMIT;

-- Verify
SELECT
  'config.rag.top_k'::text AS key,
  (data::jsonb #>> '{{rag,top_k}}')::text AS value
FROM config WHERE id = 1
UNION ALL SELECT
  'config.rag.top_k_reranker',
  (data::jsonb #>> '{{rag,top_k_reranker}}')
FROM config WHERE id = 1
UNION ALL SELECT
  'config.rag.template_length',
  length(data::jsonb #>> '{{rag,template}}')::text
FROM config WHERE id = 1
UNION ALL SELECT
  'model.system_length',
  length(params::jsonb ->> 'system')::text
FROM model WHERE id = '{model_id}';
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--container", default=DEFAULT_CONTAINER,
                   help=f"Postgres container name (default: {DEFAULT_CONTAINER}; "
                        f"override via KAIROS_POSTGRES_CONTAINER env)")
    p.add_argument("--db", default=DEFAULT_DB,
                   help=f"Database name (default: {DEFAULT_DB})")
    p.add_argument("--user", default=DEFAULT_USER,
                   help=f"Postgres user (default: {DEFAULT_USER})")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help=f"Model id to receive the system prompt "
                        f"(default: {DEFAULT_MODEL_ID})")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the SQL without executing it")
    args = p.parse_args()

    for path in (TEMPLATE_PATH, SYSPROMPT_PATH):
        if not path.exists():
            print(f"!! missing file: {path}", file=sys.stderr)
            return 2
    template = TEMPLATE_PATH.read_text()
    sysprompt = SYSPROMPT_PATH.read_text()

    sql = build_sql(template, sysprompt, args.model_id)

    if args.dry_run:
        print(sql)
        return 0

    cmd = ["docker", "exec", "-i", args.container,
           "psql", "-U", args.user, "-d", args.db,
           "-v", "ON_ERROR_STOP=1"]
    try:
        r = subprocess.run(cmd, input=sql, text=True, capture_output=True)
    except FileNotFoundError:
        print("!! docker CLI not found on PATH", file=sys.stderr)
        return 1

    sys.stdout.write(r.stdout)
    if r.stderr:
        sys.stderr.write(r.stderr)

    if r.returncode != 0:
        print(
            f"!! psql exited {r.returncode}; container={args.container} db={args.db}",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nOK: applied template ({len(template)} chars) + system prompt "
        f"({len(sysprompt)} chars) to {args.container}:{args.db}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
