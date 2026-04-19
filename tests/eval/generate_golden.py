"""Auto-generate a synthetic RAG golden set from the live Qdrant collections.

For each sampled chunk we ask the chat model: "given this excerpt, what question
would a user ask where this excerpt is the answer?" The (chunk, back-generated
question) pair becomes one row of ``tests/eval/golden.jsonl``.

NOTE: synthetic queries are a plumbing test only. ``source_text -> query ->
source_text`` is close to trivial for any working retriever; they catch
regressions in the pipeline (bad filtering, broken embedder, etc.) but say
almost nothing about real-world quality. Human-labeled queries are still
required before the CI regression gate can be treated as meaningful.

Usage:
  python -m tests.eval.generate_golden \\
      --qdrant-url http://localhost:6333 \\
      --chat-url http://localhost:8000/v1 \\
      --chat-model Qwen/Qwen2.5-14B-Instruct-AWQ \\
      --collections kb_1,kb_3,kb_4,kb_5 \\
      --samples-per-collection 10 \\
      --out tests/eval/golden.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Optional

import httpx
from qdrant_client import AsyncQdrantClient


PROMPT_TEMPLATE = (
    "Read this document excerpt. Write ONE natural-sounding search question "
    "that a user might ask where this excerpt would be the correct answer. "
    "Output ONLY the question, no preamble, no quotes.\n\n"
    "Excerpt:\n{excerpt}"
)


async def sample_points(
    client: AsyncQdrantClient,
    collection: str,
    n: int,
) -> list[dict]:
    """Pull up to ``n`` points with payload from ``collection``.

    Uses scroll with a generous limit then samples client-side so we don't bias
    toward the first segment. Returns a list of ``{"id", "payload"}`` dicts.
    """
    # Pull 5x n or up to 500 candidates, whichever is smaller, then sample.
    fetch_limit = min(max(n * 5, n), 500)
    try:
        points, _next = await client.scroll(
            collection_name=collection,
            limit=fetch_limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        print(f"WARN: scroll failed for {collection}: {exc}", file=sys.stderr)
        return []

    rows: list[dict] = []
    for p in points:
        payload = p.payload or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue
        rows.append({"id": str(p.id), "payload": payload})

    if not rows:
        return []

    random.shuffle(rows)
    return rows[:n]


async def generate_query(
    client: httpx.AsyncClient,
    chat_model: str,
    excerpt: str,
    *,
    timeout: float = 30.0,
    max_tokens: int = 128,
) -> Optional[str]:
    """Call the OpenAI-compatible /chat/completions endpoint to back-generate a query.

    Returns ``None`` on any error so callers can skip and continue.
    """
    # Keep excerpt short to leave room for the question; long excerpts waste
    # context and slow vLLM down.
    trimmed = excerpt.strip()
    if len(trimmed) > 1500:
        trimmed = trimmed[:1500]

    payload = {
        "model": chat_model,
        "messages": [
            {"role": "user", "content": PROMPT_TEMPLATE.format(excerpt=trimmed)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    try:
        r = await client.post("/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        question = body["choices"][0]["message"]["content"].strip()
        # Strip wrapping quotes or trailing junk the model sometimes emits.
        if (question.startswith('"') and question.endswith('"')) or (
            question.startswith("'") and question.endswith("'")
        ):
            question = question[1:-1].strip()
        # Keep only the first line to guard against preamble leakage.
        question = question.splitlines()[0].strip()
        if not question or len(question) < 3:
            return None
        return question
    except httpx.HTTPStatusError as exc:
        print(
            f"WARN: chat HTTP {exc.response.status_code} -> {exc.response.text[:200]}",
            file=sys.stderr,
        )
        return None
    except Exception as exc:
        print(f"WARN: chat call failed: {exc}", file=sys.stderr)
        return None


def _coerce_int(v) -> Optional[int]:
    """Coerce a payload value that might be int or numeric str -> int, else None."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


async def build_golden(args) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        print(
            f"{out_path} already exists — pass --force to overwrite, or move it aside.",
            file=sys.stderr,
        )
        return 2

    random.seed(args.seed)
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    chat = httpx.AsyncClient(base_url=args.chat_url, timeout=args.timeout)
    # Some vLLM deployments require a bearer token even for local use.
    if args.chat_api_key:
        chat.headers["Authorization"] = f"Bearer {args.chat_api_key}"

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    total_rows = 0
    total_skipped = 0

    # Open for write (we already checked that it doesn't exist unless --force).
    with out_path.open("w", encoding="utf-8") as fh:
        for collection in collections:
            print(f"=== {collection} ===", file=sys.stderr)
            points = await sample_points(qdrant, collection, args.samples_per_collection)
            if not points:
                print(f"  (no usable points)", file=sys.stderr)
                continue
            for p in points:
                payload = p["payload"]
                text = payload.get("text", "") or ""
                question = await generate_query(
                    chat,
                    args.chat_model,
                    text,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
                if not question:
                    total_skipped += 1
                    continue

                row = {
                    "query": question,
                    "gold_chunk_id": p["id"],
                    "gold_doc_id": _coerce_int(payload.get("doc_id")),
                    "kb_id": _coerce_int(payload.get("kb_id")),
                    "subtag_id": _coerce_int(payload.get("subtag_id")),
                    "source_text": text[:200],
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                total_rows += 1
                print(
                    f"  + {p['id'][:8]}  kb={row['kb_id']}  doc={row['gold_doc_id']}  "
                    f"q={question[:70]}",
                    file=sys.stderr,
                )

    await qdrant.close()
    await chat.aclose()
    print(f"\nwrote {total_rows} rows to {out_path} (skipped {total_skipped})", file=sys.stderr)
    return 0 if total_rows > 0 else 3


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--chat-url", default="http://localhost:8000/v1",
                   help="OpenAI-compatible base URL for the chat model (include /v1)")
    p.add_argument("--chat-model", default="Qwen/Qwen2.5-14B-Instruct-AWQ")
    p.add_argument("--chat-api-key", default=None,
                   help="Optional bearer token (e.g. the sk-internal-dummy from compose/.env)")
    p.add_argument("--collections", default="kb_1,kb_3,kb_4,kb_5")
    p.add_argument("--samples-per-collection", type=int, default=10)
    p.add_argument("--out", default="tests/eval/golden.jsonl")
    p.add_argument("--force", action="store_true",
                   help="overwrite output file if it already exists")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--max-tokens", type=int, default=128)
    return p.parse_args(argv)


if __name__ == "__main__":
    ns = _parse_args()
    raise SystemExit(asyncio.run(build_golden(ns)))
