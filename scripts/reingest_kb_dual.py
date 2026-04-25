#!/usr/bin/env python3
"""Dual-collection re-ingest: kb_1_v2 → kb_1_v3 with full Plan-A pipeline.

Reads every point from a SOURCE Qdrant collection (e.g. kb_1_v2), groups
them by ``doc_id``, reconstructs the per-document text from the chunk
bodies (stripping any pre-existing ``contextual-v1`` prefix), and writes
them into a TARGET collection with the new tri-vector pipeline:

    * dense (TEI bge-m3, 1024-d, COSINE)
    * sparse bm25 (fastembed Qdrant/bm25, IDF modifier)
    * colbert multivec (fastembed colbert-ir/colbertv2.0, 128-d, MAX_SIM)

Each chunk is augmented in-place with an LLM-generated ``context_prefix``
field (Anthropic Contextual Retrieval recipe). The prefix is also
prepended to the dense-embedded ``text`` so retrieval finds it via BM25
and dense both. The original ``text`` (pre-prefix) is recoverable by
splitting on the first ``\\n\\n``.

Doc-summary points (``level=doc``, ``chunk_index=-1``) are migrated AS-IS
— no contextualization (the summary is already self-contained), but the
dense vector is re-embedded for safety and BM25 + ColBERT vectors are
added so the new collection has a homogenous tri-vector shape.

Throttle: before each per-doc upsert, query Prometheus for the chat
endpoint's p95 TTFT. If above ``--throttle-ceiling-ms``, sleep 30s and
retry. When the metric isn't published (e.g. Prometheus isn't scraping
vllm-chat — the case in the live stack as of 2026-04-24), the query
returns empty and we treat it as 0ms (no-op throttle).

Failure handling: per-doc try/except. A doc that fails contextualization
or embedding is logged and skipped — the run continues. Idempotent: the
point ID is ``UUID5(URL_NS, "doc:{doc_id}:chunk:{chunk_index}")`` so a
re-run upserts onto the same row.

Usage::

    PYTHONPATH=/home/vogic/LocalRAG-plan-a \\
      /home/vogic/LocalRAG/.venv/bin/python \\
      scripts/reingest_kb_dual.py \\
        --source kb_1_v2 \\
        --target kb_1_v3 \\
        --kb-id 1 \\
        --qdrant-url http://localhost:6333 \\
        --prom-url http://localhost:9090 \\
        --throttle-ceiling-ms 3000

This script is a one-off forensic tool — kept under ``scripts/`` for
traceability of the kb_1_v2 → kb_1_v3 migration on 2026-04-24.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

# --------------------------------------------------------------------------
# PYTHONPATH bootstrap so this script works whether run from the worktree
# root or via the explicit ``PYTHONPATH=/home/vogic/LocalRAG-plan-a`` form
# above. Ensures the LOCAL ``ext.services.*`` modules win over any system
# install.
# --------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ext.services.contextualizer import contextualize_chunks_with_prefix  # noqa: E402
from ext.services.embedder import TEIEmbedder, colbert_embed  # noqa: E402
from ext.services.sparse_embedder import embed_sparse  # noqa: E402


log = logging.getLogger("reingest_kb_dual")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# Stable UUID5 namespace for deterministic point IDs. Same value as
# ``ext.services.ingest._POINT_NS`` so re-running this script overwrites
# rather than duplicates.
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Pipeline version stamp. Mirrors ext.services.pipeline_version.current_version
# under contextual-v1 + bge-m3 + chunker-v2 (no model bump in this run).
_PIPELINE_VERSION = "chunker=v2|extractor=v2|embedder=bge-m3|ctx=contextual-v1"

# Filename → YYYY-MM-DD if it parses cleanly. Used as the document_date
# anchor in the contextualizer prompt (the temporal anchor Anthropic flags
# as load-bearing for cross-document retrieval).
_DATE_PATTERNS = [
    # "10 Feb 2026.docx", "10 Feb 2026 (rev).docx"
    (re.compile(r"(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})", re.I),
     "%d %b %Y"),
    # "2026-04-15.pdf"
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), "%Y-%m-%d"),
    # "15-04-2026.pdf"
    (re.compile(r"(\d{2})-(\d{2})-(\d{4})"), "%d-%m-%Y"),
]


def _extract_doc_date(filename: str) -> Optional[str]:
    """Best-effort filename → ISO date string. Returns None on no match."""
    if not filename:
        return None
    from datetime import datetime as _dt
    for pat, fmt in _DATE_PATTERNS:
        m = pat.search(filename)
        if m:
            try:
                # Reconstruct the source string from the matched groups in
                # the same order the format string expects.
                src = m.group(0)
                d = _dt.strptime(src, fmt)
                return d.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def _strip_context_prefix(text: str, *, had_ctx_marker: bool) -> str:
    """Strip a previously-prepended LLM context prefix from a chunk body.

    Pre-3.7 contextualizer prepended the prefix as ``f"{prefix}\\n\\n{body}"``.
    If the chunk is marked with ``ctx_version=contextual-v1`` (legacy stamp)
    or carries an explicit ``context_prefix`` payload field, we split on
    the FIRST ``\\n\\n`` and return the tail. Otherwise return the text
    unchanged (no false positives — body paragraphs also use ``\\n\\n``,
    so we only strip when we KNOW a prefix was applied).
    """
    if not had_ctx_marker or not text:
        return text
    parts = text.split("\n\n", 1)
    if len(parts) == 2:
        return parts[1]
    return text


async def _scroll_all_points(
    client: AsyncQdrantClient, collection: str
) -> dict[Any, list[dict]]:
    """Read every point in ``collection``, group by doc_id.

    Returns ``{doc_id: [payload_dict, ...]}``. Each payload dict carries
    its original ``id`` field (preserved from Qdrant) so callers can
    decide between recomputing the UUID5 (chunks) or reusing the source
    ID (doc-summary points use a different seed pattern).
    """
    grouped: dict[Any, list[dict]] = defaultdict(list)
    offset = None
    n_seen = 0
    while True:
        points, offset = await client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            break
        for p in points:
            payload = dict(p.payload or {})
            payload["_source_id"] = p.id  # preserve for doc-summary parity
            did = payload.get("doc_id")
            if did is None:
                # No doc_id → orphan; skip (would have been deleted by
                # delete_orphan_chunks.py anyway).
                continue
            grouped[did].append(payload)
        n_seen += len(points)
        if n_seen % 1024 == 0:
            log.info("  scrolled %d points so far", n_seen)
        if offset is None:
            break
    log.info("scroll complete: %d total points across %d docs",
             n_seen, len(grouped))
    return grouped


async def _chat_p95_ms(
    client: httpx.AsyncClient, prom_url: str, *, model_pat: str = "orgchat-chat"
) -> float:
    """Best-effort chat-endpoint p95 in ms. Returns 0.0 when unavailable.

    Tries two metric names in order:
      1. ``rag_llm_ttft_seconds_bucket`` — the ext-app-emitted histogram
         (will exist once the patched orgchat container is in place).
      2. ``vllm:time_to_first_token_seconds_bucket`` — vllm's native
         metric (only useful if Prometheus is scraping vllm-chat directly).

    If both queries fail or return no data, returns 0.0 → throttling
    becomes a no-op rather than blocking the run.
    """
    queries = [
        f'histogram_quantile(0.95, rate(rag_llm_ttft_seconds_bucket{{model=~".*"}}[5m])) * 1000',
        f'histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m])) * 1000',
    ]
    for q in queries:
        try:
            r = await client.get(
                f"{prom_url}/api/v1/query",
                params={"query": q},
                timeout=5.0,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("data", {}).get("result", [])
            if results:
                val_str = results[0]["value"][1]
                if val_str not in ("NaN", "+Inf", "-Inf"):
                    return float(val_str)
        except Exception as e:
            log.debug("prom query %s failed: %s", q[:60], e)
            continue
    return 0.0


async def _throttle(
    client: httpx.AsyncClient,
    prom_url: str,
    ceiling_ms: float,
    state: dict,
) -> None:
    """Sleep 30s repeatedly while chat p95 > ``ceiling_ms``. No-op when 0."""
    while True:
        p95 = await _chat_p95_ms(client, prom_url)
        if p95 <= ceiling_ms or p95 == 0.0:
            return
        state["throttle_count"] += 1
        log.warning(
            "chat p95 %.0fms > ceiling %.0fms — sleeping 30s (activations=%d)",
            p95, ceiling_ms, state["throttle_count"],
        )
        await asyncio.sleep(30)


def _build_doc_metadata(
    *, filename: str, kb_name: str, subtag_name: Optional[str]
) -> dict:
    """Build the ``document_metadata`` dict the contextualizer prompt expects.

    Intentionally omits ``related_doc_titles`` — populating that requires a
    cross-document graph we don't compute in this script. Date is parsed
    from filename when possible; KB+subtag come from Postgres lookups.
    """
    return {
        "filename": filename,
        "kb_name": kb_name,
        "subtag_name": subtag_name,
        "document_date": _extract_doc_date(filename),
        "related_doc_titles": [],
    }


async def _process_one_doc(
    *,
    doc_id: Any,
    payloads: list[dict],
    target_collection: str,
    kb_name: str,
    subtag_name_by_id: dict[int, str],
    embedder: TEIEmbedder,
    qdrant: AsyncQdrantClient,
    state: dict,
) -> tuple[int, int]:
    """Re-embed + re-upsert one doc's chunks. Returns (n_chunks, n_failed_ctx).

    Splits the doc's points into chunk vs doc-summary, processes each
    accordingly, then upserts both groups in one batched call (preserves
    the source point IDs for doc-summary points, recomputes UUID5 for
    chunks).
    """
    chunks = [p for p in payloads if p.get("level") != "doc"]
    summaries = [p for p in payloads if p.get("level") == "doc"]

    chunks.sort(key=lambda p: p.get("chunk_index", 0))

    # ----- Reconstruct doc text + per-chunk dicts -----------------------
    # Strip any pre-existing context_prefix from each chunk's text. We
    # detect "this chunk had a prefix" via either an explicit
    # ``context_prefix`` field OR the legacy ``ctx_version=contextual-v1``
    # marker (the live kb_1_v2 was recontextualized in-place earlier and
    # only carries the marker, not the explicit field).
    chunk_dicts: list[dict] = []
    raw_texts: list[str] = []
    for p in chunks:
        had_marker = (
            p.get("ctx_version") == "contextual-v1"
            or p.get("context_prefix") is not None
        )
        raw = _strip_context_prefix(p.get("text", "") or "", had_ctx_marker=had_marker)
        raw_texts.append(raw)
        chunk_dicts.append({"text": raw, "context_prefix": None})

    # Document text: concatenate raw chunks in order, separated by blank
    # lines. This is an approximation of the source doc — chunker/extractor
    # added/removed whitespace, so we won't exactly recover the original
    # bytes, but the contextualizer only needs enough text to situate one
    # chunk relative to the rest.
    document_text = "\n\n".join(t for t in raw_texts if t)

    if not chunks and not summaries:
        return 0, 0

    # ----- Pull per-doc filename + subtag from the source payload --------
    sample = chunks[0] if chunks else summaries[0]
    filename = sample.get("filename") or f"doc_{doc_id}"
    subtag_id = sample.get("subtag_id")
    subtag_name = subtag_name_by_id.get(subtag_id) if subtag_id is not None else None
    owner_user_id = sample.get("owner_user_id")

    n_failed_ctx = 0

    # ----- Contextualize chunks (mutates chunk_dicts in place) ----------
    if chunk_dicts and document_text:
        meta = _build_doc_metadata(
            filename=filename, kb_name=kb_name, subtag_name=subtag_name,
        )
        try:
            await contextualize_chunks_with_prefix(
                chunk_dicts,
                document_text=document_text,
                document_metadata=meta,
                concurrency=int(os.environ.get("RAG_CONTEXTUALIZE_CONCURRENCY", "8")),
                timeout_s=float(os.environ.get("RAG_CONTEXTUALIZE_TIMEOUT", "30")),
            )
        except Exception as e:
            # Whole-batch fail-open: log and continue with raw chunks.
            log.warning("contextualize_batch failed for doc %s: %s", doc_id, e)
            n_failed_ctx = len(chunk_dicts)
        else:
            # Count per-chunk fail-opens (context_prefix==None means the
            # contextualizer fell open for that chunk).
            n_failed_ctx = sum(1 for cd in chunk_dicts if not cd.get("context_prefix"))

    # ----- Embed dense (TEI bge-m3) -------------------------------------
    chunk_texts = [cd["text"] for cd in chunk_dicts]
    summary_texts = [s.get("text", "") or "" for s in summaries]
    all_texts = chunk_texts + summary_texts
    if not all_texts:
        return 0, n_failed_ctx

    dense_vecs = await embedder.embed(all_texts)
    chunk_dense = dense_vecs[: len(chunk_texts)]
    summary_dense = dense_vecs[len(chunk_texts) :]

    # ----- Embed sparse (BM25) ------------------------------------------
    # Run in a thread executor — fastembed is sync and we don't want to
    # block the event loop on docs with hundreds of chunks.
    loop = asyncio.get_event_loop()
    try:
        sparse_all = await loop.run_in_executor(
            None, lambda: list(embed_sparse(all_texts))
        )
    except Exception as e:
        log.warning("BM25 sparse embed failed for doc %s: %s — skipping sparse arm", doc_id, e)
        sparse_all = [None] * len(all_texts)
    chunk_sparse = sparse_all[: len(chunk_texts)]
    summary_sparse = sparse_all[len(chunk_texts) :]

    # ----- Embed ColBERT (multivec) -------------------------------------
    try:
        colbert_all = await loop.run_in_executor(
            None, lambda: list(colbert_embed(all_texts))
        )
    except Exception as e:
        log.warning("ColBERT embed failed for doc %s: %s — skipping colbert arm", doc_id, e)
        colbert_all = [None] * len(all_texts)
    chunk_colbert = colbert_all[: len(chunk_texts)]
    summary_colbert = colbert_all[len(chunk_texts) :]

    # ----- Build Qdrant point structs -----------------------------------
    pts: list[qm.PointStruct] = []
    now_ns = time.time_ns()

    # Chunk points: regenerate UUID5 from doc:chunk_index for idempotency.
    for i, (cd, src) in enumerate(zip(chunk_dicts, chunks)):
        chunk_idx = src.get("chunk_index", i)
        # Preserve the original payload identity fields, overlay the new
        # text + context_prefix + bumped pipeline_version.
        payload = {
            "kb_id": src.get("kb_id"),
            "doc_id": src.get("doc_id"),
            "subtag_id": src.get("subtag_id"),
            "owner_user_id": src.get("owner_user_id"),
            "chat_id": src.get("chat_id"),
            "filename": filename,
            "chunk_index": chunk_idx,
            "text": cd["text"],
            "context_prefix": cd.get("context_prefix"),
            "page": src.get("page"),
            "heading_path": list(src.get("heading_path") or []),
            "sheet": src.get("sheet"),
            "uploaded_at": src.get("uploaded_at") or now_ns,
            "deleted": False,
            "model_version": _PIPELINE_VERSION,
            "level": "chunk",
            # Carry forward the legacy ctx_version marker for backward
            # compat with retrievers that still check it.
            "ctx_version": "contextual-v1",
        }
        # Preserve any other payload fields we don't recognize so we
        # don't silently lose data on the migration.
        for k, v in src.items():
            if k not in payload and not k.startswith("_"):
                payload[k] = v

        vec_map: dict = {"dense": chunk_dense[i]}
        sv = chunk_sparse[i] if i < len(chunk_sparse) else None
        if sv is not None:
            indices, values = sv
            vec_map["bm25"] = qm.SparseVector(
                indices=list(indices), values=list(values),
            )
        cv = chunk_colbert[i] if i < len(chunk_colbert) else None
        if cv is not None:
            vec_map["colbert"] = cv

        point_id = str(uuid.uuid5(_POINT_NS, f"doc:{doc_id}:chunk:{chunk_idx}"))
        pts.append(qm.PointStruct(id=point_id, vector=vec_map, payload=payload))

    # Doc-summary points: reuse the source UUID5 (already derived from
    # doc:{doc_id}:doc_summary by the original ingest), preserve text
    # verbatim, just re-attach the new tri-vector embeddings.
    for i, src in enumerate(summaries):
        payload = dict(src)
        payload.pop("_source_id", None)
        payload["model_version"] = _PIPELINE_VERSION
        payload.setdefault("level", "doc")
        payload.setdefault("kind", "doc_summary")

        vec_map = {"dense": summary_dense[i]}
        sv = summary_sparse[i] if i < len(summary_sparse) else None
        if sv is not None:
            indices, values = sv
            vec_map["bm25"] = qm.SparseVector(
                indices=list(indices), values=list(values),
            )
        cv = summary_colbert[i] if i < len(summary_colbert) else None
        if cv is not None:
            vec_map["colbert"] = cv

        # Reuse the source point ID so a re-run idempotently overwrites.
        point_id = src.get("_source_id") or str(uuid.uuid5(
            _POINT_NS, f"doc:{doc_id}:doc_summary"
        ))
        pts.append(qm.PointStruct(id=point_id, vector=vec_map, payload=payload))

    # ----- Upsert in batches of 32 --------------------------------------
    batch = int(os.environ.get("RAG_REINGEST_BATCH", "32"))
    for i in range(0, len(pts), batch):
        await qdrant.upsert(
            collection_name=target_collection,
            points=pts[i : i + batch],
            wait=True,
        )

    return len(pts), n_failed_ctx


async def _amain(args: argparse.Namespace) -> int:
    # ----- Load Postgres lookups ----------------------------------------
    # We only need KB name + subtag-id → name mapping for the contextualizer
    # prompt's anchor fields. Falls back to "unknown" if Postgres is down.
    kb_name = "unknown"
    subtag_name_by_id: dict[int, str] = {}
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql://") and "+asyncpg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if db_url:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            from sqlalchemy import text as _sql
            engine = create_async_engine(db_url, pool_pre_ping=True)
            Session = async_sessionmaker(engine, expire_on_commit=False)
            async with Session() as s:
                row = (await s.execute(
                    _sql("SELECT name FROM knowledge_bases WHERE id = :k"),
                    {"k": args.kb_id},
                )).first()
                if row:
                    kb_name = row[0]
                rows = (await s.execute(
                    _sql("SELECT id, name FROM kb_subtags WHERE kb_id = :k"),
                    {"k": args.kb_id},
                )).all()
                for r in rows:
                    subtag_name_by_id[int(r[0])] = str(r[1])
            await engine.dispose()
            log.info("kb_name=%r subtags=%s", kb_name, subtag_name_by_id)
        except Exception as e:
            log.warning("Postgres lookup failed (%s) — using kb_name=%r", e, kb_name)
    else:
        log.warning("DATABASE_URL unset — using kb_name=%r", kb_name)

    # ----- Wire up Qdrant + TEI -----------------------------------------
    qdrant = AsyncQdrantClient(url=args.qdrant_url, timeout=120.0)
    tei_url = (
        os.environ.get("TEI_URL")
        or os.environ.get("RAG_EMBEDDING_OPENAI_API_BASE_URL", "")
        or "http://localhost:8080"
    )
    if "/v1" in tei_url:
        tei_url = tei_url.split("/v1")[0]
    embedder = TEIEmbedder(base_url=tei_url)
    log.info("TEI url=%r", tei_url)

    # ----- Scroll source -------------------------------------------------
    log.info("scrolling source collection %r", args.source)
    grouped = await _scroll_all_points(qdrant, args.source)
    n_docs_total = len(grouped)
    if not n_docs_total:
        log.warning("source collection has no points — nothing to do")
        await qdrant.close()
        await embedder.aclose()
        return 0
    log.info("found %d docs / %d points in source",
             n_docs_total, sum(len(v) for v in grouped.values()))

    # Smoke-test filters (off by default).
    if args.only_doc_id is not None:
        if args.only_doc_id not in grouped:
            log.error("--only-doc-id=%s not found in source", args.only_doc_id)
            await qdrant.close(); await embedder.aclose(); return 1
        grouped = {args.only_doc_id: grouped[args.only_doc_id]}
        n_docs_total = 1
        log.info("smoke-test mode: processing only doc_id=%s (%d points)",
                 args.only_doc_id, len(grouped[args.only_doc_id]))
    elif args.limit_docs is not None:
        keys = sorted(grouped.keys())[: args.limit_docs]
        grouped = {k: grouped[k] for k in keys}
        n_docs_total = len(grouped)
        log.info("smoke-test mode: limited to first %d docs", n_docs_total)

    # ----- Re-ingest loop ------------------------------------------------
    state = {"throttle_count": 0}
    n_docs_ok = 0
    n_docs_err = 0
    n_points_total = 0
    n_chunks_failed_ctx_total = 0
    failed_doc_samples: list[tuple[Any, str]] = []

    stopping = {"sig": False}
    def _handler(sig, frame):
        log.warning("signal %s — finishing current doc then exiting", sig)
        stopping["sig"] = True
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=10.0) as prom_client:
        for idx, (doc_id, payloads) in enumerate(sorted(grouped.items(), key=lambda kv: kv[0])):
            if stopping["sig"]:
                break
            try:
                await _throttle(
                    prom_client, args.prom_url, args.throttle_ceiling_ms, state,
                )
                n_pts, n_failed = await _process_one_doc(
                    doc_id=doc_id,
                    payloads=payloads,
                    target_collection=args.target,
                    kb_name=kb_name,
                    subtag_name_by_id=subtag_name_by_id,
                    embedder=embedder,
                    qdrant=qdrant,
                    state=state,
                )
                n_docs_ok += 1
                n_points_total += n_pts
                n_chunks_failed_ctx_total += n_failed
            except Exception as e:
                n_docs_err += 1
                msg = repr(e)[:200]
                failed_doc_samples.append((doc_id, msg))
                log.error("doc %s failed: %s", doc_id, msg)
                continue

            elapsed = time.perf_counter() - t0
            done = idx + 1
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_docs_total - done) / rate if rate > 0 else 0
            log.info(
                "[%d/%d] doc=%s pts=%d ctx_fail=%d ok=%d err=%d "
                "elapsed=%.1fs ETA=%.1fmin",
                done, n_docs_total, doc_id, n_points_total,
                n_chunks_failed_ctx_total, n_docs_ok, n_docs_err,
                elapsed, eta / 60,
            )

    elapsed = time.perf_counter() - t0
    log.info(
        "DONE wall=%.1fs(%.1fmin) docs_ok=%d docs_err=%d points=%d "
        "ctx_fail=%d throttle_pauses=%d",
        elapsed, elapsed / 60, n_docs_ok, n_docs_err, n_points_total,
        n_chunks_failed_ctx_total, state["throttle_count"],
    )
    if failed_doc_samples:
        log.warning("Failed doc samples (up to 10):")
        for did, msg in failed_doc_samples[:10]:
            log.warning("  doc_id=%s: %s", did, msg)

    await qdrant.close()
    await embedder.aclose()
    return 0 if n_docs_err == 0 else 1


def _parse(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, help="Source Qdrant collection")
    p.add_argument("--target", required=True, help="Target Qdrant collection")
    p.add_argument("--kb-id", type=int, required=True, help="KB id (Postgres)")
    p.add_argument("--qdrant-url", default="http://localhost:6333")
    p.add_argument("--prom-url", default="http://localhost:9090")
    p.add_argument("--throttle-ceiling-ms", type=float, default=3000.0,
                   help="Pause if chat p95 exceeds this many ms")
    p.add_argument("--limit-docs", type=int, default=None,
                   help="Process at most N docs (smoke-test only)")
    p.add_argument("--only-doc-id", type=int, default=None,
                   help="Process only this single doc_id (smoke-test only)")
    return p.parse_args(argv)


def main() -> int:
    args = _parse()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
