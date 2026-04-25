"""Extract → chunk → embed → upsert pipeline.

Each extracted block carries structural metadata (``page`` / ``heading_path`` /
``sheet``). We chunk the block's text independently and inherit the block's
metadata onto every resulting chunk so Qdrant payloads can surface hints like
"from page 7" or "under heading 'Rollout'" at retrieval time.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Mapping

from . import flags
from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract
from .pipeline_version import current_version
from .vector_store import VectorStore


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID at call time so tests can toggle it without reimport.

    Default on as of 2026-04-19 — eval showed +12pp chunk_recall at +3ms.
    Set RAG_HYBRID=0 to force dense-only. Any non-"0" value means "on".
    Runtime fallback: even with hybrid on, ingest only computes sparse vectors
    when the target collection was created with sparse support (via
    ``_collection_has_sparse``) — legacy collections remain dense-only.
    """
    return os.environ.get("RAG_HYBRID", "1") != "0"


def _contextualize_enabled() -> bool:
    """Read RAG_CONTEXTUALIZE_KBS at call time.

    Default OFF. When OFF, the contextualizer module is not imported here,
    so the default ingest path stays byte-identical to the pre-P2.7
    behaviour (no chat-model calls, no httpx churn, no extra imports).
    """
    return os.environ.get("RAG_CONTEXTUALIZE_KBS", "0") == "1"


def _raptor_enabled() -> bool:
    """Read RAG_RAPTOR at call time via the flags overlay.

    Default OFF. When OFF, the raptor module is not imported here, so the
    default ingest path stays byte-identical to the pre-P3.4 behaviour
    (no tree building, no extra chat-model calls, no sklearn import).
    Per-KB overrides from ``rag_config`` flow through ``flags.get``.
    """
    return flags.get("RAG_RAPTOR", "0") == "1"


async def _maybe_contextualize_chunks(
    chunks_and_blocks: list[tuple[object, object]],
    *,
    doc_title: str,
) -> bool:
    """Optionally augment chunks in place with a per-chunk LLM context prefix.

    Returns True when augmentation ran (so ``pipeline_version`` can stamp
    ``ctx=contextual-v1``), False otherwise. Mutates ``chunks_and_blocks``
    by replacing each chunk's ``text`` with its augmented version.

    Fail-open at the batch level: if the whole call errors (e.g. the
    contextualizer module can't be imported, env vars missing, etc.)
    we log and return False so ingest continues with raw chunks and the
    default ``ctx=none`` version stamp.

    Extracted into a helper so the main ingest body stays linear and
    ``ingest.py`` has no top-level dependency on ``contextualizer.py``
    when the flag is off.
    """
    try:
        from .contextualizer import contextualize_batch  # local import — off-path has no cost
        chat_url = os.environ.get("OPENAI_API_BASE_URL")
        chat_model = os.environ.get("CHAT_MODEL", "orgchat-chat")
        if not chat_url:
            return False
        api_key = os.environ.get("OPENAI_API_KEY")
        concurrency = int(os.environ.get("RAG_CONTEXTUALIZE_CONCURRENCY", "8"))
        pairs = [(c.text, doc_title) for c, _ in chunks_and_blocks]
        augmented = await contextualize_batch(
            pairs,
            chat_url=chat_url,
            chat_model=chat_model,
            api_key=api_key,
            concurrency=concurrency,
        )
        # Contextualizer already falls open per-chunk; we just copy the
        # resulting (maybe-augmented, maybe-unchanged) text back onto the
        # chunk objects. Chunks are dataclasses with frozen=True, so we
        # rebuild via a tuple-replace pattern: replace the chunk object
        # in the paired list with a new one whose ``text`` has been bumped.
        from dataclasses import replace as dc_replace
        for idx, (c, b) in enumerate(chunks_and_blocks):
            new_text = augmented[idx]
            if new_text is not c.text:
                # Chunk is frozen; build a copy. token_count stays stale
                # (now slightly larger) — acceptable because retrieval only
                # uses text; budget.py re-tokenizes when it cares.
                chunks_and_blocks[idx] = (dc_replace(c, text=new_text), b)
        return True
    except Exception:  # noqa: BLE001 — fail-open
        return False

# Stable namespace for deterministic point IDs (UUID5 based on doc_id + chunk_index).
# Using the well-known URL namespace UUID so the value is fixed across deploys.
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


async def ingest_bytes(
    *,
    data: bytes,
    mime_type: str,
    filename: str,
    collection: str,
    payload_base: Mapping[str, int | str],
    vector_store: VectorStore,
    embedder: Embedder,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> int:
    """Full ingest: returns number of chunks upserted."""
    blocks = extract(data, mime_type, filename)
    if not blocks:
        return 0

    # Chunk per block; carry the source block forward so we can stamp its
    # structural metadata onto each resulting chunk.
    paired: list[tuple[object, object]] = []  # (Chunk, ExtractedBlock)
    for b in blocks:
        for c in chunk_text(
            b.text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        ):
            paired.append((c, b))
    if not paired:
        return 0

    # P2.7: optional per-chunk context augmentation (Anthropic Contextual
    # Retrieval). OFF by default — the default path does not import the
    # contextualizer module at all. When the flag is on, each chunk gets
    # a short LLM-generated prefix ("this chunk is about X from section Y
    # of document Z") that is then embedded + stored + indexed together
    # with the raw chunk body. Fail-open at both chunk and batch level so
    # a chat endpoint hiccup doesn't crash ingest.
    context_augmented = False
    if _contextualize_enabled():
        context_augmented = await _maybe_contextualize_chunks(
            paired, doc_title=filename
        )

    texts = [c.text for c, _ in paired]
    vectors = await embedder.embed(texts)

    # Sparse vectors are only computed when hybrid is on AND the target
    # collection was created with sparse support. When either condition fails
    # we produce no sparse vectors and the upsert path takes the legacy
    # dense-only shape (byte-identical to the pre-hybrid behaviour). We use
    # getattr with defaults so test doubles / minimal VectorStore substitutes
    # that don't implement the sparse detection helpers still work.
    sparse_vectors: list[tuple[list[int], list[float]] | None] = [None] * len(paired)
    if _hybrid_enabled():
        refresh = getattr(vector_store, "_refresh_sparse_cache", None)
        has_sparse = getattr(vector_store, "_collection_has_sparse", None)
        if refresh is not None and has_sparse is not None:
            try:
                await refresh(collection)
            except Exception:
                pass  # fall through — has_sparse below will be False
            if has_sparse(collection):
                try:
                    from .sparse_embedder import embed_sparse
                    sparse_vectors = list(embed_sparse(texts))  # type: ignore[assignment]
                except Exception:
                    # fastembed missing or failed — silently skip sparse arm.
                    sparse_vectors = [None] * len(paired)

    now = time.time_ns()
    pv = current_version(context_augmented=context_augmented)

    # Defensive coercion — main historically passed str(doc.id); we now store
    # doc_id as int consistently. If the caller supplied a numeric string
    # (legacy callers, worker retries), coerce it; non-numeric values are
    # left untouched so we don't mask genuine misuse.
    if "doc_id" in payload_base and payload_base["doc_id"] is not None:
        try:
            payload_base = {**payload_base, "doc_id": int(payload_base["doc_id"])}
        except (ValueError, TypeError):
            pass  # non-numeric doc_id — leave as-is (shouldn't happen in practice)

    doc_id = payload_base.get("doc_id")
    chat_id = payload_base.get("chat_id")

    # P3.4: optional RAPTOR tree expansion. OFF by default — the default
    # path does not import the raptor module at all. When on, we replace
    # the flat list of chunk points with every tree node (leaves at
    # level 0 plus intermediate LLM summaries at level 1+). Each node
    # becomes one Qdrant point with a ``chunk_level`` payload field so
    # retrieval / admin tooling can distinguish tiers. Fail-open at the
    # batch level — a runaway chat endpoint or a missing dep (sklearn)
    # drops us back to flat ingest.
    tree_nodes = None  # type: ignore[var-annotated]
    if _raptor_enabled():
        try:
            from .raptor import build_tree  # local import — off-path has no cost
            chat_url = os.environ.get("OPENAI_API_BASE_URL")
            chat_model = os.environ.get("CHAT_MODEL", "orgchat-chat")
            api_key = os.environ.get("OPENAI_API_KEY")
            max_levels = int(flags.get("RAG_RAPTOR_MAX_LEVELS", "3") or 3)
            cluster_min = int(flags.get("RAG_RAPTOR_CLUSTER_MIN", "5") or 5)
            concurrency = int(flags.get("RAG_RAPTOR_CONCURRENCY", "4") or 4)
            if chat_url:
                leaves = [
                    {"text": c.text, "index": i, "embedding": vectors[i]}
                    for i, (c, _b) in enumerate(paired)
                ]
                tree_nodes = await build_tree(
                    leaves,
                    chat_url=chat_url,
                    chat_model=chat_model,
                    api_key=api_key,
                    embedder=embedder,
                    max_levels=max_levels,
                    cluster_min=cluster_min,
                    concurrency=concurrency,
                )
        except Exception:  # noqa: BLE001 — fail-open
            tree_nodes = None

    points = []
    if tree_nodes:
        # Tree-node path: one point per RAPTOR node (leaves + summaries).
        # Leaves index aligns with ``paired[i]`` so structural metadata
        # (page / heading_path / sheet) + sparse vectors still attach.
        # Summary nodes get None for those fields (they don't live in any
        # one source block) and no sparse arm (BM25 over a LLM summary is
        # not meaningful and we don't have it anyway).
        for nidx, node in enumerate(tree_nodes):
            if node.embedding is None:
                continue  # build_tree couldn't embed this summary — skip
            payload = dict(payload_base)
            payload["chunk_index"] = nidx
            payload["text"] = node.text
            payload["uploaded_at"] = now
            payload["deleted"] = False
            payload["model_version"] = pv
            payload["chunk_level"] = int(node.level)
            payload["source_chunk_ids"] = list(node.source_chunk_ids)
            if node.level == 0 and nidx < len(paired):
                # Leaf: inherit the source block's structural metadata.
                _chunk, block = paired[nidx]
                payload["page"] = block.page
                payload["heading_path"] = list(block.heading_path)
                payload["sheet"] = block.sheet
            else:
                payload["page"] = None
                payload["heading_path"] = []
                payload["sheet"] = None

            if doc_id is not None:
                id_seed = f"doc:{doc_id}:chunk:{nidx}"
            else:
                id_seed = f"chat:{chat_id}:chunk:{nidx}"
            point_id = str(uuid.uuid5(_POINT_NS, id_seed))
            point: dict = {
                "id": point_id,
                "vector": node.embedding,
                "payload": payload,
            }
            if node.level == 0 and nidx < len(sparse_vectors):
                sv = sparse_vectors[nidx]
                if sv is not None:
                    point["sparse_vector"] = sv
            points.append(point)
    else:
        for gidx, ((chunk, block), vec) in enumerate(zip(paired, vectors)):
            payload = dict(payload_base)
            payload["chunk_index"] = gidx
            payload["text"] = chunk.text
            payload["uploaded_at"] = now
            payload["deleted"] = False
            # Structural metadata from the source block (may be None / []).
            payload["page"] = block.page
            payload["heading_path"] = list(block.heading_path)
            payload["sheet"] = block.sheet
            payload["model_version"] = pv

            # Deterministic point ID: same doc + global chunk index always maps
            # to the same Qdrant point. This lets delete_by_doc reconstruct point
            # IDs or use payload filtering to remove vectors when a document is
            # soft-deleted.
            if doc_id is not None:
                id_seed = f"doc:{doc_id}:chunk:{gidx}"
            else:
                id_seed = f"chat:{chat_id}:chunk:{gidx}"
            point_id = str(uuid.uuid5(_POINT_NS, id_seed))

            point: dict = {
                "id": point_id,
                "vector": vec,
                "payload": payload,
            }
            sv = sparse_vectors[gidx]
            if sv is not None:
                point["sparse_vector"] = sv
            points.append(point)
    await vector_store.upsert(collection, points)

    # Tier 1: per-document summary point.
    # Gated by RAG_DOC_SUMMARIES (default OFF). When ON and we have a
    # doc_id (KB uploads — not ephemeral chat uploads), call the chat
    # model to produce a 3-sentence summary, embed it, and upsert one
    # more point into the same collection with level="doc". Also mirror
    # the summary into kb_documents.doc_summary so it's queryable from
    # Postgres (UI "what does this doc cover?" previews).
    #
    # Fail-open at every step: a failed summary never blocks the
    # chunk-level ingest (which already succeeded above). Default path
    # (flag off) does not import doc_summarizer — zero cost.
    if flags.get("RAG_DOC_SUMMARIES", "0") == "1" and doc_id is not None and texts:
        try:
            await _emit_doc_summary_point(
                chunks_texts=texts,
                filename=filename,
                doc_id=int(doc_id),
                payload_base=payload_base,
                collection=collection,
                vector_store=vector_store,
                embedder=embedder,
                pipeline_version=pv,
                now_ns=now,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            import logging as _log
            _log.getLogger("orgchat.ingest").warning(
                "doc summary emit failed for doc_id=%s: %s", doc_id, e
            )

    return len(points)


async def _emit_doc_summary_point(
    *,
    chunks_texts: list[str],
    filename: str,
    doc_id: int,
    payload_base: Mapping[str, int | str],
    collection: str,
    vector_store: VectorStore,
    embedder: Embedder,
    pipeline_version: str,
    now_ns: int,
) -> None:
    """Summarize a doc, embed the summary, and upsert one Qdrant point.

    Also UPDATEs ``kb_documents.doc_summary`` so the text is queryable
    from Postgres (no Qdrant round-trip needed for UI previews).

    Fail-open at every boundary: caller wraps in try/except and logs.
    The chunk-level ingest has already succeeded so any failure here
    leaves the system in a consistent state (document indexed at the
    chunk tier; summary tier simply absent until the backfill script
    fills it in later).
    """
    import logging as _log
    import uuid as _uuid
    log = _log.getLogger("orgchat.ingest")

    from .doc_summarizer import summarize_document

    chat_url = os.environ.get("OPENAI_API_BASE_URL")
    if not chat_url:
        log.debug("RAG_DOC_SUMMARIES=1 but OPENAI_API_BASE_URL unset — skipping")
        return
    chat_model = os.environ.get("SUMMARY_MODEL",
                                os.environ.get("CHAT_MODEL", "orgchat-chat"))
    api_key = os.environ.get("OPENAI_API_KEY")

    summary = await summarize_document(
        chunks=chunks_texts,
        filename=filename,
        chat_url=chat_url,
        chat_model=chat_model,
        api_key=api_key,
        timeout=float(os.environ.get("RAG_DOC_SUMMARY_TIMEOUT", "30.0")),
    )
    if not summary:
        log.debug("empty summary for doc_id=%s — skipping summary point", doc_id)
        return

    # Embed the summary as a single-element batch.
    [summary_vec] = await embedder.embed([summary])

    summary_payload: dict = dict(payload_base)
    summary_payload["chunk_index"] = -1
    summary_payload["text"] = summary
    summary_payload["uploaded_at"] = now_ns
    summary_payload["deleted"] = False
    summary_payload["model_version"] = pipeline_version
    summary_payload["level"] = "doc"
    summary_payload["kind"] = "doc_summary"
    summary_payload["filename"] = filename
    # Keep the structural fields present so payload shape is homogenous
    # with chunk points.
    summary_payload.setdefault("page", None)
    summary_payload.setdefault("heading_path", [])
    summary_payload.setdefault("sheet", None)

    point_id = str(_uuid.uuid5(_POINT_NS, f"doc:{doc_id}:doc_summary"))
    point = {"id": point_id, "vector": summary_vec, "payload": summary_payload}

    await vector_store.upsert(collection, [point])

    # Mirror into Postgres. Import lazily — ingest.py doesn't currently
    # depend on the chat_rag_bridge module registry, so we reach into its
    # configured sessionmaker (set by ext.app.build_ext_routers at
    # startup). If unset (unlikely in prod, possible in tests), skip the
    # mirror silently — Qdrant is the retrieval source of truth.
    try:
        from .chat_rag_bridge import _sessionmaker as _sm
        if _sm is not None:
            from sqlalchemy import text as _sql_text
            async with _sm() as s:
                await s.execute(
                    _sql_text(
                        "UPDATE kb_documents SET doc_summary = :s WHERE id = :d"
                    ),
                    {"s": summary, "d": doc_id},
                )
                await s.commit()
    except Exception as e:  # noqa: BLE001 — best-effort mirror
        log.warning("kb_documents.doc_summary mirror failed doc_id=%s: %s", doc_id, e)
