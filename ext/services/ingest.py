"""Extract → chunk → embed → upsert pipeline.

Each extracted block carries structural metadata (``page`` / ``heading_path`` /
``sheet``). We chunk the block's text independently and inherit the block's
metadata onto every resulting chunk so Qdrant payloads can surface hints like
"from page 7" or "under heading 'Rollout'" at retrieval time.
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import uuid
from typing import Mapping

from . import flags
from .chunker import chunk_text
from .embedder import Embedder
from .extractor import extract
from .kb_config import get_chunking_strategy
from .ocr import OCRBackend, ocr_pdf, select_ocr_backend
from .pipeline_version import current_version
from .temporal_shard import ShardKeyOrigin, extract_shard_key
from .vector_store import VectorStore


log = logging.getLogger("orgchat.ingest")


def _sharding_enabled() -> bool:
    """Read RAG_SHARDING_ENABLED at call time so tests / operators can toggle.

    Default OFF — when off, the ingest path is byte-identical to pre-Phase-5.2:
    no shard_key payload, normal ``upsert`` (not ``upsert_temporal``).
    Plan B Phase 5.2.
    """
    return os.environ.get("RAG_SHARDING_ENABLED", "0") == "1"


def _hybrid_enabled() -> bool:
    """Read RAG_HYBRID at call time so tests can toggle it without reimport.

    Default on as of 2026-04-19 — eval showed +12pp chunk_recall at +3ms.
    Set RAG_HYBRID=0 to force dense-only. Any non-"0" value means "on".
    Runtime fallback: even with hybrid on, ingest only computes sparse vectors
    when the target collection was created with sparse support (via
    ``_collection_has_sparse``) — legacy collections remain dense-only.
    """
    return os.environ.get("RAG_HYBRID", "1") != "0"


def _colbert_enabled() -> bool:
    """Read RAG_COLBERT at call time so tests / operators can toggle it.

    Default OFF — ColBERT model isn't part of the default fastembed
    cache hydration (Appendix A.2) and produces large multi-vector
    payloads (~10kB / chunk for typical 60-token chunks). When OFF, the
    embedder.colbert_embed function is never imported, so the default
    ingest path stays byte-identical to pre-Task-3.4 behaviour.
    Same gating semantics as RAG_HYBRID: any non-"0" value enables.
    """
    return os.environ.get("RAG_COLBERT", "0") == "1"


def _contextualize_enabled() -> bool:
    """Read RAG_CONTEXTUALIZE_KBS at call time.

    Default OFF. When OFF, the contextualizer module is not imported here,
    so the default ingest path stays byte-identical to the pre-P2.7
    behaviour (no chat-model calls, no httpx churn, no extra imports).

    Retained for legacy callers; new code should use ``should_contextualize``
    so per-KB ``rag_config`` opt-in/out is honoured.
    """
    return os.environ.get("RAG_CONTEXTUALIZE_KBS", "0") == "1"


def should_contextualize(*, env_flag: str | None, kb_rag_config: dict | None) -> bool:
    """Decide whether to contextualize for a given ingest based on the
    per-KB rag_config first (explicit opt-in/out), falling back to global env.

    Precedence: per-KB value wins (True OR False — both are explicit
    operator decisions; we don't want a global ``"1"`` to override a KB
    that opted out, nor a global ``"0"`` to suppress a KB that opted in).
    Missing per-KB key → fall back to the global env flag.

    Args:
        env_flag: raw value of ``RAG_CONTEXTUALIZE_KBS`` (or any other env
            string). ``None`` / ``""`` / ``"0"`` all mean disabled; only
            the literal string ``"1"`` enables. This matches the
            convention used by ``_contextualize_enabled`` and the rest of
            the RAG_* env reads in this module.
        kb_rag_config: the merged per-KB rag_config dict (post
            ``kb_config.merge_configs`` for multi-KB ingest, or a single
            KB's raw config). Looks for the literal key ``"contextualize"``;
            ignores ``contextualize_on_ingest`` (a separate, future-
            facing key kept in the kb_config schema for retrieval-time
            tracking but not consulted here — that key is for the request
            overlay path, this helper is the ingest-time gate).

    Returns:
        ``True`` to run the chunk-context augmentation pass, ``False``
        to skip it (default-off path stays byte-identical).
    """
    if kb_rag_config and "contextualize" in kb_rag_config:
        return bool(kb_rag_config["contextualize"])
    return (env_flag or "0") == "1"


def should_caption_images(
    *, env_flag: str | None, kb_rag_config: dict | None,
) -> bool:
    """Decide whether to extract+caption images for a given ingest.

    Mirrors :func:`should_contextualize` precedence: per-KB explicit
    value wins (True or False — both are explicit operator decisions),
    falling back to the global ``RAG_IMAGE_CAPTIONS`` env flag.

    The reason this matters: per-image vision-LLM round-trips dominate
    ingest wall-time on PDFs with many diagrams, and routinely OOM the
    celery worker on memory-pressured deployments. A pure-text KB
    (e.g., a security-policy doc whose diagrams are decorative) can
    opt out via ``rag_config.image_captions=false`` while the env
    default stays on for image-rich corpora.

    Args:
        env_flag: raw value of ``RAG_IMAGE_CAPTIONS``. Only the literal
            ``"1"`` enables; everything else (including ``None``)
            disables.
        kb_rag_config: the merged per-KB rag_config dict. Looks for the
            literal key ``"image_captions"``.

    Returns:
        ``True`` to run image extraction + captioning, ``False`` to
        skip it (no images extracted, no chunks emitted).
    """
    if kb_rag_config and "image_captions" in kb_rag_config:
        return bool(kb_rag_config["image_captions"])
    return (env_flag or "0") == "1"


def _raptor_enabled() -> bool:
    """Read RAG_RAPTOR at call time via the flags overlay.

    Default OFF. When OFF, the raptor module is not imported here, so the
    default ingest path stays byte-identical to the pre-P3.4 behaviour
    (no tree building, no extra chat-model calls, no sklearn import).
    Per-KB overrides from ``rag_config`` flow through ``flags.get``.
    """
    return flags.get("RAG_RAPTOR", "0") == "1"


async def _persist_doc_pipeline_version(
    doc_id: int | str,
    pipeline_version: str,
) -> None:
    """Refresh ``kb_documents.pipeline_version`` to reflect the actual
    pipeline that produced the embeddings.

    Bug-fix campaign §1.12: ``upload.py`` stamps the column at upload
    time (``ctx=none`` because contextualize is a runtime decision based
    on per-KB ``rag_config``). When ``ingest_bytes`` later flips
    ``context_augmented=True`` and stamps every Qdrant point's
    ``pipeline_version`` payload with ``ctx=contextual-v1``, the
    Postgres column was left lying — the kb_admin drift dashboard and
    ``reembed_all.py`` checkpoint thought the doc was un-contextualized.

    Best-effort: any DB error is logged + swallowed. The Qdrant points
    already carry the correct value; the column update is a metadata
    convenience. Indirected via the ``ingest_worker`` engine singleton
    (lazy import — keeps ingest.py free of any sqlalchemy import when
    the column update isn't needed) so tests can monkeypatch this whole
    function on the ``ext.services.ingest`` module.
    """
    if doc_id is None:
        return
    try:
        from ..workers.ingest_worker import _get_engine
        from sqlalchemy import text as _sql
        engine = _get_engine()
        if engine is None:
            return
        async with engine.begin() as conn:
            await conn.execute(
                _sql(
                    "UPDATE kb_documents SET pipeline_version = :pv "
                    "WHERE id = :i"
                ),
                {"pv": pipeline_version, "i": int(doc_id)},
            )
    except Exception as e:  # noqa: BLE001 — best-effort
        import logging as _log
        _log.getLogger("orgchat.ingest").warning(
            "ingest: failed to refresh kb_documents.pipeline_version "
            "doc_id=%s pv=%s err=%s",
            doc_id, pipeline_version, e,
        )


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


def build_point_payload(
    *,
    kb_id: int,
    doc_id: int | None,
    subtag_id: int | None,
    filename: str,
    owner_user_id: str,
    chunk_meta: Mapping[str, object],
    chat_id: str | None = None,
    level: str = "chunk",
) -> dict:
    """Return the canonical Qdrant payload dict for one point.

    Centralizes payload construction so the upsert loop in ``ingest_bytes``
    (and any future callers — eval harness, repair scripts) all produce
    payloads with the same shape and the same field set as
    ``ext/db/qdrant_schema.py``'s ``canonical_payload_schema``. Phase 3.2
    adds the ``context_prefix`` field; centralizing means the next field
    addition only changes one spot.

    The function is pure: no I/O, no mutation of inputs. The caller
    attaches the vector + sparse_vector and computes the point ID
    separately — vectors are large and shouldn't round-trip through this
    helper, and point IDs depend on the caller's deterministic scheme.

    Args:
        kb_id: knowledge-base ID this point belongs to.
        doc_id: ``kb_documents.id`` for KB uploads, or ``None`` for
            ephemeral chat-scoped uploads (in which case ``chat_id``
            must be set).
        subtag_id: optional ``kb_subtags.id`` (None → unfiled in KB).
        filename: original filename — surfaced at retrieval time for
            citations and KB catalog rendering.
        owner_user_id: principal who owns this point. RBAC filters use
            this for chat-scoped (private) uploads.
        chunk_meta: dict carrying per-chunk fields. Required keys:
            ``text``, ``chunk_index``. Optional keys (all canonical):
            ``context_prefix``, ``page``, ``heading_path``, ``sheet``,
            ``uploaded_at``, ``deleted``, ``model_version``,
            ``chunk_level``, ``source_chunk_ids``, ``kind``.
        chat_id: chat namespace for ephemeral uploads; mutually
            exclusive with ``doc_id`` for KB uploads but the schema
            tolerates both being present (filters key off doc_id when set).
        level: 'chunk' (default) for chunk-level points or 'doc' for
            tier-1 doc-summary points.

    Returns:
        Dict matching the canonical Qdrant payload shape, including the
        ``context_prefix`` field (None when no prefix was generated).
    """
    payload: dict = {
        "kb_id": kb_id,
        "doc_id": doc_id,
        "subtag_id": subtag_id,
        "owner_user_id": owner_user_id,
        "chat_id": chat_id,
        "filename": filename,
        "level": level,
        "chunk_index": chunk_meta.get("chunk_index"),
        "text": chunk_meta.get("text"),
        "context_prefix": chunk_meta.get("context_prefix"),
        "page": chunk_meta.get("page"),
        "heading_path": list(chunk_meta.get("heading_path") or []),
        "sheet": chunk_meta.get("sheet"),
    }
    # Optional canonical fields — only stamp when the caller provided them
    # so callers that don't compute these (tests, simple repair scripts)
    # produce minimal but still-valid payloads.
    # Plan B Phase 6.6 / 6.7: structural fields stamped by chunker dispatch
    # (``chunk_text_for_kb`` / ``extract_images_as_chunks``).
    for k in ("uploaded_at", "deleted", "model_version", "chunk_level",
              "source_chunk_ids", "kind",
              "chunk_type", "language", "continuation"):
        if k in chunk_meta:
            payload[k] = chunk_meta[k]
    return payload


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
    kb_rag_config: dict | None = None,
) -> int:
    """Full ingest: returns number of chunks upserted.

    ``kb_rag_config`` is the per-KB ``rag_config`` JSONB dict (or ``None``
    for callers that don't have one — chat-private uploads, repair
    scripts, tests). Currently consulted only for the ingest-time
    ``contextualize`` opt-in (Phase 3.3); future ingest-time keys
    (e.g. raptor opt-in, doc_summaries opt-in per-KB) will read from the
    same dict. Default ``None`` preserves byte-identical behaviour for
    callers that don't opt in.
    """
    blocks = extract(data, mime_type, filename)
    if not blocks:
        return 0

    # Plan B Phase 5.2 — derive a per-document shard_key once when temporal
    # sharding is enabled. The same key is stamped on every chunk's payload
    # and used as the ``shard_key_selector`` on the upsert. Default OFF —
    # legacy ingest path is byte-identical.
    sharding_on = _sharding_enabled()
    doc_shard_key: str | None = None
    doc_shard_origin: ShardKeyOrigin | None = None
    if sharding_on:
        # Use the first block's text as the body sample for date extraction.
        # Most ingest pipelines emit blocks in source order so block[0] is
        # the document head — the same place YAML frontmatter and the
        # opening date typically live.
        body_sample = blocks[0].text if blocks else ""
        doc_shard_key, doc_shard_origin = extract_shard_key(
            filename=filename, body=body_sample,
        )
        import logging as _log
        _log.getLogger("orgchat.ingest").info(
            "ingest doc=%s shard_key=%s origin=%s",
            filename, doc_shard_key, doc_shard_origin.value,
        )

    # Chunk per block via the per-KB dispatcher (Plan B Phase 6.6).
    # ``chunk_text_for_kb`` reads ``rag_config.chunking_strategy`` and
    # double-gates on ``RAG_STRUCTURED_CHUNKER`` env. Default 'window'
    # behaves byte-identically to the legacy chunk_text path.
    # We carry per-chunk extras (chunk_type / language / continuation)
    # in a parallel list keyed by global chunk index.
    from .chunker import Chunk as _Chunk
    paired: list[tuple[object, object]] = []  # (Chunk, ExtractedBlock)
    chunk_extras: list[dict] = []  # parallel to ``paired``
    for b in blocks:
        chunk_dicts = chunk_text_for_kb(
            text=b.text,
            rag_config=kb_rag_config,
            chunk_size_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )
        for cd in chunk_dicts:
            paired.append((
                _Chunk(index=len(paired), text=cd["text"]), b,
            ))
            extras: dict = {}
            ct = cd.get("chunk_type", "prose")
            if ct != "prose":
                extras["chunk_type"] = ct
            if cd.get("language"):
                extras["language"] = cd["language"]
            if cd.get("continuation"):
                extras["continuation"] = True
            chunk_extras.append(extras)

    # Plan B Phase 6.7 — image-caption chunks. Quadruple-gated:
    # (1) PDF mime, (2) per-KB ``image_captions`` rag_config (if set,
    # wins absolute), (3) global ``RAG_IMAGE_CAPTIONS=1`` env, (4)
    # vision service reachable. The per-KB gate runs HERE so a text-
    # only KB can short-circuit the entire image extraction + vision
    # LLM round-trip — the helper's internal env gate is left in
    # place as a final safety net.
    _do_captions = (
        mime_type == "application/pdf"
        and should_caption_images(
            env_flag=os.environ.get("RAG_IMAGE_CAPTIONS"),
            kb_rag_config=kb_rag_config,
        )
    )
    if _do_captions:
        try:
            image_chunks = await extract_images_as_chunks(
                pdf_bytes=data, filename=filename,
            )
        except Exception:  # noqa: BLE001 — fail-open
            image_chunks = []
    else:
        image_chunks = []
    # Bug-fix campaign §1.13: pair each image chunk with the block whose
    # ``page`` matches the image's page so the downstream payload
    # inherits the *page-local* heading_path. The previous
    # ``blocks[0]`` fallback caused images on page 5 of a 100-page PDF
    # to inherit the page-1 chapter heading — citations and faceted
    # retrieval were silently mis-attributed. When no block matches
    # (image on a page that yielded no extracted text, or no ``page``
    # in the image dict), we pair with ``None`` so heading_path falls
    # back to ``[]`` and the payload is honest about its provenance.
    _block_by_page = {b.page: b for b in blocks if b.page is not None}
    for ic in image_chunks:
        page = ic.get("page")
        host_block = _block_by_page.get(page) if page is not None else None
        paired.append((
            _Chunk(index=len(paired), text=ic["text"]), host_block,
        ))
        extras: dict = {"chunk_type": "image_caption"}
        if page is not None:
            extras["page"] = page
        if ic.get("language"):
            extras["language"] = ic["language"]
        chunk_extras.append(extras)

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
    if should_contextualize(
        env_flag=os.environ.get("RAG_CONTEXTUALIZE_KBS"),
        kb_rag_config=kb_rag_config,
    ):
        context_augmented = await _maybe_contextualize_chunks(
            paired, doc_title=filename
        )

    texts = [c.text for c, _ in paired]

    # Bug-fix campaign §3.8 — run dense / sparse / colbert embedders
    # concurrently. The three are independent (each consumes ``texts``;
    # none reads another's output) so they can interleave. Dense is
    # async network I/O (TEI HTTP). Sparse and ColBERT are synchronous
    # ONNX runs — wrapped in ``asyncio.to_thread`` so they don't block
    # the event loop while dense waits on the network. Net effect on a
    # large batch: the slowest of the three caps total embed time
    # instead of the sum (was: dense + sparse + colbert; now: max).
    #
    # Each arm's two-gate logic (flag + collection has slot) is
    # preserved — when a gate is closed, the coroutine resolves to the
    # ``None``-filled fallback list immediately without touching the
    # underlying embedder.
    async def _dense_arm() -> list[list[float]]:
        return await embedder.embed(texts)

    async def _sparse_arm() -> list[tuple[list[int], list[float]] | None]:
        # Sparse vectors are only computed when hybrid is on AND the
        # target collection was created with sparse support. When
        # either condition fails we produce no sparse vectors and the
        # upsert path takes the legacy dense-only shape (byte-identical
        # to the pre-hybrid behaviour). We use getattr with defaults so
        # test doubles / minimal VectorStore substitutes that don't
        # implement the sparse detection helpers still work.
        if not _hybrid_enabled():
            return [None] * len(paired)
        refresh = getattr(vector_store, "_refresh_sparse_cache", None)
        has_sparse = getattr(vector_store, "_collection_has_sparse", None)
        if refresh is None or has_sparse is None:
            return [None] * len(paired)
        try:
            await refresh(collection)
        except Exception:
            return [None] * len(paired)  # fall through — collection check failed
        if not has_sparse(collection):
            return [None] * len(paired)
        try:
            from .sparse_embedder import embed_sparse
            # ONNX run — blocks the event loop in-process. Push to a
            # thread so dense (network IO) and colbert (also ONNX,
            # different model) can overlap.
            return list(await asyncio.to_thread(embed_sparse, texts))  # type: ignore[arg-type]
        except Exception:
            # fastembed missing or failed — silently skip sparse arm.
            return [None] * len(paired)

    async def _colbert_arm() -> list[list[list[float]] | None]:
        # P3.4: ColBERT multi-vectors. Same two-gate pattern as sparse:
        # ``RAG_COLBERT=1`` AND the target collection was created with
        # the ``colbert`` named slot. When either is false we leave a
        # list of Nones and the upsert path skips the colbert arm —
        # leaving dense (and sparse, if applicable) byte-identical to
        # pre-Task-3.4. Failures (model missing, fastembed not
        # installed) silently fall through to dense+sparse only — never
        # block ingest.
        if not _colbert_enabled():
            return [None] * len(paired)
        refresh_cb = getattr(vector_store, "_refresh_colbert_cache", None)
        has_colbert = getattr(vector_store, "_collection_has_colbert", None)
        if refresh_cb is None or has_colbert is None:
            return [None] * len(paired)
        try:
            await refresh_cb(collection)
        except Exception:
            return [None] * len(paired)
        if not has_colbert(collection):
            return [None] * len(paired)
        try:
            from .embedder import colbert_embed
            # Same to_thread treatment as sparse — colbert_embed runs a
            # separate ONNX session, also blocking.
            return list(await asyncio.to_thread(colbert_embed, texts))  # type: ignore[arg-type]
        except Exception:
            return [None] * len(paired)

    vectors, sparse_vectors, colbert_vectors = await asyncio.gather(
        _dense_arm(), _sparse_arm(), _colbert_arm(),
    )

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

    # Bug-fix campaign §1.12: refresh kb_documents.pipeline_version when
    # contextualize ran. Upload-time stamps ``ctx=none`` (runtime decision);
    # the Qdrant points stamp ``ctx=contextual-v1``. Without this update,
    # the column drifts from the points and reembed_all + the kb_admin
    # drift dashboard mis-classify the doc. Skipped for chat-private
    # uploads (doc_id is None) since they have no kb_documents row.
    if context_augmented and doc_id is not None:
        try:
            await _persist_doc_pipeline_version(doc_id, pv)
        except Exception:  # noqa: BLE001 — fail-open
            pass

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

    # Identity fields used by build_point_payload — extracted once. We tolerate
    # legacy callers that pass kb_id/subtag_id as None or omit them entirely
    # (e.g. private-chat ingest where chat_id is the routing key).
    pb_kb_id = payload_base.get("kb_id")  # type: ignore[union-attr]
    pb_subtag_id = payload_base.get("subtag_id")  # type: ignore[union-attr]
    pb_owner = payload_base.get("owner_user_id")  # type: ignore[union-attr]
    pb_filename = payload_base.get("filename") or filename
    # Pull any extra payload_base fields that aren't part of the canonical
    # build_point_payload signature so we preserve them on the way out
    # (forward-compat: routers / scripts may stamp custom fields and the
    # legacy code path used to carry them via ``dict(payload_base)``).
    _CANONICAL_KEYS = {
        "kb_id", "subtag_id", "doc_id", "owner_user_id", "chat_id", "filename",
    }
    extra_payload_fields = {
        k: v for k, v in dict(payload_base).items() if k not in _CANONICAL_KEYS
    }
    # Preserve the legacy "lift forward only what was provided" contract:
    # if payload_base did NOT carry a given identity key, the resulting
    # payload should not stamp it either (private-chat ingest historically
    # produced payloads without ``doc_id``; ingest_bytes callers depend on
    # that — see ``test_doc_id_coercion.test_missing_doc_id_does_not_crash``).
    _absent_identity_keys = tuple(
        k for k in ("doc_id", "subtag_id", "kb_id", "chat_id", "owner_user_id")
        if k not in payload_base  # type: ignore[operator]
    )

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
            if node.level == 0 and nidx < len(paired):
                # Leaf: inherit the source block's structural metadata.
                src_chunk, block = paired[nidx]
                page = block.page
                heading_path = list(block.heading_path)
                sheet = block.sheet
                # Source chunk's context_prefix (set by contextualize_chunks_with_prefix
                # when contextual augmentation ran). Frozen-dataclass chunks may not
                # carry this attribute — getattr with default keeps the default-off
                # path byte-identical (no context_prefix in payload at all when None).
                ctx_prefix = getattr(src_chunk, "context_prefix", None)
            else:
                page = None
                heading_path = []
                sheet = None
                ctx_prefix = None
            chunk_meta: dict = {
                "chunk_index": nidx,
                "text": node.text,
                "context_prefix": ctx_prefix,
                "page": page,
                "heading_path": heading_path,
                "sheet": sheet,
                "uploaded_at": now,
                "deleted": False,
                "model_version": pv,
                "chunk_level": int(node.level),
                "source_chunk_ids": list(node.source_chunk_ids),
            }
            payload = build_point_payload(
                kb_id=pb_kb_id, doc_id=doc_id, subtag_id=pb_subtag_id,
                filename=pb_filename, owner_user_id=pb_owner,
                chunk_meta=chunk_meta, chat_id=chat_id, level="chunk",
            )
            # Re-stamp any extra payload_base fields (forward-compat for
            # custom payloads we don't recognize in the canonical schema).
            payload.update(extra_payload_fields)
            for k in _absent_identity_keys:
                payload.pop(k, None)
            # Plan B Phase 5.2: temporal shard_key stamp (default-off).
            if sharding_on and doc_shard_key is not None:
                payload["shard_key"] = doc_shard_key
                if doc_shard_origin is not None:
                    payload["shard_key_origin"] = doc_shard_origin.value

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
            # P3.4: only leaves (level 0, indexed by source chunk) get
            # ColBERT vectors. RAPTOR summary nodes (level >= 1) are LLM
            # paraphrases — token-level late interaction over a paraphrase
            # is not meaningful, and skipping them keeps payload bloat
            # bounded by the leaf count.
            if node.level == 0 and nidx < len(colbert_vectors):
                cv = colbert_vectors[nidx]
                if cv is not None:
                    point["colbert_vector"] = cv
            points.append(point)
    else:
        for gidx, ((chunk, block), vec) in enumerate(zip(paired, vectors)):
            extras = chunk_extras[gidx] if gidx < len(chunk_extras) else {}
            chunk_meta = {
                "chunk_index": gidx,
                "text": chunk.text,
                "context_prefix": getattr(chunk, "context_prefix", None),
                # block may be None for image_caption chunks (no host block).
                "page": extras.get("page", block.page if block is not None else None),
                "heading_path": (
                    list(block.heading_path) if block is not None else []
                ),
                "sheet": block.sheet if block is not None else None,
                "uploaded_at": now,
                "deleted": False,
                "model_version": pv,
            }
            # Plan B Phase 6.6 / 6.7 — structural / kind metadata from the
            # chunker dispatcher. ``chunk_type``, ``language``, ``continuation``
            # are surfaced verbatim so downstream retrieval / re-embedding
            # tooling can distinguish prose / table / code / image_caption.
            for k in ("chunk_type", "language", "continuation"):
                if k in extras:
                    chunk_meta[k] = extras[k]
            payload = build_point_payload(
                kb_id=pb_kb_id, doc_id=doc_id, subtag_id=pb_subtag_id,
                filename=pb_filename, owner_user_id=pb_owner,
                chunk_meta=chunk_meta, chat_id=chat_id, level="chunk",
            )
            payload.update(extra_payload_fields)
            for k in _absent_identity_keys:
                payload.pop(k, None)
            # Plan B Phase 5.2: temporal shard_key stamp (default-off).
            if sharding_on and doc_shard_key is not None:
                payload["shard_key"] = doc_shard_key
                if doc_shard_origin is not None:
                    payload["shard_key_origin"] = doc_shard_origin.value

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
            # P3.4 ColBERT multi-vector — attached only when both the
            # opt-in env flag is on AND the collection has the slot
            # (the embed list is all-Nones otherwise; the upsert path
            # also gates again on collection support so a stray
            # colbert_vector field on a non-colbert collection is a
            # no-op rather than a Qdrant error).
            cv = colbert_vectors[gidx]
            if cv is not None:
                point["colbert_vector"] = cv
            points.append(point)
    # Plan B Phase 5.2: route to per-shard upsert when temporal sharding is on.
    # The named shard receives the entire batch (one-doc-per-shard invariant
    # — ``doc_shard_key`` is derived per-document, not per-chunk).
    if sharding_on and doc_shard_key is not None:
        upsert_temporal = getattr(vector_store, "upsert_temporal", None)
        if upsert_temporal is not None:
            await upsert_temporal(
                collection, points, shard_key=doc_shard_key,
            )
        else:
            # Test double / minimal substitute: fall back to plain upsert.
            await vector_store.upsert(collection, points)
    else:
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
    # Wave 2 (review §2.8): doc-summary points have no logical "chunk_index"
    # in the source — the legacy -1 magic value conflicted with `WHERE
    # chunk_index >= 0` filters. The level="doc" + kind="doc_summary" fields
    # below are the canonical discriminator. Setting None instead of -1.
    # D-1's RRF dedup fix (commit 5b7ce80) handles None correctly.
    summary_payload["chunk_index"] = None
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

    # Stamp the same shard_key the chunk points used so this doc-summary
    # point lands in the correct temporal shard. Without this, upsert
    # into a custom-sharded collection raises "every point's payload
    # must include a 'shard_key'". Body sample reuses the first chunk
    # which mirrors what the chunk-side derivation does.
    if _sharding_enabled():
        from .temporal_shard import extract_shard_key as _extract_sk
        body_sample = chunks_texts[0] if chunks_texts else ""
        _sk, _sk_origin = _extract_sk(filename=filename, body=body_sample)
        summary_payload["shard_key"] = _sk
        summary_payload["shard_key_origin"] = _sk_origin.value

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


# ---------------------------------------------------------------------------
# Plan B Phase 6.4 — OCR fallback for PDFs with low extracted text.
#
# The wrapper detects pages where pdfplumber recovered fewer than the
# configured threshold of characters (default 50) and re-runs the page
# through the OCR pipeline (Tesseract by default — see ext/services/ocr.py).
# Per-KB ``ocr_policy`` overrides the global ``RAG_OCR_ENABLED`` flag so
# operators can disable OCR for KBs known to be all-text (faster ingest)
# or switch a specific KB to a cloud backend (operator opt-in only).
# ---------------------------------------------------------------------------

async def extract_pdf_with_ocr_fallback(
    *,
    pdf_bytes: bytes,
    filename: str,
    ocr_policy: dict | None,
) -> str:
    """Extract text via pdfplumber; OCR pages where text < threshold.

    Plan B Phase 6.4. Returns concatenated text. Per-page OCR is
    triggered when ``len(page_text) < trigger_chars``. Falls open at
    every layer:

    * RAG_OCR_ENABLED=0 → return raw pdfplumber text (default-off path
      stays byte-identical).
    * Per-KB ocr_policy.enabled=False → skip OCR even if the global flag
      is on.
    * No low-text pages → no OCR call (cheap path stays cheap).

    The OCR text replaces the pdfplumber text only for pages below the
    threshold; pages with sufficient text are kept as-is.
    """
    pages = _extract_pdf_text_per_page(pdf_bytes)

    if os.environ.get("RAG_OCR_ENABLED", "0") != "1":
        return "\n\n".join(pages)

    if ocr_policy and not ocr_policy.get("enabled", True):
        return "\n\n".join(pages)

    threshold = int(
        (ocr_policy or {}).get(
            "trigger_chars_per_page",
            os.environ.get("RAG_OCR_TRIGGER_CHARS", "50"),
        )
    )

    backend = select_ocr_backend(ocr_policy)
    language = (ocr_policy or {}).get("language", "eng")

    needs_ocr = any(len(p.strip()) < threshold for p in pages)
    if not needs_ocr:
        return "\n\n".join(pages)

    log.info(
        "ocr trigger: %s has %d/%d low-text pages",
        filename,
        sum(1 for p in pages if len(p.strip()) < threshold),
        len(pages),
    )
    ocr_text = await _ocr_pdf_pages(
        pdf_bytes, backend=backend, language=language,
    )

    # Splice OCR text in for low-text pages; keep pdfplumber text for
    # the rest. ocr_text is a single concatenated string (one segment per
    # page joined by "\n\n"); split + index by page so the output stays
    # aligned 1:1 with the pdfplumber page list. If OCR returned fewer
    # segments than pages (unlikely but possible — rasterize errors), the
    # fallback is to keep the pdfplumber text for that page.
    out_pages = []
    ocr_segments = ocr_text.split("\n\n")
    for i, p in enumerate(pages):
        if len(p.strip()) < threshold and i < len(ocr_segments):
            out_pages.append(ocr_segments[i])
        else:
            out_pages.append(p)
    return "\n\n".join(out_pages)


async def _ocr_pdf_pages(pdf_bytes, *, backend, language):
    """Indirection for tests to patch.

    Plan B Phase 6.4. Wraps ``ocr_pdf`` so the trigger threshold tests
    can stub the call site without touching the underlying OCR module.
    """
    return await ocr_pdf(pdf_bytes, backend=backend, language=language)


def _extract_pdf_text_per_page(pdf_bytes: bytes) -> list[str]:
    """Extract per-page PDF text via pdfplumber.

    Plan B Phase 6.4. Returns one string per page in source order so
    callers (notably ``extract_pdf_with_ocr_fallback``) can spot
    low-text pages and route them through OCR independently. Empty
    pages return ``""`` rather than ``None`` so downstream length
    checks are unambiguous.
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return pages


# ---------------------------------------------------------------------------
# Plan B Phase 6.6 — per-KB chunking strategy dispatch.
#
# ``chunk_text_for_kb`` is the new dispatch point: it reads
# ``rag_config.chunking_strategy`` (window or structured) and routes the
# extracted text to the right chunker. The window chunker keeps the
# pre-Plan-B byte-identical behavior; the structured chunker preserves
# tables + code blocks as atomic units (Plan B Phase 6.5).
#
# The structured path is double-gated by the env flag
# ``RAG_STRUCTURED_CHUNKER`` so an operator can globally roll it back
# without touching every KB row.
# ---------------------------------------------------------------------------

def chunk_text_for_kb(
    *,
    text: str,
    rag_config: dict | None,
    chunk_size_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[dict]:
    """Dispatch to the right chunker per KB strategy.

    Plan B Phase 6.6. Always returns a list of chunk dicts with at
    least ``text`` and ``chunk_type`` keys. The window chunker emits
    every chunk with ``chunk_type="prose"``; the structured chunker
    distinguishes prose / table / code (and stamps ``language`` for
    code chunks).
    """
    strategy = get_chunking_strategy(rag_config)
    if strategy == "structured" and \
            os.environ.get("RAG_STRUCTURED_CHUNKER", "0") == "1":
        from .chunker_structured import chunk_structured
        return chunk_structured(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    # Window default — wrap each Chunk's text into the dict shape.
    return [
        {"text": w, "chunk_type": "prose"}
        for w in _chunk_window(
            text,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


def _chunk_window(text, *, chunk_size_tokens=800, overlap_tokens=100):
    """Existing window-chunker entry point.

    Wraps ``ext.services.chunker.chunk_text`` so callers downstream of
    ``chunk_text_for_kb`` get a list of plain text strings (matching the
    legacy contract). ``chunk_text`` returns ``Chunk`` dataclasses; we
    flatten to ``.text`` strings here.
    """
    return [
        c.text for c in chunk_text(
            text,
            chunk_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    ]


# ---------------------------------------------------------------------------
# Plan B Phase 6.7 — image caption extraction.
#
# When a PDF carries embedded images (charts, screenshots, diagrams), we
# extract them, send each through the vllm-vision service, and emit a
# chunk with ``chunk_type="image_caption"``. This restores recall on
# visual-content queries that today silently lose the image content.
#
# Triple-gated soft-fail:
#   * RAG_IMAGE_CAPTIONS=0 (default) → never extract images at all
#   * Image extraction fails → return [] + log
#   * Vision service unreachable → skip per-image + tick
#     ``rag_image_skip_total`` metric
# ---------------------------------------------------------------------------

async def extract_images_as_chunks(
    *, pdf_bytes: bytes, filename: str,
) -> list[dict]:
    """Plan B Phase 6.7 + page-render fallback — emit image_caption chunks.

    Two-tier extraction:
      1. ``_extract_pdf_images`` returns embedded raster images (PDFs that
         actually carry photos / screenshots). Cheap and fast.
      2. Page-render fallback — when the PDF's diagrams are vector-drawn
         (network maps, flowcharts, org charts), pymupdf finds only the
         tiny decorative shapes. ``_render_pdf_pages_as_images`` renders
         each page to PNG at ``RAG_RENDER_PDF_DPI`` (default 200) and
         captions the page-image. Triggered when tier 1 yields no images
         of useful size (>= ``RAG_VISION_RASTER_MIN_BYTES``, default 5 KB).
         Disable with ``RAG_RENDER_PDF_PAGES_FOR_VISION=0``.

    Triple-gated soft-fail:
      * RAG_IMAGE_CAPTIONS=0 (default) → never extract images at all
      * Image extraction OR page-render fails → return [] + log
      * Vision service unreachable → skip per-image + tick
        ``rag_image_skip_total`` metric
    """
    if os.environ.get("RAG_IMAGE_CAPTIONS", "0") != "1":
        return []

    # Tier 1 — embedded rasters.
    try:
        images = await _extract_pdf_images(pdf_bytes)
    except Exception as e:
        log.warning("image extraction failed for %s: %s", filename, e)
        images = []

    # Filter to "useful" images. Tiny decorative shapes (e.g. the 263-byte
    # bullet glyphs in NFS.pdf) waste vision budget without conveying
    # information — they trigger generic captions and pollute retrieval.
    min_bytes = int(os.environ.get("RAG_VISION_RASTER_MIN_BYTES", "5000"))
    useful = [
        img for img in images
        if len(img.get("image_bytes") or b"") >= min_bytes
    ]

    # Tier 2 — page-render fallback for vector-drawn PDFs.
    if not useful and os.environ.get("RAG_RENDER_PDF_PAGES_FOR_VISION", "1") == "1":
        try:
            dpi = int(os.environ.get("RAG_RENDER_PDF_DPI", "200"))
            max_pages = int(os.environ.get("RAG_RENDER_PDF_MAX_PAGES", "50"))
            useful = await _render_pdf_pages_as_images(
                pdf_bytes, dpi=dpi, max_pages=max_pages,
            )
            log.info(
                "extract_images_as_chunks: %s — no embedded raster >= %d B; "
                "page-render fallback produced %d page images at %d DPI",
                filename, min_bytes, len(useful), dpi,
            )
        except Exception as e:
            log.warning(
                "page-render fallback failed for %s: %s", filename, e,
            )
            return []

    if not useful:
        return []

    out: list[dict] = []
    for img in useful:
        try:
            caption = await _caption_image(img["image_bytes"])
        except Exception as e:
            log.warning(
                "image caption failed for %s page %s: %s",
                filename, img.get("page"), e,
            )
            try:
                from .metrics import RAG_IMAGE_SKIP
                RAG_IMAGE_SKIP.inc()
            except Exception:
                pass
            continue
        if not caption:
            continue
        out.append({
            "text": caption,
            "chunk_type": "image_caption",
            "payload": {
                "page": img.get("page"),
                "position": img.get("position"),
            },
        })
    return out


async def _extract_pdf_images(pdf_bytes: bytes) -> list[dict]:
    """Use pymupdf to enumerate images. Returns list of dicts.

    Plan B Phase 6.7. Each entry is
    ``{"page": int, "image_bytes": bytes, "position": tuple}``.
    """
    import pymupdf

    out = []
    with pymupdf.open(stream=pdf_bytes) as doc:
        for page_num, page in enumerate(doc, start=1):
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base_img = doc.extract_image(xref)
                out.append({
                    "page": page_num,
                    "image_bytes": base_img["image"],
                    "position": (img_idx,),
                })
    return out


async def _render_pdf_pages_as_images(
    pdf_bytes: bytes, *, dpi: int = 200, max_pages: int = 50,
) -> list[dict]:
    """Render each PDF page to a PNG raster.

    Used as a fallback when ``_extract_pdf_images`` returns no useful
    embedded images (typical for vector-drawn diagrams: pymupdf finds
    only the PDF's tiny decorative shapes, not the actual rendered
    graphic). Each rendered page is treated as a single image-caption
    candidate with the same payload shape as embedded-image extraction
    so downstream retrieval can render "see page N" hints.

    ``dpi``: 200 is a reasonable trade between OCR-quality (text inside
    diagrams stays legible at this resolution) and payload size
    (~300 KB per A4 page).
    ``max_pages``: cap to avoid runaway cost on long PDFs. Pages beyond
    the cap are skipped silently — the fallback is best-effort.
    """
    import pymupdf

    out: list[dict] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num in range(min(doc.page_count, max_pages)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            png = pix.tobytes("png")
            out.append({
                "page": page_num + 1,
                "image_bytes": png,
                "position": ("page-render",),
            })
    return out


async def _caption_image(image_bytes: bytes) -> str:
    """Send an image to vllm-vision and return the caption.

    Plan B Phase 6.7. Soft-fails if the vision service is unreachable —
    the caller catches the exception and ticks
    ``rag_image_skip_total``.
    """
    import base64

    import httpx

    vision_url = os.environ.get(
        "RAG_VISION_URL", "http://vllm-vision:8000/v1",
    )
    vision_model = os.environ.get("RAG_VISION_MODEL", "qwen2-vl-7b")
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": vision_model,
        "messages": [
            {"role": "user", "content": [
                {"type": "text",
                 "text": (
                     "Analyze this image and produce a structured description "
                     "preserving every load-bearing detail. Cover, in this order:\n"
                     "1. Layout — single-column / multi-column / header-body-footer / "
                     "diagram / table / mixed; reading order.\n"
                     "2. Text — verbatim transcription of all visible text in reading "
                     "order. Preserve numbers, dates, IDs, and proper nouns exactly.\n"
                     "3. Tables — render each as a Markdown table (header row + data "
                     "rows). Preserve cell content; keep row/column count exact.\n"
                     "4. Diagrams — name the diagram type (flowchart / sequence / "
                     "hierarchy / network / sankey / swim-lane / state machine). "
                     "Enumerate every node (box / circle / shape) with its label and "
                     "every edge (arrow / line) with its direction and any edge label. "
                     "Use Markdown bullet trees or `A -> B (label)` notation when the "
                     "structure is graph-shaped.\n"
                     "5. Visual hierarchy — emphasized text, headings, callouts, "
                     "color-coded groupings.\n"
                     "6. Charts / graphs — describe axes, scale, legend, and the 3–5 "
                     "key data points or trends.\n\n"
                     "Format as concise structured prose with markdown. No 'I think' / "
                     "'this appears'. If the image is purely decorative (logo, photo, "
                     "icon) with no informational content, say so in one line and stop."
                 )},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        "max_tokens": int(os.environ.get("RAG_VISION_MAX_TOKENS", "800")),
        "temperature": 0.0,
    }
    # Default 120s. Page-rendered diagram captions on a 31B vision-LLM
    # routinely take 30-90s; the prior 20s default silently dropped most
    # of them. Tunable per deploy.
    vision_timeout = float(os.environ.get("RAG_VISION_TIMEOUT_SECS", "120"))
    async with httpx.AsyncClient(timeout=vision_timeout) as c:
        r = await c.post(f"{vision_url}/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
