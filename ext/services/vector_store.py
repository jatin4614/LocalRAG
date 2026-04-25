"""Thin async wrapper over qdrant-client."""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from .metrics import qdrant_search_latency_seconds, qdrant_upsert_latency_seconds
from .obs import span
from time import perf_counter as _perf_counter


# --- P2.4 HNSW tuning knobs ---------------------------------------------------
# Env-tunable with sane defaults. Defaults match Qdrant's built-in defaults
# EXCEPT ``ef_construct`` (200, up from 100 — +2-3pp recall at modest index-time
# cost). All knobs are read lazily at call time so tests / operators can flip
# them via env without re-importing the module.


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def _hnsw_config_diff() -> qm.HnswConfigDiff:
    """Build the HNSW config applied at collection creation."""
    return qm.HnswConfigDiff(
        m=_env_int("RAG_QDRANT_M", 16),
        ef_construct=_env_int("RAG_QDRANT_EF_CONSTRUCT", 200),
        full_scan_threshold=_env_int("RAG_QDRANT_FULL_SCAN_THRESHOLD", 10000),
    )


# --- P3.2 scalar quantization -----------------------------------------------
# INT8 scalar quantization @ quantile=0.99 gives ~4× vector-RAM reduction with
# <2% recall loss. Binary quantization is intentionally skipped — bge-m3 is
# 1024d, the known borderline where binary starts degrading.


def _quantization_config() -> qm.ScalarQuantization:
    """Build the ScalarQuantization config for new collections.

    ``always_ram=True`` keeps the 4×-smaller quantized index in RAM for fast
    approximate search; originals spill to disk (``on_disk=True`` on
    VectorParams) and are only loaded when a query opts into rescoring.
    """
    return qm.ScalarQuantization(
        scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8,
            quantile=_env_float("RAG_QDRANT_QUANTILE", 0.99),
            always_ram=True,
        )
    )


def _should_quantize() -> bool:
    """Default-on switch for creating new collections with quantization.

    Off by default so zero-config deployments keep byte-identical behaviour.
    Operators opt in via ``RAG_QDRANT_QUANTIZE=1``.
    """
    return _env_bool("RAG_QDRANT_QUANTIZE", False)


def _should_rescore() -> bool:
    """Per-query rescore switch. On by default when quantization is involved.

    Env knob ``RAG_QDRANT_RESCORE=0`` disables globally. The per-call
    ``rescore=False`` kwarg on ``search`` / ``hybrid_search`` always wins.
    """
    return _env_bool("RAG_QDRANT_RESCORE", True)


def _env_oversampling() -> float:
    """Oversampling ratio used during quantized rescoring.

    With oversampling=2.0, Qdrant pulls 2× the requested limit via the
    quantized index, then rescores those candidates with the original fp32
    vectors and returns the final top-N. Higher → better recall, more I/O.
    """
    return _env_float("RAG_QDRANT_OVERSAMPLING", 2.0)


def _quantization_search_params() -> qm.QuantizationSearchParams:
    """Build the per-query QuantizationSearchParams (rescore=True, oversampling)."""
    return qm.QuantizationSearchParams(
        rescore=True,
        oversampling=_env_oversampling(),
    )


def _search_params(*, rescore: Optional[bool] = None) -> qm.SearchParams:
    """Build the per-query SearchParams (hnsw_ef + optional quantization hint).

    ``rescore`` (default None → env-driven):
      * True  → attach QuantizationSearchParams(rescore=True, oversampling=2)
      * False → omit quantization params (quantized approx only, no rescore)
      * None  → follow ``RAG_QDRANT_RESCORE`` env (default True)

    For legacy, unquantized collections the ``quantization=`` arg is a no-op
    on Qdrant's side, so it's always safe to include.
    """
    if rescore is None:
        rescore = _should_rescore()
    qp = _quantization_search_params() if rescore else None
    return qm.SearchParams(
        hnsw_ef=_env_int("RAG_QDRANT_EF", 128),
        quantization=qp,
    )

_PAYLOAD_FIELDS = [
    "text", "kb_id", "subtag_id", "doc_id", "chat_id",
    "filename", "chunk_index", "deleted",
    # P0.4 structural metadata + pipeline provenance.
    "page", "heading_path", "sheet", "model_version",
    # P2.2 tenant-owner attribution — indexed as tenant field for RBAC
    # hot-path filters (admin-wide scoped queries skip other owners' subgraphs).
    "owner_user_id",
    # P3.4 RAPTOR tree provenance: ``chunk_level`` (0 leaf, 1+ summary),
    # ``source_chunk_ids`` (leaf indices a summary node covers).
    "chunk_level", "source_chunk_ids",
    # Tier 1 doc-summary index (2026-04-22): ``level`` distinguishes
    # summary points (``"doc"``) from chunks; ``kind`` is the producer
    # tag (``"doc_summary"``). Required in the allowlist so retriever
    # post-filters on ``level`` actually see the field.
    "level", "kind",
]

# Name of the named dense vector when a collection is created with sparse vectors
# alongside. Legacy collections (no sparse) use the unnamed single-vector form,
# so ``_DENSE_NAME`` is *only* used on hybrid-enabled collections.
_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


# P2.3: Single consolidated collection for all private chat documents.
# Replaces the unbounded-growth pattern of ``chat_{chat_id}`` — one collection
# per chat — with a single hybrid-shaped collection tenant-partitioned on
# ``chat_id`` + ``owner_user_id``. Qdrant's ``is_tenant=True`` indexes
# (registered in ``ensure_collection`` above) make those filters
# O(tenant-size) not O(total-points), so one big collection scales where
# N per-chat collections did not.
CHAT_PRIVATE_COLLECTION = "chat_private"


@dataclass
class Hit:
    id: int | str | uuid.UUID
    score: float
    payload: dict


class VectorStore:
    def __init__(self, *, url: str, vector_size: int, distance: str = "Cosine") -> None:
        # P2.4: configurable connection pool + longer timeout so batch upserts
        # under high concurrency don't starve or time out. ``pool_size`` maps
        # to httpx's max_connections; ``timeout`` is the per-request deadline
        # (the default 5s is fine for reads but tight for big upsert batches).
        self._client = AsyncQdrantClient(
            url=url,
            timeout=30.0,
            pool_size=_env_int("RAG_QDRANT_MAX_CONNS", 32),
        )
        self._vector_size = vector_size
        self._distance = distance
        self._known: set[str] = set()
        # Cache of collection name → bool (does this collection have a sparse
        # named vector?). Populated lazily; cleared when collection is deleted.
        self._sparse_cache: dict[str, bool] = {}

    async def close(self) -> None:
        await self._client.close()

    async def list_collections(self) -> list[str]:
        cols = (await self._client.get_collections()).collections
        return [c.name for c in cols]

    async def ensure_collection(
        self,
        name: str,
        *,
        with_sparse: bool = False,
        with_quantization: Optional[bool] = None,
    ) -> None:
        """Idempotently create a collection.

        By default creates a legacy-shaped collection with a single unnamed
        dense vector (backward-compatible with every collection created before
        this change). When ``with_sparse=True``, creates a collection with both
        a named ``dense`` vector AND a named ``bm25`` sparse vector (IDF
        modifier) — required for server-side hybrid retrieval via RRF.

        P3.2: ``with_quantization`` opts the collection into scalar INT8
        quantization (``ScalarQuantization(INT8, quantile=0.99, always_ram=True)``).
        When ``None`` (default), the ``RAG_QDRANT_QUANTIZE`` env switch decides.
        When enabled, the dense VectorParams also gets ``on_disk=True`` so the
        original fp32 vectors spill to disk (only paged in for rescoring);
        the 4×-smaller quantized index stays RAM-resident.

        This method never upgrades an existing collection; if ``name`` already
        exists with the wrong shape, callers must recreate it externally
        (or use ``scripts/enable_quantization.py`` to retrofit quantization
        settings on-the-fly — Qdrant rebuilds the quantized index in the
        background without rewriting the raw vectors).
        """
        if name in self._known:
            return
        # P2.4: HNSW tuning — built from env on every create call so operators
        # can change defaults without restarting. ``on_disk_payload`` lets very
        # large KBs spill payloads to disk (keep RAM for HNSW graph).
        hnsw = _hnsw_config_diff()
        on_disk_payload = _env_bool("RAG_QDRANT_ON_DISK_PAYLOAD", False)
        # P3.2: per-call arg wins; env is the fallback. ``None`` means "ask env".
        if with_quantization is None:
            with_quantization = _should_quantize()
        quant_config = _quantization_config() if with_quantization else None
        # Originals-on-disk pairs with always-RAM quantized — that's the whole
        # tier trade: tiny fast index in RAM, big precise index on disk.
        dense_on_disk = True if with_quantization else None
        try:
            if with_sparse:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        _DENSE_NAME: qm.VectorParams(
                            size=self._vector_size,
                            distance=qm.Distance[self._distance.upper()],
                            on_disk=dense_on_disk,
                        ),
                    },
                    sparse_vectors_config={
                        _SPARSE_NAME: qm.SparseVectorParams(
                            modifier=qm.Modifier.IDF,
                        ),
                    },
                    hnsw_config=hnsw,
                    on_disk_payload=on_disk_payload,
                    quantization_config=quant_config,
                )
                self._sparse_cache[name] = True
            else:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=qm.VectorParams(
                        size=self._vector_size,
                        distance=qm.Distance[self._distance.upper()],
                        on_disk=dense_on_disk,
                    ),
                    hnsw_config=hnsw,
                    on_disk_payload=on_disk_payload,
                    quantization_config=quant_config,
                )
                self._sparse_cache[name] = False
        except Exception:
            # Already exists — verify via BOTH collections and aliases before
            # giving up. Qdrant rejects create_collection when ``name`` matches
            # an existing alias ("Alias with the same name already exists")
            # even though aliases resolve transparently on every other
            # operation (search/upsert/payload_index), so treat an
            # alias-hit the same as a collection-hit.
            cols = await self.list_collections()
            aliases: set[str] = set()
            try:
                resp = await self._client.get_aliases()
                aliases = {a.alias_name for a in resp.aliases}
            except Exception:
                pass
            if name not in cols and name not in aliases:
                raise
        # Create payload indexes for fast filtered queries (idempotent — Qdrant
        # silently ignores duplicate index creation).
        #
        # Tenant fields get ``KeywordIndexParams(is_tenant=True)``: hints to
        # Qdrant 1.11+ that each unique value partitions the data into a
        # tenant sub-graph, unlocking filtered-HNSW optimizations that skip
        # cross-tenant sub-graphs during search. Within-tenant filters
        # (subtag/doc/deleted) stay as plain KEYWORD — those discriminate
        # rows within a tenant, not between tenants.
        # ``chat_id`` + ``owner_user_id`` are UUID strings in payload and get
        # ``KeywordIndexParams(is_tenant=True)`` — Qdrant's filtered-HNSW
        # tenant optimization only applies to keyword indexes, so true
        # string-tenants stay here. ``kb_id``, ``subtag_id``, ``doc_id`` are
        # int-typed in payload (autoincrement Postgres PKs) and need
        # ``IntegerIndexParams``; a keyword index over ints silently indexes
        # nothing — previous bug that caused ``payload_schema.kb_id.points=0``
        # on all KB collections. ``deleted`` is boolean-ish, stays keyword.
        tenant_fields = ("chat_id", "owner_user_id")
        int_filter_fields = ("kb_id", "subtag_id", "doc_id")
        filter_fields = ("deleted",)
        for field in tenant_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=qm.KeywordIndexParams(
                        type="keyword",
                        is_tenant=True,
                    ),
                )
            except Exception:
                pass  # index already exists
        for field in int_filter_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=qm.IntegerIndexParams(
                        type="integer",
                        lookup=True,
                        range=False,
                    ),
                )
            except Exception:
                pass  # index already exists
        for field in filter_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # index already exists
        self._known.add(name)

    async def delete_collection(self, name: str) -> None:
        try:
            await self._client.delete_collection(name)
        except Exception:
            pass
        self._known.discard(name)
        self._sparse_cache.pop(name, None)

    async def _refresh_sparse_cache(self, name: str) -> bool:
        """Ask Qdrant whether ``name`` has the ``bm25`` sparse named vector.

        Cached; legacy collections always return False. Safe to call on a
        non-existent collection (returns False).
        """
        if name in self._sparse_cache:
            return self._sparse_cache[name]
        try:
            info = await self._client.get_collection(collection_name=name)
        except Exception:
            self._sparse_cache[name] = False
            return False
        sparse = getattr(info.config.params, "sparse_vectors", None) if info and info.config else None
        has = bool(sparse) and _SPARSE_NAME in sparse
        self._sparse_cache[name] = has
        return has

    def _collection_has_sparse(self, name: str) -> bool:
        """Return cached sparse-support flag without hitting the network.

        Callers should ensure ``_refresh_sparse_cache(name)`` has been awaited
        at least once for ``name`` (typically during ``ensure_collection`` or
        the first retrieval for that collection in this process). Returns
        False for unknown collections (fail-closed → dense-only fallback).
        """
        return self._sparse_cache.get(name, False)

    async def upsert(self, name: str, points: Iterable[dict]) -> None:
        """Upsert points. Each point may carry ``sparse_vector: (indices, values)``.

        When any point has a sparse_vector AND the collection supports sparse
        (hybrid-enabled), points are written in the named-vector form
        (``{dense: [...], bm25: SparseVector(...)}``). Otherwise the legacy
        single-unnamed-vector path is used (byte-identical to before).
        """
        points = list(points)
        # Decide encoding: sparse is only used if (a) any point carries it AND
        # (b) the collection was created with sparse support.
        has_sparse_points = any(p.get("sparse_vector") is not None for p in points)
        use_sparse = False
        if has_sparse_points:
            # Make sure the cache is warm — caller usually called
            # ensure_collection first, but fall back to a Qdrant lookup.
            await self._refresh_sparse_cache(name)
            use_sparse = self._collection_has_sparse(name)

        n_points = len(points)
        with span(
            "qdrant.upsert",
            collection=name,
            n_points=n_points,
            vector_size=self._vector_size,
            hybrid=use_sparse,
        ):
            _t = _perf_counter()
            try:
                if not use_sparse:
                    # Legacy path — byte-identical to pre-hybrid behavior.
                    pts = [
                        qm.PointStruct(
                            id=p["id"], vector=p["vector"], payload=p.get("payload", {})
                        )
                        for p in points
                    ]
                    await self._client.upsert(collection_name=name, points=pts, wait=True)
                    return

                # Hybrid path — pack dense under _DENSE_NAME and sparse under _SPARSE_NAME.
                pts = []
                for p in points:
                    vec_map: dict = {_DENSE_NAME: p["vector"]}
                    sv = p.get("sparse_vector")
                    if sv is not None:
                        indices, values = sv
                        vec_map[_SPARSE_NAME] = qm.SparseVector(indices=list(indices), values=list(values))
                    pts.append(
                        qm.PointStruct(id=p["id"], vector=vec_map, payload=p.get("payload", {}))
                    )
                await self._client.upsert(collection_name=name, points=pts, wait=True)
            finally:
                try:
                    qdrant_upsert_latency_seconds.observe(_perf_counter() - _t)
                except Exception:
                    pass

    @staticmethod
    def _build_filter(
        *,
        subtag_ids: Optional[list[int]] = None,
        doc_ids: Optional[list[int]] = None,
        owner_user_id: Optional[int | str] = None,
        chat_id: Optional[int | str] = None,
        level: Optional[str] = None,
    ) -> qm.Filter:
        """Build the standard Qdrant filter for every read path.

        When ``owner_user_id`` is passed, a ``must`` condition pins matches to
        that owner. Numeric strings are coerced to int so callers that pass
        ``"7"`` still match points stamped with integer ``7`` (defensive —
        mirrors ingest's ``doc_id`` coercion). Non-numeric strings (UUIDs from
        upstream Open WebUI) pass through unchanged.

        Default ``owner_user_id=None`` means no owner filter — byte-identical
        to the pre-P2.2 behaviour. This preserves KB retrieval semantics where
        every user with KB access sees every chunk regardless of uploader.

        P2.3: ``chat_id`` adds a second ``must`` condition pinning matches to
        a single chat. Required for reads against the consolidated
        ``chat_private`` collection (many chats share the same collection,
        tenant-partitioned by ``chat_id`` + ``owner_user_id``). Default
        ``chat_id=None`` = no chat filter = byte-identical to pre-P2.3.
        Chat ids are typically UUID strings from upstream Open WebUI; they
        pass through unchanged.
        """
        must_conditions = []
        if subtag_ids:
            must_conditions.append(
                qm.FieldCondition(key="subtag_id", match=qm.MatchAny(any=subtag_ids))
            )
        if doc_ids:
            # Tier-2 ``specific_date`` router uses this to pin retrieval to
            # the exact document(s) whose filename matches the date in the
            # query. Without the filter, ranking signals can't reliably
            # distinguish "5 Jan 2026" from "5 Feb 2026" or "4 Jan 2026"
            # (all share the same numeric tokens in BM25 and overlap
            # structurally in dense space on daily-reporting corpora).
            must_conditions.append(
                qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))
            )
        if owner_user_id is not None:
            match_val: int | str = owner_user_id
            if isinstance(owner_user_id, str):
                try:
                    match_val = int(owner_user_id)
                except (ValueError, TypeError):
                    match_val = owner_user_id  # UUID string — leave alone
            must_conditions.append(
                qm.FieldCondition(
                    key="owner_user_id",
                    match=qm.MatchValue(value=match_val),
                )
            )
        if chat_id is not None:
            must_conditions.append(
                qm.FieldCondition(
                    key="chat_id",
                    match=qm.MatchValue(value=chat_id),
                )
            )
        must_not_conditions = [
            qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True))
        ]
        # Tier 1 (2026-04-22): ``level`` pre-filter for the doc-summary
        # index. ``"doc"`` → keep only summary points; ``"chunk"`` → keep
        # only leaf chunks (summaries excluded via ``must_not`` since
        # legacy chunks have no ``level`` field and ``MatchValue`` on
        # missing keys rejects the point). Default ``None`` = no level
        # constraint (byte-identical to pre-Tier-1).
        if level == "doc":
            must_conditions.append(
                qm.FieldCondition(key="level", match=qm.MatchValue(value="doc"))
            )
        elif level == "chunk":
            must_not_conditions.append(
                qm.FieldCondition(key="level", match=qm.MatchValue(value="doc"))
            )
        return qm.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions,
        )

    async def search(
        self,
        name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        subtag_ids: Optional[list[int]] = None,
        doc_ids: Optional[list[int]] = None,
        owner_user_id: Optional[int | str] = None,
        chat_id: Optional[int | str] = None,
        level: Optional[str] = None,
        rescore: Optional[bool] = None,
    ) -> List[Hit]:
        """Dense-only search.

        For legacy collections (unnamed single vector) this is byte-identical to
        before. For hybrid-shaped collections (named ``dense`` + ``bm25``), Qdrant
        requires the caller to name which vector it's querying — we route via
        ``using=_DENSE_NAME`` when the sparse cache indicates this collection
        was created with ``with_sparse=True``.

        P2.2: ``owner_user_id`` adds a ``must`` condition on the owner field.
        Default None = no owner filter = byte-identical to pre-P2.2.

        P2.3: ``chat_id`` adds a ``must`` condition on the chat scope —
        required when reading from the consolidated ``chat_private``
        collection. Default None = no chat filter = byte-identical to pre-P2.3.

        P3.2: ``rescore`` controls quantization rescoring. True (default,
        env-backed via ``RAG_QDRANT_RESCORE=1``) attaches
        ``QuantizationSearchParams(rescore=True, oversampling=2.0)`` to the
        search params so Qdrant uses the INT8 index for approximate search,
        then rescores the top ``limit*oversampling`` candidates with the full
        fp32 vectors. On a collection *without* quantization the param is a
        no-op. Set ``rescore=False`` to skip the rescore pass (fastest, lowest
        recall — use for latency-critical queries where ±2% recall is fine).
        """
        flt = self._build_filter(
            subtag_ids=subtag_ids,
            doc_ids=doc_ids,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            level=level,
        )
        # Warm the sparse cache lazily (cheap, cached on first call) so legacy
        # callers that only use dense still route correctly against hybrid collections.
        if name not in self._sparse_cache:
            try:
                await self._refresh_sparse_cache(name)
            except Exception:
                pass
        kwargs = {
            "collection_name": name,
            "query": query_vector,
            "limit": limit,
            "query_filter": flt,
            "with_payload": _PAYLOAD_FIELDS,
            # P2.4: per-query HNSW ``ef`` knob. Higher → better recall, slower.
            # P3.2: SearchParams also carries the QuantizationSearchParams hint
            # when rescore is on (no-op on unquantized collections).
            "search_params": _search_params(rescore=rescore),
        }
        if self._sparse_cache.get(name):
            kwargs["using"] = _DENSE_NAME
        with span("qdrant.client.search", collection=name, limit=limit, mode="dense"):
            _t = _perf_counter()
            try:
                response = await self._client.query_points(**kwargs)
            finally:
                try:
                    qdrant_search_latency_seconds.labels(collection=name).observe(
                        _perf_counter() - _t
                    )
                except Exception:
                    pass
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in response.points]

    async def hybrid_search(
        self,
        name: str,
        query_vector: list[float],
        query_text: str,
        *,
        limit: int = 10,
        subtag_ids: Optional[list[int]] = None,
        doc_ids: Optional[list[int]] = None,
        owner_user_id: Optional[int | str] = None,
        chat_id: Optional[int | str] = None,
        level: Optional[str] = None,
        rescore: Optional[bool] = None,
    ) -> List[Hit]:
        """Server-side RRF fusion of dense + BM25 sparse results.

        Only valid for collections created with ``ensure_collection(..., with_sparse=True)``.
        Uses Qdrant's ``query_points`` with two ``Prefetch`` arms (dense + sparse)
        fused via ``FusionQuery(Fusion.RRF)``. Each prefetch pulls ``limit*2``
        candidates to give RRF enough material to rerank. Requires Qdrant ≥ 1.11.

        P2.2: ``owner_user_id`` propagates to both prefetch arms and the outer
        filter so RRF can only surface chunks owned by the given user. Default
        None = byte-identical to pre-P2.2.

        P2.3: ``chat_id`` propagates identically — required to scope the
        consolidated ``chat_private`` collection to a single chat.

        P3.2: ``rescore`` attaches ``QuantizationSearchParams(rescore=True,
        oversampling=2.0)`` to the dense prefetch arm so quantized approx
        search is rescored against the original fp32 vectors. The sparse
        (BM25) arm has no vectors and ignores the hint. Default follows the
        ``RAG_QDRANT_RESCORE`` env knob.
        """
        from .sparse_embedder import embed_sparse_query

        flt = self._build_filter(
            subtag_ids=subtag_ids,
            doc_ids=doc_ids,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            level=level,
        )
        indices, values = embed_sparse_query(query_text)
        # P2.4 + P3.2: SearchParams carries hnsw_ef AND (when rescore is on)
        # QuantizationSearchParams(rescore=True, oversampling=2.0). Attached
        # only to the dense prefetch arm; BM25 sparse has no HNSW / quantization.
        sp = _search_params(rescore=rescore)
        prefetch = [
            qm.Prefetch(
                query=query_vector,
                using=_DENSE_NAME,
                filter=flt,
                limit=limit * 2,
                params=sp,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=indices, values=values),
                using=_SPARSE_NAME,
                filter=flt,
                limit=limit * 2,
            ),
        ]
        with span("qdrant.client.search", collection=name, limit=limit, mode="hybrid"):
            _t = _perf_counter()
            try:
                response = await self._client.query_points(
                    collection_name=name,
                    prefetch=prefetch,
                    query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                    limit=limit,
                    query_filter=flt,
                    with_payload=_PAYLOAD_FIELDS,
                )
            finally:
                try:
                    qdrant_search_latency_seconds.labels(collection=name).observe(
                        _perf_counter() - _t
                    )
                except Exception:
                    pass
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in response.points]

    async def delete_by_doc(self, collection: str, doc_id: int | str) -> int:
        """Delete all points in ``collection`` whose payload ``doc_id`` matches.

        Tries int match first (new canonical form, enforced in ingest.py and
        by scripts/normalize_doc_ids.py), falls back to string match for any
        legacy rows that haven't been normalised yet. Returns 1 on success,
        0 on any error (best-effort — caller should log).
        """
        candidates: list[int | str] = []
        try:
            candidates.append(int(doc_id))
        except (ValueError, TypeError):
            pass
        candidates.append(str(doc_id))

        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(should=[
                        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=v))
                        for v in candidates
                    ])
                ),
                wait=True,
            )
            return 1
        except Exception:
            return 0

    async def optimize_collection(self, name: str) -> None:
        """P2.4: nudge Qdrant to re-balance the HNSW graph.

        Setting ``indexing_threshold=0`` forces the optimizer to index the
        full graph immediately — useful after bulk upserts or after raising
        ``ef_construct`` on an existing collection (the change only affects
        new inserts until a rebuild). Best-effort: errors are swallowed so
        callers can fire-and-forget from admin scripts.
        """
        try:
            await self._client.update_collection(
                collection_name=name,
                optimizer_config=qm.OptimizersConfigDiff(indexing_threshold=0),
            )
        except Exception:
            pass
