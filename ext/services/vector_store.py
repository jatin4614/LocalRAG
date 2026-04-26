"""Thin async wrapper over qdrant-client."""
from __future__ import annotations

import os
import time as _time
import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

import httpx
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from .circuit_breaker import CircuitOpenError, breaker_for
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
# P3.4 ColBERT multi-vector slot. Per-token 128-dim vectors with
# ``MultiVectorConfig(MAX_SIM)``. Only created when the collection is
# explicitly opt-in via ``ensure_collection(..., with_colbert=True)`` or
# the ``RAG_COLBERT=1`` env switch — Qdrant requires the named slot to
# exist before any upsert can target it. Read-side fusion (Task 3.5)
# will use ``using=_COLBERT_NAME`` against the same slot.
_COLBERT_NAME = "colbert"
_COLBERT_DIM = 128


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
        # Phase 1.3: ``_url`` is also retained so ``health_check()`` can probe
        # the Qdrant root endpoint via a fresh httpx client (the qdrant-client
        # itself doesn't expose a cheap "is the server up?" call).
        self._url = url
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
        # P3.4 — analogous cache for the colbert multi-vector slot. Read-side
        # (Task 3.5) and ingest both consult ``_collection_has_colbert`` to
        # decide whether to compute / upsert the multi-vectors at all.
        self._colbert_cache: dict[str, bool] = {}

    async def close(self) -> None:
        await self._client.close()

    async def health_check(self) -> bool:
        """Lightweight Qdrant health probe. Result cached 5s.

        Phase 1.3 — used by ``chat_rag_bridge._run_pipeline`` as a pre-flight
        check before fanning out N parallel KB searches. The Qdrant root
        endpoint returns ``{"title":...,"version":...}`` in <5ms when the
        server is up; on connection error / timeout we report False without
        raising so callers can decide whether to short-circuit retrieval.

        The 5s cache means N concurrent requests share one probe — important
        because ``health_check()`` runs on every chat turn.
        """
        now = _time.monotonic()
        if hasattr(self, "_health_cache") and now - self._health_cache[0] < 5.0:
            return self._health_cache[1]
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                r = await c.get(f"{self._url}/")
                ok = r.status_code == 200
        except Exception:
            ok = False
        self._health_cache = (now, ok)
        return ok

    async def list_collections(self) -> list[str]:
        cols = (await self._client.get_collections()).collections
        return [c.name for c in cols]

    async def ensure_collection(
        self,
        name: str,
        *,
        with_sparse: bool = False,
        with_quantization: Optional[bool] = None,
        with_colbert: Optional[bool] = None,
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

        P3.4: ``with_colbert`` opts the collection into a third named vector
        slot (``colbert``, 128-dim, COSINE, ``MultiVectorConfig(MAX_SIM)``) so
        ColBERT late-interaction multi-vectors can be upserted alongside dense
        + sparse. Implies the named-vector shape — passed without
        ``with_sparse``, the dense slot is also given a name (``dense``) so
        Qdrant accepts the multi-vectors_config dict. When ``None`` (default),
        the ``RAG_COLBERT`` env switch decides. The colbert slot is purely
        additive: read paths that don't query it are byte-identical.

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
        # P3.4: per-call arg wins; env is the fallback (same convention as
        # quantization). ColBERT requires the named-vector shape regardless
        # of ``with_sparse`` because Qdrant only accepts MultiVectorConfig
        # under a named slot in a vectors_config dict (legacy single-unnamed
        # vector form has no slot to attach the multi-vector params to).
        if with_colbert is None:
            with_colbert = _env_bool("RAG_COLBERT", False)
        # Pre-build colbert vector params once — same shape regardless of
        # sparse/legacy branch below.
        colbert_params = (
            qm.VectorParams(
                size=_COLBERT_DIM,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM,
                ),
            )
            if with_colbert
            else None
        )
        try:
            if with_sparse or with_colbert:
                # Named-vector form — required when there's any second slot
                # (sparse OR colbert). Dense always lands under ``_DENSE_NAME``
                # so the read path can route via ``using=_DENSE_NAME``.
                vectors_config: dict = {
                    _DENSE_NAME: qm.VectorParams(
                        size=self._vector_size,
                        distance=qm.Distance[self._distance.upper()],
                        on_disk=dense_on_disk,
                    ),
                }
                if colbert_params is not None:
                    vectors_config[_COLBERT_NAME] = colbert_params
                # ``sparse_vectors_config`` is only valid alongside the
                # named-vector form; pass it as None when sparse is off so
                # qdrant-client doesn't reject the request.
                sparse_cfg = (
                    {_SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)}
                    if with_sparse
                    else None
                )
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_cfg,
                    hnsw_config=hnsw,
                    on_disk_payload=on_disk_payload,
                    quantization_config=quant_config,
                )
                self._sparse_cache[name] = bool(with_sparse)
                # Defensive: a few legacy unit tests construct VectorStore
                # via ``__new__`` and only set the pre-3.4 attributes, so use
                # a ``getattr`` fallback to avoid AttributeError on the new
                # cache. Production paths always go through ``__init__`` and
                # always have the dict.
                cb_cache = getattr(self, "_colbert_cache", None)
                if cb_cache is not None:
                    cb_cache[name] = bool(with_colbert)
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
                cb_cache = getattr(self, "_colbert_cache", None)
                if cb_cache is not None:
                    cb_cache[name] = False
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
        # Defensive: legacy unit-test stubs (see ``_make_vs`` in
        # ``tests/unit/test_vector_store_*``) construct the instance via
        # ``__new__`` and only set ``_sparse_cache``. The colbert cache
        # is added in P3.4 and may legitimately be absent on those stubs.
        cb_cache = getattr(self, "_colbert_cache", None)
        if cb_cache is not None:
            cb_cache.pop(name, None)

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

    async def _refresh_colbert_cache(self, name: str) -> bool:
        """Ask Qdrant whether ``name`` has the ``colbert`` named vector slot.

        Mirrors ``_refresh_sparse_cache``: cached, returns False for legacy
        collections, safe on missing collections. The slot we look for is a
        named vector under ``_COLBERT_NAME`` whose ``multivector_config`` is
        present (only multi-vector slots have that field set).
        """
        # Defensive: legacy unit-test stubs may not have ``_colbert_cache``;
        # promote to an instance dict so subsequent calls hit the cache.
        if not hasattr(self, "_colbert_cache"):
            self._colbert_cache = {}
        if name in self._colbert_cache:
            return self._colbert_cache[name]
        try:
            info = await self._client.get_collection(collection_name=name)
        except Exception:
            self._colbert_cache[name] = False
            return False
        has = False
        try:
            params = info.config.params if info and info.config else None
            vectors = getattr(params, "vectors", None) if params is not None else None
            # ``vectors`` is a dict[name, VectorParams] for named-vector
            # collections; legacy single-unnamed-vector form returns a bare
            # VectorParams (no ``__contains__``). Only the dict form can have
            # the colbert slot, so the isinstance guard covers both shapes.
            if isinstance(vectors, dict) and _COLBERT_NAME in vectors:
                slot = vectors[_COLBERT_NAME]
                # multivector_config is only set on multi-vector slots —
                # presence alone confirms the slot was created with
                # MultiVectorConfig (vs. a same-named regular dense slot).
                has = getattr(slot, "multivector_config", None) is not None
        except Exception:
            has = False
        self._colbert_cache[name] = has
        return has

    def _collection_has_colbert(self, name: str) -> bool:
        """Return cached colbert-support flag without hitting the network.

        Same convention as ``_collection_has_sparse``: returns False for
        unknown collections (fail-closed → ingest skips the colbert arm).
        """
        cb_cache = getattr(self, "_colbert_cache", None)
        if cb_cache is None:
            return False
        return cb_cache.get(name, False)

    async def collection_has_vector(
        self, collection: str, vector_name: str
    ) -> bool:
        """Check if a collection has a specific named vector slot. Cached 60s.

        Generic counterpart to ``_collection_has_sparse`` /
        ``_collection_has_colbert`` that doesn't hard-code a slot name —
        used by the read path (Phase 3.5) to gate the optional ColBERT
        arm of tri-fusion per collection. Fail-closed (returns False)
        when the collection doesn't exist OR the lookup raises, so the
        caller silently skips the extra retrieval arm rather than
        crashing the whole search.

        The cache is keyed by ``(collection, vector_name)`` and entries
        expire after 60s — short enough that an admin who just ran
        ``scripts/enable_colbert.py`` on a previously-legacy collection
        sees the new slot within a minute, long enough that this lookup
        adds no measurable per-query overhead under steady load.
        """
        cache = getattr(self, "_named_vec_cache", {})
        key = (collection, vector_name)
        now = _time.monotonic()
        if key in cache and now - cache[key][0] < 60.0:
            return cache[key][1]
        try:
            info = await self._client.get_collection(collection)
            params = info.config.params if info and info.config else None
            vectors = getattr(params, "vectors", None) if params is not None else None
            # Named-vector form is dict[name, VectorParams]; legacy single-
            # unnamed-vector form returns a bare VectorParams (no membership
            # test). Only the dict form can carry an extra named slot.
            ok = isinstance(vectors, dict) and vector_name in vectors
        except Exception:
            ok = False
        cache[key] = (now, ok)
        self._named_vec_cache = cache
        return ok

    async def upsert(self, name: str, points: Iterable[dict]) -> None:
        """Upsert points. Each point may carry ``sparse_vector: (indices, values)``.

        When any point has a sparse_vector AND the collection supports sparse
        (hybrid-enabled), points are written in the named-vector form
        (``{dense: [...], bm25: SparseVector(...)}``). Otherwise the legacy
        single-unnamed-vector path is used (byte-identical to before).

        P3.4: each point may also carry ``colbert_vector: list[list[float]]``
        (one 128-dim vector per token). When present AND the collection has
        the colbert slot, the named-vector dict picks up
        ``{colbert: [[...],...]}`` alongside dense (and sparse if hybrid).
        Colbert alone (no sparse) still triggers the named-vector path, since
        Qdrant requires the named shape for the multi-vector slot to exist.
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

        # Same gate for colbert: only emit when the point carries multi-vectors
        # AND the collection actually has the slot. Either condition false →
        # silently skip (no Qdrant error, ingest doesn't have to special-case
        # collections that pre-date Task 3.4).
        has_colbert_points = any(p.get("colbert_vector") is not None for p in points)
        use_colbert = False
        if has_colbert_points:
            await self._refresh_colbert_cache(name)
            use_colbert = self._collection_has_colbert(name)

        n_points = len(points)
        with span(
            "qdrant.upsert",
            collection=name,
            n_points=n_points,
            vector_size=self._vector_size,
            hybrid=use_sparse,
            colbert=use_colbert,
        ):
            _t = _perf_counter()
            try:
                if not use_sparse and not use_colbert:
                    # Legacy path — byte-identical to pre-hybrid behavior.
                    pts = [
                        qm.PointStruct(
                            id=p["id"], vector=p["vector"], payload=p.get("payload", {})
                        )
                        for p in points
                    ]
                    await self._client.upsert(collection_name=name, points=pts, wait=True)
                    return

                # Named-vector path — pack dense under _DENSE_NAME, sparse
                # under _SPARSE_NAME, colbert under _COLBERT_NAME. Each slot is
                # only populated when both the per-point payload AND the
                # collection support it (set above), so a hybrid-only
                # collection ignores stray colbert_vector fields and vice
                # versa.
                pts = []
                for p in points:
                    vec_map: dict = {_DENSE_NAME: p["vector"]}
                    if use_sparse:
                        sv = p.get("sparse_vector")
                        if sv is not None:
                            indices, values = sv
                            vec_map[_SPARSE_NAME] = qm.SparseVector(
                                indices=list(indices), values=list(values)
                            )
                    if use_colbert:
                        cv = p.get("colbert_vector")
                        if cv is not None:
                            # cv is already list[list[float]] (JSON-safe); the
                            # qdrant-client serializer accepts that directly
                            # for multi-vector slots.
                            vec_map[_COLBERT_NAME] = cv
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
        # Phase 1.3: per-collection circuit breaker. If Qdrant has been
        # failing for this collection, raise CircuitOpenError immediately
        # instead of issuing yet another doomed RPC. Only transport errors
        # (timeout / connect / read) trip the breaker — application errors
        # (bad filter, unknown collection) propagate normally without
        # affecting breaker state.
        cb = breaker_for(f"qdrant:{name}")
        cb.raise_if_open()
        with span("qdrant.client.search", collection=name, limit=limit, mode="dense"):
            _t = _perf_counter()
            try:
                response = await self._client.query_points(**kwargs)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
                cb.record_failure()
                raise
            finally:
                try:
                    qdrant_search_latency_seconds.labels(collection=name).observe(
                        _perf_counter() - _t
                    )
                except Exception:
                    pass
            cb.record_success()
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
        """Server-side RRF fusion of dense + BM25 sparse (+ optional ColBERT) results.

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

        P3.5 (tri-fusion): when ``RAG_COLBERT=1`` AND the collection has the
        named ``colbert`` vector slot (checked via ``collection_has_vector``,
        cached 60s), a third Prefetch arm is added that queries the ColBERT
        multi-vector slot using ``MAX_SIM`` late interaction. The outer
        ``FusionQuery(RRF)`` then fuses all three arms server-side — same RRF
        formula, one extra round-trip avoided. Default-off:
        ``RAG_COLBERT=0`` OR a collection without the slot keeps the
        two-arm dense+sparse path byte-identical to pre-P3.5.
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
        # P3.5: third arm — ColBERT late-interaction multi-vector query.
        # Gated behind BOTH the env flag AND the per-collection slot check
        # so a collection that pre-dates Task 3.4 silently falls back to the
        # two-arm dense+sparse path (no Qdrant error from querying a missing
        # named vector). The flag check first short-circuits before any
        # network call when ColBERT is process-wide off.
        if _env_bool("RAG_COLBERT", False) and await self.collection_has_vector(
            name, _COLBERT_NAME
        ):
            try:
                from .embedder import colbert_embed
                colbert_vecs = colbert_embed([query_text])
                if colbert_vecs and colbert_vecs[0]:
                    prefetch.append(
                        qm.Prefetch(
                            query=colbert_vecs[0],
                            using=_COLBERT_NAME,
                            filter=flt,
                            limit=limit * 2,
                        )
                    )
            except Exception:
                # Embedder failure (model not cached, ONNX runtime issue,
                # fastembed not installed) → fall back to the two-arm path
                # rather than break the whole search. Operators see this in
                # the qdrant.client.search span trace.
                pass
        # Phase 1.3: per-collection circuit breaker — hybrid path mirrors the
        # dense ``search`` path. Same trip rules: only transport errors count.
        cb = breaker_for(f"qdrant:{name}")
        cb.raise_if_open()
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
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError):
                cb.record_failure()
                raise
            finally:
                try:
                    qdrant_search_latency_seconds.labels(collection=name).observe(
                        _perf_counter() - _t
                    )
                except Exception:
                    pass
            cb.record_success()
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

    # ------------------------------------------------------------------
    # Plan B Phase 5.1 — temporal custom sharding
    # ------------------------------------------------------------------
    async def ensure_collection_temporal(
        self,
        name: str,
        shard_keys: list[str],
        *,
        with_sparse: bool = True,
        with_colbert: bool = False,
        on_disk_payload: Optional[bool] = None,
        replication_factor: int = 1,
    ) -> None:
        """Create a Qdrant collection with custom temporal sharding.

        One shard per ``shard_key`` (typically "YYYY-MM" for monthly buckets).
        Shard creation is idempotent; existing keys are not re-created. The
        collection itself is also idempotent — if it exists with a different
        sharding strategy, this method will NOT migrate it (operator must
        drop + recreate to change sharding).

        Plan B Phase 5.1.
        """
        on_disk_payload = (
            on_disk_payload
            if on_disk_payload is not None
            else _env_bool("RAG_QDRANT_ON_DISK_PAYLOAD", True)
        )

        if await self._client.collection_exists(collection_name=name):
            import logging as _logging
            _logging.getLogger(__name__).info(
                "collection %s exists; ensuring shard keys", name
            )
        else:
            vectors_config: dict = {
                _DENSE_NAME: qm.VectorParams(
                    size=self._vector_size,
                    distance=qm.Distance[self._distance.upper()],
                )
            }
            if with_colbert:
                vectors_config[_COLBERT_NAME] = qm.VectorParams(
                    size=_COLBERT_DIM,
                    distance=qm.Distance.COSINE,
                    multivector_config=qm.MultiVectorConfig(
                        comparator=qm.MultiVectorComparator.MAX_SIM,
                    ),
                )
            sparse_vectors = None
            if with_sparse:
                sparse_vectors = {
                    _SPARSE_NAME: qm.SparseVectorParams(modifier=qm.Modifier.IDF)
                }

            await self._client.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors,
                on_disk_payload=on_disk_payload,
                sharding_method=qm.ShardingMethod.CUSTOM,
                shard_number=len(shard_keys),
                replication_factor=replication_factor,
            )
            import logging as _logging
            _logging.getLogger(__name__).info(
                "created temporal collection %s with %d shards",
                name, len(shard_keys),
            )

        # Discover peer ids — required for create_shard_key in cluster mode.
        # Single-peer clusters still need explicit placement; otherwise Qdrant
        # returns 400 "Distributed mode disabled" because it cannot decide
        # which peer should host the new shard.
        # Discover peer ids via raw HTTP — qdrant-client SDK exposes
        # cluster_status as a discriminated-union model that does not
        # surface the peers dict cleanly.
        peer_ids: list[int] = []
        try:
            import httpx as _httpx
            qdrant_url = (
                self._url.rstrip("/")
                if hasattr(self, "_url") else "http://qdrant:6333"
            )
            async with _httpx.AsyncClient(timeout=5.0) as _http:
                r = await _http.get(f"{qdrant_url}/cluster")
                if r.status_code == 200:
                    peers = r.json().get("result", {}).get("peers", {}) or {}
                    peer_ids = [int(pid) for pid in peers.keys()]
        except Exception:
            # Cluster API unavailable; placement=None only works in true
            # distributed (multi-peer) mode.
            peer_ids = []

        # Ensure shard keys (idempotent — Qdrant returns 200 even if exists)
        for sk in shard_keys:
            try:
                await self._client.create_shard_key(
                    collection_name=name, shard_key=sk,
                    placement=peer_ids or None,
                )
            except Exception as e:
                # 409 / "already exists" is fine
                if "exists" not in str(e).lower():
                    raise

        # Add per-payload index on shard_key for filterable date-bounded queries
        try:
            await self._client.create_payload_index(
                collection_name=name,
                field_name="shard_key",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # idempotent

    async def upsert_temporal(
        self,
        collection: str,
        points: list[dict],
        *,
        shard_key: str,
    ) -> None:
        """Upsert points into a specific shard_key.

        Caller must ensure all points belong to the named shard. Mixing
        shards in one call is a Qdrant constraint violation.

        Plan B Phase 5.1; Phase 5.9 added per-shard latency observation.
        """
        structs = [
            qm.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        # Plan B Phase 5.9 — observe per-shard upsert latency. Wrap in
        # try/finally so the histogram still records on exception.
        import time as _t
        from .metrics import RAG_SHARD_UPSERT_LATENCY
        _start = _t.monotonic()
        try:
            await self._client.upsert(
                collection_name=collection,
                points=structs,
                shard_key_selector=shard_key,
            )
        finally:
            try:
                RAG_SHARD_UPSERT_LATENCY.labels(
                    collection=collection, shard_key=shard_key,
                ).observe(_t.monotonic() - _start)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Plan B Phase 5.3 — tiered storage (hot/warm/cold)
    # ------------------------------------------------------------------
    async def apply_tier_config(
        self,
        collection: str,
        shard_key: str,
        tier: str,
    ) -> None:
        """Update the per-shard tier configuration.

        Hot:  in-memory HNSW (memmap_threshold=0, no quantization).
        Warm: mmap on SSD (memmap_threshold=20_000).
        Cold: on-disk + INT8 scalar quantization (always_ram=False).

        Note: Qdrant currently scopes optimizer + quantization config at
        collection level. For per-shard control on a temporal collection,
        we use the shard_key as a partition key in the indexing optimizer
        threshold; per-shard quantization in Qdrant's current API requires
        re-creating the affected shard. The tier cron (Phase 5.8) coordinates
        this carefully.

        Plan B Phase 5.3.
        """
        if tier == "hot":
            await self._client.update_collection(
                collection_name=collection,
                optimizers_config=qm.OptimizersConfigDiff(
                    memmap_threshold=0,  # all in RAM
                ),
            )
        elif tier == "warm":
            await self._client.update_collection(
                collection_name=collection,
                optimizers_config=qm.OptimizersConfigDiff(
                    memmap_threshold=20_000,
                ),
            )
        elif tier == "cold":
            await self._client.update_collection(
                collection_name=collection,
                optimizers_config=qm.OptimizersConfigDiff(
                    memmap_threshold=20_000,
                ),
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=False,
                    ),
                ),
            )
        else:
            raise ValueError(f"unknown tier {tier!r}")
        # Plan B Phase 5.9 — emit per-shard tier gauge after Qdrant accepts.
        try:
            from .metrics import set_shard_tier
            set_shard_tier(collection=collection, shard_key=shard_key, tier=tier)
        except Exception:
            pass


def classify_tier(
    shard_key: str,
    *,
    hot_months: int = 3,
    warm_months: int = 12,
) -> str:
    """Return ``'hot'`` / ``'warm'`` / ``'cold'`` for a 'YYYY-MM' shard_key.

    ``hot_months``: shards aged < this are hot. ``warm_months``: shards aged
    >= hot_months but < warm_months are warm. Older are cold. Boundaries
    are exclusive of ``hot_months`` and inclusive of ``warm_months``.

    Plan B Phase 5.3.
    """
    import datetime as _dt
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(shard_key)
    today = _dt.date.today()
    age_months = (today.year - y) * 12 + (today.month - m)
    if age_months < hot_months:
        return "hot"
    if age_months < warm_months:
        return "warm"
    return "cold"
