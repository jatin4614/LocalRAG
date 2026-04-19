"""Thin async wrapper over qdrant-client."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

_PAYLOAD_FIELDS = [
    "text", "kb_id", "subtag_id", "doc_id", "chat_id",
    "filename", "chunk_index", "deleted",
    # P0.4 structural metadata + pipeline provenance.
    "page", "heading_path", "sheet", "model_version",
    # P2.2 tenant-owner attribution — indexed as tenant field for RBAC
    # hot-path filters (admin-wide scoped queries skip other owners' subgraphs).
    "owner_user_id",
]

# Name of the named dense vector when a collection is created with sparse vectors
# alongside. Legacy collections (no sparse) use the unnamed single-vector form,
# so ``_DENSE_NAME`` is *only* used on hybrid-enabled collections.
_DENSE_NAME = "dense"
_SPARSE_NAME = "bm25"


@dataclass
class Hit:
    id: int | str | uuid.UUID
    score: float
    payload: dict


class VectorStore:
    def __init__(self, *, url: str, vector_size: int, distance: str = "Cosine") -> None:
        self._client = AsyncQdrantClient(url=url)
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

    async def ensure_collection(self, name: str, *, with_sparse: bool = False) -> None:
        """Idempotently create a collection.

        By default creates a legacy-shaped collection with a single unnamed
        dense vector (backward-compatible with every collection created before
        this change). When ``with_sparse=True``, creates a collection with both
        a named ``dense`` vector AND a named ``bm25`` sparse vector (IDF
        modifier) — required for server-side hybrid retrieval via RRF.

        This method never upgrades an existing collection; if ``name`` already
        exists with the wrong shape, callers must recreate it externally.
        """
        if name in self._known:
            return
        try:
            if with_sparse:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        _DENSE_NAME: qm.VectorParams(
                            size=self._vector_size,
                            distance=qm.Distance[self._distance.upper()],
                        ),
                    },
                    sparse_vectors_config={
                        _SPARSE_NAME: qm.SparseVectorParams(
                            modifier=qm.Modifier.IDF,
                        ),
                    },
                )
                self._sparse_cache[name] = True
            else:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config=qm.VectorParams(
                        size=self._vector_size,
                        distance=qm.Distance[self._distance.upper()],
                    ),
                )
                self._sparse_cache[name] = False
        except Exception:
            # Already exists — verify it's actually there before caching
            cols = await self.list_collections()
            if name not in cols:
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
        tenant_fields = ("kb_id", "chat_id", "owner_user_id")
        filter_fields = ("subtag_id", "doc_id", "deleted")
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

    @staticmethod
    def _build_filter(
        *, subtag_ids: Optional[list[int]] = None
    ) -> qm.Filter:
        must_conditions = []
        if subtag_ids:
            must_conditions.append(
                qm.FieldCondition(key="subtag_id", match=qm.MatchAny(any=subtag_ids))
            )
        must_not_conditions = [
            qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True))
        ]
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
    ) -> List[Hit]:
        """Dense-only search.

        For legacy collections (unnamed single vector) this is byte-identical to
        before. For hybrid-shaped collections (named ``dense`` + ``bm25``), Qdrant
        requires the caller to name which vector it's querying — we route via
        ``using=_DENSE_NAME`` when the sparse cache indicates this collection
        was created with ``with_sparse=True``.
        """
        flt = self._build_filter(subtag_ids=subtag_ids)
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
        }
        if self._sparse_cache.get(name):
            kwargs["using"] = _DENSE_NAME
        response = await self._client.query_points(**kwargs)
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in response.points]

    async def hybrid_search(
        self,
        name: str,
        query_vector: list[float],
        query_text: str,
        *,
        limit: int = 10,
        subtag_ids: Optional[list[int]] = None,
    ) -> List[Hit]:
        """Server-side RRF fusion of dense + BM25 sparse results.

        Only valid for collections created with ``ensure_collection(..., with_sparse=True)``.
        Uses Qdrant's ``query_points`` with two ``Prefetch`` arms (dense + sparse)
        fused via ``FusionQuery(Fusion.RRF)``. Each prefetch pulls ``limit*2``
        candidates to give RRF enough material to rerank. Requires Qdrant ≥ 1.11.
        """
        from .sparse_embedder import embed_sparse_query

        flt = self._build_filter(subtag_ids=subtag_ids)
        indices, values = embed_sparse_query(query_text)
        prefetch = [
            qm.Prefetch(
                query=query_vector,
                using=_DENSE_NAME,
                filter=flt,
                limit=limit * 2,
            ),
            qm.Prefetch(
                query=qm.SparseVector(indices=indices, values=values),
                using=_SPARSE_NAME,
                filter=flt,
                limit=limit * 2,
            ),
        ]
        response = await self._client.query_points(
            collection_name=name,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            query_filter=flt,
            with_payload=_PAYLOAD_FIELDS,
        )
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
