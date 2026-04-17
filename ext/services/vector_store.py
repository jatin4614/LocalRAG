"""Thin async wrapper over qdrant-client."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Iterable, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm


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

    async def close(self) -> None:
        await self._client.close()

    async def list_collections(self) -> list[str]:
        cols = (await self._client.get_collections()).collections
        return [c.name for c in cols]

    async def ensure_collection(self, name: str) -> None:
        if name in self._known:
            return
        try:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=qm.VectorParams(
                    size=self._vector_size,
                    distance=qm.Distance[self._distance.upper()],
                ),
            )
        except Exception:
            # Already exists — verify it's actually there before caching
            cols = await self.list_collections()
            if name not in cols:
                raise
        # Create payload indexes for fast filtered queries (idempotent — Qdrant
        # silently ignores duplicate index creation).
        for field in ("kb_id", "subtag_id", "doc_id", "chat_id", "deleted"):
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

    async def upsert(self, name: str, points: Iterable[dict]) -> None:
        pts = [
            qm.PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
            for p in points
        ]
        await self._client.upsert(collection_name=name, points=pts, wait=True)

    async def search(
        self,
        name: str,
        query_vector: list[float],
        *,
        limit: int = 10,
        subtag_ids: Optional[list[int]] = None,
    ) -> List[Hit]:
        must_conditions = []
        if subtag_ids:
            must_conditions.append(
                qm.FieldCondition(key="subtag_id", match=qm.MatchAny(any=subtag_ids))
            )
        # Exclude soft-deleted points regardless of whether a subtag filter is active.
        must_not_conditions = [
            qm.FieldCondition(key="deleted", match=qm.MatchValue(value=True))
        ]
        flt = qm.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions,
        )
        response = await self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in response.points]

    async def delete_by_doc(self, collection: str, doc_id: int | str) -> int:
        """Delete all points in ``collection`` whose payload ``doc_id`` matches.

        doc_id is stored as a string in Qdrant payloads (normalised in ingest.py).
        Returns 1 on success, 0 on any error (best-effort — caller should log).
        """
        try:
            await self._client.delete(
                collection_name=collection,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(must=[
                        qm.FieldCondition(
                            key="doc_id",
                            match=qm.MatchValue(value=str(doc_id)),
                        )
                    ])
                ),
                wait=True,
            )
            return 1
        except Exception:
            return 0
