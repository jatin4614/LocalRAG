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

    async def close(self) -> None:
        await self._client.close()

    async def list_collections(self) -> list[str]:
        cols = (await self._client.get_collections()).collections
        return [c.name for c in cols]

    async def ensure_collection(self, name: str) -> None:
        cols = await self.list_collections()
        if name in cols:
            return
        await self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(
                size=self._vector_size,
                distance=qm.Distance[self._distance.upper()],
            ),
        )

    async def delete_collection(self, name: str) -> None:
        try:
            await self._client.delete_collection(name)
        except Exception:
            pass

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
        flt = None
        if subtag_ids:
            flt = qm.Filter(must=[
                qm.FieldCondition(key="subtag_id", match=qm.MatchAny(any=subtag_ids))
            ])
        response = await self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [Hit(id=r.id, score=r.score, payload=r.payload or {}) for r in response.points]
