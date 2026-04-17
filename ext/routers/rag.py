"""RAG retrieval endpoint: pulls from selected KBs + chat-private namespace."""
from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import validate_selected_kb_config
from ..services.auth import CurrentUser, get_current_user
from ..services.budget import budget_chunks
from ..services.embedder import Embedder
from ..services.rbac import get_allowed_kb_ids
from ..services.reranker import rerank
from ..services.retriever import retrieve
from ..services.vector_store import VectorStore


router = APIRouter(tags=["rag"])

_SM: async_sessionmaker[AsyncSession] | None = None
_VS: VectorStore | None = None
_EMB: Embedder | None = None


def configure(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    vector_store: VectorStore,
    embedder: Embedder,
) -> None:
    global _SM, _VS, _EMB
    _SM = sessionmaker
    _VS = vector_store
    _EMB = embedder


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SM is None:
        raise RuntimeError("rag router not configured")
    async with _SM() as s:
        yield s


class RetrieveRequest(BaseModel):
    chat_id: str
    query: str
    selected_kb_config: List[Any] = []
    max_tokens: int = 4000
    top_k: int = 10


class HitOut(BaseModel):
    score: float
    text: str
    kb_id: Optional[int] = None
    subtag_id: Optional[int] = None
    chat_id: Optional[int] = None
    doc_id: Optional[int] = None


class RetrieveResponse(BaseModel):
    hits: List[HitOut]


@router.post("/api/rag/retrieve", response_model=RetrieveResponse)
async def rag_retrieve(
    body: RetrieveRequest,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> RetrieveResponse:
    if _VS is None or _EMB is None:
        raise RuntimeError("rag router not fully configured")

    row = (await session.execute(
        text('SELECT user_id FROM chat WHERE id = :cid'),
        {"cid": body.chat_id},
    )).first()
    if row is None or str(row[0]) != str(user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    try:
        parsed = validate_selected_kb_config(body.selected_kb_config) or []
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    allowed = set(await get_allowed_kb_ids(session, user_id=user.id))
    for entry in parsed:
        if entry.kb_id not in allowed:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                detail=f"no access to kb_id={entry.kb_id}")

    raw = await retrieve(
        query=body.query,
        selected_kbs=[{"kb_id": e.kb_id, "subtag_ids": e.subtag_ids} for e in parsed],
        chat_id=body.chat_id,
        vector_store=_VS, embedder=_EMB,
    )
    reranked = rerank(raw, top_k=body.top_k)
    budgeted = budget_chunks(reranked, max_tokens=body.max_tokens)

    return RetrieveResponse(hits=[
        HitOut(
            score=h.score,
            text=str(h.payload.get("text", "")),
            kb_id=h.payload.get("kb_id"),
            subtag_id=h.payload.get("subtag_id"),
            chat_id=h.payload.get("chat_id"),
            doc_id=h.payload.get("doc_id"),
        )
        for h in budgeted
    ])
