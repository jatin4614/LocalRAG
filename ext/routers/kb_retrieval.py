"""User-facing read routes: list available KBs, set per-chat KB selection."""
from __future__ import annotations

from typing import Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import Chat, validate_selected_kb_config
from ..services import kb_service
from ..services.auth import CurrentUser, get_current_user
from ..services.rbac import get_allowed_kb_ids


router = APIRouter(tags=["kb-retrieval"])

_SESSIONMAKER: async_sessionmaker[AsyncSession] | None = None


def set_sessionmaker(sm: async_sessionmaker[AsyncSession]) -> None:
    global _SESSIONMAKER
    _SESSIONMAKER = sm


async def _get_session() -> AsyncGenerator[AsyncSession, None]:
    if _SESSIONMAKER is None:
        raise RuntimeError("sessionmaker not configured")
    async with _SESSIONMAKER() as s:
        yield s


class KBAvailable(BaseModel):
    id: int
    name: str
    description: Optional[str]


@router.get("/api/kb/available", response_model=list[KBAvailable])
async def available_kbs(
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> list[KBAvailable]:
    allowed = await get_allowed_kb_ids(session, user_id=user.id)
    kbs = await kb_service.list_kbs(session, kb_ids=allowed)
    return [KBAvailable(id=k.id, name=k.name, description=k.description) for k in kbs]


class ChatKBConfig(BaseModel):
    config: Optional[List[Any]] = None


@router.put("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def set_chat_kb_config(
    chat_id: int,
    body: ChatKBConfig,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> ChatKBConfig:
    chat = (await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    try:
        parsed = validate_selected_kb_config(body.config)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if parsed:
        allowed = set(await get_allowed_kb_ids(session, user_id=user.id))
        for entry in parsed:
            if entry.kb_id not in allowed:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    detail=f"no access to kb_id={entry.kb_id}",
                )

    chat.selected_kb_config = body.config
    await session.commit()
    return ChatKBConfig(config=body.config)


@router.get("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def get_chat_kb_config(
    chat_id: int,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
) -> ChatKBConfig:
    chat = (await session.execute(
        select(Chat).where(Chat.id == chat_id, Chat.user_id == user.id)
    )).scalar_one_or_none()
    if chat is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")
    return ChatKBConfig(config=chat.selected_kb_config)
