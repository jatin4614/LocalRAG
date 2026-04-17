"""User-facing read routes: list available KBs, set per-chat KB selection."""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..db.models import validate_selected_kb_config
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
):
    allowed = await get_allowed_kb_ids(session, user_id=user.id)
    kbs = await kb_service.list_kbs(session, kb_ids=allowed)
    return [KBAvailable(id=k.id, name=k.name, description=k.description) for k in kbs]


class ChatKBConfig(BaseModel):
    config: Optional[List[Any]] = None


@router.put("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def set_chat_kb_config(
    chat_id: str,
    body: ChatKBConfig,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    # Verify chat ownership using upstream's "chat" table (singular)
    row = (await session.execute(
        text('SELECT user_id, meta FROM chat WHERE id = :cid'),
        {"cid": chat_id},
    )).first()
    if row is None or str(row[0]) != str(user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    try:
        parsed = validate_selected_kb_config(body.config)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if parsed:
        allowed = set(await get_allowed_kb_ids(session, user_id=user.id))
        for entry in parsed:
            if entry.kb_id not in allowed:
                raise HTTPException(status.HTTP_403_FORBIDDEN,
                                    detail=f"no access to kb_id={entry.kb_id}")

    # Store in chat.meta JSON column
    current_meta = row[1] if row[1] else {}
    if isinstance(current_meta, str):
        current_meta = json.loads(current_meta)
    current_meta["kb_config"] = body.config
    await session.execute(
        text('UPDATE chat SET meta = :meta WHERE id = :cid'),
        {"meta": json.dumps(current_meta), "cid": chat_id},
    )
    await session.commit()
    return ChatKBConfig(config=body.config)


@router.get("/api/chats/{chat_id}/kb_config", response_model=ChatKBConfig)
async def get_chat_kb_config(
    chat_id: str,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    row = (await session.execute(
        text('SELECT user_id, meta FROM chat WHERE id = :cid'),
        {"cid": chat_id},
    )).first()
    if row is None or str(row[0]) != str(user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="chat not found")

    meta = row[1] if row[1] else {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    return ChatKBConfig(config=meta.get("kb_config"))
