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


class SubtagInfo(BaseModel):
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


@router.get("/api/kb/{kb_id}/subtags", response_model=list[SubtagInfo])
async def list_kb_subtags(
    kb_id: int,
    user: CurrentUser = Depends(get_current_user),
    session: AsyncSession = Depends(_get_session),
):
    allowed = await get_allowed_kb_ids(session, user_id=user.id)
    if kb_id not in allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")
    subs = await kb_service.list_subtags(session, kb_id=kb_id)
    return [SubtagInfo(id=s.id, name=s.name, description=s.description) for s in subs]


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

        # Guard 1: every submitted subtag_id must belong to the kb_id it is grouped under.
        all_sids = [sid for entry in parsed for sid in entry.subtag_ids]
        if all_sids:
            sub_rows = (await session.execute(
                text('SELECT id, kb_id FROM kb_subtags WHERE id = ANY(:sids)'),
                {"sids": all_sids},
            )).all()
            sid_to_kb = {int(r[0]): int(r[1]) for r in sub_rows}
            for entry in parsed:
                for sid in entry.subtag_ids:
                    if sid_to_kb.get(sid) != entry.kb_id:
                        raise HTTPException(
                            status.HTTP_400_BAD_REQUEST,
                            detail=f"subtag_id={sid} does not belong to kb_id={entry.kb_id}",
                        )

    # Guard 2: kb_config is locked once the chat has any user message (design §2.4).
    # Upstream stores messages as a dict at chat.chat['history']['messages'], keyed by
    # message_id; each message carries a 'role'. We count role=='user' entries.
    chat_blob = (await session.execute(
        text('SELECT chat FROM chat WHERE id = :cid'),
        {"cid": chat_id},
    )).scalar()
    if chat_blob:
        if isinstance(chat_blob, str):
            try:
                chat_blob = json.loads(chat_blob)
            except (ValueError, TypeError):
                chat_blob = {}
        messages_map = (chat_blob or {}).get("history", {}).get("messages", {}) or {}
        has_user_message = any(
            isinstance(m, dict) and m.get("role") == "user"
            for m in messages_map.values()
        )
        if has_user_message:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                detail="chat already has messages; kb_config is locked",
            )

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
