"""SSE endpoint for streaming KB ingest progress to the admin UI.

When the admin opens a KB in the Knowledge Base admin page, the page
subscribes to this endpoint to receive real-time per-document ingest
progress events. Without this stream the only signal of progress was a
manual page refresh, which made async uploads look stuck.

Shape::

    GET /api/kb/{kb_id}/ingest-stream?token=<jwt>
    Accept: text/event-stream

Each event::

    event: ingest
    data: {"doc_id": 155, "filename": "Jan 23.docx", "stage": "processing"}

    event: ingest
    data: {"doc_id": 155, "filename": "Jan 23.docx", "stage": "done", "chunks": 219}

    event: ingest
    data: {"doc_id": 156, "filename": "feB 23.docx",
           "stage": "failed", "error": "OCR timeout"}

The stream stays open for the lifetime of the page; the client closes
it on unmount. Per-event payloads are produced by the celery worker
via ``ext.services.ingest_progress.emit_sync``.

Auth: standard ``Bearer`` header is preferred; we also accept ``?token=``
because the browser ``EventSource`` constructor can't carry custom
headers and admin-only access keeps the surface small.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from ..services.auth import CurrentUser, get_current_user
from ..services.ingest_progress import subscribe_async

log = logging.getLogger("orgchat.ingest_stream")

router = APIRouter(tags=["ingest-stream"])


def _sse_frame(event: str, data: str) -> bytes:
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


async def _resolve_user(
    request: Request, token: Optional[str],
) -> CurrentUser:
    """Resolve the user from either the Authorization header or ?token=.

    EventSource doesn't support custom headers, so we accept the JWT as
    a query parameter as a controlled fallback. The header path stays
    canonical for non-browser clients (curl, etc.).
    """
    # Header path — let the standard dependency do its work.
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return await get_current_user(request)
    # Query-param path — graft the token onto a fake Authorization header
    # so jwt verification stays in one place.
    if token:
        request.headers.__dict__["_list"].append(
            (b"authorization", f"Bearer {token}".encode())
        )
        return await get_current_user(request)
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="missing auth")


@router.get("/api/kb/{kb_id}/ingest-stream")
async def ingest_stream(
    kb_id: int,
    request: Request,
    token: Optional[str] = Query(default=None),
):
    """SSE stream of ingest progress events for ``kb_id``.

    Returns 401 without auth and 404 if the user has no read grant on
    the KB. Otherwise streams events until the client disconnects.
    """
    user = await _resolve_user(request, token)

    # RBAC: a non-admin must have an access grant on this KB. Reuse the
    # already-imported sessionmaker from chat_rag_bridge so we don't
    # spin up a second pool. Admin bypass mirrors kb_admin's policy.
    if user.role != "admin":
        from ..services.chat_rag_bridge import _sessionmaker
        if _sessionmaker is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="not ready")
        async with _sessionmaker() as s:  # type: ignore[misc]
            row = (await s.execute(
                text(
                    "SELECT 1 FROM kb_access "
                    "WHERE kb_id = :kb AND (user_id = :uid OR group_id IN ("
                    "  SELECT group_id FROM group_member WHERE user_id = :uid"
                    ")) LIMIT 1"
                ),
                {"kb": kb_id, "uid": user.id},
            )).first()
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, detail="kb not found")

    async def _gen() -> AsyncIterator[bytes]:
        # Initial "ready" frame so the browser EventSource fires its
        # ``open`` handler — useful for the UI to flip from "subscribing"
        # to "subscribed" indicator.
        yield _sse_frame("ready", '{"kb_id":' + str(kb_id) + "}")
        try:
            async for payload in subscribe_async(kb_id):
                if await request.is_disconnected():
                    break
                yield _sse_frame("ingest", payload)
        except asyncio.CancelledError:
            # Client disconnected — clean up upstream subscription.
            raise

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            # Disable proxy buffering so events stream live instead of
            # accumulating until the connection closes.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
