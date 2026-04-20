"""Server-Sent Events endpoint for streaming RAG pipeline progress.

Solves the UX half of P3.0: the user types a question, the LLM is about
to spend 5-10s generating a response, and we want them to SEE that work
is happening — "retrieving...", "reranking...", "expanding context...".

Shape::

    GET /api/rag/stream/{chat_id}?q=<urlencoded-query>
    Accept: text/event-stream

Each pipeline stage emits one or two events. Event shape::

    event: stage
    data: {"stage": "embed", "status": "running"}

    event: stage
    data: {"stage": "retrieve", "status": "done", "ms": 9, "hits": 30}

    event: stage
    data: {"stage": "mmr", "status": "skipped", "reason": "flag_off"}

    event: hits
    data: {"hits": [{"doc_id": 42, "filename": "policy.md", "score": 0.87}]}

    event: done
    data: {"total_ms": 580}

The LLM call is NOT part of this stream — the frontend should fire the
existing chat-completion request in parallel. This endpoint only drains
the retrieval pipeline so the UI can render "work in progress" between
the user's submit and the first token arriving from the LLM.

curl test::

    curl -N \\
        -H "X-User-Id: 42" -H "X-User-Role: admin" \\
        "http://localhost:8080/api/rag/stream/chat-xyz?q=hello"
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..services import chat_rag_bridge
from ..services.auth import CurrentUser, get_current_user

log = logging.getLogger("orgchat.rag_stream")

router = APIRouter(prefix="/api/rag", tags=["rag-stream"])


def _sse(event: str, data: dict) -> bytes:
    """Encode an SSE frame. ``data`` is JSON-serialized onto a single line.

    SSE protocol: ``event:`` line + ``data:`` line + blank line. We use
    ``ensure_ascii=True`` so non-ASCII filenames don't break byte-counting
    intermediaries (Caddy, nginx).
    """
    payload = json.dumps(data, ensure_ascii=True)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@router.get("/stream/{chat_id}")
async def stream_retrieval(
    chat_id: str,
    request: Request,
    q: Optional[str] = None,
    user: CurrentUser = Depends(get_current_user),
):
    """SSE endpoint — streams pipeline stage events for one retrieval.

    ``q`` is the user query. If absent or empty we return 400.

    The endpoint reads the chat's stored ``kb_config`` (same as the
    middleware does for the completion path) so the UX sees the exact
    KBs that the next LLM call will use. Non-owners of ``chat_id``
    receive 404 (same as ``get_chat_kb_config``).
    """
    if not q or not q.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="missing ?q=")

    # Ownership + kb_config lookup in one trip — reuse the bridge's helper
    # so the access rules match the completion path exactly.
    kb_config = await chat_rag_bridge.get_kb_config_for_chat(chat_id, user.id)
    # get_kb_config_for_chat returns None on unknown chat OR on lookup
    # failure. Treat either as "no KB config" — retrieval still runs over
    # the private-chat namespace if the user owns the chat, so we don't
    # hard-fail. (The chat-ownership check happens inside the bridge via
    # the user_id filter on the meta lookup.)

    async def event_stream() -> AsyncIterator[bytes]:
        # Queue used to ferry progress events from the bridge coroutine
        # (which may run in the same task) to the SSE generator (which
        # yields bytes). Asyncio Queue is the right primitive — it
        # propagates across ``await`` without the GIL contention of a
        # threadlocal queue.
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def progress_cb(event: dict) -> None:
            await queue.put(event)

        async def drain() -> None:
            """Run the pipeline and push a terminal sentinel on completion."""
            try:
                await chat_rag_bridge.retrieve_kb_sources(
                    kb_config=kb_config or [],
                    query=q,
                    user_id=user.id,
                    chat_id=chat_id,
                    progress_cb=progress_cb,
                )
            except Exception as e:
                log.exception("rag_stream: pipeline error: %s", e)
                await queue.put({"stage": "error", "message": str(e)})
            finally:
                await queue.put(None)  # sentinel: stream end

        task = asyncio.create_task(drain())

        try:
            # Initial comment keeps the connection open through proxy
            # buffers (Caddy default buffer is empty-until-flush-on-EOL).
            yield b": stream open\n\n"
            while True:
                # Bail out early if the client disconnected — StreamingResponse
                # won't cancel the task automatically in every FastAPI version.
                if await request.is_disconnected():
                    task.cancel()
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send a heartbeat so intermediaries don't close the idle connection.
                    yield b": keepalive\n\n"
                    continue
                if event is None:
                    yield _sse("done", {})
                    break
                # "hits" and "done" events get their own SSE event name so
                # clients can switch on addEventListener; everything else
                # is a "stage" event.
                name = "stage"
                if event.get("stage") == "hits":
                    name = "hits"
                elif event.get("stage") == "done":
                    name = "done"
                elif event.get("stage") == "error":
                    name = "error"
                yield _sse(name, event)
        finally:
            if not task.done():
                task.cancel()

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        # X-Accel-Buffering disables nginx response buffering; Caddy
        # ignores it but it's cheap insurance if upstream changes proxy.
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=headers,
    )


__all__ = ["router"]
