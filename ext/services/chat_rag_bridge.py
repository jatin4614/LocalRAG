"""Bridge between upstream's chat middleware and our KB RAG pipeline.

Called from the patched process_chat_payload() in middleware.py.
Retrieves KB context and returns it in upstream's source dict format.
"""
from __future__ import annotations

import contextvars
import logging
import os as _os
from typing import List, Optional

logger = logging.getLogger("orgchat.rag_bridge")

# P0.5 — history-aware query rewrite (flag-gated OFF by default).
# When RAG_DISABLE_REWRITE=1 (default), behavior is byte-identical to pre-P0.5.
_RAG_DISABLE_REWRITE = _os.environ.get("RAG_DISABLE_REWRITE", "1") == "1"

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

# Module-level registry — set by ext.app.build_ext_routers() at startup
_vector_store = None
_embedder = None
_sessionmaker = None


def configure(*, vector_store, embedder, sessionmaker) -> None:
    global _vector_store, _embedder, _sessionmaker
    _vector_store = vector_store
    _embedder = embedder
    _sessionmaker = sessionmaker


async def get_kb_config_for_chat(chat_id: str, user_id: str) -> Optional[list]:
    """Read the chat's selected KB config from upstream's chat.meta JSON column."""
    if _sessionmaker is None:
        return None
    from sqlalchemy import text
    import json
    try:
        async with _sessionmaker() as s:
            row = (await s.execute(
                text('SELECT meta FROM chat WHERE id = :cid AND user_id = :uid'),
                {"cid": chat_id, "uid": user_id},
            )).first()
            if row and row[0]:
                meta = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                return meta.get("kb_config")
    except Exception as e:
        logger.debug("kb_config lookup failed: %s", e)
    return None


async def retrieve_kb_sources(
    kb_config: list,
    query: str,
    user_id: str,
    chat_id: Optional[str] = None,  # NEW: enables private chat doc retrieval
    history: Optional[List[dict]] = None,  # P0.5: prior turns for query rewrite
) -> List[dict]:
    """Retrieve from our KB pipeline and format as upstream-compatible source dicts.

    Args:
        kb_config: [{"kb_id": 1, "subtag_ids": [...]}, ...]
        query: the user's message text
        user_id: for RBAC verification
        chat_id: current chat ID — if set, private docs in collection chat_{id} are included
        history: prior conversation turns (role/content dicts). When set AND
            RAG_DISABLE_REWRITE=0, the query is rewritten into a standalone form
            before retrieval. Default None preserves pre-P0.5 behavior.

    Returns:
        List of source dicts in upstream's format:
        [{"source": {"id": ..., "name": ...}, "document": [...], "metadata": [...]}, ...]
    """
    if _vector_store is None or _embedder is None or _sessionmaker is None:
        logger.warning("RAG bridge not configured — skipping KB retrieval")
        return []

    if not query:
        return []

    # P0.5: optionally rewrite the query using conversation history.
    # Default OFF (RAG_DISABLE_REWRITE=1) → code path below is byte-identical
    # to the previous release. Flip the flag to engage.
    if not _RAG_DISABLE_REWRITE and history:
        from .query_rewriter import rewrite_query as _rewrite_query
        query = await _rewrite_query(
            latest_turn=query,
            history=history,
            chat_url=_os.environ.get("OPENAI_API_BASE_URL", "http://vllm-chat:8000/v1"),
            chat_model=_os.environ.get("REWRITE_MODEL", _os.environ.get("CHAT_MODEL", "orgchat-chat")),
            api_key=_os.environ.get("OPENAI_API_KEY"),
        )

    # If neither KBs nor chat has private docs, early exit
    if not kb_config and not chat_id:
        return []

    import uuid as _uuid
    request_id_var.set(str(_uuid.uuid4())[:8])
    user_id_var.set(user_id)
    logger.info("rag: request started req=%s user=%s kbs=%d chat=%s",
                request_id_var.get(), user_id_var.get(),
                len(kb_config or []), bool(chat_id))

    # RBAC check — only if kb_config is non-empty
    selected_kbs = []
    if kb_config:
        from .rbac import get_allowed_kb_ids
        async with _sessionmaker() as s:
            allowed = set(await get_allowed_kb_ids(s, user_id=user_id))
        selected_kbs = [cfg for cfg in kb_config if cfg.get("kb_id") in allowed]
        if not selected_kbs and not chat_id:
            return []

    # Retrieve using our pipeline
    from .retriever import retrieve
    from .reranker import rerank, rerank_with_flag
    from .budget import budget_chunks

    try:
        raw_hits = await retrieve(
            query=query,
            selected_kbs=selected_kbs,
            chat_id=chat_id,  # NOW PASSED THROUGH
            vector_store=_vector_store,
            embedder=_embedder,
            per_kb_limit=10,
            total_limit=30,
        )
        # P1.2 — dispatch through rerank_with_flag. Default (RAG_RERANK unset/0)
        # calls ``rerank`` which is byte-identical to the previous behaviour.
        reranked = rerank_with_flag(query, raw_hits, top_k=10, fallback_fn=rerank)

        # P1.3 — MMR diversification (flag-gated). Reads the flag at call time
        # so tests can monkeypatch the env without module reload. When the flag
        # is off (default) the ``mmr`` module is never imported. Fails open:
        # any MMR error is swallowed and we keep the reranker output.
        if _os.environ.get("RAG_MMR", "0") == "1" and reranked:
            try:
                from .mmr import mmr_rerank_from_hits
                _mmr_lambda = float(_os.environ.get("RAG_MMR_LAMBDA", "0.7"))
                reranked = await mmr_rerank_from_hits(
                    query, reranked, _embedder,
                    top_k=len(reranked), lambda_=_mmr_lambda,
                )
            except Exception:
                # Fail open: on any MMR error, stick with reranker output.
                pass

        # P1.4 context-expansion hook goes here — leave a blank line.

        budgeted = budget_chunks(reranked, max_tokens=4000)
    except Exception as e:
        logger.exception("KB retrieval failed: %s", e)
        return []

    if not budgeted:
        return []

    # Group by source document for cleaner citation display
    sources_by_doc: dict[str, dict] = {}
    for hit in budgeted:
        # For private chat docs, payload may have chat_id instead of doc_id
        doc_id = str(hit.payload.get("doc_id", ""))
        chat_scope = hit.payload.get("chat_id")
        filename = str(hit.payload.get("filename", ""))
        if not filename:
            if chat_scope is not None:
                filename = f"private-doc (chat {chat_scope})"
            else:
                filename = f"kb-{hit.payload.get('kb_id', '?')}"
        text_content = str(hit.payload.get("text", ""))

        # P0.6 — spotlighting against indirect prompt injection.
        # Read flag at call time so tests can toggle via monkeypatch.setenv
        # without module reload. Default (flag=0) → byte-identical to pre-P0.6.
        if _os.environ.get("RAG_SPOTLIGHT", "0") == "1" and text_content:
            from .spotlight import wrap_context as _spotlight_wrap
            text_content = _spotlight_wrap(text_content)

        # Key: prefer doc_id for KB docs, fall back to (chat_id, chunk_index) for private
        if doc_id:
            key = f"kb_{hit.payload.get('kb_id')}_{doc_id}"
        else:
            key = f"chat_{chat_scope}_{hit.payload.get('chunk_index', id(hit))}"

        if key not in sources_by_doc:
            sources_by_doc[key] = {
                "source": {
                    "id": key,
                    "name": filename,
                },
                "document": [],
                "metadata": [],
            }
        sources_by_doc[key]["document"].append(text_content)
        sources_by_doc[key]["metadata"].append({
            "source": filename,
            "kb_id": hit.payload.get("kb_id"),
            "subtag_id": hit.payload.get("subtag_id"),
            "doc_id": doc_id or None,
            "chat_id": chat_scope,
        })

    return list(sources_by_doc.values())
