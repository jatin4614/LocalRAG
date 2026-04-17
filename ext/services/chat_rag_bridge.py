"""Bridge between upstream's chat middleware and our KB RAG pipeline.

Called from the patched process_chat_payload() in middleware.py.
Retrieves KB context and returns it in upstream's source dict format.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("orgchat.rag_bridge")

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
) -> List[dict]:
    """Retrieve from our KB pipeline and format as upstream-compatible source dicts.

    Args:
        kb_config: [{"kb_id": 1, "subtag_ids": [...]}, ...]
        query: the user's message text
        user_id: for RBAC verification

    Returns:
        List of source dicts in upstream's format:
        [{"source": {"id": ..., "name": ...}, "document": [...], "metadata": [...]}, ...]
    """
    if _vector_store is None or _embedder is None or _sessionmaker is None:
        logger.warning("RAG bridge not configured — skipping KB retrieval")
        return []

    if not kb_config or not query:
        return []

    # RBAC check: verify user still has access to these KBs
    from .rbac import get_allowed_kb_ids
    async with _sessionmaker() as s:
        allowed = set(await get_allowed_kb_ids(s, user_id=user_id))

    selected_kbs = [
        cfg for cfg in kb_config
        if cfg.get("kb_id") in allowed
    ]
    if not selected_kbs:
        return []

    # Retrieve using our pipeline
    from .retriever import retrieve
    from .reranker import rerank
    from .budget import budget_chunks

    try:
        raw_hits = await retrieve(
            query=query,
            selected_kbs=selected_kbs,
            chat_id=None,  # KB retrieval only, not private docs here
            vector_store=_vector_store,
            embedder=_embedder,
            per_kb_limit=10,
            total_limit=30,
        )
        reranked = rerank(raw_hits, top_k=10)
        budgeted = budget_chunks(reranked, max_tokens=4000)
    except Exception as e:
        logger.exception("KB retrieval failed: %s", e)
        return []

    if not budgeted:
        return []

    # Group by source document for cleaner citation display
    sources_by_doc: dict[str, dict] = {}
    for hit in budgeted:
        doc_id = str(hit.payload.get("doc_id", "unknown"))
        filename = str(hit.payload.get("filename", f"kb-{hit.payload.get('kb_id', '?')}"))
        text_content = str(hit.payload.get("text", ""))

        key = f"kb_{hit.payload.get('kb_id')}_{doc_id}"
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
            "doc_id": doc_id,
        })

    return list(sources_by_doc.values())
