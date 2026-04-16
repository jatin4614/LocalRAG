"""Parallel multi-KB + optional chat-private retrieval."""
from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence

from .embedder import Embedder
from .vector_store import Hit, VectorStore


async def retrieve(
    *,
    query: str,
    selected_kbs: Sequence[dict],
    chat_id: Optional[int],
    vector_store: VectorStore,
    embedder: Embedder,
    per_kb_limit: int = 10,
    total_limit: int = 30,
) -> List[Hit]:
    """Run parallel searches against each selected KB and an optional chat namespace.

    selected_kbs shape: [{"kb_id": int, "subtag_ids": [int, ...]}, ...]
        empty subtag_ids → search all subtags in that KB.
    Returns a flat list of Hit objects sorted by raw score descending, trimmed to total_limit.
    """
    [qvec] = await embedder.embed([query])

    async def _search_kb(cfg: dict) -> List[Hit]:
        kb_id = cfg["kb_id"]
        subtag_ids = cfg.get("subtag_ids") or None
        try:
            return await vector_store.search(
                f"kb_{kb_id}", qvec, limit=per_kb_limit, subtag_ids=subtag_ids,
            )
        except Exception:
            return []

    async def _search_chat() -> List[Hit]:
        if chat_id is None:
            return []
        try:
            return await vector_store.search(f"chat_{chat_id}", qvec, limit=per_kb_limit)
        except Exception:
            return []

    tasks = [_search_kb(cfg) for cfg in selected_kbs]
    tasks.append(_search_chat())
    results = await asyncio.gather(*tasks)

    flat: list[Hit] = []
    for lst in results:
        flat.extend(lst)
    flat.sort(key=lambda h: h.score, reverse=True)
    return flat[:total_limit]
