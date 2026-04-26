"""Temporal-then-semantic RAPTOR tree builder.

Plan B Phase 5.5. Replaces the flat RAPTOR for collections that have
``shard_key`` payload (i.e. temporally-sharded collections from Phase 5.4).

Tree layout:

  L0 — original chunks (verbatim, untouched, NOT emitted by this module).
  L1 — per-month subtree summaries. A small (≤ 2-3 chunks) month gets a
       single summary node; a larger month would in principle be subdivided
       semantically, but for the v0 of this builder we collapse to one
       monthly summary per shard_key. Future work: re-introduce GMM
       clustering inside a month for very-large monthly buckets.
  L2 — per-quarter summaries (3 monthly nodes → 1 quarterly node, prompted
       to highlight changes vs the prior quarter).
  L3 — per-year summaries (4 quarterly nodes → 1 yearly node).
  L4 — 3-year meta summary (only if corpus spans >1 year).

Each node payload includes:

  - ``level``: 1 | 2 | 3 | 4
  - ``shard_key``: the lowest-level shard_key (or one of them) the node covers
  - ``time_range``: ``{"start": "YYYY-MM", "end": "YYYY-MM"}``
  - ``source_chunk_ids``: ``[int]`` (the L0 leaves the L1 node covers; only
    populated on L1 nodes — higher levels are summaries-of-summaries)

Concurrency for ``summarize`` calls is governed by an internal semaphore
(``RAG_TEMPORAL_RAPTOR_CONCURRENCY``, default 4) so a tree build cannot
runaway against vllm-chat. Embedding is batched per-level.

The injected ``summarize`` and ``embed`` callbacks let the caller plug in
any LLM (vllm-chat in production, fake echoes in unit tests). They are
async to align with the rest of the ingest pipeline.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
from typing import Awaitable, Callable, Optional, Tuple


log = logging.getLogger("orgchat.temporal_raptor")


_SummarizeCallable = Callable[[str], Awaitable[str]]
_EmbedCallable = Callable[[list[str]], Awaitable[list[list[float]]]]


def quarter_for_shard_key(sk: str) -> Tuple[str, int, int]:
    """Return ``(quarter_label, year, quarter_num)`` for a 'YYYY-MM' shard_key."""
    from .temporal_shard import parse_shard_key
    y, m = parse_shard_key(sk)
    q = (m - 1) // 3 + 1
    return f"{y:04d}-Q{q}", y, q


def group_chunks_by_shard_key(chunks: list[dict]) -> dict[str, list[dict]]:
    """Bucket chunks by their ``shard_key`` payload (chunks without one are dropped)."""
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for c in chunks:
        sk = c.get("shard_key")
        if sk:
            grouped[sk].append(c)
    return grouped


def build_quarter_prompt(
    quarter_label: str,
    month_summaries: list[str],
    prior_quarter_summary: Optional[str],
) -> str:
    months_block = "\n\n".join(
        f"  Month {i + 1}: {ms}" for i, ms in enumerate(month_summaries)
    )
    if prior_quarter_summary:
        prior_block = (
            f"\nPrior quarter ({quarter_label}'s predecessor):\n"
            f"  {prior_quarter_summary}\n"
        )
        instruction = (
            f"Summarize the following month-summaries from {quarter_label}. "
            "Note what changed compared to the prior quarter."
        )
    else:
        prior_block = ""
        instruction = (
            f"Summarize the following month-summaries from {quarter_label}. "
            "This is the first quarter in the corpus — no prior to compare "
            "against."
        )

    return f"""{instruction}

{months_block}{prior_block}

Quarterly summary:"""


def build_year_prompt(year: int, quarter_summaries: list[str]) -> str:
    qb = "\n\n".join(
        f"  Q{i + 1}: {qs}" for i, qs in enumerate(quarter_summaries)
    )
    return f"""Synthesize the following quarterly summaries from {year} into a year-in-review. Highlight cross-quarter trends.

{qb}

Annual summary:"""


def build_meta_prompt(year_summaries: list[str]) -> str:
    yb = "\n\n".join(year_summaries)
    return f"""Synthesize the following yearly summaries into a 3-year overview. Highlight long-term trends and inflection points.

{yb}

3-year synthesis:"""


def build_month_prompt(shard_key: str, chunk_texts: list[str]) -> str:
    """Prompt for the monthly (L1) summary fallback path."""
    joined = "\n\n".join(t[:2000] for t in chunk_texts)
    return f"""Summarize the following content from {shard_key} in 3-5 declarative sentences. Capture themes, named entities, and specific facts. Do NOT use phrases like 'the document says'.

{joined}

Monthly summary:"""


async def _gather_with_concurrency(coros, *, limit: int):
    sem = asyncio.Semaphore(max(1, limit))

    async def _wrap(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(_wrap(c) for c in coros))


async def build_temporal_tree(
    *,
    chunks: list[dict],
    summarize: _SummarizeCallable,
    embed: _EmbedCallable,
    chat_model: str,
    concurrency: Optional[int] = None,
) -> list[dict]:
    """Build the temporal-semantic tree.

    ``chunks`` is the flat list of L0 leaves with ``shard_key`` payload.
    ``summarize(prompt) -> str`` is an injected LLM caller.
    ``embed(texts) -> [[float]]`` embeds the resulting summary text.

    Returns a list of node dicts ready for upsert. Each has::

        {
          "text": str,
          "embedding": list[float],
          "payload": {
            "level": int,                 # 1, 2, 3, or 4
            "shard_key": str,             # anchor shard_key
            "time_range": {"start": "YYYY-MM", "end": "YYYY-MM"},
            ...
          }
        }

    Plan B Phase 5.5.
    """
    if concurrency is None:
        try:
            concurrency = int(
                os.environ.get("RAG_TEMPORAL_RAPTOR_CONCURRENCY", "4")
            )
        except (ValueError, TypeError):
            concurrency = 4

    by_sk = group_chunks_by_shard_key(chunks)
    if not by_sk:
        return []

    nodes: list[dict] = []
    monthly_summaries: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Stage 1: per-month L1 summary nodes.
    # ------------------------------------------------------------------
    sk_order = sorted(by_sk)

    async def _summarize_month(sk: str) -> tuple[str, str, list[int]]:
        month_chunks = by_sk[sk]
        prompt = build_month_prompt(sk, [c["text"] for c in month_chunks])
        try:
            text = await summarize(prompt)
        except Exception as e:  # noqa: BLE001 — fail-open per month
            log.warning("L1 summarize failed for %s: %s", sk, e)
            text = ""
        if not text:
            text = ""
        source_ids = [int(c.get("chunk_index", 0)) for c in month_chunks]
        return sk, text, source_ids

    month_results = await _gather_with_concurrency(
        [_summarize_month(sk) for sk in sk_order], limit=concurrency,
    )

    # Embed all monthly summaries in one batch
    month_texts = [t for _, t, _ in month_results if t]
    if month_texts:
        month_embeds = await embed(month_texts)
    else:
        month_embeds = []
    embed_iter = iter(month_embeds)
    for sk, text, source_ids in month_results:
        if not text:
            continue
        emb = next(embed_iter)
        monthly_summaries[sk] = text
        nodes.append({
            "text": text,
            "embedding": emb,
            "payload": {
                "level": 1,
                "shard_key": sk,
                "time_range": {"start": sk, "end": sk},
                "source_chunk_ids": source_ids,
            },
        })

    # ------------------------------------------------------------------
    # Stage 2a: L2 per-quarter (sequential — needs prior quarter summary).
    # ------------------------------------------------------------------
    quarterly_summaries: dict[tuple[int, int], str] = {}
    by_quarter: dict[tuple[int, int], list[tuple[str, str]]] = (
        collections.defaultdict(list)
    )
    for sk, summary in monthly_summaries.items():
        _, y, q = quarter_for_shard_key(sk)
        by_quarter[(y, q)].append((sk, summary))

    quarter_pending: list[tuple[tuple[int, int], list[tuple[str, str]]]] = (
        sorted(by_quarter.items())
    )
    for (y, q), pairs in quarter_pending:
        pairs.sort()  # by shard_key
        quarter_label = f"{y:04d}-Q{q}"
        prior_key = (y, q - 1) if q > 1 else (y - 1, 4)
        prior = quarterly_summaries.get(prior_key)
        prompt = build_quarter_prompt(
            quarter_label=quarter_label,
            month_summaries=[s for _, s in pairs],
            prior_quarter_summary=prior,
        )
        try:
            text = await summarize(prompt)
        except Exception as e:  # noqa: BLE001
            log.warning("L2 summarize failed for %s: %s", quarter_label, e)
            text = ""
        if not text:
            continue
        [embedding] = await embed([text])
        quarterly_summaries[(y, q)] = text
        first_sk = pairs[0][0]
        last_sk = pairs[-1][0]
        nodes.append({
            "text": text,
            "embedding": embedding,
            "payload": {
                "level": 2,
                "shard_key": first_sk,
                "time_range": {"start": first_sk, "end": last_sk},
                "quarter_label": quarter_label,
            },
        })

    # ------------------------------------------------------------------
    # Stage 2b: L3 per-year.
    # ------------------------------------------------------------------
    yearly_summaries: dict[int, str] = {}
    by_year: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    for (y, q), s in quarterly_summaries.items():
        by_year[y].append((q, s))
    for y, qpairs in sorted(by_year.items()):
        qpairs.sort()
        prompt = build_year_prompt(
            year=y, quarter_summaries=[s for _, s in qpairs],
        )
        try:
            text = await summarize(prompt)
        except Exception as e:  # noqa: BLE001
            log.warning("L3 summarize failed for %d: %s", y, e)
            text = ""
        if not text:
            continue
        [embedding] = await embed([text])
        yearly_summaries[y] = text
        nodes.append({
            "text": text,
            "embedding": embedding,
            "payload": {
                "level": 3,
                "shard_key": f"{y:04d}-12",  # year-end as anchor
                "time_range": {"start": f"{y:04d}-01", "end": f"{y:04d}-12"},
                "year": y,
            },
        })

    # ------------------------------------------------------------------
    # Stage 2c: L4 meta — only if more than one year.
    # ------------------------------------------------------------------
    if len(yearly_summaries) > 1:
        years_sorted = sorted(yearly_summaries)
        prompt = build_meta_prompt(
            year_summaries=[
                f"{y}: {yearly_summaries[y]}" for y in years_sorted
            ],
        )
        try:
            text = await summarize(prompt)
        except Exception as e:  # noqa: BLE001
            log.warning("L4 meta summarize failed: %s", e)
            text = ""
        if text:
            [embedding] = await embed([text])
            nodes.append({
                "text": text,
                "embedding": embedding,
                "payload": {
                    "level": 4,
                    "shard_key": f"{years_sorted[-1]:04d}-12",
                    "time_range": {
                        "start": f"{years_sorted[0]:04d}-01",
                        "end": f"{years_sorted[-1]:04d}-12",
                    },
                    "is_meta": True,
                },
            })

    return nodes


__all__ = [
    "build_temporal_tree",
    "group_chunks_by_shard_key",
    "build_quarter_prompt",
    "build_year_prompt",
    "build_meta_prompt",
    "build_month_prompt",
    "quarter_for_shard_key",
]
