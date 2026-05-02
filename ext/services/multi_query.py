"""Multi-query decomposition (Phase 6.X — Method 3).

Pure logic — no I/O, no LLM, no Qdrant. Three functions feed the bridge's
``retrieve_kb_sources`` decompose branch:

* :func:`should_decompose` — gate predicate.
* :func:`build_sub_queries` — one focus-shifted sub-query per entity.
* :func:`merge_with_quota` — merges N per-entity hit lists with a
  per-entity floor (so the lowest-frequency entity is not crowded out
  at the rerank cut), deduping by point id and capping at a total.

Why focus-shifted sub-queries instead of "<entity>" alone:
the original query carries the *intent* (date, topic, format
preference). An embedding of just ``"32 Inf Bde"`` retrieves
generic mentions across the corpus including unrelated months. We
keep the original query and append a focus suffix so the dense
vector tilts toward the entity while the date / temporal signal is
preserved.

The merge algorithm:
  1. Per-entity round-robin until each entity hit its floor (or ran out).
  2. Then top-up by raw score across all remaining candidates (deduped
     by point id — the same chunk can show up in two entity buckets).
  3. Final list is sorted by score descending and capped at
     ``k_total``. Ties broken by stable original order.

Hit shape contract: any object with ``.id`` and ``.score`` attributes
works (``ext.services.retriever.Hit`` is the canonical type; tests use
a fake dataclass). We do NOT touch ``.payload``.
"""
from __future__ import annotations

from typing import Any, Sequence


# The bridge's ``_intent`` value when classification didn't run; treated
# as "default — decompose if entities exist". Metadata is the only intent
# that explicitly skips decomposition (catalog questions don't fan out).
_NON_DECOMPOSING_INTENTS = frozenset({"metadata"})


def should_decompose(
    *,
    entities: Sequence[str],
    flag_on: bool,
    intent: str | None,
) -> bool:
    """Gate predicate for multi-query decomposition.

    Returns ``True`` iff all of:
      * ``flag_on`` — the per-KB or global ``RAG_MULTI_ENTITY_DECOMPOSE``
        flag is enabled.
      * ``len(entities) >= 2`` — the extractor produced ≥2 entities.
        Single-entity queries go through the existing single-query path.
      * ``intent`` is not ``"metadata"``. Catalog/enumeration queries
        (``"list documents"``, ``"what files do I have"``) don't fan out.

    ``intent=None`` is treated as decomposable — defensive for the case
    where the intent classifier failed and the bridge is operating
    blind. Better to fan out and accept the small extra cost than to
    silently fall through to the broken single-query path.
    """
    if not flag_on:
        return False
    if not entities or len(entities) < 2:
        return False
    if intent in _NON_DECOMPOSING_INTENTS:
        return False
    return True


def build_sub_queries(
    query: str,
    entities: Sequence[str],
) -> list[tuple[str, str]]:
    """Return ``[(entity, sub_query), ...]`` one per entity.

    Each sub-query is the original query plus a focus suffix naming
    the entity. Order matches ``entities``. Empty input yields an
    empty list.

    Shape: ``"<original query> (focus on <entity>)"``. Suffix is
    appended (not prefixed) so date/topic tokens at the start of the
    query stay near the front of the embedding (BAAI/bge-m3 weighs
    early tokens slightly higher in cosine similarity).
    """
    if not entities:
        return []
    base = (query or "").strip() or "(no query)"
    return [(e, f"{base} (focus on {e})") for e in entities]


def merge_with_quota(
    *,
    per_entity_hits: dict[str, list[Any]],
    k_min_per_entity: int,
    k_total: int,
) -> list[Any]:
    """Merge per-entity hit lists with a per-entity floor + total cap.

    Algorithm:
      1. **Quota pass.** Round-robin across entities. For each entity,
         keep its top ``k_min_per_entity`` hits (or all of them, if it
         has fewer). Hits are taken in their existing per-bucket order
         — the caller is responsible for sorting per-bucket by score.
      2. **Top-up pass.** From the remaining candidates across all
         entities, take the highest-scoring ones until ``k_total``
         is reached.
      3. **Final sort.** Output is sorted by ``score`` descending. Ties
         are broken by stable insertion order from the union of input
         buckets — first occurrence wins on tie.

    Dedup is by ``hit.id``. When the same id appears in multiple
    buckets (the same chunk semantically matched two entity sub-queries),
    we keep the **higher-scoring copy** before applying the algorithm
    above. This is important when Method 4 (per-entity text filter) is
    OFF — without the filter, dense similarity can rank the same chunk
    high under two different sub-queries, and we want the better
    score-per-entity to count.

    Returns a flat list of hits, length ≤ ``k_total``.
    """
    # Step 0 — flatten + dedupe by id, keeping the higher-scoring copy.
    # Track which entity contributed each surviving hit (the first one
    # that produced its winning score) so the quota pass can iterate.
    best_for_id: dict[Any, tuple[Any, str]] = {}
    for entity, hits in per_entity_hits.items():
        for h in hits:
            prev = best_for_id.get(h.id)
            if prev is None or h.score > prev[0].score:
                best_for_id[h.id] = (h, entity)

    # Build per-entity bucket from deduped hits (entity → list of hits,
    # ordered by score desc within each bucket).
    bucket: dict[str, list[Any]] = {e: [] for e in per_entity_hits.keys()}
    for hit, entity in best_for_id.values():
        bucket[entity].append(hit)
    for e in bucket:
        bucket[e].sort(key=lambda h: h.score, reverse=True)

    # Step 1 — quota pass.
    quota: list[Any] = []
    quota_ids: set[Any] = set()
    for e, hits in bucket.items():
        for h in hits[:k_min_per_entity]:
            if h.id not in quota_ids:
                quota.append(h)
                quota_ids.add(h.id)

    # Step 2 — top-up pass.
    leftover: list[Any] = []
    for e, hits in bucket.items():
        for h in hits[k_min_per_entity:]:
            if h.id not in quota_ids:
                leftover.append(h)
    leftover.sort(key=lambda h: h.score, reverse=True)
    remaining_slots = max(0, k_total - len(quota))
    quota.extend(leftover[:remaining_slots])

    # Step 3 — final sort by score, capped.
    quota.sort(key=lambda h: h.score, reverse=True)
    return quota[:k_total]


__all__ = [
    "should_decompose",
    "build_sub_queries",
    "merge_with_quota",
]
