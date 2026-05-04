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
    subtopics: Sequence[str] = (),
    flag_on: bool,
    intent: str | None,
) -> tuple[str, bool]:
    """Two-axis gate predicate for multi-query decomposition.

    Returns ``(mode, enabled)`` where ``mode`` is one of:

    * ``"none"``     — single-axis behaviour, no decompose
    * ``"entity"``   — N entities, M < 2 subtopics — current single-axis path
    * ``"subtopic"`` — M subtopics, N < 2 entities — fan out by subtopic only
    * ``"both"``     — N×M sub-queries with two-axis quotas

    Bound conditions (independent of mode):
    * ``flag_on`` — env or per-KB master gate
    * ``intent`` is not ``"metadata"`` — catalog questions never decompose

    ``intent=None`` is treated as decomposable (defensive — mirror of the
    original gate behaviour).
    """
    if not flag_on:
        return ("none", False)
    if intent in _NON_DECOMPOSING_INTENTS:
        return ("none", False)

    n_e = len(entities or [])
    n_s = len(subtopics or [])

    if n_e >= 2 and n_s >= 2:
        return ("both", True)
    if n_e >= 2:
        return ("entity", True)
    if n_s >= 2:
        return ("subtopic", True)
    return ("none", False)


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


def build_sub_queries_two_axis(
    query: str,
    entities: Sequence[str],
    subtopics: Sequence[str],
) -> list[tuple[str, str, str]]:
    """Return ``[(entity, subtopic, sub_query), ...]`` one per cell.

    For each (entity, subtopic) pair, build a focus-suffixed sub-query.
    Order is entity-major, subtopic-minor (so all of A's subtopics come
    before B's). Empty entity OR empty subtopic list yields ``[]``.

    Suffix format: ``"<original> (focus on <entity> — <subtopic>)"``.
    """
    if not entities or not subtopics:
        return []
    base = (query or "").strip() or "(no query)"
    return [
        (e, s, f"{base} (focus on {e} — {s})")
        for e in entities
        for s in subtopics
    ]


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


def merge_with_two_axis_quota(
    *,
    per_cell_hits: dict[tuple[str, str], list[Any]],
    k_min_per_cell: int,
    k_min_per_entity: int,
    k_min_per_subtopic: int,
    k_total: int,
) -> list[Any]:
    """Merge per-(entity, subtopic) hit lists with quotas at three levels.

    Algorithm:
      1. **Cell-quota pass.** For each (entity, subtopic) cell, take its
         top ``k_min_per_cell`` hits in score-desc order.
      2. **Entity-floor recovery.** For each entity, ensure its total
         pool size is ≥ ``k_min_per_entity`` — if a cell was empty,
         pull more from cells that did have hits for that entity.
      3. **Subtopic-floor recovery.** Same for subtopics — if a subtopic
         is under-represented, pull from any cell that had hits for it.
      4. **Top-up pass.** Fill remaining slots up to ``k_total`` by
         score-desc across deduped non-quota leftovers.
      5. **Final sort + cap.** Sort by score desc, cap at ``k_total``.

    Dedup is by ``hit.id``. When the same hit appears in multiple cells
    (the same chunk semantically matched two sub-queries), the highest
    score copy wins, then the algorithm above runs.

    Hit shape contract: any object with ``.id`` and ``.score``
    (matches ``merge_with_quota``).

    Returns a flat list of hits, length ≤ ``k_total``.
    """
    # Step 0 — flatten + dedupe by id, keep highest score copy + remember
    # which (entity, subtopic) cell first picked it.
    best_for_id: dict[Any, tuple[Any, str, str]] = {}
    for (entity, subtopic), hits in per_cell_hits.items():
        for h in hits:
            prev = best_for_id.get(h.id)
            if prev is None or h.score > prev[0].score:
                best_for_id[h.id] = (h, entity, subtopic)

    # Rebuild bucket dict from deduped hits (cell-keyed)
    bucket: dict[tuple[str, str], list[Any]] = {
        k: [] for k in per_cell_hits.keys()
    }
    for hit, e, s in best_for_id.values():
        bucket[(e, s)].append(hit)
    for k in bucket:
        bucket[k].sort(key=lambda h: h.score, reverse=True)

    selected_ids: set = set()
    selected: list[Any] = []

    def _take(hit: Any) -> None:
        if hit.id not in selected_ids:
            selected_ids.add(hit.id)
            selected.append(hit)

    # Step 1 — cell-quota
    for (e, s), hits in bucket.items():
        for h in hits[:k_min_per_cell]:
            _take(h)

    # Step 2 — entity-floor recovery
    # Filter out the __leftover__ synthetic cell — it's not a real entity.
    # Leftover hits flow through the top-up pass (step 4) instead.
    entities = sorted({
        e for (e, _) in bucket.keys() if e != "__leftover__"
    })
    for entity in entities:
        # Count current selection for this entity
        ent_selected = [
            h for h in selected
            if any(
                h.id in {hh.id for hh in bucket.get((entity, s), [])}
                for s in {ss for (ee, ss) in bucket.keys() if ee == entity}
            )
        ]
        if len(ent_selected) >= k_min_per_entity:
            continue
        deficit = k_min_per_entity - len(ent_selected)
        # Pull more from any cell of this entity, score-desc
        candidates = []
        for s in {ss for (ee, ss) in bucket.keys() if ee == entity}:
            for h in bucket.get((entity, s), []):
                if h.id not in selected_ids:
                    candidates.append(h)
        candidates.sort(key=lambda h: h.score, reverse=True)
        for h in candidates[:deficit]:
            _take(h)

    # Step 3 — subtopic-floor recovery (mirror of step 2)
    subtopics = sorted({
        s for (_, s) in bucket.keys() if s != "__leftover__"
    })
    for subtopic in subtopics:
        sub_selected = [
            h for h in selected
            if any(
                h.id in {hh.id for hh in bucket.get((e, subtopic), [])}
                for e in {ee for (ee, ss) in bucket.keys() if ss == subtopic}
            )
        ]
        if len(sub_selected) >= k_min_per_subtopic:
            continue
        deficit = k_min_per_subtopic - len(sub_selected)
        candidates = []
        for e in {ee for (ee, ss) in bucket.keys() if ss == subtopic}:
            for h in bucket.get((e, subtopic), []):
                if h.id not in selected_ids:
                    candidates.append(h)
        candidates.sort(key=lambda h: h.score, reverse=True)
        for h in candidates[:deficit]:
            _take(h)

    # Step 4 — top-up by score across remaining leftovers
    if len(selected) < k_total:
        leftover = [
            h for hits in bucket.values() for h in hits
            if h.id not in selected_ids
        ]
        leftover.sort(key=lambda h: h.score, reverse=True)
        for h in leftover[: k_total - len(selected)]:
            _take(h)

    # Step 5 — final sort + cap
    selected.sort(key=lambda h: h.score, reverse=True)
    return selected[:k_total]


__all__ = [
    "should_decompose",
    "build_sub_queries",
    "build_sub_queries_two_axis",
    "merge_with_quota",
    "merge_with_two_axis_quota",
]
