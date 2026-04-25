"""RBAC: resolve which KB ids a user is allowed to read."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import KBAccess, KnowledgeBase
from .obs import span

_logger = logging.getLogger(__name__)


async def get_allowed_kb_ids(session: AsyncSession, *, user_id: str) -> List[int]:
    """Return list of kb_ids the given user can read.

    Admins see every non-deleted KB. Regular users see KBs matched by:
      - direct user grant in kb_access, OR
      - group grant for a group they belong to.
    """
    with span("rbac.check", user_id=str(user_id)) as _sp:
        # Use raw SQL for upstream's "user" table (singular name, UUID id)
        row = (await session.execute(
            text('SELECT role FROM "user" WHERE id = :uid'), {"uid": user_id}
        )).first()
        if row is None:
            try:
                _sp.set_attribute("decision", "no_user")
                _sp.set_attribute("kb_ids", "")
            except Exception:
                pass
            return []
        role = row[0]

        if role == "admin":
            rows = (await session.execute(
                select(KnowledgeBase.id).where(KnowledgeBase.deleted_at.is_(None))
            )).scalars().all()
            allowed = list(rows)
            try:
                _sp.set_attribute("decision", "admin_all")
                _sp.set_attribute("kb_ids", ",".join(str(x) for x in allowed))
            except Exception:
                pass
            return allowed

        # Get user's group IDs from upstream's "group_member" table
        group_rows = (await session.execute(
            text('SELECT group_id FROM group_member WHERE user_id = :uid'), {"uid": user_id}
        )).scalars().all()
        group_ids = list(group_rows)

        conditions = [KBAccess.user_id == user_id]
        if group_ids:
            conditions.append(KBAccess.group_id.in_(group_ids))

        rows = (await session.execute(
            select(KBAccess.kb_id).where(or_(*conditions))
        )).scalars().all()
        allowed = sorted(set(rows))
        try:
            _sp.set_attribute("decision", "granted" if allowed else "no_grants")
            _sp.set_attribute("kb_ids", ",".join(str(x) for x in allowed))
        except Exception:
            pass
        return allowed


async def resolved_allowed_kb_ids(
    session: AsyncSession,
    *,
    user_id: str,
    redis: Optional[Any] = None,
) -> set[int]:
    """Cache-first wrapper around :func:`get_allowed_kb_ids`.

    Mirrors the production flow used in
    :mod:`ext.services.chat_rag_bridge._run_pipeline` so every KB-RBAC
    call site shares one read path:

    1. Look up ``rbac:user:{user_id}`` in Redis (TTL =
       ``RAG_RBAC_CACHE_TTL_SECS``, default 30s).
    2. Cache hit -> return immediately.
    3. Cache miss / cache outage / corrupt value -> hit Postgres via
       :func:`get_allowed_kb_ids`, then write the result back to the
       cache. Cache write failure is non-fatal (next request just
       re-fetches).

    Sacred CLAUDE.md §2 invariant: the DB miss path MUST always run
    when the cache returns ``None``. The cache is purely an
    accelerator — never a gate that could weaken isolation under a
    Redis outage.

    Args:
        session:  active AsyncSession for the DB-side fallback.
        user_id:  caller's user id (string for namespace stability).
        redis:    optional async redis handle for the cache layer. If
                  ``None`` the cache is bypassed and we go straight to
                  the DB — this is the "pure DB lookup" path, used by
                  call sites that don't have a redis handle wired up
                  yet (e.g. unit tests with a fake session).

    Returns:
        ``set[int]`` of allowed kb_ids.
    """
    if redis is not None:
        try:
            from .rbac_cache import get_shared_cache

            cache = get_shared_cache(redis=redis)
            cached = await cache.get(user_id=str(user_id))
            if cached is not None:
                return cached
        except Exception as exc:  # noqa: BLE001
            # Fail-open on cache errors: log and fall through to DB.
            _logger.debug("rbac cache get failed: %s", exc)

    allowed = set(await get_allowed_kb_ids(session, user_id=user_id))

    if redis is not None:
        try:
            from .rbac_cache import get_shared_cache

            cache = get_shared_cache(redis=redis)
            await cache.set(user_id=str(user_id), allowed_kb_ids=allowed)
        except Exception as exc:  # noqa: BLE001
            _logger.debug("rbac cache set failed: %s", exc)

    return allowed


async def users_affected_by_grant(
    session: AsyncSession, grant: Any
) -> List[str]:
    """Return user ids whose ``allowed_kb_ids`` could change due to ``grant``.

    Used by the RBAC cache (Phase 1.5) to publish targeted invalidation
    after a ``kb_access`` mutation. We MUST NOT under-report -- a missed
    user keeps a stale cache entry until TTL expiry, which can leak a
    revoked KB for up to ``RAG_RBAC_CACHE_TTL_SECS`` seconds.

    Two grant shapes:

    * Direct user grant (``grant.user_id`` is set, ``grant.group_id`` is
      None) -> exactly one user is affected.
    * Group grant (``grant.group_id`` is set, ``grant.user_id`` is None)
      -> every current member of the group is affected. We resolve
      membership against upstream's ``group_member`` table (the same one
      :func:`get_allowed_kb_ids` reads, so the cache invariant holds).
    """
    user_id = getattr(grant, "user_id", None)
    if user_id is not None:
        return [str(user_id)]
    group_id = getattr(grant, "group_id", None)
    if group_id is None:
        # Defensive: a malformed grant with neither id should never reach
        # us (KBAccess.__init__ enforces XOR), but if it does we return
        # an empty list rather than raising -- cache invalidation is
        # best-effort and the TTL safety net catches the gap.
        return []
    rows = (
        await session.execute(
            text(
                'SELECT user_id FROM group_member WHERE group_id = :gid'
            ),
            {"gid": group_id},
        )
    ).scalars().all()
    return [str(r) for r in rows]
