"""Six isolation tests for RBAC cache (CLAUDE.md §2 invariant).

Sacred matrix:

1. User A in group X -> query -> sees kb_X results.
2. Admin revokes A from X -> pub/sub fires -> A's cache invalidated ->
   next query denies.
3. Cache TTL expires naturally -> permission re-fetch picks up the
   revocation even when the pub/sub message was lost.
4. Pub/sub message dropped (subscriber disabled) -> TTL acts as the
   safety net.
5. Concurrent queries from A during a revocation window -> no query
   returns a partial union of pre/post grants.
6. Two users A and B with different KB access -> their cache keys do
   not collide.

The fixtures (``rbac_db_session``, ``redis_client``, ``test_user_a``,
``test_user_b``, ``test_kb_x``, ``test_kb_y``, ``test_group_x``,
``assign_user_to_group``, ``revoke_user_from_group``,
``grant_kb_to_group``, ``grant_kb_to_user``, ``query_allowed_ids``,
``monkeypatch_ttl_to_1sec``, ``disable_pubsub_subscribe``) are defined
in ``tests/integration/conftest.py``.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_1_user_in_group_sees_kb(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_kb_x,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    query_allowed_ids,
):
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    allowed = await query_allowed_ids(test_user_a)
    assert test_kb_x in allowed, (
        f"user A ({test_user_a}) in group X ({test_group_x}) "
        f"with grant on kb_x ({test_kb_x}) should see kb_x; got {allowed}"
    )


@pytest.mark.asyncio
async def test_2_revocation_invalidates_cache_via_pubsub(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_kb_x,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    revoke_user_from_group,
    query_allowed_ids,
):
    # Set up access, populate the cache.
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    allowed_before = await query_allowed_ids(test_user_a)
    assert test_kb_x in allowed_before

    # Revoke -> the fixture publishes on rbac:invalidate.
    await revoke_user_from_group(test_user_a, test_group_x)
    # Allow the in-process subscriber to drop the key.
    await asyncio.sleep(0.2)

    allowed_after = await query_allowed_ids(test_user_a)
    assert test_kb_x not in allowed_after, (
        "post-revocation query must not see kb_x; got " f"{allowed_after}"
    )


@pytest.mark.asyncio
async def test_3_ttl_expiry_as_safety_net(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_kb_x,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    revoke_user_from_group,
    query_allowed_ids,
    monkeypatch_ttl_to_1sec,
):
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    # Warm the (now 1-second TTL) cache.
    await query_allowed_ids(test_user_a)
    # Revoke WITHOUT a pub/sub event -- simulates a dropped message.
    await revoke_user_from_group(test_user_a, test_group_x, skip_pubsub=True)
    # After TTL, the safety net forces a DB re-fetch.
    await asyncio.sleep(1.4)
    allowed = await query_allowed_ids(test_user_a)
    assert test_kb_x not in allowed, (
        "TTL expiry must force a re-fetch even when pub/sub dropped; "
        f"got {allowed}"
    )


@pytest.mark.asyncio
async def test_4_pubsub_dropped_message_ttl_fallback(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_kb_x,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    revoke_user_from_group,
    query_allowed_ids,
    monkeypatch_ttl_to_1sec,
    disable_pubsub_subscribe,
):
    # Distinct from test 3 by intent: this scenario is "the publisher
    # raised the message but the subscriber on this replica never saw
    # it" -- a Redis restart between PUBLISH and SUBSCRIBE delivery.
    # Concretely we still bypass the cache eviction path so the entry
    # depends entirely on the TTL safety net.
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    await query_allowed_ids(test_user_a)
    await revoke_user_from_group(test_user_a, test_group_x, skip_pubsub=True)
    await asyncio.sleep(1.4)  # past the 1-second TTL
    allowed = await query_allowed_ids(test_user_a)
    assert test_kb_x not in allowed, (
        "TTL must catch even when pub/sub subscriber is dropped; "
        f"got {allowed}"
    )


@pytest.mark.asyncio
async def test_5_concurrent_queries_during_revocation_are_consistent(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_kb_x,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    revoke_user_from_group,
    query_allowed_ids,
):
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    await query_allowed_ids(test_user_a)  # warm

    async def q():
        return await query_allowed_ids(test_user_a)

    # Kick off the revocation alongside 10 concurrent reads.
    task = asyncio.create_task(
        revoke_user_from_group(test_user_a, test_group_x)
    )
    results = await asyncio.gather(*(q() for _ in range(10)))
    await task

    # Contract: each individual query must return a *coherent* snapshot --
    # either the pre-revoke set OR the post-revoke set, never a fragment
    # that's "half pre / half post". Concretely the only two valid result
    # values are ``{test_kb_x}`` (pre-revoke) and ``set()`` (post-revoke);
    # any other value would prove a torn read.
    for r in results:
        assert isinstance(r, set), f"expected set, got {type(r)}"
    valid_snapshots = ({test_kb_x}, set())
    bad = [r for r in results if r not in valid_snapshots]
    assert not bad, (
        f"queries returned torn / partial-union results: {bad}; "
        f"valid snapshots are {valid_snapshots}"
    )


@pytest.mark.asyncio
async def test_6_user_a_and_user_b_caches_do_not_collide(
    rbac_db_session,
    redis_client,
    test_user_a,
    test_user_b,
    test_kb_x,
    test_kb_y,
    test_group_x,
    assign_user_to_group,
    grant_kb_to_group,
    grant_kb_to_user,
    query_allowed_ids,
):
    await assign_user_to_group(test_user_a, test_group_x)
    await grant_kb_to_group(test_kb_x, test_group_x)
    await grant_kb_to_user(test_kb_y, test_user_b)

    allowed_a = await query_allowed_ids(test_user_a)
    allowed_b = await query_allowed_ids(test_user_b)

    assert test_kb_x in allowed_a, f"A should see kb_x; got {allowed_a}"
    assert test_kb_y not in allowed_a, (
        f"A must NOT see B's kb_y (CLAUDE.md §2 leak); got {allowed_a}"
    )
    assert test_kb_x not in allowed_b, (
        f"B must NOT see A's kb_x (CLAUDE.md §2 leak); got {allowed_b}"
    )
    assert test_kb_y in allowed_b, f"B should see kb_y; got {allowed_b}"
