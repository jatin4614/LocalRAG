"""Background idle-sleep loop."""
from __future__ import annotations

import asyncio
import logging
from typing import Mapping, Protocol

from .client import ModelClientError
from .tracker import IdleTracker


logger = logging.getLogger("model_manager.sleeper")


class _SleepCapable(Protocol):
    async def sleep_model(self) -> bool: ...


async def sleep_idle_models(
    tracker: IdleTracker,
    clients: Mapping[str, _SleepCapable],
    *,
    idle_secs: float,
) -> list[str]:
    """Sleep every awake model idle for >= idle_secs.

    Returns names actually slept. ModelClientError is logged and the tracker
    state is kept 'awake' so the next cycle retries.
    """
    idle = tracker.idle_models(idle_secs=idle_secs)
    slept: list[str] = []
    for name in idle:
        client = clients.get(name)
        if client is None:
            logger.warning("no client for %s; skipping", name)
            continue
        try:
            await client.sleep_model()
        except ModelClientError as e:
            logger.warning("failed to sleep %s: %s", name, e)
            continue
        tracker.mark_asleep(name)
        slept.append(name)
        logger.info("slept %s", name)
    return slept


async def sleeper_loop(
    tracker: IdleTracker,
    clients: Mapping[str, _SleepCapable],
    *,
    idle_secs: float,
    poll_secs: float,
    stop: asyncio.Event,
) -> None:
    """Forever loop until stop is set. Never raises."""
    while not stop.is_set():
        try:
            await sleep_idle_models(tracker, clients, idle_secs=idle_secs)
        except Exception:  # noqa: BLE001
            logger.exception("sleeper loop iteration failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_secs)
        except asyncio.TimeoutError:
            pass
