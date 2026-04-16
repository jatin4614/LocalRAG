import pytest
from model_manager.tracker import IdleTracker
from model_manager.sleeper import sleep_idle_models


class StubClient:
    async def sleep_model(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_sleeps_only_idle_models():
    clock = [1000.0]
    tracker = IdleTracker(now_fn=lambda: clock[0])
    tracker.register("vision",  initial_state="awake")
    tracker.register("whisper", initial_state="awake")

    clients = {"vision": StubClient(), "whisper": StubClient()}
    clock[0] = 1400.0

    sleeps = await sleep_idle_models(tracker, clients, idle_secs=300)

    assert sorted(sleeps) == ["vision", "whisper"]
    snap = tracker.snapshot()
    assert snap["vision"]["state"] == "asleep"
    assert snap["whisper"]["state"] == "asleep"


@pytest.mark.asyncio
async def test_no_op_if_nothing_idle():
    clock = [1000.0]
    tracker = IdleTracker(now_fn=lambda: clock[0])
    tracker.register("vision", initial_state="awake")
    clock[0] = 1100.0
    sleeps = await sleep_idle_models(tracker, {"vision": StubClient()}, idle_secs=300)
    assert sleeps == []
    assert tracker.snapshot()["vision"]["state"] == "awake"


@pytest.mark.asyncio
async def test_sleep_error_leaves_tracker_state_awake():
    from model_manager.client import ModelClientError

    class FailingStub:
        async def sleep_model(self) -> bool:
            raise ModelClientError("boom")

    clock = [1000.0]
    tracker = IdleTracker(now_fn=lambda: clock[0])
    tracker.register("vision", initial_state="awake")
    clock[0] = 1500.0
    sleeps = await sleep_idle_models(tracker, {"vision": FailingStub()}, idle_secs=300)
    assert sleeps == []
    assert tracker.snapshot()["vision"]["state"] == "awake"
