import pytest
from model_manager.tracker import IdleTracker


def test_register_and_snapshot():
    t = IdleTracker(now_fn=lambda: 1000.0)
    t.register("vision", initial_state="asleep")
    t.register("whisper", initial_state="awake")
    snap = t.snapshot()
    assert set(snap) == {"vision", "whisper"}
    assert snap["vision"]["state"] == "asleep"
    assert snap["whisper"]["state"] == "awake"


def test_mark_awake_updates_timestamp():
    clock = [1000.0]
    t = IdleTracker(now_fn=lambda: clock[0])
    t.register("vision", initial_state="asleep")
    clock[0] = 2000.0
    t.mark_awake("vision")
    snap = t.snapshot()
    assert snap["vision"]["state"] == "awake"
    assert snap["vision"]["last_active"] == 2000.0


def test_touch_only_updates_timestamp_when_awake():
    clock = [1000.0]
    t = IdleTracker(now_fn=lambda: clock[0])
    t.register("vision", initial_state="asleep")
    clock[0] = 1500.0
    t.touch("vision")
    assert t.snapshot()["vision"]["state"] == "asleep"
    t.mark_awake("vision")
    clock[0] = 2000.0
    t.touch("vision")
    assert t.snapshot()["vision"]["last_active"] == 2000.0


def test_idle_list_returns_expired_awake_models():
    clock = [1000.0]
    t = IdleTracker(now_fn=lambda: clock[0])
    t.register("vision",  initial_state="awake")
    t.register("whisper", initial_state="awake")
    t.mark_awake("vision")
    clock[0] = 1010.0
    t.mark_awake("whisper")
    clock[0] = 1305.0
    assert t.idle_models(idle_secs=300) == ["vision"]


def test_idle_list_ignores_asleep_models():
    clock = [1000.0]
    t = IdleTracker(now_fn=lambda: clock[0])
    t.register("vision", initial_state="asleep")
    clock[0] = 5000.0
    assert t.idle_models(idle_secs=300) == []


def test_unknown_name_raises():
    t = IdleTracker()
    with pytest.raises(KeyError):
        t.mark_awake("missing")
