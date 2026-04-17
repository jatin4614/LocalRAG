# Phase 3 — Model Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute task-by-task.

**Goal:** Ship the `model-manager` sidecar — a tiny always-running FastAPI daemon that watches idle times on the two sleepable GPU services (vllm-vision + whisper), wakes them on demand via HTTP, and puts them back to sleep after `MODEL_UNLOAD_IDLE_SECS` idle. Exposes `/models/status` + `/models/{name}/wake|sleep|touch` to the rest of the stack.

**Architecture:** Separate container, no Docker-socket access. Talks to `vllm-vision:8000` and `whisper:9000` over HTTP. State kept in-process (single replica; a restart is harmless — state rebuilds from `GET /health` on each backend). Background asyncio task polls idle timestamps every `POLL_SECS` (default 15 s) and calls `POST /sleep` on any awake model whose last_active is older than `IDLE_SECS`.

**Tech Stack:** Python ≥ 3.10, FastAPI, httpx, pytest-asyncio, asyncio, pydantic-settings.

**Working directory:** `/home/vogic/LocalRAG/` (main, tagged `phase-2-kb-management`).

---

## Decisions (Phase 3 scope)

| # | Decision | Revise-by |
|---|----------|-----------|
| D20 | Model manager is an independent Python package under `model_manager/` — does NOT import from `ext/`. Shares only the dev venv for tests. | Phase 5 |
| D21 | Two sleepable models: `vision` (→ `vllm-vision`) and `whisper`. Chat stays always-on and is NOT managed. | — |
| D22 | State is in-process (no Redis/DB). A restart rebuilds state by calling `GET /health` on each backend. For a single sidecar replica this is simpler than shared state. | If we scale to multiple model-manager replicas |
| D23 | If a backend's `/sleep` or `/wake_up` returns 4xx, treat as fatal (bug) and 502 back to caller. If it returns 5xx or times out, treat as transient and surface 503. | — |
| D24 | `POLL_SECS=15`, `IDLE_SECS=300` by default. Overridable via env. | Phase 6 perf tuning |
| D25 | The compose `model-manager` service depends on `vllm-vision` and `whisper` but does NOT `depends_on: service_healthy` on them — the manager comes up first and retries. | — |

---

## File structure

```
model_manager/
├── Dockerfile               T7
├── requirements.txt         T7
├── __init__.py              T1
├── config.py                T1
├── client.py                T2
├── tracker.py               T3
├── sleeper.py               T4
└── app.py                   T5

tests/
├── unit/
│   ├── test_mm_config.py    T1
│   ├── test_mm_tracker.py   T3
│   └── test_mm_client.py    T2
└── integration/
    └── test_mm_app.py       T8 (spins up stub backends via ASGITransport)

compose/docker-compose.yml   T7 (append model-manager service)
```

---

## Task 1: Settings for model-manager

**Files:** `model_manager/__init__.py` (empty), `model_manager/config.py`, `tests/unit/test_mm_config.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_mm_config.py`:

```python
import pytest
from model_manager.config import ModelManagerSettings


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("VISION_URL", "http://v:8000")
    monkeypatch.setenv("WHISPER_URL", "http://w:9000")
    s = ModelManagerSettings()
    assert s.vision_url == "http://v:8000"
    assert s.whisper_url == "http://w:9000"
    assert s.idle_secs == 300
    assert s.poll_secs == 15
    assert s.host == "0.0.0.0"
    assert s.port == 8080


def test_settings_override_idle(monkeypatch):
    monkeypatch.setenv("VISION_URL", "http://v:8000")
    monkeypatch.setenv("WHISPER_URL", "http://w:9000")
    monkeypatch.setenv("MODEL_UNLOAD_IDLE_SECS", "60")
    monkeypatch.setenv("MODEL_MANAGER_POLL_SECS", "5")
    s = ModelManagerSettings()
    assert s.idle_secs == 60
    assert s.poll_secs == 5


def test_settings_require_backend_urls(monkeypatch):
    monkeypatch.delenv("VISION_URL", raising=False)
    monkeypatch.delenv("WHISPER_URL", raising=False)
    with pytest.raises(Exception):
        ModelManagerSettings()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_config.py -v
```

- [ ] **Step 3: Write files**

Create `/home/vogic/LocalRAG/model_manager/__init__.py` (empty).

Create `/home/vogic/LocalRAG/model_manager/config.py`:

```python
"""Model manager settings."""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelManagerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, case_sensitive=False, extra="ignore")

    vision_url:  str = Field(..., alias="VISION_URL")
    whisper_url: str = Field(..., alias="WHISPER_URL")
    idle_secs:   int = Field(300, alias="MODEL_UNLOAD_IDLE_SECS")
    poll_secs:   int = Field(15,  alias="MODEL_MANAGER_POLL_SECS")
    host:        str = Field("0.0.0.0", alias="MODEL_MANAGER_HOST")
    port:        int = Field(8080,      alias="MODEL_MANAGER_PORT")
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_config.py -v && ruff check . && mypy .
```

Expected: 3 PASSED, lint clean.

- [ ] **Step 5: Commit**

```bash
git add model_manager/__init__.py model_manager/config.py tests/unit/test_mm_config.py
git commit -m "feat(mm): model-manager settings"
```

---

## Task 2: ModelClient — HTTP wrapper for backends

**Files:** `model_manager/client.py`, `tests/unit/test_mm_client.py`.

Abstraction: same sleep/wake contract for vision and whisper, but health endpoints differ (vllm: `GET /v1/models`, whisper: `GET /health`). The client takes a configurable `health_path` per model. Returns a normalized dict `{"state": "awake"|"asleep"|"unknown"}` for health. For `wake_up`/`sleep`, returns a bool (success) — raises `ModelClientError` on network failure.

- [ ] **Step 1: Write failing test**

`tests/unit/test_mm_client.py`:

```python
import pytest
import httpx
from model_manager.client import ModelClient, ModelClientError


@pytest.mark.asyncio
async def test_wake_success_returns_true():
    # Mock transport that returns 200 OK.
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/wake_up"
        return httpx.Response(200, json={"state": "awake"})

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.wake_up() is True
    await client.aclose()


@pytest.mark.asyncio
async def test_sleep_success_returns_true():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sleep"
        return httpx.Response(200, json={"state": "asleep"})

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.sleep_model() is True
    await client.aclose()


@pytest.mark.asyncio
async def test_health_awake_when_reachable():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.health() == "awake"
    await client.aclose()


@pytest.mark.asyncio
async def test_health_unknown_on_connection_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nope")

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    assert await client.health() == "unknown"
    await client.aclose()


@pytest.mark.asyncio
async def test_wake_raises_on_5xx():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    client = ModelClient(base_url="http://fake", health_path="/v1/models", transport=transport)
    with pytest.raises(ModelClientError):
        await client.wake_up()
    await client.aclose()
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_client.py -v
```

- [ ] **Step 3: Write `model_manager/client.py`**

```python
"""HTTP client for a single backend model server (vllm or whisper wrapper)."""
from __future__ import annotations

from typing import Literal, Optional

import httpx


class ModelClientError(RuntimeError):
    """Raised on 4xx/5xx or transport failure when the caller expected success."""


HealthState = Literal["awake", "asleep", "unknown"]


class ModelClient:
    def __init__(
        self, *, base_url: str, health_path: str,
        timeout: float = 10.0, transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._base_url = base_url
        self._health_path = health_path
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> HealthState:
        """Light probe: 2xx → awake, connection error → unknown.

        We cannot distinguish "asleep but container up" vs "awake" from the
        health endpoint alone — vLLM's /v1/models returns 200 in both states.
        The idle tracker is the source of truth for sleep state; health is
        only used to bootstrap state after manager restart.
        """
        try:
            r = await self._client.get(self._health_path)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
            return "unknown"
        if r.status_code < 300:
            return "awake"
        return "unknown"

    async def wake_up(self) -> bool:
        try:
            r = await self._client.post("/wake_up")
        except httpx.HTTPError as e:
            raise ModelClientError(f"wake_up transport failure: {e}") from e
        if r.status_code >= 500:
            raise ModelClientError(f"wake_up returned {r.status_code}")
        if r.status_code >= 400:
            raise ModelClientError(f"wake_up 4xx: {r.status_code} — backend rejected request")
        return True

    async def sleep_model(self) -> bool:
        try:
            r = await self._client.post("/sleep")
        except httpx.HTTPError as e:
            raise ModelClientError(f"sleep transport failure: {e}") from e
        if r.status_code >= 500:
            raise ModelClientError(f"sleep returned {r.status_code}")
        if r.status_code >= 400:
            raise ModelClientError(f"sleep 4xx: {r.status_code}")
        return True
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_client.py -v && ruff check . && mypy .
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add model_manager/client.py tests/unit/test_mm_client.py
git commit -m "feat(mm): async ModelClient (wake_up/sleep/health)"
```

---

## Task 3: IdleTracker — per-model state + timestamps

**Files:** `model_manager/tracker.py`, `tests/unit/test_mm_tracker.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_mm_tracker.py`:

```python
import time
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
    t.touch("vision")   # should NOT change state
    assert t.snapshot()["vision"]["state"] == "asleep"
    # Now wake, then touch.
    t.mark_awake("vision")
    clock[0] = 2000.0
    t.touch("vision")
    assert t.snapshot()["vision"]["last_active"] == 2000.0


def test_idle_list_returns_expired_awake_models():
    clock = [1000.0]
    t = IdleTracker(now_fn=lambda: clock[0])
    t.register("vision",  initial_state="awake")
    t.register("whisper", initial_state="awake")
    t.mark_awake("vision");  # last_active=1000
    clock[0] = 1010.0
    t.mark_awake("whisper")  # last_active=1010
    clock[0] = 1305.0        # vision idle 305s, whisper idle 295s
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
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_tracker.py -v
```

- [ ] **Step 3: Write `model_manager/tracker.py`**

```python
"""In-process state tracker for model sleep/wake + idle timestamps."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Literal


State = Literal["awake", "asleep", "unknown"]


class IdleTracker:
    def __init__(self, now_fn: Callable[[], float] = time.monotonic) -> None:
        self._now = now_fn
        self._state: Dict[str, State] = {}
        self._last_active: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register(self, name: str, *, initial_state: State = "asleep") -> None:
        with self._lock:
            self._state[name] = initial_state
            self._last_active[name] = self._now()

    def mark_awake(self, name: str) -> None:
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            self._state[name] = "awake"
            self._last_active[name] = self._now()

    def mark_asleep(self, name: str) -> None:
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            self._state[name] = "asleep"

    def touch(self, name: str) -> None:
        """Bump the last_active timestamp ONLY if awake; no-op if asleep."""
        with self._lock:
            if name not in self._state:
                raise KeyError(name)
            if self._state[name] == "awake":
                self._last_active[name] = self._now()

    def snapshot(self) -> Dict[str, Dict[str, float | str]]:
        with self._lock:
            return {
                n: {"state": self._state[n], "last_active": self._last_active[n]}
                for n in self._state
            }

    def idle_models(self, *, idle_secs: float) -> List[str]:
        now = self._now()
        with self._lock:
            return [
                n for n, s in self._state.items()
                if s == "awake" and (now - self._last_active[n]) >= idle_secs
            ]
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_tracker.py -v && ruff check . && mypy .
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add model_manager/tracker.py tests/unit/test_mm_tracker.py
git commit -m "feat(mm): IdleTracker (per-model state + timestamps)"
```

---

## Task 4: Background sleeper task

**Files:** `model_manager/sleeper.py`, `tests/unit/test_mm_sleeper.py`.

- [ ] **Step 1: Write failing test**

`tests/unit/test_mm_sleeper.py`:

```python
import asyncio
import pytest

from model_manager.tracker import IdleTracker
from model_manager.sleeper import sleep_idle_models


class StubClient:
    def __init__(self):
        self.sleep_calls: list[str] = []

    async def sleep_model(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_sleeps_only_idle_models():
    clock = [1000.0]
    tracker = IdleTracker(now_fn=lambda: clock[0])
    tracker.register("vision",  initial_state="awake")
    tracker.register("whisper", initial_state="awake")

    vision_stub  = StubClient()
    whisper_stub = StubClient()
    clients = {"vision": vision_stub, "whisper": whisper_stub}

    clock[0] = 1400.0  # both 400s old → both idle at 300s threshold

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
    vision_stub = StubClient()
    # 100s elapsed, threshold 300s → not idle.
    clock[0] = 1100.0
    sleeps = await sleep_idle_models(tracker, {"vision": vision_stub}, idle_secs=300)
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
    clock[0] = 1500.0  # idle
    sleeps = await sleep_idle_models(tracker, {"vision": FailingStub()}, idle_secs=300)
    assert sleeps == []  # nothing actually slept
    assert tracker.snapshot()["vision"]["state"] == "awake"  # keep truth
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_sleeper.py -v
```

- [ ] **Step 3: Write `model_manager/sleeper.py`**

```python
"""Background idle-sleep loop."""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Mapping, Protocol

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
    """Sleep every awake model that's been idle for at least `idle_secs`.

    Returns the list of models actually slept. Failures are logged and kept
    in awake state so the next cycle retries.
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
    """Forever loop: every `poll_secs`, sleep any expired awake models."""
    while not stop.is_set():
        try:
            await sleep_idle_models(tracker, clients, idle_secs=idle_secs)
        except Exception:  # noqa: BLE001 — loop must not die
            logger.exception("sleeper loop iteration failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=poll_secs)
        except asyncio.TimeoutError:
            pass
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_sleeper.py -v && ruff check . && mypy .
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add model_manager/sleeper.py tests/unit/test_mm_sleeper.py
git commit -m "feat(mm): background sleeper (sleep_idle_models + sleeper_loop)"
```

---

## Task 5: FastAPI app

**Files:** `model_manager/app.py`, `tests/integration/test_mm_app.py` (wake/sleep/touch via ASGI transport with stubbed backends).

- [ ] **Step 1: Write failing test**

`tests/integration/test_mm_app.py`:

```python
"""Integration test for model-manager with stubbed vllm-vision + whisper."""
import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport


def _stub_backend(state: dict) -> FastAPI:
    """Simulates vllm-vision OR whisper: tracks 'awake'/'asleep' state in the dict."""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"state": state["state"]}

    @app.get("/v1/models")
    async def models():  # vllm-style
        return {"data": [{"id": "stub"}]}

    @app.post("/wake_up")
    async def wake():
        state["state"] = "awake"
        return {"state": "awake"}

    @app.post("/sleep")
    async def sleep():
        state["state"] = "asleep"
        return {"state": "asleep"}

    return app


@pytest_asyncio.fixture
async def vision_state():
    return {"state": "asleep"}


@pytest_asyncio.fixture
async def whisper_state():
    return {"state": "asleep"}


@pytest_asyncio.fixture
async def mm_client(vision_state, whisper_state, monkeypatch):
    # Build stub backends and wire the ModelClients to talk to them over ASGITransport.
    vision_app  = _stub_backend(vision_state)
    whisper_app = _stub_backend(whisper_state)

    from model_manager.client import ModelClient
    from model_manager.tracker import IdleTracker
    from model_manager.app import build_app

    vision_client  = ModelClient(
        base_url="http://stub-vision", health_path="/v1/models",
        transport=httpx.ASGITransport(app=vision_app),
    )
    whisper_client = ModelClient(
        base_url="http://stub-whisper", health_path="/health",
        transport=httpx.ASGITransport(app=whisper_app),
    )

    # Inject both clients into the app. build_app(...) supports clients=... override.
    monkeypatch.setenv("VISION_URL", "http://stub-vision")
    monkeypatch.setenv("WHISPER_URL", "http://stub-whisper")
    app = build_app(clients={"vision": vision_client, "whisper": whisper_client},
                    tracker=IdleTracker(now_fn=lambda: 1000.0),
                    start_sleeper=False)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

    await vision_client.aclose()
    await whisper_client.aclose()


@pytest.mark.asyncio
async def test_status_lists_all_models(mm_client):
    r = await mm_client.get("/models/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"vision", "whisper"}


@pytest.mark.asyncio
async def test_wake_updates_state(mm_client, vision_state):
    r = await mm_client.post("/models/vision/wake")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "awake"
    assert vision_state["state"] == "awake"

    r = await mm_client.get("/models/status")
    assert r.json()["vision"]["state"] == "awake"


@pytest.mark.asyncio
async def test_sleep_updates_state(mm_client, whisper_state):
    await mm_client.post("/models/whisper/wake")
    r = await mm_client.post("/models/whisper/sleep")
    assert r.status_code == 200
    assert r.json()["state"] == "asleep"
    assert whisper_state["state"] == "asleep"


@pytest.mark.asyncio
async def test_touch_does_not_wake(mm_client, vision_state):
    r = await mm_client.post("/models/vision/touch")
    assert r.status_code == 200
    # Still asleep on backend and in tracker.
    assert vision_state["state"] == "asleep"
    r = await mm_client.get("/models/status")
    assert r.json()["vision"]["state"] == "asleep"


@pytest.mark.asyncio
async def test_unknown_model_404(mm_client):
    r = await mm_client.post("/models/chat/wake")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_healthz(mm_client):
    r = await mm_client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_mm_app.py -v
```

- [ ] **Step 3: Write `model_manager/app.py`**

```python
"""Model-manager FastAPI app."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Mapping, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from .client import ModelClient, ModelClientError
from .config import ModelManagerSettings
from .sleeper import sleeper_loop
from .tracker import IdleTracker


logger = logging.getLogger("model_manager.app")


class ModelState(BaseModel):
    state: str
    last_active: float


class WakeResponse(BaseModel):
    state: str


def _default_clients(settings: ModelManagerSettings) -> dict[str, ModelClient]:
    return {
        "vision":  ModelClient(base_url=settings.vision_url,  health_path="/v1/models"),
        "whisper": ModelClient(base_url=settings.whisper_url, health_path="/health"),
    }


def build_app(
    *,
    settings: Optional[ModelManagerSettings] = None,
    clients: Optional[Mapping[str, ModelClient]] = None,
    tracker: Optional[IdleTracker] = None,
    start_sleeper: bool = True,
) -> FastAPI:
    settings = settings or ModelManagerSettings()  # type: ignore[call-arg]
    tracker = tracker or IdleTracker()
    clients = dict(clients) if clients is not None else _default_clients(settings)

    for name in clients:
        tracker.register(name, initial_state="asleep")

    stop = asyncio.Event()
    sleeper_task: asyncio.Task | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal sleeper_task
        if start_sleeper:
            sleeper_task = asyncio.create_task(
                sleeper_loop(tracker, clients,
                             idle_secs=settings.idle_secs,
                             poll_secs=settings.poll_secs,
                             stop=stop)
            )
        yield
        stop.set()
        if sleeper_task is not None:
            try:
                await asyncio.wait_for(sleeper_task, timeout=5.0)
            except asyncio.TimeoutError:
                sleeper_task.cancel()
        for c in clients.values():
            await c.aclose()

    app = FastAPI(title="model-manager", version="0.3.0", lifespan=lifespan)

    def _ensure_known(name: str) -> None:
        if name not in clients:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"unknown model: {name}")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/models/status", response_model=dict[str, ModelState])
    async def status_all():
        return {n: ModelState(**v) for n, v in tracker.snapshot().items()}

    @app.post("/models/{name}/wake", response_model=WakeResponse)
    async def wake(name: str):
        _ensure_known(name)
        try:
            await clients[name].wake_up()
        except ModelClientError as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
        tracker.mark_awake(name)
        return WakeResponse(state="awake")

    @app.post("/models/{name}/sleep", response_model=WakeResponse)
    async def sleep(name: str):
        _ensure_known(name)
        try:
            await clients[name].sleep_model()
        except ModelClientError as e:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
        tracker.mark_asleep(name)
        return WakeResponse(state="asleep")

    @app.post("/models/{name}/touch", response_model=WakeResponse)
    async def touch(name: str):
        _ensure_known(name)
        tracker.touch(name)
        snap = tracker.snapshot()[name]
        return WakeResponse(state=str(snap["state"]))

    return app
```

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/integration/test_mm_app.py -v && ruff check . && mypy .
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add model_manager/app.py tests/integration/test_mm_app.py
git commit -m "feat(mm): FastAPI app (status, wake, sleep, touch, healthz)"
```

---

## Task 6: Dockerfile + requirements.txt

**Files:** `model_manager/Dockerfile`, `model_manager/requirements.txt`.

- [ ] **Step 1: Write a tiny sanity test**

Create `/home/vogic/LocalRAG/tests/unit/test_mm_dockerfile.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_dockerfile_has_healthcheck():
    content = (ROOT / "model_manager/Dockerfile").read_text()
    assert "HEALTHCHECK" in content
    assert "uvicorn" in content

def test_requirements_lists_fastapi_httpx():
    content = (ROOT / "model_manager/requirements.txt").read_text()
    for pkg in ["fastapi", "uvicorn", "httpx", "pydantic-settings"]:
        assert pkg in content, f"missing dep: {pkg}"
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_dockerfile.py -v
```

- [ ] **Step 3: Write files**

Create `/home/vogic/LocalRAG/model_manager/requirements.txt`:

```
fastapi>=0.115
uvicorn[standard]>=0.30
httpx>=0.27
pydantic>=2.8
pydantic-settings>=2.4
```

Create `/home/vogic/LocalRAG/model_manager/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY __init__.py config.py client.py tracker.py sleeper.py app.py ./

ENV MODEL_MANAGER_HOST=0.0.0.0
ENV MODEL_MANAGER_PORT=8080

HEALTHCHECK --interval=20s --timeout=5s --retries=5 --start-period=10s \
  CMD curl -sf http://localhost:${MODEL_MANAGER_PORT}/healthz || exit 1

EXPOSE 8080
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--factory"]
```

**Note on `--factory`:** `app.py` exports `build_app` — but `uvicorn ... --factory` expects a callable that RETURNS the app. Our entrypoint is `build_app`, so the CMD should be:

```dockerfile
CMD ["python", "-m", "uvicorn", "app:build_app", "--host", "0.0.0.0", "--port", "8080", "--factory"]
```

Use the second form (`app:build_app`).

- [ ] **Step 4: Run — PASS**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_dockerfile.py -v
```

- [ ] **Step 5: Commit**

```bash
git add model_manager/Dockerfile model_manager/requirements.txt tests/unit/test_mm_dockerfile.py
git commit -m "feat(mm): Dockerfile + requirements"
```

---

## Task 7: Compose wiring

**Files:** Edit `compose/docker-compose.yml`; extend `compose/.env.example`.

- [ ] **Step 1: Write failing test**

Create `/home/vogic/LocalRAG/tests/unit/test_mm_compose.py`:

```python
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_model_manager_service_in_compose():
    c = (ROOT / "compose/docker-compose.yml").read_text()
    assert "model-manager:" in c
    assert "VISION_URL: http://vllm-vision:8000" in c
    assert "WHISPER_URL: http://whisper:9000" in c
    assert "../model_manager" in c  # build context

def test_env_example_has_model_manager_keys():
    c = (ROOT / "compose/.env.example").read_text()
    assert "MODEL_UNLOAD_IDLE_SECS=" in c
```

- [ ] **Step 2: Run — FAIL**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_mm_compose.py -v
```

- [ ] **Step 3: Append to `compose/docker-compose.yml`**

Under `services:` at the same indentation as the existing services, append:

```yaml
  model-manager:
    build:
      context: ../model_manager
    container_name: orgchat-model-manager
    restart: unless-stopped
    environment:
      VISION_URL: http://vllm-vision:8000
      WHISPER_URL: http://whisper:9000
      MODEL_UNLOAD_IDLE_SECS: ${MODEL_UNLOAD_IDLE_SECS:-300}
      MODEL_MANAGER_POLL_SECS: "15"
    ports:
      - "8080:8080"
    depends_on:
      - vllm-vision
      - whisper
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8080/healthz || exit 1"]
      interval: 20s
      timeout: 5s
      retries: 5
      start_period: 10s
```

- [ ] **Step 4: Verify `MODEL_UNLOAD_IDLE_SECS` is in `.env.example`**

The Phase 1 `.env.example` already has `MODEL_UNLOAD_IDLE_SECS=300` under `# --- Model manager ---`. If missing, append to `compose/.env.example`:

```
MODEL_UNLOAD_IDLE_SECS=300
```

- [ ] **Step 5: Run — PASS**

```bash
source .venv/bin/activate
python -m pytest tests/unit/test_mm_compose.py -v
docker compose -f compose/docker-compose.yml --env-file compose/.env.example config > /dev/null && echo OK
```

Expected: 2 PASSED + OK.

- [ ] **Step 6: Commit**

```bash
git add compose/docker-compose.yml tests/unit/test_mm_compose.py
# .env.example only if it was edited
git add compose/.env.example 2>/dev/null || true
git commit -m "feat(mm): compose wiring (model-manager service)"
```

---

## Task 8: Full regression + Phase 3 tag

- [ ] **Step 1: Run unit + integration**

```bash
source .venv/bin/activate
python -m pytest tests/unit -v 2>&1 | tail -5
SKIP_GPU_SMOKE=1 python -m pytest tests/integration -v 2>&1 | tail -15
ruff check . && mypy .
```

Expected: unit ≥ 60 passed; integration ≥ 44 passed + 1 skipped; lint clean.

- [ ] **Step 2: Tag**

```bash
git tag -a phase-3-model-manager -m "Phase 3 complete: model-manager sidecar with idle-sleep + HTTP wake/sleep/touch API"
```

- [ ] **Step 3: Commission Phase 4 plan**

Request controller: "Write Phase 4 plan at `docs/superpowers/plans/2026-04-16-phase-4-rag-pipeline.md`: document upload → text extraction → chunking → embedding (TEI) → Qdrant upsert, parallel KB retrieval, tiered reranking, token budgeting, private-doc session namespace."

---

## Phase 3 acceptance checklist

- [ ] `model_manager/` is a self-contained Python package (no `ext/` imports).
- [ ] `GET /models/status` reflects tracker state.
- [ ] `POST /models/vision/wake` triggers backend `POST /wake_up` and sets tracker awake.
- [ ] `POST /models/vision/sleep` triggers backend `POST /sleep` and sets tracker asleep.
- [ ] `POST /models/vision/touch` updates timestamp only (does not wake).
- [ ] Background sleeper sleeps idle models; failures don't kill the loop.
- [ ] `docker compose config` validates.
- [ ] `phase-3-model-manager` tag exists.
- [ ] `ruff` + `mypy` clean.
