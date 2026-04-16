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
