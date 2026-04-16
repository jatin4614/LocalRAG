"""Thin wrapper over faster-whisper with sleep/wake endpoints matching vLLM's contract."""
from __future__ import annotations

import asyncio
import gc
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("whisper_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

MODEL_NAME  = os.environ.get("WHISPER_MODEL", "medium")
MODEL_CACHE = os.environ.get("WHISPER_CACHE", "/models")

_model = None          # faster_whisper.WhisperModel | None
_model_lock = asyncio.Lock()
_state: str = "asleep"  # "awake" | "asleep"


async def _load_to_gpu() -> None:
    global _model, _state
    from faster_whisper import WhisperModel
    logger.info("loading whisper %s to cuda", MODEL_NAME)
    _model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16",
                          download_root=MODEL_CACHE)
    _state = "awake"


async def _unload_from_gpu() -> None:
    global _model, _state
    logger.info("unloading whisper from cuda")
    _model = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    _state = "asleep"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start asleep; wake on demand via /wake_up.
    yield
    async with _model_lock:
        await _unload_from_gpu()


app = FastAPI(title="whisper_service", lifespan=lifespan)


class StateResponse(BaseModel):
    state: str
    model: str


@app.get("/health", response_model=StateResponse)
async def health() -> StateResponse:
    return StateResponse(state=_state, model=MODEL_NAME)


@app.post("/wake_up", response_model=StateResponse)
async def wake_up() -> StateResponse:
    async with _model_lock:
        if _state == "asleep":
            await _load_to_gpu()
    return StateResponse(state=_state, model=MODEL_NAME)


@app.post("/sleep", response_model=StateResponse)
async def sleep() -> StateResponse:
    async with _model_lock:
        if _state == "awake":
            await _unload_from_gpu()
    return StateResponse(state=_state, model=MODEL_NAME)


class TranscriptionResponse(BaseModel):
    text: str


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    async with _model_lock:
        if _state == "asleep":
            await _load_to_gpu()
        current_model = _model
    if current_model is None:
        raise HTTPException(status_code=503, detail="model failed to load")

    content = await file.read()
    tmp_path = f"/tmp/{file.filename or 'upload.wav'}"
    with open(tmp_path, "wb") as fh:
        fh.write(content)

    segments, _info = current_model.transcribe(tmp_path, beam_size=5)
    text = "".join(seg.text for seg in segments).strip()
    os.remove(tmp_path)
    return TranscriptionResponse(text=text)
