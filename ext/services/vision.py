"""Vision preprocessor — turns user-attached images into text context.

Flow (called from the upstream middleware, before RAG):
  1. Scan the current user message for image_url content parts.
  2. For each image, call vllm-vision for a detailed description + OCR.
  3. Persist the description to chat_{chat_id} Qdrant so later messages in
     the same chat can cite it like any other private doc.
  4. Replace the image_url part with a text part so the downstream text-only
     chat model (Qwen2.5-14B) can reason over it.

Images are content-hashed so re-submitting the same image (e.g. regenerate
response) reuses the cached description instead of re-running vision.
"""
from __future__ import annotations

import base64
import contextvars
import hashlib
import logging
import os
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger("kairos.vision")

_VISION_URL = os.environ.get("VISION_API_BASE_URL", "http://vllm-vision:8000/v1")
_VISION_MODEL = os.environ.get("VISION_MODEL_SERVED_NAME", "orgchat-vision")
_VISION_PROMPT = (
    "Describe this image in detail for use as retrieval context. "
    "Include any visible text verbatim, tables, charts, logos, and key visual elements. "
    "Be concise but thorough (200 words max)."
)
_VISION_MAX_TOKENS = 400
_VISION_TIMEOUT_S = 60

# content_hash -> description  (per-process cache; small and bounded)
_cache: dict[str, str] = {}
_CACHE_MAX = 512

# Injected by build_ext_routers at startup
_vector_store = None
_embedder = None
_sessionmaker = None


def configure(*, vector_store, embedder, sessionmaker) -> None:
    global _vector_store, _embedder, _sessionmaker
    _vector_store = vector_store
    _embedder = embedder
    _sessionmaker = sessionmaker


def _hash_image_url(url: str) -> str:
    # For data: URIs, hash the decoded bytes; for http URLs, hash the URL itself.
    if url.startswith("data:"):
        try:
            _, b64 = url.split(",", 1)
            return hashlib.sha256(base64.b64decode(b64, validate=False)).hexdigest()[:24]
        except Exception:
            return hashlib.sha256(url.encode()).hexdigest()[:24]
    return hashlib.sha256(url.encode()).hexdigest()[:24]


async def _describe_image(url: str) -> Optional[str]:
    h = _hash_image_url(url)
    if h in _cache:
        return _cache[h]

    payload = {
        "model": _VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
        "max_tokens": _VISION_MAX_TOKENS,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=_VISION_TIMEOUT_S) as client:
            r = await client.post(f"{_VISION_URL}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            desc = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            desc = (desc or "").strip()
            if not desc:
                return None
            if len(_cache) >= _CACHE_MAX:
                # Evict oldest — dicts preserve insertion order in py3.7+
                _cache.pop(next(iter(_cache)))
            _cache[h] = desc
            return desc
    except Exception as e:
        logger.warning("vision describe failed for image %s: %s", h, e)
        return None


async def _persist_description(
    *, description: str, chat_id: str, user_id: str, image_hash: str
) -> None:
    """Chunk + embed + upsert the description into chat_{chat_id}."""
    if _vector_store is None or _embedder is None:
        return
    try:
        from .ingest import ingest_bytes
        filename = f"attached-image-{image_hash}.txt"
        await _vector_store.ensure_collection(f"chat_{chat_id}")
        await ingest_bytes(
            data=description.encode("utf-8"),
            mime_type="text/plain",
            filename=filename,
            collection=f"chat_{chat_id}",
            payload_base={
                "chat_id": chat_id,
                "owner_user_id": user_id,
                "filename": filename,
                "image_hash": image_hash,
            },
            vector_store=_vector_store,
            embedder=_embedder,
        )
        logger.info(
            "vision: persisted image description chat=%s hash=%s len=%d",
            chat_id, image_hash, len(description),
        )
    except Exception as e:
        logger.exception("vision persist failed: %s", e)


def _extract_image_parts(content: Any) -> tuple[list[dict], list[dict], str]:
    """Return (kept_parts, image_parts, accumulated_text). Handles both str and list content."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}], [], content
    if not isinstance(content, list):
        return [], [], ""

    kept: list[dict] = []
    images: list[dict] = []
    text_accum = ""
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "image_url":
            images.append(part)
        else:
            kept.append(part)
            if part.get("type") == "text":
                text_accum += part.get("text", "")
    return kept, images, text_accum


async def preprocess_images(messages: list[dict], *, chat_id: Optional[str], user_id: str) -> int:
    """Mutates `messages` in place: replaces image_url parts with text descriptions.

    Returns the number of images processed.
    """
    processed = 0
    if not messages:
        return 0

    # Only the most recent user message typically has attachments;
    # but we scan all to be safe (regenerate, edits, etc.)
    for msg in messages:
        if msg.get("role") != "user":
            continue
        kept, images, _ = _extract_image_parts(msg.get("content"))
        if not images:
            continue

        descriptions: list[tuple[str, str]] = []  # (hash, description)
        for img_part in images:
            url = (img_part.get("image_url") or {}).get("url", "")
            if not url:
                continue
            desc = await _describe_image(url)
            if desc:
                descriptions.append((_hash_image_url(url), desc))
            else:
                descriptions.append((_hash_image_url(url), "[attached image could not be processed]"))
            processed += 1

        if not descriptions:
            continue

        # Persist each description to chat_{chat_id} (best-effort, non-blocking would be nice
        # but we want the chunk upserted before RAG retrieval runs).
        if chat_id:
            for h, desc in descriptions:
                if not desc.startswith("[attached image"):
                    await _persist_description(
                        description=desc, chat_id=chat_id, user_id=user_id, image_hash=h,
                    )

        # Build a combined text part describing all images, appended to any existing text.
        combined = "\n\n".join(
            f"[Attached image {i+1}]: {d}" for i, (_, d) in enumerate(descriptions)
        )
        text_part = next((p for p in kept if p.get("type") == "text"), None)
        if text_part is not None:
            text_part["text"] = f"{text_part.get('text','')}\n\n{combined}".strip()
        else:
            kept.append({"type": "text", "text": combined})

        msg["content"] = kept

    return processed
