"""B1 — pin core retrieval flag passthrough in compose/docker-compose.yml.

Several retrieval flags (RAG_HYBRID, RAG_COLBERT, RAG_RERANK, RAG_MMR,
RAG_CONTEXT_EXPAND) are read at module-import time by ext.services.* and
must be propagated from the host environment into open-webui (and the
ingest-relevant ones into celery-worker). Without explicit passthrough
the chat process inherits Python module defaults that have drifted from
production reality.

Concrete impact prevented by this test:
    Live kb_1_v4 has the ``colbert`` named slot. RAG_COLBERT default in
    code is OFF. New ingests via celery-worker therefore would skip
    colbert vectors, producing mixed-schema points and breaking
    late-interaction reranking on freshly-uploaded documents.

This test reads the YAML directly (no docker subprocess) so it stays
fast and runnable in CI without docker.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "compose" / "docker-compose.yml"
ENV_EXAMPLE = ROOT / "compose" / ".env.example"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def _service_env(svc_name: str) -> dict[str, str]:
    """Return the environment block for a service as a dict."""
    services = _load_compose()["services"]
    assert svc_name in services, f"service {svc_name!r} missing from compose"
    env = services[svc_name].get("environment") or {}
    # docker-compose accepts both list (KEY=VALUE) and dict forms.
    if isinstance(env, list):
        out: dict[str, str] = {}
        for item in env:
            if "=" in item:
                k, v = item.split("=", 1)
                out[k] = v
            else:
                out[item] = ""
        return out
    return {str(k): str(v) for k, v in env.items()}


# --- open-webui: full retrieval flag set ---------------------------------

CHAT_REQUIRED_FLAGS = {
    # Each value is the *expected default substring*. We compare against
    # the right-hand side of the env entry (which is the ${VAR:-default}
    # form). This catches both: (a) flag missing entirely, (b) flag
    # present but with the wrong default.
    "RAG_HYBRID": ":-1",
    "RAG_COLBERT": ":-1",
    "RAG_RERANK": ":-1",
    "RAG_MMR": ":-0",
    "RAG_CONTEXT_EXPAND": ":-0",
}


def test_open_webui_passes_retrieval_flags() -> None:
    env = _service_env("open-webui")
    missing = [k for k in CHAT_REQUIRED_FLAGS if k not in env]
    assert not missing, (
        f"open-webui environment is missing retrieval flag(s): {missing}. "
        "Without these, the chat process inherits Python module defaults "
        "(which have drifted from production)."
    )
    for key, expected_default in CHAT_REQUIRED_FLAGS.items():
        val = env[key]
        assert expected_default in val, (
            f"open-webui {key} default mismatch: got {val!r}, "
            f"expected substring {expected_default!r}"
        )


# --- celery-worker: ingest-relevant subset --------------------------------

WORKER_REQUIRED_FLAGS = {
    # Worker must agree with chat on hybrid/colbert/rerank so new ingests
    # write the same vector slots that retrieval expects.
    "RAG_HYBRID": ":-1",
    "RAG_COLBERT": ":-1",
    "RAG_RERANK": ":-1",
}


def test_celery_worker_passes_ingest_relevant_flags() -> None:
    env = _service_env("celery-worker")
    missing = [k for k in WORKER_REQUIRED_FLAGS if k not in env]
    assert not missing, (
        f"celery-worker environment is missing flag(s): {missing}. "
        "Without these, the async ingest path may write vectors with a "
        "different schema than the synchronous one (especially RAG_COLBERT)."
    )
    for key, expected_default in WORKER_REQUIRED_FLAGS.items():
        val = env[key]
        assert expected_default in val, (
            f"celery-worker {key} default mismatch: got {val!r}, "
            f"expected substring {expected_default!r}"
        )


# --- .env.example documents the new flags ---------------------------------

def test_env_example_documents_retrieval_flags() -> None:
    content = ENV_EXAMPLE.read_text()
    for key in (
        "RAG_HYBRID=",
        "RAG_COLBERT=",
        "RAG_RERANK=",
        "RAG_MMR=",
        "RAG_CONTEXT_EXPAND=",
    ):
        assert key in content, (
            f".env.example is missing {key!r}; operators have nothing to "
            "override and discoverability of the flag drops to zero."
        )
