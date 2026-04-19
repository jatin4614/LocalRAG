import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "compose/docker-compose.yml"),
           "--env-file", str(ROOT / "compose/.env")]

SKIP_GPU = os.environ.get("SKIP_GPU_SMOKE") == "1"


def _wait(pred, timeout_s=300, interval_s=5):
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        try:
            if pred():
                return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False


@pytest.fixture(scope="module", autouse=True)
def compose_stack():
    env_file = ROOT / "compose/.env"
    created_env = False
    if not env_file.exists():
        env_file.write_text((ROOT / "compose/.env.example").read_text())
        created_env = True
    # On non-GPU hosts, bring up only the stateful data-plane services.
    # Caddy is excluded from the base list because it binds host ports 80/443,
    # which may already be allocated on a shared/dev server.  It is implicitly
    # tested on a GPU host via the full-stack path below.
    target_services = ["postgres", "redis", "qdrant"]
    if not SKIP_GPU:
        target_services += ["caddy", "vllm-chat", "vllm-vision", "tei", "whisper"]
    subprocess.run(COMPOSE + ["up", "-d", *target_services], check=True)
    yield
    subprocess.run(COMPOSE + ["down"], check=True)
    if created_env:
        env_file.unlink()


def test_postgres_ready():
    ok = _wait(
        lambda: subprocess.run(
            ["docker", "exec", "orgchat-postgres", "pg_isready", "-U", "orgchat"],
            capture_output=True,
        ).returncode == 0
    )
    assert ok, "postgres never became healthy"


def test_redis_ready():
    ok = _wait(
        lambda: subprocess.run(
            ["docker", "exec", "orgchat-redis", "redis-cli", "ping"],
            capture_output=True,
        ).returncode == 0
    )
    assert ok, "redis never became healthy"


def test_qdrant_ready():
    ok = _wait(
        lambda: httpx.get("http://localhost:6333/readyz", timeout=2).status_code == 200
    )
    assert ok, "qdrant never became healthy"


@pytest.mark.skipif(SKIP_GPU, reason="GPU services skipped (SKIP_GPU_SMOKE=1)")
def test_vllm_chat_ready():
    ok = _wait(
        lambda: httpx.get("http://localhost:8000/v1/models", timeout=2).status_code == 200,
        timeout_s=600,
    )
    assert ok, "vllm-chat never became healthy"
