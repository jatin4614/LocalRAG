"""Plan B Phase 4.1 — `vllm-qu` service in docker-compose."""
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPOSE_FILE = ROOT / "compose" / "docker-compose.yml"


def _load_compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text())


def test_vllm_qu_service_defined():
    compose = _load_compose()
    assert "vllm-qu" in compose["services"], (
        "Plan B Phase 4.1 requires a vllm-qu service in compose/docker-compose.yml"
    )


def test_vllm_qu_service_pinned_to_gpu_1():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    # Accept either the legacy environment-based pinning or the modern
    # deploy.resources.reservations.devices form (matches existing tei /
    # vllm-chat services in this compose file).
    env = svc.get("environment", {}) or {}
    if isinstance(env, list):
        env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in env if "=" in item}
    visible = env.get("NVIDIA_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        assert str(visible) == "1", (
            f"vllm-qu must be pinned to GPU 1, got {visible!r}. "
            "GPU 0 is reserved for vllm-chat."
        )
    else:
        # Fall back to deploy reservation
        deploy = svc.get("deploy") or {}
        devices = (
            deploy.get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
        ids: list[str] = []
        for d in devices:
            ids.extend(str(x) for x in (d.get("device_ids") or []))
        assert ids == ["1"], (
            f"vllm-qu must be pinned to GPU 1 via deploy.reservations, got {ids!r}"
        )


def test_vllm_qu_uses_qwen3_4b_awq():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    cmd = svc.get("command") or []
    if isinstance(cmd, str):
        cmd = cmd.split()
    joined = " ".join(str(x) for x in cmd)
    assert "Qwen3-4B-Instruct-2507-AWQ" in joined, (
        f"vllm-qu must serve Qwen3-4B-AWQ, got command: {joined}"
    )


def test_vllm_qu_caps_gpu_memory():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    cmd = svc.get("command") or []
    if isinstance(cmd, str):
        cmd = cmd.split()
    joined = " ".join(str(x) for x in cmd)
    assert "--gpu-memory-utilization" in joined, (
        "vllm-qu must cap GPU memory utilization to leave room for TEI + reranker"
    )


def test_vllm_qu_health_check():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    assert "healthcheck" in svc, "vllm-qu must declare a healthcheck"


def test_vllm_qu_mounts_offline_cache():
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    volumes = svc.get("volumes") or []
    cache_mount = any(
        ("/root/.cache/huggingface" in str(v))
        or ("hf_cache" in str(v))
        or ("hf-cache" in str(v))
        or ("volumes/models" in str(v))
        for v in volumes
    )
    assert cache_mount, "vllm-qu must mount the model cache for offline weights"


def test_vllm_qu_exposes_host_port():
    """vllm-qu publishes 8101 on the host for direct integration testing."""
    compose = _load_compose()
    svc = compose["services"]["vllm-qu"]
    ports = svc.get("ports") or []
    joined = " ".join(str(p) for p in ports)
    assert "8101" in joined, f"vllm-qu must publish host port 8101, got {ports!r}"


def test_open_webui_has_qu_env_vars():
    """Open-WebUI must expose RAG_QU_* env vars to the bridge."""
    compose = _load_compose()
    env = compose["services"]["open-webui"].get("environment", {}) or {}
    if isinstance(env, list):
        env = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in env if "=" in item}
    for k in ("RAG_QU_URL", "RAG_QU_MODEL", "RAG_QU_ENABLED",
              "RAG_QU_LATENCY_BUDGET_MS", "RAG_QU_CACHE_ENABLED",
              "RAG_QU_CACHE_TTL_SECS", "RAG_QU_SHADOW_MODE"):
        assert k in env, f"open-webui environment missing {k}"
