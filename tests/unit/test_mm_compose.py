from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# Removed: test_model_manager_service_in_compose
# The model-manager service was removed from compose/docker-compose.yml
# (per CLAUDE.md, GPU strategy changed; Open-WebUI now gets GPU directly —
# see commit 50d6ea6). This test asserted the block existed and was stale.


def test_env_example_has_model_manager_keys():
    """MODEL_UNLOAD_IDLE_SECS env key is still documented in .env.example
    even though the service it gated has been removed; keep the
    back-compat key around so an operator with an old .env doesn't see
    a broken reference."""
    c = (ROOT / "compose/.env.example").read_text()
    assert "MODEL_UNLOAD_IDLE_SECS=" in c
