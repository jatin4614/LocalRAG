import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_compose_config_valid():
    result = subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "compose/docker-compose.yml"),
         "--env-file", str(ROOT / "compose/.env.example"), "config"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"compose config failed:\n{result.stderr}"

def test_stateful_services_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    for svc in ["postgres:", "redis:", "qdrant:"]:
        assert svc in content, f"missing service block: {svc}"

def test_env_example_keys():
    content = (ROOT / "compose/.env.example").read_text()
    for key in ["POSTGRES_PASSWORD=", "ADMIN_EMAIL=", "ADMIN_PASSWORD=",
                "SESSION_SECRET=", "WEBUI_NAME=", "DOMAIN="]:
        assert key in content, f"missing env key: {key}"
