"""Plan B Phase 4.2 — stage Qwen3-4B-AWQ weights script."""
import pathlib
import stat

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage_qwen3_qu.sh"


def test_script_exists():
    assert SCRIPT.exists(), "stage_qwen3_qu.sh missing"


def test_script_is_executable():
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, "stage_qwen3_qu.sh must be executable"


def test_script_uses_offline_cache_path():
    """The script must target the docker-mounted cache directory."""
    content = SCRIPT.read_text()
    assert "volumes/models" in content, (
        "Script must pre-cache to volumes/models (the path mounted into vllm-qu)"
    )


def test_script_downloads_qwen3_4b_awq():
    content = SCRIPT.read_text()
    assert "Qwen3-4B-Instruct-2507-AWQ" in content, (
        "Script must download the Qwen3-4B AWQ model"
    )


def test_script_verifies_size():
    content = SCRIPT.read_text()
    assert "du -sh" in content or "du -h" in content, (
        "Script must verify the cached size to catch incomplete downloads"
    )


def test_script_handles_optional_token():
    """The cpatonn build is public so HF_TOKEN is optional, but the script must
    surface it so users on private mirrors don't get a silent 401."""
    content = SCRIPT.read_text()
    assert "HF_TOKEN" in content, (
        "Script must mention HF_TOKEN (even if optional) so users on private "
        "mirrors get a clear hint"
    )


def test_script_uses_set_e_safety():
    """Bash safety header — bail on first error so partial caches don't survive."""
    content = SCRIPT.read_text()
    assert "set -e" in content or "set -euo" in content, (
        "Script must use bash strict mode (set -euo pipefail)"
    )


def test_script_prints_next_step():
    """Operator UX: the script should hand off to the next command."""
    content = SCRIPT.read_text()
    assert "vllm-qu" in content, (
        "Script should mention 'docker compose up -d vllm-qu' as the next step"
    )
