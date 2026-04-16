import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOT = ROOT / "scripts" / "bootstrap.sh"


def test_bootstrap_executable():
    assert BOOT.is_file()
    assert BOOT.stat().st_mode & stat.S_IXUSR, "script not executable"


def test_bootstrap_mentions_key_steps():
    content = BOOT.read_text()
    for step in ["preflight", "gen_self_signed_cert", "apply_patches",
                 "docker compose", "seed_admin", "apply_migrations"]:
        assert step in content, f"bootstrap missing step: {step}"


def test_bootstrap_refuses_without_env(tmp_path):
    """Running from a directory with no compose/.env should exit non-zero."""
    # Copy script into a staging dir with no compose/.env
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "compose").mkdir()
    # No compose/.env in stage dir.
    import shutil
    shutil.copy(BOOT, stage / "bootstrap.sh")
    r = subprocess.run(
        ["bash", str(stage / "bootstrap.sh"), "--dry-run"],
        capture_output=True, text=True,
        cwd=str(stage),
    )
    assert r.returncode != 0
    assert "compose/.env missing" in r.stderr or "compose/.env missing" in r.stdout
