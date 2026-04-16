import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "preflight_models.py"


def test_script_dryrun_prints_plan():
    env = os.environ.copy()
    env.update(
        {
            "EMBED_MODEL": "BAAI/bge-m3",
            "CHAT_MODEL": "Qwen/Qwen2.5-14B-Instruct-AWQ",
            "VISION_MODEL": "Qwen/Qwen2-VL-7B-Instruct",
            "WHISPER_MODEL": "medium",
            "MODEL_CACHE": "/tmp/orgchat-models-preflight-test",
        }
    )
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    for m in [
        "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "Qwen/Qwen2-VL-7B-Instruct",
        "BAAI/bge-m3",
        "medium",
    ]:
        assert m in r.stdout, f"expected {m!r} in stdout, got:\n{r.stdout}"
