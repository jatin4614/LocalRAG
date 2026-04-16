import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gen_self_signed_cert.sh"

def test_script_exists_and_executable():
    assert SCRIPT.is_file()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable"

def test_script_produces_cert_and_key(tmp_path):
    env = os.environ.copy()
    env["CERT_DIR"]     = str(tmp_path)
    env["CERT_CN"]      = "test.orgchat.lan"
    env["CERT_DAYS"]    = "30"
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "orgchat.crt").is_file()
    assert (tmp_path / "orgchat.key").is_file()
