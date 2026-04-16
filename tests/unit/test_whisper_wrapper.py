import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "whisper_service"))

def test_app_module_has_expected_endpoints():
    mod = importlib.import_module("app")
    routes = {r.path for r in mod.app.routes}
    for path in ["/health", "/sleep", "/wake_up", "/v1/audio/transcriptions"]:
        assert path in routes, f"endpoint missing: {path}"
