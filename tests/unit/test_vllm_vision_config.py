from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_vllm_vision_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "vllm-vision:" in content
    assert "${VISION_MODEL}" in content
    assert "--enable-sleep-mode" in content
