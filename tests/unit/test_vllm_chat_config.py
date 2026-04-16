from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_vllm_chat_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "vllm-chat:" in content
    assert "vllm/vllm-openai" in content
    assert "${CHAT_MODEL}" in content
    assert "--gpu-memory-utilization" in content
    assert "--enable-prefix-caching" in content
    assert "driver: nvidia" in content
