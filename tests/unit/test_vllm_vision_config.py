from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_vllm_vision_block_present():
    """Sanity: vllm-vision service is defined and parameterized by VISION_MODEL.

    Historically this test also pinned ``--enable-sleep-mode``. Main's WIP
    replaced that flag with ``--enforce-eager`` (safer default for the
    Qwen2-VL-2B vision model); accept either so the test tolerates operator
    choice. Once the Gemma 4 swap lands and vllm-vision is retired, this
    test will be deleted.
    """
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "vllm-vision:" in content
    assert "${VISION_MODEL}" in content
    assert ("--enable-sleep-mode" in content) or ("--enforce-eager" in content)
