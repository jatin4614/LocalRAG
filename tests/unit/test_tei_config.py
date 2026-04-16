from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_tei_block_present():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "tei:" in content
    assert "ghcr.io/huggingface/text-embeddings-inference" in content
    assert "${EMBED_MODEL}" in content
