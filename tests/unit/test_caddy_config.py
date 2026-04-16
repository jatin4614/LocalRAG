from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def test_caddyfile_routes_root_to_openwebui():
    content = (ROOT / "compose/caddy/Caddyfile").read_text()
    assert "reverse_proxy open-webui:8080" in content

def test_caddy_service_in_compose():
    content = (ROOT / "compose/docker-compose.yml").read_text()
    assert "caddy:" in content
    assert "./caddy/Caddyfile:/etc/caddy/Caddyfile:ro" in content
    assert "../volumes/certs:/certs:ro" in content
