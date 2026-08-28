"""Keep local Gateway listeners private while preserving container networking."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_serve_script_defaults_gateway_to_loopback() -> None:
    """The host launcher must not expose the Gateway on every interface."""
    content = _read("scripts/serve.sh")

    assert '--host \\"\\${GATEWAY_HOST:-127.0.0.1}\\"' in content
    assert "--host 0.0.0.0" not in content


def test_backend_make_targets_default_gateway_to_loopback() -> None:
    """Both local backend targets should retain an explicit override escape hatch."""
    content = _read("backend/Makefile")

    expected = "--host $${GATEWAY_HOST:-127.0.0.1} --port 8001"
    assert content.count(expected) == 2
    assert "--host 0.0.0.0" not in content


def test_container_gateway_listeners_remain_reachable_on_compose_network() -> None:
    """Container processes must bind broadly so nginx can reach them by service name."""
    compose = _read("docker/docker-compose.yaml")
    entrypoint = _read("docker/dev-entrypoint.sh")

    assert "uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001" in compose
    assert "--host 0.0.0.0 --port 8001" in entrypoint
