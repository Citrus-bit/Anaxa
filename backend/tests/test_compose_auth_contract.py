"""Contract tests for auth configuration in the Docker Compose stacks.

The frontend owns the outer UI gate and nginx auth-request token, while the
Gateway validates the same proxy token and optional admin token. Keeping the
values explicit on both containers prevents a root ``.env`` setting from
silently reaching only one side of that boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATHS = {
    "prod": REPO_ROOT / "docker" / "docker-compose.yaml",
    "dev": REPO_ROOT / "docker" / "docker-compose-dev.yaml",
}


def _environment(variant: str, service: str) -> set[str]:
    compose = yaml.safe_load(COMPOSE_PATHS[variant].read_text(encoding="utf-8"))
    values = compose["services"][service]["environment"]
    return {str(value) for value in values}


@pytest.mark.parametrize(
    ("variant", "environment_entry", "secret_suffix"),
    [
        ("prod", "MEDRIX_FLOW_ENV=production", ""),
        ("dev", "MEDRIX_FLOW_ENV=${MEDRIX_FLOW_ENV:-development}", ":-"),
    ],
)
@pytest.mark.parametrize("service", ["frontend", "gateway"])
def test_compose_propagates_shared_auth_configuration(
    variant: str,
    environment_entry: str,
    secret_suffix: str,
    service: str,
) -> None:
    environment = _environment(variant, service)
    assert f"BETTER_AUTH_SECRET=${{BETTER_AUTH_SECRET{secret_suffix}}}" in environment
    assert environment_entry in environment
    assert "MEDRIX_GATEWAY_ADMIN_TOKEN=${MEDRIX_GATEWAY_ADMIN_TOKEN:-}" in environment


@pytest.mark.parametrize("variant", sorted(COMPOSE_PATHS))
def test_ui_password_is_injected_only_into_frontend(variant: str) -> None:
    entry = "MEDRIX_FLOW_UI_PASSWORD=${MEDRIX_FLOW_UI_PASSWORD:-}"
    assert entry in _environment(variant, "frontend")
    for service in ("gateway", "provisioner"):
        environment = _environment(variant, service)
        assert entry not in environment
        assert "MEDRIX_FLOW_UI_PASSWORD=" in environment


def test_local_launcher_clears_ui_password_only_for_gateway() -> None:
    content = (REPO_ROOT / "scripts" / "serve.sh").read_text(encoding="utf-8")
    assert 'run_service "Gateway" \\\n    "cd backend && env MEDRIX_FLOW_UI_PASSWORD=' in content
    assert 'run_service "Frontend" \\\n    "cd frontend && env MEDRIX_FLOW_UI_PASSWORD=' not in content


@pytest.mark.parametrize("path", [REPO_ROOT / ".env.example", REPO_ROOT / "frontend" / ".env.example"])
def test_auth_variables_are_documented_in_env_examples(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    for name in (
        "BETTER_AUTH_SECRET",
        "MEDRIX_FLOW_ENV",
        "MEDRIX_FLOW_UI_PASSWORD",
        "MEDRIX_GATEWAY_ADMIN_TOKEN",
    ):
        assert name in content, f"{name} must be documented in {path}"
