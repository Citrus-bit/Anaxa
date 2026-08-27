"""Regression tests for docker sandbox mode detection logic."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "docker.sh"
COMPOSE_PATH = REPO_ROOT / "docker" / "docker-compose-dev.yaml"


def _detect_mode_with_config(config_content: str) -> str:
    """Write config content into a temp project root and execute detect_sandbox_mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        (tmp_root / "config.yaml").write_text(config_content)

        command = f"source '{SCRIPT_PATH}' && PROJECT_ROOT='{tmp_root}' && detect_sandbox_mode"

        output = subprocess.check_output(
            ["bash", "-lc", command],
            text=True,
        ).strip()

        return output


def test_detect_mode_defaults_to_local_when_config_missing():
    """No config file should default to local mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        command = f"source '{SCRIPT_PATH}' && PROJECT_ROOT='{tmpdir}' && detect_sandbox_mode"
        output = subprocess.check_output(["bash", "-lc", command], text=True).strip()

    assert output == "local"


def test_detect_mode_local_provider():
    """Local sandbox provider should map to local mode."""
    config = """
sandbox:
  use: medrix_flow.sandbox.local:LocalSandboxProvider
""".strip()

    assert _detect_mode_with_config(config) == "local"


def test_detect_mode_aio_without_provisioner_url():
    """AIO sandbox without provisioner_url should map to aio mode."""
    config = """
sandbox:
  use: medrix_flow.community.aio_sandbox:AioSandboxProvider
""".strip()

    assert _detect_mode_with_config(config) == "aio"


def test_detect_mode_provisioner_with_url():
    """AIO sandbox with provisioner_url should map to provisioner mode."""
    config = """
sandbox:
  use: medrix_flow.community.aio_sandbox:AioSandboxProvider
  provisioner_url: http://provisioner:6204
""".strip()

    assert _detect_mode_with_config(config) == "provisioner"


def test_detect_mode_ignores_commented_provisioner_url():
    """Commented provisioner_url should not activate provisioner mode."""
    config = """
sandbox:
  use: medrix_flow.community.aio_sandbox:AioSandboxProvider
  # provisioner_url: http://provisioner:6204
""".strip()

    assert _detect_mode_with_config(config) == "aio"


def test_detect_mode_unknown_provider_falls_back_to_local():
    """Unknown sandbox provider should default to local mode."""
    config = """
sandbox:
  use: custom.module:UnknownProvider
""".strip()

    assert _detect_mode_with_config(config) == "local"


def _seed_compose_file(tmp_root: Path) -> None:
    (tmp_root / "docker-compose-dev.yaml").write_text("services: {}\n", encoding="utf-8")


def _seed_env_examples(tmp_root: Path) -> None:
    (tmp_root / ".env.example").write_text("# test\n", encoding="utf-8")
    frontend = tmp_root / "frontend"
    frontend.mkdir(exist_ok=True)
    (frontend / ".env.example").write_text("# test\n", encoding="utf-8")


def _run_script_against(tmp_root: Path, body: str) -> subprocess.CompletedProcess[str]:
    command = f"""
source '{SCRIPT_PATH}'
PROJECT_ROOT='{tmp_root}'
DOCKER_DIR='{tmp_root}'
{body}
"""
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


@pytest.mark.parametrize("command", ["logs --gateway", "stop", "restart"])
def test_read_only_commands_do_not_create_env_files(command: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        _seed_compose_file(tmp_root)
        _seed_env_examples(tmp_root)
        result = _run_script_against(
            tmp_root,
            f"""
docker() {{ echo '2.24.0'; }}
COMPOSE_CMD=true
{command}
""",
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert not (tmp_root / ".env").exists()
        assert not (tmp_root / "frontend" / ".env").exists()


def test_start_env_files_are_created_from_examples():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        _seed_env_examples(tmp_root)
        result = _run_script_against(tmp_root, "ensure_env_files")

        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_root / ".env").read_text(encoding="utf-8") == "# test\n"
        assert (tmp_root / "frontend" / ".env").read_text(encoding="utf-8") == "# test\n"


def test_existing_env_files_are_not_overwritten():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        _seed_env_examples(tmp_root)
        (tmp_root / ".env").write_text("KEEP=root\n", encoding="utf-8")
        (tmp_root / "frontend" / ".env").write_text("KEEP=frontend\n", encoding="utf-8")
        result = _run_script_against(tmp_root, "ensure_env_files")

        assert result.returncode == 0, result.stdout + result.stderr
        assert (tmp_root / ".env").read_text(encoding="utf-8") == "KEEP=root\n"
        assert (tmp_root / "frontend" / ".env").read_text(encoding="utf-8") == "KEEP=frontend\n"


@pytest.mark.parametrize(
    ("version", "expected_returncode"),
    [("2.23.3", 1), ("2.24.0", 0), ("v2.40.2-desktop.1", 0), ("", 0)],
)
def test_compose_version_floor(version: str, expected_returncode: int):
    result = _run_script_against(
        Path(tempfile.gettempdir()),
        f"""
docker() {{ echo '{version}'; }}
docker-compose() {{ echo '{version}'; }}
require_compose_version
""",
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr


def test_hyphenated_compose_binary_is_reused_for_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        _seed_compose_file(tmp_root)
        marker = tmp_root / "compose-call.txt"
        result = _run_script_against(
            tmp_root,
            f"""
docker() {{ return 1; }}
docker-compose() {{
  if [ "$1" = version ]; then echo '2.24.0'; return 0; fi
  printf '%s\\n' "$*" > '{marker}'
}}
stop
""",
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert "down" in marker.read_text(encoding="utf-8")


def test_dev_compose_env_files_are_optional():
    compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
    expected = {
        "provisioner": "../.env",
        "frontend": "../frontend/.env",
        "gateway": "../.env",
        "langgraph": "../.env",
    }
    for service_name, path in expected.items():
        assert compose["services"][service_name]["env_file"] == [
            {"path": path, "required": False}
        ]
