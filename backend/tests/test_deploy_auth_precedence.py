"""Regression tests for production deploy secret precedence."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deployment_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    worktree = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", worktree / "scripts")
    shutil.copytree(REPO_ROOT / "docker", worktree / "docker")
    (worktree / "backend").mkdir()
    (worktree / "config.yaml").write_text(
        "sandbox:\n  use: deerflow.sandbox:LocalSandboxProvider\n",
        encoding="utf-8",
    )
    (worktree / "extensions_config.json").write_text(
        '{"mcpServers":{},"skills":{}}\n',
        encoding="utf-8",
    )
    (worktree / ".env").write_text(
        "BETTER_AUTH_SECRET=dotenv-auth-secret\nDEER_FLOW_INTERNAL_AUTH_TOKEN=dotenv-internal-token\n",
        encoding="utf-8",
    )

    capture = tmp_path / "docker-env.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env sh\n"
        'printf \'%s\\n\' \
        "BETTER_AUTH_SECRET=$BETTER_AUTH_SECRET" \
        "DEER_FLOW_INTERNAL_AUTH_TOKEN=$DEER_FLOW_INTERNAL_AUTH_TOKEN" \
        > "$CAPTURE_DOCKER_ENV"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment.pop("BETTER_AUTH_SECRET", None)
    environment.pop("DEER_FLOW_INTERNAL_AUTH_TOKEN", None)
    environment["CAPTURE_DOCKER_ENV"] = str(capture)
    return worktree, environment, capture


def test_deploy_honors_unexported_dotenv_secrets(tmp_path: Path) -> None:
    worktree, environment, capture = _deployment_fixture(tmp_path)

    result = subprocess.run(
        ["bash", str(worktree / "scripts" / "deploy.sh"), "start"],
        cwd=worktree,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "BETTER_AUTH_SECRET=dotenv-auth-secret",
        "DEER_FLOW_INTERNAL_AUTH_TOKEN=dotenv-internal-token",
    ]
    assert not (worktree / "backend" / ".deer-flow" / ".better-auth-secret").exists()
    assert not (worktree / "backend" / ".deer-flow" / ".internal-auth-token").exists()


def test_deploy_shell_secrets_override_dotenv(tmp_path: Path) -> None:
    worktree, environment, capture = _deployment_fixture(tmp_path)
    environment["BETTER_AUTH_SECRET"] = "shell-auth-secret"
    environment["DEER_FLOW_INTERNAL_AUTH_TOKEN"] = "shell-internal-token"

    result = subprocess.run(
        ["bash", str(worktree / "scripts" / "deploy.sh"), "start"],
        cwd=worktree,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "BETTER_AUTH_SECRET=shell-auth-secret",
        "DEER_FLOW_INTERNAL_AUTH_TOKEN=shell-internal-token",
    ]


def test_deploy_down_does_not_create_runtime_files_in_fresh_checkout(tmp_path: Path) -> None:
    """Stopping must be read-only when no deployment has been initialized yet."""
    worktree = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", worktree / "scripts")
    shutil.copytree(REPO_ROOT / "docker", worktree / "docker")
    (worktree / "backend").mkdir()

    capture = tmp_path / "docker-down-args.txt"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        '#!/usr/bin/env sh\nprintf "%s\\n" "$@" > "$CAPTURE_DOCKER_ARGS"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["CAPTURE_DOCKER_ARGS"] = str(capture)
    for name in (
        "BETTER_AUTH_SECRET",
        "DEER_FLOW_HOME",
        "DEER_FLOW_CONFIG_PATH",
        "DEER_FLOW_EXTENSIONS_CONFIG_PATH",
        "DEER_FLOW_INTERNAL_AUTH_TOKEN",
    ):
        environment.pop(name, None)

    result = subprocess.run(
        ["bash", str(worktree / "scripts" / "deploy.sh"), "down"],
        cwd=worktree,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[-1] == "down"
    assert not (worktree / "config.yaml").exists()
    assert not (worktree / "extensions_config.json").exists()
    assert not (worktree / "backend" / ".deer-flow").exists()
