from __future__ import annotations

from unittest.mock import MagicMock, patch

from medrix_flow.sandbox.env_policy import build_sandbox_env, is_blocked_env_name
from medrix_flow.sandbox.local.local_sandbox import LocalSandbox


def test_secret_like_environment_names_are_blocked() -> None:
    for name in (
        "OPENAI_API_KEY",
        "SERVICE_SECRET",
        "ACCESS_TOKEN",
        "DB_PASS",
        "GIT_ASKPASS",
        "DATABASE_URL",
        "MYSQL_PWD",
        "PGSERVICEFILE",
    ):
        assert is_blocked_env_name(name)


def test_normal_runtime_environment_names_are_allowed() -> None:
    for name in ("PATH", "HOME", "LANG", "PWD", "OLDPWD", "VIRTUAL_ENV"):
        assert not is_blocked_env_name(name)


def test_build_environment_scrubs_inherited_values_and_allows_injected_values(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = build_sandbox_env({"REQUEST_TOKEN": "scoped-value"})

    assert "OPENAI_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    assert env["REQUEST_TOKEN"] == "scoped-value"


def test_local_sandbox_passes_scrubbed_environment_to_subprocess(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_API_KEY", "host-secret")
    completed = MagicMock(stdout="ok", stderr="", returncode=0)

    with patch("medrix_flow.sandbox.local.local_sandbox.subprocess.run", return_value=completed) as run:
        assert LocalSandbox("test").execute_command("printf ok") == "ok"

    environment = run.call_args.kwargs["env"]
    assert "PROVIDER_API_KEY" not in environment
    assert "PATH" in environment
