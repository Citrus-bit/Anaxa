"""Isolated auth contracts for the local ``scripts/serve.sh`` launcher.

The fixture replaces every service executable with a recorder, so these tests
exercise the launcher's real environment construction without opening ports or
starting Gateway, Next.js, or nginx.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_SH = REPO_ROOT / "scripts" / "serve.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _isolated_launcher(tmp_path: Path) -> tuple[Path, dict[str, str], Path, Path]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise serve.sh")

    worktree = tmp_path / "repo"
    scripts = worktree / "scripts"
    bin_dir = tmp_path / "bin"
    scripts.mkdir(parents=True)
    bin_dir.mkdir()
    (worktree / "backend").mkdir()
    (worktree / "frontend").mkdir()
    (worktree / "docker" / "nginx").mkdir(parents=True)
    shutil.copy2(SERVE_SH, scripts / "serve.sh")
    (worktree / "config.yaml").write_text("models: []\n", encoding="utf-8")

    frontend_capture = tmp_path / "frontend-env.jsonl"
    gateway_capture = tmp_path / "gateway-env.txt"

    _write_executable(scripts / "config-upgrade.sh", "#!/bin/sh\nexit 0\n")
    _write_executable(scripts / "cleanup-containers.sh", "#!/bin/sh\nexit 0\n")
    _write_executable(
        scripts / "wait-for-port.sh",
        """#!/bin/sh
case "$1" in
  8001) capture="$CAPTURE_GATEWAY" ;;
  3000) capture="$CAPTURE_FRONTEND" ;;
  *) exit 0 ;;
esac
expected="${EXPECTED_CAPTURE_COUNT:-1}"
i=0
while [ "$i" -lt 200 ]; do
  if [ -f "$capture" ] && [ "$(wc -l < "$capture" | tr -d ' ')" -ge "$expected" ]; then
    exit 0
  fi
  i=$((i + 1))
  command sleep 0.01
done
exit 1
""",
    )
    (scripts / "detect_uv_extras.py").write_text("print('')\n", encoding="utf-8")
    (scripts / "pnpm.py").write_text(
        """import json
import os
from pathlib import Path

record = {
    "better_auth_secret": os.environ.get("BETTER_AUTH_SECRET"),
    "ui_password": os.environ.get("MEDRIX_FLOW_UI_PASSWORD"),
}
with Path(os.environ["CAPTURE_FRONTEND"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record) + "\\n")
""",
        encoding="utf-8",
    )

    _write_executable(
        bin_dir / "uv",
        """#!/bin/sh
if [ "${MEDRIX_FLOW_UI_PASSWORD+x}" = x ]; then
  ui_state="set:${MEDRIX_FLOW_UI_PASSWORD}"
else
  ui_state="unset"
fi
printf '%s\\n' "$ui_state" >> "$CAPTURE_GATEWAY"
""",
    )
    for name in ("lsof", "pgrep", "ss", "netstat"):
        _write_executable(bin_dir / name, "#!/bin/sh\nexit 1\n")
    _write_executable(bin_dir / "nginx", "#!/bin/sh\nexit 0\n")
    _write_executable(bin_dir / "sleep", "#!/bin/sh\nexit 0\n")

    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["CAPTURE_FRONTEND"] = str(frontend_capture)
    environment["CAPTURE_GATEWAY"] = str(gateway_capture)
    for name in (
        "BETTER_AUTH_SECRET",
        "DEER_FLOW_HOME",
        "DEER_FLOW_PROJECT_ROOT",
        "MEDRIX_FLOW_UI_PASSWORD",
    ):
        environment.pop(name, None)
    environment["BASH_EXECUTABLE"] = bash
    return worktree, environment, frontend_capture, gateway_capture


def _run_prod(worktree: Path, environment: dict[str, str], *, expected_count: int) -> None:
    environment = environment.copy()
    environment["EXPECTED_CAPTURE_COUNT"] = str(expected_count)
    result = subprocess.run(
        [environment["BASH_EXECUTABLE"], str(worktree / "scripts" / "serve.sh"), "--prod", "--daemon", "--skip-install"],
        cwd=worktree,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _frontend_records(path: Path) -> list[dict[str, str | None]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_serve_prod_preserves_existing_secret_and_clears_gateway_ui_password(tmp_path: Path) -> None:
    worktree, environment, frontend_capture, gateway_capture = _isolated_launcher(tmp_path)
    environment["BETTER_AUTH_SECRET"] = "configured-auth-secret"
    environment["MEDRIX_FLOW_UI_PASSWORD"] = "frontend-only-password"

    _run_prod(worktree, environment, expected_count=1)

    assert _frontend_records(frontend_capture) == [
        {
            "better_auth_secret": "configured-auth-secret",
            "ui_password": "frontend-only-password",
        }
    ]
    assert gateway_capture.read_text(encoding="utf-8").splitlines() == ["set:"]


def test_serve_prod_persists_generated_secret_across_restarts(tmp_path: Path) -> None:
    worktree, environment, frontend_capture, _ = _isolated_launcher(tmp_path)

    _run_prod(worktree, environment, expected_count=1)
    _run_prod(worktree, environment, expected_count=2)

    records = _frontend_records(frontend_capture)
    secrets = [record["better_auth_secret"] for record in records]
    assert len(secrets) == 2
    assert secrets[0]
    assert secrets[1] == secrets[0]

    secret_file = worktree / "backend" / ".deer-flow" / ".better-auth-secret"
    assert secret_file.read_text(encoding="utf-8").strip() == secrets[0]
