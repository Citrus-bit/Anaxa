from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.sandbox.security import is_host_bash_allowed, uses_local_sandbox_provider


def _safe_stem(value: str | None) -> str:
    raw = (value or "matlab_run").strip().replace(" ", "_")
    stem = "".join(ch for ch in raw if ch.isalnum() or ch in {"_", "-", "."}).strip("._-")
    return stem or "matlab_run"


def _virtual_output_path(outputs_dir: Path, filepath: Path) -> str:
    relative = filepath.resolve().relative_to(outputs_dir.resolve())
    return f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}"


def _resolve_workspace_dir(thread_id: str, thread_data: dict | None) -> Path:
    workspace_path = (thread_data or {}).get("workspace_path")
    if workspace_path:
        return Path(workspace_path).resolve()
    return get_paths().sandbox_workspace_dir(thread_id).resolve()


@tool("matlab_execution", parse_docstring=True)
def matlab_execution_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    script_content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    script_name: str = "matlab_run",
    matlab_executable: str = "matlab",
    timeout_seconds: int = 600,
) -> Command:
    """Run MATLAB in trusted local batch mode and collect output artifacts.

    Use this only when MATLAB command-line execution is required. It writes a
    `.m` script into `/mnt/user-data/workspace`, runs `matlab -batch`, and
    collects logs plus files written under a dedicated `/mnt/user-data/outputs`
    directory. It does not control the MATLAB GUI.

    Host MATLAB execution is allowed only for trusted local configuration:
    `LocalSandboxProvider` with `sandbox.allow_host_bash: true`. Containerized
    MATLAB should be handled by preparing MATLAB inside the configured sandbox
    and using normal sandbox/bash tools instead.

    Args:
        script_content: Full MATLAB `.m` source to execute.
        script_name: Safe script/output stem.
        matlab_executable: MATLAB executable name or absolute path. Defaults to `matlab`.
        timeout_seconds: Maximum execution time for `matlab -batch`.
    """
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        return Command(update={"messages": [ToolMessage("Error: thread_id is required for matlab_execution.", tool_call_id=tool_call_id)]})
    thread_data = runtime.state.get("thread_data") if runtime.state else {}
    outputs_path = (thread_data or {}).get("outputs_path")
    if not outputs_path:
        return Command(update={"messages": [ToolMessage("Error: thread outputs path is not available.", tool_call_id=tool_call_id)]})
    if not uses_local_sandbox_provider() or not is_host_bash_allowed():
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: MATLAB host execution requires LocalSandboxProvider with sandbox.allow_host_bash: true in a trusted local environment.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    matlab_path = shutil.which(matlab_executable) if not os.path.isabs(matlab_executable) else matlab_executable
    if not matlab_path or not Path(matlab_path).exists():
        return Command(update={"messages": [ToolMessage(f"Error: MATLAB executable not found: {matlab_executable}", tool_call_id=tool_call_id)]})

    stem = _safe_stem(script_name)
    workspace_dir = _resolve_workspace_dir(str(thread_id), thread_data)
    outputs_dir = Path(outputs_path).resolve()
    run_dir = outputs_dir / "matlab-execution" / stem
    workspace_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    script_path = workspace_dir / f"{stem}.m"
    log_path = run_dir / "matlab_execution.log"
    manifest_path = run_dir / "matlab_execution_manifest.json"
    escaped_run_dir = str(run_dir).replace("'", "''")
    escaped_script_path = str(script_path).replace("'", "''")
    wrapped_script = "\n".join(
        [
            f"outputDir = '{escaped_run_dir}';",
            "if ~exist(outputDir, 'dir'); mkdir(outputDir); end",
            "cd(outputDir);",
            script_content.strip(),
            "",
        ]
    )
    script_path.write_text(wrapped_script, encoding="utf-8")

    command = [str(matlab_path), "-batch", f"run('{escaped_script_path}')"]
    try:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_seconds)),
            check=False,
        )
        status = "success" if completed.returncode == 0 else "error"
        log_path.write_text(
            "\n".join(
                [
                    "$ " + " ".join(command),
                    "",
                    "## stdout",
                    completed.stdout,
                    "",
                    "## stderr",
                    completed.stderr,
                ]
            ),
            encoding="utf-8",
        )
        output_files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path.name != manifest_path.name)
        manifest = {
            "status": status,
            "returncode": completed.returncode,
            "script_path": str(script_path),
            "run_dir": str(run_dir),
            "command": command,
            "output_files": [str(path) for path in output_files],
        }
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        log_path.write_text(
            "\n".join(
                [
                    "$ " + " ".join(command),
                    "",
                    f"MATLAB execution timed out after {timeout_seconds} seconds.",
                    "",
                    "## stdout",
                    exc.stdout or "",
                    "",
                    "## stderr",
                    exc.stderr or "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = {
            "status": status,
            "returncode": None,
            "script_path": str(script_path),
            "run_dir": str(run_dir),
            "command": command,
            "output_files": [str(log_path)],
        }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [_virtual_output_path(outputs_dir, manifest_path), _virtual_output_path(outputs_dir, log_path)]
    for path in Path(run_dir).rglob("*"):
        if path.is_file() and path not in {manifest_path, log_path}:
            artifacts.append(_virtual_output_path(outputs_dir, path))

    run_relative = run_dir.resolve().relative_to(outputs_dir)
    virtual_run_dir = f"{VIRTUAL_PATH_PREFIX}/outputs/{run_relative.as_posix()}"
    message = f"MATLAB batch execution {status}. Outputs were collected under `{virtual_run_dir}`."
    return Command(update={"artifacts": artifacts, "messages": [ToolMessage(message, tool_call_id=tool_call_id)]})
