from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

matlab_tool_module = importlib.import_module("medrix_flow.tools.builtins.matlab_execution_tool")


def _runtime(tmp_path: Path) -> SimpleNamespace:
    workspace = tmp_path / "workspace"
    outputs = tmp_path / "outputs"
    workspace.mkdir()
    outputs.mkdir()
    return SimpleNamespace(
        state={"thread_data": {"workspace_path": str(workspace), "outputs_path": str(outputs)}},
        context={"thread_id": "thread-matlab"},
    )


def test_matlab_execution_requires_host_bash_permission(tmp_path, monkeypatch):
    monkeypatch.setattr(matlab_tool_module, "uses_local_sandbox_provider", lambda: True)
    monkeypatch.setattr(matlab_tool_module, "is_host_bash_allowed", lambda: False)

    result = matlab_tool_module.matlab_execution_tool.func(
        runtime=_runtime(tmp_path),
        script_content="disp('hello')",
        tool_call_id="tc-1",
    )

    assert "sandbox.allow_host_bash: true" in result.update["messages"][0].content


def test_matlab_execution_runs_fake_batch_and_collects_outputs(tmp_path, monkeypatch):
    fake_matlab = tmp_path / "matlab"
    fake_matlab.write_text(
        "#!/usr/bin/env bash\necho fake matlab \"$@\"\nprintf 'value\\n1\\n' > result.csv\n",
        encoding="utf-8",
    )
    fake_matlab.chmod(0o755)
    monkeypatch.setattr(matlab_tool_module, "uses_local_sandbox_provider", lambda: True)
    monkeypatch.setattr(matlab_tool_module, "is_host_bash_allowed", lambda: True)

    result = matlab_tool_module.matlab_execution_tool.func(
        runtime=_runtime(tmp_path),
        script_content="disp('hello')",
        script_name="demo",
        matlab_executable=str(fake_matlab),
        tool_call_id="tc-1",
    )

    outputs = tmp_path / "outputs"
    manifest = json.loads((outputs / "matlab-execution/demo/matlab_execution_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "success"
    assert manifest["command"][1] == "-batch"
    assert (outputs / "matlab-execution/demo/result.csv").exists()
    assert "/mnt/user-data/outputs/matlab-execution/demo/result.csv" in result.update["artifacts"]
