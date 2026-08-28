"""Regression tests for the extensions-config singleton source tracking."""

from __future__ import annotations

import json
from pathlib import Path

from deerflow.config.extensions_config import (
    ExtensionsConfig,
    McpServerConfig,
    get_extensions_config,
    reset_extensions_config,
    set_extensions_config,
)


def _write_config(path: Path, server_name: str) -> None:
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    server_name: {
                        "type": "stdio",
                        "command": "echo",
                    }
                },
                "skills": {},
            }
        ),
        encoding="utf-8",
    )


def test_cached_config_reloads_when_legacy_env_path_changes(monkeypatch, tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_config(first_path, "first")
    _write_config(second_path, "second")

    monkeypatch.delenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH", raising=False)
    monkeypatch.setenv("MEDRIX_FLOW_EXTENSIONS_CONFIG_PATH", str(first_path))
    reset_extensions_config()

    try:
        injected = ExtensionsConfig(
            mcp_servers={
                "injected": McpServerConfig(type="stdio", command="echo"),
            }
        )
        set_extensions_config(injected)

        assert get_extensions_config() is injected

        monkeypatch.setenv("MEDRIX_FLOW_EXTENSIONS_CONFIG_PATH", str(second_path))

        reloaded = get_extensions_config()
        assert reloaded is not injected
        assert set(reloaded.mcp_servers) == {"second"}
    finally:
        reset_extensions_config()
