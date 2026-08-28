"""Validation helpers for MCP server configuration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from medrix_flow.config.extensions_config import McpServerConfig

_DEFAULT_ALLOWED_MCP_COMMANDS = {
    "bunx",
    "deno",
    "docker",
    "node",
    "npx",
    "pnpm",
    "pnpx",
    "python",
    "python3",
    "uv",
    "uvx",
}
_BLOCKED_SHELL_EXECUTABLES = {"bash", "cmd", "dash", "fish", "ksh", "powershell", "pwsh", "sh", "zsh"}
_BLOCKED_EVAL_FLAGS = {"--command", "--eval", "--print", "-c", "-e", "-p", "/c"}


def get_allowed_mcp_commands() -> set[str]:
    """Return the effective MCP stdio command allowlist."""

    extra = {item.strip() for item in os.getenv("MEDRIX_FLOW_ALLOWED_MCP_COMMANDS", "").split(",") if item.strip()}
    return {command.lower() for command in _DEFAULT_ALLOWED_MCP_COMMANDS | extra}


def validate_mcp_server_config(server_name: str, config: McpServerConfig) -> None:
    """Validate a server config before persisting or launching it."""

    transport_type = config.type or "stdio"

    if transport_type == "stdio":
        if not config.command:
            raise ValueError(f"MCP server '{server_name}' with stdio transport requires 'command' field")
        validate_stdio_command(config.command, config.args)
        return

    if transport_type in ("sse", "http"):
        if not config.url:
            raise ValueError(f"MCP server '{server_name}' with {transport_type} transport requires 'url' field")
        return

    raise ValueError(f"MCP server '{server_name}' has unsupported transport type: {transport_type}")


def validate_stdio_command(command: str, args: list[str] | None = None) -> None:
    """Reject shell launchers and one-shot eval patterns for stdio MCP servers."""

    args = args or []
    command_path = Path(command)
    command_name = command_path.name.lower()

    if command_name in _BLOCKED_SHELL_EXECUTABLES:
        raise ValueError(f"Command '{command}' is not allowed for MCP stdio servers.")

    if command_name not in get_allowed_mcp_commands():
        raise ValueError(f"Command '{command}' is not in the MCP allowlist. Set MEDRIX_FLOW_ALLOWED_MCP_COMMANDS to extend the allowlist.")

    blocked_flags = sorted(_BLOCKED_EVAL_FLAGS & set(args))
    if blocked_flags:
        raise ValueError(f"Command '{command}' uses blocked inline-eval flags: {', '.join(blocked_flags)}")

    if command_path.is_absolute():
        if not command_path.exists() or not os.access(command_path, os.X_OK):
            raise ValueError(f"Command '{command}' is not executable.")
        return

    if shutil.which(command) is None:
        raise ValueError(f"Command '{command}' not found in PATH.")
