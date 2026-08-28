#!/usr/bin/env python3
"""Release safety check for local-only Anaxa data.

The check is intentionally conservative: local runtime data, secrets, memories,
logs, caches, and uploaded/generated artifacts must never be tracked or staged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import PurePosixPath


SENSITIVE_EXACT = {
    ".env",
    ".mcp.json",
    "config.yaml",
    "config.yaml.bak",
    "extensions_config.json",
    "mcp_config.json",
    "frontend/.env",
    "frontend/.env.local",
    "frontend/.env.production",
    "frontend/.env.development",
    "frontend/.env.test",
}

SENSITIVE_PARTS = {
    ".cache",
    ".claude",
    ".code-review-graph",
    ".codex",
    ".deer-flow",
    ".langgraph_api",
    ".medrix-flow",
    ".next",
    ".obsidian",
    ".omc",
    ".pnpm-store",
    "__pycache__",
    "logs",
    "node_modules",
}

SENSITIVE_PREFIXES = (
    ".cache/",
    ".claude/",
    ".code-review-graph/",
    ".codex/",
    ".deer-flow/",
    ".obsidian/",
    ".omc/",
    ".pnpm-store/",
    "backend/.langgraph_api/",
    "backend/.deer-flow/",
    "backend/.medrix-flow/",
    "cache/",
    "checkpoints/",
    "log/",
    "logs/",
    "outputs/",
    "skills/custom/",
    "uploads/",
)

SENSITIVE_SUFFIXES = (
    ".db",
    ".db-shm",
    ".db-wal",
    ".log",
    ".pckl",
    ".pickle",
    ".pyc",
    ".sqlite",
    ".sqlite3",
    ".sqlite3-shm",
    ".sqlite3-wal",
    ".tsbuildinfo",
)

SENSITIVE_NAMES = {
    "memory.json",
    "project-memory.json",
    "session.json",
    "settings.local.json",
    "store.pckl",
}

ALLOWLIST_EXACT = {
    ".env.example",
    "frontend/.env.example",
}


def run_git(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.rstrip("\n") for line in result.stdout.splitlines() if line.strip()]


def normalize_status_path(line: str) -> str | None:
    if not line:
        return None
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    path = path.strip().strip('"')
    return path or None


def is_sensitive(path_text: str) -> bool:
    path_text = path_text.strip().lstrip("./")
    if not path_text or path_text in ALLOWLIST_EXACT:
        return False
    if path_text in SENSITIVE_EXACT:
        return True
    if path_text.startswith(SENSITIVE_PREFIXES):
        return True

    path = PurePosixPath(path_text)
    parts = set(path.parts)
    if parts & SENSITIVE_PARTS:
        return True
    if path.name in SENSITIVE_NAMES:
        return True
    if path.name.endswith(SENSITIVE_SUFFIXES):
        return True
    if path.name.startswith(".langgraph_checkpoint"):
        return True
    if path.name.startswith(".env.") and path.name != ".env.example":
        return True
    return False


def main() -> int:
    print("==========================================")
    print("  Anaxa Release Safety Check")
    print("==========================================")
    print()

    tracked = run_git(["ls-files"])
    tracked_hits = sorted(path for path in tracked if is_sensitive(path))

    status_paths = [
        path
        for line in run_git(["status", "--short", "--untracked-files=all"])
        if (path := normalize_status_path(line))
    ]
    status_hits = sorted(path for path in status_paths if is_sensitive(path))

    failed = False
    if tracked_hits:
        failed = True
        print("✗ Sensitive local files are tracked by Git:")
        for path in tracked_hits:
            print(f"  - {path}")
        print()

    if status_hits:
        failed = True
        print("✗ Sensitive local files are staged or visible in Git status:")
        for path in status_hits:
            print(f"  - {path}")
        print()

    if failed:
        print("Do not publish yet. Remove these paths from Git tracking or update .gitignore.")
        print("This script does not delete local files.")
        return 1

    print("✓ No tracked or staged local secrets/runtime data detected.")
    print("✓ Safe to continue with normal code review before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
