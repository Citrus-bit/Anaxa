"""Compatibility checks for Anaxa's root-level developer entry points."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_root_makefile_keeps_anaxa_compatibility_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("bootstrap", "verify", "release-check", "clean-cache"):
        assert f"{target}:" in makefile

    clean_cache = makefile.split("clean-cache:", 1)[1].split("\nclean:", 1)[0]
    assert "backend/.deer-flow" not in clean_cache.replace("Retained: backend/.deer-flow", "")
    assert "backend/.medrix-flow" not in clean_cache.replace("backend/.medrix-flow", "", 1)
    assert "backend/.venv" not in clean_cache.replace("backend/.venv", "", 1)
    assert "frontend/node_modules" not in clean_cache.replace("frontend/node_modules", "", 1)


def test_release_check_covers_current_and_legacy_runtime_directories() -> None:
    module_path = REPO_ROOT / "scripts" / "release_check.py"
    spec = importlib.util.spec_from_file_location("anaxa_release_check", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.is_sensitive(".deer-flow/memory.json")
    assert module.is_sensitive("backend/.deer-flow/data/deerflow.db")
    assert module.is_sensitive(".medrix-flow/memory.json")
    assert module.is_sensitive("backend/.medrix-flow/data/medrix.db")
