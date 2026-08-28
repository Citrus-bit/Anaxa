"""Compatibility runtime for Anaxa's research persistence.

DeerFlow owns the primary Gateway runtime and SQLAlchemy persistence.  Anaxa's
academic, experiment, and research features intentionally keep their existing
SQLite schemas for backwards compatibility, so they are initialized as an
optional, independently closable sidecar.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_COMPONENTS: tuple[tuple[str, str, str, str], ...] = (
    ("academic", "medrix_flow.academic", "AcademicRepository", "AcademicResearchService"),
    ("experiment", "medrix_flow.experiments", "ExperimentRepository", "ExperimentService"),
    ("research", "medrix_flow.research", "ResearchRepository", "ResearchQuestService"),
)


@dataclass
class ResearchRuntimeStatus:
    """Observable state for the optional research sidecar."""

    enabled: bool = False
    base_dir: str | None = None
    components: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def _resolve_base_dir() -> Path:
    """Resolve a legacy data directory without mixing it with DeerFlow state."""
    for variable in ("ANAXA_RESEARCH_HOME", "MEDRIX_FLOW_HOME"):
        value = os.environ.get(variable)
        if value:
            return Path(value).expanduser().resolve()

    # Existing Anaxa data wins so an upgrade never silently starts a new DB.
    candidates = (
        Path.cwd() / ".medrix-flow" if Path.cwd().name == "backend" else None,
        _PROJECT_ROOT / "backend" / ".medrix-flow",
        _PROJECT_ROOT / ".medrix-flow",
    )
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()

    # A fresh DeerFlow checkout gets an isolated sidecar under its configured
    # home; this keeps core ``deerflow.db`` and legacy SQLite schemas separate.
    deer_home = os.environ.get("DEER_FLOW_HOME")
    if deer_home:
        return (Path(deer_home).expanduser() / "anaxa-research").resolve()
    return (_PROJECT_ROOT / "backend" / ".medrix-flow").resolve()


def _patch_legacy_paths(base_dir: Path) -> tuple[Any, Any]:
    """Point the old package's dynamic ``get_paths`` singleton at *base_dir*.

    The old services import ``get_paths`` as a function, but that function
    reads the module-level singleton on every call.  Swapping that singleton
    during the sidecar lifetime preserves the old schema and output layout
    without changing DeerFlow's path resolver.
    """
    paths_module = importlib.import_module("medrix_flow.config.paths")
    previous = getattr(paths_module, "_paths", None)
    paths_module._paths = paths_module.Paths(base_dir=base_dir)
    return paths_module, previous


def _db_path(paths: Any, name: str, base_dir: Path) -> Path:
    attribute = f"{name}_db_file"
    value = getattr(paths, attribute, base_dir / f"{name}.sqlite3")
    return Path(value)


async def _close_db(db: Any) -> None:
    try:
        await db.close()
    except Exception:
        logger.warning("Failed to close legacy research database", exc_info=True)


@asynccontextmanager
async def legacy_research_runtime(app: Any) -> AsyncIterator[ResearchRuntimeStatus]:
    """Initialize Anaxa research services as an optional Gateway sidecar.

    Each component is isolated: an unavailable scientific dependency (for
    example the experiment stack) does not disable academic or quest APIs.
    Startup errors are recorded and logged, while request accessors return a
    precise 503 for the unavailable component.
    """
    status = ResearchRuntimeStatus()
    app.state.research_runtime_status = status

    if os.environ.get("ANAXA_RESEARCH_DISABLED", "").strip() == "1":
        status.components = {name: "disabled" for name, *_ in _COMPONENTS}
        yield status
        return

    try:
        db_module = importlib.import_module("medrix_flow.runtime.db")
        sqlite_db_type = db_module.SQLiteRuntimeDB
        paths_module, previous_paths = _patch_legacy_paths(_resolve_base_dir())
        paths = paths_module.get_paths()
        base_dir = Path(paths.base_dir)
        status.base_dir = str(base_dir)
    except Exception as exc:  # pragma: no cover - only exercised by broken installs
        message = f"legacy research package unavailable: {exc}"
        logger.warning(message, exc_info=True)
        status.errors["runtime"] = message
        for name, *_ in _COMPONENTS:
            status.components[name] = "unavailable"
        yield status
        return

    databases: dict[str, Any] = {}
    opened: list[Any] = []
    try:
        for name, module_name, repository_name, service_name in _COMPONENTS:
            try:
                module = importlib.import_module(module_name)
                repository_type = getattr(module, repository_name)
                service_type = getattr(module, service_name)
                db = sqlite_db_type(_db_path(paths, name, base_dir))
                await db.connect()
                opened.append(db)
                repository = repository_type(db)
                await repository.setup()
                service = service_type(repository)
                databases[name] = db
                setattr(app.state, f"{name}_db", db)
                setattr(app.state, f"{name}_repository", repository)
                setattr(app.state, f"{name}_service", service)
                status.components[name] = "ready"
                status.enabled = True
            except Exception as exc:  # component-level degradation is intentional
                message = f"{type(exc).__name__}: {exc}"
                status.errors[name] = message
                status.components[name] = "unavailable"
                setattr(app.state, f"{name}_db", None)
                setattr(app.state, f"{name}_repository", None)
                setattr(app.state, f"{name}_service", None)
                logger.warning("Legacy %s research component unavailable: %s", name, message, exc_info=True)

        app.state.research_runtime = status
        yield status
    finally:
        for db in reversed(opened):
            await _close_db(db)
        # Restore the process-global legacy resolver so tests or another app
        # instance cannot inherit this app's path selection.
        try:
            paths_module._paths = previous_paths
        except Exception:
            logger.debug("Failed to restore legacy path resolver", exc_info=True)
        for name, *_ in _COMPONENTS:
            for suffix in ("service", "repository", "db"):
                setattr(app.state, f"{name}_{suffix}", None)
        app.state.research_runtime = None


def get_research_runtime_status(app: Any) -> ResearchRuntimeStatus:
    """Return status without forcing sidecar initialization."""
    value = getattr(app.state, "research_runtime_status", None)
    if isinstance(value, ResearchRuntimeStatus):
        return value
    return ResearchRuntimeStatus()
