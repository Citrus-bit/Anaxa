"""Compatibility checks for Anaxa research modules hosted by DeerFlow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from deerflow.config.paths import Paths
from deerflow.runtime.utils import now_iso


def test_research_runtime_compatibility_paths(tmp_path: Path) -> None:
    paths = Paths(tmp_path)

    assert paths.runtime_db_file == tmp_path / "runtime.sqlite3"
    assert paths.academic_db_file == tmp_path / "academic.sqlite3"
    assert paths.experiment_db_file == tmp_path / "experiment.sqlite3"
    assert paths.research_db_file == tmp_path / "research.sqlite3"
    assert paths.thread_memory_file("thread-1") == tmp_path / "threads" / "thread-1" / "memory.json"
    assert paths.thread_memory_file("thread-1", user_id="user-1") == tmp_path / "users" / "user-1" / "threads" / "thread-1" / "memory.json"
    assert paths.sandbox_workspace_dir("thread-1", user_id="user-1") == paths.sandbox_work_dir("thread-1", user_id="user-1")
    assert paths.sandbox_outputs_dir("thread-1", user_id="user-1") == tmp_path / "users" / "user-1" / "threads" / "thread-1" / "user-data" / "outputs"


def test_research_runtime_now_iso_is_timezone_aware() -> None:
    timestamp = datetime.fromisoformat(now_iso())

    assert timestamp.tzinfo is not None


def test_research_domain_modules_import() -> None:
    from deerflow.academic import AcademicResearchService
    from deerflow.experiments import ExperimentService
    from deerflow.research import ResearchQuestService

    assert AcademicResearchService.__name__ == "AcademicResearchService"
    assert ExperimentService.__name__ == "ExperimentService"
    assert ResearchQuestService.__name__ == "ResearchQuestService"
