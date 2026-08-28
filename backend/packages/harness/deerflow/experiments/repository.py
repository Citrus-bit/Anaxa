from __future__ import annotations

import json
from typing import Any

from deerflow.runtime.db import SQLiteRuntimeDB

from .types import (
    ExperimentArtifact,
    ExperimentFigureSpec,
    ExperimentProject,
    ExperimentRun,
)


def _to_json(value: Any) -> str:
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class ExperimentRepository:
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiment_projects (
                    project_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    dataset_ids_json TEXT NOT NULL,
                    linked_academic_project_id TEXT,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    method_key TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES experiment_projects(project_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_runs_project
                    ON experiment_runs(project_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS experiment_figures (
                    figure_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    chart_type TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    source_tables_json TEXT NOT NULL,
                    output_files_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_figures_run
                    ON experiment_figures(run_id);

                CREATE TABLE IF NOT EXISTS experiment_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    filepath TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES experiment_projects(project_id) ON DELETE CASCADE,
                    FOREIGN KEY(run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_artifacts_project
                    ON experiment_artifacts(project_id, created_at ASC);
                """
            )
            await self._db.conn.commit()

    async def create_project(self, project: ExperimentProject) -> ExperimentProject:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO experiment_projects (
                    project_id, thread_id, agent_name, domain, topic,
                    dataset_ids_json, linked_academic_project_id, status,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.project_id,
                    project.thread_id,
                    project.agent_name,
                    project.domain,
                    project.topic,
                    _to_json(project.dataset_ids),
                    project.linked_academic_project_id,
                    project.status,
                    _to_json(project.metadata),
                    project.created_at,
                    project.updated_at,
                ),
            )
            await self._db.conn.commit()
        created = await self.get_project(project.project_id)
        if created is None:
            raise RuntimeError(f"Failed to create experiment project {project.project_id}")
        return created

    async def update_project(self, project: ExperimentProject) -> ExperimentProject:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                UPDATE experiment_projects
                SET thread_id = ?, agent_name = ?, domain = ?, topic = ?,
                    dataset_ids_json = ?, linked_academic_project_id = ?,
                    status = ?, metadata_json = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (
                    project.thread_id,
                    project.agent_name,
                    project.domain,
                    project.topic,
                    _to_json(project.dataset_ids),
                    project.linked_academic_project_id,
                    project.status,
                    _to_json(project.metadata),
                    project.updated_at,
                    project.project_id,
                ),
            )
            await self._db.conn.commit()
        updated = await self.get_project(project.project_id)
        if updated is None:
            raise RuntimeError(f"Failed to update experiment project {project.project_id}")
        return updated

    async def get_project(self, project_id: str) -> ExperimentProject | None:
        cursor = await self._db.conn.execute(
            "SELECT * FROM experiment_projects WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ExperimentProject(
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            agent_name=row["agent_name"],
            domain=row["domain"],
            topic=row["topic"],
            dataset_ids=_from_json(row["dataset_ids_json"], []),
            linked_academic_project_id=row["linked_academic_project_id"],
            status=row["status"],
            metadata=_from_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def upsert_run(self, run: ExperimentRun) -> ExperimentRun:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_id, project_id, stage, status, method_key,
                    metrics_json, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    stage = excluded.stage,
                    status = excluded.status,
                    method_key = excluded.method_key,
                    metrics_json = excluded.metrics_json,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    run.run_id,
                    run.project_id,
                    run.stage,
                    run.status,
                    run.method_key,
                    _to_json(run.metrics_json),
                    run.notes,
                    run.created_at,
                    run.updated_at,
                ),
            )
            await self._db.conn.commit()
        stored = await self.get_run(run.run_id)
        if stored is None:
            raise RuntimeError(f"Failed to upsert experiment run {run.run_id}")
        return stored

    async def get_run(self, run_id: str) -> ExperimentRun | None:
        cursor = await self._db.conn.execute(
            "SELECT * FROM experiment_runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ExperimentRun(
            run_id=row["run_id"],
            project_id=row["project_id"],
            stage=row["stage"],
            status=row["status"],
            method_key=row["method_key"],
            metrics_json=_from_json(row["metrics_json"], {}),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def get_latest_run(self, project_id: str) -> ExperimentRun | None:
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM experiment_runs
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (project_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ExperimentRun(
            run_id=row["run_id"],
            project_id=row["project_id"],
            stage=row["stage"],
            status=row["status"],
            method_key=row["method_key"],
            metrics_json=_from_json(row["metrics_json"], {}),
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def count_runs(self, project_id: str) -> int:
        cursor = await self._db.conn.execute(
            "SELECT COUNT(*) AS count FROM experiment_runs WHERE project_id = ?",
            (project_id,),
        )
        row = await cursor.fetchone()
        return int(row["count"]) if row is not None else 0

    async def replace_figures(self, run_id: str, figures: list[ExperimentFigureSpec]) -> list[ExperimentFigureSpec]:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM experiment_figures WHERE run_id = ?", (run_id,))
            for figure in figures:
                await self._db.conn.execute(
                    """
                    INSERT INTO experiment_figures (
                        figure_id, run_id, intent, chart_type, grade,
                        source_tables_json, output_files_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        figure.figure_id,
                        figure.run_id,
                        figure.intent,
                        figure.chart_type,
                        figure.grade,
                        _to_json(figure.source_tables),
                        _to_json(figure.output_files),
                        _to_json(figure.metadata),
                    ),
                )
            await self._db.conn.commit()
        return await self.list_figures(run_id)

    async def list_figures(self, run_id: str) -> list[ExperimentFigureSpec]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM experiment_figures WHERE run_id = ? ORDER BY figure_id ASC",
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [
            ExperimentFigureSpec(
                figure_id=row["figure_id"],
                run_id=row["run_id"],
                intent=row["intent"],
                chart_type=row["chart_type"],
                grade=row["grade"],
                source_tables=_from_json(row["source_tables_json"], []),
                output_files=_from_json(row["output_files_json"], []),
                metadata=_from_json(row["metadata_json"], {}),
            )
            for row in rows
        ]

    async def replace_artifacts(self, project_id: str, run_id: str, artifacts: list[ExperimentArtifact]) -> list[ExperimentArtifact]:
        async with self._db.lock:
            await self._db.conn.execute(
                "DELETE FROM experiment_artifacts WHERE project_id = ?",
                (project_id,),
            )
            for artifact in artifacts:
                await self._db.conn.execute(
                    """
                    INSERT INTO experiment_artifacts (
                        artifact_id, project_id, run_id, filepath, artifact_type,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact.artifact_id,
                        artifact.project_id,
                        artifact.run_id,
                        artifact.filepath,
                        artifact.artifact_type,
                        _to_json(artifact.metadata),
                        artifact.created_at,
                    ),
                )
            await self._db.conn.commit()
        return await self.list_artifacts(project_id)

    async def list_artifacts(self, project_id: str) -> list[ExperimentArtifact]:
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM experiment_artifacts
            WHERE project_id = ?
            ORDER BY created_at ASC, filepath ASC
            """,
            (project_id,),
        )
        rows = await cursor.fetchall()
        return [
            ExperimentArtifact(
                artifact_id=row["artifact_id"],
                project_id=row["project_id"],
                run_id=row["run_id"],
                filepath=row["filepath"],
                artifact_type=row["artifact_type"],
                metadata=_from_json(row["metadata_json"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]
