from __future__ import annotations

import json
from typing import Any

from deerflow.runtime.db import SQLiteRuntimeDB

from .types import (
    ClaimEvidenceRecord,
    ExperimentBranchRecord,
    ManuscriptSectionRecord,
    NoveltyCheckRecord,
    ResearchGate,
    ResearchLedgerEntry,
    ResearchQualityAuditRecord,
    ResearchQuest,
    ReviewerReportRecord,
)


def _to_json(value: Any) -> str:
    if value is None:
        value = {}
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _from_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


class ResearchRepository:
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_quests (
                    quest_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    scope TEXT,
                    objective TEXT,
                    domain TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    academic_project_id TEXT,
                    experiment_project_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_research_quests_thread_updated
                    ON research_quests(thread_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS research_ledger (
                    entry_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    inputs_json TEXT NOT NULL,
                    outputs_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    tool_name TEXT,
                    model_name TEXT,
                    error TEXT,
                    gate_decision TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_ledger_quest_created
                    ON research_ledger(quest_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS research_gates (
                    gate_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    gate_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision TEXT,
                    reason TEXT,
                    required INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_research_gates_unique
                    ON research_gates(quest_id, stage, gate_type);

                CREATE TABLE IF NOT EXISTS research_claim_evidence (
                    claim_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    paper_id TEXT,
                    source_title TEXT,
                    locator TEXT,
                    snippet TEXT,
                    quote_text TEXT,
                    support_status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    artifact_path TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_claim_evidence_quest
                    ON research_claim_evidence(quest_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS research_novelty_checks (
                    check_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    idea TEXT NOT NULL,
                    overlap_risk TEXT NOT NULL,
                    closest_papers_json TEXT NOT NULL,
                    hypotheses_json TEXT NOT NULL,
                    minimum_experiment TEXT,
                    decision TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_novelty_quest
                    ON research_novelty_checks(quest_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS research_experiment_branches (
                    branch_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    experiment_project_id TEXT,
                    parent_branch_id TEXT,
                    name TEXT NOT NULL,
                    branch_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    seed INTEGER,
                    metrics_json TEXT NOT NULL,
                    artifact_paths_json TEXT NOT NULL,
                    failure_summary TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_branches_quest_priority
                    ON research_experiment_branches(quest_id, priority DESC);

                CREATE TABLE IF NOT EXISTS research_manuscript_sections (
                    section_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    section_key TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    claim_ids_json TEXT NOT NULL,
                    artifact_paths_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_research_sections_unique
                    ON research_manuscript_sections(quest_id, section_key);

                CREATE TABLE IF NOT EXISTS research_reviewer_reports (
                    report_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    reviewer_profile TEXT NOT NULL,
                    score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    required_actions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_reviews_quest
                    ON research_reviewer_reports(quest_id, created_at ASC);

                CREATE TABLE IF NOT EXISTS research_quality_audits (
                    audit_id TEXT PRIMARY KEY,
                    quest_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL NOT NULL,
                    metrics_json TEXT NOT NULL,
                    findings_json TEXT NOT NULL,
                    repair_actions_json TEXT NOT NULL,
                    recommended_queries_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(quest_id) REFERENCES research_quests(quest_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_quality_audits_quest
                    ON research_quality_audits(quest_id, created_at ASC);
                """
            )
            await self._db.conn.commit()

    async def create_quest(self, quest: ResearchQuest) -> ResearchQuest:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_quests (
                    quest_id, thread_id, title, topic, scope, objective, domain,
                    stage, status, academic_project_id, experiment_project_ids_json,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quest.quest_id,
                    quest.thread_id,
                    quest.title,
                    quest.topic,
                    quest.scope,
                    quest.objective,
                    quest.domain,
                    quest.stage,
                    quest.status,
                    quest.academic_project_id,
                    _to_json(quest.experiment_project_ids),
                    _to_json(quest.metadata),
                    quest.created_at,
                    quest.updated_at,
                ),
            )
            await self._db.conn.commit()
        created = await self.get_quest(quest.quest_id)
        if created is None:
            raise RuntimeError(f"Failed to create research quest {quest.quest_id}")
        return created

    async def update_quest(self, quest: ResearchQuest) -> ResearchQuest:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                UPDATE research_quests
                SET thread_id = ?, title = ?, topic = ?, scope = ?, objective = ?,
                    domain = ?, stage = ?, status = ?, academic_project_id = ?,
                    experiment_project_ids_json = ?, metadata_json = ?, updated_at = ?
                WHERE quest_id = ?
                """,
                (
                    quest.thread_id,
                    quest.title,
                    quest.topic,
                    quest.scope,
                    quest.objective,
                    quest.domain,
                    quest.stage,
                    quest.status,
                    quest.academic_project_id,
                    _to_json(quest.experiment_project_ids),
                    _to_json(quest.metadata),
                    quest.updated_at,
                    quest.quest_id,
                ),
            )
            await self._db.conn.commit()
        updated = await self.get_quest(quest.quest_id)
        if updated is None:
            raise RuntimeError(f"Failed to update research quest {quest.quest_id}")
        return updated

    async def get_quest(self, quest_id: str) -> ResearchQuest | None:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_quests WHERE quest_id = ?",
            (quest_id,),
        )
        row = await cursor.fetchone()
        return self._quest_from_row(row) if row is not None else None

    async def list_quests(self, thread_id: str | None = None) -> list[ResearchQuest]:
        if thread_id:
            cursor = await self._db.conn.execute(
                """
                SELECT * FROM research_quests
                WHERE thread_id = ?
                ORDER BY updated_at DESC
                """,
                (thread_id,),
            )
        else:
            cursor = await self._db.conn.execute("SELECT * FROM research_quests ORDER BY updated_at DESC LIMIT 100")
        rows = await cursor.fetchall()
        return [self._quest_from_row(row) for row in rows]

    async def add_ledger_entry(self, entry: ResearchLedgerEntry) -> ResearchLedgerEntry:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_ledger (
                    entry_id, quest_id, stage, event_type, summary, inputs_json,
                    outputs_json, artifacts_json, tool_name, model_name, error,
                    gate_decision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.quest_id,
                    entry.stage,
                    entry.event_type,
                    entry.summary,
                    _to_json(entry.inputs),
                    _to_json(entry.outputs),
                    _to_json(entry.artifacts),
                    entry.tool_name,
                    entry.model_name,
                    entry.error,
                    entry.gate_decision,
                    entry.created_at,
                ),
            )
            await self._db.conn.commit()
        return entry

    async def list_ledger(self, quest_id: str) -> list[ResearchLedgerEntry]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_ledger WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ResearchLedgerEntry(
                entry_id=row["entry_id"],
                quest_id=row["quest_id"],
                stage=row["stage"],
                event_type=row["event_type"],
                summary=row["summary"],
                inputs=_from_json(row["inputs_json"], {}),
                outputs=_from_json(row["outputs_json"], {}),
                artifacts=_from_json(row["artifacts_json"], []),
                tool_name=row["tool_name"],
                model_name=row["model_name"],
                error=row["error"],
                gate_decision=row["gate_decision"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def upsert_gate(self, gate: ResearchGate) -> ResearchGate:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_gates (
                    gate_id, quest_id, stage, gate_type, status, decision,
                    reason, required, created_at, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(quest_id, stage, gate_type) DO UPDATE SET
                    status = excluded.status,
                    decision = excluded.decision,
                    reason = excluded.reason,
                    required = excluded.required,
                    decided_at = excluded.decided_at
                """,
                (
                    gate.gate_id,
                    gate.quest_id,
                    gate.stage,
                    gate.gate_type,
                    gate.status,
                    gate.decision,
                    gate.reason,
                    int(gate.required),
                    gate.created_at,
                    gate.decided_at,
                ),
            )
            await self._db.conn.commit()
        stored = await self.get_gate(gate.quest_id, gate.stage, gate.gate_type)
        if stored is None:
            raise RuntimeError(f"Failed to upsert research gate {gate.gate_id}")
        return stored

    async def get_gate(self, quest_id: str, stage: str, gate_type: str) -> ResearchGate | None:
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM research_gates
            WHERE quest_id = ? AND stage = ? AND gate_type = ?
            """,
            (quest_id, stage, gate_type),
        )
        row = await cursor.fetchone()
        return self._gate_from_row(row) if row is not None else None

    async def list_gates(self, quest_id: str) -> list[ResearchGate]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_gates WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [self._gate_from_row(row) for row in rows]

    async def add_claim_evidence(self, record: ClaimEvidenceRecord) -> ClaimEvidenceRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_claim_evidence (
                    claim_id, quest_id, claim, paper_id, source_title, locator,
                    snippet, quote_text, support_status, confidence, artifact_path,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.claim_id,
                    record.quest_id,
                    record.claim,
                    record.paper_id,
                    record.source_title,
                    record.locator,
                    record.snippet,
                    record.quote,
                    record.support_status,
                    record.confidence,
                    record.artifact_path,
                    _to_json(record.metadata),
                    record.created_at,
                ),
            )
            await self._db.conn.commit()
        return record

    async def list_claim_evidence(self, quest_id: str) -> list[ClaimEvidenceRecord]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_claim_evidence WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ClaimEvidenceRecord(
                claim_id=row["claim_id"],
                quest_id=row["quest_id"],
                claim=row["claim"],
                paper_id=row["paper_id"],
                source_title=row["source_title"],
                locator=row["locator"],
                snippet=row["snippet"],
                quote=row["quote_text"],
                support_status=row["support_status"],
                confidence=float(row["confidence"]),
                artifact_path=row["artifact_path"],
                metadata=_from_json(row["metadata_json"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def add_novelty_check(self, record: NoveltyCheckRecord) -> NoveltyCheckRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_novelty_checks (
                    check_id, quest_id, idea, overlap_risk, closest_papers_json,
                    hypotheses_json, minimum_experiment, decision, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.check_id,
                    record.quest_id,
                    record.idea,
                    record.overlap_risk,
                    _to_json(record.closest_papers),
                    _to_json(record.hypotheses),
                    record.minimum_experiment,
                    record.decision,
                    record.created_at,
                ),
            )
            await self._db.conn.commit()
        return record

    async def list_novelty_checks(self, quest_id: str) -> list[NoveltyCheckRecord]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_novelty_checks WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            NoveltyCheckRecord(
                check_id=row["check_id"],
                quest_id=row["quest_id"],
                idea=row["idea"],
                overlap_risk=row["overlap_risk"],
                closest_papers=_from_json(row["closest_papers_json"], []),
                hypotheses=_from_json(row["hypotheses_json"], []),
                minimum_experiment=row["minimum_experiment"],
                decision=row["decision"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def upsert_experiment_branch(self, record: ExperimentBranchRecord) -> ExperimentBranchRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_experiment_branches (
                    branch_id, quest_id, experiment_project_id, parent_branch_id,
                    name, branch_type, status, priority, seed, metrics_json,
                    artifact_paths_json, failure_summary, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(branch_id) DO UPDATE SET
                    experiment_project_id = excluded.experiment_project_id,
                    parent_branch_id = excluded.parent_branch_id,
                    name = excluded.name,
                    branch_type = excluded.branch_type,
                    status = excluded.status,
                    priority = excluded.priority,
                    seed = excluded.seed,
                    metrics_json = excluded.metrics_json,
                    artifact_paths_json = excluded.artifact_paths_json,
                    failure_summary = excluded.failure_summary,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.branch_id,
                    record.quest_id,
                    record.experiment_project_id,
                    record.parent_branch_id,
                    record.name,
                    record.branch_type,
                    record.status,
                    record.priority,
                    record.seed,
                    _to_json(record.metrics),
                    _to_json(record.artifact_paths),
                    record.failure_summary,
                    _to_json(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
            await self._db.conn.commit()
        branches = await self.list_experiment_branches(record.quest_id)
        return next(branch for branch in branches if branch.branch_id == record.branch_id)

    async def list_experiment_branches(self, quest_id: str) -> list[ExperimentBranchRecord]:
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM research_experiment_branches
            WHERE quest_id = ?
            ORDER BY priority DESC, created_at ASC
            """,
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ExperimentBranchRecord(
                branch_id=row["branch_id"],
                quest_id=row["quest_id"],
                experiment_project_id=row["experiment_project_id"],
                parent_branch_id=row["parent_branch_id"],
                name=row["name"],
                branch_type=row["branch_type"],
                status=row["status"],
                priority=float(row["priority"]),
                seed=row["seed"],
                metrics=_from_json(row["metrics_json"], {}),
                artifact_paths=_from_json(row["artifact_paths_json"], []),
                failure_summary=row["failure_summary"],
                metadata=_from_json(row["metadata_json"], {}),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def upsert_manuscript_section(self, record: ManuscriptSectionRecord) -> ManuscriptSectionRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_manuscript_sections (
                    section_id, quest_id, section_key, title, content,
                    claim_ids_json, artifact_paths_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(quest_id, section_key) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    claim_ids_json = excluded.claim_ids_json,
                    artifact_paths_json = excluded.artifact_paths_json,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    record.section_id,
                    record.quest_id,
                    record.section_key,
                    record.title,
                    record.content,
                    _to_json(record.claim_ids),
                    _to_json(record.artifact_paths),
                    record.status,
                    record.created_at,
                    record.updated_at,
                ),
            )
            await self._db.conn.commit()
        sections = await self.list_manuscript_sections(record.quest_id)
        return next(section for section in sections if section.section_key == record.section_key)

    async def list_manuscript_sections(self, quest_id: str) -> list[ManuscriptSectionRecord]:
        cursor = await self._db.conn.execute(
            """
            SELECT * FROM research_manuscript_sections
            WHERE quest_id = ?
            ORDER BY created_at ASC
            """,
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ManuscriptSectionRecord(
                section_id=row["section_id"],
                quest_id=row["quest_id"],
                section_key=row["section_key"],
                title=row["title"],
                content=row["content"],
                claim_ids=_from_json(row["claim_ids_json"], []),
                artifact_paths=_from_json(row["artifact_paths_json"], []),
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def add_reviewer_report(self, record: ReviewerReportRecord) -> ReviewerReportRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_reviewer_reports (
                    report_id, quest_id, stage, reviewer_profile, score, verdict,
                    findings_json, required_actions_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.report_id,
                    record.quest_id,
                    record.stage,
                    record.reviewer_profile,
                    record.score,
                    record.verdict,
                    _to_json(record.findings),
                    _to_json(record.required_actions),
                    record.created_at,
                ),
            )
            await self._db.conn.commit()
        return record

    async def add_quality_audit(self, record: ResearchQualityAuditRecord) -> ResearchQualityAuditRecord:
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_quality_audits (
                    audit_id, quest_id, stage, status, score, metrics_json,
                    findings_json, repair_actions_json, recommended_queries_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.audit_id,
                    record.quest_id,
                    record.stage,
                    record.status,
                    record.score,
                    _to_json(record.metrics),
                    _to_json(record.findings),
                    _to_json(record.repair_actions),
                    _to_json(record.recommended_queries),
                    record.created_at,
                ),
            )
            await self._db.conn.commit()
        return record

    async def list_quality_audits(self, quest_id: str) -> list[ResearchQualityAuditRecord]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_quality_audits WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ResearchQualityAuditRecord(
                audit_id=row["audit_id"],
                quest_id=row["quest_id"],
                stage=row["stage"],
                status=row["status"],
                score=float(row["score"]),
                metrics=_from_json(row["metrics_json"], {}),
                findings=_from_json(row["findings_json"], []),
                repair_actions=_from_json(row["repair_actions_json"], []),
                recommended_queries=_from_json(row["recommended_queries_json"], []),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def list_reviewer_reports(self, quest_id: str) -> list[ReviewerReportRecord]:
        cursor = await self._db.conn.execute(
            "SELECT * FROM research_reviewer_reports WHERE quest_id = ? ORDER BY created_at ASC",
            (quest_id,),
        )
        rows = await cursor.fetchall()
        return [
            ReviewerReportRecord(
                report_id=row["report_id"],
                quest_id=row["quest_id"],
                stage=row["stage"],
                reviewer_profile=row["reviewer_profile"],
                score=float(row["score"]),
                verdict=row["verdict"],
                findings=_from_json(row["findings_json"], []),
                required_actions=_from_json(row["required_actions_json"], []),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _quest_from_row(row: Any) -> ResearchQuest:
        return ResearchQuest(
            quest_id=row["quest_id"],
            thread_id=row["thread_id"],
            title=row["title"],
            topic=row["topic"],
            scope=row["scope"],
            objective=row["objective"],
            domain=row["domain"],
            stage=row["stage"],
            status=row["status"],
            academic_project_id=row["academic_project_id"],
            experiment_project_ids=_from_json(row["experiment_project_ids_json"], []),
            metadata=_from_json(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _gate_from_row(row: Any) -> ResearchGate:
        return ResearchGate(
            gate_id=row["gate_id"],
            quest_id=row["quest_id"],
            stage=row["stage"],
            gate_type=row["gate_type"],
            status=row["status"],
            decision=row["decision"],
            reason=row["reason"],
            required=bool(row["required"]),
            created_at=row["created_at"],
            decided_at=row["decided_at"],
        )
