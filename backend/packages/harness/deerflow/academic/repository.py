from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from deerflow.runtime.db import SQLiteRuntimeDB
from deerflow.runtime.utils import now_iso

from .types import (
    AcademicGraph,
    EvidenceCardRecord,
    OutlineNodeRecord,
    PaperAuthor,
    PaperEdgeRecord,
    PaperRecord,
    ReportExportRecord,
    ResearchProject,
    SearchQueryRecord,
)


class AcademicRepository:
    def __init__(self, db: SQLiteRuntimeDB) -> None:
        self._db = db

    async def _ensure_paper_columns(self) -> None:
        cursor = await self._db.conn.execute("PRAGMA table_info(papers)")
        rows = await cursor.fetchall()
        existing = {row["name"] for row in rows}
        migrations = [
            ("source_kind", "ALTER TABLE papers ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'metadata'"),
            ("venue_type", "ALTER TABLE papers ADD COLUMN venue_type TEXT NOT NULL DEFAULT 'unknown'"),
            ("venue_tier", "ALTER TABLE papers ADD COLUMN venue_tier TEXT NOT NULL DEFAULT 'unknown'"),
            ("publication_status", "ALTER TABLE papers ADD COLUMN publication_status TEXT NOT NULL DEFAULT 'unknown'"),
            ("is_preprint", "ALTER TABLE papers ADD COLUMN is_preprint INTEGER NOT NULL DEFAULT 0"),
            ("canonical_source", "ALTER TABLE papers ADD COLUMN canonical_source TEXT"),
            ("quality_signals_json", "ALTER TABLE papers ADD COLUMN quality_signals_json TEXT NOT NULL DEFAULT '{}'"),
        ]
        for column, statement in migrations:
            if column not in existing:
                await self._db.conn.execute(statement)

    async def setup(self) -> None:
        async with self._db.lock:
            await self._db.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_projects (
                    project_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    topic_key TEXT NOT NULL,
                    scope TEXT,
                    status TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_research_projects_thread_topic
                    ON research_projects(thread_id, topic_key);

                CREATE TABLE IF NOT EXISTS search_queries (
                    query_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    rationale TEXT,
                    query_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_search_queries_project_text
                    ON search_queries(project_id, query_text);

                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    canonical_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    year INTEGER,
                    venue TEXT,
                    abstract TEXT,
                    doi TEXT,
                    pmid TEXT,
                    pmcid TEXT,
                    arxiv_id TEXT,
                    cited_by_count INTEGER,
                    provider TEXT NOT NULL,
                    provider_id TEXT,
                    source_url TEXT,
                    oa_url TEXT,
                    metadata_only INTEGER NOT NULL DEFAULT 0,
                    source_kind TEXT NOT NULL DEFAULT 'metadata',
                    venue_type TEXT NOT NULL DEFAULT 'unknown',
                    venue_tier TEXT NOT NULL DEFAULT 'unknown',
                    publication_status TEXT NOT NULL DEFAULT 'unknown',
                    is_preprint INTEGER NOT NULL DEFAULT 0,
                    canonical_source TEXT,
                    quality_signals_json TEXT NOT NULL DEFAULT '{}',
                    keywords_json TEXT NOT NULL,
                    methods_json TEXT NOT NULL,
                    populations_json TEXT NOT NULL,
                    conflict_flags_json TEXT NOT NULL,
                    raw_source_json TEXT NOT NULL,
                    relevance_score REAL NOT NULL DEFAULT 0,
                    recency_score REAL NOT NULL DEFAULT 0,
                    completeness_score REAL NOT NULL DEFAULT 0,
                    rank_score REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_project_canonical
                    ON papers(project_id, canonical_id);

                CREATE TABLE IF NOT EXISTS paper_aliases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    alias_type TEXT NOT NULL,
                    alias_value TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_aliases_unique
                    ON paper_aliases(paper_id, alias_type, alias_value);

                CREATE TABLE IF NOT EXISTS paper_authors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    display_name TEXT NOT NULL,
                    given_name TEXT,
                    family_name TEXT,
                    orcid TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS paper_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_links_unique
                    ON paper_links(paper_id, link_type, url);

                CREATE TABLE IF NOT EXISTS paper_edges (
                    edge_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    source_paper_id TEXT,
                    relation_type TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_ref TEXT NOT NULL,
                    target_paper_id TEXT,
                    target_outline_node_id TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS evidence_cards (
                    evidence_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    claim_spans_json TEXT NOT NULL,
                    outline_node_ids_json TEXT NOT NULL,
                    relevance_score REAL NOT NULL DEFAULT 0,
                    recency_score REAL NOT NULL DEFAULT 0,
                    evidence_level TEXT NOT NULL,
                    method_tags_json TEXT NOT NULL,
                    novelty_tags_json TEXT NOT NULL,
                    source_snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS outline_nodes (
                    node_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    section_order INTEGER NOT NULL,
                    claim_spans_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS report_exports (
                    export_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    export_kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(project_id) REFERENCES research_projects(project_id) ON DELETE CASCADE
                );
                """
            )
            await self._ensure_paper_columns()
            await self._db.conn.commit()

    async def create_project(
        self,
        *,
        project_id: str,
        thread_id: str,
        topic: str,
        topic_key: str,
        scope: str | None,
        status: str,
        domain: str,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchProject:
        now = now_iso()
        async with self._db.lock:
            await self._db.conn.execute(
                """
                INSERT INTO research_projects (
                    project_id, thread_id, topic, topic_key, scope, status, domain,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    thread_id,
                    topic,
                    topic_key,
                    scope,
                    status,
                    domain,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            await self._db.conn.commit()
        created = await self.get_project(project_id)
        if created is None:
            raise RuntimeError(f"Failed to create research project {project_id}")
        return created

    async def find_project_by_thread_topic(self, thread_id: str, topic_key: str) -> ResearchProject | None:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                """
                SELECT * FROM research_projects
                WHERE thread_id = ? AND topic_key = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (thread_id, topic_key),
            )
            row = await cursor.fetchone()
        return self._project_from_row(row) if row is not None else None

    async def update_project_status(
        self,
        project_id: str,
        *,
        status: str,
        metadata: dict[str, Any] | None = None,
        domain: str | None = None,
    ) -> None:
        async with self._db.lock:
            existing = await self._db.conn.execute(
                "SELECT metadata_json, domain FROM research_projects WHERE project_id = ?",
                (project_id,),
            )
            row = await existing.fetchone()
            metadata_payload = metadata or {}
            if row is not None and row["metadata_json"]:
                prior = json.loads(row["metadata_json"])
                prior.update(metadata_payload)
                metadata_payload = prior
            await self._db.conn.execute(
                """
                UPDATE research_projects
                SET status = ?, domain = COALESCE(?, domain), metadata_json = ?, updated_at = ?
                WHERE project_id = ?
                """,
                (
                    status,
                    domain,
                    json.dumps(metadata_payload, ensure_ascii=False),
                    now_iso(),
                    project_id,
                ),
            )
            await self._db.conn.commit()

    async def get_project(self, project_id: str) -> ResearchProject | None:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM research_projects WHERE project_id = ?",
                (project_id,),
            )
            row = await cursor.fetchone()
        return self._project_from_row(row) if row is not None else None

    async def replace_search_queries(self, project_id: str, queries: list[SearchQueryRecord]) -> None:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM search_queries WHERE project_id = ?", (project_id,))
            await self._db.conn.executemany(
                """
                INSERT INTO search_queries (
                    query_id, project_id, query_text, rationale, query_type, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        query.query_id,
                        project_id,
                        query.query_text,
                        query.rationale,
                        query.query_type,
                        query.source,
                        query.created_at,
                    )
                    for query in queries
                ],
            )
            await self._db.conn.commit()

    async def list_search_queries(self, project_id: str) -> list[SearchQueryRecord]:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM search_queries WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [
            SearchQueryRecord(
                query_id=row["query_id"],
                project_id=row["project_id"],
                query_text=row["query_text"],
                rationale=row["rationale"],
                query_type=row["query_type"],
                source=row["source"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def upsert_papers(self, project_id: str, papers: list[PaperRecord]) -> list[PaperRecord]:
        if not papers:
            return []

        async with self._db.lock:
            for paper in papers:
                existing_cursor = await self._db.conn.execute(
                    "SELECT paper_id FROM papers WHERE project_id = ? AND canonical_id = ?",
                    (project_id, paper.canonical_id),
                )
                existing = await existing_cursor.fetchone()
                paper_id = existing["paper_id"] if existing is not None else paper.paper_id
                paper.paper_id = paper_id
                paper.project_id = project_id
                paper.updated_at = now_iso()
                if existing is None:
                    await self._db.conn.execute(
                        """
                        INSERT INTO papers (
                            paper_id, project_id, canonical_id, title, year, venue, abstract,
                            doi, pmid, pmcid, arxiv_id, cited_by_count, provider, provider_id,
                            source_url, oa_url, metadata_only, source_kind, venue_type, venue_tier,
                            publication_status, is_preprint, canonical_source, quality_signals_json, keywords_json, methods_json,
                            populations_json, conflict_flags_json, raw_source_json,
                            relevance_score, recency_score, completeness_score, rank_score,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        self._paper_to_row_tuple(paper),
                    )
                else:
                    await self._db.conn.execute(
                        """
                        UPDATE papers SET
                            title = ?, year = ?, venue = ?, abstract = ?, doi = ?, pmid = ?, pmcid = ?, arxiv_id = ?,
                            cited_by_count = ?, provider = ?, provider_id = ?, source_url = ?, oa_url = ?,
                            metadata_only = ?, source_kind = ?, venue_type = ?, venue_tier = ?,
                            publication_status = ?, is_preprint = ?, canonical_source = ?, quality_signals_json = ?,
                            keywords_json = ?, methods_json = ?, populations_json = ?,
                            conflict_flags_json = ?, raw_source_json = ?, relevance_score = ?, recency_score = ?,
                            completeness_score = ?, rank_score = ?, updated_at = ?
                        WHERE paper_id = ?
                        """,
                        (
                            paper.title,
                            paper.year,
                            paper.venue,
                            paper.abstract,
                            paper.doi,
                            paper.pmid,
                            paper.pmcid,
                            paper.arxiv_id,
                            paper.cited_by_count,
                            paper.provider,
                            paper.provider_id,
                            paper.source_url,
                            paper.oa_url,
                            int(paper.metadata_only),
                            paper.source_kind,
                            paper.venue_type,
                            paper.venue_tier,
                            paper.publication_status,
                            int(paper.is_preprint),
                            paper.canonical_source,
                            json.dumps(paper.quality_signals, ensure_ascii=False),
                            json.dumps(paper.keywords, ensure_ascii=False),
                            json.dumps(paper.methods, ensure_ascii=False),
                            json.dumps(paper.populations, ensure_ascii=False),
                            json.dumps(paper.conflict_flags, ensure_ascii=False),
                            json.dumps(paper.raw_source, ensure_ascii=False),
                            paper.relevance_score,
                            paper.recency_score,
                            paper.completeness_score,
                            paper.rank_score,
                            paper.updated_at,
                            paper_id,
                        ),
                    )

                await self._db.conn.execute("DELETE FROM paper_aliases WHERE paper_id = ?", (paper_id,))
                await self._db.conn.execute("DELETE FROM paper_authors WHERE paper_id = ?", (paper_id,))
                await self._db.conn.execute("DELETE FROM paper_links WHERE paper_id = ?", (paper_id,))

                aliases = [
                    ("doi", paper.doi),
                    ("pmid", paper.pmid),
                    ("pmcid", paper.pmcid),
                    ("arxiv_id", paper.arxiv_id),
                    ("provider_id", paper.provider_id),
                ]
                await self._db.conn.executemany(
                    "INSERT OR IGNORE INTO paper_aliases (paper_id, alias_type, alias_value) VALUES (?, ?, ?)",
                    [(paper_id, alias_type, alias_value) for alias_type, alias_value in aliases if alias_value],
                )

                await self._db.conn.executemany(
                    """
                    INSERT INTO paper_authors (paper_id, ordinal, display_name, given_name, family_name, orcid)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            paper_id,
                            author.ordinal,
                            author.display_name,
                            author.given_name,
                            author.family_name,
                            author.orcid,
                        )
                        for author in paper.authors
                    ],
                )

                links = [("source", paper.source_url), ("open_access", paper.oa_url)]
                await self._db.conn.executemany(
                    "INSERT OR IGNORE INTO paper_links (paper_id, link_type, url) VALUES (?, ?, ?)",
                    [(paper_id, link_type, url) for link_type, url in links if url],
                )

            await self._db.conn.commit()

        return await self.list_project_papers(project_id)

    async def list_project_papers(self, project_id: str) -> list[PaperRecord]:
        async with self._db.lock:
            paper_cursor = await self._db.conn.execute(
                "SELECT * FROM papers WHERE project_id = ? ORDER BY rank_score DESC, updated_at DESC",
                (project_id,),
            )
            paper_rows = await paper_cursor.fetchall()
            author_cursor = await self._db.conn.execute(
                """
                SELECT paper_id, ordinal, display_name, given_name, family_name, orcid
                FROM paper_authors
                WHERE paper_id IN (SELECT paper_id FROM papers WHERE project_id = ?)
                ORDER BY paper_id, ordinal ASC
                """,
                (project_id,),
            )
            author_rows = await author_cursor.fetchall()

        authors_by_paper: dict[str, list[PaperAuthor]] = defaultdict(list)
        for row in author_rows:
            authors_by_paper[row["paper_id"]].append(
                PaperAuthor(
                    display_name=row["display_name"],
                    given_name=row["given_name"],
                    family_name=row["family_name"],
                    orcid=row["orcid"],
                    ordinal=row["ordinal"],
                )
            )

        return [self._paper_from_row(row, authors_by_paper[row["paper_id"]]) for row in paper_rows]

    async def replace_outline_nodes(self, project_id: str, nodes: list[OutlineNodeRecord]) -> None:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM outline_nodes WHERE project_id = ?", (project_id,))
            await self._db.conn.executemany(
                """
                INSERT INTO outline_nodes (
                    node_id, project_id, title, purpose, section_order, claim_spans_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.node_id,
                        project_id,
                        node.title,
                        node.purpose,
                        node.section_order,
                        json.dumps(node.claim_spans, ensure_ascii=False),
                        json.dumps(node.metadata, ensure_ascii=False),
                        node.created_at,
                        node.updated_at,
                    )
                    for node in nodes
                ],
            )
            await self._db.conn.commit()

    async def list_outline_nodes(self, project_id: str) -> list[OutlineNodeRecord]:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM outline_nodes WHERE project_id = ? ORDER BY section_order ASC",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [
            OutlineNodeRecord(
                node_id=row["node_id"],
                project_id=row["project_id"],
                title=row["title"],
                purpose=row["purpose"],
                section_order=row["section_order"],
                claim_spans=json.loads(row["claim_spans_json"] or "[]"),
                metadata=json.loads(row["metadata_json"] or "{}"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def replace_evidence_cards(self, project_id: str, cards: list[EvidenceCardRecord]) -> None:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM evidence_cards WHERE project_id = ?", (project_id,))
            await self._db.conn.executemany(
                """
                INSERT INTO evidence_cards (
                    evidence_id, project_id, paper_id, summary, claim_spans_json, outline_node_ids_json,
                    relevance_score, recency_score, evidence_level, method_tags_json, novelty_tags_json,
                    source_snapshot_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        card.evidence_id,
                        project_id,
                        card.paper_id,
                        card.summary,
                        json.dumps(card.claim_spans, ensure_ascii=False),
                        json.dumps(card.outline_node_ids, ensure_ascii=False),
                        card.relevance_score,
                        card.recency_score,
                        card.evidence_level,
                        json.dumps(card.method_tags, ensure_ascii=False),
                        json.dumps(card.novelty_tags, ensure_ascii=False),
                        json.dumps(card.source_snapshot, ensure_ascii=False),
                        card.created_at,
                        card.updated_at,
                    )
                    for card in cards
                ],
            )
            await self._db.conn.commit()

    async def list_evidence_cards(self, project_id: str) -> list[EvidenceCardRecord]:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM evidence_cards WHERE project_id = ? ORDER BY relevance_score DESC, recency_score DESC",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [
            EvidenceCardRecord(
                evidence_id=row["evidence_id"],
                project_id=row["project_id"],
                paper_id=row["paper_id"],
                summary=row["summary"],
                claim_spans=json.loads(row["claim_spans_json"] or "[]"),
                outline_node_ids=json.loads(row["outline_node_ids_json"] or "[]"),
                relevance_score=row["relevance_score"],
                recency_score=row["recency_score"],
                evidence_level=row["evidence_level"],
                method_tags=json.loads(row["method_tags_json"] or "[]"),
                novelty_tags=json.loads(row["novelty_tags_json"] or "[]"),
                source_snapshot=json.loads(row["source_snapshot_json"] or "{}"),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def replace_paper_edges(self, project_id: str, edges: list[PaperEdgeRecord]) -> None:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM paper_edges WHERE project_id = ?", (project_id,))
            await self._db.conn.executemany(
                """
                INSERT INTO paper_edges (
                    edge_id, project_id, source_paper_id, relation_type, target_kind, target_ref,
                    target_paper_id, target_outline_node_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.edge_id,
                        project_id,
                        edge.source_paper_id,
                        edge.relation_type,
                        edge.target_kind,
                        edge.target_ref,
                        edge.target_paper_id,
                        edge.target_outline_node_id,
                        json.dumps(edge.metadata, ensure_ascii=False),
                        edge.created_at,
                    )
                    for edge in edges
                ],
            )
            await self._db.conn.commit()

    async def get_graph(self, project_id: str) -> AcademicGraph:
        papers = await self.list_project_papers(project_id)
        outline = await self.list_outline_nodes(project_id)
        evidence = await self.list_evidence_cards(project_id)
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM paper_edges WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,),
            )
            edge_rows = await cursor.fetchall()

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        for paper in papers:
            nodes.append(
                {
                    "id": paper.paper_id,
                    "type": "paper",
                    "label": paper.title,
                    "provider": paper.provider,
                    "year": paper.year,
                }
            )
            for author in paper.authors:
                author_id = f"author:{author.display_name}"
                nodes.append({"id": author_id, "type": "author", "label": author.display_name})
                edges.append(
                    {
                        "id": f"authored:{paper.paper_id}:{author_id}",
                        "source": paper.paper_id,
                        "target": author_id,
                        "relation": "authored_by",
                    }
                )

        for node in outline:
            nodes.append({"id": node.node_id, "type": "outline", "label": node.title})

        for card in evidence:
            card_id = f"evidence:{card.evidence_id}"
            nodes.append({"id": card_id, "type": "evidence", "label": card.summary[:80]})
            edges.append(
                {
                    "id": f"supports:{card.paper_id}:{card_id}",
                    "source": card.paper_id,
                    "target": card_id,
                    "relation": "supports",
                }
            )
            for outline_node_id in card.outline_node_ids:
                edges.append(
                    {
                        "id": f"belongs:{card.evidence_id}:{outline_node_id}",
                        "source": card_id,
                        "target": outline_node_id,
                        "relation": "belongs_to",
                    }
                )

        for row in edge_rows:
            edges.append(
                {
                    "id": row["edge_id"],
                    "source": row["source_paper_id"],
                    "target": row["target_paper_id"] or row["target_outline_node_id"] or row["target_ref"],
                    "relation": row["relation_type"],
                    "target_kind": row["target_kind"],
                }
            )

        deduped_nodes = list({node["id"]: node for node in nodes}.values())
        deduped_edges = list({edge["id"]: edge for edge in edges if edge.get("source") and edge.get("target")}.values())
        return AcademicGraph(project_id=project_id, nodes=deduped_nodes, edges=deduped_edges)

    async def replace_report_exports(self, project_id: str, exports: list[ReportExportRecord]) -> None:
        async with self._db.lock:
            await self._db.conn.execute("DELETE FROM report_exports WHERE project_id = ?", (project_id,))
            await self._db.conn.executemany(
                """
                INSERT INTO report_exports (export_id, project_id, export_kind, path, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.export_id,
                        project_id,
                        item.export_kind,
                        item.path,
                        json.dumps(item.summary, ensure_ascii=False),
                        item.created_at,
                    )
                    for item in exports
                ],
            )
            await self._db.conn.commit()

    async def list_report_exports(self, project_id: str) -> list[ReportExportRecord]:
        async with self._db.lock:
            cursor = await self._db.conn.execute(
                "SELECT * FROM report_exports WHERE project_id = ? ORDER BY created_at ASC",
                (project_id,),
            )
            rows = await cursor.fetchall()
        return [
            ReportExportRecord(
                export_id=row["export_id"],
                project_id=row["project_id"],
                export_kind=row["export_kind"],
                path=row["path"],
                summary=json.loads(row["summary_json"] or "{}"),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _project_from_row(row) -> ResearchProject:
        return ResearchProject(
            project_id=row["project_id"],
            thread_id=row["thread_id"],
            topic=row["topic"],
            topic_key=row["topic_key"],
            scope=row["scope"],
            status=row["status"],
            domain=row["domain"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _paper_to_row_tuple(paper: PaperRecord) -> tuple[Any, ...]:
        return (
            paper.paper_id,
            paper.project_id,
            paper.canonical_id,
            paper.title,
            paper.year,
            paper.venue,
            paper.abstract,
            paper.doi,
            paper.pmid,
            paper.pmcid,
            paper.arxiv_id,
            paper.cited_by_count,
            paper.provider,
            paper.provider_id,
            paper.source_url,
            paper.oa_url,
            int(paper.metadata_only),
            paper.source_kind,
            paper.venue_type,
            paper.venue_tier,
            paper.publication_status,
            int(paper.is_preprint),
            paper.canonical_source,
            json.dumps(paper.quality_signals, ensure_ascii=False),
            json.dumps(paper.keywords, ensure_ascii=False),
            json.dumps(paper.methods, ensure_ascii=False),
            json.dumps(paper.populations, ensure_ascii=False),
            json.dumps(paper.conflict_flags, ensure_ascii=False),
            json.dumps(paper.raw_source, ensure_ascii=False),
            paper.relevance_score,
            paper.recency_score,
            paper.completeness_score,
            paper.rank_score,
            paper.created_at,
            paper.updated_at,
        )

    @staticmethod
    def _paper_from_row(row, authors: list[PaperAuthor]) -> PaperRecord:
        return PaperRecord(
            paper_id=row["paper_id"],
            project_id=row["project_id"],
            canonical_id=row["canonical_id"],
            title=row["title"],
            authors=authors,
            year=row["year"],
            venue=row["venue"],
            abstract=row["abstract"],
            doi=row["doi"],
            pmid=row["pmid"],
            pmcid=row["pmcid"],
            arxiv_id=row["arxiv_id"],
            cited_by_count=row["cited_by_count"],
            provider=row["provider"],
            provider_id=row["provider_id"],
            source_url=row["source_url"],
            oa_url=row["oa_url"],
            metadata_only=bool(row["metadata_only"]),
            source_kind=row["source_kind"],
            venue_type=row["venue_type"],
            venue_tier=row["venue_tier"],
            publication_status=row["publication_status"],
            is_preprint=bool(row["is_preprint"]),
            canonical_source=row["canonical_source"],
            quality_signals=json.loads(row["quality_signals_json"] or "{}"),
            keywords=json.loads(row["keywords_json"] or "[]"),
            methods=json.loads(row["methods_json"] or "[]"),
            populations=json.loads(row["populations_json"] or "[]"),
            conflict_flags=json.loads(row["conflict_flags_json"] or "[]"),
            raw_source=json.loads(row["raw_source_json"] or "{}"),
            relevance_score=row["relevance_score"],
            recency_score=row["recency_score"],
            completeness_score=row["completeness_score"],
            rank_score=row["rank_score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
