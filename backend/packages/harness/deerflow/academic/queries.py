from __future__ import annotations

import uuid

from deerflow.runtime.utils import now_iso

from .types import ResearchProject, SearchQueryRecord
from .utils import detect_domain


def build_query_expansions(project: ResearchProject) -> list[SearchQueryRecord]:
    topic = project.topic.strip()
    scoped_topic = f"{topic} {project.scope}".strip() if project.scope else topic
    domain = project.domain or detect_domain(topic, project.scope)

    templates: list[tuple[str, str, str]] = [
        (topic, "core", "Direct topic query for baseline coverage."),
        (f"{scoped_topic} review", "review", "Find broad survey and review papers."),
        (f"{scoped_topic} systematic review", "review", "Capture evidence synthesis and literature surveys."),
        (f"{scoped_topic} methods", "methods", "Surface methodological approaches."),
        (f"{scoped_topic} applications", "applications", "Find applied or empirical usage scenarios."),
        (f"{scoped_topic} limitations", "limitations", "Collect reported limitations and failure cases."),
        (f"{scoped_topic} controversy", "controversy", "Look for conflicting claims and unresolved debates."),
        (f"{scoped_topic} future directions", "future", "Capture explicit research gaps and future work."),
    ]

    if domain == "biomedical":
        templates.extend(
            [
                (f"{scoped_topic} clinical trial", "methods", "Biomedical enhancement for clinical evidence."),
                (f"{scoped_topic} patient cohort", "applications", "Biomedical enhancement for patient populations."),
            ]
        )
    elif domain == "cs_ai":
        templates.extend(
            [
                (f"{scoped_topic} benchmark dataset", "methods", "CS/AI enhancement for benchmarks and datasets."),
                (f"{scoped_topic} evaluation framework", "methods", "CS/AI enhancement for evaluation protocols."),
            ]
        )

    deduped: list[SearchQueryRecord] = []
    seen: set[str] = set()
    created_at = now_iso()
    for query_text, query_type, rationale in templates:
        normalized = " ".join(query_text.split())
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        deduped.append(
            SearchQueryRecord(
                query_id=str(uuid.uuid4()),
                project_id=project.project_id,
                query_text=normalized,
                rationale=rationale,
                query_type=query_type,
                source="planner",
                created_at=created_at,
            )
        )

    return deduped[:10]
