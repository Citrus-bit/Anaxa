from __future__ import annotations

from medrix_flow.research.quality import build_quality_audit
from medrix_flow.research.types import ClaimEvidenceRecord, ManuscriptSectionRecord


def test_research_quality_audit_detects_review_weaknesses():
    evidence = [
        ClaimEvidenceRecord(
            claim_id="claim-1",
            quest_id="rq-quality",
            claim="Representation is not enough for reliable evaluation.",
            paper_id="paper-1",
            source_title="Evaluation Systems",
            snippet="A benchmark dataset reports empirical result gaps.",
            support_status="supported",
            confidence=0.8,
            metadata={"citation_key": "smith2024", "evidence_type": "benchmark"},
            created_at="2026-05-10T00:00:00Z",
        ),
        ClaimEvidenceRecord(
            claim_id="claim-2",
            quest_id="rq-quality",
            claim="The framework is universally superior.",
            support_status="unsupported",
            confidence=0.1,
            metadata={},
            created_at="2026-05-10T00:00:00Z",
        ),
    ]
    sections = [
        ManuscriptSectionRecord(
            section_id="section-1",
            quest_id="rq-quality",
            section_key="intro",
            title="Introduction",
            content=("Representation is not enough for reliable systems \\cite{smith2024}.\n\nRepresentation is not enough, and all systems must support every workflow. bibliography keys are synchronized."),
            claim_ids=["claim-1", "claim-2"],
            created_at="2026-05-10T00:00:00Z",
            updated_at="2026-05-10T00:00:00Z",
        )
    ]

    audit = build_quality_audit(
        audit_id="qa-1",
        quest_id="rq-quality",
        stage="review",
        topic="reliable evaluation systems",
        evidence=evidence,
        sections=sections,
        reference_count=12,
        cited_reference_count=1,
        created_at="2026-05-10T00:00:00Z",
    )

    assert audit.status == "block"
    assert audit.metrics["reference_count"] == 12
    assert audit.metrics["unsupported_claim_count"] == 1
    assert audit.metrics["quantitative_evidence_count"] == 1
    assert audit.metrics["repeated_phrase_count"] >= 1
    assert audit.metrics["absolute_phrase_count"] >= 1
    assert audit.metrics["author_note_count"] == 1
    assert any("Expand scholarly search" in action for action in audit.repair_actions)
