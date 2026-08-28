from __future__ import annotations

import re
from typing import Any

from .types import (
    ClaimEvidenceRecord,
    ManuscriptSectionRecord,
    QualityAuditStatus,
    ResearchQualityAuditRecord,
    ResearchStage,
)

QUANTITATIVE_EVIDENCE_TERMS = {
    "ablation",
    "accuracy",
    "auc",
    "baseline",
    "benchmark",
    "case study",
    "confidence interval",
    "dataset",
    "effect size",
    "empirical",
    "experiment",
    "f1",
    "metric",
    "p-value",
    "result",
    "roc",
    "statistical",
}

ABSOLUTE_PHRASES = {
    "are not enough",
    "is not enough",
    "must support",
    "must provide",
    "always",
    "never",
    "universally",
    "without exception",
}

AUTHOR_NOTE_PATTERNS = {
    "bibliography keys are synchronized",
    "citation keys are synchronized",
    "as an ai",
    "i cannot",
    "i can't",
}


def _word_tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text.lower())}


def _topic_fit(text: str, topic_terms: set[str], required_topics: list[str]) -> float:
    haystack = text.lower()
    required_hits = sum(1 for topic in required_topics if topic.lower() in haystack)
    if required_topics:
        return required_hits / len(required_topics)
    if not topic_terms:
        return 1.0
    text_terms = _word_tokens(text)
    return len(topic_terms & text_terms) / len(topic_terms)


def _count_latex_citations(text: str) -> int:
    return len(re.findall(r"\\(?:[A-Za-z]*cite[A-Za-z]*|nocite)\s*(?:\[[^\]]*\]\s*){0,2}\{[^{}]+\}", text))


def _section_text(sections: list[ManuscriptSectionRecord]) -> str:
    return "\n\n".join(section.content for section in sections if section.content)


def _repeated_phrases(text: str) -> list[str]:
    lowered = re.sub(r"\s+", " ", text.lower())
    phrases = []
    candidates = [
        "representation is not enough",
        "dynamic perturbation causal",
        "not enough",
        "must support",
    ]
    for phrase in candidates:
        if lowered.count(phrase) >= 2:
            phrases.append(phrase)
    return phrases


def build_quality_audit(
    *,
    audit_id: str,
    quest_id: str,
    stage: ResearchStage,
    topic: str,
    evidence: list[ClaimEvidenceRecord],
    sections: list[ManuscriptSectionRecord],
    reference_count: int | None = None,
    cited_reference_count: int | None = None,
    min_reference_count: int = 50,
    min_cited_reference_count: int = 30,
    required_topics: list[str] | None = None,
    created_at: str,
) -> ResearchQualityAuditRecord:
    required_topics = [item.strip() for item in (required_topics or []) if item.strip()]
    text = _section_text(sections)
    topic_terms = _word_tokens(topic)
    supported = [item for item in evidence if item.support_status == "supported"]
    unsupported = [item for item in evidence if item.support_status in {"unsupported", "contradicted"}]
    source_backed = [item for item in evidence if item.paper_id and item.source_title and (item.snippet or item.quote or item.locator)]
    reference_count = reference_count if reference_count is not None else len({item.paper_id for item in evidence if item.paper_id})
    cited_reference_count = cited_reference_count if cited_reference_count is not None else len({item.metadata.get("citation_key") for item in evidence if item.metadata.get("citation_key")})
    latex_citation_count = _count_latex_citations(text)
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if len(part.strip()) >= 80]
    citation_density = latex_citation_count / max(len(paragraphs), 1)
    quantitative_evidence = [item for item in evidence if any(term in f"{item.claim} {item.snippet or ''} {item.quote or ''} {' '.join(map(str, item.metadata.values()))}".lower() for term in QUANTITATIVE_EVIDENCE_TERMS)]
    topic_fit_scores = [_topic_fit(f"{item.claim} {item.source_title or ''} {item.snippet or ''}", topic_terms, required_topics) for item in evidence]
    off_topic_count = len([score for score in topic_fit_scores if score < 0.08])
    absolute_hits = sorted({phrase for phrase in ABSOLUTE_PHRASES if phrase in text.lower()})
    author_note_hits = sorted({phrase for phrase in AUTHOR_NOTE_PATTERNS if phrase in text.lower()})
    repeated = _repeated_phrases(text)
    has_feasibility_section = any(
        ("challenge" in section.title.lower() or "mitigation" in section.title.lower() or "feasibility" in section.title.lower()) or ("challenge" in section.content.lower() and "mitigation" in section.content.lower())
        for section in sections
    )

    findings: list[str] = []
    repair_actions: list[str] = []
    recommended_queries: list[str] = []

    if reference_count < min_reference_count:
        findings.append(f"Reference coverage is thin: {reference_count}/{min_reference_count} usable references.")
        repair_actions.append("Expand scholarly search and rebuild the core literature set before final manuscript export.")
        recommended_queries.extend([f"{topic} review benchmark", f"{topic} systematic review", f"{topic} limitations empirical results"])
    if cited_reference_count < min_cited_reference_count and latex_citation_count < min_cited_reference_count:
        findings.append(f"Inline citation coverage is thin: {max(cited_reference_count, latex_citation_count)}/{min_cited_reference_count}.")
        repair_actions.append("Replace bibliography-only coverage with paragraph-level citations tied to exact BibTeX keys.")
    if unsupported:
        findings.append(f"{len(unsupported)} claim(s) are unsupported or contradicted.")
        repair_actions.append("Remove, weaken, or relabel unsupported claims before final release.")
    if len(source_backed) < max(3, min(10, len(evidence))):
        findings.append("Claim evidence lacks enough paper/snippet/locator bindings.")
        repair_actions.append("Attach paper_id, citation_key, snippet or locator to core claims.")
    if not quantitative_evidence:
        findings.append("No quantitative, benchmark, case-study, metric, or empirical evidence was detected.")
        repair_actions.append("Search for benchmark, dataset, case study, empirical result, ablation, or metric evidence.")
        recommended_queries.extend([f"{topic} benchmark dataset", f"{topic} case study empirical", f"{topic} ablation metric"])
    if off_topic_count:
        findings.append(f"{off_topic_count} evidence item(s) appear weakly related to the topic.")
        repair_actions.append("Exclude off-topic or cross-domain references from final references unless explicitly justified.")
    if not has_feasibility_section and sections:
        findings.append("No implementation challenges and mitigations discussion was detected.")
        repair_actions.append("Add an implementation challenges and mitigations section with evidence-backed risks.")
    if repeated:
        findings.append("Repeated phrases detected: " + ", ".join(repeated) + ".")
        repair_actions.append("Consolidate repeated claims and add transitions between sections.")
    if absolute_hits:
        findings.append("Over-absolute academic wording detected: " + ", ".join(absolute_hits) + ".")
        repair_actions.append("Hedge absolute claims unless directly supported by evidence.")
    if author_note_hits:
        findings.append("Author/tool process notes remain in manuscript text.")
        repair_actions.append("Remove process notes such as citation-key synchronization comments before export.")

    metrics: dict[str, Any] = {
        "reference_count": reference_count,
        "min_reference_count": min_reference_count,
        "cited_reference_count": cited_reference_count,
        "min_cited_reference_count": min_cited_reference_count,
        "latex_citation_count": latex_citation_count,
        "citation_density": round(citation_density, 4),
        "evidence_count": len(evidence),
        "supported_claim_count": len(supported),
        "unsupported_claim_count": len(unsupported),
        "source_backed_claim_count": len(source_backed),
        "quantitative_evidence_count": len(quantitative_evidence),
        "off_topic_evidence_count": off_topic_count,
        "repeated_phrase_count": len(repeated),
        "absolute_phrase_count": len(absolute_hits),
        "author_note_count": len(author_note_hits),
        "has_feasibility_section": has_feasibility_section,
    }

    score = 1.0
    score -= 0.2 if reference_count < min_reference_count else 0.0
    score -= 0.18 if max(cited_reference_count, latex_citation_count) < min_cited_reference_count else 0.0
    score -= min(0.2, len(unsupported) * 0.08)
    score -= 0.15 if not quantitative_evidence else 0.0
    score -= 0.1 if off_topic_count else 0.0
    score -= 0.08 if sections and not has_feasibility_section else 0.0
    score -= min(0.12, (len(repeated) + len(absolute_hits) + len(author_note_hits)) * 0.04)
    score = max(0.0, min(1.0, round(score, 4)))

    status: QualityAuditStatus = "pass"
    if any(
        [
            reference_count < min_reference_count,
            max(cited_reference_count, latex_citation_count) < min_cited_reference_count,
            unsupported,
            not quantitative_evidence,
            author_note_hits,
        ]
    ):
        status = "block"
    elif findings:
        status = "revise"

    return ResearchQualityAuditRecord(
        audit_id=audit_id,
        quest_id=quest_id,
        stage=stage,
        status=status,
        score=score,
        metrics=metrics,
        findings=findings,
        repair_actions=repair_actions,
        recommended_queries=sorted(set(recommended_queries)),
        created_at=created_at,
    )
