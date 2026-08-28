from __future__ import annotations

from collections import defaultdict

from .quality import preprint_penalty, published_version_bonus, venue_tier_score
from .types import PaperRecord
from .utils import (
    citation_score,
    completeness_score,
    local_upload_bonus,
    recency_score,
    relevance_score,
)


def score_papers(
    papers: list[PaperRecord],
    *,
    terms: set[str],
    quality_mode: str = "strict",
) -> list[PaperRecord]:
    scored: list[PaperRecord] = []
    for paper in papers:
        paper.relevance_score = round(relevance_score(paper, terms), 4)
        paper.recency_score = round(recency_score(paper.year), 4)
        paper.completeness_score = round(completeness_score(paper), 4)
        cite_score = citation_score(paper.cited_by_count)
        tier_score = venue_tier_score(paper.venue_tier)
        publish_bonus = published_version_bonus(paper)
        preprint_bias = preprint_penalty(paper) if quality_mode == "strict" else 0.0
        paper.rank_score = round(
            paper.relevance_score * 0.45
            + paper.completeness_score * 0.2
            + paper.recency_score * 0.15
            + cite_score * 0.15
            + tier_score * 0.12
            + publish_bonus
            + preprint_bias
            + local_upload_bonus(paper)
            + (0.05 if paper.provider in {"dblp", "openalex", "pubmed", "acl-anthology"} else 0.0),
            4,
        )
        paper.quality_signals["venue_tier_score"] = tier_score
        paper.quality_signals["published_version_bonus"] = publish_bonus
        paper.quality_signals["preprint_penalty"] = preprint_bias
        scored.append(paper)
    return sorted(scored, key=lambda item: item.rank_score, reverse=True)


def select_core_papers(
    papers: list[PaperRecord],
    *,
    limit: int,
) -> list[PaperRecord]:
    selected: list[PaperRecord] = []
    provider_counts: dict[str, int] = defaultdict(int)
    keyword_signatures: set[tuple[str, ...]] = set()
    preprint_count = 0

    for paper in papers:
        signature = tuple(sorted(paper.keywords[:4]))
        provider_cap = max(3, limit // 2)
        if provider_counts[paper.provider] >= provider_cap and paper.provider != "local-upload":
            continue
        if paper.is_preprint and limit >= 5 and preprint_count >= max(1, limit // 5):
            continue
        if signature and signature in keyword_signatures and paper.rank_score < 0.75:
            continue

        selected.append(paper)
        provider_counts[paper.provider] += 1
        if paper.is_preprint:
            preprint_count += 1
        if signature:
            keyword_signatures.add(signature)
        if len(selected) >= limit:
            break

    if len(selected) < min(limit, len(papers)):
        seen_ids = {paper.paper_id for paper in selected}
        for paper in papers:
            if paper.paper_id in seen_ids:
                continue
            selected.append(paper)
            if len(selected) >= limit:
                break

    return selected
