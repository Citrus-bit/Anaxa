from __future__ import annotations

from typing import Any

from .types import PaperRecord
from .utils import normalize_whitespace

_T1_VENUES = {
    "aaai",
    "acl",
    "colm",
    "coling",
    "cvpr",
    "eccv",
    "emnlp",
    "iccv",
    "iclr",
    "icml",
    "ijcai",
    "jmlr",
    "kdd",
    "naacl",
    "neurips",
    "sigir",
    "tacl",
    "tmlr",
    "wsdm",
    "www",
}

_VENUE_PATTERNS: list[tuple[str, str]] = [
    ("conference on empirical methods in natural language processing", "emnlp"),
    ("annual meeting of the association for computational linguistics", "acl"),
    ("north american chapter of the association for computational linguistics", "naacl"),
    ("transactions of the association for computational linguistics", "tacl"),
    ("empirical methods in natural language processing", "emnlp"),
    ("international conference on machine learning", "icml"),
    ("international conference on learning representations", "iclr"),
    ("conference on neural information processing systems", "neurips"),
    ("advances in neural information processing systems", "neurips"),
    ("computer vision and pattern recognition", "cvpr"),
    ("international conference on computer vision", "iccv"),
    ("european conference on computer vision", "eccv"),
    ("association for the advancement of artificial intelligence", "aaai"),
    ("international joint conference on artificial intelligence", "ijcai"),
    ("journal of machine learning research", "jmlr"),
    ("transactions on machine learning research", "tmlr"),
    ("conference on language modeling", "colm"),
    ("world wide web conference", "www"),
    ("web search and data mining", "wsdm"),
    ("knowledge discovery and data mining", "kdd"),
    ("international acm sigir conference", "sigir"),
    ("computational linguistics", "coling"),
]

_NLP_HINTS = {
    "acl",
    "alignment",
    "dialogue",
    "embedding",
    "evaluation",
    "generation",
    "instruction tuning",
    "language model",
    "large language model",
    "llm",
    "machine translation",
    "multilingual",
    "natural language",
    "nlp",
    "prompting",
    "question answering",
    "retrieval augmented generation",
    "summarization",
    "translation",
}


def is_nlp_topic(topic: str, scope: str | None = None) -> bool:
    haystack = f"{topic} {scope or ''}".lower()
    return any(token in haystack for token in _NLP_HINTS)


def hydrate_quality_metadata(paper: PaperRecord) -> PaperRecord:
    paper.source_kind = _source_kind_for_provider(paper.provider)
    paper.venue_type = detect_venue_type(paper)
    paper.venue_tier = detect_venue_tier(paper.venue)
    paper.publication_status = detect_publication_status(paper)
    paper.is_preprint = paper.venue_type == "preprint" or paper.publication_status == "preprint"
    paper.canonical_source = f"{paper.provider}:{paper.provider_id or paper.canonical_id}"
    paper.quality_signals = {
        "source_kind": paper.source_kind,
        "venue_type": paper.venue_type,
        "venue_tier": paper.venue_tier,
        "publication_status": paper.publication_status,
        "is_preprint": paper.is_preprint,
        "version_priority": round(version_priority_score(paper), 4),
        "venue_tier_score": venue_tier_score(paper.venue_tier),
        "published_version_bonus": published_version_bonus(paper),
        "preprint_penalty": preprint_penalty(paper),
    }
    return paper


def detect_venue_tier(venue: str | None) -> str:
    normalized = _normalize_venue(venue)
    if normalized in _T1_VENUES:
        return "t1"
    if normalized == "unknown":
        return "unknown"
    if normalized:
        return "unranked"
    return "unknown"


def venue_tier_score(venue_tier: str) -> float:
    return {
        "t1": 1.0,
        "t2": 0.8,
        "unranked": 0.45,
        "unknown": 0.15,
    }.get(venue_tier, 0.15)


def detect_venue_type(paper: PaperRecord) -> str:
    venue = normalize_whitespace(paper.venue).lower()
    source_url = normalize_whitespace(paper.source_url).lower()
    raw_type = str((paper.raw_source or {}).get("type") or "").lower()
    normalized_venue = _normalize_venue(venue)
    if paper.provider == "arxiv" or "arxiv" in venue or "corr" in venue or "arxiv.org" in source_url:
        return "preprint"
    if normalized_venue in _T1_VENUES:
        return "conference"
    if "workshop" in venue or "workshop" in raw_type:
        return "workshop"
    if raw_type in {"journal-article", "article"}:
        return "journal"
    if raw_type in {"proceedings-article", "conference-paper", "inproceedings"}:
        return "conference"
    if any(token in venue for token in ("journal", "transactions", "letters")):
        return "journal"
    if any(token in venue for token in ("conference", "proceedings", "symposium", "convention")):
        return "conference"
    return "unknown"


def detect_publication_status(paper: PaperRecord) -> str:
    if detect_venue_type(paper) == "preprint":
        return "preprint"

    raw_source = paper.raw_source or {}
    status = normalize_whitespace(str(raw_source.get("status") or "")).lower()
    venue = normalize_whitespace(paper.venue).lower()
    venueid = normalize_whitespace(str(raw_source.get("venueid") or "")).lower()
    if status in {"published", "accepted", "preprint"}:
        return status
    if paper.provider in {"dblp", "crossref", "openalex", "acl-anthology"} and detect_venue_type(paper) in {"conference", "journal"}:
        return "published"
    if paper.provider == "openreview":
        if any(token in venueid for token in ("/conference", "/journal")) or _normalize_venue(venue) in _T1_VENUES:
            return "accepted"
    if detect_venue_type(paper) in {"conference", "journal"}:
        return "published"
    return "unknown"


def published_version_bonus(paper: PaperRecord) -> float:
    if paper.publication_status == "published":
        return 0.12
    if paper.publication_status == "accepted":
        return 0.08
    return 0.0


def preprint_penalty(paper: PaperRecord) -> float:
    return -0.18 if paper.is_preprint else 0.0


def version_priority_score(
    paper: PaperRecord,
    *,
    quality_mode: str = "strict",
    preprint_policy: str = "prefer_final",
) -> float:
    base = 0.0
    source_weight = {
        "dblp": 0.34,
        "openalex": 0.3,
        "crossref": 0.28,
        "semantic-scholar": 0.24,
        "acl-anthology": 0.32,
        "openreview": 0.22,
        "pubmed": 0.3,
        "local-upload": 0.35,
        "arxiv": 0.05,
    }
    base += source_weight.get(paper.provider, 0.12)
    base += venue_tier_score(paper.venue_tier) * 0.35
    base += published_version_bonus(paper)
    if preprint_policy == "prefer_final":
        base += preprint_penalty(paper)
    if paper.doi:
        base += 0.15
    if paper.abstract:
        base += 0.05
    if paper.cited_by_count:
        base += min(0.08, paper.cited_by_count / 1000)
    if quality_mode == "strict" and paper.is_preprint:
        base -= 0.05
    return base


def provider_breakdown(papers: list[PaperRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paper in papers:
        counts[paper.provider] = counts.get(paper.provider, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def venue_breakdown(papers: list[PaperRecord], *, limit: int = 12) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paper in papers:
        venue = normalize_whitespace(paper.venue) or "Unknown venue"
        counts[venue] = counts.get(venue, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:limit])


def preprint_ratio(papers: list[PaperRecord]) -> float:
    if not papers:
        return 0.0
    preprints = sum(1 for paper in papers if paper.is_preprint)
    return round(preprints / len(papers), 4)


def _normalize_venue(venue: str | None) -> str:
    normalized = normalize_whitespace(venue).lower()
    if not normalized:
        return "unknown"
    normalized = normalized.replace("proceedings of the ", "")
    normalized = normalized.replace("proceedings of ", "")
    normalized = normalized.replace("the ", "")
    for pattern, canonical in _VENUE_PATTERNS:
        if pattern in normalized:
            return canonical
    if normalized in _T1_VENUES:
        return normalized
    return normalized


def _source_kind_for_provider(provider: str) -> str:
    mapping: dict[str, str] = {
        "acl-anthology": "anthology",
        "arxiv": "repository",
        "crossref": "metadata",
        "dblp": "metadata",
        "local-upload": "local_library",
        "openalex": "metadata",
        "openreview": "peer_review_platform",
        "pubmed": "metadata",
        "semantic-scholar": "metadata",
    }
    return mapping.get(provider, "metadata")


def canonical_reason(paper: PaperRecord) -> dict[str, Any]:
    return {
        "provider": paper.provider,
        "venue": paper.venue,
        "venue_type": paper.venue_type,
        "venue_tier": paper.venue_tier,
        "publication_status": paper.publication_status,
        "is_preprint": paper.is_preprint,
        "canonical_source": paper.canonical_source,
    }
