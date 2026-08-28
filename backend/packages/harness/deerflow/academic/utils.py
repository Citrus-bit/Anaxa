from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.parse import quote

from .types import PaperAuthor, PaperRecord

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "study",
    "system",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
}

_BIOMEDICAL_HINTS = {
    "biomedical",
    "biology",
    "cancer",
    "cell",
    "clinical",
    "disease",
    "drug",
    "gene",
    "genomic",
    "health",
    "hospital",
    "medical",
    "medicine",
    "molecule",
    "patient",
    "pharmacology",
    "protein",
    "therapy",
}

_CS_AI_HINTS = {
    "ai",
    "algorithm",
    "artificial intelligence",
    "computer vision",
    "dataset",
    "deep learning",
    "diffusion",
    "embedding",
    "evaluation",
    "foundation model",
    "language model",
    "llm",
    "machine learning",
    "multimodal",
    "neural",
    "reinforcement learning",
    "transformer",
}

_METHOD_HINTS = {
    "benchmark": "benchmarking",
    "causal": "causal inference",
    "case study": "case study",
    "clinical trial": "clinical trial",
    "diffusion": "diffusion models",
    "experiment": "experimental design",
    "framework": "framework design",
    "graph": "graph modeling",
    "meta-analysis": "meta-analysis",
    "neural network": "neural networks",
    "randomized": "randomized study",
    "regression": "regression analysis",
    "review": "review",
    "simulation": "simulation",
    "survey": "survey",
    "transformer": "transformer models",
}

_POPULATION_HINTS = {
    "adolescent": "adolescents",
    "adult": "adults",
    "child": "children",
    "elderly": "older adults",
    "hospital": "clinical settings",
    "student": "students",
    "teacher": "teachers",
    "user": "users",
    "patient": "patients",
}

_CONFLICT_HINTS = {
    "challenge": "challenge",
    "contradict": "conflicting findings",
    "controvers": "controversy",
    "limitation": "limitation",
    "risk": "risk",
    "uncertain": "uncertainty",
}


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def slugify(text: str, *, fallback: str = "project") -> str:
    normalized = normalize_whitespace(text).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    normalized = normalized.strip("-")
    return normalized or fallback


def normalize_title_key(title: str) -> str:
    text = normalize_whitespace(title).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = normalize_whitespace(doi)
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    return value.lower() or None


def doi_url(doi: str | None) -> str | None:
    normalized = normalize_doi(doi)
    if not normalized:
        return None
    return f"https://doi.org/{quote(normalized, safe='/')}"


def normalize_pmid(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = normalize_whitespace(str(value))
    return text or None


def normalize_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    text = normalize_whitespace(value)
    text = re.sub(r"^arxiv:", "", text, flags=re.IGNORECASE)
    return text or None


def split_author_name(display_name: str) -> tuple[str | None, str | None]:
    text = normalize_whitespace(display_name)
    if not text:
        return None, None
    if "," in text:
        family, given = [part.strip() for part in text.split(",", 1)]
        return given or None, family or None
    parts = text.split(" ")
    if len(parts) == 1:
        return None, parts[0]
    return " ".join(parts[:-1]) or None, parts[-1] or None


def initials(given_name: str | None) -> str:
    if not given_name:
        return ""
    parts = re.split(r"[\s\-]+", given_name.strip())
    chars = [part[0].upper() + "." for part in parts if part]
    return " ".join(chars)


def detect_domain(topic: str, scope: str | None = None) -> str:
    haystack = f"{topic} {scope or ''}".lower()
    if any(hint in haystack for hint in _BIOMEDICAL_HINTS):
        return "biomedical"
    if any(hint in haystack for hint in _CS_AI_HINTS):
        return "cs_ai"
    return "general"


def extract_keywords(*texts: str | None, limit: int = 12) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", normalize_whitespace(text).lower()):
            if token in _STOPWORDS:
                continue
            counter[token] += 1
    return [word for word, _count in counter.most_common(limit)]


def infer_method_tags(*texts: str | None) -> list[str]:
    haystack = " ".join(normalize_whitespace(text).lower() for text in texts if text)
    tags = [label for needle, label in _METHOD_HINTS.items() if needle in haystack]
    return sorted(set(tags))


def infer_population_tags(*texts: str | None) -> list[str]:
    haystack = " ".join(normalize_whitespace(text).lower() for text in texts if text)
    tags = [label for needle, label in _POPULATION_HINTS.items() if needle in haystack]
    return sorted(set(tags))


def infer_conflict_flags(*texts: str | None) -> list[str]:
    haystack = " ".join(normalize_whitespace(text).lower() for text in texts if text)
    tags = [label for needle, label in _CONFLICT_HINTS.items() if needle in haystack]
    return sorted(set(tags))


def canonical_id_for(
    *,
    doi: str | None = None,
    pmid: str | None = None,
    pmcid: str | None = None,
    arxiv_id: str | None = None,
    title: str,
    first_author: str | None = None,
    year: int | None = None,
    provider: str,
    provider_id: str | None = None,
) -> str:
    if normalized_doi := normalize_doi(doi):
        return f"doi:{normalized_doi}"
    if pmid:
        return f"pmid:{normalize_pmid(pmid)}"
    if pmcid:
        return f"pmcid:{normalize_pmid(pmcid)}"
    if normalized_arxiv := normalize_arxiv_id(arxiv_id):
        return f"arxiv:{normalized_arxiv}"
    title_key = normalize_title_key(title)
    author_key = slugify(first_author or "unknown-author")
    year_key = str(year) if year is not None else "nd"
    if title_key:
        return f"title:{title_key}:{author_key}:{year_key}"
    provider_key = slugify(provider_id or title or provider or "paper")
    return f"{provider}:{provider_key}"


def author_display_list(authors: list[PaperAuthor]) -> list[str]:
    values: list[str] = []
    for author in sorted(authors, key=lambda item: item.ordinal):
        family = author.family_name or split_author_name(author.display_name)[1]
        given = author.given_name or split_author_name(author.display_name)[0]
        if family:
            rendered = family
            if initials(given):
                rendered = f"{family}, {initials(given)}"
            values.append(rendered)
        else:
            values.append(author.display_name)
    return values


def topic_terms(topic: str, scope: str | None = None) -> set[str]:
    terms = set(extract_keywords(topic, scope, limit=24))
    for part in normalize_whitespace(topic).lower().split(" "):
        if len(part) >= 3 and part not in _STOPWORDS:
            terms.add(part)
    return terms


def relevance_score(paper: PaperRecord, terms: set[str]) -> float:
    haystack = f"{paper.title} {paper.abstract or ''}".lower()
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if term in haystack)
    return min(1.0, matched / max(3, len(terms) * 0.5))


def completeness_score(paper: PaperRecord) -> float:
    fields = [
        bool(paper.title),
        bool(paper.authors),
        paper.year is not None,
        bool(paper.venue),
        bool(paper.abstract),
        bool(paper.doi or paper.pmid or paper.arxiv_id),
        bool(paper.source_url),
    ]
    return sum(1 for value in fields if value) / len(fields)


def recency_score(year: int | None) -> float:
    if year is None:
        return 0.2
    current_year = datetime.now(UTC).year
    age = max(0, current_year - year)
    return max(0.1, 1.0 - min(age, 12) / 12)


def citation_score(cited_by_count: int | None) -> float:
    if not cited_by_count:
        return 0.0
    return min(1.0, math.log1p(cited_by_count) / math.log(101))


def local_upload_bonus(paper: PaperRecord) -> float:
    return 0.2 if paper.provider == "local-upload" else 0.0


def summarize_abstract(text: str | None, *, limit: int = 280) -> str:
    content = normalize_whitespace(text)
    if not content:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", content, maxsplit=1)[0]
    summary = sentence or content
    if len(summary) <= limit:
        return summary
    return summary[: limit - 1].rstrip() + "…"


def novelty_tags_for_paper(paper: PaperRecord) -> list[str]:
    tags: list[str] = []
    text = f"{paper.title} {paper.abstract or ''}".lower()
    if paper.populations:
        tags.append("research-object-gap")
    if len(paper.methods) >= 2:
        tags.append("method-combination-gap")
    if paper.conflict_flags:
        tags.append("conflicting-findings")
    if "dataset" in text or "benchmark" in text or "sample size" in text:
        tags.append("dataset-or-design-limitation")
    return sorted(set(tags))


def today_stamp() -> str:
    return datetime.now(UTC).date().isoformat()


def merge_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
