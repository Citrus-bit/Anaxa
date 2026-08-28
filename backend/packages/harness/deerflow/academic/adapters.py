from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from html import unescape
from typing import Protocol
from urllib.parse import urlencode

import httpx

from deerflow.runtime.utils import now_iso

from .quality import hydrate_quality_metadata, is_nlp_topic
from .types import PaperAuthor, PaperRecord
from .utils import (
    canonical_id_for,
    extract_keywords,
    infer_conflict_flags,
    infer_method_tags,
    infer_population_tags,
    normalize_arxiv_id,
    normalize_doi,
    normalize_pmid,
    normalize_whitespace,
    split_author_name,
    summarize_abstract,
)

logger = logging.getLogger(__name__)


class AcademicSourceAdapter(Protocol):
    name: str

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]: ...


class HTTPAcademicAdapter:
    name = "base"

    def __init__(self, *, timeout: float = 20.0) -> None:
        self._timeout = timeout

    async def _client(self, *, headers: dict[str, str] | None = None) -> httpx.AsyncClient:
        default_headers = {
            "User-Agent": "MedrixFlowAcademicResearch/0.1",
            "Accept": "application/json",
        }
        if headers:
            default_headers.update(headers)
        return httpx.AsyncClient(timeout=self._timeout, headers=default_headers, follow_redirects=True)

    @staticmethod
    def _paper(
        *,
        project_id: str,
        provider: str,
        provider_id: str | None,
        title: str,
        authors: list[PaperAuthor],
        year: int | None,
        venue: str | None,
        abstract: str | None,
        doi: str | None = None,
        pmid: str | None = None,
        pmcid: str | None = None,
        arxiv_id: str | None = None,
        cited_by_count: int | None = None,
        source_url: str | None = None,
        oa_url: str | None = None,
        metadata_only: bool = False,
        raw_source: dict | None = None,
    ) -> PaperRecord:
        first_author = authors[0].display_name if authors else None
        title_text = title.strip()
        abstract_text = abstract.strip() if abstract else None
        created_at = now_iso()
        canonical_id = canonical_id_for(
            doi=doi,
            pmid=pmid,
            pmcid=pmcid,
            arxiv_id=arxiv_id,
            title=title_text,
            first_author=first_author,
            year=year,
            provider=provider,
            provider_id=provider_id,
        )
        return hydrate_quality_metadata(
            PaperRecord(
                paper_id=f"{project_id}:{provider}:{provider_id or canonical_id}",
                project_id=project_id,
                canonical_id=canonical_id,
                title=title_text,
                authors=authors,
                year=year,
                venue=venue,
                abstract=abstract_text,
                doi=normalize_doi(doi),
                pmid=normalize_pmid(pmid),
                pmcid=normalize_pmid(pmcid),
                arxiv_id=normalize_arxiv_id(arxiv_id),
                cited_by_count=cited_by_count,
                provider=provider,
                provider_id=provider_id,
                source_url=source_url,
                oa_url=oa_url,
                metadata_only=metadata_only,
                keywords=extract_keywords(title_text, abstract_text),
                methods=infer_method_tags(title_text, abstract_text),
                populations=infer_population_tags(title_text, abstract_text),
                conflict_flags=infer_conflict_flags(title_text, abstract_text),
                raw_source=raw_source or {},
                created_at=created_at,
                updated_at=created_at,
            )
        )


class DBLPAdapter(HTTPAcademicAdapter):
    name = "dblp"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {
            "q": query,
            "h": str(min(limit, 12)),
            "format": "json",
        }
        url = f"https://dblp.org/search/publ/api?{urlencode(params)}"
        try:
            async with await self._client(headers={"Accept": "application/json"}) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("DBLP search failed for query=%s", query, exc_info=True)
            return []

        hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
        if isinstance(hits, dict):
            hits = [hits]

        papers: list[PaperRecord] = []
        for hit in hits[:limit]:
            info = hit.get("info") or {}
            title = _strip_markup(info.get("title"))
            if not title:
                continue
            authors = _dblp_authors((info.get("authors") or {}).get("author"))
            year_text = str(info.get("year") or "")
            year = int(year_text) if year_text.isdigit() else None
            venue = info.get("venue") or info.get("booktitle") or info.get("journal")
            doi = info.get("doi")
            ee = info.get("ee")
            if isinstance(ee, list):
                source_url = next((item for item in ee if isinstance(item, str) and item.startswith("http")), None)
            else:
                source_url = ee if isinstance(ee, str) else None
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=info.get("key") or doi or source_url,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=venue,
                    abstract=None,
                    doi=doi,
                    source_url=source_url,
                    oa_url=source_url,
                    metadata_only=True,
                    raw_source={
                        "type": info.get("type"),
                        "dblp_key": info.get("key"),
                        "venue": venue,
                    },
                )
            )
        return papers


class OpenAlexAdapter(HTTPAcademicAdapter):
    name = "openalex"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {"search": query, "per_page": str(min(limit, 25))}
        if api_key := os.getenv("OPENALEX_API_KEY"):
            params["api_key"] = api_key
        url = f"https://api.openalex.org/works?{urlencode(params)}"
        try:
            async with await self._client() as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("OpenAlex search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        for item in payload.get("results", [])[:limit]:
            authors = []
            for idx, authorship in enumerate(item.get("authorships", []) or []):
                author_info = authorship.get("author", {}) or {}
                display_name = author_info.get("display_name")
                if not display_name:
                    continue
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            doi = item.get("doi")
            oa_location = item.get("best_oa_location") or {}
            source = item.get("primary_location", {}).get("source", {}) or {}
            year = item.get("publication_year")
            title = item.get("display_name") or item.get("title") or ""
            if not title:
                continue
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=item.get("id"),
                    title=title,
                    authors=authors,
                    year=year if isinstance(year, int) else None,
                    venue=source.get("display_name"),
                    abstract=_openalex_abstract(item),
                    doi=doi,
                    cited_by_count=item.get("cited_by_count"),
                    source_url=item.get("id"),
                    oa_url=oa_location.get("landing_page_url") or oa_location.get("pdf_url"),
                    metadata_only=False,
                    raw_source={
                        "referenced_works": item.get("referenced_works", []),
                        "concepts": [concept.get("display_name") for concept in item.get("concepts", [])[:8] if concept.get("display_name")],
                    },
                )
            )
        return papers


def _openalex_abstract(item: dict) -> str | None:
    inverted = item.get("abstract_inverted_index")
    if not isinstance(inverted, dict):
        return None
    positions: dict[int, str] = {}
    for token, indexes in inverted.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions[index] = token
    if not positions:
        return None
    return " ".join(token for _index, token in sorted(positions.items()))


def _strip_markup(value: str | None) -> str | None:
    if not value:
        return None
    return normalize_whitespace(re.sub(r"<[^>]+>", " ", unescape(value)))


def _dblp_authors(value) -> list[PaperAuthor]:
    if not value:
        return []
    items = value if isinstance(value, list) else [value]
    authors: list[PaperAuthor] = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            display_name = item.get("text") or item.get("#text") or item.get("name")
        else:
            display_name = str(item)
        display_name = normalize_whitespace(display_name)
        if not display_name:
            continue
        given, family = split_author_name(display_name)
        authors.append(
            PaperAuthor(
                display_name=display_name,
                given_name=given,
                family_name=family,
                ordinal=idx,
            )
        )
    return authors


class CrossrefAdapter(HTTPAcademicAdapter):
    name = "crossref"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {
            "query.bibliographic": query,
            "rows": str(min(limit, 15)),
            "select": ",".join(
                [
                    "DOI",
                    "URL",
                    "title",
                    "author",
                    "published-print",
                    "published-online",
                    "created",
                    "container-title",
                    "abstract",
                    "is-referenced-by-count",
                    "type",
                ]
            ),
            "mailto": os.getenv("MEDRIX_FLOW_CONTACT_EMAIL", ""),
        }
        headers = {}
        if contact_email := os.getenv("MEDRIX_FLOW_CONTACT_EMAIL"):
            headers["User-Agent"] = f"MedrixFlowAcademicResearch/0.1 (mailto:{contact_email})"
        url = f"https://api.crossref.org/works?{urlencode(params)}"
        try:
            async with await self._client(headers=headers or None) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("Crossref search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        for item in payload.get("message", {}).get("items", [])[:limit]:
            authors = []
            for idx, author in enumerate(item.get("author", []) or []):
                display_name = " ".join(part for part in [author.get("given"), author.get("family")] if part).strip()
                display_name = display_name or author.get("name")
                if not display_name:
                    continue
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=author.get("given"),
                        family_name=author.get("family"),
                        ordinal=idx,
                    )
                )
            title = _first_text(item.get("title"))
            if not title:
                continue
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=item.get("DOI"),
                    title=title,
                    authors=authors,
                    year=_crossref_year(item),
                    venue=_first_text(item.get("container-title")),
                    abstract=_strip_jats(item.get("abstract")),
                    doi=item.get("DOI"),
                    cited_by_count=item.get("is-referenced-by-count"),
                    source_url=item.get("URL"),
                    metadata_only=False,
                    raw_source={"type": item.get("type")},
                )
            )
        return papers


def _first_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        first = value[0]
        return first if isinstance(first, str) else None
    return None


def _crossref_year(item: dict) -> int | None:
    for key in ("published-print", "published-online", "created"):
        date_parts = (item.get(key) or {}).get("date-parts") or []
        if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int):
                return year
    return None


def _strip_jats(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub(r"<[^>]+>", " ", text).strip()


class ArxivAdapter(HTTPAcademicAdapter):
    name = "arxiv"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(min(limit, 10)),
        }
        url = f"http://export.arxiv.org/api/query?{urlencode(params)}"
        try:
            async with await self._client(headers={"Accept": "application/atom+xml"}) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.text
        except Exception:
            logger.warning("arXiv search failed for query=%s", query, exc_info=True)
            return []

        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            logger.warning("Failed to parse arXiv response for query=%s", query, exc_info=True)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        papers: list[PaperRecord] = []
        for entry in root.findall("atom:entry", ns)[:limit]:
            title = normalize_xml_text(entry.findtext("atom:title", default="", namespaces=ns))
            if not title:
                continue
            summary = normalize_xml_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = entry.findtext("atom:published", default="", namespaces=ns)
            year = int(published[:4]) if published[:4].isdigit() else None
            authors = []
            for idx, author in enumerate(entry.findall("atom:author", ns)):
                display_name = normalize_xml_text(author.findtext("atom:name", default="", namespaces=ns))
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            entry_id = entry.findtext("atom:id", default="", namespaces=ns)
            arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else None
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=arxiv_id,
                    title=title,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    abstract=summary,
                    arxiv_id=arxiv_id,
                    source_url=entry_id or None,
                    oa_url=entry_id or None,
                    metadata_only=False,
                    raw_source={
                        "categories": [item.attrib.get("term") for item in entry.findall("atom:category", ns)],
                    },
                )
            )
        return papers


def normalize_xml_text(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def _openreview_value(value) -> str | None:
    if isinstance(value, dict):
        raw = value.get("value")
        return normalize_whitespace(str(raw)) if raw is not None else None
    if value is None:
        return None
    return normalize_whitespace(str(value))


def _openreview_list(value) -> list[str]:
    if isinstance(value, dict):
        payload = value.get("value")
        if isinstance(payload, list):
            return [normalize_whitespace(str(item)) for item in payload if normalize_whitespace(str(item))]
    if isinstance(value, list):
        return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]
    return []


def _is_high_quality_openreview_note(*, venue: str | None, venueid: str | None) -> bool:
    combined = f"{venue or ''} {venueid or ''}".lower()
    if not combined or "corr" in combined or "public_article" in combined:
        return False
    return any(token in combined for token in _OPENREVIEW_VENUE_HINTS)


def _looks_like_acl_family(source_name: str | None, source_url: str | None) -> bool:
    combined = f"{source_name or ''} {source_url or ''}".lower()
    return any(token in combined for token in _ACL_FAMILY_HINTS)


def _year_from_text(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", value)
    return int(match.group(0)) if match else None


def _extract_doi_from_text(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", value, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _openreview_absolute_url(value: str | None, *, note_id: str | None = None, pdf: bool = False) -> str | None:
    normalized = normalize_whitespace(value)
    if normalized:
        if normalized.startswith("http://") or normalized.startswith("https://"):
            return normalized
        if normalized.startswith("/"):
            return f"https://openreview.net{normalized}"
    if note_id:
        route = "pdf" if pdf else "forum"
        return f"https://openreview.net/{route}?id={note_id}"
    return None


_OPENREVIEW_VENUE_HINTS = {
    "aaai",
    "acl",
    "colm",
    "cvpr",
    "eccv",
    "emnlp",
    "iccv",
    "iclr",
    "icml",
    "ijcai",
    "naacl",
    "neurips",
    "sigir",
    "tmlr",
    "wsdm",
    "www",
}

_ACL_FAMILY_HINTS = {
    "aclanthology.org",
    "acl ",
    " emnlp",
    " naacl",
    " tacl",
    " coling",
    " findings of the association for computational linguistics",
    "association for computational linguistics",
}


class OpenReviewAdapter(HTTPAcademicAdapter):
    name = "openreview"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {
            "term": query,
            "limit": str(min(limit, 8)),
        }
        url = f"https://api2.openreview.net/notes/search?{urlencode(params)}"
        try:
            async with await self._client() as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("OpenReview search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        for note in payload.get("notes", [])[:limit]:
            note_id = note.get("id")
            content = note.get("content") or {}
            title = _openreview_value(content.get("title"))
            venue = _openreview_value(content.get("venue"))
            venueid = _openreview_value(content.get("venueid"))
            if not title or not _is_high_quality_openreview_note(venue=venue, venueid=venueid):
                continue
            authors = []
            for idx, display_name in enumerate(_openreview_list(content.get("authors"))):
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            source_url = _openreview_absolute_url(_openreview_value(content.get("html")), note_id=note_id)
            pdf_url = _openreview_absolute_url(_openreview_value(content.get("pdf")), note_id=note_id, pdf=True)
            abstract = _openreview_value(content.get("abstract"))
            doi = _extract_doi_from_text("\n".join(filter(None, [source_url, _openreview_value(content.get("_bibtex"))])))
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=note_id,
                    title=title,
                    authors=authors,
                    year=_year_from_text(_openreview_value(content.get("venue")) or _openreview_value(content.get("_bibtex"))),
                    venue=venue,
                    abstract=abstract,
                    doi=doi,
                    source_url=source_url,
                    oa_url=pdf_url or source_url,
                    metadata_only=not bool(abstract),
                    raw_source={
                        "status": "accepted",
                        "venueid": venueid,
                        "pdf": pdf_url,
                    },
                )
            )
        return papers


class ACLAnthologyAdapter(HTTPAcademicAdapter):
    name = "acl-anthology"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        params = {"search": query, "per_page": str(min(limit * 2, 20))}
        if api_key := os.getenv("OPENALEX_API_KEY"):
            params["api_key"] = api_key
        url = f"https://api.openalex.org/works?{urlencode(params)}"
        try:
            async with await self._client() as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("ACL Anthology search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        for item in payload.get("results", []):
            source = item.get("primary_location", {}).get("source", {}) or {}
            source_name = normalize_whitespace(source.get("display_name"))
            source_url = (item.get("best_oa_location") or {}).get("landing_page_url") or (item.get("best_oa_location") or {}).get("pdf_url") or item.get("doi") or item.get("id")
            if not _looks_like_acl_family(source_name, source_url):
                continue
            title = item.get("display_name") or item.get("title") or ""
            if not title:
                continue
            authors: list[PaperAuthor] = []
            for idx, authorship in enumerate(item.get("authorships", []) or []):
                author_info = authorship.get("author", {}) or {}
                display_name = author_info.get("display_name")
                if not display_name:
                    continue
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=item.get("id"),
                    title=title,
                    authors=authors,
                    year=item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
                    venue=source_name or source.get("host_organization_name"),
                    abstract=_openalex_abstract(item),
                    doi=item.get("doi"),
                    cited_by_count=item.get("cited_by_count"),
                    source_url=source_url,
                    oa_url=(item.get("best_oa_location") or {}).get("pdf_url"),
                    metadata_only=False,
                    raw_source={
                        "upstream_provider": "openalex",
                        "referenced_works": item.get("referenced_works", []),
                    },
                )
            )
            if len(papers) >= limit:
                break
        return papers


class PubMedAdapter(HTTPAcademicAdapter):
    name = "pubmed"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        search_params = {
            "db": "pubmed",
            "retmode": "json",
            "retmax": str(min(limit, 10)),
            "term": query,
        }
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{urlencode(search_params)}"
        try:
            async with await self._client() as client:
                search_resp = await client.get(search_url)
                search_resp.raise_for_status()
                id_list = (search_resp.json().get("esearchresult") or {}).get("idlist", [])
                if not id_list:
                    return []
                summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode({"db": "pubmed", "retmode": "json", "id": ",".join(id_list)})
                summary_resp = await client.get(summary_url)
                summary_resp.raise_for_status()
                payload = summary_resp.json()
        except Exception:
            logger.warning("PubMed search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        result = payload.get("result") or {}
        for pmid in result.get("uids", [])[:limit]:
            item = result.get(pmid) or {}
            title = item.get("title")
            if not title:
                continue
            authors = []
            for idx, author in enumerate(item.get("authors", []) or []):
                display_name = author.get("name")
                if not display_name:
                    continue
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            article_ids = item.get("articleids", []) or []
            doi = next((entry.get("value") for entry in article_ids if entry.get("idtype") == "doi"), None)
            pmcid = next((entry.get("value") for entry in article_ids if entry.get("idtype") == "pmc"), None)
            pubdate = item.get("pubdate") or ""
            year_match = re.search(r"(19|20)\d{2}", pubdate)
            year = int(year_match.group(0)) if year_match else None
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=pmid,
                    title=title,
                    authors=authors,
                    year=year,
                    venue=item.get("fulljournalname") or item.get("source"),
                    abstract=summarize_abstract(item.get("sorttitle") or title),
                    doi=doi,
                    pmid=pmid,
                    pmcid=pmcid,
                    source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    oa_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None,
                    metadata_only=False,
                    raw_source={"pubtype": item.get("pubtype", [])},
                )
            )
        return papers


class SemanticScholarAdapter(HTTPAcademicAdapter):
    name = "semantic-scholar"

    async def search(self, query: str, *, project_id: str, limit: int) -> list[PaperRecord]:
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        params = {
            "query": query,
            "limit": str(min(limit, 10)),
            "fields": ",".join(
                [
                    "title",
                    "abstract",
                    "authors",
                    "year",
                    "venue",
                    "externalIds",
                    "url",
                    "citationCount",
                    "openAccessPdf",
                ]
            ),
        }
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urlencode(params)}"
        try:
            headers = {"x-api-key": api_key} if api_key else None
            async with await self._client(headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning("Semantic Scholar search failed for query=%s", query, exc_info=True)
            return []

        papers: list[PaperRecord] = []
        for item in payload.get("data", [])[:limit]:
            title = item.get("title")
            if not title:
                continue
            authors = []
            for idx, author in enumerate(item.get("authors", []) or []):
                display_name = author.get("name")
                if not display_name:
                    continue
                given, family = split_author_name(display_name)
                authors.append(
                    PaperAuthor(
                        display_name=display_name,
                        given_name=given,
                        family_name=family,
                        ordinal=idx,
                    )
                )
            external_ids = item.get("externalIds") or {}
            papers.append(
                self._paper(
                    project_id=project_id,
                    provider=self.name,
                    provider_id=item.get("paperId"),
                    title=title,
                    authors=authors,
                    year=item.get("year"),
                    venue=item.get("venue"),
                    abstract=item.get("abstract"),
                    doi=external_ids.get("DOI"),
                    pmid=external_ids.get("PubMed"),
                    arxiv_id=external_ids.get("ArXiv"),
                    cited_by_count=item.get("citationCount"),
                    source_url=item.get("url"),
                    oa_url=(item.get("openAccessPdf") or {}).get("url"),
                    metadata_only=False,
                    raw_source={"paperId": item.get("paperId")},
                )
            )
        return papers


def build_default_adapters(
    domain: str,
    *,
    topic: str = "",
    scope: str | None = None,
    source_profile: str | None = None,
) -> Sequence[AcademicSourceAdapter]:
    adapters: list[AcademicSourceAdapter] = []
    if domain == "cs_ai":
        if source_profile in {None, "", "default", "cs_ai_premium"}:
            adapters.append(DBLPAdapter())
            adapters.append(OpenReviewAdapter())
            if is_nlp_topic(topic, scope):
                adapters.append(ACLAnthologyAdapter())
            adapters.append(SemanticScholarAdapter())
            adapters.extend([OpenAlexAdapter(), CrossrefAdapter(), ArxivAdapter()])
            return adapters
    adapters = [OpenAlexAdapter(), CrossrefAdapter()]
    if domain in {"general", "cs_ai"}:
        adapters.append(SemanticScholarAdapter())
        adapters.append(ArxivAdapter())
    if domain == "biomedical":
        adapters.append(PubMedAdapter())
    return adapters
