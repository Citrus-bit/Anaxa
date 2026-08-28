import asyncio
import json

import httpx

from medrix_flow.academic.adapters import (
    ACLAnthologyAdapter,
    DBLPAdapter,
    OpenReviewAdapter,
    SemanticScholarAdapter,
    build_default_adapters,
)


class FakeResponse:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.org")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self):
        return self._payload

    @property
    def text(self) -> str:
        if isinstance(self._payload, str):
            return self._payload
        return json.dumps(self._payload)


class FakeClient:
    def __init__(self, payload, *, status_code: int = 200) -> None:
        self._response = FakeResponse(payload, status_code=status_code)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str):
        return self._response


def test_build_default_adapters_uses_cs_ai_premium_stack_for_nlp_topics():
    adapters = build_default_adapters(
        "cs_ai",
        topic="large language model reasoning",
        scope="NLP evaluation and prompting",
        source_profile="cs_ai_premium",
    )

    assert [adapter.name for adapter in adapters] == [
        "dblp",
        "openreview",
        "acl-anthology",
        "semantic-scholar",
        "openalex",
        "crossref",
        "arxiv",
    ]


def test_dblp_adapter_normalizes_markup_and_quality_metadata(monkeypatch):
    adapter = DBLPAdapter()
    payload = {
        "result": {
            "hits": {
                "hit": [
                    {
                        "info": {
                            "title": "<i>Retriever</i> Fusion for <sup>LLM</sup> Evaluation",
                            "authors": {"author": ["Alice Smith", {"text": "Bob Lee"}]},
                            "year": "2025",
                            "venue": "NeurIPS",
                            "doi": "10.1000/dblp.demo",
                            "ee": "https://doi.org/10.1000/dblp.demo",
                            "key": "conf/neurips/demo2025",
                            "type": "Conference and Workshop Papers",
                        }
                    }
                ]
            }
        }
    }

    async def fake_client(*, headers=None):
        return FakeClient(payload)

    monkeypatch.setattr(adapter, "_client", fake_client)
    papers = asyncio.run(adapter.search("retrieval evaluation", project_id="project-1", limit=5))

    assert len(papers) == 1
    paper = papers[0]
    assert paper.title == "Retriever Fusion for LLM Evaluation"
    assert [author.display_name for author in paper.authors] == ["Alice Smith", "Bob Lee"]
    assert paper.venue_type == "conference"
    assert paper.publication_status == "published"
    assert paper.venue_tier == "t1"
    assert paper.canonical_source == "dblp:conf/neurips/demo2025"


def test_openreview_adapter_filters_low_quality_notes_and_builds_absolute_urls(monkeypatch):
    adapter = OpenReviewAdapter()
    payload = {
        "notes": [
            {
                "id": "note-1",
                "content": {
                    "title": {"value": "Accepted LLM Evaluation Benchmarks"},
                    "venue": {"value": "ICLR 2026"},
                    "venueid": {"value": "ICLR.cc/2026/Conference"},
                    "authors": {"value": ["Alice Smith", "Bob Lee"]},
                    "abstract": {"value": "Benchmark study for evaluation pipelines."},
                    "pdf": {"value": "/pdf?id=note-1"},
                },
            },
            {
                "id": "note-2",
                "content": {
                    "title": {"value": "CoRR-only preprint"},
                    "venue": {"value": "CoRR"},
                    "venueid": {"value": "Public_Article"},
                    "authors": {"value": ["Ignored Author"]},
                },
            },
        ]
    }

    async def fake_client(*, headers=None):
        return FakeClient(payload)

    monkeypatch.setattr(adapter, "_client", fake_client)
    papers = asyncio.run(adapter.search("llm evaluation", project_id="project-2", limit=5))

    assert len(papers) == 1
    paper = papers[0]
    assert paper.provider == "openreview"
    assert paper.source_url == "https://openreview.net/forum?id=note-1"
    assert paper.oa_url == "https://openreview.net/pdf?id=note-1"
    assert paper.publication_status == "accepted"
    assert paper.is_preprint is False


def test_acl_anthology_adapter_filters_to_acl_family_results(monkeypatch):
    adapter = ACLAnthologyAdapter()
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "Reasoning with LLM Judges",
                "publication_year": 2025,
                "doi": "https://doi.org/10.18653/v1/2025.acl-long.1",
                "authorships": [{"author": {"display_name": "Alice Smith"}}],
                "cited_by_count": 12,
                "primary_location": {"source": {"display_name": "Annual Meeting of the Association for Computational Linguistics"}},
                "best_oa_location": {
                    "landing_page_url": "https://aclanthology.org/2025.acl-long.1/",
                    "pdf_url": "https://aclanthology.org/2025.acl-long.1.pdf",
                },
                "abstract_inverted_index": {"Reasoning": [0], "with": [1], "judges": [2]},
                "referenced_works": ["https://openalex.org/W2"],
            },
            {
                "id": "https://openalex.org/W2",
                "display_name": "Irrelevant Workshop Paper",
                "publication_year": 2025,
                "authorships": [{"author": {"display_name": "Ignored Author"}}],
                "primary_location": {"source": {"display_name": "Some Other Venue"}},
            },
        ]
    }

    async def fake_client(*, headers=None):
        return FakeClient(payload)

    monkeypatch.setattr(adapter, "_client", fake_client)
    papers = asyncio.run(adapter.search("llm judges", project_id="project-3", limit=5))

    assert len(papers) == 1
    paper = papers[0]
    assert paper.provider == "acl-anthology"
    assert paper.venue_tier == "t1"
    assert paper.publication_status == "published"
    assert paper.raw_source["upstream_provider"] == "openalex"


def test_semantic_scholar_adapter_normalizes_external_ids_without_api_key(monkeypatch):
    adapter = SemanticScholarAdapter()
    payload = {
        "data": [
            {
                "paperId": "s2-1",
                "title": "Multimodal Evaluation for LLM Agents",
                "abstract": "A study of agent evaluation for multimodal settings.",
                "authors": [{"name": "Alice Smith"}],
                "year": 2024,
                "venue": "ICML",
                "externalIds": {
                    "DOI": "10.1000/s2.demo",
                    "ArXiv": "2501.12345",
                    "PubMed": "12345678",
                },
                "url": "https://www.semanticscholar.org/paper/s2-1",
                "citationCount": 42,
                "openAccessPdf": {"url": "https://example.org/demo.pdf"},
            }
        ]
    }

    async def fake_client(*, headers=None):
        assert headers is None
        return FakeClient(payload)

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.setattr(adapter, "_client", fake_client)
    papers = asyncio.run(adapter.search("multimodal agents", project_id="project-4", limit=5))

    assert len(papers) == 1
    paper = papers[0]
    assert paper.doi == "10.1000/s2.demo"
    assert paper.arxiv_id == "2501.12345"
    assert paper.pmid == "12345678"
    assert paper.venue_type == "conference"
    assert paper.venue_tier == "t1"
