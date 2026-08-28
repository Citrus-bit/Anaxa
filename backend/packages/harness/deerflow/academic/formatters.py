from __future__ import annotations

import re

from .types import PaperRecord, ReferenceEntry
from .utils import author_display_list, doi_url, normalize_whitespace, slugify

DEFAULT_REFERENCE_STYLE = "apa7"

REFERENCE_STYLE_LABELS: dict[str, str] = {
    "apa7": "APA 7",
    "mla9": "MLA 9",
    "chicago": "Chicago",
    "gbt7714": "GB/T 7714",
    "plain": "Plain text",
    "bibtex": "BibTeX",
}

_REFERENCE_STYLE_ALIASES: dict[str, str] = {
    "apa": "apa7",
    "apa7": "apa7",
    "apaedition7": "apa7",
    "americanpsychologicalassociation": "apa7",
    "mla": "mla9",
    "mla9": "mla9",
    "chicago": "chicago",
    "chicagoauthordate": "chicago",
    "chicagonotes": "chicago",
    "gb": "gbt7714",
    "gbt": "gbt7714",
    "gb7714": "gbt7714",
    "gbt7714": "gbt7714",
    "gbt77142015": "gbt7714",
    "gbt77142005": "gbt7714",
    "bib": "bibtex",
    "bibtex": "bibtex",
    "plain": "plain",
    "text": "plain",
}


def normalize_reference_style(style: str | None) -> str:
    if not style:
        return DEFAULT_REFERENCE_STYLE
    key = re.sub(r"[^a-z0-9]+", "", normalize_whitespace(style).lower())
    normalized = _REFERENCE_STYLE_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(REFERENCE_STYLE_LABELS)
        raise ValueError(f"Unsupported reference style {style!r}. Supported styles: {supported}.")
    return normalized


def reference_style_label(style: str | None) -> str:
    normalized = normalize_reference_style(style)
    return REFERENCE_STYLE_LABELS[normalized]


def _sentence_case(title: str) -> str:
    text = normalize_whitespace(title)
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _source_url(paper: PaperRecord) -> str | None:
    return doi_url(paper.doi) or paper.oa_url or paper.source_url


def _reference_completeness(paper: PaperRecord) -> str:
    if not paper.authors:
        return "missing-author"
    if paper.year is None:
        return "missing-year"
    if not normalize_whitespace(paper.venue):
        return "missing-venue"
    if not _source_url(paper):
        return "incomplete"
    return "complete"


def _included_in_final(completeness: str, paper: PaperRecord) -> bool:
    return completeness in {"complete", "missing-venue", "missing-year", "missing-author"} and bool(paper.provider and paper.canonical_source)


def _display_authors(paper: PaperRecord, *, fallback: str | None = None) -> str:
    names = [normalize_whitespace(author.display_name) for author in sorted(paper.authors, key=lambda item: item.ordinal) if normalize_whitespace(author.display_name)]
    if not names:
        return fallback or _sentence_case(paper.title)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def format_apa7_reference(paper: PaperRecord) -> ReferenceEntry:
    authors = author_display_list(paper.authors)
    completeness = "complete"

    if authors:
        if len(authors) <= 20:
            if len(authors) == 1:
                author_part = authors[0]
            else:
                author_part = ", ".join(authors[:-1]) + f", & {authors[-1]}"
        else:
            author_part = ", ".join(authors[:19]) + ", ... " + authors[-1]
    else:
        author_part = _sentence_case(paper.title)
        completeness = "missing-author"

    year_text = str(paper.year) if paper.year is not None else "n.d."
    if paper.year is None and completeness == "complete":
        completeness = "missing-year"

    title_text = _sentence_case(paper.title)
    venue_text = normalize_whitespace(paper.venue)
    doi_link = doi_url(paper.doi)
    url = _source_url(paper)

    if authors:
        parts = [f"{author_part} ({year_text}). {title_text}."]
    else:
        parts = [f"{author_part} ({year_text})."]

    if venue_text:
        parts.append(f"{venue_text}.")
    elif completeness == "complete":
        completeness = "missing-venue"

    if paper.is_preprint and not venue_text:
        parts.append("arXiv.")

    if url:
        parts.append(url)
    elif completeness == "complete":
        completeness = "incomplete"

    formatted = " ".join(part.strip() for part in parts if part.strip())
    included = _included_in_final(completeness, paper)
    return ReferenceEntry(
        paper_id=paper.paper_id,
        style="apa7",
        formatted_text=re.sub(r"\s+", " ", formatted).strip(),
        doi_url=doi_link,
        completeness=completeness,
        included_in_final=included,
    )


def format_mla9_reference(paper: PaperRecord) -> ReferenceEntry:
    completeness = _reference_completeness(paper)
    authors = _display_authors(paper)
    title = normalize_whitespace(paper.title)
    venue = normalize_whitespace(paper.venue)
    year = str(paper.year) if paper.year is not None else "n.d."
    url = _source_url(paper)
    parts = [f'{authors}. "{title}."']
    if venue:
        parts.append(f"{venue}, {year}.")
    else:
        parts.append(f"{year}.")
    if url:
        parts.append(url)
    formatted = " ".join(part.strip() for part in parts if part.strip())
    return ReferenceEntry(
        paper_id=paper.paper_id,
        style="mla9",
        formatted_text=re.sub(r"\s+", " ", formatted).strip(),
        doi_url=doi_url(paper.doi),
        completeness=completeness,
        included_in_final=_included_in_final(completeness, paper),
    )


def format_chicago_reference(paper: PaperRecord) -> ReferenceEntry:
    completeness = _reference_completeness(paper)
    authors = _display_authors(paper)
    title = normalize_whitespace(paper.title)
    venue = normalize_whitespace(paper.venue)
    year = str(paper.year) if paper.year is not None else "n.d."
    url = _source_url(paper)
    parts = [f'{authors}. "{title}."']
    if venue:
        parts.append(f"{venue} ({year}).")
    else:
        parts.append(f"({year}).")
    if url:
        parts.append(url)
    formatted = " ".join(part.strip() for part in parts if part.strip())
    return ReferenceEntry(
        paper_id=paper.paper_id,
        style="chicago",
        formatted_text=re.sub(r"\s+", " ", formatted).strip(),
        doi_url=doi_url(paper.doi),
        completeness=completeness,
        included_in_final=_included_in_final(completeness, paper),
    )


def format_gbt7714_reference(paper: PaperRecord) -> ReferenceEntry:
    completeness = _reference_completeness(paper)
    authors = _display_authors(paper)
    title = normalize_whitespace(paper.title)
    venue = normalize_whitespace(paper.venue)
    year = str(paper.year) if paper.year is not None else "n.d."
    url = _source_url(paper)
    source_type = "J"
    if paper.is_preprint or paper.provider == "arxiv":
        source_type = "OL"
    elif venue and any(word in venue.lower() for word in ("conference", "proceedings", "workshop")):
        source_type = "C"
    if venue:
        formatted = f"{authors}. {title}[{source_type}]. {venue}, {year}."
    else:
        formatted = f"{authors}. {title}[{source_type}]. {year}."
    if url:
        formatted = f"{formatted} {url}"
    return ReferenceEntry(
        paper_id=paper.paper_id,
        style="gbt7714",
        formatted_text=re.sub(r"\s+", " ", formatted).strip(),
        doi_url=doi_url(paper.doi),
        completeness=completeness,
        included_in_final=_included_in_final(completeness, paper),
    )


def format_plain_reference(paper: PaperRecord) -> ReferenceEntry:
    completeness = _reference_completeness(paper)
    authors = _display_authors(paper)
    year = str(paper.year) if paper.year is not None else "n.d."
    title = normalize_whitespace(paper.title)
    venue = normalize_whitespace(paper.venue)
    url = _source_url(paper)
    parts = [f"{authors} ({year}). {title}."]
    if venue:
        parts.append(venue + ".")
    if url:
        parts.append(url)
    formatted = " ".join(part.strip() for part in parts if part.strip())
    return ReferenceEntry(
        paper_id=paper.paper_id,
        style="plain",
        formatted_text=re.sub(r"\s+", " ", formatted).strip(),
        doi_url=doi_url(paper.doi),
        completeness=completeness,
        included_in_final=_included_in_final(completeness, paper),
    )


def format_reference(paper: PaperRecord, style: str | None = None) -> ReferenceEntry:
    normalized = normalize_reference_style(style)
    if normalized == "apa7":
        return format_apa7_reference(paper)
    if normalized == "mla9":
        return format_mla9_reference(paper)
    if normalized == "chicago":
        return format_chicago_reference(paper)
    if normalized == "gbt7714":
        return format_gbt7714_reference(paper)
    if normalized == "plain":
        return format_plain_reference(paper)
    if normalized == "bibtex":
        completeness = _reference_completeness(paper)
        return ReferenceEntry(
            paper_id=paper.paper_id,
            style="bibtex",
            formatted_text=format_bibtex_entry(paper),
            doi_url=doi_url(paper.doi),
            completeness=completeness,
            included_in_final=_included_in_final(completeness, paper),
        )
    raise AssertionError(f"Unhandled reference style: {normalized}")


def _bibtex_type(paper: PaperRecord) -> str:
    venue = (paper.venue or "").lower()
    if paper.is_preprint:
        return "misc"
    if "conference" in venue or "proceedings" in venue or "workshop" in venue:
        return "inproceedings"
    return "article"


def _bibtex_escape(value: str) -> str:
    return re.sub(r"([{}])", r"\\\1", value)


def format_bibtex_entry(paper: PaperRecord) -> str:
    author_names = []
    for author in sorted(paper.authors, key=lambda item: item.ordinal):
        if author.family_name and author.given_name:
            author_names.append(f"{author.family_name}, {author.given_name}")
        elif author.display_name:
            author_names.append(author.display_name)

    lead_author = paper.authors[0].family_name if paper.authors and paper.authors[0].family_name else "paper"
    key = f"{slugify(lead_author)}{paper.year or 'nd'}{slugify((paper.title or 'ref').split(' ')[0])}"

    fields: list[tuple[str, str]] = [
        ("title", paper.title),
    ]
    if author_names:
        fields.append(("author", " and ".join(author_names)))
    if paper.year is not None:
        fields.append(("year", str(paper.year)))
    if paper.venue:
        fields.append(("journal" if _bibtex_type(paper) == "article" else "booktitle", paper.venue))
    if paper.doi:
        fields.append(("doi", paper.doi))
    if url := (doi_url(paper.doi) or paper.oa_url or paper.source_url):
        fields.append(("url", url))
    if paper.abstract:
        fields.append(("abstract", paper.abstract))

    rendered_fields = ",\n".join(f"  {name} = {{{_bibtex_escape(value)}}}" for name, value in fields if value)
    return f"@{_bibtex_type(paper)}{{{key},\n{rendered_fields}\n}}"
