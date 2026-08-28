"""Deterministic BibTeX and LaTeX citation auditing utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_BIBTEX_ENTRY_RE = re.compile(r"@\s*([A-Za-z]+)\s*([{\(])\s*([^,\s{}()]+)\s*,", re.MULTILINE)
_LATEX_CITE_RE = re.compile(
    r"\\(?P<command>(?:[A-Za-z]*cite[A-Za-z]*|nocite))\s*(?:\[[^\]]*\]\s*){0,2}\{(?P<keys>[^{}]+)\}",
    re.MULTILINE,
)
_UNESCAPED_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_LATEX_SIMPLE_COMMAND_ARG_RE = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\]\s*)*\{([^{}]*)\}")
_LATEX_REMAINING_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\]\s*)*")
_IGNORED_BIBTEX_ENTRY_TYPES = {"comment", "preamble", "string"}
_EXPERIMENTAL_CLAIM_TERMS = {
    "ablation",
    "accuracy",
    "auc",
    "auroc",
    "baseline",
    "benchmark",
    "experiment",
    "experimental",
    "f1",
    "metric",
    "outperform",
    "performance",
    "result",
    "robust",
    "sota",
    "state-of-the-art",
    "superior",
}
_SUPPORTED_STATUSES = {"supported", "verified", "supported_by_experiment", "supported_by_literature", "supported_by_simulation"}
_EXPERIMENT_STATUSES = {"supported_by_experiment", "experiment-supported", "experiment_supported", "supported_by_simulation"}
_SIMULATION_STATUSES = {"supported_by_simulation", "simulation-supported", "simulation_supported"}
_SIMULATION_DISCLOSURE_KEYS = {
    "simulation_assumptions",
    "simulation_assumptions_path",
    "simulation_disclosure",
    "simulation_method",
    "simulated_experiment_contract",
}


@dataclass(frozen=True)
class BibTeXEntry:
    """Parsed BibTeX entry fields needed by deterministic quality gates."""

    entry_type: str
    key: str
    fields: dict[str, str]
    raw: str


@dataclass(frozen=True)
class CitationAuditResult:
    """Structured result for a LaTeX/BibTeX citation audit."""

    status: str
    bibtex_path: str
    tex_path: str | None
    citation_keys: list[str]
    cited_keys: list[str]
    missing_keys: list[str]
    unused_keys: list[str]
    nocite_all: bool
    violations: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    stale_claims: list[str] = field(default_factory=list)
    paragraph_count: int = 0
    uncited_paragraph_count: int = 0
    author_notes: list[str] = field(default_factory=list)
    reference_count: int = 0
    cited_reference_count: int = 0
    recent_reference_count: int = 0
    recent_year_cutoff: int | None = None
    invalid_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "bibtex_path": self.bibtex_path,
            "tex_path": self.tex_path,
            "citation_keys": self.citation_keys,
            "cited_keys": self.cited_keys,
            "missing_keys": self.missing_keys,
            "unused_keys": self.unused_keys,
            "nocite_all": self.nocite_all,
            "violations": self.violations,
            "unsupported_claims": self.unsupported_claims,
            "stale_claims": self.stale_claims,
            "paragraph_count": self.paragraph_count,
            "uncited_paragraph_count": self.uncited_paragraph_count,
            "author_notes": self.author_notes,
            "reference_count": self.reference_count,
            "cited_reference_count": self.cited_reference_count,
            "recent_reference_count": self.recent_reference_count,
            "recent_year_cutoff": self.recent_year_cutoff,
            "invalid_references": self.invalid_references,
            "metadata": self.metadata,
        }


def parse_bibtex_entries(source: str) -> dict[str, BibTeXEntry]:
    """Parse BibTeX entries and common fields without model inference."""

    entries: dict[str, BibTeXEntry] = {}
    for match in _BIBTEX_ENTRY_RE.finditer(source):
        entry_type = match.group(1).lower()
        if entry_type in _IGNORED_BIBTEX_ENTRY_TYPES:
            continue
        key = match.group(3).strip()
        if not key or key in entries:
            continue
        raw = _extract_raw_bibtex_entry(source, match)
        fields = _parse_bibtex_fields(raw[match.end() - match.start() : -1] if raw.endswith(("}", ")")) else raw)
        entries[key] = BibTeXEntry(entry_type=entry_type, key=key, fields=fields, raw=raw)
    return entries


def extract_bibtex_keys(source: str) -> list[str]:
    """Extract citation keys from BibTeX source without model inference."""

    return list(parse_bibtex_entries(source).keys())


def _extract_raw_bibtex_entry(source: str, match: re.Match[str]) -> str:
    opener = match.group(2)
    closer = "}" if opener == "{" else ")"
    depth = 1
    index = match.end()
    while index < len(source):
        char = source[index]
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1]
        index += 1
    return source[match.start() :]


def _parse_bibtex_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    index = 0
    length = len(body)
    while index < length:
        while index < length and body[index] in " \t\r\n,":
            index += 1
        field_start = index
        while index < length and re.match(r"[A-Za-z0-9_-]", body[index]):
            index += 1
        if field_start == index:
            index += 1
            continue
        field_name = body[field_start:index].strip().lower()
        while index < length and body[index].isspace():
            index += 1
        if index >= length or body[index] != "=":
            continue
        index += 1
        while index < length and body[index].isspace():
            index += 1
        value, index = _parse_bibtex_value(body, index)
        fields[field_name] = _normalize_bibtex_value(value)
    return fields


def _parse_bibtex_value(body: str, index: int) -> tuple[str, int]:
    if index >= len(body):
        return "", index
    if body[index] == "{":
        depth = 1
        index += 1
        start = index
        while index < len(body):
            char = body[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return body[start:index], index + 1
            index += 1
        return body[start:], index
    if body[index] == '"':
        index += 1
        start = index
        while index < len(body):
            if body[index] == '"' and (index == 0 or body[index - 1] != "\\"):
                return body[start:index], index + 1
            index += 1
        return body[start:], index

    start = index
    while index < len(body) and body[index] != ",":
        index += 1
    return body[start:index], index


def _normalize_bibtex_value(value: str) -> str:
    value = value.strip().strip("{}").strip()
    return re.sub(r"\s+", " ", value)


def extract_latex_citations(source: str) -> tuple[list[str], bool]:
    """Extract cited BibTeX keys from LaTeX source and flag ``\\nocite{*}``."""

    stripped = "\n".join(_UNESCAPED_COMMENT_RE.sub("", line) for line in source.splitlines())
    cited: list[str] = []
    seen: set[str] = set()
    nocite_all = False

    for match in _LATEX_CITE_RE.finditer(stripped):
        command = match.group("command").lower()
        raw_keys = match.group("keys")
        for raw_key in raw_keys.split(","):
            key = raw_key.strip()
            if not key:
                continue
            if command == "nocite" and key == "*":
                nocite_all = True
                continue
            if key not in seen:
                cited.append(key)
                seen.add(key)

    return cited, nocite_all


def _latex_visible_text(source: str) -> str:
    stripped = "\n".join(_UNESCAPED_COMMENT_RE.sub("", line) for line in source.splitlines())
    text = _LATEX_CITE_RE.sub(" ", stripped)
    text = re.sub(r"\\(?:begin|end)\{[^{}]*\}", " ", text)
    text = re.sub(r"\\([%&_$#{}])", r"\1", text)
    for _ in range(8):
        next_text = _LATEX_SIMPLE_COMMAND_ARG_RE.sub(r" \1 ", text)
        if next_text == text:
            break
        text = next_text
    text = _LATEX_REMAINING_COMMAND_RE.sub(" ", text)
    return text.replace("{", " ").replace("}", " ")


def _normalize_claim_match_text(text: str) -> str:
    lowered = text.casefold()
    without_punctuation = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", without_punctuation).strip()


def _claim_occurs_in_tex(claim_text: str, tex_source: str) -> bool:
    normalized_claim = _normalize_claim_match_text(claim_text)
    if not normalized_claim:
        return False
    stripped_source = "\n".join(_UNESCAPED_COMMENT_RE.sub("", line) for line in tex_source.splitlines())
    normalized_raw = _normalize_claim_match_text(stripped_source)
    normalized_visible = _normalize_claim_match_text(_latex_visible_text(tex_source))
    return normalized_claim in normalized_raw or normalized_claim in normalized_visible


def _claim_support_issue_texts(claims: Any) -> list[str]:
    """Return claim texts that are explicitly marked unsupported or lack evidence."""

    if isinstance(claims, dict):
        candidates = claims.get("claims", claims.get("claim_table", claims.get("items", [])))
        global_simulation_disclosure = _has_simulation_disclosure(claims)
    else:
        candidates = claims
        global_simulation_disclosure = False

    unsupported: list[str] = []
    if not isinstance(candidates, list):
        return unsupported

    for item in candidates:
        if not isinstance(item, dict):
            continue
        claim_text = str(item.get("claim") or item.get("text") or item.get("statement") or "").strip()
        status = str(item.get("support_status") or item.get("status") or "").strip().lower()
        evidence = item.get("evidence") or item.get("citations") or item.get("citation_keys") or item.get("artifact_path")
        evidence_type = str(item.get("evidence_type") or item.get("support_type") or "").strip().lower()
        experimental_claim = _is_experimental_claim(claim_text, item)
        lacks_evidence = evidence in (None, "", []) and status not in _SUPPORTED_STATUSES
        literature_only_experimental_claim = experimental_claim and status not in _EXPERIMENT_STATUSES and evidence_type != "experiment"
        simulation_claim_without_assumptions = status in _SIMULATION_STATUSES and not (global_simulation_disclosure or _has_simulation_disclosure(item))
        if status in {"unsupported", "contradicted", "missing"} or lacks_evidence:
            unsupported.append(claim_text or json.dumps(item, ensure_ascii=False, sort_keys=True))
        elif literature_only_experimental_claim:
            unsupported.append(claim_text or json.dumps(item, ensure_ascii=False, sort_keys=True))
        elif simulation_claim_without_assumptions:
            unsupported.append(claim_text or json.dumps(item, ensure_ascii=False, sort_keys=True))

    return unsupported


def find_unsupported_and_stale_claims(claims: Any, tex_source: str | None = None) -> tuple[list[str], list[str]]:
    """Split unsupported claim-map entries into active manuscript issues and stale entries."""

    unsupported: list[str] = []
    stale: list[str] = []
    for claim_text in _claim_support_issue_texts(claims):
        if tex_source is not None and _claim_occurs_in_tex(claim_text, tex_source):
            unsupported.append(claim_text)
        elif tex_source is not None and claim_text and not claim_text.lstrip().startswith("{"):
            stale.append(claim_text)
        else:
            unsupported.append(claim_text)
    return unsupported, stale


def find_unsupported_claims(claims: Any) -> list[str]:
    """Return unsupported claim-map entries using the legacy no-manuscript behavior."""

    unsupported, _ = find_unsupported_and_stale_claims(claims)
    return unsupported


def _has_simulation_disclosure(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if key_lower in _SIMULATION_DISCLOSURE_KEYS and item not in (None, "", [], {}):
                return True
            if key_lower in {"evidence", "artifact_path", "artifacts"} and _has_simulation_disclosure(item):
                return True
        return False
    if isinstance(value, list):
        return any(_has_simulation_disclosure(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(key in lowered for key in _SIMULATION_DISCLOSURE_KEYS)
    return False


def _is_experimental_claim(claim_text: str, item: dict[str, Any]) -> bool:
    text = " ".join(
        [
            claim_text,
            str(item.get("section") or ""),
            str(item.get("claim_type") or ""),
            str(item.get("evidence_type") or ""),
        ]
    ).lower()
    return any(term in text for term in _EXPERIMENTAL_CLAIM_TERMS)


def find_author_notes(source: str) -> list[str]:
    notes = []
    lowered = source.lower()
    for pattern in (
        "bibliography keys are synchronized",
        "citation keys are synchronized",
        "cannot actually",
        "i cannot",
        "i can't",
        "as an ai",
    ):
        if pattern in lowered:
            notes.append(pattern)
    return notes


def citation_paragraph_stats(source: str) -> tuple[int, int]:
    stripped = "\n".join(_UNESCAPED_COMMENT_RE.sub("", line) for line in source.splitlines())
    body_match = re.search(r"\\begin\{document\}(?P<body>.*)\\end\{document\}", stripped, flags=re.DOTALL)
    body = body_match.group("body") if body_match else stripped
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", body) if len(re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", "", paragraph).strip()) >= 80]
    uncited = [paragraph for paragraph in paragraphs if not _LATEX_CITE_RE.search(paragraph)]
    return len(paragraphs), len(uncited)


def audit_latex_citations(
    *,
    bibtex_path: Path,
    tex_path: Path | None = None,
    claim_map_path: Path | None = None,
    allow_nocite_all: bool = False,
    min_references: int = 0,
    min_cited_references: int = 0,
    min_recent_references: int = 0,
    recent_year_cutoff: int | None = None,
    required_reference_fields: tuple[str, ...] = (),
    required_cited_fields: tuple[str, ...] = (),
) -> CitationAuditResult:
    """Audit LaTeX citations against a BibTeX file."""

    bibtex_source = bibtex_path.read_text(encoding="utf-8")
    entries = parse_bibtex_entries(bibtex_source)
    citation_keys = list(entries.keys())
    cited_keys: list[str] = []
    nocite_all = False
    tex_source: str | None = None
    paragraph_count = 0
    uncited_paragraph_count = 0
    author_notes: list[str] = []
    resolved_recent_year_cutoff = recent_year_cutoff
    if min_recent_references > 0 and resolved_recent_year_cutoff is None:
        resolved_recent_year_cutoff = datetime.now().year - 10

    if tex_path is not None:
        tex_source = tex_path.read_text(encoding="utf-8")
        cited_keys, nocite_all = extract_latex_citations(tex_source)
        paragraph_count, uncited_paragraph_count = citation_paragraph_stats(tex_source)
        author_notes = find_author_notes(tex_source)

    missing_keys = sorted(key for key in cited_keys if key not in set(citation_keys))
    unused_keys = sorted(key for key in citation_keys if key not in set(cited_keys))

    unsupported_claims: list[str] = []
    stale_claims: list[str] = []
    if claim_map_path is not None and claim_map_path.exists():
        claim_map = json.loads(claim_map_path.read_text(encoding="utf-8"))
        unsupported_claims, stale_claims = find_unsupported_and_stale_claims(claim_map, tex_source)

    effective_cited_keys = citation_keys if nocite_all and allow_nocite_all else cited_keys
    recent_reference_count = _recent_reference_count(entries, resolved_recent_year_cutoff)
    invalid_references = _reference_field_violations(
        entries=entries,
        required_reference_fields=required_reference_fields,
        required_cited_fields=required_cited_fields,
        cited_keys=effective_cited_keys,
    )

    violations: list[str] = []
    if not citation_keys:
        violations.append("No BibTeX citation keys were found.")
    if min_references > 0 and len(citation_keys) < min_references:
        violations.append(f"BibTeX reference count below threshold: {len(citation_keys)}/{min_references}.")
    if min_cited_references > 0 and len(effective_cited_keys) < min_cited_references:
        violations.append(f"Inline cited reference count below threshold: {len(effective_cited_keys)}/{min_cited_references}.")
    if min_recent_references > 0 and recent_reference_count < min_recent_references:
        violations.append(f"Recent reference count below threshold: {recent_reference_count}/{min_recent_references} since {resolved_recent_year_cutoff}.")
    if invalid_references:
        violations.append("BibTeX entries are missing required fields: " + "; ".join(invalid_references) + ".")
    if missing_keys:
        violations.append(f"Missing BibTeX keys cited in LaTeX: {', '.join(missing_keys)}.")
    if nocite_all and not allow_nocite_all:
        violations.append(r"\nocite{*} is not allowed unless the user explicitly asks to include every reference.")
    if tex_path is not None and not cited_keys and not (nocite_all and allow_nocite_all):
        violations.append("No inline LaTeX citations were found in the manuscript body.")
    if paragraph_count and uncited_paragraph_count / paragraph_count > 0.5:
        violations.append(f"Too many manuscript paragraphs lack inline citations: {uncited_paragraph_count}/{paragraph_count}.")
    if unsupported_claims:
        violations.append(f"Unsupported manuscript claims: {len(unsupported_claims)}.")
    if author_notes:
        violations.append("Author/tool process notes remain in manuscript text: " + ", ".join(author_notes) + ".")

    metadata: dict[str, Any] = {}
    if claim_map_path is not None and claim_map_path.exists():
        metadata["claim_map_sync_status"] = "stale_entries" if stale_claims else "current"
        metadata["stale_claim_count"] = len(stale_claims)

    return CitationAuditResult(
        status="fail" if violations else "pass",
        bibtex_path=str(bibtex_path),
        tex_path=str(tex_path) if tex_path is not None else None,
        citation_keys=citation_keys,
        cited_keys=cited_keys,
        missing_keys=missing_keys,
        unused_keys=unused_keys,
        nocite_all=nocite_all,
        violations=violations,
        unsupported_claims=unsupported_claims,
        stale_claims=stale_claims,
        paragraph_count=paragraph_count,
        uncited_paragraph_count=uncited_paragraph_count,
        author_notes=author_notes,
        reference_count=len(citation_keys),
        cited_reference_count=len(effective_cited_keys),
        recent_reference_count=recent_reference_count,
        recent_year_cutoff=resolved_recent_year_cutoff,
        invalid_references=invalid_references,
        metadata=metadata,
    )


def _recent_reference_count(entries: dict[str, BibTeXEntry], recent_year_cutoff: int | None) -> int:
    if recent_year_cutoff is None:
        return 0
    count = 0
    for entry in entries.values():
        year = _entry_year(entry)
        if year is not None and year >= recent_year_cutoff:
            count += 1
    return count


def _entry_year(entry: BibTeXEntry) -> int | None:
    match = re.search(r"\d{4}", entry.fields.get("year", ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _reference_field_violations(
    *,
    entries: dict[str, BibTeXEntry],
    required_reference_fields: tuple[str, ...],
    required_cited_fields: tuple[str, ...],
    cited_keys: list[str],
) -> list[str]:
    violations: list[str] = []
    for key, entry in entries.items():
        missing = _missing_required_fields(entry, required_reference_fields)
        if missing:
            violations.append(f"{key} missing {', '.join(missing)}")
    for key in cited_keys:
        entry = entries.get(key)
        if entry is None:
            continue
        missing = _missing_required_fields(entry, required_cited_fields)
        if missing:
            violations.append(f"{key} missing {', '.join(missing)}")
    return list(dict.fromkeys(violations))


def _missing_required_fields(entry: BibTeXEntry, required_fields: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for field_name in required_fields:
        normalized = field_name.strip().lower()
        if not normalized:
            continue
        if normalized == "venue":
            present = any(_field_has_value(entry, candidate) for candidate in ("journal", "booktitle", "publisher", "school", "institution", "organization"))
        elif normalized in {"url_or_doi", "doi_or_url"}:
            present = _field_has_value(entry, "doi") or _field_has_value(entry, "url")
        else:
            present = _field_has_value(entry, normalized)
        if not present:
            missing.append(field_name)
    return missing


def _field_has_value(entry: BibTeXEntry, field_name: str) -> bool:
    value = entry.fields.get(field_name, "").strip()
    return bool(value and value not in {"-", "n/a", "N/A"})


def write_citation_audit(result: CitationAuditResult, output_path: Path) -> Path:
    """Write a citation audit JSON file and return its path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path
