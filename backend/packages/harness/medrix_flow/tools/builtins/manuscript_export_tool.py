"""One-shot LaTeX manuscript export tool."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX
from medrix_flow.utils.citations import (
    CitationAuditResult,
    audit_latex_citations,
    extract_latex_citations,
    parse_bibtex_entries,
    write_citation_audit,
)
from medrix_flow.utils.latex import (
    PdfTextAuditResult,
    audit_pdf_text_extractability,
    compile_latex_to_pdf,
    prepare_latex_preview,
    write_pdf_text_audit,
)
from medrix_flow.utils.manuscript_quality import (
    ManuscriptQualityAuditResult,
    audit_manuscript_quality,
    write_manuscript_quality_audit,
)

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
_SAFE_FILENAME_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_QUALITY_DISABLED_VALUES = {"", "none", "off", "false", "disabled"}
_FORMAL_REVIEW_PROFILE_VALUES = {"academic_review", "formal_review", "review", "review_article", "survey", "paper"}
_ENGLISH_FORMAL_REVIEW_TERMS = (
    "literature review",
    "review article",
    "systematic review",
    "survey paper",
    "research survey",
)
_CHINESE_FORMAL_REVIEW_TERMS = ("综述", "文献综述", "论文", "研究进展", "参考文献")


@dataclass(frozen=True)
class ReferencePolicy:
    min_references: int = 15
    min_cited_references: int = 15
    min_recent_references: int = 8
    required_cited_fields: tuple[str, ...] = ("author", "title", "year")


@dataclass(frozen=True)
class AcademicReferenceBundle:
    bibtex_path: Path
    content: str
    keys: list[str]
    canonical_reference_count: int | None
    retrieval_audit_path: Path | None
    references_md_path: Path | None
    evidence_map_path: Path | None


@dataclass(frozen=True)
class ReferenceResolution:
    bibtex_content: str
    academic_bundle: AcademicReferenceBundle | None
    metadata: dict[str, Any]


def _safe_filename_stem(filename_stem: str | None) -> str:
    stem = (filename_stem or "manuscript").strip()
    if stem.lower().endswith((".tex", ".pdf")):
        stem = stem.rsplit(".", 1)[0]

    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError("filename_stem must be a filename stem, not a path")
    if not _SAFE_FILENAME_STEM_RE.fullmatch(stem):
        raise ValueError("filename_stem may only contain ASCII letters, numbers, dots, underscores, and hyphens")
    return stem


def _outputs_dir(runtime: ToolRuntime[ContextT, ThreadState]) -> Path:
    if runtime.state is None:
        raise ValueError("Thread runtime state is not available")
    if not runtime.context.get("thread_id"):
        raise ValueError("Thread ID is not available in runtime context")

    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("Thread outputs path is not available in runtime state")

    outputs_dir = Path(outputs_path).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir


def _virtual_output_path(outputs_dir: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Tool attempted to write outside {OUTPUTS_VIRTUAL_PREFIX}: {path}") from exc
    return f"{OUTPUTS_VIRTUAL_PREFIX}/{relative.as_posix()}"


def _parse_claim_map(claim_map_json: str | None) -> Any | None:
    if claim_map_json is None or not claim_map_json.strip():
        return None
    try:
        return json.loads(claim_map_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"claim_map_json is not valid JSON: {exc}") from exc


def _write_claim_map(claim_map: Any | None, claim_map_path: Path) -> Path | None:
    if claim_map is None:
        return None
    claim_map_path.write_text(json.dumps(claim_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return claim_map_path


def _write_manuscript_inputs(
    *,
    tex_path: Path,
    bibtex_path: Path,
    claim_map_path: Path,
    tex_content: str,
    bibtex_content: str,
    claim_map: Any | None,
) -> Path | None:
    tex_path.write_text(tex_content, encoding="utf-8")
    bibtex_path.write_text(bibtex_content, encoding="utf-8")
    return _write_claim_map(claim_map, claim_map_path)


def _reference_policy_for(tex_content: str, quality_profile: str | None) -> ReferencePolicy | None:
    requested_profile = (quality_profile or "auto").strip().lower()
    if requested_profile in _QUALITY_DISABLED_VALUES:
        return None
    if requested_profile in _FORMAL_REVIEW_PROFILE_VALUES or _looks_like_formal_review(tex_content):
        return ReferencePolicy()
    return None


def _looks_like_formal_review(tex_content: str) -> bool:
    lowered = tex_content.lower()
    if any(term in lowered for term in _ENGLISH_FORMAL_REVIEW_TERMS):
        return True
    return bool(_CJK_RE.search(tex_content) and any(term in tex_content for term in _CHINESE_FORMAL_REVIEW_TERMS))


def _resolve_references(
    *,
    outputs_dir: Path,
    tex_content: str,
    bibtex_content: str,
    reference_policy: ReferencePolicy | None,
) -> ReferenceResolution:
    if reference_policy is None:
        return ReferenceResolution(bibtex_content=bibtex_content, academic_bundle=None, metadata={})

    academic_bundle = _find_best_academic_reference_bundle(outputs_dir)
    if academic_bundle is None:
        return ReferenceResolution(bibtex_content=bibtex_content, academic_bundle=None, metadata={})

    input_entries = parse_bibtex_entries(bibtex_content)
    cited_keys, _ = extract_latex_citations(tex_content)
    cited_key_set = set(cited_keys)
    academic_key_set = set(academic_bundle.keys)
    reuse_mode = "input"
    resolved_bibtex_content = bibtex_content
    if len(academic_bundle.keys) > len(input_entries):
        if cited_key_set and cited_key_set.issubset(academic_key_set):
            resolved_bibtex_content = academic_bundle.content
            reuse_mode = "academic_references"
        else:
            resolved_bibtex_content = _merge_bibtex_sources(bibtex_content, academic_bundle.content)
            reuse_mode = "merged_academic_references"

    final_entries = parse_bibtex_entries(resolved_bibtex_content)
    cited_overlap_count = len(cited_key_set & academic_key_set)
    metadata = {
        "reference_resolution": {
            "mode": reuse_mode,
            "input_reference_count": len(input_entries),
            "academic_reference_count": len(academic_bundle.keys),
            "final_reference_count": len(final_entries),
            "academic_cited_overlap_count": cited_overlap_count,
            "cited_key_count": len(cited_keys),
            "academic_bibtex_path": _virtual_output_path(outputs_dir, academic_bundle.bibtex_path),
            "canonical_reference_count": academic_bundle.canonical_reference_count,
            "retrieval_audit_path": _optional_virtual_output_path(outputs_dir, academic_bundle.retrieval_audit_path),
            "references_md_path": _optional_virtual_output_path(outputs_dir, academic_bundle.references_md_path),
            "evidence_map_path": _optional_virtual_output_path(outputs_dir, academic_bundle.evidence_map_path),
        }
    }
    return ReferenceResolution(
        bibtex_content=resolved_bibtex_content,
        academic_bundle=academic_bundle,
        metadata=metadata,
    )


def _find_best_academic_reference_bundle(outputs_dir: Path) -> AcademicReferenceBundle | None:
    candidates: list[AcademicReferenceBundle] = []
    for bibtex_path in outputs_dir.glob("academic-research/**/references.bib"):
        try:
            content = bibtex_path.read_text(encoding="utf-8")
        except OSError:
            continue
        keys = list(parse_bibtex_entries(content).keys())
        if not keys:
            continue
        retrieval_audit_path = bibtex_path.with_name("retrieval_audit.json")
        candidates.append(
            AcademicReferenceBundle(
                bibtex_path=bibtex_path,
                content=content,
                keys=keys,
                canonical_reference_count=_read_canonical_reference_count(retrieval_audit_path),
                retrieval_audit_path=retrieval_audit_path if retrieval_audit_path.exists() else None,
                references_md_path=_existing_sibling(bibtex_path, "references.md"),
                evidence_map_path=_existing_sibling(bibtex_path, "evidence_map.json"),
            )
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (item.canonical_reference_count or len(item.keys), len(item.keys), item.bibtex_path.stat().st_mtime),
    )


def _existing_sibling(path: Path, name: str) -> Path | None:
    sibling = path.with_name(name)
    return sibling if sibling.exists() else None


def _read_canonical_reference_count(retrieval_audit_path: Path) -> int | None:
    if not retrieval_audit_path.exists():
        return None
    try:
        payload = json.loads(retrieval_audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("canonical_reference_count")
    if isinstance(value, int):
        return value
    return None


def _merge_bibtex_sources(primary: str, secondary: str) -> str:
    primary_entries = parse_bibtex_entries(primary)
    secondary_entries = parse_bibtex_entries(secondary)
    pieces = [entry.raw.strip() for entry in primary_entries.values()]
    for key, entry in secondary_entries.items():
        if key not in primary_entries:
            pieces.append(entry.raw.strip())
    return "\n\n".join(piece for piece in pieces if piece) + "\n"


def _optional_virtual_output_path(outputs_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return _virtual_output_path(outputs_dir, path)


def _apply_reference_consistency(
    *,
    result: CitationAuditResult,
    reference_policy: ReferencePolicy | None,
    academic_bundle: AcademicReferenceBundle | None,
    metadata: dict[str, Any],
) -> CitationAuditResult:
    violations = list(result.violations)
    if reference_policy is not None and academic_bundle is not None:
        academic_keys = set(academic_bundle.keys)
        cited_keys = set(result.cited_keys)
        cited_overlap_count = len(cited_keys & academic_keys)
        if cited_keys and cited_overlap_count / len(cited_keys) < 0.5:
            violations.append(f"Final manuscript citations are detached from academic_research references: {cited_overlap_count}/{len(cited_keys)} cited keys exist in the academic references.bib.")
        canonical_count = academic_bundle.canonical_reference_count
        if canonical_count is not None and canonical_count >= reference_policy.min_references:
            if result.reference_count < reference_policy.min_references:
                violations.append(f"Final references.bib is smaller than the academic_research corpus: {result.reference_count} final entries vs {canonical_count} canonical references.")

    return replace(
        result,
        status="fail" if violations else "pass",
        violations=list(dict.fromkeys(violations)),
        metadata={**result.metadata, **metadata},
    )


def _artifact_paths(outputs_dir: Path, *paths: Path) -> list[str]:
    return [_virtual_output_path(outputs_dir, path) for path in paths]


def _format_audit_failure(result: CitationAuditResult) -> str:
    parts = ["FAIL: citation audit blocked PDF generation."]
    if result.violations:
        parts.append("Violations: " + " ".join(result.violations))
    if result.missing_keys:
        parts.append("Missing keys: " + ", ".join(result.missing_keys) + ".")
    if result.unsupported_claims:
        parts.append(f"Unsupported claims: {len(result.unsupported_claims)}.")
    return " ".join(parts)


def _format_quality_failure(result: ManuscriptQualityAuditResult) -> str:
    parts = ["FAIL: manuscript quality audit blocked PDF generation."]
    if result.violations:
        parts.append("Violations: " + " ".join(result.violations))
    if result.recommendations:
        parts.append("Required repairs: " + " ".join(result.recommendations))
    return " ".join(parts)


def _format_pdf_text_failure(result: PdfTextAuditResult) -> str:
    parts = ["FAIL: PDF text audit blocked PDF generation."]
    if result.violations:
        parts.append("Violations: " + " ".join(result.violations))
    if result.error:
        parts.append(f"Extraction note: {result.error}.")
    return " ".join(parts)


def _compile_pdf(tex_path: Path, bibtex_path: Path) -> Path:
    prepared_path = prepare_latex_preview(tex_path)
    prepared_bibtex_path = prepared_path.parent / bibtex_path.name
    if prepared_bibtex_path.resolve() != bibtex_path.resolve():
        shutil.copy2(bibtex_path, prepared_bibtex_path)

    preview_pdf = compile_latex_to_pdf(prepared_path, prepared_path.parent)
    final_pdf = tex_path.with_suffix(".pdf")
    if preview_pdf.resolve() != final_pdf.resolve():
        shutil.copy2(preview_pdf, final_pdf)
    return final_pdf


@tool("manuscript_export", parse_docstring=True)
def manuscript_export_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tex_content: str,
    bibtex_content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    claim_map_json: str | None = None,
    filename_stem: str = "manuscript",
    allow_nocite_all: bool = False,
    quality_profile: str = "auto",
) -> Command:
    """Write a LaTeX manuscript bundle, audit citations, compile PDF, and present artifacts.

    Use this tool for final paper, review article, experiment-paper, or formal
    manuscript deliverables. It writes the manuscript files under
    `/mnt/user-data/outputs`, validates BibTeX citation keys, blocks unsafe
    `\\nocite{*}` fallbacks by default, compiles the PDF when the audit passes,
    and returns the generated artifacts to the user.

    Args:
        tex_content: Full LaTeX source for the manuscript.
        bibtex_content: Full BibTeX source for `references.bib`.
        claim_map_json: Optional JSON claim/evidence map used to block unsupported claims.
        filename_stem: Safe output filename stem for `{filename_stem}.tex` and `{filename_stem}.pdf`.
        allow_nocite_all: Set true only when the user explicitly requested `\\nocite{*}`.
        quality_profile: Manuscript quality profile. Use `auto` by default; mathematical
            modeling competition manuscripts are automatically audited for page,
            figure, table, reference, recency, and simulation-evidence coverage.
    """
    try:
        stem = _safe_filename_stem(filename_stem)
        claim_map = _parse_claim_map(claim_map_json)
        outputs_dir = _outputs_dir(runtime)
        reference_policy = _reference_policy_for(tex_content, quality_profile)
        reference_resolution = _resolve_references(
            outputs_dir=outputs_dir,
            tex_content=tex_content,
            bibtex_content=bibtex_content,
            reference_policy=reference_policy,
        )
        resolved_bibtex_content = reference_resolution.bibtex_content

        tex_path = outputs_dir / f"{stem}.tex"
        bibtex_path = outputs_dir / "references.bib"
        audit_path = outputs_dir / "citation_audit.json"
        pdf_text_audit_path = outputs_dir / "pdf_text_audit.json"
        quality_audit_path = outputs_dir / "manuscript_quality_audit.json"
        claim_map_path = outputs_dir / "claim_map.json"
        final_pdf_path = tex_path.with_suffix(".pdf")
        if final_pdf_path.exists():
            final_pdf_path.unlink()

        resolved_claim_map_path = _write_manuscript_inputs(
            tex_path=tex_path,
            bibtex_path=bibtex_path,
            claim_map_path=claim_map_path,
            tex_content=tex_content,
            bibtex_content=resolved_bibtex_content,
            claim_map=claim_map,
        )

        audit_kwargs: dict[str, Any] = {}
        if reference_policy is not None:
            audit_kwargs = {
                "min_references": reference_policy.min_references,
                "min_cited_references": reference_policy.min_cited_references,
                "min_recent_references": reference_policy.min_recent_references,
                "required_cited_fields": reference_policy.required_cited_fields,
            }
        audit_result = audit_latex_citations(
            bibtex_path=bibtex_path,
            tex_path=tex_path,
            claim_map_path=resolved_claim_map_path,
            allow_nocite_all=allow_nocite_all,
            **audit_kwargs,
        )
        audit_result = _apply_reference_consistency(
            result=audit_result,
            reference_policy=reference_policy,
            academic_bundle=reference_resolution.academic_bundle,
            metadata=reference_resolution.metadata,
        )
        write_citation_audit(audit_result, audit_path)

        if not audit_result.passed:
            artifacts = _artifact_paths(outputs_dir, audit_path, tex_path, bibtex_path)
            return Command(
                update={
                    "artifacts": artifacts,
                    "messages": [ToolMessage(_format_audit_failure(audit_result), tool_call_id=tool_call_id)],
                }
            )

        preflight_quality = audit_manuscript_quality(
            tex_content=tex_content,
            bibtex_content=resolved_bibtex_content,
            claim_map=claim_map,
            quality_profile=quality_profile,
        )
        if preflight_quality.applied:
            write_manuscript_quality_audit(preflight_quality, quality_audit_path)
        if not preflight_quality.passed:
            artifacts = _artifact_paths(outputs_dir, quality_audit_path, audit_path, tex_path, bibtex_path)
            return Command(
                update={
                    "artifacts": artifacts,
                    "messages": [ToolMessage(_format_quality_failure(preflight_quality), tool_call_id=tool_call_id)],
                }
            )

        try:
            final_pdf_path = _compile_pdf(tex_path, bibtex_path)
        except Exception as exc:
            if final_pdf_path.exists():
                final_pdf_path.unlink()
            artifacts = _artifact_paths(outputs_dir, tex_path, bibtex_path, audit_path)
            message = f"FAIL: LaTeX compilation failed after citation audit passed: {exc}"
            return Command(
                update={
                    "artifacts": artifacts,
                    "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
                }
            )

        final_quality = audit_manuscript_quality(
            tex_content=tex_content,
            bibtex_content=resolved_bibtex_content,
            claim_map=claim_map,
            quality_profile=quality_profile,
            pdf_path=final_pdf_path,
        )
        if final_quality.applied:
            write_manuscript_quality_audit(final_quality, quality_audit_path)
        if not final_quality.passed:
            if final_pdf_path.exists():
                final_pdf_path.unlink()
            artifacts = _artifact_paths(outputs_dir, quality_audit_path, audit_path, tex_path, bibtex_path)
            return Command(
                update={
                    "artifacts": artifacts,
                    "messages": [ToolMessage(_format_quality_failure(final_quality), tool_call_id=tool_call_id)],
                }
            )

        pdf_text_audit_result: PdfTextAuditResult | None = None
        if reference_policy is not None or _CJK_RE.search(tex_content):
            pdf_text_audit_result = audit_pdf_text_extractability(final_pdf_path)
            write_pdf_text_audit(pdf_text_audit_result, pdf_text_audit_path)
            if not pdf_text_audit_result.passed:
                if final_pdf_path.exists():
                    final_pdf_path.unlink()
                artifact_files = [pdf_text_audit_path]
                if final_quality.applied:
                    artifact_files.append(quality_audit_path)
                artifact_files.extend([audit_path, tex_path, bibtex_path])
                artifacts = _artifact_paths(outputs_dir, *artifact_files)
                return Command(
                    update={
                        "artifacts": artifacts,
                        "messages": [ToolMessage(_format_pdf_text_failure(pdf_text_audit_result), tool_call_id=tool_call_id)],
                    }
                )

        artifact_files = [final_pdf_path, tex_path, bibtex_path, audit_path]
        if final_quality.applied:
            artifact_files.append(quality_audit_path)
        if pdf_text_audit_result is not None:
            artifact_files.append(pdf_text_audit_path)
        artifacts = _artifact_paths(outputs_dir, *artifact_files)
        message = f"PASS: manuscript_export wrote `{_virtual_output_path(outputs_dir, final_pdf_path)}`. BibTeX keys: {len(audit_result.citation_keys)}; cited keys: {len(audit_result.cited_keys)}."
        if audit_result.stale_claims:
            message += f" Citation audit noted {len(audit_result.stale_claims)} stale claim_map entr{'y' if len(audit_result.stale_claims) == 1 else 'ies'} that no longer appear in the manuscript."
        return Command(
            update={
                "artifacts": artifacts,
                "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
            }
        )
    except Exception as exc:
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})
