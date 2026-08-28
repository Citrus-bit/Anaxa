"""Deterministic manuscript quality audits for formal paper exports."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_BIBTEX_ENTRY_RE = re.compile(r"@\s*[A-Za-z]+\s*[{(]\s*([^,\s{}()]+)\s*,", re.MULTILINE)
_BIBTEX_YEAR_RE = re.compile(r"\byear\s*=\s*[{\"]?\s*(\d{4})", re.IGNORECASE)
_LATEX_VISIBLE_COMMAND_RE = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|subparagraph|caption|"
    r"textbf|textit|emph|underline|texttt|textsc|textsf|textrm)\*?"
    r"(?:\[[^\]]*\])?\{([^{}]*)\}"
)
_LATEX_NON_TEXT_COMMAND_WITH_ARG_RE = re.compile(
    r"\\(?:bibliography|bibliographystyle|cite|citet|citep|citealp|citeauthor|citeyear|"
    r"ref|eqref|autoref|cref|Cref|label|includegraphics|url|href|input|include|"
    r"addbibresource)(?:\[[^\]]*\])*(?:\{[^{}]*\}){1,2}"
)
_LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])*")
_LATEX_INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
_LATEX_DISPLAY_MATH_RE = re.compile(
    r"\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\\begin\{(?:equation|align|gather|multline)\*?\}[\s\S]*?\\end\{(?:equation|align|gather|multline)\*?\}",
    re.MULTILINE,
)
_LATEX_ENV_RE = re.compile(r"\\(?:begin|end)\{[^{}]+\}")
_LATEX_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|[\u4e00-\u9fff]")

MATH_MODELING_QUALITY_PROFILE = "math_modeling_competition"
_AUTO_PROFILE_TERMS = (
    "mathematical contest in modeling",
    "interdisciplinary contest in modeling",
    "mathematical modeling competition",
    "math modeling competition",
    "modeling contest",
    "modeling competition manuscript",
    "数学建模",
    "建模大赛",
    "建模竞赛",
    "数学模型竞赛",
    "数学模型竞赛论文",
)
_AUTO_PROFILE_PATTERNS = (re.compile(r"\b(?:mcm|icm)\b", re.IGNORECASE),)
_SYNTHETIC_TERMS = (
    "synthetic experiment mode",
    "synthetic_data_mode",
    "supported_by_simulation",
    "simulated_experiment",
    "simulation_assumptions",
    "synthetic_results",
)


@dataclass(frozen=True)
class ManuscriptQualityThresholds:
    min_words: int = 4500
    min_pages: int = 10
    min_figures: int = 5
    min_tables: int = 5
    min_references: int = 15
    min_recent_references: int = 8
    recent_year_cutoff: int = field(default_factory=lambda: datetime.now().year - 10)


@dataclass(frozen=True)
class ManuscriptQualityAuditResult:
    status: str
    profile: str | None
    applied: bool
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    violations: list[str]
    recommendations: list[str]

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "profile": self.profile,
            "applied": self.applied,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "violations": self.violations,
            "recommendations": self.recommendations,
        }


def resolve_quality_profile(tex_content: str, quality_profile: str | None) -> str | None:
    requested = (quality_profile or "auto").strip().lower()
    if requested in {"", "none", "off", "false", "disabled"}:
        return None
    if requested in {"math", "math_modeling", "mathematical_modeling", MATH_MODELING_QUALITY_PROFILE}:
        return MATH_MODELING_QUALITY_PROFILE
    if requested != "auto":
        return requested

    lowered = tex_content.lower()
    if any(term in lowered for term in _AUTO_PROFILE_TERMS) or any(pattern.search(tex_content) for pattern in _AUTO_PROFILE_PATTERNS):
        return MATH_MODELING_QUALITY_PROFILE
    return None


def audit_manuscript_quality(
    *,
    tex_content: str,
    bibtex_content: str,
    claim_map: Any | None = None,
    quality_profile: str | None = "auto",
    pdf_path: Path | None = None,
    thresholds: ManuscriptQualityThresholds | None = None,
) -> ManuscriptQualityAuditResult:
    profile = resolve_quality_profile(tex_content, quality_profile)
    if profile != MATH_MODELING_QUALITY_PROFILE:
        return ManuscriptQualityAuditResult(
            status="pass",
            profile=profile,
            applied=False,
            metrics={},
            thresholds={},
            violations=[],
            recommendations=[],
        )

    resolved_thresholds = thresholds or ManuscriptQualityThresholds()
    metrics = _quality_metrics(tex_content, bibtex_content, claim_map, pdf_path, resolved_thresholds.recent_year_cutoff)
    violations: list[str] = []
    recommendations: list[str] = []

    checks = [
        ("word_count", resolved_thresholds.min_words, "Expand the manuscript body with full modeling derivations, algorithm design, result interpretation, sensitivity analysis, and limitations."),
        ("figure_count", resolved_thresholds.min_figures, "Add required figures: parameter sensitivity, Monte Carlo convergence, probability heatmap, scheme comparison, and robustness/error analysis."),
        ("table_count", resolved_thresholds.min_tables, "Add required tables for notation, parameter grid, simulation results, ablation, robustness, and error analysis."),
        ("reference_count", resolved_thresholds.min_references, "Run academic_research and cite at least 15 verifiable sources."),
        ("recent_reference_count", resolved_thresholds.min_recent_references, "Repair the bibliography with at least 8 references from the last 10 years."),
    ]
    if pdf_path is not None:
        checks.insert(1, ("page_count", resolved_thresholds.min_pages, "Extend the compiled PDF to at least 10 pages for a formal modeling competition manuscript."))

    for metric_name, minimum, recommendation in checks:
        if metric_name == "page_count" and metrics.get("page_count_status") not in {None, "ok"}:
            status = str(metrics.get("page_count_status") or "unknown")
            error = str(metrics.get("page_count_error") or "page count could not be read")
            violations.append(f"page_count unavailable ({status}): {error}.")
            recommendations.append(recommendation)
            continue
        value = int(metrics.get(metric_name) or 0)
        if value < minimum:
            violations.append(f"{metric_name} below threshold: {value}/{minimum}.")
            recommendations.append(recommendation)

    if metrics.get("synthetic_mode_detected"):
        synthetic_requirements = {
            "has_simulation_assumptions": "Include simulation_assumptions.json or equivalent assumption metadata in the claim map.",
            "has_synthetic_results": "Include synthetic_results.csv/json in the experiment evidence bundle.",
            "has_ablation_or_robustness": "Include ablation_results.json and robustness_results.json or equivalent simulated analysis artifacts.",
            "has_error_analysis": "Include error_analysis.md or a concrete error-analysis artifact.",
        }
        for metric_name, recommendation in synthetic_requirements.items():
            if not metrics.get(metric_name):
                violations.append(f"{metric_name} is required for synthetic math-modeling manuscripts.")
                recommendations.append(recommendation)

    return ManuscriptQualityAuditResult(
        status="fail" if violations else "pass",
        profile=profile,
        applied=True,
        metrics=metrics,
        thresholds={
            "min_words": resolved_thresholds.min_words,
            "min_pages": resolved_thresholds.min_pages,
            "min_figures": resolved_thresholds.min_figures,
            "min_tables": resolved_thresholds.min_tables,
            "min_references": resolved_thresholds.min_references,
            "min_recent_references": resolved_thresholds.min_recent_references,
            "recent_year_cutoff": resolved_thresholds.recent_year_cutoff,
        },
        violations=list(dict.fromkeys(violations)),
        recommendations=list(dict.fromkeys(recommendations)),
    )


def write_manuscript_quality_audit(result: ManuscriptQualityAuditResult, output_path: Path) -> Path:
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def _quality_metrics(
    tex_content: str,
    bibtex_content: str,
    claim_map: Any | None,
    pdf_path: Path | None,
    recent_year_cutoff: int,
) -> dict[str, Any]:
    reference_years = _bibtex_years(bibtex_content)
    claim_map_text = json.dumps(claim_map, ensure_ascii=False).lower() if claim_map is not None else ""
    combined_text = f"{tex_content}\n{bibtex_content}\n{claim_map_text}".lower()
    page_count_result = _pdf_page_count_result(pdf_path)
    return {
        "word_count": _latex_word_count(tex_content),
        "page_count": page_count_result["page_count"],
        "page_count_status": page_count_result["page_count_status"],
        "page_count_error": page_count_result["page_count_error"],
        "figure_count": _latex_figure_count(tex_content),
        "table_count": _latex_table_count(tex_content),
        "reference_count": len(_BIBTEX_ENTRY_RE.findall(bibtex_content)),
        "recent_reference_count": sum(1 for year in reference_years if year >= recent_year_cutoff),
        "recent_year_cutoff": recent_year_cutoff,
        "synthetic_mode_detected": any(term in combined_text for term in _SYNTHETIC_TERMS),
        "has_simulation_assumptions": "simulation_assumptions" in combined_text,
        "has_synthetic_results": "synthetic_results" in combined_text,
        "has_ablation_or_robustness": "ablation_results" in combined_text or "robustness_results" in combined_text,
        "has_error_analysis": "error_analysis" in combined_text,
    }


def _latex_word_count(tex_content: str) -> int:
    body_match = re.search(r"\\begin\{document\}(?P<body>.*)\\end\{document\}", tex_content, flags=re.DOTALL)
    body = body_match.group("body") if body_match else tex_content
    body = _LATEX_COMMENT_RE.sub(" ", body)
    for _ in range(4):
        next_body = _LATEX_VISIBLE_COMMAND_RE.sub(r" \1 ", body)
        if next_body == body:
            break
        body = next_body
    body = _LATEX_INLINE_MATH_RE.sub(" ", body)
    body = _LATEX_DISPLAY_MATH_RE.sub(" ", body)
    body = _LATEX_NON_TEXT_COMMAND_WITH_ARG_RE.sub(" ", body)
    body = _LATEX_ENV_RE.sub(" ", body)
    body = _LATEX_COMMAND_RE.sub(" ", body)
    body = body.replace("{", " ").replace("}", " ")
    return len(_WORD_RE.findall(body))


def _latex_figure_count(tex_content: str) -> int:
    figure_envs = len(re.findall(r"\\begin\{figure\*?\}", tex_content))
    includegraphics = len(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{[^{}]+\}", tex_content))
    tikz = len(re.findall(r"\\begin\{tikzpicture\}", tex_content))
    return max(figure_envs, includegraphics + tikz)


def _latex_table_count(tex_content: str) -> int:
    table_envs = len(re.findall(r"\\begin\{table\*?\}", tex_content))
    tabulars = len(re.findall(r"\\begin\{(?:tabular|longtable|tabularx)\}", tex_content))
    return max(table_envs, tabulars)


def _bibtex_years(bibtex_content: str) -> list[int]:
    years: list[int] = []
    for match in _BIBTEX_YEAR_RE.finditer(bibtex_content):
        try:
            years.append(int(match.group(1)))
        except ValueError:
            continue
    return years


def _pdf_page_count_result(pdf_path: Path | None) -> dict[str, Any]:
    if pdf_path is None:
        return {"page_count": None, "page_count_status": "not_requested", "page_count_error": None}
    if not pdf_path.exists():
        return {"page_count": None, "page_count_status": "missing", "page_count_error": f"{pdf_path} does not exist"}

    errors: list[str] = []
    try:
        count = _pdf_page_count(pdf_path)
        if count is not None:
            return {"page_count": count, "page_count_status": "ok", "page_count_error": None}
    except Exception as exc:
        errors.append(f"pypdf: {exc}")

    try:
        completed = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                if line.lower().startswith("pages:"):
                    return {
                        "page_count": int(line.split(":", 1)[1].strip()),
                        "page_count_status": "ok",
                        "page_count_error": None,
                    }
        stderr = completed.stderr.strip()
        if stderr:
            errors.append(f"pdfinfo: {stderr}")
    except Exception as exc:
        errors.append(f"pdfinfo: {exc}")

    return {
        "page_count": None,
        "page_count_status": "unavailable",
        "page_count_error": "; ".join(errors) or "no PDF page-count backend succeeded",
    }


def _pdf_page_count(pdf_path: Path | None) -> int | None:
    if pdf_path is None or not pdf_path.exists():
        return None
    from pypdf import PdfReader

    return len(PdfReader(str(pdf_path)).pages)
