"""Academic research pipeline for literature-heavy report generation."""

from .adapters import AcademicSourceAdapter, build_default_adapters
from .formatters import (
    DEFAULT_REFERENCE_STYLE,
    REFERENCE_STYLE_LABELS,
    format_apa7_reference,
    format_bibtex_entry,
    format_reference,
    normalize_reference_style,
    reference_style_label,
)
from .repository import AcademicRepository
from .service import AcademicResearchService
from .types import (
    AcademicGraph,
    EvidenceCardRecord,
    OutlineNodeRecord,
    PaperAuthor,
    PaperEdgeRecord,
    PaperRecord,
    ReferenceEntry,
    ReportExportRecord,
    ResearchProject,
    SearchQueryRecord,
)

__all__ = [
    "AcademicGraph",
    "AcademicRepository",
    "AcademicResearchService",
    "AcademicSourceAdapter",
    "DEFAULT_REFERENCE_STYLE",
    "EvidenceCardRecord",
    "OutlineNodeRecord",
    "PaperAuthor",
    "PaperEdgeRecord",
    "PaperRecord",
    "ReferenceEntry",
    "REFERENCE_STYLE_LABELS",
    "ReportExportRecord",
    "ResearchProject",
    "SearchQueryRecord",
    "build_default_adapters",
    "format_apa7_reference",
    "format_bibtex_entry",
    "format_reference",
    "normalize_reference_style",
    "reference_style_label",
]
