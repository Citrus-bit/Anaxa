from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from medrix_flow.utils.citations import audit_latex_citations, write_citation_audit

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"


def _thread_roots(runtime: ToolRuntime[ContextT, ThreadState]) -> list[Path]:
    thread_data = runtime.state.get("thread_data") or {}
    return [
        Path(path).resolve()
        for path in (
            thread_data.get("outputs_path"),
            thread_data.get("uploads_path"),
            thread_data.get("workspace_path"),
        )
        if path
    ]


def _resolve_thread_path(runtime: ToolRuntime[ContextT, ThreadState], path: str) -> Path:
    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        raise ValueError("thread_id is required")

    stripped = path.lstrip("/")
    virtual_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")
    if stripped == virtual_prefix or stripped.startswith(virtual_prefix + "/"):
        actual_path = get_paths().resolve_virtual_path(thread_id, path)
    else:
        actual_path = Path(path).expanduser().resolve()

    roots = _thread_roots(runtime)
    if not roots:
        raise ValueError("thread runtime paths are not available")

    for root in roots:
        try:
            actual_path.relative_to(root)
            return actual_path
        except ValueError:
            continue

    raise ValueError(f"Only files under {VIRTUAL_PATH_PREFIX} can be audited: {path}")


def _resolve_output_path(runtime: ToolRuntime[ContextT, ThreadState], path: str) -> tuple[Path, str]:
    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("thread outputs path is not available")

    outputs_dir = Path(outputs_path).resolve()
    actual_path = _resolve_thread_path(runtime, path)
    try:
        relative = actual_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Audit output must be under {OUTPUTS_VIRTUAL_PREFIX}: {path}") from exc
    return actual_path, f"{OUTPUTS_VIRTUAL_PREFIX}/{relative.as_posix()}"


@tool("citation_audit", parse_docstring=True)
def citation_audit_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    bibtex_path: str = "/mnt/user-data/outputs/references.bib",
    tex_path: str | None = None,
    claim_map_path: str | None = None,
    allow_nocite_all: bool = False,
    output_audit_path: str = "/mnt/user-data/outputs/citation_audit.json",
) -> Command:
    """Extract BibTeX keys and audit LaTeX citations before manuscript PDF export.

    Use this tool for paper, literature review, or experiment-report manuscripts
    that include a `references.bib` file. It deterministically reads BibTeX
    citation keys, checks LaTeX `\\cite{...}` commands against those keys, and
    writes `citation_audit.json` into the outputs directory.

    Args:
        bibtex_path: Path to the BibTeX file, usually `/mnt/user-data/outputs/references.bib`.
        tex_path: Optional path to the LaTeX manuscript to audit.
        claim_map_path: Optional JSON claim/evidence map to flag unsupported claims.
        allow_nocite_all: Set true only when the user explicitly asked to include every reference via `\\nocite{*}`.
        output_audit_path: Output JSON path for the audit. Must be under `/mnt/user-data/outputs`.
    """
    try:
        resolved_bibtex = _resolve_thread_path(runtime, bibtex_path)
        resolved_tex = _resolve_thread_path(runtime, tex_path) if tex_path else None
        resolved_claim_map = _resolve_thread_path(runtime, claim_map_path) if claim_map_path else None
        resolved_output, virtual_output = _resolve_output_path(runtime, output_audit_path)

        result = audit_latex_citations(
            bibtex_path=resolved_bibtex,
            tex_path=resolved_tex,
            claim_map_path=resolved_claim_map,
            allow_nocite_all=allow_nocite_all,
        )
        write_citation_audit(result, resolved_output)
    except Exception as exc:
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})

    summary = f"{result.status.upper()}: citation audit wrote `{virtual_output}`. BibTeX keys: {len(result.citation_keys)}; cited keys: {len(result.cited_keys)}; missing keys: {len(result.missing_keys)}; nocite_all: {result.nocite_all}."
    if result.violations:
        summary += " Violations: " + " ".join(result.violations)

    return Command(
        update={
            "artifacts": [virtual_output],
            "messages": [ToolMessage(summary, tool_call_id=tool_call_id)],
        }
    )
