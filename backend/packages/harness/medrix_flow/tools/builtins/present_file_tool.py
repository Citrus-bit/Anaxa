import logging
import shutil
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from medrix_flow.utils.citations import audit_latex_citations, write_citation_audit
from medrix_flow.utils.latex import compile_latex_to_pdf, prepare_latex_preview

OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"
logger = logging.getLogger(__name__)


def _normalize_presented_filepath(
    runtime: ToolRuntime[ContextT, ThreadState],
    filepath: str,
) -> str:
    """Normalize a presented file path to the `/mnt/user-data/outputs/*` contract.

    Accepts either:
    - A virtual sandbox path such as `/mnt/user-data/outputs/report.md`
    - A host-side thread outputs path such as
      `/app/backend/.medrix-flow/threads/<thread>/user-data/outputs/report.md`

    Returns:
        The normalized virtual path.

    Raises:
        ValueError: If runtime metadata is missing or the path is outside the
            current thread's outputs directory.
    """
    if runtime.state is None:
        raise ValueError("Thread runtime state is not available")

    thread_id = runtime.context.get("thread_id")
    if not thread_id:
        raise ValueError("Thread ID is not available in runtime context")

    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("Thread outputs path is not available in runtime state")

    outputs_dir = Path(outputs_path).resolve()
    stripped = filepath.lstrip("/")
    virtual_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")

    if stripped == virtual_prefix or stripped.startswith(virtual_prefix + "/"):
        actual_path = get_paths().resolve_virtual_path(thread_id, filepath)
    else:
        actual_path = Path(filepath).expanduser().resolve()

    try:
        relative_path = actual_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Only files in {OUTPUTS_VIRTUAL_PREFIX} can be presented: {filepath}") from exc

    return f"{OUTPUTS_VIRTUAL_PREFIX}/{relative_path.as_posix()}"


@tool("present_files", parse_docstring=True)
def present_file_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    filepaths: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Make files visible to the user for viewing and rendering in the client interface.

    When to use the present_files tool:

    - Making any file available for the user to view, download, or interact with
    - Presenting multiple related files at once
    - After creating files that should be presented to the user

    When NOT to use the present_files tool:
    - When you only need to read file contents for your own processing
    - For temporary or intermediate files not meant for user viewing

    Notes:
    - You should call this tool after creating files and moving them to the `/mnt/user-data/outputs` directory.
    - This tool can be safely called in parallel with other tools. State updates are handled by a reducer to prevent conflicts.

    Args:
        filepaths: List of absolute file paths to present to the user. **Only** files in `/mnt/user-data/outputs` can be presented.
    """
    try:
        normalized_paths: list[str] = []
        diagnostics: list[str] = []
        thread_id = runtime.context.get("thread_id")
        outputs_dir = Path(runtime.state.get("thread_data", {}).get("outputs_path", "")).resolve()
        for filepath in filepaths:
            normalized_path = _normalize_presented_filepath(runtime, filepath)

            if normalized_path.lower().endswith(".tex") and thread_id:
                try:
                    tex_relative = Path(normalized_path.removeprefix(OUTPUTS_VIRTUAL_PREFIX).lstrip("/"))
                    tex_path = outputs_dir / tex_relative
                    bibtex_path = tex_path.with_name("references.bib")
                    if bibtex_path.exists():
                        audit_result = audit_latex_citations(bibtex_path=bibtex_path, tex_path=tex_path)
                        audit_path = write_citation_audit(audit_result, tex_path.parent / "citation_audit.json")
                        normalized_audit_path = f"{OUTPUTS_VIRTUAL_PREFIX}/{audit_path.relative_to(outputs_dir).as_posix()}"
                        normalized_paths.append(normalized_audit_path)
                        if not audit_result.passed:
                            normalized_paths.append(normalized_path)
                            diagnostics.append(f"Citation audit failed for {normalized_path}; PDF was not generated. " + " ".join(audit_result.violations))
                            logger.warning("LaTeX citation audit failed for %s: %s", normalized_path, audit_result.violations)
                            continue

                    prepared_path = prepare_latex_preview(tex_path)
                    preview_pdf = compile_latex_to_pdf(prepared_path, prepared_path.parent)
                    final_pdf = tex_path.with_suffix(".pdf")
                    shutil.copy2(preview_pdf, final_pdf)
                    normalized_paths.append(f"{OUTPUTS_VIRTUAL_PREFIX}/{final_pdf.relative_to(outputs_dir).as_posix()}")
                    normalized_paths.append(normalized_path)
                    logger.info("Compiled LaTeX preview: %s -> %s", tex_path, final_pdf)
                    continue
                except Exception as exc:
                    diagnostics.append(f"LaTeX preview compilation failed for {normalized_path}: {exc}")
                    logger.warning("LaTeX preview compilation failed for %s: %s", normalized_path, exc)

            normalized_paths.append(normalized_path)
    except ValueError as exc:
        return Command(
            update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]},
        )

    # The merge_artifacts reducer will handle merging and deduplication
    message = "Successfully presented files"
    if diagnostics:
        message = "Presented files with issues:\n" + "\n".join(f"- {item}" for item in diagnostics)

    return Command(
        update={
            "artifacts": normalized_paths,
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        },
    )
