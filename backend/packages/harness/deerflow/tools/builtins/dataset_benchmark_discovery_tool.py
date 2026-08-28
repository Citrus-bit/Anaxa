from __future__ import annotations

from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadState
from deerflow.benchmarks import DatasetBenchmarkDiscoveryService
from deerflow.config.paths import VIRTUAL_PATH_PREFIX


def _virtual_output_path(outputs_dir: Path, filepath: Path) -> str:
    relative = filepath.resolve().relative_to(outputs_dir.resolve())
    return f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}"


@tool("dataset_benchmark_discovery", parse_docstring=True)
async def dataset_benchmark_discovery_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    topic: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    scope: str | None = None,
    max_results: int = 30,
    sources: list[str] | None = None,
) -> Command:
    """Discover datasets and benchmarks for experiments without downloading restricted data.

    Use this before manuscript or experiment planning when the user asks for
    recent datasets, benchmarks, leaderboards, baselines, SOTA comparisons, or
    a dataset/benchmark map for a research topic.

    The tool searches dataset and benchmark indexes, normalizes candidates, and
    writes `dataset_benchmark_map.json` under `/mnt/user-data/outputs`. It marks
    access, license, version/date, metrics, baseline/SOTA hints, and risks. It
    does not bypass login, license, or data-use restrictions.

    Args:
        topic: Research or experiment topic to discover datasets and benchmarks for.
        scope: Optional domain, modality, task, date, or benchmark constraint.
        max_results: Maximum normalized candidate entries to retain.
        sources: Optional source names such as papers-with-code, huggingface,
            openml, kaggle, zenodo, github, uci, figshare, osf, or leaderboard.
    """
    thread_data = runtime.state.get("thread_data") if runtime.state else {}
    outputs_path = (thread_data or {}).get("outputs_path")
    if not outputs_path:
        return Command(update={"messages": [ToolMessage("Error: thread outputs path is not available.", tool_call_id=tool_call_id)]})

    outputs_dir = Path(outputs_path)
    try:
        service = DatasetBenchmarkDiscoveryService()
        benchmark_map, output_path = await service.discover(
            topic=topic,
            scope=scope,
            output_dir=outputs_dir,
            max_results=max_results,
            sources=sources,
        )
    except Exception as exc:
        return Command(update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]})

    source_summary = ", ".join(benchmark_map.sources_requested)
    message = f"Dataset/benchmark map ready with {len(benchmark_map.entries)} candidate(s) from {source_summary}. Restricted or uncertain sources were marked for manual verification."
    return Command(
        update={
            "artifacts": [_virtual_output_path(outputs_dir, output_path)],
            "messages": [ToolMessage(message, tool_call_id=tool_call_id)],
        }
    )
