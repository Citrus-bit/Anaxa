"""Decision recording tool for auditable workflow visualization."""

from __future__ import annotations

import json
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command


def _clean_text(value: str | None, *, limit: int = 800) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _clean_list(values: list[str] | None, *, limit: int = 8) -> list[str]:
    return [_clean_text(item, limit=240) for item in (values or [])[:limit] if _clean_text(item, limit=240)]


@tool("record_decision", parse_docstring=True)
def record_decision_tool(
    title: str,
    decision_type: str,
    rationale: str,
    next_step: str,
    status: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    alternatives: list[str] | None = None,
    related_tool: str | None = None,
    outcome: str | None = None,
) -> Command:
    """Record a concise, user-visible decision summary for the Flow tab.

    Use this tool only for important route-changing decisions: planning choices,
    tool selection, fallback/retry decisions, failure recovery, and final quality
    checks. Record a short auditable summary, not hidden reasoning or chain of
    thought.

    Args:
        title: Short decision title.
        decision_type: Decision category, such as planning, tool_selection,
            retry, fallback, validation, or delivery_check.
        rationale: Brief visible reason for the decision. Do not include private
            chain-of-thought.
        next_step: The concrete action that follows from this decision.
        status: Current decision status, such as planned, running, success,
            error, interrupted, or blocked.
        alternatives: Optional alternatives that were considered at a high
            level.
        related_tool: Optional tool or subsystem this decision is about.
        outcome: Optional result observed after the decision.
    """
    payload = {
        "title": _clean_text(title, limit=160) or "Decision",
        "decision_type": _clean_text(decision_type, limit=80) or "decision",
        "rationale": _clean_text(rationale),
        "next_step": _clean_text(next_step),
        "status": _clean_text(status, limit=80) or "success",
        "alternatives": _clean_list(alternatives),
        "related_tool": _clean_text(related_tool, limit=120) or None,
        "outcome": _clean_text(outcome) or None,
    }
    return Command(
        update={
            "messages": [
                ToolMessage(
                    json.dumps(payload, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                    name="record_decision",
                )
            ]
        }
    )
