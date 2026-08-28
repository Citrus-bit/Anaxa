"""VisualQualityMiddleware - enforce quality check before presenting visual output."""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)

# File extensions that indicate visual output
_VISUAL_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",  # images
        ".pptx",
        ".ppt",  # presentations
        ".pdf",  # documents/reports
        ".html",  # charts (rendered)
    }
)


def _has_visual_files(filepaths: list[str]) -> bool:
    """Check if any of the filepaths are visual output files."""
    return any(fp.lower().endswith(tuple(_VISUAL_EXTENSIONS)) for fp in filepaths)


class VisualQualityMiddleware(AgentMiddleware[ThreadState]):
    """Enforce visual_quality_check before present_files for visual output.

    When the agent calls ``present_files`` with visual file types (images,
    PPT, PDF, HTML charts) and has NOT called ``visual_quality_check`` in
    the current agent turn, this middleware injects a reminder into the
    tool result asking the agent to run the quality check first.

    This is a soft gate: it does not block delivery, but adds a visible
    nudge that the model will see and (in most cases) act on.
    """

    state_schema = ThreadState

    def __init__(self) -> None:
        super().__init__()
        self._quality_checked = False

    def _handle(
        self,
        request: ToolCallRequest,
        result: ToolMessage | Command,
    ) -> ToolMessage | Command:
        """Append quality reminder if presenting visual files without prior check."""
        if not isinstance(result, Command):
            return result

        # If quality was already checked this turn, pass through
        if self._quality_checked:
            return result

        # Extract filepaths from the present_files args
        args = request.tool_call.get("args", {})
        filepaths = args.get("filepaths", [])

        if not _has_visual_files(filepaths):
            return result

        # Visual files being presented without quality check — inject reminder
        logger.info("[VisualQuality] present_files called with visual output but no prior quality check")

        reminder = (
            "\n\n📋 **Quality Reminder**: You are presenting visual output without running "
            "`visual_quality_check` first. For professional-grade deliverables, call "
            "`visual_quality_check` to verify design standards before presenting to the user."
        )
        if any(fp.lower().endswith((".pdf", ".tex")) for fp in filepaths):
            reminder += " For LaTeX/PDF reports, also verify equations render correctly, symbols are legible, and every figure is backed by validated local data or assets."

        # Append reminder to the messages in the Command update
        update = dict(result.update) if result.update else {}
        messages = list(update.get("messages", []))
        if messages and isinstance(messages[-1], ToolMessage):
            original = messages[-1]
            content = str(original.content) + reminder
            messages[-1] = ToolMessage(
                content=content,
                tool_call_id=original.tool_call_id,
                name=original.name,
                status=original.status,
            )
            update["messages"] = messages
            return Command(update=update, goto=result.goto if hasattr(result, "goto") else None)

        return result

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "")

        # Track quality check calls
        if tool_name == "visual_quality_check":
            self._quality_checked = True
            return handler(request)

        # Reset tracking on non-present_files, non-quality-check calls
        if tool_name != "present_files":
            return handler(request)

        result = handler(request)
        return self._handle(request, result)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        tool_name = request.tool_call.get("name", "")

        if tool_name == "visual_quality_check":
            self._quality_checked = True
            return await handler(request)

        if tool_name != "present_files":
            return await handler(request)

        result = await handler(request)
        return self._handle(request, result)
