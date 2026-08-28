"""Task tool for delegating work to subagents."""

import logging
import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Annotated, Literal

from langchain.tools import InjectedToolCallId, ToolRuntime, tool
from langgraph.config import get_stream_writer
from langgraph.typing import ContextT

from medrix_flow.agents.lead_agent.prompt import get_skills_prompt_section
from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.subagents import MIN_SUBAGENT_MAX_TURNS, SubagentExecutor, get_subagent_config
from medrix_flow.subagents.executor import SubagentStatus, cleanup_background_task, get_background_task_result, mark_background_task_timed_out

logger = logging.getLogger(__name__)
TASK_HEARTBEAT_INTERVAL_SECONDS = 15
TASK_POLL_INTERVAL_SECONDS = 5


@tool("task", parse_docstring=True)
def task_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: Literal["general-purpose", "bash", "academic-researcher", "cs-ai-experimenter", "bioinformatics-analyst"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    Subagents help you:
    - Preserve context by keeping exploration and implementation separate
    - Handle complex multi-step tasks autonomously
    - Execute commands or operations in isolated contexts

    Available subagent types:
    - **general-purpose**: A capable agent for complex, multi-step tasks that require
      both exploration and action. Use when the task requires complex reasoning,
      multiple dependent steps, or would benefit from isolated context.
    - **bash**: Command execution specialist for running bash commands. Use for
      git operations, build processes, or when command output would be verbose.
    - **academic-researcher**: Specialized for literature-heavy academic work that
      benefits from structured report generation, reference normalization, and
      evidence mapping.
    - **cs-ai-experimenter**: Structured CS/AI experiment specialist for model
      evaluation, diagnostics, and reproducible experiment bundles.
    - **bioinformatics-analyst**: Bioinformatics specialist for bulk expression,
      differential analysis, QC, and single-cell starter workflows.

    When to use this tool:
    - Complex tasks requiring multiple steps or tools
    - Tasks that produce verbose output
    - When you want to isolate context from the main conversation
    - Parallel research or exploration tasks

    When NOT to use this tool:
    - Simple, single-step operations (use tools directly)
    - Tasks requiring user interaction or clarification

    Args:
        description: A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD.
        max_turns: Optional maximum number of agent turns. Values below the
            safe minimum are raised to avoid premature recursion-limit failures.
    """
    # Get subagent configuration
    config = get_subagent_config(subagent_type)
    if config is None:
        return f"Error: Unknown subagent type '{subagent_type}'. Available: academic-researcher, bash, bioinformatics-analyst, cs-ai-experimenter, general-purpose"

    # Build config overrides
    overrides: dict = {}

    skills_section = get_skills_prompt_section()
    if skills_section:
        overrides["system_prompt"] = config.system_prompt + "\n\n" + skills_section

    if max_turns is not None:
        overrides["max_turns"] = max(MIN_SUBAGENT_MAX_TURNS, max_turns)

    if overrides:
        config = replace(config, **overrides)

    # Extract parent context from runtime
    sandbox_state = None
    thread_data = None
    thread_id = None
    parent_model = None
    trace_id = None

    if runtime is not None:
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")
        thread_id = runtime.context.get("thread_id")

        # Prefer runtime context because it reflects the current run. Metadata is
        # retained as a fallback for older call paths.
        parent_model = runtime.context.get("model_name") or runtime.context.get("model")
        metadata = runtime.config.get("metadata", {})
        parent_model = parent_model or metadata.get("model_name")

        # Get or generate trace_id for distributed tracing
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    # Get available tools (excluding task tool to prevent nesting)
    # Lazy import to avoid circular dependency
    from medrix_flow.tools import get_available_tools

    # Subagents should not have subagent tools enabled (prevent recursive nesting)
    tools = get_available_tools(model_name=parent_model, subagent_enabled=False)

    # Create executor
    executor = SubagentExecutor(
        config=config,
        tools=tools,
        parent_model=parent_model,
        sandbox_state=sandbox_state,
        thread_data=thread_data,
        thread_id=thread_id,
        trace_id=trace_id,
    )

    # Start background execution (always async to prevent blocking)
    # Use tool_call_id as task_id for better traceability
    task_id = executor.execute_async(prompt, task_id=tool_call_id)

    # Poll for task completion in backend (removes need for LLM to poll)
    poll_count = 0
    last_status = None
    last_message_count = 0  # Track how many AI messages we've already sent
    last_heartbeat_poll = -TASK_HEARTBEAT_INTERVAL_SECONDS // TASK_POLL_INTERVAL_SECONDS
    # Safety timeout: execution timeout + 60s buffer, counted from execution start.
    max_poll_count = (config.timeout_seconds + 60) // TASK_POLL_INTERVAL_SECONDS

    logger.info(f"[trace={trace_id}] Started background task {task_id} (subagent={subagent_type}, timeout={config.timeout_seconds}s, polling_limit={max_poll_count} polls)")

    writer = get_stream_writer()
    # Send Task Started message'
    writer({"type": "task_started", "task_id": task_id, "description": description})

    while True:
        result = get_background_task_result(task_id)

        if result is None:
            logger.error(f"[trace={trace_id}] Task {task_id} not found in background tasks")
            writer({"type": "task_failed", "task_id": task_id, "error": "Task disappeared from background tasks"})
            cleanup_background_task(task_id)
            return f"Error: Task {task_id} disappeared from background tasks"

        # Log status changes for debugging
        if result.status != last_status:
            logger.info(f"[trace={trace_id}] Task {task_id} status: {result.status.value}")
            last_status = result.status

        # Check for new AI messages and send task_running events
        current_message_count = len(result.ai_messages)
        if current_message_count > last_message_count:
            # Send task_running event for each new message
            for i in range(last_message_count, current_message_count):
                message = result.ai_messages[i]
                writer(
                    {
                        "type": "task_running",
                        "task_id": task_id,
                        "message": message,
                        "message_index": i + 1,  # 1-based index for display
                        "total_messages": current_message_count,
                    }
                )
                logger.info(f"[trace={trace_id}] Task {task_id} sent message #{i + 1}/{current_message_count}")
            last_message_count = current_message_count
        elif result.status == SubagentStatus.RUNNING and poll_count - last_heartbeat_poll >= TASK_HEARTBEAT_INTERVAL_SECONDS // TASK_POLL_INTERVAL_SECONDS:
            # Emit a lightweight heartbeat so the UI can keep showing progress
            # even when the subagent has not produced a new AI message yet.
            writer(
                {
                    "type": "task_running",
                    "task_id": task_id,
                    "message_index": last_message_count,
                    "total_messages": current_message_count,
                    "heartbeat": True,
                }
            )
            last_heartbeat_poll = poll_count

        # Check if task completed, failed, or timed out
        if result.status == SubagentStatus.COMPLETED:
            writer({"type": "task_completed", "task_id": task_id, "result": result.result})
            logger.info(f"[trace={trace_id}] Task {task_id} completed after {poll_count} polls")
            cleanup_background_task(task_id)
            return f"Task Succeeded. Result: {result.result}"
        elif result.status == SubagentStatus.FAILED:
            writer({"type": "task_failed", "task_id": task_id, "error": result.error})
            logger.error(f"[trace={trace_id}] Task {task_id} failed: {result.error}")
            cleanup_background_task(task_id)
            return f"Task failed. Error: {result.error}"
        elif result.status == SubagentStatus.TIMED_OUT:
            writer({"type": "task_timed_out", "task_id": task_id, "error": result.error})
            logger.warning(f"[trace={trace_id}] Task {task_id} timed out: {result.error}")
            cleanup_background_task(task_id)
            return f"Task timed out. Error: {result.error}"

        # Still running, wait before next poll
        time.sleep(TASK_POLL_INTERVAL_SECONDS)
        poll_count += 1

        # Polling safety timeout as a backstop in case the scheduler timeout
        # does not fire. Queue time is excluded because started_at is assigned
        # by the execution worker.
        if result.started_at is not None:
            elapsed_seconds = (datetime.now() - result.started_at).total_seconds()
        else:
            elapsed_seconds = 0

        if result.started_at is not None and elapsed_seconds > config.timeout_seconds + 60:
            timeout_minutes = config.timeout_seconds // 60
            error = f"Polling safety timeout after {timeout_minutes} minutes. This may indicate the background task is stuck."
            logger.error(f"[trace={trace_id}] Task {task_id} polling timed out after {elapsed_seconds:.1f}s of execution (should have been caught by thread pool timeout)")
            mark_background_task_timed_out(task_id, error)
            writer({"type": "task_timed_out", "task_id": task_id, "error": error})
            cleanup_background_task(task_id)
            return f"Task polling timed out after {timeout_minutes} minutes. This may indicate the background task is stuck. Status: {result.status.value}"
