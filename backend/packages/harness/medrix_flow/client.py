"""MedrixFlowClient — Embedded Python client for MedrixFlow agent system.

Provides direct programmatic access to MedrixFlow's agent capabilities
without requiring LangGraph Server or Gateway API processes.

Usage:
    from medrix_flow.client import MedrixFlowClient

    client = MedrixFlowClient()
    response = client.chat("Analyze this paper for me", thread_id="my-thread")
    print(response)

    # Streaming
    for event in client.stream("hello"):
        print(event)
"""

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from medrix_flow.agents.lead_agent.agent import _build_middlewares
from medrix_flow.agents.lead_agent.prompt import apply_prompt_template
from medrix_flow.agents.thread_state import ThreadState
from medrix_flow.config.agents_config import AGENT_NAME_PATTERN
from medrix_flow.config.app_config import get_app_config, reload_app_config
from medrix_flow.config.extensions_config import ExtensionsConfig, get_extensions_config, reload_extensions_config
from medrix_flow.config.paths import get_paths
from medrix_flow.config.subagents_config import get_subagents_app_config
from medrix_flow.models import create_chat_model
from medrix_flow.skills.service import SkillService

logger = logging.getLogger(__name__)


def _thread_memory_mtime(thread_id: str | None) -> float | None:
    if not thread_id:
        return None
    try:
        path = get_paths().thread_memory_file(thread_id)
        return path.stat().st_mtime if path.exists() else None
    except (OSError, ValueError):
        return None


@dataclass
class StreamEvent:
    """A single event from the streaming agent response.

    Event types align with the LangGraph SSE protocol:
        - ``"values"``: Full state snapshot (title, messages, artifacts).
        - ``"messages-tuple"``: Per-message update (AI text, tool calls, tool results).
        - ``"end"``: Stream finished.

    Attributes:
        type: Event type.
        data: Event payload. Contents vary by type.
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)


class MedrixFlowClient:
    """Embedded Python client for MedrixFlow agent system.

    Provides direct programmatic access to MedrixFlow's agent capabilities
    without requiring LangGraph Server or Gateway API processes.

    Note:
        Multi-turn conversations require a ``checkpointer``. Without one,
        each ``stream()`` / ``chat()`` call is stateless — ``thread_id``
        is only used for file isolation (uploads / artifacts).

        The system prompt (including date, memory, and skills context) is
        generated when the internal agent is first created and cached until
        the configuration key changes. Call :meth:`reset_agent` to force
        a refresh in long-running processes.

    Example::

        from medrix_flow.client import MedrixFlowClient

        client = MedrixFlowClient()

        # Simple one-shot
        print(client.chat("hello"))

        # Streaming
        for event in client.stream("hello"):
            print(event.type, event.data)

        # Configuration queries
        print(client.list_models())
        print(client.list_skills())
    """

    def __init__(
        self,
        config_path: str | None = None,
        checkpointer=None,
        *,
        model_name: str | None = None,
        thinking_enabled: bool = True,
        reasoning_effort: str | None = None,
        subagent_enabled: bool = False,
        plan_mode: bool = False,
        agent_name: str | None = None,
    ):
        """Initialize the client.

        Loads configuration but defers agent creation to first use.

        Args:
            config_path: Path to config.yaml. Uses default resolution if None.
            checkpointer: LangGraph checkpointer instance for state persistence.
                Required for multi-turn conversations on the same thread_id.
                Without a checkpointer, each call is stateless.
            model_name: Override the default model name from config.
            thinking_enabled: Enable model's extended thinking.
            reasoning_effort: Optional reasoning depth override for capable models.
            subagent_enabled: Enable subagent delegation.
            plan_mode: Enable TodoList middleware for plan mode.
            agent_name: Name of the agent to use.
        """
        if config_path is not None:
            reload_app_config(config_path)
        self._app_config = get_app_config()

        if agent_name is not None and not AGENT_NAME_PATTERN.match(agent_name):
            raise ValueError(f"Invalid agent name '{agent_name}'. Must match pattern: {AGENT_NAME_PATTERN.pattern}")

        self._checkpointer = checkpointer
        self._model_name = model_name
        self._thinking_enabled = thinking_enabled
        self._reasoning_effort = reasoning_effort
        self._subagent_enabled = subagent_enabled
        self._plan_mode = plan_mode
        self._agent_name = agent_name

        # Lazy agent — created on first call, recreated when config changes.
        self._agent = None
        self._agent_config_key: tuple | None = None

    def reset_agent(self) -> None:
        """Force the internal agent to be recreated on the next call.

        Use this after external changes (e.g. memory updates, skill
        installations) that should be reflected in the system prompt
        or tool set.
        """
        self._agent = None
        self._agent_config_key = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON to *path* atomically (temp file + replace)."""
        fd = tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        )
        try:
            json.dump(data, fd, indent=2)
            fd.close()
            Path(fd.name).replace(path)
        except BaseException:
            fd.close()
            Path(fd.name).unlink(missing_ok=True)
            raise

    def _get_runnable_config(self, thread_id: str, **overrides) -> RunnableConfig:
        """Build a RunnableConfig for agent invocation."""
        configurable = {
            "thread_id": thread_id,
            "model_name": overrides.get("model_name", self._model_name),
            "thinking_enabled": overrides.get("thinking_enabled", self._thinking_enabled),
            "reasoning_effort": overrides.get("reasoning_effort", self._reasoning_effort),
            "is_plan_mode": overrides.get("plan_mode", self._plan_mode),
            "subagent_enabled": overrides.get("subagent_enabled", self._subagent_enabled),
            "visual_output_intent": overrides.get("visual_output_intent", False),
            "synthetic_data_mode": overrides.get("synthetic_data_mode", False),
        }
        return RunnableConfig(
            configurable=configurable,
            recursion_limit=overrides.get("recursion_limit", 100),
        )

    def _ensure_agent(self, config: RunnableConfig):
        """Create (or recreate) the agent when config-dependent params change."""
        cfg = config.get("configurable", {})
        key = (
            cfg.get("model_name"),
            cfg.get("thinking_enabled"),
            cfg.get("reasoning_effort"),
            cfg.get("is_plan_mode"),
            cfg.get("subagent_enabled"),
            cfg.get("visual_output_intent"),
            cfg.get("synthetic_data_mode"),
            cfg.get("thread_id"),
            _thread_memory_mtime(cfg.get("thread_id")),
        )

        if self._agent is not None and self._agent_config_key == key:
            return

        thinking_enabled = cfg.get("thinking_enabled", True)
        reasoning_effort = cfg.get("reasoning_effort")
        model_name = cfg.get("model_name")
        subagent_enabled = cfg.get("subagent_enabled", False)
        visual_output_intent = bool(cfg.get("visual_output_intent", False))
        synthetic_data_mode = bool(cfg.get("synthetic_data_mode", False))
        max_concurrent_subagents = cfg.get("max_concurrent_subagents", get_subagents_app_config().pool_size)
        thread_id = cfg.get("thread_id")

        kwargs: dict[str, Any] = {
            "model": create_chat_model(
                name=model_name,
                thinking_enabled=thinking_enabled,
                reasoning_effort=reasoning_effort,
            ),
            "tools": self._get_tools(
                model_name=model_name,
                subagent_enabled=subagent_enabled,
                plan_mode=bool(cfg.get("is_plan_mode", False)),
                visual_output_intent=visual_output_intent,
            ),
            "middleware": _build_middlewares(config, model_name=model_name, agent_name=self._agent_name),
            "system_prompt": apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                agent_name=self._agent_name,
                thread_id=thread_id,
                plan_mode=bool(cfg.get("is_plan_mode", False)),
                visual_output_intent=visual_output_intent,
                synthetic_data_mode=synthetic_data_mode,
            ),
            "state_schema": ThreadState,
        }
        checkpointer = self._checkpointer
        if checkpointer is None:
            from medrix_flow.agents.checkpointer import get_checkpointer

            try:
                checkpointer = get_checkpointer()
            except Exception:
                logger.warning("Failed to initialize default checkpointer; continuing without persistence.", exc_info=True)
                checkpointer = None
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer

        self._agent = create_agent(**kwargs)
        self._agent_config_key = key
        logger.info(
            "Agent created: agent_name=%s, model=%s, thinking=%s, reasoning_effort=%s",
            self._agent_name,
            model_name,
            thinking_enabled,
            reasoning_effort,
        )

    @staticmethod
    def _get_tools(
        *,
        model_name: str | None,
        subagent_enabled: bool,
        plan_mode: bool = False,
        visual_output_intent: bool = False,
    ):
        """Lazy import to avoid circular dependency at module level."""
        from medrix_flow.tools import get_available_tools

        return get_available_tools(
            model_name=model_name,
            subagent_enabled=subagent_enabled,
            plan_mode=plan_mode,
            visual_output_intent=visual_output_intent,
        )

    @staticmethod
    def _serialize_message(msg) -> dict:
        """Serialize a LangChain message to a plain dict for values events."""
        if isinstance(msg, AIMessage):
            d: dict[str, Any] = {"type": "ai", "content": msg.content, "id": getattr(msg, "id", None)}
            if msg.tool_calls:
                d["tool_calls"] = [{"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in msg.tool_calls]
            if getattr(msg, "usage_metadata", None):
                d["usage_metadata"] = msg.usage_metadata
            return d
        if isinstance(msg, ToolMessage):
            payload = {
                "type": "tool",
                "content": MedrixFlowClient._extract_text(msg.content),
                "name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "id": getattr(msg, "id", None),
            }
            if getattr(msg, "additional_kwargs", None):
                payload["additional_kwargs"] = msg.additional_kwargs
            return payload
        if isinstance(msg, HumanMessage):
            payload = {"type": "human", "content": msg.content, "id": getattr(msg, "id", None)}
            if getattr(msg, "name", None):
                payload["name"] = getattr(msg, "name")
            return payload
        if isinstance(msg, SystemMessage):
            return {"type": "system", "content": msg.content, "id": getattr(msg, "id", None)}
        return {"type": "unknown", "content": str(msg), "id": getattr(msg, "id", None)}

    @staticmethod
    def _extract_text(content) -> str:
        """Extract plain text from AIMessage content (str or list of blocks).

        String chunks are concatenated without separators to avoid corrupting
        token/character deltas or chunked JSON payloads. Dict-based text blocks
        are treated as full text blocks and joined with newlines to preserve
        readability.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            if content and all(isinstance(block, str) for block in content):
                chunk_like = len(content) > 1 and all(isinstance(block, str) and len(block) <= 20 and any(ch in block for ch in '{}[]":,') for block in content)
                return "".join(content) if chunk_like else "\n".join(content)

            pieces: list[str] = []
            pending_str_parts: list[str] = []

            def flush_pending_str_parts() -> None:
                if pending_str_parts:
                    pieces.append("".join(pending_str_parts))
                    pending_str_parts.clear()

            for block in content:
                if isinstance(block, str):
                    pending_str_parts.append(block)
                elif isinstance(block, dict):
                    flush_pending_str_parts()
                    text_val = block.get("text")
                    if isinstance(text_val, str):
                        pieces.append(text_val)

            flush_pending_str_parts()
            return "\n".join(pieces) if pieces else ""
        return str(content)

    # ------------------------------------------------------------------
    # Public API — conversation
    # ------------------------------------------------------------------

    def stream(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        **kwargs,
    ) -> Generator[StreamEvent, None, None]:
        """Stream a conversation turn, yielding events incrementally.

        Each call sends one user message and yields events until the agent
        finishes its turn. A ``checkpointer`` must be provided at init time
        for multi-turn context to be preserved across calls.

        Event types align with the LangGraph SSE protocol so that
        consumers can switch between HTTP streaming and embedded mode
        without changing their event-handling logic.

        Args:
            message: User message text.
            thread_id: Thread ID for conversation context. Auto-generated if None.
            **kwargs: Override client defaults (model_name, thinking_enabled,
                plan_mode, subagent_enabled, recursion_limit).

        Yields:
            StreamEvent with one of:
            - type="values"          data={"title": str|None, "messages": [...], "artifacts": [...]}
            - type="messages-tuple"  data={"type": "ai", "content": str, "id": str}
            - type="messages-tuple"  data={"type": "ai", "content": str, "id": str, "usage_metadata": {...}}
            - type="messages-tuple"  data={"type": "ai", "content": "", "id": str, "tool_calls": [...]}
            - type="messages-tuple"  data={"type": "tool", "content": str, "name": str, "tool_call_id": str, "id": str}
            - type="end"             data={"usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}}
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        config = self._get_runnable_config(thread_id, **kwargs)
        self._ensure_agent(config)

        state: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
        context = dict(config.get("configurable", {}))
        context["thread_id"] = thread_id
        if self._agent_name:
            context["agent_name"] = self._agent_name

        seen_ids: set[str] = set()
        cumulative_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for chunk in self._agent.stream(state, config=config, context=context, stream_mode="values"):
            messages = chunk.get("messages", [])

            for msg in messages:
                msg_id = getattr(msg, "id", None)
                if msg_id and msg_id in seen_ids:
                    continue
                if msg_id:
                    seen_ids.add(msg_id)

                if isinstance(msg, AIMessage):
                    # Track token usage from AI messages
                    usage = getattr(msg, "usage_metadata", None)
                    if usage:
                        cumulative_usage["input_tokens"] += usage.get("input_tokens", 0) or 0
                        cumulative_usage["output_tokens"] += usage.get("output_tokens", 0) or 0
                        cumulative_usage["total_tokens"] += usage.get("total_tokens", 0) or 0

                    if msg.tool_calls:
                        yield StreamEvent(
                            type="messages-tuple",
                            data={
                                "type": "ai",
                                "content": "",
                                "id": msg_id,
                                "tool_calls": [{"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in msg.tool_calls],
                            },
                        )

                    text = self._extract_text(msg.content)
                    if text:
                        event_data: dict[str, Any] = {"type": "ai", "content": text, "id": msg_id}
                        if usage:
                            event_data["usage_metadata"] = {
                                "input_tokens": usage.get("input_tokens", 0) or 0,
                                "output_tokens": usage.get("output_tokens", 0) or 0,
                                "total_tokens": usage.get("total_tokens", 0) or 0,
                            }
                        yield StreamEvent(type="messages-tuple", data=event_data)

                elif isinstance(msg, ToolMessage):
                    yield StreamEvent(
                        type="messages-tuple",
                        data={
                            "type": "tool",
                            "content": self._extract_text(msg.content),
                            "name": getattr(msg, "name", None),
                            "tool_call_id": getattr(msg, "tool_call_id", None),
                            "id": msg_id,
                        },
                    )

            # Emit a values event for each state snapshot
            yield StreamEvent(
                type="values",
                data={
                    "title": chunk.get("title"),
                    "messages": [self._serialize_message(m) for m in messages],
                    "plan": chunk.get("plan"),
                    "artifacts": chunk.get("artifacts", []),
                },
            )

        yield StreamEvent(type="end", data={"usage": cumulative_usage})

    def chat(self, message: str, *, thread_id: str | None = None, **kwargs) -> str:
        """Send a message and return the final text response.

        Convenience wrapper around :meth:`stream` that returns only the
        **last** AI text from ``messages-tuple`` events. If the agent emits
        multiple text segments in one turn, intermediate segments are
        discarded. Use :meth:`stream` directly to capture all events.

        Args:
            message: User message text.
            thread_id: Thread ID for conversation context. Auto-generated if None.
            **kwargs: Override client defaults (same as stream()).

        Returns:
            The last AI message text, or empty string if no response.
        """
        last_text = ""
        for event in self.stream(message, thread_id=thread_id, **kwargs):
            if event.type == "messages-tuple" and event.data.get("type") == "ai":
                content = event.data.get("content", "")
                if content:
                    last_text = content
        return last_text

    # ------------------------------------------------------------------
    # Public API — configuration queries
    # ------------------------------------------------------------------

    def list_models(self) -> dict:
        """List available models from configuration.

        Returns:
            Dict with "models" key containing list of model info dicts,
            matching the Gateway API ``ModelsListResponse`` schema.
        """
        return {
            "models": [
                {
                    "name": model.name,
                    "model": getattr(model, "model", None),
                    "display_name": getattr(model, "display_name", None),
                    "description": getattr(model, "description", None),
                    "supports_thinking": getattr(model, "supports_thinking", False),
                    "supports_reasoning_effort": getattr(model, "supports_reasoning_effort", False),
                }
                for model in self._app_config.models
            ]
        }

    def list_skills(self, enabled_only: bool = False) -> dict:
        """List available skills.

        Args:
            enabled_only: If True, only return enabled skills.

        Returns:
            Dict with "skills" key containing list of skill info dicts,
            matching the Gateway API ``SkillsListResponse`` schema.
        """
        from medrix_flow.skills.loader import load_skills

        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "license": s.license,
                    "category": s.category,
                    "enabled": s.enabled,
                }
                for s in load_skills(enabled_only=enabled_only)
            ]
        }

    def get_memory(self) -> dict:
        """Get current memory data.

        Returns:
            Memory data dict (see src/agents/memory/updater.py for structure).
        """
        from medrix_flow.agents.memory.updater import get_memory_data

        return get_memory_data()

    def get_model(self, name: str) -> dict | None:
        """Get a specific model's configuration by name.

        Args:
            name: Model name.

        Returns:
            Model info dict matching the Gateway API ``ModelResponse``
            schema, or None if not found.
        """
        model = self._app_config.get_model_config(name)
        if model is None:
            return None
        return {
            "name": model.name,
            "model": getattr(model, "model", None),
            "display_name": getattr(model, "display_name", None),
            "description": getattr(model, "description", None),
            "supports_thinking": getattr(model, "supports_thinking", False),
            "supports_reasoning_effort": getattr(model, "supports_reasoning_effort", False),
        }

    # ------------------------------------------------------------------
    # Public API — MCP configuration
    # ------------------------------------------------------------------

    def get_mcp_config(self) -> dict:
        """Get MCP server configurations.

        Returns:
            Dict with "mcp_servers" key mapping server name to config,
            matching the Gateway API ``McpConfigResponse`` schema.
        """
        config = get_extensions_config()
        return {"mcp_servers": {name: server.model_dump() for name, server in config.mcp_servers.items()}}

    def update_mcp_config(self, mcp_servers: dict[str, dict]) -> dict:
        """Update MCP server configurations.

        Writes to extensions_config.json and reloads the cache.

        Args:
            mcp_servers: Dict mapping server name to config dict.
                Each value should contain keys like enabled, type, command, args, env, url, etc.

        Returns:
            Dict with "mcp_servers" key, matching the Gateway API
            ``McpConfigResponse`` schema.

        Raises:
            OSError: If the config file cannot be written.
        """
        config_path = ExtensionsConfig.resolve_config_path()
        if config_path is None:
            raise FileNotFoundError("Cannot locate extensions_config.json. Set MEDRIX_FLOW_EXTENSIONS_CONFIG_PATH or ensure it exists in the project root.")

        current_config = get_extensions_config()

        config_data = {
            "mcpServers": mcp_servers,
            "skills": {name: {"enabled": skill.enabled} for name, skill in current_config.skills.items()},
        }

        self._atomic_write_json(config_path, config_data)

        self._agent = None
        reloaded = reload_extensions_config()
        return {"mcp_servers": {name: server.model_dump() for name, server in reloaded.mcp_servers.items()}}

    # ------------------------------------------------------------------
    # Public API — skills management
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> dict | None:
        """Get a specific skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill info dict, or None if not found.
        """
        service = SkillService()
        try:
            skill = service.get_skill(name)
        except FileNotFoundError:
            return None
        return {
            "name": skill.name,
            "description": skill.description,
            "license": skill.license,
            "category": skill.category,
            "enabled": skill.enabled,
        }

    def update_skill(self, name: str, *, enabled: bool) -> dict:
        """Update a skill's enabled status.

        Args:
            name: Skill name.
            enabled: New enabled status.

        Returns:
            Updated skill info dict.

        Raises:
            ValueError: If the skill is not found.
            OSError: If the config file cannot be written.
        """
        service = SkillService()
        try:
            updated = service.update_skill_enabled(name, enabled=enabled)
        except FileNotFoundError as exc:
            raise ValueError(str(exc)) from exc
        self._agent = None
        return {
            "name": updated.name,
            "description": updated.description,
            "license": updated.license,
            "category": updated.category,
            "enabled": updated.enabled,
        }

    def install_skill(self, skill_path: str | Path) -> dict:
        """Install a skill from a .skill archive (ZIP).

        Args:
            skill_path: Path to the .skill file.

        Returns:
            Dict with success, skill_name, message.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is invalid.
        """
        result = SkillService().install_skill_from_archive(skill_path)
        self._agent = None
        return result

    # ------------------------------------------------------------------
    # Public API — memory management
    # ------------------------------------------------------------------

    def reload_memory(self) -> dict:
        """Reload memory data from file, forcing cache invalidation.

        Returns:
            The reloaded memory data dict.
        """
        from medrix_flow.agents.memory.updater import reload_memory_data

        return reload_memory_data()

    def get_memory_config(self) -> dict:
        """Get memory system configuration.

        Returns:
            Memory config dict.
        """
        from medrix_flow.config.memory_config import get_memory_config

        config = get_memory_config()
        return {
            "enabled": config.enabled,
            "storage_path": config.storage_path,
            "debounce_seconds": config.debounce_seconds,
            "max_facts": config.max_facts,
            "fact_confidence_threshold": config.fact_confidence_threshold,
            "injection_enabled": config.injection_enabled,
            "max_injection_tokens": config.max_injection_tokens,
        }

    def get_memory_status(self) -> dict:
        """Get memory status: config + current data.

        Returns:
            Dict with "config" and "data" keys.
        """
        return {
            "config": self.get_memory_config(),
            "data": self.get_memory(),
        }

    # ------------------------------------------------------------------
    # Public API — file uploads
    # ------------------------------------------------------------------

    @staticmethod
    def _get_uploads_dir(thread_id: str) -> Path:
        """Get (and create) the uploads directory for a thread."""
        base = get_paths().sandbox_uploads_dir(thread_id)
        base.mkdir(parents=True, exist_ok=True)
        return base

    def upload_files(self, thread_id: str, files: list[str | Path]) -> dict:
        """Upload local files into a thread's uploads directory.

        For PDF, PPT, Excel, and Word files, they are also converted to Markdown.

        Args:
            thread_id: Target thread ID.
            files: List of local file paths to upload.

        Returns:
            Dict with success, files, message — matching the Gateway API
            ``UploadResponse`` schema.

        Raises:
            FileNotFoundError: If any file does not exist.
            ValueError: If any supplied path exists but is not a regular file.
        """
        from medrix_flow.utils.file_conversion import CONVERTIBLE_EXTENSIONS, convert_file_to_markdown

        # Validate all files upfront to avoid partial uploads.
        resolved_files = []
        convertible_extensions = {ext.lower() for ext in CONVERTIBLE_EXTENSIONS}
        has_convertible_file = False
        for f in files:
            p = Path(f)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {f}")
            if not p.is_file():
                raise ValueError(f"Path is not a file: {f}")
            resolved_files.append(p)
            if not has_convertible_file and p.suffix.lower() in convertible_extensions:
                has_convertible_file = True

        uploads_dir = self._get_uploads_dir(thread_id)
        uploaded_files: list[dict] = []

        conversion_pool = None
        if has_convertible_file:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                conversion_pool = None
            else:
                import concurrent.futures

                # Reuse one worker when already inside an event loop to avoid
                # creating a new ThreadPoolExecutor per converted file.
                conversion_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _convert_in_thread(path: Path):
            return asyncio.run(convert_file_to_markdown(path))

        try:
            for src_path in resolved_files:
                dest = uploads_dir / src_path.name
                shutil.copy2(src_path, dest)

                info: dict[str, Any] = {
                    "filename": src_path.name,
                    "size": dest.stat().st_size,
                    "path": str(dest),
                    "virtual_path": f"/mnt/user-data/uploads/{src_path.name}",
                    "artifact_url": f"/api/threads/{thread_id}/artifacts/mnt/user-data/uploads/{src_path.name}",
                }

                if src_path.suffix.lower() in convertible_extensions:
                    try:
                        if conversion_pool is not None:
                            md_path = conversion_pool.submit(_convert_in_thread, dest).result()
                        else:
                            md_path = asyncio.run(convert_file_to_markdown(dest))
                    except Exception:
                        logger.warning(
                            "Failed to convert %s to markdown",
                            src_path.name,
                            exc_info=True,
                        )
                        md_path = None

                    if md_path is not None:
                        info["markdown_file"] = md_path.name
                        info["markdown_virtual_path"] = f"/mnt/user-data/uploads/{md_path.name}"
                        info["markdown_artifact_url"] = f"/api/threads/{thread_id}/artifacts/mnt/user-data/uploads/{md_path.name}"

                uploaded_files.append(info)
        finally:
            if conversion_pool is not None:
                conversion_pool.shutdown(wait=True)

        return {
            "success": True,
            "files": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} file(s)",
        }

    def list_uploads(self, thread_id: str) -> dict:
        """List files in a thread's uploads directory.

        Args:
            thread_id: Thread ID.

        Returns:
            Dict with "files" and "count" keys, matching the Gateway API
            ``list_uploaded_files`` response.
        """
        uploads_dir = self._get_uploads_dir(thread_id)
        if not uploads_dir.exists():
            return {"files": [], "count": 0}

        files = []
        with os.scandir(uploads_dir) as entries:
            file_entries = [entry for entry in entries if entry.is_file()]

        for entry in sorted(file_entries, key=lambda item: item.name):
            stat = entry.stat()
            filename = entry.name
            files.append(
                {
                    "filename": filename,
                    "size": stat.st_size,
                    "path": str(Path(entry.path)),
                    "virtual_path": f"/mnt/user-data/uploads/{filename}",
                    "artifact_url": f"/api/threads/{thread_id}/artifacts/mnt/user-data/uploads/{filename}",
                    "extension": Path(filename).suffix,
                    "modified": stat.st_mtime,
                }
            )
        return {"files": files, "count": len(files)}

    def delete_upload(self, thread_id: str, filename: str) -> dict:
        """Delete a file from a thread's uploads directory.

        Args:
            thread_id: Thread ID.
            filename: Filename to delete.

        Returns:
            Dict with success and message, matching the Gateway API
            ``delete_uploaded_file`` response.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If path traversal is detected.
        """
        uploads_dir = self._get_uploads_dir(thread_id)
        file_path = (uploads_dir / filename).resolve()

        try:
            file_path.relative_to(uploads_dir.resolve())
        except ValueError as exc:
            raise PermissionError("Access denied: path traversal detected") from exc

        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {filename}")

        file_path.unlink()
        return {"success": True, "message": f"Deleted {filename}"}

    # ------------------------------------------------------------------
    # Public API — artifacts
    # ------------------------------------------------------------------

    def get_artifact(self, thread_id: str, path: str) -> tuple[bytes, str]:
        """Read an artifact file produced by the agent.

        Args:
            thread_id: Thread ID.
            path: Virtual path (e.g. "mnt/user-data/outputs/file.txt").

        Returns:
            Tuple of (file_bytes, mime_type).

        Raises:
            FileNotFoundError: If the artifact does not exist.
            ValueError: If the path is invalid.
        """
        virtual_prefix = "mnt/user-data"
        clean_path = path.lstrip("/")
        if not clean_path.startswith(virtual_prefix):
            raise ValueError(f"Path must start with /{virtual_prefix}")

        relative = clean_path[len(virtual_prefix) :].lstrip("/")
        base_dir = get_paths().sandbox_user_data_dir(thread_id)
        actual = (base_dir / relative).resolve()

        try:
            actual.relative_to(base_dir.resolve())
        except ValueError as exc:
            raise PermissionError("Access denied: path traversal detected") from exc
        if not actual.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        if not actual.is_file():
            raise ValueError(f"Path is not a file: {path}")

        mime_type, _ = mimetypes.guess_type(actual)
        return actual.read_bytes(), mime_type or "application/octet-stream"
