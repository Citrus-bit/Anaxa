"""Middleware for intercepting clarification requests and presenting them to the user."""

import re
from collections.abc import Callable
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

ASK_CLARIFICATION_TOOL_NAME = "ask_clarification"


class ClarificationMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


SYNTHETIC_SUBSTITUTABLE_RE = re.compile(
    r"(experiment|experimental|data|dataset|parameter|metric|baseline|ablation|robustness|figure|table|plot|chart|"
    r"simulation|simulated|synthetic|sample|seed|effect size|model setting|hyperparameter|appendix|code|pdf|latex|"
    r"实验|数据|数据集|参数|指标|基线|baseline|消融|鲁棒|图表|作图|绘图|表格|样本|随机种子|效应|模型|超参数|附录|代码|论文|PDF)",
    re.IGNORECASE,
)

NON_SUBSTITUTABLE_RE = re.compile(
    r"(official contest statement|official problem statement|official template|credential|login|"
    r"destructive|delete|overwrite|官方赛题|官方模板|账号|登录|凭据|删除|覆盖|销毁)",
    re.IGNORECASE,
)

STRICT_FORMAT_RE = re.compile(
    r"((strict|exact|mandated|required|official).{0,40}(template|format|page limit|citation style|author|deadline)|"
    r"(严格|精确|必须|指定|官方).{0,20}(模板|格式|页数|引用格式|作者|署名|截止))",
    re.IGNORECASE,
)


def _runtime_synthetic_mode(runtime: Any) -> bool:
    context = getattr(runtime, "context", None)
    if isinstance(context, dict) and context.get("synthetic_data_mode"):
        return True

    config = getattr(runtime, "config", None)
    if isinstance(config, dict):
        configurable = config.get("configurable")
        if isinstance(configurable, dict) and configurable.get("synthetic_data_mode"):
            return True

    return False


def _clarification_text(args: dict) -> str:
    parts: list[str] = []
    for key in ("question", "context", "clarification_type"):
        value = args.get(key)
        if isinstance(value, str):
            parts.append(value)
    options = args.get("options")
    if isinstance(options, list):
        parts.extend(str(item) for item in options if item is not None)
    return "\n".join(parts)


def _is_synthetic_substitutable_clarification(args: dict) -> bool:
    text = _clarification_text(args)
    if not text:
        return False
    if NON_SUBSTITUTABLE_RE.search(text):
        return False
    if STRICT_FORMAT_RE.search(text) and not SYNTHETIC_SUBSTITUTABLE_RE.search(text):
        return False
    return bool(SYNTHETIC_SUBSTITUTABLE_RE.search(text))


class ClarificationMiddleware(AgentMiddleware[ClarificationMiddlewareState]):
    """Intercepts clarification tool calls and interrupts execution to present questions to the user.

    When the model calls the `ask_clarification` tool, this middleware:
    1. Intercepts the tool call before execution
    2. Extracts the clarification question and metadata
    3. Formats a user-friendly message
    4. Returns a Command that interrupts execution and presents the question
    5. Waits for user response before continuing

    This replaces the tool-based approach where clarification continued the conversation flow.
    """

    state_schema = ClarificationMiddlewareState

    @staticmethod
    def _filter_tool_use_content(content: Any, kept_ids: set[str], kept_names: set[str]) -> Any:
        """Remove provider content blocks for tool calls dropped from the AI message."""
        if not isinstance(content, list):
            return content
        filtered: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"tool_use", "function_call"}:
                block_id = block.get("id")
                if isinstance(block_id, str) and block_id:
                    if block_id not in kept_ids:
                        continue
                elif block.get("type") == "function_call":
                    name = block.get("name")
                    if not isinstance(name, str) or name not in kept_names:
                        continue
            filtered.append(block)
        return filtered

    @staticmethod
    def _filter_raw_tool_calls(
        additional_kwargs: dict[str, Any], kept_ids: set[str], kept_names: set[str]
    ) -> dict[str, Any]:
        """Keep raw provider tool-call metadata in sync with structured calls."""
        raw_tool_calls = additional_kwargs.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return additional_kwargs

        filtered: list[Any] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                filtered.append(raw_call)
                continue
            call_id = raw_call.get("id")
            function = raw_call.get("function")
            call_name = raw_call.get("name")
            if isinstance(function, dict):
                call_name = function.get("name", call_name)
            if isinstance(call_id, str) and call_id:
                if call_id in kept_ids:
                    filtered.append(raw_call)
            elif isinstance(call_name, str) and call_name in kept_names:
                filtered.append(raw_call)

        if filtered:
            additional_kwargs["tool_calls"] = filtered
        else:
            additional_kwargs.pop("tool_calls", None)
        return additional_kwargs

    def _drop_parallel_siblings(self, state: ClarificationMiddlewareState, runtime: Any) -> dict[str, Any] | None:
        """Keep clarification calls from executing alongside unrelated tools."""
        context = getattr(runtime, "context", None)
        if isinstance(context, dict) and context.get("disable_clarification"):
            return None

        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        last = messages[-1]
        tool_calls = [call for call in (last.tool_calls or []) if isinstance(call, dict)]
        invalid_tool_calls = [
            call
            for call in (getattr(last, "invalid_tool_calls", None) or [])
            if isinstance(call, dict)
        ]
        clarification_calls = [
            call for call in tool_calls if call.get("name") == ASK_CLARIFICATION_TOOL_NAME
        ]
        invalid_clarification_calls = [
            call for call in invalid_tool_calls if call.get("name") == ASK_CLARIFICATION_TOOL_NAME
        ]
        sibling_calls = [
            call for call in tool_calls if call.get("name") != ASK_CLARIFICATION_TOOL_NAME
        ]
        if (not clarification_calls and not invalid_clarification_calls) or not sibling_calls:
            return None

        kept_ids = {
            str(call["id"])
            for call in [*clarification_calls, *invalid_clarification_calls]
            if isinstance(call.get("id"), str) and call["id"]
        }
        kept_names = {
            str(call["name"])
            for call in [*clarification_calls, *invalid_clarification_calls]
            if isinstance(call.get("name"), str) and call["name"]
        }
        content = self._filter_tool_use_content(last.content, kept_ids, kept_names)
        additional_kwargs = self._filter_raw_tool_calls(
            dict(getattr(last, "additional_kwargs", {}) or {}), kept_ids, kept_names
        )
        patched = last.model_copy(
            update={
                "tool_calls": clarification_calls,
                "content": content,
                "additional_kwargs": additional_kwargs,
            }
        )
        return {"messages": [patched]}

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters.

        Args:
            text: Text to check

        Returns:
            True if text contains Chinese characters
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _format_clarification_message(self, args: dict) -> str:
        """Format the clarification arguments into a user-friendly message.

        Args:
            args: The tool call arguments containing clarification details

        Returns:
            Formatted message string
        """
        question = args.get("question", "")
        clarification_type = args.get("clarification_type", "missing_info")
        context = args.get("context")
        options = args.get("options", [])

        # Type-specific icons
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "🤔",
            "approach_choice": "🔀",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }

        icon = type_icons.get(clarification_type, "❓")

        # Build the message naturally
        message_parts = []

        # Add icon and question together for a more natural flow
        if context:
            # If there's context, present it first as background
            message_parts.append(f"{icon} {context}")
            message_parts.append(f"\n{question}")
        else:
            # Just the question with icon
            message_parts.append(f"{icon} {question}")

        # Add options in a cleaner format
        if options and len(options) > 0:
            message_parts.append("")  # blank line for spacing
            for i, option in enumerate(options, 1):
                message_parts.append(f"  {i}. {option}")

        return "\n".join(message_parts)

    def _build_clarification_payload(self, args: dict) -> dict:
        """Build a structured payload for the frontend clarification card."""
        options = args.get("options", []) or []
        return {
            "question": args.get("question", ""),
            "clarification_type": args.get("clarification_type", "missing_info"),
            "context": args.get("context"),
            "options": options,
            "allow_custom_input": True,
        }

    def _handle_synthetic_substitution(self, request: ToolCallRequest) -> ToolMessage:
        tool_call_id = request.tool_call.get("id", "")
        return ToolMessage(
            content=(
                "Synthetic Experiment Mode is enabled. Do not ask the user for missing "
                "personal experiment data, parameters, ablation settings, plotting data, "
                "or figure/table values. Continue by generating reasonable simulation "
                "assumptions, synthetic results, analyses, and manuscript-ready artifacts. "
                "Only ask again if the missing information is an official template, exact "
                "contest statement, required formatting constraint, credential, or destructive-operation approval."
            ),
            tool_call_id=tool_call_id,
            name="ask_clarification",
        )

    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """Handle clarification request and return command to interrupt execution.

        Args:
            request: Tool call request

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # Extract clarification arguments
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        if _runtime_synthetic_mode(getattr(request, "runtime", None)) and _is_synthetic_substitutable_clarification(args):
            print("[ClarificationMiddleware] Suppressed substitutable clarification in Synthetic Experiment Mode")
            print(f"[ClarificationMiddleware] Question: {question}")
            return Command(update={"messages": [self._handle_synthetic_substitution(request)]})

        print("[ClarificationMiddleware] Intercepted clarification request")
        print(f"[ClarificationMiddleware] Question: {question}")

        # Format the clarification message
        formatted_message = self._format_clarification_message(args)

        # Get the tool call ID
        tool_call_id = request.tool_call.get("id", "")

        # Create a ToolMessage with the formatted question
        # This will be added to the message history
        tool_message = ToolMessage(
            content=formatted_message,
            tool_call_id=tool_call_id,
            name="ask_clarification",
            additional_kwargs={"clarification": self._build_clarification_payload(args)},
        )

        # Return a Command that:
        # 1. Adds the formatted tool message
        # 2. Interrupts execution by going to __end__
        # Note: We don't add an extra AIMessage here - the frontend will detect
        # and display ask_clarification tool messages directly
        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (sync version).

        Args:
            request: Tool call request
            handler: Original tool execution handler

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # Check if this is an ask_clarification tool call
        if request.tool_call.get("name") != ASK_CLARIFICATION_TOOL_NAME:
            # Not a clarification call, execute normally
            return handler(request)

        return self._handle_clarification(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (async version).

        Args:
            request: Tool call request
            handler: Original tool execution handler (async)

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # Check if this is an ask_clarification tool call
        if request.tool_call.get("name") != ASK_CLARIFICATION_TOOL_NAME:
            # Not a clarification call, execute normally
            return await handler(request)

        return self._handle_clarification(request)

    @override
    def after_model(self, state: ClarificationMiddlewareState, runtime: Any) -> dict[str, Any] | None:
        return self._drop_parallel_siblings(state, runtime)

    @override
    async def aafter_model(self, state: ClarificationMiddlewareState, runtime: Any) -> dict[str, Any] | None:
        return self._drop_parallel_siblings(state, runtime)
