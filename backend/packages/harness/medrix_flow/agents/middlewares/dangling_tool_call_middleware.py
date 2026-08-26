"""Middleware to fix dangling tool calls in message history.

A dangling tool call occurs when an AIMessage contains tool_calls but there are
no corresponding ToolMessages in the history (e.g., due to user interruption or
request cancellation). This causes LLM errors due to incomplete message format.

This middleware intercepts the model call to detect and patch such gaps by
inserting synthetic ToolMessages with an error indicator immediately after the
AIMessage that made the tool calls, ensuring correct message ordering.

Note: Uses wrap_model_call instead of before_model to ensure patches are inserted
at the correct positions (immediately after each dangling AIMessage), not appended
to the end of the message list as before_model + add_messages reducer would do.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

_UNKNOWN_TOOL_NAME = "unknown_tool"
_MAX_RECOVERY_ERROR_DETAIL_LEN = 500


def _valid_tool_name(name: object) -> bool:
    return isinstance(name, str) and bool(name.strip())


def _normalize_tool_name(name: object) -> str:
    return name.strip() if _valid_tool_name(name) else _UNKNOWN_TOOL_NAME


def _parse_json_object(value: object) -> dict | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_tool_arguments(arguments: object) -> str:
    """Return a JSON object string safe for OpenAI-compatible serialization."""
    if isinstance(arguments, dict):
        try:
            return json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return "{}"
    return arguments if _parse_json_object(arguments) is not None else "{}"


class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """Inserts placeholder ToolMessages for dangling tool calls before model invocation.

    Scans the message history for AIMessages whose tool_calls lack corresponding
    ToolMessages, and injects synthetic error responses immediately after the
    offending AIMessage so the LLM receives a well-formed conversation.
    """

    @staticmethod
    def _message_tool_calls(msg) -> list[dict]:
        """Return the provider-visible tool calls, including malformed calls."""
        normalized: list[dict] = []
        tool_calls = getattr(msg, "tool_calls", None) or []
        invalid_tool_calls = getattr(msg, "invalid_tool_calls", None) or []

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            normalized_call = dict(tool_call)
            original_name = tool_call.get("name")
            normalized_call["name"] = _normalize_tool_name(original_name)
            if not _valid_tool_name(original_name):
                normalized_call["invalid_tool_name"] = True
            normalized.append(normalized_call)

        # Raw provider calls are a fallback view. Do not count them alongside
        # structured or invalid calls, or one provider call would get two results.
        raw_tool_calls = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []
        if not tool_calls and not invalid_tool_calls and isinstance(raw_tool_calls, list):
            for raw_call in raw_tool_calls:
                if not isinstance(raw_call, dict):
                    continue
                function = raw_call.get("function")
                name = raw_call.get("name")
                if not name and isinstance(function, dict):
                    name = function.get("name")
                args = raw_call.get("args", {})
                if not args and isinstance(function, dict):
                    parsed_args = _parse_json_object(function.get("arguments"))
                    args = parsed_args if parsed_args is not None else {}
                normalized_call = {
                    "id": raw_call.get("id"),
                    "name": _normalize_tool_name(name),
                    "args": args if isinstance(args, dict) else {},
                }
                if not _valid_tool_name(name):
                    normalized_call["invalid_tool_name"] = True
                normalized.append(normalized_call)

        for invalid_call in invalid_tool_calls:
            if not isinstance(invalid_call, dict):
                continue
            original_name = invalid_call.get("name")
            normalized.append(
                {
                    "id": invalid_call.get("id"),
                    "name": _normalize_tool_name(original_name),
                    "args": {},
                    "invalid": True,
                    "invalid_tool_name": not _valid_tool_name(original_name),
                    "error": invalid_call.get("error"),
                }
            )
        return normalized

    @staticmethod
    def _synthetic_tool_message_content(tool_call: dict) -> str:
        if tool_call.get("invalid_tool_name"):
            return "[Tool call could not be executed because its name was missing or empty.]"
        if tool_call.get("invalid"):
            error = tool_call.get("error")
            detail = error[:_MAX_RECOVERY_ERROR_DETAIL_LEN] if isinstance(error, str) else ""
            suffix = f": {detail}" if detail else ""
            return f"[Tool call could not be executed because its arguments were invalid{suffix}]"
        return "[Tool call was interrupted and did not return a result.]"

    @staticmethod
    def _sanitize_ai_message_tool_calls(msg):
        if getattr(msg, "type", None) != "ai":
            return msg

        changed = False
        update: dict = {}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            sanitized_tool_calls = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    sanitized_tool_calls.append(tool_call)
                    continue
                sanitized = dict(tool_call)
                normalized_name = _normalize_tool_name(sanitized.get("name"))
                if sanitized.get("name") != normalized_name:
                    sanitized["name"] = normalized_name
                    changed = True
                sanitized_tool_calls.append(sanitized)
            if changed:
                update["tool_calls"] = sanitized_tool_calls

        invalid_tool_calls = getattr(msg, "invalid_tool_calls", None)
        if invalid_tool_calls:
            sanitized_invalid_tool_calls = []
            invalid_changed = False
            for invalid_call in invalid_tool_calls:
                if not isinstance(invalid_call, dict):
                    sanitized_invalid_tool_calls.append(invalid_call)
                    continue
                sanitized = dict(invalid_call)
                normalized_name = _normalize_tool_name(sanitized.get("name"))
                normalized_args = _normalize_tool_arguments(sanitized.get("args"))
                if sanitized.get("name") != normalized_name:
                    sanitized["name"] = normalized_name
                    invalid_changed = True
                if sanitized.get("args") != normalized_args:
                    sanitized["args"] = normalized_args
                    invalid_changed = True
                sanitized_invalid_tool_calls.append(sanitized)
            if invalid_changed:
                update["invalid_tool_calls"] = sanitized_invalid_tool_calls
                changed = True

        additional_kwargs = dict(getattr(msg, "additional_kwargs", {}) or {})
        raw_tool_calls = additional_kwargs.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            sanitized_raw_calls = []
            raw_changed = False
            for raw_call in raw_tool_calls:
                if not isinstance(raw_call, dict):
                    sanitized_raw_calls.append(raw_call)
                    continue
                sanitized_raw = dict(raw_call)
                function = sanitized_raw.get("function")
                if isinstance(function, dict):
                    sanitized_function = dict(function)
                    normalized_name = _normalize_tool_name(sanitized_function.get("name"))
                    normalized_args = _normalize_tool_arguments(sanitized_function.get("arguments"))
                    if sanitized_function.get("name") != normalized_name:
                        sanitized_function["name"] = normalized_name
                        raw_changed = True
                    if sanitized_function.get("arguments") != normalized_args:
                        sanitized_function["arguments"] = normalized_args
                        raw_changed = True
                    if sanitized_function != function:
                        sanitized_raw["function"] = sanitized_function
                else:
                    normalized_name = _normalize_tool_name(sanitized_raw.get("name"))
                    if sanitized_raw.get("name") != normalized_name:
                        sanitized_raw["name"] = normalized_name
                        raw_changed = True
                sanitized_raw_calls.append(sanitized_raw)
            if raw_changed:
                additional_kwargs["tool_calls"] = sanitized_raw_calls
                update["additional_kwargs"] = additional_kwargs
                changed = True

        return msg.model_copy(update=update) if changed else msg

    def _build_patched_messages(self, messages: list) -> list | None:
        """Return a new message list with patches inserted at the correct positions.

        For each AIMessage with dangling tool_calls (no corresponding ToolMessage),
        a synthetic ToolMessage is inserted immediately after that AIMessage.
        Returns None if no patches are needed.
        """
        # Collect IDs of all existing ToolMessages
        existing_tool_msg_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                existing_tool_msg_ids.add(msg.tool_call_id)

        # Check if any patching is needed, including invalid/raw provider calls.
        needs_patch = False
        for msg in messages:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                if tc_id and tc_id not in existing_tool_msg_ids:
                    needs_patch = True
                    break
            if needs_patch:
                break

        # Build new list with patches inserted right after each dangling AIMessage
        patched: list = []
        patched_ids: set[str] = set()
        patch_count = 0
        for msg in messages:
            sanitized_msg = self._sanitize_ai_message_tool_calls(msg)
            patched.append(sanitized_msg)
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                if tc_id and tc_id not in existing_tool_msg_ids and tc_id not in patched_ids:
                    patched.append(
                        ToolMessage(
                            content=self._synthetic_tool_message_content(tc),
                            tool_call_id=tc_id,
                            name=tc.get("name", _UNKNOWN_TOOL_NAME),
                            status="error",
                        )
                    )
                    patched_ids.add(tc_id)
                    patch_count += 1

        if patched == messages and not needs_patch:
            return None
        if patch_count:
            logger.warning("Injecting %d placeholder ToolMessage(s) for dangling tool calls", patch_count)
        return patched

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return await handler(request)
