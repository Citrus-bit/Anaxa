"""Regression tests for malformed and interrupted tool-call history."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai.chat_models.base import _convert_message_to_dict

from medrix_flow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware


def _tool_call(name: str, call_id: str) -> dict:
    return {"name": name, "id": call_id, "args": {}}


def _tool_message(call_id: str, name: str = "bash") -> ToolMessage:
    return ToolMessage(content="result", tool_call_id=call_id, name=name)


def test_inserts_placeholder_for_dangling_structured_call() -> None:
    patched = DanglingToolCallMiddleware()._build_patched_messages(
        [AIMessage(content="", tool_calls=[_tool_call("bash", "call-1")])]
    )

    assert patched is not None
    assert isinstance(patched[1], ToolMessage)
    assert patched[1].tool_call_id == "call-1"
    assert patched[1].status == "error"


def test_invalid_tool_call_gets_safe_arguments_and_recovery_message() -> None:
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "invalid-1",
                "name": "write_file",
                "args": '{"path":"/tmp/report.md"}}',
                "error": "Failed to parse tool arguments: malformed JSON",
            }
        ],
    )

    patched = DanglingToolCallMiddleware()._build_patched_messages([message])

    assert patched is not None
    assert patched[0].invalid_tool_calls[0]["args"] == "{}"
    assert json.loads(_convert_message_to_dict(patched[0])["tool_calls"][0]["function"]["arguments"]) == {}
    assert "arguments were invalid" in patched[1].content
    assert patched[1].tool_call_id == "invalid-1"


def test_raw_provider_arguments_are_sanitized_without_structured_view() -> None:
    message = AIMessage.model_construct(
        content="",
        type="ai",
        tool_calls=[],
        invalid_tool_calls=[],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "raw-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/tmp/a"}}'},
                }
            ]
        },
        response_metadata={},
    )

    patched = DanglingToolCallMiddleware()._build_patched_messages([message])

    assert patched is not None
    raw_args = patched[0].additional_kwargs["tool_calls"][0]["function"]["arguments"]
    assert json.loads(raw_args) == {}
    assert patched[1].tool_call_id == "raw-1"


def test_invalid_tool_call_with_existing_result_is_sanitized_without_duplicate_result() -> None:
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call-1",
                "name": "read_file",
                "args": "[1, 2]",
                "error": "schema validation failed",
            }
        ],
    )

    patched = DanglingToolCallMiddleware()._build_patched_messages(
        [message, _tool_message("call-1", "read_file")]
    )

    assert patched is not None
    assert patched[0].invalid_tool_calls[0]["args"] == "{}"
    assert [msg.tool_call_id for msg in patched if isinstance(msg, ToolMessage)] == ["call-1"]


def test_raw_fallback_is_not_counted_twice_when_invalid_view_exists() -> None:
    message = AIMessage.model_construct(
        content="",
        type="ai",
        tool_calls=[],
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "call-1",
                "name": "read_file",
                "args": "{",
                "error": "invalid JSON",
            }
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{"},
                }
            ]
        },
        response_metadata={},
    )

    patched = DanglingToolCallMiddleware()._build_patched_messages([message])

    assert patched is not None
    assert [msg.tool_call_id for msg in patched if isinstance(msg, ToolMessage)] == ["call-1"]


def test_non_tool_messages_are_preserved_around_inserted_placeholder() -> None:
    messages = [
        HumanMessage(content="start"),
        AIMessage(content="", tool_calls=[_tool_call("bash", "call-1")]),
        HumanMessage(content="interrupted"),
    ]

    patched = DanglingToolCallMiddleware()._build_patched_messages(messages)

    assert patched is not None
    assert [type(message) for message in patched] == [HumanMessage, AIMessage, ToolMessage, HumanMessage]
