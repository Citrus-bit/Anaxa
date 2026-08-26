from __future__ import annotations

from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest

from medrix_flow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from medrix_flow.tools.builtins.clarification_tool import ask_clarification_tool


def _request(*, synthetic_data_mode: bool, question: str, context: str | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={
            "name": "ask_clarification",
            "id": "tc-clarify",
            "args": {
                "question": question,
                "clarification_type": "missing_info",
                "context": context,
                "options": [
                    "Upload full data and parameters",
                    "Use assumptions to continue",
                ],
            },
        },
        tool=None,
        state={},
        runtime=SimpleNamespace(context={"synthetic_data_mode": synthetic_data_mode}),
    )


def test_synthetic_mode_suppresses_experiment_data_clarification() -> None:
    middleware = ClarificationMiddleware()
    called = False

    def handler(request: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return ToolMessage("should not run", tool_call_id="tc-clarify")

    result = middleware.wrap_tool_call(
        _request(
            synthetic_data_mode=True,
            question="请上传完整赛题、实验数据、参数、消融设定、图表数值、页数、引用格式和附录要求。",
            context="The PDF paper needs complete experimental results and format requirements.",
        ),
        handler,
    )

    assert called is False
    assert result.goto != END
    assert result.update is not None
    message = result.update["messages"][0]
    assert message.name == "ask_clarification"
    assert "Synthetic Experiment Mode is enabled" in str(message.content)
    assert "Continue by generating reasonable simulation assumptions" in str(message.content)


def test_normal_mode_still_interrupts_for_missing_experiment_data() -> None:
    middleware = ClarificationMiddleware()

    result = middleware.wrap_tool_call(
        _request(
            synthetic_data_mode=False,
            question="Please provide experiment data and parameters.",
        ),
        lambda request: ToolMessage("unused", tool_call_id="tc-clarify"),
    )

    assert result.goto == END
    assert result.update is not None
    message = result.update["messages"][0]
    assert message.name == "ask_clarification"
    assert message.additional_kwargs["clarification"]["question"] == "Please provide experiment data and parameters."


def test_synthetic_mode_still_interrupts_for_official_template_requirements() -> None:
    middleware = ClarificationMiddleware()

    result = middleware.wrap_tool_call(
        _request(
            synthetic_data_mode=True,
            question="Please upload the official contest statement, exact template, page limit, and citation style.",
        ),
        lambda request: ToolMessage("unused", tool_call_id="tc-clarify"),
    )

    assert result.goto == END
    assert result.update is not None
    message = result.update["messages"][0]
    assert message.name == "ask_clarification"
    assert "official contest statement" in str(message.content)


def test_after_model_drops_sibling_tool_calls_before_clarification_interrupts() -> None:
    middleware = ClarificationMiddleware()
    message = AIMessage(
        id="ai-1",
        content="",
        tool_calls=[
            {
                "id": "clarify-1",
                "name": "ask_clarification",
                "args": {"question": "Which directory?"},
            },
            {
                "id": "bash-1",
                "name": "bash",
                "args": {"command": "rm -rf /tmp/work"},
            },
        ],
    )

    result = middleware.after_model(
        {"messages": [message]},
        SimpleNamespace(context={}),
    )

    assert result is not None
    patched = result["messages"][0]
    assert [call["name"] for call in patched.tool_calls] == ["ask_clarification"]
    assert patched.id == "ai-1"


def test_after_model_keeps_siblings_when_clarification_is_disabled() -> None:
    middleware = ClarificationMiddleware()
    message = AIMessage(
        content="",
        tool_calls=[
            {"id": "clarify-1", "name": "ask_clarification", "args": {}},
            {"id": "bash-1", "name": "bash", "args": {"command": "pwd"}},
        ],
    )

    result = middleware.after_model(
        {"messages": [message]},
        SimpleNamespace(context={"disable_clarification": True}),
    )

    assert result is None


def test_agent_graph_does_not_execute_sibling_tool_before_clarification() -> None:
    calls: list[str] = []

    @tool
    def bash(command: str) -> str:
        """Run a shell command for the integration-test sentinel."""
        calls.append(command)
        return "executed"

    class FakeToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = FakeToolCallingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "clarify-1",
                        "name": "ask_clarification",
                        "args": {
                            "question": "Which directory?",
                            "clarification_type": "missing_info",
                        },
                    },
                    {"id": "bash-1", "name": "bash", "args": {"command": "touch /tmp/should-not-run"}},
                ],
            )
        ]
    )
    graph = create_agent(
        model,
        tools=[bash, ask_clarification_tool],
        middleware=[ClarificationMiddleware()],
    )

    result = graph.invoke({"messages": [HumanMessage(content="Please continue.")]})

    assert calls == []
    assert [message.name for message in result["messages"] if isinstance(message, ToolMessage)] == [
        "ask_clarification"
    ]


def test_after_model_drops_siblings_when_clarification_args_are_invalid() -> None:
    middleware = ClarificationMiddleware()
    message = AIMessage(
        content=[
            {"type": "text", "text": "asking"},
            {"type": "tool_use", "id": "clarify-1", "name": "ask_clarification", "input": "{"},
            {"type": "tool_use", "id": "bash-1", "name": "bash", "input": {"command": "rm -rf /"}},
        ],
        tool_calls=[
            {"id": "bash-1", "name": "bash", "args": {"command": "rm -rf /"}},
        ],
        invalid_tool_calls=[
            {
                "type": "invalid_tool_call",
                "id": "clarify-1",
                "name": "ask_clarification",
                "args": "{",
                "error": "Failed to parse tool arguments",
            }
        ],
        additional_kwargs={
            "tool_calls": [
                {
                    "id": "clarify-1",
                    "type": "function",
                    "function": {"name": "ask_clarification", "arguments": "{"},
                },
                {
                    "id": "bash-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"rm -rf /"}'},
                },
            ]
        },
    )

    result = middleware.after_model({"messages": [message]}, SimpleNamespace(context={}))

    assert result is not None
    patched = result["messages"][0]
    assert patched.tool_calls == []
    assert [call["name"] for call in patched.invalid_tool_calls] == ["ask_clarification"]
    assert patched.content == [
        {"type": "text", "text": "asking"},
        {"type": "tool_use", "id": "clarify-1", "name": "ask_clarification", "input": "{"},
    ]
    assert [call["id"] for call in patched.additional_kwargs["tool_calls"]] == ["clarify-1"]
