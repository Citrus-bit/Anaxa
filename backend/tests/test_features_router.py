from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.deps import get_config
from app.gateway.routers import features
from deerflow.config.agents_config import AgentConfig
from deerflow.config.extensions_config import ExtensionsConfig, McpServerConfig
from deerflow.skills import Skill
from deerflow.skills.types import SkillCategory
from deerflow.subagents import SubagentConfig


def _app_with_config(
    *,
    agents_api_enabled: bool,
    browser_enabled: bool = False,
    browser_extra: dict | None = None,
    mcp_tasks_available: bool = False,
    subagent_batches_available: bool = False,
    subagent_batch_repo_available: bool | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.mcp_tasks_available = mcp_tasks_available
    app.state.subagent_batches_available = subagent_batches_available
    if subagent_batch_repo_available is None:
        subagent_batch_repo_available = subagent_batches_available
    app.state.subagent_batch_repo = object() if subagent_batch_repo_available else None
    app.include_router(features.router)
    tools = (
        [
            SimpleNamespace(name="browser_navigate", model_extra=browser_extra or {}),
        ]
        if browser_enabled
        else []
    )
    fake_config = SimpleNamespace(
        agents_api=SimpleNamespace(enabled=agents_api_enabled),
        tools=tools,
        subagent_runtime=SimpleNamespace(max_running=3),
    )
    app.dependency_overrides[get_config] = lambda: fake_config
    return app


def test_features_reports_agents_api_enabled() -> None:
    with TestClient(_app_with_config(agents_api_enabled=True)) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json() == {
        "agents_api": {"enabled": True},
        "browser_control": {"enabled": False},
        "mcp_tasks": {"enabled": False},
        "subagent_batches": {
            "enabled": False,
            "repository_available": False,
            "worker_running": False,
            "max_running": 3,
        },
    }


def test_features_reports_agents_api_disabled() -> None:
    with TestClient(_app_with_config(agents_api_enabled=False)) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json() == {
        "agents_api": {"enabled": False},
        "browser_control": {"enabled": False},
        "mcp_tasks": {"enabled": False},
        "subagent_batches": {
            "enabled": False,
            "repository_available": False,
            "worker_running": False,
            "max_running": 3,
        },
    }


def test_features_reports_mcp_tasks_startup_capability() -> None:
    with TestClient(_app_with_config(agents_api_enabled=True, mcp_tasks_available=True)) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["mcp_tasks"] == {"enabled": True}


def test_features_reports_subagent_batch_startup_capability() -> None:
    with TestClient(
        _app_with_config(
            agents_api_enabled=True,
            subagent_batches_available=True,
        )
    ) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["subagent_batches"] == {
        "enabled": True,
        "repository_available": True,
        "worker_running": True,
        "max_running": 3,
    }


def test_features_distinguishes_batch_history_from_worker_availability() -> None:
    with TestClient(
        _app_with_config(
            agents_api_enabled=True,
            subagent_batches_available=False,
            subagent_batch_repo_available=True,
        )
    ) as client:
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["subagent_batches"] == {
        "enabled": False,
        "repository_available": True,
        "worker_running": False,
        "max_running": 3,
    }


def test_features_reports_browser_control_enabled_when_configured_and_runtime_available() -> None:
    with (
        patch("app.gateway.browser_capability.importlib.util.find_spec", return_value=object()),
        TestClient(_app_with_config(agents_api_enabled=True, browser_enabled=True)) as client,
    ):
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["browser_control"] == {"enabled": True}


def test_features_reports_browser_control_disabled_when_runtime_missing() -> None:
    with (
        patch("app.gateway.browser_capability.importlib.util.find_spec", return_value=None),
        TestClient(_app_with_config(agents_api_enabled=True, browser_enabled=True)) as client,
    ):
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["browser_control"] == {"enabled": False}


def test_features_reports_browser_control_disabled_for_unguarded_cdp() -> None:
    with (
        patch("app.gateway.browser_capability.importlib.util.find_spec", return_value=object()),
        TestClient(
            _app_with_config(
                agents_api_enabled=True,
                browser_enabled=True,
                browser_extra={"cdp_url": "http://127.0.0.1:9222"},
            ),
        ) as client,
    ):
        response = client.get("/api/features")
    assert response.status_code == 200
    assert response.json()["browser_control"] == {"enabled": False}


def test_feature_inventory_lists_deerflow_registries_without_secrets() -> None:
    extensions = ExtensionsConfig(
        mcp_servers={
            "paper-search": McpServerConfig(
                enabled=True,
                type="http",
                url="https://mcp.example.com/sse?token=secret#frag",
                headers={"Authorization": "Bearer secret", "X-Empty": ""},
                env={"API_KEY": "secret"},
                description="Academic search tools",
            )
        },
        skills={},
    )
    storage = SimpleNamespace(
        load_skills=lambda enabled_only=False: [
            Skill(
                name="literature-finder",
                description="Find relevant papers",
                license="MIT",
                skill_dir=Path("/skills/public/literature-finder"),
                skill_file=Path("/skills/public/literature-finder/SKILL.md"),
                relative_path=Path("literature-finder"),
                category=SkillCategory.PUBLIC,
                enabled=True,
            )
        ]
    )

    with (
        patch(
            "app.gateway.routers.features.list_custom_agents",
            return_value=[
                AgentConfig(
                    name="domain-reviewer",
                    description="Custom reviewer",
                    model="gpt-5.5",
                    tool_groups=["academic"],
                )
            ],
        ) as list_agents,
        patch(
            "app.gateway.routers.features.list_subagents",
            return_value=[
                SubagentConfig(
                    name="academic-researcher",
                    description="Research subagent",
                    model="inherit",
                    tools=["paper_search"],
                )
            ],
        ) as list_registered_subagents,
        patch("app.gateway.routers.features.ExtensionsConfig.from_file", return_value=extensions),
        patch("app.gateway.routers.features.get_effective_user_id", return_value="test-user"),
        patch(
            "app.gateway.routers.features.get_or_new_user_skill_storage",
            return_value=storage,
        ) as get_skill_storage,
        TestClient(_app_with_config(agents_api_enabled=True)) as client,
    ):
        response = client.get("/api/features/inventory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agents"] == [
        {
            "name": "default",
            "description": "Primary Anaxa orchestrator for chat, research routing, artifact generation, tool use, memory, and human-gated long-running workflows.",
            "model": None,
            "tool_groups": None,
            "kind": "system",
            "readonly": True,
        },
        {
            "name": "domain-reviewer",
            "description": "Custom reviewer",
            "model": "gpt-5.5",
            "tool_groups": ["academic"],
            "kind": "custom",
            "readonly": False,
        },
        {
            "name": "academic-researcher",
            "description": "Research subagent",
            "model": None,
            "tool_groups": ["paper_search"],
            "kind": "subagent",
            "readonly": True,
        },
    ]
    assert payload["tools"] == [
        {
            "name": "paper-search",
            "enabled": True,
            "transport": "http",
            "description": "Academic search tools",
            "command": None,
            "url": "https://mcp.example.com/sse",
            "args": [],
            "env_keys": [{"key": "API_KEY", "configured": True}],
            "header_keys": [
                {"key": "Authorization", "configured": True},
                {"key": "X-Empty", "configured": False},
            ],
            "oauth_enabled": False,
        }
    ]
    assert payload["skills"] == [
        {
            "name": "literature-finder",
            "description": "Find relevant papers",
            "license": "MIT",
            "category": "public",
            "enabled": True,
        }
    ]
    assert "secret" not in response.text
    list_agents.assert_called_once_with(user_id="test-user")
    list_registered_subagents.assert_called_once()
    get_skill_storage.assert_called_once()


def test_feature_inventory_handles_empty_deerflow_registries() -> None:
    storage = SimpleNamespace(load_skills=lambda enabled_only=False: [])
    with (
        patch("app.gateway.routers.features.list_custom_agents", return_value=[]),
        patch("app.gateway.routers.features.list_subagents", return_value=[]),
        patch(
            "app.gateway.routers.features.ExtensionsConfig.from_file",
            return_value=ExtensionsConfig(mcp_servers={}, skills={}),
        ),
        patch("app.gateway.routers.features.get_or_new_user_skill_storage", return_value=storage),
        TestClient(_app_with_config(agents_api_enabled=False)) as client,
    ):
        response = client.get("/api/features/inventory")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {
                "name": "default",
                "description": "Primary Anaxa orchestrator for chat, research routing, artifact generation, tool use, memory, and human-gated long-running workflows.",
                "model": None,
                "tool_groups": None,
                "kind": "system",
                "readonly": True,
            }
        ],
        "tools": [],
        "skills": [],
    }
