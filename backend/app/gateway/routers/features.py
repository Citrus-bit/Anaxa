"""Read-only feature flags and configured runtime inventory.

Reports which optional features are exposed over HTTP so the frontend can gate
UI and avoid firing requests that the backend would reject. Config-only flags
read through ``get_config`` so edits to ``config.yaml`` take effect on the next
request, while startup-scoped capabilities report the runtime that actually
started. The separate inventory endpoint preserves Anaxa's read-only catalog of
agents, MCP servers, and skills.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.browser_capability import browser_capability
from app.gateway.deps import get_config
from deerflow.config.agents_config import AgentConfig, list_custom_agents
from deerflow.config.app_config import AppConfig
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.skills import Skill
from deerflow.skills.storage import get_or_new_user_skill_storage
from deerflow.skills.types import SkillCategory
from deerflow.subagents import SubagentConfig, list_subagents
from deerflow.subagents.capacity import configured_subagent_max_running

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["features"])


class AgentsApiFeature(BaseModel):
    """Availability of the custom-agent management API."""

    enabled: bool = Field(..., description="Whether the agents_api routes are exposed over HTTP")


class BrowserControlFeature(BaseModel):
    """Availability of live agentic browser control."""

    enabled: bool = Field(..., description="Whether the live browser routes and UI are available")


class McpTasksFeature(BaseModel):
    """Availability of the durable MCP task runtime."""

    enabled: bool = Field(..., description="Whether durable MCP task APIs and UI are available")


class SubagentBatchesFeature(BaseModel):
    """Persistence, worker, and process capacity for native-subagent batches."""

    enabled: bool = Field(..., description="Compatibility alias for worker_running")
    repository_available: bool = Field(..., description="Whether durable batch history APIs are available")
    worker_running: bool = Field(..., description="Whether this Gateway process is executing durable batch work")
    max_running: int = Field(..., description="Native subagent execution slots in this Gateway process")


class FeaturesResponse(BaseModel):
    """Frontend-facing feature availability flags."""

    agents_api: AgentsApiFeature
    browser_control: BrowserControlFeature
    mcp_tasks: McpTasksFeature
    subagent_batches: SubagentBatchesFeature


class FeatureAgentResponse(BaseModel):
    """Read-only agent or subagent catalog entry."""

    name: str = Field(..., description="Agent name")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    kind: str = Field(default="custom", description="Agent kind: system, custom, or subagent")
    readonly: bool = Field(default=False, description="Whether the agent is read-only")


class RedactedConfigKey(BaseModel):
    """Configuration-key presence without its secret value."""

    key: str = Field(..., description="Configuration key name")
    configured: bool = Field(default=True, description="Whether a value is configured")


class FeatureToolResponse(BaseModel):
    """Sanitized MCP server catalog entry."""

    name: str = Field(..., description="MCP server name")
    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")
    transport: str = Field(default="stdio", description="MCP transport type")
    description: str = Field(default="", description="Server description")
    command: str | None = Field(default=None, description="Command summary for stdio servers")
    url: str | None = Field(default=None, description="URL without query or fragment")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env_keys: list[RedactedConfigKey] = Field(default_factory=list, description="Redacted environment variable keys")
    header_keys: list[RedactedConfigKey] = Field(default_factory=list, description="Redacted HTTP header keys")
    oauth_enabled: bool = Field(default=False, description="Whether OAuth token injection is configured")


class FeatureSkillResponse(BaseModel):
    """Read-only skill catalog entry."""

    name: str = Field(..., description="Skill name")
    description: str = Field(default="", description="Skill description")
    license: str | None = Field(default=None, description="Skill license")
    category: SkillCategory | str = Field(..., description="Skill category")
    enabled: bool = Field(default=True, description="Whether this skill is enabled")


class FeatureInventoryResponse(BaseModel):
    """Read-only inventory kept separate from frontend capability flags."""

    agents: list[FeatureAgentResponse] = Field(default_factory=list)
    tools: list[FeatureToolResponse] = Field(default_factory=list)
    skills: list[FeatureSkillResponse] = Field(default_factory=list)


def _agent_to_response(agent: AgentConfig) -> FeatureAgentResponse:
    return FeatureAgentResponse(
        name=agent.name,
        description=agent.description,
        model=agent.model,
        tool_groups=agent.tool_groups,
        kind="custom",
        readonly=False,
    )


def _default_agent_response() -> FeatureAgentResponse:
    return FeatureAgentResponse(
        name="default",
        description=("Primary Anaxa orchestrator for chat, research routing, artifact generation, tool use, memory, and human-gated long-running workflows."),
        model=None,
        tool_groups=None,
        kind="system",
        readonly=True,
    )


def _subagent_to_response(subagent: SubagentConfig) -> FeatureAgentResponse:
    return FeatureAgentResponse(
        name=subagent.name,
        description=subagent.description,
        model=None if subagent.model == "inherit" else subagent.model,
        tool_groups=subagent.tools,
        kind="subagent",
        readonly=True,
    )


def _load_agents(config: AppConfig | None = None, user_id: str | None = None) -> list[FeatureAgentResponse]:
    """Build the agent catalog from DeerFlow's custom-agent/subagent registries."""
    if config is None:
        from deerflow.config.app_config import get_app_config

        config = get_app_config()
    if user_id is None:
        user_id = get_effective_user_id()

    configured_agents = [_agent_to_response(agent) for agent in list_custom_agents(user_id=user_id)]
    registered_subagents = [_subagent_to_response(subagent) for subagent in list_subagents(app_config=config)]

    agents_by_name: dict[str, FeatureAgentResponse] = {}
    for agent in [_default_agent_response(), *configured_agents, *registered_subagents]:
        agents_by_name.setdefault(agent.name, agent)
    return list(agents_by_name.values())


def _redacted_keys(values: dict[str, str]) -> list[RedactedConfigKey]:
    return [RedactedConfigKey(key=key, configured=bool(str(value).strip())) for key, value in sorted(values.items())]


def _sanitize_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return value
    # Strip credentials as well as query/fragment material. ``netloc`` may
    # contain ``user:password@host`` for basic-auth MCP endpoints.
    authority = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))


def _load_tools() -> list[FeatureToolResponse]:
    config = ExtensionsConfig.from_file()
    return [
        FeatureToolResponse(
            name=name,
            enabled=server.enabled,
            transport=server.type,
            description=server.description,
            command=server.command,
            url=_sanitize_url(server.url),
            args=list(server.args),
            env_keys=_redacted_keys(server.env),
            header_keys=_redacted_keys(server.headers),
            oauth_enabled=server.oauth is not None and server.oauth.enabled,
        )
        for name, server in sorted(config.mcp_servers.items())
    ]


def _skill_to_response(skill: Skill) -> FeatureSkillResponse:
    return FeatureSkillResponse(
        name=skill.name,
        description=skill.description,
        license=skill.license,
        category=skill.category,
        enabled=skill.enabled,
    )


def _load_inventory(config: AppConfig, user_id: str) -> FeatureInventoryResponse:
    skills = get_or_new_user_skill_storage(user_id, app_config=config).load_skills(enabled_only=False)
    return FeatureInventoryResponse(
        agents=_load_agents(config, user_id),
        tools=_load_tools(),
        skills=[_skill_to_response(skill) for skill in skills],
    )


@router.get(
    "/features",
    response_model=FeaturesResponse,
    summary="List Feature Flags",
    description="Report which optional features are available, so the frontend can gate UI before issuing requests.",
)
async def list_features(request: Request, config: AppConfig = Depends(get_config)) -> FeaturesResponse:
    """Return availability of optional frontend features."""
    browser = browser_capability(config)
    subagent_batch_worker_running = bool(getattr(request.app.state, "subagent_batches_available", False))
    return FeaturesResponse(
        agents_api=AgentsApiFeature(enabled=config.agents_api.enabled),
        browser_control=BrowserControlFeature(enabled=browser.available),
        # MCP task bindings and the submitter are startup-scoped. Report the
        # capability that actually started rather than a hot-reloaded config
        # value that would require a Gateway restart to take effect.
        mcp_tasks=McpTasksFeature(enabled=bool(getattr(request.app.state, "mcp_tasks_available", False))),
        subagent_batches=SubagentBatchesFeature(
            # Keep the historical `enabled` field as a compatibility alias
            # while exposing read persistence independently from execution.
            # A stopped/disabled worker must not hide durable history/export.
            enabled=subagent_batch_worker_running,
            repository_available=getattr(request.app.state, "subagent_batch_repo", None) is not None,
            worker_running=subagent_batch_worker_running,
            max_running=configured_subagent_max_running(),
        ),
    )


@router.get(
    "/features/inventory",
    response_model=FeatureInventoryResponse,
    summary="List Read-only Feature Inventory",
    description="Return a read-only inventory of visible agents, MCP servers, and skills.",
)
async def list_feature_inventory(config: AppConfig = Depends(get_config)) -> FeatureInventoryResponse:
    """Return the current user's configured runtime inventory without secrets."""
    user_id = get_effective_user_id()
    try:
        return await asyncio.to_thread(_load_inventory, config, user_id)
    except Exception as exc:
        logger.error("Failed to load feature inventory: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load feature inventory: {str(exc)}") from exc
