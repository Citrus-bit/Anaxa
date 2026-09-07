# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedrixFlow is a LangGraph-based AI super agent system with a full-stack architecture. The backend provides a "super agent" with sandbox execution, persistent memory, subagent delegation, and extensible tool integration - all operating in per-thread isolated environments.

**Architecture**:
- **LangGraph Server** (port 6203): Agent runtime and workflow execution
- **Gateway API** (port 6202): REST API for models, MCP, skills, memory, artifacts, uploads, and local thread cleanup
- **Frontend** (port 6201): Next.js web interface
- **Nginx** (port 6200): Unified reverse proxy entry point
- **Provisioner** (port 6204, optional in Docker dev): Started only when sandbox is configured for provisioner/Kubernetes mode

**Project Structure**:
```
medrix-flow/
├── Makefile                    # Root commands (check, install, dev, stop)
├── config.yaml                 # Main application configuration
├── extensions_config.json      # MCP servers and skills configuration
├── backend/                    # Backend application (this directory)
│   ├── Makefile               # Backend-only commands (dev, gateway, lint)
│   ├── langgraph.json         # LangGraph server configuration
│   ├── packages/
│   │   └── harness/           # medrix-flow-harness package (import: medrix_flow.*)
│   │       ├── pyproject.toml
│   │       └── medrix_flow/
│   │           ├── agents/            # LangGraph agent system
│   │           │   ├── lead_agent/    # Main agent (factory + system prompt)
│   │           │   ├── middlewares/   # 10 middleware components
│   │           │   ├── memory/        # Memory extraction, queue, prompts
│   │           │   └── thread_state.py # ThreadState schema
│   │           ├── sandbox/           # Sandbox execution system
│   │           │   ├── local/         # Local filesystem provider
│   │           │   ├── sandbox.py     # Abstract Sandbox interface
│   │           │   ├── tools.py       # bash, ls, read/write/str_replace
│   │           │   └── middleware.py  # Sandbox lifecycle management
│   │           ├── subagents/         # Subagent delegation system
│   │           │   ├── builtins/      # general-purpose, bash agents
│   │           │   ├── executor.py    # Background execution engine
│   │           │   └── registry.py    # Agent registry
│   │           ├── tools/builtins/    # Built-in tools (present_files, ask_clarification, view_image)
│   │           ├── mcp/               # MCP integration (tools, cache, client)
│   │           ├── models/            # Model factory with thinking/vision support
│   │           ├── skills/            # Skills discovery, loading, parsing
│   │           ├── config/            # Configuration system (app, model, sandbox, tool, etc.)
│   │           ├── community/         # Community tools (tavily, jina_ai, firecrawl, image_search, aio_sandbox)
│   │           ├── reflection/        # Dynamic module loading (resolve_variable, resolve_class)
│   │           ├── utils/             # Utilities (network, readability)
│   │           └── client.py          # Embedded Python client (MedrixFlowClient)
│   ├── app/                   # Application layer (import: app.*)
│   │   ├── gateway/           # FastAPI Gateway API
│   │   │   ├── app.py         # FastAPI application
│   │   │   └── routers/       # FastAPI route modules (models, mcp, memory, skills, uploads, threads, artifacts, agents, suggestions, channels)
│   │   └── channels/          # IM platform integrations
│   ├── tests/                 # Test suite
│   └── docs/                  # Documentation
├── frontend/                   # Next.js frontend application
└── skills/                     # Agent skills directory
    ├── public/                # Public skills (committed)
    └── custom/                # Custom skills (gitignored)
```

## Important Development Guidelines

### Documentation Update Policy
**CRITICAL: Always update README.md and CLAUDE.md after every code change**

When making code changes, you MUST update the relevant documentation:
- Update `README.md` for user-facing changes (features, setup, usage instructions)
- Update `CLAUDE.md` for development changes (architecture, commands, workflows, internal systems)
- Keep documentation synchronized with the codebase at all times
- Ensure accuracy and timeliness of all documentation

## Commands

**Root directory** (for full application):
```bash
make check      # Check system requirements
make install    # Install all dependencies (frontend + backend)
make dev        # Start all services (LangGraph + Gateway + Frontend + Nginx), with config.yaml preflight
make stop       # Stop all services
```

**Backend directory** (for backend development only):
```bash
make install    # Install backend dependencies
make dev        # Run LangGraph server only (port 2024)
make gateway    # Run Gateway API only (port 8001)
make test       # Run all backend tests
make lint       # Lint with ruff
make format     # Format code with ruff
```

Regression tests related to Docker/provisioner behavior:
- `tests/test_docker_sandbox_mode_detection.py` (mode detection from `config.yaml`)
- `tests/test_provisioner_kubeconfig.py` (kubeconfig file/directory handling)

Boundary check (harness → app import firewall):
- `tests/test_harness_boundary.py` — ensures `packages/harness/medrix_flow/` never imports from `app.*`

CI runs these regression tests for every pull request via [.github/workflows/backend-unit-tests.yml](../.github/workflows/backend-unit-tests.yml).

## Architecture

### Harness / App Split

The backend is split into two layers with a strict dependency direction:

- **Harness** (`packages/harness/medrix_flow/`): Publishable agent framework package (`medrix-flow-harness`). Import prefix: `medrix_flow.*`. Contains agent orchestration, tools, sandbox, models, MCP, skills, config — everything needed to build and run agents.
- **App** (`app/`): Unpublished application code. Import prefix: `app.*`. Contains the FastAPI Gateway API and IM channel integrations (Feishu, Slack, Telegram).

**Dependency rule**: App imports medrix_flow, but medrix_flow never imports app. This boundary is enforced by `tests/test_harness_boundary.py` which runs in CI.

**Import conventions**:
```python
# Harness internal
from medrix_flow.agents import make_lead_agent
from medrix_flow.models import create_chat_model

# App internal
from app.gateway.app import app
from app.channels.service import start_channel_service

# App → Harness (allowed)
from medrix_flow.config import get_app_config

# Harness → App (FORBIDDEN — enforced by test_harness_boundary.py)
# from app.gateway.routers.uploads import ...  # ← will fail CI
```

### Agent System

**Lead Agent** (`packages/harness/medrix_flow/agents/lead_agent/agent.py`):
- Entry point: `make_lead_agent(config: RunnableConfig)` registered in `langgraph.json`
- Dynamic model selection via `create_chat_model()` with thinking/vision support
- Tools loaded via `get_available_tools()` - combines sandbox, built-in, MCP, community, and subagent tools
- System prompt generated by `apply_prompt_template()` with skills, memory, and subagent instructions

**ThreadState** (`packages/harness/medrix_flow/agents/thread_state.py`):
- Extends `AgentState` with: `sandbox`, `thread_data`, `title`, `artifacts`, `todos`, `uploaded_files`, `viewed_images`
- Uses custom reducers: `merge_artifacts` (deduplicate), `merge_viewed_images` (merge/clear)

**Runtime Configuration** (via `config.configurable`):
- `thinking_enabled` - Enable model's extended thinking
- `model_name` - Select specific LLM model
- `is_plan_mode` - Enable TodoList middleware
- `subagent_enabled` - Enable task delegation tool

### Middleware Chain

Middlewares execute in strict order in `packages/harness/medrix_flow/agents/lead_agent/agent.py`:

1. **ThreadDataMiddleware** - Creates per-thread directories (`backend/.medrix-flow/threads/{thread_id}/user-data/{workspace,uploads,outputs}`); Web UI thread deletion now follows LangGraph thread removal with Gateway cleanup of the local `.medrix-flow/threads/{thread_id}` directory
2. **UploadsMiddleware** - Tracks and injects newly uploaded files into conversation
3. **SandboxMiddleware** - Acquires sandbox, stores `sandbox_id` in state
4. **DanglingToolCallMiddleware** - Injects placeholder ToolMessages for AIMessage tool_calls that lack responses (e.g., due to user interruption)
5. **GuardrailMiddleware** - Pre-tool-call authorization via pluggable `GuardrailProvider` protocol (optional, if `guardrails.enabled` in config). Evaluates each tool call and returns error ToolMessage on deny. Three provider options: built-in `AllowlistProvider` (zero deps), OAP policy providers (e.g. `aport-agent-guardrails`), or custom providers. See [docs/GUARDRAILS.md](docs/GUARDRAILS.md) for setup, usage, and how to implement a provider.
6. **SummarizationMiddleware** - Context reduction when approaching token limits (optional, if enabled)
7. **TodoListMiddleware** - Task tracking with `write_todos` tool (optional, if plan_mode)
8. **TitleMiddleware** - Auto-generates thread title after first complete exchange and normalizes structured message content before prompting the title model
9. **MemoryMiddleware** - Queues conversations for async memory update (filters to user + final AI responses)
10. **ViewImageMiddleware** - Injects base64 image data before LLM call (conditional on vision support)
11. **SubagentLimitMiddleware** - Truncates excess `task` tool calls from model response to enforce `MAX_CONCURRENT_SUBAGENTS` limit (optional, if subagent_enabled)
12. **ClarificationMiddleware** - Intercepts `ask_clarification` tool calls, interrupts via `Command(goto=END)` (must be last)

### Configuration System

**Main Configuration** (`config.yaml`):

Setup: Copy `config.example.yaml` to `config.yaml` in the **project root** directory.

**Config Versioning**: `config.example.yaml` has a `config_version` field. On startup, `AppConfig.from_file()` compares user version vs example version and emits a warning if outdated. Missing `config_version` = version 0. Run `make config-upgrade` to auto-merge missing fields. When changing the config schema, bump `config_version` in `config.example.yaml`.

**Config Caching**: `get_app_config()` caches the parsed config, but automatically reloads it when the resolved config path changes or the file's mtime increases. This keeps Gateway and LangGraph reads aligned with `config.yaml` edits without requiring a manual process restart.

Configuration priority:
1. Explicit `config_path` argument
2. `MEDRIX_FLOW_CONFIG_PATH` environment variable
3. `config.yaml` in current directory (backend/)
4. `config.yaml` in parent directory (project root - **recommended location**)

Config values starting with `$` are resolved as environment variables (e.g., `$OPENAI_API_KEY`).
`ModelConfig` also declares `use_responses_api` and `output_version` so OpenAI `/v1/responses` can be enabled explicitly while still using `langchain_openai:ChatOpenAI`.

**Research Configuration** (`config.yaml` → `research`):
- `manuscript_model`: optional model name for manuscript section generation in `research_assistant action="run_pipeline"`; `null` inherits the current thread model.
- `fast_draft_model`: optional model for the first-pass editable manuscript draft; `null` falls back to `manuscript_model` or the thread model.
- `finalization_model`: optional model for background review/final-bundle continuation; `null` falls back to `manuscript_model`.
- `manuscript_section_concurrency`: max concurrent manuscript section generation calls; defaults to `3`.
- `default_auto_gates`: gate types that `run_pipeline` may auto-approve by default. Keep this empty unless the product deliberately allows unattended gates.
- `default_max_stages`: maximum lifecycle stages advanced by one `run_pipeline` tool call; defaults to `5` to avoid long blocking tool calls.
- `default_quality_mode`: final-bundle quality gate mode for `run_pipeline`; `auto_repair` is the default, with `audit_only` and `strict_gate` available for less/more enforcement.
- `default_quality_repair_budget`: maximum automatic quality-repair approvals before returning to a human gate; defaults to `2`.
- `research_assistant action="run_pipeline"` accepts `delivery_mode`. `fast_draft_first` is the default tool behavior and pauses after `manuscript_draft`, writes a fast draft artifact, records `draft_ready_at` / `finalization_run_id`, and starts a tracked background finalization run. `final_only` keeps the previous one-shot final-bundle behavior.
- Academic reference formatting is user-selected. `academic_research`, `/api/academic/projects/{project_id}/synthesize`, and `/api/academic/projects/{project_id}/references` accept `reference_style`/`style` values such as `apa7`, `mla9`, `chicago`, `gbt7714`, `plain`, and `bibtex`; APA 7 is only the compatibility default when no style is specified.

**Extensions Configuration** (`extensions_config.json`):

MCP servers and skills are configured together in `extensions_config.json` in project root:

Configuration priority:
1. Explicit `config_path` argument
2. `MEDRIX_FLOW_EXTENSIONS_CONFIG_PATH` environment variable
3. `extensions_config.json` in current directory (backend/)
4. `extensions_config.json` in parent directory (project root - **recommended location**)

### Gateway API (`app/gateway/`)

FastAPI application on port 8001 with health check at `GET /health`.

**Routers**:

| Router | Endpoints |
|--------|-----------|
| **Models** (`/api/models`) | `GET /` - list models; `GET /{name}` - model details |
| **MCP** (`/api/mcp`) | `GET /config` - get config; `PUT /config` - update config (saves to extensions_config.json) |
| **Skills** (`/api/skills`) | `GET /` - list skills; `GET /{name}` - details; `PUT /{name}` - update enabled; `POST /install` - install from .skill archive (accepts standard optional frontmatter like `version`, `author`, `compatibility`) |
| **Memory** (`/api/memory`) | `GET /` - memory data; `POST /reload` - force reload; `GET /config` - config; `GET /status` - config + data |
| **Uploads** (`/api/threads/{id}/uploads`) | `POST /` - upload files (auto-converts PDF/PPT/Excel/Word); non-mounted sandbox syncs always release the acquired sandbox after success or failure; `GET /list` - list; `DELETE /{filename}` - delete |
| **Threads** (`/api/threads/{id}`) | `DELETE /` - remove MedrixFlow-managed local thread data after LangGraph thread deletion; unexpected failures are logged server-side and return a generic 500 detail |
| **Artifacts** (`/api/threads/{id}/artifacts`) | `GET /{path}` - serve artifacts; `?download=true` for file download |
| **Suggestions** (`/api/threads/{id}/suggestions`) | `POST /` - generate follow-up questions; rich list/block model content is normalized before JSON parsing |

Proxied through nginx: `/api/langgraph/*` → LangGraph, all other `/api/*` → Gateway.

### Sandbox System (`packages/harness/medrix_flow/sandbox/`)

**Interface**: Abstract `Sandbox` with `execute_command`, `read_file`, `write_file`, `list_dir`
**Provider Pattern**: `SandboxProvider` with `acquire`, `get`, `release` lifecycle
**Implementations**:
- `LocalSandboxProvider` - Singleton local filesystem execution with path mappings
- `AioSandboxProvider` (`packages/harness/medrix_flow/community/`) - Docker-based isolation

**Virtual Path System**:
- Agent sees: `/mnt/user-data/{workspace,uploads,outputs}`, `/mnt/skills`
- Physical: `backend/.medrix-flow/threads/{thread_id}/user-data/...`, `medrix-flow/skills/`
- Translation: `replace_virtual_path()` / `replace_virtual_paths_in_command()`
- Detection: `is_local_sandbox()` checks `sandbox_id == "local"`

**Sandbox Tools** (in `packages/harness/medrix_flow/sandbox/tools.py`):
- `bash` - Execute commands with path translation and error handling
- `ls` - Directory listing (tree format, max 2 levels)
- `read_file` - Read file contents with optional line range
- `write_file` - Write/append to files, creates directories
- `str_replace` - Substring replacement (single or all occurrences)

### Subagent System (`packages/harness/medrix_flow/subagents/`)

**Built-in Agents**: `general-purpose` (all tools except `task`) and `bash` (command specialist)
**Execution**: Dual thread pool - `_scheduler_pool` (3 workers) + `_execution_pool` (3 workers)
**Concurrency**: `MAX_CONCURRENT_SUBAGENTS = 3` enforced by `SubagentLimitMiddleware` (truncates excess tool calls in `after_model`), 15-minute timeout
**Flow**: `task()` tool → `SubagentExecutor` → background thread → poll 5s → SSE events → result
**Events**: `task_started`, `task_running`, `task_completed`/`task_failed`/`task_timed_out`

### Tool System (`packages/harness/medrix_flow/tools/`)

`get_available_tools(groups, include_mcp, model_name, subagent_enabled)` assembles:
1. **Config-defined tools** - Resolved from `config.yaml` via `resolve_variable()`
2. **MCP tools** - From enabled MCP servers (lazy initialized, cached with mtime invalidation)
3. **Built-in tools**:
   - `present_files` - Make output files visible to user (only `/mnt/user-data/outputs`)
   - `ask_clarification` - Request clarification (intercepted by ClarificationMiddleware → interrupts)
   - `view_image` - Read image as base64 (added only if model supports vision)
4. **Subagent tool** (if enabled):
   - `task` - Delegate to subagent (description, prompt, subagent_type, max_turns)

**Community tools** (`packages/harness/medrix_flow/community/`):
- `tavily/` - Web search (5 results default) and web fetch (4KB limit)
- `jina_ai/` - Web fetch via Jina reader API with readability extraction
- `firecrawl/` - Web scraping via Firecrawl API
- `image_search/` - Image search via DuckDuckGo

### Research Pipeline (`packages/harness/medrix_flow/research/`)

The research layer is a harness-only lifecycle system. It must not import `app.*`.

- `types.py` defines the lifecycle stages, gates, `PipelineStageEvent`, and `PipelineRunResult`.
- `service.py` owns deterministic persistence and stage handlers. `advance_quest()` remains the single-stage primitive and accepts optional generator callbacks for stages that need LLM content.
- `quality.py` owns deterministic `ResearchQualityAudit` checks for literature coverage, inline citation density, off-topic evidence, quantitative/benchmark evidence, feasibility discussion, repeated/absolute wording, and author/tool process-note residue.
- `orchestrator.py` owns longer coordination. `run_pipeline()` loops over `advance_quest()` until `final_bundle`, a non-auto human gate, cancellation, error, or `max_stages`.
- `tools/builtins/research_assistant_tool.py` exposes `action="run_pipeline"` for end-to-end autonomous research requests. The tool injects manuscript content generation via `create_chat_model()` and keeps service/orchestrator free of model factory dependencies.
- Human gate defaults are conservative: `experiment_execution`, `pre_review`, and `final_release` stop the pipeline unless explicitly included in `auto_gates` or configured in `research.default_auto_gates`. `final_quality_repair` is auto-approved only by `run_pipeline(quality_mode="auto_repair")` and only up to the configured repair budget.

### Academic Research Quality Coverage

`academic_research` remains the literature retrieval/report primitive, but review/manuscript deliverables now activate a review-quality coverage profile:
- Tool/API optional parameters: `deliverable_type`, `min_reference_count`, `target_reference_count`, `required_topics`, and `required_evidence_types`.
- Review/manuscript/survey/paper deliverables default to at least 50 usable references, target 80 references, and at least 30 core papers for synthesis.
- The ingest step stores `reference_coverage_audit` in project metadata, including coverage status, off-topic reference count, quantitative evidence count, missing required topics/evidence types, and recommended expansion queries.
- Keep this logic generic; do not hard-code one domain's model or benchmark list as truth. User-provided coverage hints are allowed.

Regression coverage:
- `tests/test_research_pipeline.py` covers pipeline result serialization, auto gates, max-stage stops, blocked gates, and cancelled quests.
- `tests/test_research_service.py` covers manuscript content generation fallback and deterministic result synthesis.
- `tests/test_research_assistant_tool.py` covers `research_assistant action="run_pipeline"` dispatch and config defaults.
- `tests/test_research_quality.py` covers deterministic quality audit findings.
- `tests/test_academic_service.py` covers review-quality reference coverage targets and missing coverage hints.

### MCP System (`packages/harness/medrix_flow/mcp/`)

- Uses `langchain-mcp-adapters` `MultiServerMCPClient` for multi-server management
- **Lazy initialization**: Tools loaded on first use via `get_cached_mcp_tools()`
- **Cache invalidation**: Detects config file changes via mtime comparison
- **Transports**: stdio (command-based), SSE, HTTP
- **OAuth (HTTP/SSE)**: Supports token endpoint flows (`client_credentials`, `refresh_token`) with automatic token refresh + Authorization header injection
- **Runtime updates**: Gateway API saves to extensions_config.json; LangGraph detects via mtime

### Skills System (`packages/harness/medrix_flow/skills/`)

- **Location**: `medrix-flow/skills/{public,custom}/`
- **Format**: Directory with `SKILL.md` (YAML frontmatter: name, description, license, allowed-tools)
- **Loading**: `load_skills()` recursively scans `skills/{public,custom}` for `SKILL.md`, parses metadata, and reads enabled state from extensions_config.json
- **Injection**: Enabled skills listed in agent system prompt with container paths
- **Installation**: `POST /api/skills/install` extracts .skill ZIP archive to custom/ directory

### Model Factory (`packages/harness/medrix_flow/models/factory.py`)

- `create_chat_model(name, thinking_enabled)` instantiates LLM from config via reflection
- Supports `thinking_enabled` flag with per-model `when_thinking_enabled` overrides
- Supports `supports_vision` flag for image understanding models
- Config values starting with `$` resolved as environment variables
- Missing provider modules surface actionable install hints from reflection resolvers (for example `uv add langchain-google-genai`)

### IM Channels System (`app/channels/`)

Bridges external messaging platforms (Feishu, Slack, Telegram) to the MedrixFlow agent via the LangGraph Server.

**Architecture**: Channels communicate with the LangGraph Server through `langgraph-sdk` HTTP client (same as the frontend), ensuring threads are created and managed server-side.

**Components**:
- `message_bus.py` - Async pub/sub hub (`InboundMessage` → queue → dispatcher; `OutboundMessage` → callbacks → channels)
- `store.py` - JSON-file persistence mapping `channel_name:chat_id[:topic_id]` → `thread_id` (keys are `channel:chat` for root conversations and `channel:chat:topic` for threaded conversations)
- `manager.py` - Core dispatcher: creates threads via `client.threads.create()`, routes commands, keeps Slack/Telegram on `client.runs.wait()`, and uses `client.runs.stream(["messages-tuple", "values"])` for Feishu incremental outbound updates
- `base.py` - Abstract `Channel` base class (start/stop/send lifecycle)
- `service.py` - Manages lifecycle of all configured channels from `config.yaml`
- `slack.py` / `feishu.py` / `telegram.py` - Platform-specific implementations (`feishu.py` tracks the running card `message_id` in memory and patches the same card in place)

**Message Flow**:
1. External platform -> Channel impl -> `MessageBus.publish_inbound()`
2. `ChannelManager._dispatch_loop()` consumes from queue
3. For chat: look up/create thread on LangGraph Server
4. Feishu chat: `runs.stream()` → accumulate AI text → publish multiple outbound updates (`is_final=False`) → publish final outbound (`is_final=True`)
5. Slack/Telegram chat: `runs.wait()` → extract final response → publish outbound
6. Feishu channel sends one running reply card up front, then patches the same card for each outbound update (card JSON sets `config.update_multi=true` for Feishu's patch API requirement)
7. For commands (`/new`, `/status`, `/models`, `/memory`, `/help`): handle locally or query Gateway API
8. Outbound → channel callbacks → platform reply

**Configuration** (`config.yaml` -> `channels`):
- `langgraph_url` - LangGraph Server URL (default: `http://localhost:6203`)
- `gateway_url` - Gateway API URL for auxiliary commands (default: `http://localhost:6202`)
- Per-channel configs: `feishu` (app_id, app_secret), `slack` (bot_token, app_token), `telegram` (bot_token)

### Memory System (`packages/harness/medrix_flow/agents/memory/`)

**Components**:
- `updater.py` - LLM-based memory updates with fact extraction, whitespace-normalized fact deduplication (trims leading/trailing whitespace before comparing), and atomic file I/O
- `queue.py` - Debounced update queue (per-thread deduplication, configurable wait time)
- `prompt.py` - Prompt templates for memory updates

**Data Structure** (stored in `backend/.medrix-flow/memory.json`):
- **User Context**: `workContext`, `personalContext`, `topOfMind` (1-3 sentence summaries)
- **History**: `recentMonths`, `earlierContext`, `longTermBackground`
- **Facts**: Discrete facts with `id`, `content`, `category` (preference/knowledge/context/behavior/goal), `confidence` (0-1), `createdAt`, `source`

**Workflow**:
1. `MemoryMiddleware` filters messages (user inputs + final AI responses) and queues conversation
2. Queue debounces (30s default), batches updates, deduplicates per-thread
3. Background thread invokes LLM to extract context updates and facts
4. Applies updates atomically (temp file + rename) with cache invalidation, skipping duplicate fact content before append
5. Next interaction injects top 15 facts + context into `<memory>` tags in system prompt

Focused regression coverage for the updater lives in `backend/tests/test_memory_updater.py`.

**Configuration** (`config.yaml` → `memory`):
- `enabled` / `injection_enabled` - Master switches
- `storage_path` - Path to memory.json
- `debounce_seconds` - Wait time before processing (default: 30)
- `model_name` - LLM for updates (null = default model)
- `max_facts` / `fact_confidence_threshold` - Fact storage limits (100 / 0.7)
- `max_injection_tokens` - Token limit for prompt injection (2000)

### Reflection System (`packages/harness/medrix_flow/reflection/`)

- `resolve_variable(path)` - Import module and return variable (e.g., `module.path:variable_name`)
- `resolve_class(path, base_class)` - Import and validate class against base class

### Config Schema

**`config.yaml`** key sections:
- `models[]` - LLM configs with `use` class path, `supports_thinking`, `supports_vision`, provider-specific fields
- `tools[]` - Tool configs with `use` variable path and `group`
- `tool_groups[]` - Logical groupings for tools
- `sandbox.use` - Sandbox provider class path
- `skills.path` / `skills.container_path` - Host and container paths to skills directory
- `title` - Auto-title generation (enabled, max_words, max_chars, prompt_template)
- `summarization` - Context summarization (enabled, trigger conditions, keep policy)
- `subagents.enabled` - Master switch for subagent delegation
- `memory` - Memory system (enabled, storage_path, debounce_seconds, model_name, max_facts, fact_confidence_threshold, injection_enabled, max_injection_tokens)

**`extensions_config.json`**:
- `mcpServers` - Map of server name → config (enabled, type, command, args, env, url, headers, oauth, description)
- `skills` - Map of skill name → state (enabled)

Both can be modified at runtime via Gateway API endpoints or `MedrixFlowClient` methods.

### Embedded Client (`packages/harness/medrix_flow/client.py`)

`MedrixFlowClient` provides direct in-process access to all MedrixFlow capabilities without HTTP services. All return types align with the Gateway API response schemas, so consumer code works identically in HTTP and embedded modes.

**Architecture**: Imports the same `medrix_flow` modules that LangGraph Server and Gateway API use. Shares the same config files and data directories. No FastAPI dependency.

**Agent Conversation** (replaces LangGraph Server):
- `chat(message, thread_id)` — synchronous, returns final text
- `stream(message, thread_id)` — yields `StreamEvent` aligned with LangGraph SSE protocol:
  - `"values"` — full state snapshot (title, messages, artifacts)
  - `"messages-tuple"` — per-message update (AI text, tool calls, tool results)
  - `"end"` — stream finished
- Agent created lazily via `create_agent()` + `_build_middlewares()`, same as `make_lead_agent`
- Supports `checkpointer` parameter for state persistence across turns
- `reset_agent()` forces agent recreation (e.g. after memory or skill changes)

**Gateway Equivalent Methods** (replaces Gateway API):

| Category | Methods | Return format |
|----------|---------|---------------|
| Models | `list_models()`, `get_model(name)` | `{"models": [...]}`, `{name, display_name, ...}` |
| MCP | `get_mcp_config()`, `update_mcp_config(servers)` | `{"mcp_servers": {...}}` |
| Skills | `list_skills()`, `get_skill(name)`, `update_skill(name, enabled)`, `install_skill(path)` | `{"skills": [...]}` |
| Memory | `get_memory()`, `reload_memory()`, `get_memory_config()`, `get_memory_status()` | dict |
| Uploads | `upload_files(thread_id, files)`, `list_uploads(thread_id)`, `delete_upload(thread_id, filename)` | `{"success": true, "files": [...]}`, `{"files": [...], "count": N}` |
| Artifacts | `get_artifact(thread_id, path)` → `(bytes, mime_type)` | tuple |

**Key difference from Gateway**: Upload accepts local `Path` objects instead of HTTP `UploadFile`, rejects directory paths before copying, and reuses a single worker when document conversion must run inside an active event loop. Artifact returns `(bytes, mime_type)` instead of HTTP Response. The new Gateway-only thread cleanup route deletes `.medrix-flow/threads/{thread_id}` after LangGraph thread deletion; there is no matching `MedrixFlowClient` method yet. `update_mcp_config()` and `update_skill()` automatically invalidate the cached agent.

**Tests**: `tests/test_client.py` (77 unit tests including `TestGatewayConformance`), `tests/test_client_live.py` (live integration tests, requires config.yaml)

**Gateway Conformance Tests** (`TestGatewayConformance`): Validate that every dict-returning client method conforms to the corresponding Gateway Pydantic response model. Each test parses the client output through the Gateway model — if Gateway adds a required field that the client doesn't provide, Pydantic raises `ValidationError` and CI catches the drift. Covers: `ModelsListResponse`, `ModelResponse`, `SkillsListResponse`, `SkillResponse`, `SkillInstallResponse`, `McpConfigResponse`, `UploadResponse`, `MemoryConfigResponse`, `MemoryStatusResponse`.

## Development Workflow

### Test-Driven Development (TDD) — MANDATORY

**Every new feature or bug fix MUST be accompanied by unit tests. No exceptions.**

- Write tests in `backend/tests/` following the existing naming convention `test_<feature>.py`
- Run the full suite before and after your change: `make test`
- Tests must pass before a feature is considered complete
- For lightweight config/utility modules, prefer pure unit tests with no external dependencies
- If a module causes circular import issues in tests, add a `sys.modules` mock in `tests/conftest.py` (see existing example for `medrix_flow.subagents.executor`)

```bash
# Run all tests
make test

# Run a specific test file
PYTHONPATH=. uv run pytest tests/test_<feature>.py -v
```

### Running the Full Application

From the **project root** directory:
```bash
make dev
```

This starts all services and makes the application available at `http://localhost:6200`.

**Nginx routing**:
- `/api/langgraph/*` → LangGraph Server (6203)
- `/api/*` (other) → Gateway API (6202)
- `/` (non-API) → Frontend (6201)

### Running Backend Services Separately

From the **backend** directory:

```bash
# Terminal 1: LangGraph server
make dev

# Terminal 2: Gateway API
make gateway
```

Direct access (without nginx):
- LangGraph: `http://localhost:6203`
- Gateway: `http://localhost:6202`

### Frontend Configuration

The frontend uses environment variables to connect to backend services:
- `NEXT_PUBLIC_LANGGRAPH_BASE_URL` - Defaults to `/api/langgraph` (through nginx)
- `NEXT_PUBLIC_BACKEND_BASE_URL` - Defaults to empty string (through nginx)

When using `make dev` from root, the frontend automatically connects through nginx.

## Key Features

### File Upload

Multi-file upload with automatic document conversion:
- Endpoint: `POST /api/threads/{thread_id}/uploads`
- Supports: PDF, PPT, Excel, Word documents (converted via `markitdown`)
- Rejects directory inputs before copying so uploads stay all-or-nothing
- Reuses one conversion worker per request when called from an active event loop
- Files stored in thread-isolated directories
- Agent receives uploaded file list via `UploadsMiddleware`

See [docs/FILE_UPLOAD.md](docs/FILE_UPLOAD.md) for details.

### Plan Mode

TodoList middleware for complex multi-step tasks:
- Controlled via runtime config: `config.configurable.is_plan_mode = True`
- Provides `write_todos` tool for task tracking
- One task in_progress at a time, real-time updates

See [docs/plan_mode_usage.md](docs/plan_mode_usage.md) for details.

### Context Summarization

Automatic conversation summarization when approaching token limits:
- Configured in `config.yaml` under `summarization` key
- Trigger types: tokens, messages, or fraction of max input
- Keeps recent messages while summarizing older ones

See [docs/summarization.md](docs/summarization.md) for details.

### Vision Support

For models with `supports_vision: true`:
- `ViewImageMiddleware` processes images in conversation
- `view_image_tool` added to agent's toolset
- Images automatically converted to base64 and injected into state

## Code Style

- Uses `ruff` for linting and formatting
- Line length: 240 characters
- Python 3.12+ with type hints
- Double quotes, space indentation

## Documentation

See `docs/` directory for detailed documentation:
- [CONFIGURATION.md](docs/CONFIGURATION.md) - Configuration options
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture details
- [API.md](docs/API.md) - API reference
- [SETUP.md](docs/SETUP.md) - Setup guide
- [FILE_UPLOAD.md](docs/FILE_UPLOAD.md) - File upload feature
- [PATH_EXAMPLES.md](docs/PATH_EXAMPLES.md) - Path types and usage
- [summarization.md](docs/summarization.md) - Context summarization
- [plan_mode_usage.md](docs/plan_mode_usage.md) - Plan mode with TodoList
