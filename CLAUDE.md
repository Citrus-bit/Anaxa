# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope and priority

- Treat `.github/copilot-instructions.md` as the primary operating guide for this repo.
- Use the root `Makefile` as the default entrypoint unless a command explicitly requires `backend/` or `frontend/`.

## Runtime requirements

- Python >= 3.12
- `uv`
- Node.js >= 22
- `pnpm` (lockfile is pnpm v10)
- `nginx` (required for unified local endpoint via `make dev`)

## Common commands

### Root (full stack)

```bash
make check          # verify prerequisites
make install        # install backend (uv) + frontend (pnpm)
make dev            # start Gateway+embedded agent runtime(8001) + Frontend(3000) + nginx(2026)
make stop           # stop local services
make config         # first-time config bootstrap only (aborts if config already exists)
make config-upgrade # merge new config fields from template
```

Docker flows (mode-aware from `config.yaml`):

```bash
make docker-init
make docker-start
make docker-stop
make docker-logs
```

### Backend (`backend/`)

```bash
make install        # uv sync
make dev            # FastAPI gateway + embedded DeerFlow agent runtime
make gateway        # FastAPI gateway on :8001
make lint           # ruff check
make format         # ruff fix + format
make test           # full backend test suite
```

Run a single backend test:

```bash
PYTHONPATH=. uv run pytest tests/test_<name>.py -v
# or
PYTHONPATH=. uv run pytest tests/test_<name>.py::test_<case> -v
```

### Frontend (`frontend/`)

```bash
pnpm dev
pnpm lint
pnpm typecheck
BETTER_AUTH_SECRET=local-dev-secret pnpm build
pnpm start
```

Notes:
- Prefer `pnpm lint` + `pnpm typecheck` over `pnpm check`.
- `pnpm build` is expected to need `BETTER_AUTH_SECRET` (or `SKIP_ENV_VALIDATION=1`, but secret is preferred).

## Pre-PR validation baseline

- Backend (required by CI):

```bash
cd backend && make lint && make test
```

- If frontend changed:

```bash
cd frontend && pnpm lint && pnpm typecheck
```

- If env/auth/routing/build-sensitive frontend files changed:

```bash
cd frontend && BETTER_AUTH_SECRET=local-dev-secret pnpm build
```

- If touching orchestration/config (`Makefile`, `docker/*`, `config*.yaml`), run `make dev` once and verify all three local services come up.

## High-level architecture

Anaxa uses the DeerFlow runtime with three local processes:

1. **Nginx** (`:2026`) — unified entrypoint.
2. **Gateway API** (`:8001`) — FastAPI endpoints and the embedded DeerFlow/LangGraph-compatible agent runtime.
3. **Frontend** (`:3000`) — Next.js app.

Routing through nginx:
- `/api/langgraph/*` -> Gateway's embedded LangGraph-compatible agent runtime
- `/api/*` (non-langgraph) -> Gateway API
- `/` -> Frontend

## Backend architecture (big picture)

Primary backend code lives in `backend/packages/harness/deerflow/`, with Anaxa compatibility APIs in `backend/packages/harness/medrix_flow/` and app-layer APIs in `backend/app/`.

- `agents/` — Lead agent and ordered middleware pipeline.
- `sandbox/` — thread-isolated tool execution (`bash`, file ops) with virtual paths.
- `subagents/` — delegated task execution (bounded concurrency/timeouts).
- `mcp/` — MCP client/tool integration.
- `tools/` + `community/` — built-in and external/community tools.
- `config/` — config loading and env resolution.
- `app/gateway/` — FastAPI gateway routers and app lifecycle.

Graph entrypoint is defined in `backend/langgraph.json`:
- `lead_agent` -> `deerflow.agents:make_lead_agent`

## Frontend architecture (big picture)

`frontend/src/`:

- `app/` — Next.js App Router routes.
- `components/` — UI and workspace components.
- `core/` — application logic (threads, API client, models, skills, settings, MCP integration).
- `hooks/`, `lib/`, `styles/` — shared hooks/utilities/styling.
- `env.js` — environment schema/validation (important for build behavior).

## Key repo-level gotchas

- Proxy environment variables can break frontend dependency/network operations.
- `make config` is intentionally non-idempotent when config already exists.
- `make dev` handles cleanup/startup and may emit expected shutdown noise when interrupted.
- Skills are loaded from `skills/public/` (and `skills/custom/` when present).

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

The code-review-graph MCP is disabled for local Codex Desktop sessions. Use
standard shell and file tools (`rg`, `find`, `sed`, and focused reads) for
repository exploration unless the MCP is explicitly re-enabled.
