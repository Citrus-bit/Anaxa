# Anaxa 1.0

**English** | [中文](./README.md)

<p align="center">
  <img src="./Anaxa%20logo.jpg" alt="Anaxa" width="180">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-1.0-111827?style=for-the-badge" alt="Version 1.0">
  <img src="https://img.shields.io/badge/Orchestration-LangGraph-blue?style=for-the-badge&logo=python" alt="LangGraph">
  <img src="https://img.shields.io/badge/Frontend-Next.js%2016-black?style=for-the-badge&logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react" alt="React 19">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python" alt="Python 3.12">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>An auditable, interruptible, human-steerable workspace for research agents</b><br/>
  Literature Retrieval · Evidence Maps · Experiment Loops · LaTeX/PDF Manuscripts · Human Gates
</p>

---

## What Is Anaxa

Anaxa is an open-source agent system for scientific research workflows. It is not a generic chat UI, and it is not an unsupervised paper generator. It connects literature retrieval, evidence auditing, experiment execution, manuscript drafting, reviewer-style checks, and final artifact packaging into one traceable research lifecycle.

The goal is to move research assistance from "a model writes a long answer in chat" to "each step has sources, artifacts, status, failure records, and human handoff points." Chat remains the main entry point, but research-heavy requests are routed to backend tools for academic search, experiments, manuscript export, and staged orchestration.

Anaxa is built for:

- Literature reviews, related work, research background, and reference curation.
- Manuscript drafts, experiment reports, and research memos with claim-level evidence.
- Reproducible analysis for CS/AI, data science, bioinformatics, and empirical research.
- One-pass LaTeX paper bundles with BibTeX, citation audit, and PDF output.
- Long-running research projects that require stage tracking and human approval.

## Core Research Workflow

Anaxa organizes complex research work around a `ResearchQuest`. A quest moves through explicit, recoverable stages:

```text
intake
  -> literature
  -> novelty_check
  -> evidence_verified
  -> experiment_planned
  -> experiment_running
  -> results_synthesized
  -> manuscript_draft
  -> review
  -> revision
  -> final_bundle
```

Each stage records inputs, outputs, tool calls, artifacts, failure reasons, and gate decisions in a research ledger. Riskier actions, including external code execution, long experiments, automatic experiment-code edits, and final manuscript release, require human approval by default.

## Key Capabilities

### Academic Retrieval and References

- `academic_research` handles multi-source paper retrieval, metadata normalization, deduplication, evidence-card generation, reference export, and coverage audits.
- Supported source families include Semantic Scholar, OpenAlex, Crossref, arXiv, DBLP, OpenReview, and ACL Anthology.
- References can be exported in the user's requested style, including APA 7, MLA 9, Chicago, GB/T 7714, plain text, and BibTeX.
- Review, survey, and manuscript tasks use a stricter quality profile with higher reference targets, citation-density checks, off-topic reference filtering, and automatic search repair.

### Claim-Level Evidence Map

Anaxa does not treat a bibliography at the end of a paper as sufficient evidence. Important claims should be bound to concrete support:

```text
claim -> paper_id -> citation_key -> snippet/page/abstract evidence -> support_status -> confidence
```

When full-text PDFs are unavailable, the system can fall back to title, abstract, and metadata evidence, but the audit marks that evidence as weaker. Missing citation keys, overuse of `\nocite{*}`, paragraphs without citations, unsupported claims, and leftover process notes are surfaced in `citation_audit.json` and can block the final PDF release.

### Experiment Loop

- `experiment_lab` provides Python-first experiment execution, metric tracking, figure generation, and result-bundle export.
- It supports baselines, ablations, seeds, metrics, failure summaries, branch ranking, and a reproducibility ledger.
- CS/AI workflows cover regression, classification, clustering, dimensionality reduction, diagnostics, model evaluation, and paper-ready result summaries.
- Bioinformatics workflows cover bulk expression, differential analysis, enrichment analysis, single-cell starter workflows, and common scientific figures.
- Empirical-research skills can translate DID, IV, RDD, PSM/IPW, DML, target-trial, and TMLE requirements into experiment metadata and gates.

### Manuscript and PDF Export

Manuscript, review, survey, and experiment-paper requests default to a LaTeX bundle:

- `manuscript.tex`
- `references.bib`
- `citation_audit.json`
- `manuscript.pdf`

The high-level `manuscript_export` tool handles file writing, citation audit, LaTeX compilation, and artifact presentation as one operation. Lower-level tools such as `write_file`, `citation_audit`, and `present_files` remain available, but final manuscript delivery should use the high-level export path instead of leaving file generation to one-off chat responses.

The PDF path currently prefers `tectonic`. If the compiler is missing or LaTeX compilation fails, the system returns the exact tool error and keeps the `.tex`, `.bib`, and audit files for repair.

### Quality Audit and Auto-Repair

`ResearchQualityAudit` checks common pre-submission risks:

- Insufficient literature coverage.
- Low in-text citation density.
- Off-topic or cross-domain references.
- Missing quantitative evidence, benchmark evidence, case studies, or experimental results.
- Evaluation frameworks without implementation challenges and mitigations.
- Repeated phrasing, over-absolute wording, weak transitions, and process-note residue.

The default policy is to repair first: expand search, add evidence, look for benchmarks, and rewrite weak sections. If the repair budget is exhausted, the workflow stops at a human gate and returns a concrete remediation report.

### Thread Artifacts and Thread-Scoped Memory

- Each chat/thread has isolated `workspace`, `uploads`, `outputs`, and `memory.json` storage.
- A new chat starts with empty private memory and does not inherit a global user profile or another thread's research state.
- Reports, BibTeX files, PDFs, figures, tables, and audit files are written to thread-scoped outputs and surfaced in the artifact panel.
- Legacy memory APIs remain for compatibility, but the frontend no longer exposes a shared memory settings page.

### Read-Only Feature Inventory

The frontend settings panel includes a read-only "Features" page that reflects backend configuration:

- Agents: default orchestrator, system agents, custom agents, and subagents.
- Tools: MCP servers, transport, enabled state, command or URL summaries, and redacted env/header keys.
- Skills: public and custom skills with name, description, category, enabled state, and license.

This page is an inventory, not an editor. It does not provide create, delete, edit, enable, disable, or connection-test controls. Existing backend management APIs remain available for scripts and compatibility.

## Architecture

```text
                  http://localhost:2026
                          |
                          v
                  +----------------+
                  |     Nginx      |
                  | reverse proxy  |
                  +---+--------+---+
                      |        |
      /api/langgraph/*|        |/api/*
                      v        v
        +----------------------------------------+
        | Gateway API + embedded DeerFlow runtime|
        | FastAPI :8001                          |
        | models/setup, agent runs, research     |
        | academic, experiments, artifacts       |
        +----------------+-----------+-----------+
                         |           |
                         v           v
        +----------------+   +------------------+
        | Sandbox / VFS  |   | DeerFlow state   |
        | thread-scoped  |   | .deer-flow       |
        +----------------+   +------------------+

        +----------------------------------------+
        | Frontend :3000                         |
        | Next.js 16 / React 19 / Tailwind CSS 4 |
        +----------------------------------------+
```

Request routing:

- `/api/langgraph/*` -> Gateway's embedded LangGraph-compatible runtime for agent interaction, threads, SSE streaming, and long-running runs.
- `/api/*` -> Gateway API for setup, features, academic research, research quests, experiments, uploads, artifacts, and runs.
- `/` -> Frontend, served by Next.js.

Primary DeerFlow runtime data is stored under `backend/.deer-flow` or the directory pointed to by `DEER_FLOW_HOME`. Anaxa's legacy research sidecar remains isolated under `.medrix-flow` (or `ANAXA_RESEARCH_HOME`). Each thread maps a virtual filesystem to its thread-scoped workspace, uploads, and outputs.

## Quick Start

Anaxa 1.0 is released as an open-source development build by default: download the source, initialize local files, start the app locally, then configure models in the frontend. UI password protection is disabled by default; this mode is intended for localhost or trusted LAN use, not public internet deployment.

### 1. Install Base Tools

Install these system tools first:

- Python 3.12+
- Node.js 22+
- pnpm
- uv
- nginx

If you are not sure whether they are installed, run:

```bash
make check
```

The checker prints install hints for anything missing. `tectonic` is optional and only affects LaTeX PDF generation reliability.

### 2. Download and Bootstrap

```bash
git clone <repo-url>
cd <repo-folder>
make bootstrap
```

`make bootstrap` does three things:

- Checks local dependencies.
- Creates local config files: `config.yaml`, `.env`, `frontend/.env`, and `extensions_config.json`.
- Installs backend and frontend dependencies.

Generated config files are ignored by Git. You do not need to edit API keys by hand; configure them in the frontend after startup.

### 3. Start

```bash
make dev
```

Open:

```text
http://localhost:2026
```

On first launch, open "Settings & More -> Setup" in the lower-left corner, add at least one chat model and API key, then save.

The development command starts:

- Frontend: `http://localhost:3000`
- Gateway API + embedded agent runtime: `http://localhost:8001`
- Unified Nginx entry: `http://localhost:2026`

### Optional Docker Path

If you do not want to install Python/Node/nginx locally, use Docker development mode:

```bash
make docker-init
make docker-start
```

Open the same URL:

```text
http://localhost:2026
```

Stop Docker development mode with:

```bash
make docker-stop
```

### Common Commands

| Command | Description |
|---|---|
| `make bootstrap` | First-time setup: check tools, create local config, install dependencies |
| `make check` | Check Node.js, pnpm, uv, and nginx |
| `make install` | Install backend and frontend dependencies |
| `make dev` | Start hot-reload development services |
| `make dev-daemon` | Start development services in the background |
| `make stop` | Stop local services |
| `make docker-start` | Start Docker development mode |
| `make docker-stop` | Stop Docker development mode |
| `make verify` | Run backend lint/test and frontend lint/typecheck/unit/build checks |
| `make release-check` | Check that local secrets, caches, memory, and runtime data are not tracked by Git |

## Configuration

### Normal Use: Configure in the Frontend

After startup, visit `http://localhost:2026` and open "Settings & More -> Setup":

- Add the model provider, model name, and API key.
- Configure web search, web fetch, academic retrieval, and other tool API keys.
- Configure Google AI Studio or an OpenAI-compatible image endpoint if you need image generation.
- Saving writes local config files and hot-reloads the app.

The "Features" page is read-only and shows available Agents, MCP tools, and Skills. It does not provide create, delete, edit, enable, or disable controls.

### Advanced Use: Config Files

`config.yaml` and `extensions_config.json` remain available for scripts and deeper customization:

- `config.yaml`: models, tool groups, sandbox provider, checkpointer, memory, research gates, and quality policies.
- `extensions_config.json`: MCP server and skill enablement state.
- `make config-upgrade`: merge new fields into an existing `config.yaml`.

### Release Safety Check

Your local `.env`, databases, memory, contexts, uploads, outputs, caches, and logs should not be published. Before pushing to a public repository, run:

```bash
make release-check
```

If the check fails, fix the listed paths before publishing. The command does not delete local files.

### Public Deployment Passwords

The open-source development build does not set a UI password by default. If you expose the service to the public internet, set `MEDRIX_FLOW_ENV=production`, `MEDRIX_FLOW_UI_PASSWORD`, and optionally `MEDRIX_GATEWAY_ADMIN_TOKEN`.

### Compatibility Identifiers

Anaxa 1.0 is the user-facing product name. Some technical identifiers remain unchanged to avoid breaking existing scripts, runtime directories, environment variables, and imports:

- Python import package: `medrix_flow`
- Compatibility names in Python distribution and workspace packages
- Environment variables: `MEDRIX_FLOW_*`
- Primary runtime directory: `.deer-flow`; legacy research sidecar directory: `.medrix-flow`
- Admin header: `x-medrix-admin-token`
- Some Docker container, Compose project, and cleanup prefixes

These are compatibility identifiers, not the new product brand.

## Built-In Tools and Skills

Built-in tools:

- `academic_research`: paper retrieval, evidence cards, references, and audits.
- `research_assistant`: research lifecycle, ledger, gates, quality audits, and `run_pipeline`.
- `experiment_lab`: experiment execution, figures, metrics, and reproducibility bundles.
- `manuscript_export`: one-pass LaTeX/BibTeX/audit/PDF export.
- `citation_audit`: citation key, in-text citation, and claim-support checks for LaTeX.
- `present_files`: artifact presentation and `.tex` PDF preview.
- `ask_clarification`: structured user confirmation.
- `visual_quality_check` / `visual_refinement_check`: visual artifact QA gates.
- sandbox tools: `bash`, `ls`, `read_file`, `write_file`, and `str_replace`.

Skills are discovered from `skills/public` and `skills/custom`, then injected according to `extensions_config.json`. Bundled public skills cover academic writing, citation curation, empirical methods, scientific figures, data analysis, technical diagrams, presentations, PDFs, and skill/plugin creation.

## Safety and Deployment

- Production should set `MEDRIX_FLOW_ENV=production`.
- When exposing the UI/API through Nginx, set `MEDRIX_FLOW_UI_PASSWORD`; otherwise protected pages and APIs deny access by default.
- Scripted administration can use `MEDRIX_GATEWAY_ADMIN_TOKEN` with the `x-medrix-admin-token` header.
- Local sandboxing is for trusted development. Production or untrusted code execution should use `AioSandboxProvider` or provisioner/Kubernetes mode.
- Bash tool calls go through safety auditing and path isolation; thread virtual paths are constrained to `/mnt/user-data`.
- External code execution, long experiments, automatic experiment-code modification, and final manuscript release should pass through human gates by default.

## Project Structure

```text
.
├── backend/
│   ├── app/gateway/                 # FastAPI Gateway API
│   ├── packages/harness/medrix_flow/ # LangGraph agents, tools, research, sandbox
│   └── tests/                        # backend tests
├── frontend/
│   ├── src/app/                      # Next.js app routes
│   ├── src/components/               # workspace, settings, artifact UI
│   └── test/                         # frontend tests
├── skills/
│   ├── public/                       # bundled skills
│   └── custom/                       # user skills
├── docker/
│   ├── nginx/                        # local and production reverse proxy config
│   └── provisioner/                  # optional sandbox provisioner
├── scripts/                          # setup, serve, deploy, doctor helpers
├── config.example.yaml               # app config template
└── extensions_config.example.json    # MCP and skills config template
```

## Development and Verification

Backend:

```bash
cd backend
make lint
make test
```

Frontend:

```bash
cd frontend
pnpm lint
pnpm typecheck
pnpm test:unit
BETTER_AUTH_SECRET=local-dev-secret pnpm build
```

Full local verification:

```bash
make verify
```

## License and Acknowledgements

This project is released under the MIT License. Anaxa builds on LangGraph, Next.js, FastAPI, open-source agent tooling, and ideas from scientific automation projects. Its product stance is a research copilot: models can help with research work, but they do not replace the researcher's judgment, evidence responsibility, or authorship responsibility.
