# Anaxa 1.0

[English](./README_en.md) | **中文**

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
  <b>可审计、可暂停、可接管的科研智能体工作台</b><br/>
  文献检索 · 证据映射 · 实验闭环 · LaTeX/PDF 成稿 · 人工把关
</p>

---

## Anaxa 是什么

Anaxa 是一个面向科研工作流的开源智能体系统。它不是普通聊天机器人，也不是无人监管的论文生成器，而是把文献检索、证据审计、实验执行、论文写作、同行评审式检查和最终产物整理串联到同一个可追踪的研究生命周期中。

系统的核心目标，是把科研任务从“在对话里临时生成一段文字”推进到“每一步都有来源、产物、状态和人工接管点”的工作台。普通聊天仍然是入口，但复杂研究任务会被路由到后端的学术检索、实验、论文导出和研究编排工具。

适合的场景包括：

- 系统综述、相关工作、课题背景调研和参考文献整理。
- 需要 claim-level 证据绑定的论文草稿、实验报告和研究备忘录。
- CS/AI、数据科学、生物信息和实证研究中的可复现实验分析。
- LaTeX 论文 bundle、BibTeX、citation audit 和 PDF 的一次性导出。
- 长任务、阶段性科研项目和需要人工批准的自动化研究流程。

## 论文产物示例

以下图片为论文/报告生成流程中的产物示例，仅用于展示 Anaxa 可整理的论文级表格、图表、公式和分析页面形态。示例内容只供参考，不构成本项目的正式实验结论、性能承诺或领域事实声明。

<p align="center">
  <img src="./docs/assets/paper-products/regression-metrics-predicted-actual.svg" alt="论文产物示例：回归指标表与预测值对真实值散点图" width="760">
</p>

<p align="center">
  <img src="./docs/assets/paper-products/scfm-vcfm-evaluation.svg" alt="论文产物示例：scFM 与 VCFM 评估要求对比页面" width="760">
</p>

<p align="center">
  <img src="./docs/assets/paper-products/hybrid-template-mlp-limits.svg" alt="论文产物示例：混合模型公式与 MLP 结构限制分析页面" width="760">
</p>

<p align="center">
  <img src="./docs/assets/paper-products/ablation-study-results.svg" alt="论文产物示例：消融实验结果表" width="760">
</p>

## 核心科研流程

Anaxa 的研究链路围绕 `ResearchQuest` 展开。它把一个科研任务拆成可检查、可恢复的阶段：

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

每个阶段都会把输入、输出、工具调用、产物、失败原因和 gate 决策写入研究 ledger。对于风险较高的步骤，例如外部代码执行、长实验、自动修改实验代码和最终论文发布，系统默认要求人工确认，而不是直接放任模型继续推进。

论文生成默认采用“两段式”交付：`research_assistant action="run_pipeline"` 会先在 `manuscript_draft` 阶段返回可编辑初稿和 fast draft `.tex`，并记录 `draft_ready_at`；随后创建可跟踪的后台 finalization run，继续完成审稿、修复、citation audit、质量审计、PDF 编译和最终 bundle。章节草稿会按 `research.manuscript_section_concurrency` 并发生成，且可通过 `fast_draft_model` / `finalization_model` 区分首版速度和最终质量模型。

## 关键能力

### 学术检索与引用体系

- `academic_research` 负责多源文献检索、元数据规范化、去重、证据卡生成、参考文献导出和 coverage audit。
- 支持 Semantic Scholar、OpenAlex、Crossref、arXiv、DBLP、OpenReview、ACL Anthology 等来源组合。
- 参考文献可按用户选择的格式输出，支持 APA 7、MLA 9、Chicago、GB/T 7714、plain text 和 BibTeX。
- 综述、survey、manuscript 类任务会启用更严格的质量配置：更高的 reference 目标、正文引用密度检查、离题文献过滤和自动补检索。

### Claim-Level Evidence Map

Anaxa 不把“文末列了参考文献”视为合格引用。核心论断需要绑定到具体证据：

```text
claim -> paper_id -> citation_key -> snippet/page/abstract evidence -> support_status -> confidence
```

没有全文 PDF 时，系统会降级使用标题、摘要和元数据，但会在 audit 中把这类证据标记为弱证据。缺失 citation key、滥用 `\nocite{*}`、正文缺少段落级引用、unsupported claim 或作者过程说明残留，都会在 `citation_audit.json` 中暴露，并可阻断最终 PDF release。

### 实验闭环

- `experiment_lab` 提供 Python-first 的实验执行、指标记录、图表生成和结果 bundle 导出能力。
- 支持 baseline、ablation、seed、metric、failure summary、branch ranking 和 reproducibility ledger。
- CS/AI 任务可覆盖回归、分类、聚类、降维、诊断图、模型评估和论文级结果摘要。
- 生物信息学任务可覆盖 bulk 表达、差异分析、富集分析、单细胞起步流程和常见科研图。
- 实证研究方法 skill 可把 DID、IV、RDD、PSM/IPW、DML、target trial、TMLE 等方法要求转换为实验 metadata 和 gate。

### 论文成稿与 PDF 导出

论文、综述、实验论文和正式研究报告默认生成 LaTeX bundle：

- `manuscript.tex`
- `references.bib`
- `citation_audit.json`
- `manuscript.pdf`

高层工具 `manuscript_export` 会统一完成文件写入、citation audit、LaTeX 编译和 artifacts 展示。底层仍保留 `write_file`、`citation_audit` 和 `present_files`，但正式论文交付优先使用高层出口，避免文件生成停留在临时聊天回答里。

当前 PDF 链路优先使用 `tectonic`。如果编译器缺失或 LaTeX 编译失败，系统会返回明确的工具错误，并保留 `.tex`、`.bib` 和 audit 文件，便于继续修复。

### 质量审计与自动修复

`ResearchQualityAudit` 覆盖投稿前常见质量风险：

- 文献覆盖不足。
- 正文引用密度过低。
- 离题或跨领域外围文献混入。
- 定量证据、benchmark、case study 或实验结果不足。
- 评估框架缺少实施挑战与解决方案。
- 重复表达、绝对化措辞、过渡缺失和过程说明残留。

默认策略是先自动补检索、补证据、补 benchmark、重写薄弱章节；修复预算耗尽后，再进入人工 gate，并返回具体的补救报告。

### Thread Artifacts 与对话级记忆

- 每个 chat/thread 都有独立的 `workspace`、`uploads`、`outputs` 和 `memory.json`。
- 新建对话从空记忆开始，不会继承其他对话的用户画像或研究状态。
- 当前对话的报告、BibTeX、PDF、图表、表格和审计文件会写入 thread-scoped outputs，并可在 artifact 面板中发现、预览和下载。
- 后端保留旧 memory API 作为兼容层，但前端不再提供全局记忆设置页面。

### 功能配置只读展示

前端设置中的“功能”页集中展示当前后端配置：

- Agents：默认编排器、系统 agent、自定义 agent 和 subagent。
- Tools：MCP server、transport、enabled 状态、命令或 URL 摘要、redacted env/header keys。
- Skills：公共和自定义 skill 的名称、描述、分类、启用状态和 license。

这个页面是只读清单，不提供新增、删除、编辑、启用、停用或测试连接操作。底层管理 API 仍然保留，用于脚本化管理和兼容场景。

## 系统架构

```text
                  http://localhost:6200
                          |
                          v
                  +----------------+
                  |     Nginx      |
                  | reverse proxy  |
                  +---+--------+---+
                      |        |
      /api/langgraph/*|        |/api/*
                      v        v
        +----------------+   +------------------+
        | LangGraph      |   | Gateway API      |
        | Server :6203   |   | FastAPI :6202    |
        |                |   |                  |
        | Lead agent     |   | models/setup     |
        | middleware     |   | features         |
        | tools          |   | academic         |
        | subagents      |   | research         |
        +-------+--------+   | experiments      |
                |            | artifacts/runs   |
                |            +---------+--------+
                |                      |
                v                      v
        +----------------+   +------------------+
        | Sandbox / VFS  |   | SQLite / files   |
        | /mnt/user-data |   | .medrix-flow     |
        +----------------+   +------------------+

        +----------------------------------------+
        | Frontend :6201                         |
        | Next.js 16 / React 19 / Tailwind CSS 4 |
        +----------------------------------------+
```

请求路由：

- `/api/langgraph/*` -> LangGraph Server：agent 交互、thread、SSE streaming、long-running run。
- `/api/*` -> Gateway API：配置、features、academic、research、experiments、uploads、artifacts、runs。
- `/` -> Frontend：Next.js Web UI。

运行时数据主要存储在 `backend/.medrix-flow` 或 `MEDRIX_FLOW_HOME` 指向的目录中。每个 thread 的虚拟文件系统会映射为 `/mnt/user-data/{workspace,uploads,outputs}`。

## 快速开始

Anaxa 1.0 默认以开源开发版发布：下载源码后，在本地初始化、运行服务，并通过前端配置模型，便于直接使用和二次开发。默认不启用 UI 密码；该模式适合本机或可信局域网环境，不适合作为公网生产部署。

### 1. 安装基础工具

你只需要先安装这些系统工具：

- Python 3.12+
- Node.js 22+
- pnpm
- uv
- nginx

如果不确定是否装好了，先运行：

```bash
make check
```

缺少的工具会在检查结果里给出安装提示。`tectonic` 是可选项，只影响 LaTeX PDF 生成的稳定性。

### 2. 下载并初始化

```bash
git clone <repo-url>
cd <repo-folder>
make bootstrap
```

`make bootstrap` 会完成三件事：

- 检查本机依赖是否可用。
- 创建本地配置文件：`config.yaml`、`.env`、`frontend/.env`、`extensions_config.json`。
- 安装后端和前端依赖。

生成的配置文件都在 `.gitignore` 中，不会随源码提交。真实 API Key 不需要手动写入文件，后续在前端配置即可。

### 3. 启动

```bash
make dev
```

启动后访问：

```text
http://localhost:6200
```

首次进入页面后，打开左下角“设置和更多 -> 配置”，添加至少一个聊天模型和 API Key，保存后即可开始新对话。

本地开发服务会同时启动：

- Frontend: `http://localhost:6201`
- Gateway API: `http://localhost:6202`
- LangGraph Server: `http://localhost:6203`
- Nginx unified entry: `http://localhost:6200`

### Docker 可选路径

如果你不想在本机安装 Python/Node/nginx，可以使用 Docker 开发模式：

需要 Docker Compose `v2.24+`。`make docker-start` 会检查 Compose 版本、补齐缺失的 `.env` 文件，并从 `docker/` 目录使用相对 compose 文件名；`make docker-stop`、`make docker-logs` 和重启操作不会修改配置文件。

```bash
make docker-init
make docker-start
```

直接调用 Compose 时，请从仓库根目录执行并设置宿主机路径变量，避免 provisioner 模式把宿主机目录解析到错误位置：

```bash
MEDRIX_FLOW_ROOT="$PWD" docker compose -f docker/docker-compose-dev.yaml up --build
```

访问地址同样是：

```text
http://localhost:6200
```

停止 Docker 开发环境：

```bash
make docker-stop
```

### 常用命令

| 命令 | 说明 |
|---|---|
| `make bootstrap` | 第一次使用：检查依赖、创建本地配置、安装依赖 |
| `make check` | 检查 Node.js、pnpm、uv、nginx 是否安装 |
| `make install` | 安装 backend 与 frontend 依赖 |
| `make dev` | 启动开发模式，支持热重载 |
| `make dev-daemon` | 后台启动开发服务 |
| `make stop` | 停止本地服务 |
| `make docker-start` | 启动 Docker 开发模式 |
| `make docker-stop` | 停止 Docker 开发模式 |
| `make verify` | 跑 backend lint/test、frontend lint/typecheck/unit/build |
| `make release-check` | 发布前检查本地 secrets、缓存、记忆和运行数据是否被 Git 跟踪 |

## 配置

### 普通用户：在前端配置

启动后进入 `http://localhost:6200`，打开“设置和更多 -> 配置”：

- 添加模型 provider、模型名和 API Key。
- 配置网络搜索、网页抓取、学术检索增强等工具 API Key。
- 如需图像生成，配置 Google AI Studio 或 OpenAI-compatible 图像接口。
- 保存后配置会写入本地文件并热重载。

前端“功能”页只读展示当前可用的 Agents、MCP tools 和 Skills，不提供新增、删除、编辑、启用或停用入口。

### 高级用户：配置文件

`config.yaml` 和 `extensions_config.json` 仍然可用，适合脚本化管理或深度定制：

- `config.yaml`：模型、工具组、沙箱 provider、checkpointer、memory、research gate 和质量策略。
- `extensions_config.json`：MCP server 和 skills 的启用状态。
- `make config-upgrade`：把新字段合并到已有 `config.yaml`。

### 发布前安全检查

你的本地 `.env`、数据库、记忆、上下文、上传文件、输出文件、缓存和日志不应该随项目公开。发布前运行：

```bash
make release-check
```

如果检查失败，先处理输出中列出的路径，再 push 到公开仓库。该命令不会删除任何本地文件。

### 公网部署密码

开源开发版默认不设置 UI 密码。如果要把服务暴露到公网，请设置 `MEDRIX_FLOW_ENV=production`、`MEDRIX_FLOW_UI_PASSWORD`，以及可选的 `MEDRIX_GATEWAY_ADMIN_TOKEN`。

### 兼容技术标识

Anaxa 1.0 已完成面向用户的品牌迁移。为避免破坏已有脚本、数据目录、环境变量和 import 路径，以下技术标识仍然保留：

- Python import package: `medrix_flow`
- Python distribution / workspace package names 中的兼容命名
- Environment variables: `MEDRIX_FLOW_*`
- Runtime directory: `.medrix-flow`
- Admin header: `x-medrix-admin-token`
- 部分 Docker container、Compose project 和 cleanup prefix

这些名称属于兼容层，不代表新的产品品牌。

## 内置工具与 Skills

内置工具：

- `academic_research`：文献检索、证据卡、references 和 audit。
- `research_assistant`：研究生命周期、ledger、gate、quality audit 和 `run_pipeline`。
- `experiment_lab`：实验执行、图表、指标和 reproducibility bundle。
- `manuscript_export`：LaTeX/BibTeX/audit/PDF 一键导出。
- `citation_audit`：LaTeX citation key、正文引用和 claim 支持检查。
- `present_files`：展示 artifacts，并尝试为 `.tex` 生成 PDF 预览。
- `ask_clarification`：需要用户确认时生成结构化选项。
- `visual_quality_check` / `visual_refinement_check`：视觉产物质量门控。
- sandbox tools：`bash`、`ls`、`read_file`、`write_file`、`str_replace`。

Skills 会从 `skills/public` 和 `skills/custom` 中发现，并根据 `extensions_config.json` 注入。当前公共 skills 覆盖学术写作、文献整理、实证研究方法、科研图、数据分析、技术图示、PPT、PDF、插件/技能创建等场景。

## 安全与部署

- 生产环境应设置 `MEDRIX_FLOW_ENV=production`。
- 通过 Nginx 暴露 UI/API 时，应设置 `MEDRIX_FLOW_UI_PASSWORD`；否则受保护页面和 API 会以 fail-closed 方式拒绝访问。
- 脚本化管理可设置 `MEDRIX_GATEWAY_ADMIN_TOKEN`，通过 `x-medrix-admin-token` 请求头访问受保护接口。
- 本地沙箱适合可信开发环境；生产或不可信代码执行应切换到 `AioSandboxProvider` 或 provisioner/Kubernetes 模式。
- Bash 工具调用会经过安全审计和路径隔离；thread 虚拟路径限制在 `/mnt/user-data`。
- 外部代码执行、长实验、自动修改实验代码和最终论文 release 默认应经过 human gate。

## 项目结构

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

## 开发与验证

后端：

```bash
cd backend
make lint
make test
```

前端：

```bash
cd frontend
pnpm lint
pnpm typecheck
pnpm test:unit
BETTER_AUTH_SECRET=local-dev-secret pnpm build
```

全量本地验证：

```bash
make verify
```

## 许可证与致谢

本项目使用 MIT License。Anaxa 的工程基础吸收了 LangGraph、Next.js、FastAPI、开源 agent 工具生态和多个科研自动化项目中的思想，但项目目标始终是“科研副驾驶”：让模型参与研究工作，而不是替代研究者的判断、证据责任和最终署名责任。
