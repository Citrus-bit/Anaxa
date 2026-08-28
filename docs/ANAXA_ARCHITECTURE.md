# MedrixFlow 项目架构文档

> 最后更新: 2026-04-02

## 1. 项目概述

<!-- 来源归属: 本项目继承自一个开源上游项目，归属信息见根目录 LICENSE 文件 -->
MedrixFlow 是一个全栈 AI 代理编排平台。后端基于 LangGraph 实现多代理协作与状态管理，前端基于 Next.js 16 提供现代化交互界面。

**核心定位**: AI 超级代理系统 — 沙箱执行 · 持久化记忆 · 多代理协作 · 可扩展工具生态

## 2. 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.12+ | 运行时 |
| LangGraph | 1.0.6+ | 多代理编排引擎（有向图状态机） |
| LangChain | 1.2.3+ | LLM 抽象层、工具系统、MCP 适配器 |
| FastAPI | 0.115.0+ | Gateway REST API |
| uv | latest | 包管理器 |
| agent-sandbox | - | 沙箱代码执行 |
| markitdown | - | 多格式文档转 Markdown |

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| Next.js | 16 | React 元框架 (App Router + Turbopack) |
| React | 19 | UI 库 |
| TypeScript | 5.x | 类型安全 |
| TailwindCSS | 4 | 原子化 CSS |
| Shadcn UI | - | 基础组件库 |
| MagicUI | - | 动效组件 |
| TanStack Query | 5.x | 服务端状态管理 |
| LangGraph SDK | 1.5.3+ | Agent 交互与 SSE 流式 |
| pnpm | 10.26.2 | 包管理器 |

### 基础设施

| 技术 | 用途 |
|------|------|
| Nginx | 统一反向代理入口 |
| Docker / Docker Compose | 容器化部署 |
| GitHub Actions | CI/CD (lint + 单元测试) |

## 3. 系统架构

### 3.1 服务拓扑

```
                     ┌─────────────────────────────────────────────┐
                     │            Nginx (端口 1000)                │
                     │           统一反向代理入口                   │
                     └──────┬────────────────────┬─────────────────┘
                            │                    │
          /api/langgraph/*  │                    │  /api/* (其他)
                            v                    v
          ┌──────────────────────┐  ┌──────────────────────────────┐
          │  LangGraph Server    │  │   Gateway API (端口 8001)    │
          │    (端口 2024)       │  │   FastAPI REST               │
          │                      │  │                              │
          │ ┌──────────────────┐ │  │  /api/models      模型列表   │
          │ │    Lead Agent    │ │  │  /api/mcp/config  MCP 配置   │
          │ │                  │ │  │  /api/skills      技能管理   │
          │ │  17 层中间件链    │ │  │  /api/memory      记忆数据   │
          │ │       |          │ │  │  /api/setup/*     配置管理   │
          │ │   工具系统        │ │  │  /api/threads/*   线程管理   │
          │ │       |          │ │  │                              │
          │ │  子代理(x3并行)   │ │  └──────────────────────────────┘
          │ └──────────────────┘ │
          └──────────────────────┘
                            │
          ┌──────────────────────┐
          │   Frontend (端口 3000)│
          │   Next.js 16         │
          │   React 19           │
          │   TailwindCSS 4      │
          │   Shadcn UI          │
          └──────────────────────┘
```

### 3.2 请求路由

通过 Nginx 统一代理：
- `/api/langgraph/*` → LangGraph Server：代理交互、线程管理、SSE 流式传输
- `/api/*`（其他）→ Gateway API：模型、MCP、Skills、记忆、文件上传、产物
- `/`（非 API）→ Frontend：Next.js Web 界面

### 3.3 配置加载链路

```
.env (load_dotenv) → config.yaml (yaml.safe_load)
  → AppConfig.resolve_env_variables（递归替换 $VAR）
  → AppConfig 单例缓存（mtime 自动热重载）
  → Gateway API 通过 get_app_config() 读取
```

核心文件: `backend/packages/harness/medrix_flow/config/app_config.py`

## 4. 后端架构

### 4.1 包结构

```
backend/
├── packages/harness/medrix_flow/   # 核心 harness 包
│   ├── agents/                     # 代理系统
│   │   ├── lead_agent/             # 主代理（工厂 + 提示词）
│   │   ├── middlewares/            # 17 个中间件组件
│   │   ├── memory/                 # 记忆系统（抽取、存储、纠正检测）
│   │   └── thread_state.py         # 线程状态 Schema
│   ├── sandbox/                    # 沙箱执行引擎 + 安全审计
│   ├── subagents/                  # 子代理系统（注册 + 执行器）
│   ├── tools/                      # 内置工具集
│   ├── mcp/                        # MCP 协议集成
│   ├── models/                     # 模型工厂 + Provider 补丁
│   ├── skills/                     # Skill 发现与加载
│   ├── community/                  # 社区工具（Tavily/Jina/Firecrawl/DuckDuckGo）
│   ├── config/                     # 配置系统（热重载 + 环境变量解析）
│   ├── reflection/                 # 动态模块加载
│   └── utils/                      # 工具函数
├── app/gateway/                    # FastAPI 网关
│   ├── app.py                      # 应用入口
│   └── routers/                    # 路由模块
├── tests/                          # 测试套件（277 用例）
├── langgraph.json                  # LangGraph 入口配置
├── pyproject.toml                  # Python 依赖
└── Dockerfile                      # 容器构建
```

### 4.2 Lead Agent

入口点: `medrix_flow.agents:make_lead_agent`（在 `langgraph.json` 中声明）

Lead Agent 是 LangGraph 有向图状态机的核心节点，职责：
- 动态模型选择（支持 Thinking/Vision 模式运行时切换）
- 编排 17 层中间件链
- 管理工具系统（沙箱 + MCP + 社区 + 内置）
- 委派子代理执行并行任务
- Skills 注入 + Memory 上下文注入

### 4.3 中间件链（17 层）

按严格顺序执行，处理各种横切关注点：

| # | 中间件 | 职责 | 可选 |
|---|--------|------|------|
| 1 | ThreadDataMiddleware | 创建线程专属隔离目录 | |
| 2 | UploadsMiddleware | 注入上传文件到上下文 | |
| 3 | SandboxMiddleware | 获取沙箱执行环境 | |
| 4 | DanglingToolCallMiddleware | 清理悬挂的未完成工具调用 | |
| 5 | GuardrailMiddleware | 工具调用授权守卫 | ✓ |
| 6 | ToolErrorHandlingMiddleware | 工具调用失败降级 | |
| 7 | SummarizationMiddleware | Token 超限时摘要压缩上下文 | ✓ |
| 8 | TodoListMiddleware | 计划模式任务进度跟踪 | ✓ |
| 9 | TitleMiddleware | 首轮对话后自动生成标题 | |
| 10 | MemoryMiddleware | 异步记忆抽取 + 纠正检测 | |
| 11 | ViewImageMiddleware | 视觉模型图像注入 | ✓ |
| 12 | DeferredToolFilterMiddleware | 延迟工具加载 | ✓ |
| 13 | SubagentLimitMiddleware | 子代理并发上限控制 | ✓ |
| 14 | LoopDetectionMiddleware | 代理无限循环检测 | |
| 15 | SandboxAuditMiddleware | Bash 命令安全审计（block/warn/pass） | |
| 16 | TokenUsageMiddleware | LLM 调用 Token 用量记录 | |
| 17 | ClarificationMiddleware | 澄清请求拦截（必须在最后） | |

### 4.4 子代理系统

- 内置代理: `general-purpose`（全工具集）、`bash`（命令专家）
- 并发上限: 3 个子代理/轮次，15 分钟超时
- 执行流: Agent 调用 `task()` → executor 后台运行 → 轮询结果 → SSE 事件推送

### 4.5 记忆系统

- 自动知识抽取: LLM 分析对话，提取用户背景/事实/上下文
- 纠正检测: 11 条中英文正则匹配用户纠正意图
- 防抖批处理: 可配置 debounce（默认 30s）
- 可插拔存储: 默认 `FileMemoryStorage`（JSON），可配置为 SQLite/Redis 等
- System Prompt 注入: 高置信度事实自动注入代理提示词

### 4.6 沙箱系统

- 虚拟路径映射: `/mnt/user-data/{workspace,uploads,outputs}` → 线程物理目录
- 双引擎: `LocalSandboxProvider`（本地）+ `AioSandboxProvider`（Docker）
- 安全: `SandboxAuditMiddleware` 三级审计 + `allow_host_bash` 配置开关
- 工具: `bash`, `ls`, `read_file`, `write_file`, `str_replace`

### 4.7 Gateway API 端点

| 路由 | 方法 | 用途 |
|------|------|------|
| `/api/models` | GET | 可用模型列表 |
| `/api/mcp/config` | GET/PUT | MCP 服务器配置管理 |
| `/api/mcp/test` | POST | MCP 连接测试 |
| `/api/skills` | GET/PUT | 技能管理 |
| `/api/skills/install` | POST | 安装 .skill 技能包 |
| `/api/memory` | GET | 记忆数据 |
| `/api/memory/reload` | POST | 强制重载记忆 |
| `/api/memory/config` | GET | 记忆配置 |
| `/api/memory/status` | GET | 记忆配置 + 数据 |
| `/api/threads/{id}/uploads` | POST/GET/DELETE | 文件上传管理 |
| `/api/threads/{id}` | DELETE | 删除线程本地数据 |
| `/api/threads/{id}/artifacts/{path}` | GET | 产物文件服务 |
| `/api/setup/config` | GET | 读取配置 |
| `/api/setup/models` | PUT | 保存模型配置 |
| `/api/setup/test-model` | POST | 测试模型连通性 |
| `/api/setup/test-tool-key` | POST | 测试工具 Key |
| `/health` | GET | 健康检查 |

## 5. 前端架构

### 5.1 目录结构

```
frontend/src/
├── app/                        # Next.js App Router 路由
│   ├── page.tsx                # / → 重定向 /workspace
│   ├── workspace/
│   │   ├── page.tsx            # /workspace → 重定向 /workspace/chats
│   │   ├── chats/
│   │   │   ├── page.tsx        # /workspace/chats → 聊天列表
│   │   │   └── [thread_id]/
│   │   │       └── page.tsx    # /workspace/chats/{id} → 聊天页面
│   │   └── agents/             # Agent 页面
│   └── mock/                   # Mock/Demo 页面
├── components/
│   ├── ui/                     # 基础 UI 组件 (Shadcn)
│   ├── workspace/              # 工作区组件
│   │   ├── messages/           # 消息展示组件
│   │   ├── settings/           # 设置对话框
│   │   ├── artifacts/          # 产物面板
│   │   └── ...
│   ├── ai-elements/            # AI 特有组件（推理/代码块/模型选择器）
│   └── landing/                # 着陆页组件
├── core/                       # 核心业务逻辑
│   ├── threads/                # 线程管理 + 流式传输
│   ├── setup/                  # 配置管理（类型/API/Hooks）
│   ├── i18n/                   # 国际化（中/英）
│   ├── settings/               # 本地设置 (localStorage)
│   ├── api/                    # API 客户端
│   ├── mcp/                    # MCP 集成
│   ├── skills/                 # 技能系统
│   ├── artifacts/              # 产物管理
│   ├── models/                 # 数据模型
│   ├── messages/               # 消息处理
│   ├── tasks/                  # 子任务 Context
│   ├── todos/                  # Todo 系统
│   └── utils/                  # 工具函数
├── hooks/                      # 自定义 React Hooks
├── lib/                        # 共享库
├── server/                     # 服务端代码（预留，未启用）
│   └── better-auth/            # 认证（预留）
└── styles/                     # 全局样式
```

### 5.2 状态管理

- **服务端状态**: TanStack React Query（线程列表、模型、MCP 配置等）
- **本地设置**: localStorage（`medrix_flow.local-settings`）
- **流式状态**: LangGraph SDK `useStream` + sessionStorage 断连恢复

### 5.3 流式传输

- SSE 流式渲染: Agent 响应、Thinking 过程、子代理任务进度
- 断连恢复: `reconnectOnMount` + `streamResumable` + sessionStorage 映射
- 乐观 UI: 消息发送即刻展示 + 线程列表 optimistic 插入

### 5.4 国际化

- 支持语言: 中文 (zh-CN)、英文 (en-US)
- Hook: `useI18n()` → `t.xxx.yyy`

## 6. 基础设施

### 6.1 目录结构

```
medrix-flow/
├── scripts/
│   ├── serve.sh              # 服务启动（并行 + 健康检查）
│   ├── start-daemon.sh       # 守护进程启动
│   ├── config-upgrade.sh     # 配置版本升级
│   ├── deploy.sh             # Docker 部署
│   ├── configure.py          # 配置初始化
│   ├── check.py              # 前置工具检查
│   ├── cleanup-containers.sh # 沙箱容器清理
│   └── docker.sh             # Docker 开发环境管理
├── docker/
│   ├── nginx/                # Nginx 配置
│   ├── docker-compose.yaml   # 生产编排
│   └── docker-compose-dev.yaml # 开发编排
├── skills/
│   └── public/               # 17 个公共技能包
├── config.yaml               # 主配置（gitignored）
├── config.example.yaml       # 配置模板
├── .env                      # 环境变量（gitignored）
└── Makefile                  # 统一命令入口
```

### 6.2 常用命令

| 命令 | 说明 |
|------|------|
| `make config` | 首次生成配置文件 |
| `make install` | 安装全部依赖 |
| `make dev` | 开发模式启动（热重载） |
| `make start` | 生产模式启动 |
| `make stop` | 停止所有服务 |
| `make clean` | 停止 + 清理临时文件 |
| `make up` | Docker 生产部署 |
| `make down` | 停止 Docker 容器 |

### 6.3 CI/CD

- **后端**: `.github/workflows/backend-unit-tests.yml` — PR 触发 `ruff check` + `pytest`（277 用例）
- **前端**: `.github/workflows/lint-check.yml` — lint 检查

## 7. 多渠道接入

除 Web 界面外，支持 IM 渠道接入：

| 渠道 | 传输方式 | 特点 |
|------|---------|------|
| 飞书 | 卡片消息原地更新 | 流式响应，存储 message_id 逐块 patch |
| Slack | Socket Mode WebSocket | 无需公网 IP |
| Telegram | Bot API | 每用户独立会话 |

## 8. 设计决策

### 8.1 基于上游开源项目演进

<!-- 归属说明: 上游项目的版权声明统一收敛到根目录 LICENSE 文件；此处仅记录 fork 后的主要演进点 -->
MedrixFlow 从一个开源上游项目 fork 而来（完整归属见 LICENSE），后续进行了大量二次开发：
- 品牌重命名与视觉系统重做
- 端口调整（2026 → 1000）
- 安全加固（SandboxAuditMiddleware、security.py）
- 功能增强（TokenUsageMiddleware、Memory 纠正检测、MemoryStorage 抽象）
- Citations 系统移除（简化维护负担）
- 前端配置 UI（Setup Settings Page）
- MCP 工具 CRUD + 连接测试

### 8.2 中间件链式架构

选择显式有序中间件链（而非事件驱动），确保执行顺序可预测且调试容易。中间件之间存在隐式依赖，顺序不可随意调整。

### 8.3 配置热重载

AppConfig 单例 + mtime 检测机制，避免服务重启。前端 UI 修改 → 写入 config.yaml + .env → `reload_app_config()` → 即时生效。

### 8.4 可插拔存储

Memory 系统采用 `MemoryStorage` ABC 抽象，默认 JSON 文件存储，支持通过 `storage_class` 配置切换。
