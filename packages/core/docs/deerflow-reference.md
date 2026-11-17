# DeerFlow 2.0 代码借鉴指引

> **立场**：借鉴不是照抄。DeerFlow 的架构抽象（Sandbox 三模式、SKILL.md、tool-call recovery、provider registry）是我们的蓝本；它们的具体实现是**参考点而非起点**——在 SanYuanAgent 落地时按本仓代码规范（Pydantic schema、`SandboxClient` 封装、structlog 脱敏、harness/app 边界、mypy strict）重写。
>
> **License**：DeerFlow 2.0 MIT；SanYuanAgent `packages/core/` Apache 2.0。两者兼容。**入库时 `NOTICE` 必须标注借鉴来源 + commit SHA**（见末尾）。


> **退役注记（2026-07-14）**：下文提到的 `sandbox_supervisor/` 实现已随「沙箱统一 CubeSandbox」决策退役删除；本文档保留为当年借鉴映射的历史参考，路径不再存在。

## 目录

- [0. DeerFlow 仓库拓扑](#0-deerflow-仓库拓扑)
- [1. Sandbox 模块](#1-sandbox-模块)
- [2. Skills 模块](#2-skills-模块)
- [3. Tool-Call Recovery](#3-tool-call-recovery)
- [4. Provider Registry / Models](#4-provider-registry--models)
- [5. 可选借鉴：Middleware 栈](#5-可选借鉴middleware-栈)
- [6. 可选借鉴：Subagents](#6-可选借鉴subagents)
- [7. 明确不借鉴](#7-明确不借鉴)
- [8. 代码规范对齐清单](#8-代码规范对齐清单)
- [9. License & NOTICE 义务](#9-license--notice-义务)

---

## 0. DeerFlow 仓库拓扑

关键镜像：DeerFlow 本身就是 `harness + app` 两层分层（和 SanYuanAgent 同构）。

```
github.com/bytedance/deer-flow  (default branch: main, License: MIT)
└── backend/
    ├── packages/harness/deerflow/       ← 对应我们 packages/core/sanyuan_agent/
    │   ├── client.py                    ← CLI / SDK 入口
    │   ├── agents/
    │   │   ├── factory.py
    │   │   ├── lead_agent/
    │   │   ├── memory/
    │   │   ├── middlewares/             ← ★ 15+ 个 middleware（丰富借鉴源）
    │   │   ├── checkpointer/
    │   │   └── thread_state.py
    │   ├── sandbox/                     ← 本文档核心借鉴源 #1
    │   ├── skills/                      ← 本文档核心借鉴源 #2
    │   ├── reflection/
    │   │   └── resolvers.py             ← 本文档核心借鉴源 #3
    │   ├── models/                      ← 本文档核心借鉴源 #4
    │   ├── tools/                       ← Tool 基类 + skill_manage_tool
    │   ├── subagents/
    │   ├── runtime/                     ← Run lifecycle + stream_bridge
    │   ├── guardrails/
    │   ├── tracing/
    │   ├── mcp/
    │   ├── config/
    │   ├── community/
    │   ├── uploads/
    │   └── utils/
    └── app/                             ← 对应我们 app/
        ├── gateway/                     ← 对应我们 app/gateway/
        └── channels/                    ← 对应我们 app/channels/（Phase 2）
```

**速查 URL 模式**：`https://github.com/bytedance/deer-flow/blob/main/<path>` 打开任一文件。

---

## 1. Sandbox 模块

### 1.1 目标

SanYuanAgent 要的三模式（Local / Docker / K8s）+ Supervisor 抽象 + 审计 hook + Playwright CDP。DeerFlow 目前开源的是 Local 模式 + Provider 抽象，其他模式由我们自己在 `sandbox_supervisor/` 里实现。

### 1.2 DeerFlow 文件清单（全部位于 `backend/packages/harness/deerflow/sandbox/`）

| DeerFlow 文件 | 作用 | SanYuanAgent 目标映射 | 借鉴深度 |
|---|---|---|---|
| `sandbox.py` | Sandbox 核心接口 / data class | `packages/core/sanyuan_agent/sandbox/client.py`（SandboxClient 接口） | **结构 + 命名** |
| `sandbox_provider.py` | Provider 抽象基类 | `packages/core/sanyuan_agent/sandbox/providers/base.py` | **结构** |
| `local/local_sandbox.py` | Local 模式实现（~16KB，进程式 sandbox） | `sandbox_supervisor/providers/local_provider.py` | **算法思路** |
| `local/local_sandbox_provider.py` | Local Provider 注册 | 同上 | **结构** |
| `local/list_dir.py` | 文件系统操作辅助 | 我们走 `SandboxClient.read_file/list_dir` 接口，具体实现在 Supervisor 内 | **思路** |
| `middleware.py` | Agent 图中把 Sandbox 注入 state 的 middleware | 我们用 `ToolContext.sandbox` 依赖注入模式，不走 LangGraph middleware 层 | **参考** |
| `security.py` | Sandbox 内命令白名单 / 资源限制校验 | `sandbox_supervisor/security/limits.py`（我们按 Sandbox Spec §4 重写更严格版本） | **参考** |
| `exceptions.py` | Sandbox 领域异常 | `packages/core/sanyuan_agent/sandbox/exceptions.py` | **直接改名引入** |
| `file_operation_lock.py` | workspace 文件并发锁 | `sandbox_supervisor/workspace/lock.py`（Supervisor 内） | **算法** |
| `tools.py` | Sandbox 暴露给 Agent 的 tool adapters（~54KB 最大单文件） | `packages/core/sanyuan_agent/tools/builtins/{bash,file_*,computer_use}.py` 拆解 | **拆解 + 改写** |
| `search.py` | Sandbox 内 `web_search` 工具 | `packages/core/sanyuan_agent/tools/builtins/web_search.py` | **参考 API 形状** |

### 1.3 具体借鉴建议

**1) Provider 接口**：`sandbox_provider.py` 的抽象基类定义（`provision / destroy / exec / browser` 等方法签名）是**最值得直接借鉴**的部分。我们的 `packages/core/sanyuan_agent/sandbox/providers/base.py` 按相同契约写，但：

- 所有方法参数 / 返回用 **Pydantic BaseModel**（DeerFlow 用 dataclass，我们统一 Pydantic）
- `async` 全覆盖（DeerFlow 部分是同步接口）
- 加 `tenant_id` / `account_id` 强制参数（DeerFlow 无多租户）

**2) Local provider**：`local/local_sandbox.py` 展示了进程级 sandbox 怎么做（cgroup-less 的 fallback）。我们的 `local_provider.py` 借鉴其**资源限制与清理逻辑**，但：
- 加 gVisor 可选包裹（Sandbox Spec §4.3）
- Windows 上不保证——我们的 local 只支持 Linux

**3) `tools.py` 切碎**：DeerFlow 54KB 的单文件是所有 Sandbox tool 的合集。我们按"一个 Tool 一个文件"拆（本仓 `tool-authoring` skill 的强约定）：
```
packages/core/sanyuan_agent/tools/builtins/
├── bash.py
├── file_read.py
├── file_write.py
├── file_list.py
├── web_fetch.py
├── web_search.py
├── computer_use.py
└── __init__.py    ← TOOL_REGISTRY
```

**4) 不直接 import docker**：DeerFlow 的 local provider 就是进程级，没有 docker 依赖。我们的 `docker_provider.py` 和 `k8s_provider.py` 是 SanYuanAgent 特有 —— **只能由 `sandbox_supervisor/` 内部引入 docker SDK**（本仓 `sandbox-client` skill 铁律 4）。

### 1.4 具体链接

- Provider 抽象：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/sandbox/sandbox_provider.py
- Local 实现：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py
- 工具合集（需拆）：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/sandbox/tools.py
- 并发锁：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/sandbox/file_operation_lock.py
- 异常类：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/sandbox/exceptions.py

---

## 2. Skills 模块

### 2.1 目标

SanYuanAgent 的 SKILL.md 三级渐进加载 + Ed25519 签名 + 租户可信 key 列表。DeerFlow 已经有完整的加载 / 解析 / 校验链。

### 2.2 DeerFlow 文件清单（`backend/packages/harness/deerflow/skills/`）

| DeerFlow 文件 | 作用 | SanYuanAgent 映射 | 借鉴深度 |
|---|---|---|---|
| `types.py` | `Skill` / `SkillMetadata` / `SkillBundle` data class | `packages/core/sanyuan_agent/skills/types.py` | **直接改写为 Pydantic** |
| `parser.py` | SKILL.md frontmatter + markdown 解析 | `packages/core/sanyuan_agent/skills/parser.py` | **结构 + 算法** |
| `loader.py` | 按路径 / 目录加载 Skill，返回内存对象 | `packages/core/sanyuan_agent/skills/loader.py` | **结构** |
| `manager.py` | Skill registry / 查询 / 激活 | `packages/core/sanyuan_agent/skills/registry.py` | **结构** |
| `installer.py` | 从远程 URL / tarball 安装 Skill | Phase 2 Skills 市场时再引入 | **延后借鉴** |
| `validation.py` | SKILL.md 字段校验（required fields、版本号等） | `packages/core/sanyuan_agent/skills/validator.py` | **规则表** |
| `security_scanner.py` | Skill 内容安全扫描（危险模式） | `packages/core/sanyuan_agent/skills/scanner.py` + 加上 **Ed25519 签名验证**（DeerFlow 没有） | **扩展** |

### 2.3 关键差异：签名

DeerFlow 的 `security_scanner.py` 做**静态内容扫描**（扫禁词、可疑命令）。SanYuanAgent 需要**额外一层 Ed25519 签名**（PRD 四层数据承诺的一部分）——这是我们独有的，需要自己加：

```
packages/core/sanyuan_agent/skills/
├── types.py              ← 借鉴 DeerFlow
├── parser.py             ← 借鉴 DeerFlow
├── loader.py             ← 借鉴 DeerFlow，加 verify_signature 强制调用
├── registry.py           ← 借鉴 DeerFlow 的 manager.py
├── validator.py          ← 借鉴 DeerFlow 的 validation.py
├── scanner.py            ← 借鉴 DeerFlow 的 security_scanner.py
└── signature.py          ← ★ SanYuanAgent 独有，Ed25519 签名 + 租户 trust store
```

**`loader.py` 必须在 load 路径上调用 `verify_signature`**——本仓 `skill-authoring` skill 铁律 (`verify_skill_signature` 必须在所有 loader 路径调用，`if DEV_MODE: skip` 不允许)。

### 2.4 渐进加载

DeerFlow 的 `parser.py` + `loader.py` 本来就分离了 frontmatter（L1）和正文（L2），可以直接支持**延迟加载**。L3（`examples/` 和 `assets/`）我们按需 `skill.load_assets(path)` 按方法读。

### 2.5 具体链接

- Types：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/types.py
- Parser：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/parser.py
- Loader：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/loader.py
- Manager：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/manager.py
- Validation：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/validation.py
- Security scanner：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/skills/security_scanner.py

---

## 3. Tool-Call Recovery

### 3.1 目标

LangGraph Agent 调用 LLM，LLM 可能产出"有 tool_call 但没跟 tool_result"的悬挂状态（LLM 中断、超时、工具抛异常）。Recovery 的任务是**把 state 修回合法**，让下一次 LLM 调用不炸。

### 3.2 DeerFlow 主参照

**核心文件**：`backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py`

这是 DeerFlow 对这个问题的主要解法——作为 LangGraph middleware 在 `modify_state_messages` 钩子里：
1. 扫描当前 messages 找出"dangling tool_call"（有 `AIMessage.tool_calls` 但缺对应 `ToolMessage`）
2. 补入一条合成的 `ToolMessage(content="...error / cancelled...", tool_call_id=...)`
3. 让下一轮 LLM 输入合法

### 3.3 配套文件

DeerFlow 把"Tool 相关错误处理"拆成几个 middleware：

| 文件 | 作用 | 借鉴价值 |
|---|---|---|
| `dangling_tool_call_middleware.py` | 补齐悬挂 tool_call | ★ 核心参照 |
| `tool_error_handling_middleware.py` | Tool 执行抛异常时的 graceful 处理 | ★ 一并借鉴 |
| `llm_error_handling_middleware.py` | LLM 调用本身出错（timeout / API down） | 参考 |
| `loop_detection_middleware.py` | Agent 陷入死循环的检测（重复相同 tool call） | 额外价值 |
| `deferred_tool_filter_middleware.py` | 延迟 tool（需用户确认）的过滤 | 对应我们 RiskConfirmCard 流程 |

以及 `reflection/resolvers.py`（single file，4KB）—— "当 LLM 返回 JSON 格式错误时，resolver 尝试修复"。这是**另一种"recovery"语义**（schema recovery），也值得参照。

### 3.4 SanYuanAgent 落点

```
packages/core/sanyuan_agent/recovery/
├── __init__.py
├── dangling_tool_call.py       ← 核心：悬挂 tool_call 修复
├── tool_error.py               ← Tool 异常 graceful 化
├── loop_detection.py           ← 死循环检测
├── schema_repair.py            ← LLM JSON schema 修复（借 DeerFlow reflection/resolvers.py）
└── middleware.py               ← 包装成 LangGraph middleware，注入 lead_agent
```

**本仓代码规范对齐**：
- 每个 middleware 是**纯函数**（输入 state，输出 state 增量），不持有全局状态
- 用 structlog 记录 recovery 事件（字段：`run_id`, `recovery_type`, `tool_call_id`, `reason`）——审计必需
- 每个 recovery 动作 emit 一条 audit event（`recovery.dangling_tool_call_patched` 等），走 `audit.operations` 表
- 测试在 `tests/unit/harness/recovery/` 下

### 3.5 具体链接

- **★ Dangling tool call**：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/dangling_tool_call_middleware.py
- Tool error handling：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/tool_error_handling_middleware.py
- LLM error handling：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py
- Loop detection：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py
- Schema 修复：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/reflection/resolvers.py
- Deferred tool filter（与 RiskConfirmCard 对应）：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/deferred_tool_filter_middleware.py

---

## 4. Provider Registry / Models

### 4.1 目标

SanYuanAgent Phase 1 需要 DeepSeek / Qwen / Claude 三个 provider，用 LiteLLM 统一。DeerFlow 提供了 factory + 各家 provider 的实现样本。

### 4.2 DeerFlow 文件清单（`backend/packages/harness/deerflow/models/`）

| 文件 | 作用 | 借鉴价值 |
|---|---|---|
| `factory.py` | Provider 工厂（按 provider name + config 创建 client） | ★ 结构直接借鉴 |
| `credential_loader.py` | 从 env / secret 读 API key | ★ 借鉴，但我们改造成从 Infisical 读（Phase 1 后期） |
| `claude_provider.py` | Anthropic provider 包装（~14KB） | 参考，但我们优先走 LiteLLM 统一接口 |
| `openai_codex_provider.py` | OpenAI / Codex 专用适配（~17KB，处理 Codex CLI 协议） | 我们 Phase 1 不需要 Codex |
| `vllm_provider.py` | vLLM 自部署模型适配（~11KB） | 未来私有化部署时借鉴 |
| `patched_deepseek.py` | DeepSeek 修复（API quirks） | ★ 直接借鉴（DeepSeek 是我们 Phase 1 主 provider） |
| `patched_openai.py` | OpenAI API quirks 修复 | 借鉴（若我们需要直连 OpenAI 而不走 LiteLLM） |
| `patched_minimax.py` | MiniMax API 适配 | 选借 |

### 4.3 LiteLLM vs 自写 provider

SanYuanAgent Phase 1 决定用 LiteLLM 做统一抽象。这意味着：

- **Factory 模式借鉴**：`factory.py` 的 `get_provider(name, config)` 模式可以保留，但内部改为 `return LiteLLMWrapper(model=name, ...)`
- **`patched_*` 借鉴**：LiteLLM 本身有 bug 时，`patched_deepseek.py` 这类文件是"监控 provider quirks"的最佳参照 —— 未来我们发现 LiteLLM 对某家不兼容，可以写 `patches/deepseek.py` 类似 monkey patch
- **不借鉴**：`openai_codex_provider.py` 和 `vllm_provider.py` 的直连逻辑（我们全走 LiteLLM）

### 4.4 SanYuanAgent 落点

```
packages/core/sanyuan_agent/models/
├── __init__.py
├── factory.py               ← 借鉴 DeerFlow，内部走 LiteLLM
├── registry.py              ← Provider 元信息注册（cost / context_window / capabilities）
├── cost_tracker.py          ← Phase 1 四层数据承诺要求：每次调用都要记 cost
├── credential_loader.py     ← 借鉴，加 Infisical 分支
└── patches/
    └── deepseek.py          ← 按需加入
```

### 4.5 具体链接

- Factory：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/models/factory.py
- Credential loader：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/models/credential_loader.py
- DeepSeek patch：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/models/patched_deepseek.py
- Claude provider（参考）：https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/models/claude_provider.py

---

## 5. 可选借鉴：Middleware 栈

DeerFlow 的 `agents/middlewares/` 共 15 个 middleware，除了 recovery 相关的几个，其余按需借鉴：

| Middleware | 作用 | SanYuanAgent 是否借 |
|---|---|---|
| `clarification_middleware.py` | Agent 主动发问澄清 | ⚠ Phase 2 再看 |
| `memory_middleware.py` | 自动把 memory 注入 prompt | ✅ Phase 1 借鉴，结合我们四层记忆 |
| `sandbox_audit_middleware.py` | Sandbox 调用的审计事件 | ✅ **直接借鉴**（对我们 Audit Journal 合规关键） |
| `subagent_limit_middleware.py` | Subagent 数量上限保护 | ⚠ Phase 2（Orchestrator 引入后） |
| `summarization_middleware.py` | Context 压缩 / 摘要 | ✅ 借鉴，对应我们 Working Memory compact |
| `thread_data_middleware.py` | 把 thread 元信息注入 state | ✅ 借鉴 |
| `title_middleware.py` | 自动生成对话标题 | ✅ 借鉴，前端 ThreadList 显示需要 |
| `todo_middleware.py` | Agent 自己维护 TODO 列表 | ⚠ Phase 2 |
| `token_usage_middleware.py` | 统计 token 用量 | ✅ **直接借鉴** |
| `uploads_middleware.py` | 处理用户上传的文件 | ✅ Phase 1 末期（需要 Storage service） |
| `view_image_middleware.py` | 多模态图片查看 | ⚠ Phase 2（视觉 Computer Use） |

**★ 优先借鉴**：`sandbox_audit_middleware.py`、`token_usage_middleware.py`、`memory_middleware.py`。

Base URL：`https://github.com/bytedance/deer-flow/blob/main/backend/packages/harness/deerflow/agents/middlewares/<name>.py`

---

## 6. 可选借鉴：Subagents

DeerFlow `subagents/executor.py`（~25KB）+ `subagents/registry.py`——对应我们 **Phase 2 Orchestrator-Worker 架构**。

| 文件 | 作用 | SanYuanAgent |
|---|---|---|
| `subagents/executor.py` | Subagent 执行器 | Phase 2 核心借鉴 |
| `subagents/registry.py` | Subagent 类型注册 | Phase 2 |
| `subagents/config.py` | Subagent 配置 schema | Phase 2 |
| `subagents/builtins/` | 内置 subagent（DeerFlow 有 deep-research / code 等） | 参照设计，我们定义自己的内置集 |

**不急于 Phase 1 借鉴**。但今天写下来，Phase 2 规划时直接有参照路径。

---

## 7. 明确不借鉴

| DeerFlow 部分 | 原因 |
|---|---|
| `app/gateway/` 内部业务逻辑 | 我们的 Gateway 要对接 Kun 框架 + S2S JWT + 多租户，业务语义不同 |
| `app/channels/` 的飞书 / 企微 / Slack 实现 | Phase 2 再看；DeerFlow 目前是单租户版本，适配我们多租户要重写 |
| DeerFlow 自己的 OIDC / 身份层 | 我们走 Authentik + Kun.Edge 业务身份，架构边界不同 |
| `installer.py`（Skill 远程安装） | Phase 1 只支持内置 Skills；Phase 3 企业市场再回头看 |
| `community/` 目录 | DeerFlow 社区贡献点，和我们业务无关 |
| `mcp/` 的具体集成 | MCP 我们走 harness/mcp_client 独立实现，不直接继承其内部细节 |
| `guardrails/` | 我们用 Higress AI Gateway（Phase 3），DeerFlow 的 guardrails 是应用层，不同抽象 |

---

## 8. 代码规范对齐清单

每次从 DeerFlow 拷 / 改 / 参考一个文件，**必须在 PR 里过这个清单**：

- [ ] 命名：文件路径从 `deerflow/` 改到 `sanyuan_agent/`；类名里 `DeerFlow*` 改 `SanYuanAgent*`
- [ ] 数据模型：dataclass / TypedDict 改为 **Pydantic BaseModel**
- [ ] 异步：同步方法改 `async def`；同步 IO 替换为 asyncio 对应的
- [ ] 类型标注：所有公开函数 type hints 完整，`mypy --strict` 零错误
- [ ] 租户：所有涉及数据 / 持久化的接口加 `tenant_id: UUID` / `account_id: UUID` 参数
- [ ] 日志：`print` / `logging` 全换为 `structlog`；避免直接打 prompt / tool_output（脱敏）
- [ ] Sandbox：**不**直接 `import docker` / `from kubernetes import client`（本仓 `sandbox-client` skill 铁律 4）
- [ ] Harness 边界：新文件里**不**出现 `from app.* import ...`（CI 硬检查）
- [ ] 测试：`tests/unit/harness/<module>/test_<file>.py` 跟进
- [ ] 依赖：新依赖走 `uv add`；`uv.lock` 一起提交
- [ ] License：文件头加 Apache 2.0 header；若是"重度借鉴"（>50% 结构雷同）加 `# Derived from bytedance/deer-flow (MIT): <original-path>`
- [ ] Docstring：留一行说明来源 `"""...Derived from DeerFlow <original-path>."""`（方便将来同步上游改动）

---

## 9. License & NOTICE 义务

### 9.1 License 兼容性

- DeerFlow：**MIT**
- SanYuanAgent Harness：**Apache 2.0**

MIT → Apache 2.0 **兼容**（MIT 是 Apache 2.0 的子集宽松度）。我们可以合法借鉴。但：

- 必须**保留** DeerFlow 的版权声明
- 必须**标注**哪些是借鉴的

### 9.2 `NOTICE` 文件要求

在 `packages/core/NOTICE`（Apache 2.0 惯例）加入：

```
SanYuanAgent (Harness)
Copyright <year> <你的公司 / idadawn>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.

----------------------------------------------------------------------
This product includes software developed by ByteDance and contributors:

  DeerFlow (https://github.com/bytedance/deer-flow)
  Copyright (c) ByteDance
  Licensed under the MIT License.

Specifically, the following modules are derived or adapted from DeerFlow:
  - packages/core/sanyuan_agent/sandbox/
    <- backend/packages/harness/deerflow/sandbox/
  - packages/core/sanyuan_agent/skills/
    <- backend/packages/harness/deerflow/skills/
  - packages/core/sanyuan_agent/recovery/
    <- backend/packages/harness/deerflow/agents/middlewares/{dangling,tool_error,llm_error,loop_detection}*
  - packages/core/sanyuan_agent/models/factory.py, credential_loader.py
    <- backend/packages/harness/deerflow/models/{factory,credential_loader}.py

Last synced: <commit-sha-of-deer-flow-main-at-time-of-sync>
----------------------------------------------------------------------
```

**更新时机**：每次从 DeerFlow "大范围同步" 时（不是每次改几行），更新 `Last synced` 的 commit SHA。

### 9.3 文件级 header

"重度借鉴"文件（>50% 结构雷同）在文件顶部加：

```python
# Copyright <year> <idadawn>
# Licensed under the Apache License, Version 2.0.
#
# Derived from ByteDance DeerFlow (MIT):
#   https://github.com/bytedance/deer-flow/blob/<sha>/backend/packages/harness/deerflow/<path>
# Adapted for SanYuanAgent conventions: Pydantic BaseModel, async, tenant-scoped, structlog, mypy strict.
```

**浅借鉴**（只参照接口 / 命名）的文件**不需要**加 header，标准 Apache 2.0 header 即可。

---

## 附录：快速索引（按任务查借鉴点）

| 你要干什么 | 先看 DeerFlow |
|---|---|
| 写 `SandboxClient` 接口 | `sandbox/sandbox_provider.py` + `sandbox/sandbox.py` |
| 写 Local sandbox provider | `sandbox/local/local_sandbox.py` |
| 写 Tool 基类 + builtins | `sandbox/tools.py`（54KB，需拆）+ `tools/tools.py` |
| 写 SKILL.md 解析 | `skills/parser.py` + `skills/types.py` |
| 写 Skill Loader（带签名） | `skills/loader.py`（加 Ed25519） |
| 写 dangling tool_call 修复 | **`agents/middlewares/dangling_tool_call_middleware.py`** |
| 写 Tool 异常 graceful 化 | `agents/middlewares/tool_error_handling_middleware.py` |
| 写 Loop 检测 | `agents/middlewares/loop_detection_middleware.py` |
| 写 Provider Factory | `models/factory.py` |
| 写 Credential loader | `models/credential_loader.py` |
| 写 Audit middleware | `agents/middlewares/sandbox_audit_middleware.py` |
| 写 Token / cost tracker | `agents/middlewares/token_usage_middleware.py` |
| 写对话标题自动生成 | `agents/middlewares/title_middleware.py` |
| 写 Working memory compaction | `agents/middlewares/summarization_middleware.py` |

---

*Last updated: 本文档对应 `bytedance/deer-flow@main` 扫描结果。同步 DeerFlow 上游改动时，重新跑 `mcp__github__get_file_contents` 确认路径不变。*
