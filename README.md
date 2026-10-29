# Open Deep Research · Reviewed

基于 [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research) 的研究助手，增加独立报告审查、按角色组装上下文、证据归档和运行用量记录。保留原项目的多模型、搜索工具、MCP 与 LangGraph Studio 支持，遵循 MIT 许可证。

日常 CLI 的默认流程：

```text
研究问题 → 搜索与研究 → 写报告 → 独立审查一次 → 必要时修改一次 → 输出
```

审查或修改失败时保留已有报告，并返回状态提示。`reviewed` 表示审过一次，`revised` 表示修改过一次，均不表示严格证据验收通过。

## 快速开始

需要 Python 3.10+；Studio 配置使用 Python 3.11。

```bash
git clone https://github.com/P0tenti4L-220/open-deep-research-reviewed.git
cd open-deep-research-reviewed
uv sync
cp .env.example .env
```

在 `.env` 填写所用服务的密钥。模板将各模型角色设为 `deepseek:deepseek-flash`，需要 `DEEPSEEK_API_KEY`；默认搜索使用 Tavily，需要 `TAVILY_API_KEY`。可以按 `provider:model` 格式修改各角色模型。

运行一次研究：

```bash
ODR_ALLOW_PAID_RUNS=1 uv run python -m open_deep_research.run \
  "解释 LangGraph 的研究流程，并给出来源" \
  --max-cost-usd 0.10
```

`0.10` 是单次模型费用估算上限的示例，搜索服务用量另计。CLI 在研究开始前检查付费开关、金额上限、价格表、模型能力声明及密钥是否存在。模型预算按冻结价格换算为 token 上限；估算不等于供应商账单。

附带的价格表是 **2026-10-26 的冻结快照**，仅覆盖其中列出的 DeepSeek 模型。更换模型或更新价格时，通过 `ODR_PRICING_PATH` 指定包含对应价格和能力声明的 JSON；未覆盖或未验证的价格会阻止 CLI 付费运行。

## 审查与上下文选项

```bash
# 默认：报告后审一次，最多修改一次
ODR_ALLOW_PAID_RUNS=1 uv run python -m open_deep_research.run "研究问题" --max-cost-usd 0.10

# 关闭 Reviewer
ODR_ALLOW_PAID_RUNS=1 uv run python -m open_deep_research.run "研究问题" --reviewer off --max-cost-usd 0.10

# 启用保留的多阶段审查
ODR_ALLOW_PAID_RUNS=1 uv run python -m open_deep_research.run "研究问题" --reviewer on --max-cost-usd 0.10

# 按角色组装上下文，输出 JSON 状态
ODR_ALLOW_PAID_RUNS=1 uv run python -m open_deep_research.run "研究问题" --context-policy scoped --json --max-cost-usd 0.10
```

`--arm A/B/C/D` 保留不同上下文与多阶段审查组合。CLI 未指定审查选项和 arm 时使用 `once`。Studio/API 使用配置默认值；如需同样的单次审查流程，设置 `reviewer_once=true, reviewer_enabled=false`。

多阶段审查可返回 `partial` 或 `inconclusive`。源码中存在检查机制，不代表所有真实任务都能完成验收，也不代表已证明准确率、成本或速度提升。

## LangGraph Studio

```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

在 Studio 的 `messages` 输入研究问题，通过助手配置选择模型、搜索服务和审查选项。图入口由 `langgraph.json` 指向 `deep_researcher`。

## 运行记录与隐私

运行记录默认写入 `.odr_runs/`，可能包含提问、检索原文、模型请求和报告，请保留在本地。`ARTIFACT_ROOT` 可指定另一个本地目录；需要分享报告时，先单独检查报告内容。

| `OBSERVABILITY_BACKEND` | 行为 |
| --- | --- |
| `none`（默认） | 不开启额外观测会话；研究归档仍由 artifact 配置控制 |
| `local` | 在本地保存观测记录 |
| `langsmith` | 保存本地记录，并向配置的 LangSmith 项目导出 |

公开仓库只同步运行源码、配置模板与公开说明。本地测试、评测语料、实验结果、请求与追踪记录、内部设计材料、个人项目笔记及真实 `.env` 不在发布范围内。`.gitignore` 使用顶层允许列表，新增顶层目录需要显式决定是否公开。

## 代码结构

| 路径 | 用途 |
| --- | --- |
| `src/open_deep_research/deep_researcher.py` | LangGraph 研究流程 |
| `src/open_deep_research/run.py` | CLI、运行配置和图入口适配 |
| `src/open_deep_research/reviewer.py` | 独立审查与决策校验 |
| `src/open_deep_research/context.py` | 角色上下文与模型调用边界 |
| `src/open_deep_research/artifacts.py` | 证据、请求与报告的本地归档 |
| `src/open_deep_research/preflight.py` | CLI 运行前检查 |
| `src/open_deep_research/pricing.py` | 冻结价格与费用估算 |
| `src/open_deep_research/observability.py` | 运行观测 |
| `src/open_deep_research/context_policy/` | 角色策略与提示材料 |
| `src/legacy/` | 保留的早期实现源码 |
| `src/security/auth.py` | LangGraph 部署认证 |

更多结构说明见 [公开架构说明](docs/public/ARCHITECTURE.md)。原项目的课程、部署方式和研究背景见 [上游仓库](https://github.com/langchain-ai/open_deep_research)。
