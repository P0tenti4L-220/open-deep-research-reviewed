# 以 Open Deep Research 为样本：Agent 项目的文件与目录结构拆解

本文的参考对象是本仓库的主实现 `src/open_deep_research/`。文中每一节都标注了对应的源码位置，便于逐条核对。
它要回答的是三个问题：这些文件为什么这样分、哪些是"必需品"、`CLAUDE.md` 在这套体系里到底扮演什么角色。

---

## 0. 先分清两类文件：运行期产物 vs 开发期脚手架

这是读这类项目时最容易混淆的一点。同一个根目录里，实际上叠了两套完全不同的目录逻辑。

| 类别 | 面向谁 | 生命周期 | 本仓库例子 |
| --- | --- | --- | --- |
| **运行期产物**（Runtime） | agent 本身、部署平台、审计人 | 每次 run 生成，可回放 | `langgraph.json`、`.env`、`.odr_runs/`、`context_policy/` |
| **开发期脚手架**（Dev） | 改代码的人 / 编码 agent（Claude Code、Codex 等） | 随仓库提交，长期存在 | `CLAUDE.md`、`AGENTS.md`、`README.md`、`docs/`、`.github/`、`tests/` |

很多人问"agent 的目录该长什么样"，其实问的多半是第二类；而真正决定 agent 行为的，是第一类。
两类之间没有自动联系：`CLAUDE.md` 改了不会改变 agent 的提问方式，`prompts.py` 改了才会。

---

## 1. 一套完整的 Agent 目录骨架

按"职责单一 + 依赖单向"来分，一个成熟的 agent 项目通常长这样（括号内标出本仓库对应文件）：

```
repo/
├── langgraph.json              ① 部署清单：入口在哪、依赖什么
├── pyproject.toml              ① 依赖、构建、lint 规则
├── .env.example                ① 配置模板（不含密钥）
├── .gitignore                  ① 产物、密钥、缓存不进仓库
│
├── src/<pkg>/
│   ├── <graph>.py              ② 编排：节点、边、路由
│   ├── state.py                ③ 契约：状态与结构化输出模型
│   ├── configuration.py        ③ 配置：模型、预算、开关
│   ├── prompts.py              ④ 提示词：角色规则正文
│   ├── utils.py                ⑤ 工具：搜索、抓取、解析
│   ├── context*.py             ⑥ 上下文组装（进阶）
│   ├── artifacts*.py           ⑦ 持久化 / 证据库（进阶）
│   └── <policy_dir>/           ⑥ 需要热改或按角色拆分的提示词资源
├── src/security/auth.py        ⑧ 鉴权
├── tests/                      ⑨ 离线评测 + 脚本化模型
├── docs/                       ⑩ 设计文档
└── CLAUDE.md / AGENTS.md       ⑪ 编码 agent 的仓库记忆
```

下面逐个说明"它承担什么"以及"没有它会怎样"。

---

## 2. 逐层拆解（附本仓库对应文件）

**来源标注约定**：带 **【新增】** 的条目是本地新增模块，不在 [baseline-files.json](context-review-design/baseline-files.json) 的 26 个基线文件里；未标注的条目属于基线（上游）结构。

### ① 部署清单层：平台怎么找到你的 agent

- [langgraph.json](../langgraph.json) — 只有 15 行，却是**唯一**告诉平台"哪个 Python 符号是 agent"的地方：
  ```json
  "graphs": { "Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher" },
  "dependencies": ["."], "env": "./.env",
  "auth": { "path": "./src/security/auth.py:auth" }
  ```
  注意 `:deep_researcher` 是模块级的编译结果，不是函数调用——图在 import 时就建好了。
- [pyproject.toml](../pyproject.toml) — 依赖 + `[tool.ruff]`。这里还有一段容易被忽略的"打包映射"：
  ```toml
  [tool.setuptools.package-dir]
  "open_deep_research" = "src/open_deep_research"
  ```
  没有它，部署时 `context_policy/*.md` 这类**非 Python 数据文件**不会进环境，运行期直接 `Missing context policy resource`。
- [.env.example](../.env.example) — 列出全部环境变量但值留空。它同时是一份**能力清单**：看有哪些 key，就知道这个 agent 依赖哪些外部服务。
- [.gitignore](../.gitignore) — 把 `.env`、`.odr_runs`、`.langgraph_api` 排除在外，保证"密钥和运行产物不进仓库"。

**判断标准**：这一层是必需的。没有入口声明，平台不知道跑什么；没有依赖声明，别人的机器跑不起来。

### ② 编排层：agent 的"大脑皮层"

- [deep_researcher.py](../src/open_deep_research/deep_researcher.py)（2056 行）— 所有节点在这里：`clarify_with_user` → `write_research_brief` → `research_supervisor`（内含 `supervisor` / `supervisor_tools` / `validate_proposal` / `reviewer` 子图）→ `report_writer` → `review_report` → `finalize`。
- [graph_switch.py](../src/open_deep_research/graph_switch.py) — 基线图与带审查的图是**两张**编译好的图，run 级绑定，不允许模型中途切换。
- [run.py](../src/open_deep_research/run.py) — 宿主入口：创建 ArtifactStore、选图、绑定身份。注释写得很清楚——*"Model output never influences any of these values."*

**判断标准**：必需。这是唯一"执行逻辑"，其它文件都是它的输入或输出。

**设计要点**：编排层只做三件事——决定调用谁、传什么、何时停。一旦它开始写具体业务文字（比如报告模板），说明边界破了。

### ③ 契约层：state.py + configuration.py

- [state.py](../src/open_deep_research/state.py)（630 行）承担两类契约：
  - LangGraph 的 `AgentState` / `SupervisorState` / `ResearcherState`（`MessagesState` + `Annotated[List, operator.add]` 决定消息如何累积）；
  - **模型可见结构**：`ConductResearch`、`ResearchComplete`、`ClarifyWithUser`、`ResearchQuestion`、`BriefExtraction` 等 Pydantic 模型。文件头注释点明了一个安全约束：*"Every contract model sets `extra="forbid"`：a model-facing tool argument cannot smuggle fields (identity, permissions, budgets) that the host did not provide."*
- [configuration.py](../src/open_deep_research/configuration.py)（400 行）— 所有可调参数集中一处：模型名、`max_concurrent_research_units`、各角色上下文预算、`max_review_events`、`experiment_arm`。三个细节值得学：
  - `from_runnable_config` 优先级是 `环境变量 > configurable > 默认值`（[configuration.py:369](../src/open_deep_research/configuration.py)）；
  - `Field(metadata={"x_oap_ui_config": ...})` 直接把参数暴露成 LangGraph Studio 里的表单控件；
  - `IDENTITY_CONTEXT: ContextVar` 用来在工具内部调用模型时继承宿主身份——身份不能来自模型输出。

**判断标准**：必需。哪怕是最小的 agent，也至少需要一个 `state.py` 等价物（TypedDict 就够了）。

### ④ 提示词层：两种存放哲学，本仓库**两种都用了**

这是全篇最关键的一节，因为它直接关系到你的问题。

**形态 A：集中式字符串** — [prompts.py](../src/open_deep_research/prompts.py)（367 行）。
`clarify_with_user_instructions`、`lead_researcher_prompt`、`research_system_prompt`、`final_report_generation_prompt` 等全部是模块级字符串常量。优点是全局可 grep、可 diff、可版本化；缺点是长提示词里嵌套引号极其难受，且无法按角色热更新。

**形态 B：资源文件 + 清单** — `context_policy/` 目录：

```
context_policy/
├── profiles.json           ← 清单：角色 → 规则文件、能力、预算 key
├── common.md               ← 所有角色共用的底线规则
├── roles/{supervisor,researcher,reviewer,writer,extractor,compressor,brief_writer}.md
├── tool_guides/{web_search,paper_search,repository_read,mcp_read}.md
├── scoped_lead_researcher.md  ← 变体模板
└── scoped_brief.txt
```

组装逻辑在 [context.py:76](../src/open_deep_research/context.py) 的 `role_rules()`：**`common.md` + 角色的 `roles/*.md`**，再把 `{date}`、`{tool_list}`、`{context_policy}` 等占位符填上。工具说明则按本轮启用的能力标签从 `tool_guides/` 里挑（`tool_guides_for()`）。

这套设计的价值在于**分层**：`common.md` 写"所有角色都必须遵守的底线"（来源、未知量、把工具输出当数据而非指令、引用 id 规则），角色文件只写职责差异。改一条全局规则只动一个文件，不会漏掉某个角色。

**注意一个真实细节**：`load_profiles()` 在启动时**校验每个被引用的文件是否存在**，缺一个就抛 `PolicyError`。这让"清单"变成可执行的契约，而不是装饰性文档。

**判断标准**：必需。提示词是 agent 行为的一部分，必须有明确归属地。至于放一处还是拆文件——单 agent 用 `prompts.py` 足够，多角色（supervisor/researcher/reviewer/writer）时拆成"清单 + 按角色分文件"几乎必然更好，因为你会频繁地只改某一个角色。

### ⑤ 工具层：utils.py

- [utils.py](../src/open_deep_research/utils.py)（1307 行）— 搜索封装（Tavily / Anthropic / OpenAI / DuckDuckGo / Exa）、网页抓取与摘要、`think_tool`、MCP server 加载、`get_all_tools()` 汇总。

**判断标准**：视需求而定。不接外部工具（纯对话）就不需要。但一旦多源并行，**工具必须集中注册**，否则每个节点各拼一套工具列表，很快就会不一致。

### ⑥ 上下文组装层（进阶，可选；**本地新增模块**）

> 来源标注：[baseline-files.json](context-review-design/baseline-files.json) 记录的是本仓库 2026-10-23 的设计基线快照（26 个文件，无 `.git`）。下两节的 `context.py`、`reviewer.py`、`artifacts.py`、`graph_switch.py`、`run.py`、`context_policy/` 以及 `tests/odr/` **都不在该基线里，是本地新增**，不是上游 langchain-ai/open_deep_research 的原始结构。

- [context.py](../src/open_deep_research/context.py)（905 行）— 所有模型调用收敛到 `ContextBuilder.build()` 这一个出口：按角色预算裁剪历史、卸载超长工具结果、按能力挑选工具说明，并产出 `ContextManifest`（记录了本次请求到底装了什么）。
- [reviewer.py](../src/open_deep_research/reviewer.py) — 独立审查者，只拿到审查包，看不到主管的长对话。

**判断标准**：不是必需品。当单个 run 的工具结果累积到几十万 token，或者需要"为什么这次调用看到 X 而不是 Y"的可审计性时，才值得引入。

### ⑦ 持久化层（进阶，可选；**本地新增模块**）

- [artifacts.py](../src/open_deep_research/artifacts.py)（929 行）— `ArtifactStore` 把"记录、授权、进入请求"三件事拆开（文件头 docstring 就是这句话）。落盘结构：

  ```
  .odr_runs/<run_id>/
  ├── manifest.json         ← 本次 run 的模型、限制、策略版本
  ├── evidence/objects/     ← 原始证据
  └── agents/<agent_id>/    ← 各 agent 私有目录
  ```

  `.odr_runs/` 被 gitignore，是运行产物而非源码。

### ⑧ 安全层：src/security/auth.py

[src/security/auth.py](../src/security/auth.py)（155 行）— Supabase JWT 校验，被 `langgraph.json` 的 `auth.path` 引用。它独立成目录而不是塞进主包，是因为它是**部署层关注点**，不是 agent 逻辑。

### ⑨ 测试层：tests/

- [tests/odr/conftest.py](../tests/odr/conftest.py) — 用 `ScriptedRuntime` 替换**唯一**的模型调用出口（`context.assemble`），而不是伪造模型类。这样生产路径(request 组装、manifest 记录、结构化输出处理)仍然被真实执行，"no network and no real model is used"。
- `tests/evaluators.py`、`tests/run_evaluate.py` — 评测数据集 + 打分标准（9 项质量准则）。

**判断标准**：必需。"能在没网的情况下跑通一条链路"是 agent 项目质量的分水岭。

### ⑩ 文档层：docs/ 与 examples/

[docs/context-review-design/](../docs/context-review-design/) 是设计交接文档，六篇分工明确（为什么改 / 每个角色看到什么 / 怎么测 / 接口契约 …）。[examples/](../examples/) 放参考产出样本。两者都不是运行依赖，但 `README.md` 里的链接会指向它们。

### ⑪ 编码 agent 的仓库记忆：CLAUDE.md / AGENTS.md

见下一节。

---

## 3. `CLAUDE.md` 到底是什么？

**结论：`CLAUDE.md` 不是给"被部署的 agent"看的上下文，而是给"帮你写这个 agent 的那个人/那个编码 agent"看的仓库说明书。**

### 3.1 证据

- 本仓库根目录的 [CLAUDE.md](../CLAUDE.md) 和 [AGENTS.md](../AGENTS.md) **几乎完全相同**，`diff` 只有一行：
  ```
  < - `CLAUDE.md` - Legacy-specific Claude instructions
  > - `AGENTS.md` - Legacy-specific Codex instructions
  ```
  内容本身是在描述"这个项目有哪些目录、怎么跑测试、有哪些依赖"。这明显是写给读代码的 AI 的，不是在描述 agent 的调研行为。
- 它不在 `langgraph.json`、不在 `pyproject.toml`、不被任何 Python 模块 `import` 或 `read_text()`。**运行期完全不读取**。
- 对照检查：全局唯一的系统提示词常量清单在 [prompts.py](../src/open_deep_research/prompts.py)，全部是 `Python 字符串常量`；真正的 agent 规则在 `context_policy/`。`CLAUDE.md` 不在其中。

### 3.2 同一项目里"三种同名文件"的职责区分

这是最便于记忆的对照表：

| 文件 | 消费者 | 是否进模型上下文 | 回答的问题 |
| --- | --- | --- | --- |
| `context_policy/roles/supervisor.md` | 被部署的 agent（主管） | **是**，每次跑都进 | "作为调研主管，我该怎么委派任务？" |
| `docs/context-review-design/*.md` | 改代码的人 | 否 | "为什么这么设计？" |
| `CLAUDE.md` / `AGENTS.md` | 帮你改代码的编码 agent | 否（但编码 agent 会把它读进**自己的**上下文） | "这个仓库怎么建、怎么测、别踩哪些坑？" |

一句话：**前者是 product context，中者是 design rationale，后者是 coding agent memory。**

### 3.3 那么"固定的给 agent 看的 context"长什么样？

用户直觉里"CLAUDE.md 是给 AI 看的固定指令"——这个直觉对了一半。确实存在一份固定的、每次调用都注入的文本，但它在这个仓库里是：

1. `context_policy/common.md`（所有角色共用的底线）
2. `context_policy/roles/<role>.md`（角色差异）
3. `prompts.py` 里的模块级常量（基线模式）
4. `state.py` 里 Field description（工具调用的参数说明）

而且**它不是完全固定的**：占位符（`{date}`、`{tool_list}`、`{context_policy}`、`{max_concurrent_research_units}`）在运行期填值，`role_rules()` 还会按能力标签追加工具说明。所以更准确的说法是——**"结构固定、内容按时装配"的上下文合同**。

---

## 4. 哪些文件是"必须"的

按"删掉就静默失效"的程度排序：

| 必需程度 | 文件 | 删掉会怎样 |
| --- | --- | --- |
| 硬必需 | 入口声明（`langgraph.json`）+ 依赖（`pyproject.toml`） | 平台找不到 agent / 装不上 |
| 硬必需 | 编排文件（`deep_researcher.py`） | 没有执行逻辑 |
| 硬必需 | 状态定义（`state.py` 或等价 TypedDict） | 图无法传递数据 |
| 硬必需 | 提示词（`prompts.py` 或 `context_policy/`） | agent 行为退化成裸模型 |
| 硬必需 | 配置模板（`.env.example`）+ `.gitignore` | 协作者无法复现 / 泄露密钥 |
| 强烈建议 | 测试（`tests/`） | 改动无法验证，回归靠肉眼 |
| 强烈建议 | 配置模块（`configuration.py`） | 参数散落各处，无法做实验 |
| 可选 | 上下文组装（`context.py`）、持久化（`artifacts.py`）、鉴权（`auth.py`） | 规模小/不部署时不需要 |
| 可选 | `docs/`、`examples/`、`CLAUDE.md`、`AGENTS.md` | 不影响运行，影响可维护性 |

注意"硬必需"里没有 `CLAUDE.md`。它在**开发期**是加分项，在**运行期**是零影响。

---

## 5. 可以做的最小骨架

如果你想从零搭一个同类 agent，下面这个规模就够跑通一条完整链路：

```
my_agent/
├── langgraph.json
├── pyproject.toml
├── .env.example / .gitignore
└── src/my_agent/
    ├── graph.py            # 节点与边
    ├── state.py            # AgentState + 模型可见的结构化输出模型
    ├── configuration.py    # 模型名、预算、开关
    ├── prompts.py          # 最初的提示词（单文件）
    └── tools.py            # 工具注册
tests/
└── test_smoke.py           # 脚本化模型，离线跑通整图
```

**演进路径**（与"什么时候该拆"对应）：

1. 角色从 1 个变 2 个以上 → `prompts.py` 拆成 `prompts/` 目录或 `context_policy/roles/*.md` + `profiles.json`
2. 出现"为什么这次调用看到 X"的追问 → 加 `context.py`（统一模型调用出口 + manifest）
3. 需要审计、回放、授权 → 加 `artifacts.py`
4. 要部署给外部用户 → 加 `security/auth.py`
5. 团队开始问"这仓库怎么跑" → 加 `CLAUDE.md` / `AGENTS.md`，**并且让它与 `README.md` 保持同步**

---

## 6. 三个容易踩的坑（本仓库里都能看到痕迹）

1. **文档与代码会漂移**：`CLAUDE.md` 与 `AGENTS.md` 仍写着 `src/legacy/` 下有一个同名文件——但实际目录里是 [src/legacy/CLAUDE.md](../src/legacy/CLAUDE.md)，两份根文档里没有对应文件；同时两份都没提到已经新增的 `reviewer.py`、`context.py`、`artifacts.py`、`context_policy/`。这说明**脚手架文档不自动更新**——它是手写的，必须有人看管。
2. **数据文件要显式打包**：`[tool.setuptools.package-dir]` 那段，漏了就是运行期崩溃。
3. **身份与授权不能来自模型**：`state.py` 的 `extra="forbid"`、`run.py` 的注释、`IDENTITY_CONTEXT` 的 ContextVar —— 三处独立证据，说明设计者在"模型输出 ≠ 可信输入"这一点上是刻意的。

---

## 附 1：基线 vs 本地新增（用于核对"哪些是原先就有的"）

证据来源：[baseline-files.json](context-review-design/baseline-files.json)（设计交接稿生成的快照，共 26 个文件，`git_present: false`）。用它做 SHA-256 比对，可以区分三种状态：

| 状态 | 文件 |
| --- | --- |
| **基线内、未改动** | `langgraph.json`、`pyproject.toml`、`.gitignore`、`uv.lock`、`src/security/auth.py`、`src/open_deep_research/prompts.py`、`src/legacy/*`、`tests/{evaluators,prompts,run_evaluate,pairwise_evaluation,supervisor_parallel_evaluation,extract_langsmith_data}.py` |
| **基线内、已被修改** | `src/open_deep_research/configuration.py`、`deep_researcher.py`、`state.py`、`utils.py` |
| **基线外、本地新增** | `src/open_deep_research/{context,reviewer,artifacts,graph_switch,run}.py`、`src/open_deep_research/context_policy/**`、`tests/odr/**`、`docs/context-review-design/**` |

推论：**⑥（`context.py`）、⑦（`artifacts.py`）原先是完全没有的**，`reviewer.py`、`graph_switch.py`、`run.py`、`context_policy/` 同理；原先是"单文件编排 + 集中式 `prompts.py`"，新增后变成"编排 / 上下文 / 审查 / 持久化 / 角色提示词资源"五层分离。

需要说明的边界：

- `baseline-files.json` 自身生成于 2026-10-23 的设计阶段，**不能证明**这 26 个文件就是上游 langchain-ai/open_deep_research 的某个 commit——它只是本地某一时刻的快照。
- `CLAUDE.md`、`AGENTS.md`、`.env.example`、`README.md` 等**不在基线清单中**，但它们是上游仓库自带的文件（`README.md` 里有 2025-07~08 的上游更新日志），只是生成快照时被跳过了。因此"不在清单里"≠"你新写的"。新增模块的文件时间戳（2026-10-23 15:13–15:46）与基线快照同一时段，与上述判断一致。

---

## 附 2：本文引用的源码位置

| 主题 | 位置 |
| --- | --- |
| 入口声明 | [langgraph.json](../langgraph.json) |
| 图拓扑 | [deep_researcher.py:1249-1275](../src/open_deep_research/deep_researcher.py)、[:2008](../src/open_deep_research/deep_researcher.py) |
| 状态与契约 | [state.py](../src/open_deep_research/state.py) |
| 配置优先级 | [configuration.py:369](../src/open_deep_research/configuration.py) |
| 基线提示词 | [prompts.py](../src/open_deep_research/prompts.py) |
| 角色化提示词 | [context.py:76](../src/open_deep_research/context.py)、[profiles.json](../src/open_deep_research/context_policy/profiles.json) |
| 上下文组装 | [context.py:422](../src/open_deep_research/context.py) |
| 落盘结构 | [artifacts.py:336](../src/open_deep_research/artifacts.py) |
| 离线测试 | [tests/odr/conftest.py](../tests/odr/conftest.py) |
