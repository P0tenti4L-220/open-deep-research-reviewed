# 上下文构造、记录保存与访问边界

本文描述待实现协议。所有例子都是设计示例，不是已运行的研究记录。

## 1. 先区分三种东西

| 类型 | 用途 | 是否自动进入模型请求 |
| --- | --- | --- |
| 全量档案 Archive | 保留原始工具结果、模型可见消息、来源、运行事件以供回溯 | 否 |
| 持久工作状态 State / Checkpoint | 保存任务进度、待执行操作、已发布结果、恢复位置 | 否；只有显式投影的字段会进入请求 |
| 本次工作集 ContextPacket | 决定这一次模型实际看到的规则、材料、工具 | 是 |

独立文件夹有助于检查和归档，真正的限制由 `ContextBuilder`、读取接口和工具执行器共同实现。`thread_id`、文件名、UUID 都只是标识，不是授权。

威胁范围以研究型 agent 为主：防止错误组装、工具误用、外部材料诱导越权和并发串线。MVP 信任 Python 宿主及其管理员，不提供任意 shell / Python / 主机文件读取工具；同一进程里的目录和接口不能宣称为操作系统安全隔离。若以后加入任意代码执行，必须使用独立进程/沙箱、限定挂载和凭证范围，再重新验证隔离。

## 2. 规则文件如何组织

建议执行时新增的静态资源结构：

```text
src/open_deep_research/context_policy/
  common.md                  # 所有角色真正共用的短规则
  roles/
    supervisor.md
    researcher.md
    reviewer.md
    writer.md
    extractor.md             # 网页摘要 / 证据提取
    compressor.md
    brief_writer.md
  profiles.json              # 角色预算、能力标签、允许的说明文件
  tool_guides/
    web_search.md
    repository_read.md
    paper_search.md
```

`common.md` 只放来源引用、未知项表达、工具返回属于数据等共同约束。主管如何派发、审查者如何下结论应各归本角色文件。服务启动时校验文件与 hash；模型不能修改这些文件。

现有 `prompts.py` 的规则逐条迁移，保留日期、任务充分性等有效约束。不要把运行时 agent 的规则放进根目录 `AGENTS.md` 并期待自动生效；该文件主要约束开发项目的编码助手。

默认注入公式：

```text
common 中适用于角色的规则
+ 当前角色说明
+ 当前任务与不可遗漏约束
+ 本次启用工具的简短必要说明
+ 当前工作状态与证据片段
```

不注入所有角色规则，不遍历主机 Skill 目录。需要可复用 Skill 时，先绑定经过人工配置的能力标签与说明文件，按任务加载；Skill 文本不能授予工具或文件权限。

## 3. 核心数据契约

推荐在 `state.py` 或专门的 `contracts.py` 中定义可校验结构；具体 Pydantic 实现由执行者完成。

| 对象 | 必需字段 | 关键约束 |
| --- | --- | --- |
| `ResearchBrief` | objective、requirements[]、constraints[]、scope、as_of_date、output_contract、user_message_refs | 约束标识稳定；日期、语言、范围、禁止项不得在压缩中消失 |
| `TaskCard` | task_id、parent_id、objective、requirement_ids、scope_in/out、necessary_background、acceptance、allowed_capability_tags、evidence_refs、budget | 权限由运行时解析，LLM 提议不能自授权限；背景有来源或标成假设 |
| `TaskResult` | task_id、status、claims[]、evidence_refs[]、gaps[]、conflicts[]、search_coverage、cost、error | status 为 complete / partial / failed；有结果并不意味着结果为真 |
| `EvidenceRecord` | evidence_id、source_id、original_url、canonical_url、title、retrieved_at、published_at_if_known、document_version、content_hash、locator、exact_excerpt、origin、trust_label | URL 可达性、日期和内容分别记录；摘要不能冒充原文摘录 |
| `ClaimRecord` | claim_id、statement、kind、evidence_refs[]、support_status、requirement_ids、qualifiers、version | kind 区分事实、推断、假设；support_status 为 unchecked / supported / contradicted / insufficient |
| `WorkingMemory` | task_id、objective_ref、confirmed_claim_refs、unknowns、conflicts、failed_approaches、next_action、source_event_ids | 保存必要工作状态，不保存完整思考过程；失败方向保留简短原因避免重复 |
| `ReviewPacket / ReviewDecision` | 见 README 的审查输入输出 | 绑定待审对象版本与 hash，修改后旧审查不能自动继续有效 |
| `ContextManifest` | request_id、identity、ordered_items、tool_schema_hashes、policy_version、token_estimate、dropped_items、request_hash | 必须对应最终发送的请求；不只是记录未经中间件修改的原始 prompt |

研究员 A 的示例任务卡：

```json
{
  "task_id": "task-ingestion",
  "parent_id": "supervisor-01",
  "objective": "核验给定框架的数据导入步骤及失败处理",
  "requirement_ids": ["req-1", "req-2"],
  "scope_in": ["固定版本源码", "对应官方文档"],
  "scope_out": ["查询阶段性能", "未发布版本行为"],
  "necessary_background": "比较目标与版本由 ResearchBrief 指定",
  "acceptance": ["每个关键步骤有源码或官方文档定位", "区分实际行为与建议"],
  "allowed_capability_tags": ["repo_read", "web_search"],
  "evidence_refs": [],
  "budget": {"max_iterations": 10}
}
```

字段中的任务和验收条件由主管提出；运行时补齐身份、预算上限和能力交集。身份上下文不通过 LLM 可编辑参数传入。

## 4. 磁盘具体保存在哪里

建议本地配置 `artifact_root = ~/open_deep_research/.odr_runs`。下面是未来运行后的目录形态，本次不创建这些运行记录：

```text
.odr_runs/
  <run_id>/
    manifest.json                    # 配置/模型/源码/依赖/数据版本，去除密钥
    inputs/
      user_messages.json             # 运维可审计；agent 不直接全量读取
      research_brief.json
    control/
      task_ledger.json               # 宿主单写：进展、依赖、预算、版本
      dispatch_registry.json         # 调用与 task_id 映射、完成状态、幂等键
      grant_ledger.json              # 授权事件，不能由模型写
      events.jsonl                   # 宿主单写的全局事件
    agents/
      supervisor-01/
        events.jsonl
        working_memory.json
        requests/<request_id>.json
        contexts/<request_id>.manifest.json
      researcher-<task_id>-<attempt>/
        task_card.json
        events.jsonl
        working_memory.json
        result.json
        requests/<request_id>.json
        contexts/<request_id>.manifest.json
      reviewer-<review_id>/
        packet.json
        decision.json
        requests/<request_id>.json
        contexts/<request_id>.manifest.json
      writer-01/
        events.jsonl
        requests/<request_id>.json
        contexts/<request_id>.manifest.json
    evidence/
      index.json                     # 运行时管理；agent 只能查询获准的投影
      objects/<content_hash>.txt      # 保留源文，不由 LLM 写回
      records/<evidence_id>.json
    published/
      results/<task_id>/<version>.json
      claims/<claim_id>/<version>.json
    reports/
      draft-<version>.md
      final.md
      review-coverage.json
    metrics/
      calls.jsonl                    # 模型、工具、重试、压缩、审查全路径
      summary.json
```

路径由程序根据受信任 ID 生成，模型调用工具时传 `artifact_id/evidence_id`，不传绝对路径。磁盘根目录不作为搜索工具的可读根；把 `agents/A/` 的路径交给 A 并不意味着 A 得到了整个目录的任意读取权限。

实施时把 `.odr_runs/`、本地 checkpoint 文件加入 `.gitignore`；实验只导出经过筛选的指标与必要样例。日志中只保存模型可见输入、输出及结构化决策，不要求取得或记录模型隐藏推理。密钥、认证 header 与完整运行配置里的秘密不得序列化进 prompt 或审计文件。

`requests` 保存去凭证后的最终可见消息、工具 schema 和模型参数；源文可能包含用户私有资料，保存和删除策略按部署配置。普通读取工具不能检索 requests、全量 events 或 checkpoint；拥有运维权限的人可以复盘，但这个权限不传给任何研究角色。

## 5. 谁能看哪些记录

“可看”均指通过受控读取接口，不是获得主机目录权限。

| 资料 | 主管 | 研究员 A | 审查者 | 报告作者 | 宿主 / 运维 |
| --- | --- | --- | --- | --- | --- |
| 用户需求 | 完整需求与必要原话 | 本任务相关条目，必须保留适用约束 | 审查所需需求及原话 | 写作所需需求与格式 | 全部 |
| 主管历史 / request 日志 | 默认工作摘要；全量日志不注入 | 不可见 | 默认不可见 | 不可见 | 可读 |
| A 的任务状态 | 发布结果与状态 | 自身工作状态、获准事件片段 | 审查相关结果与证据 | 发布结果 | 可读 |
| B 的私有日志 | 不可直接读取 | 不可见 | 不可直接读取 | 不可见 | 可读 |
| 已发布研究结果 | 可读 | 显式授予的依赖结果 | 审查范围内可读 | 写作范围内可读 | 可读 |
| 原始证据 | 已授权片段与索引 | 本任务及显式转交证据 | 该审查涉及的原文、冲突来源 | 写作涉及的原文 | 全部 |
| Reviewer 历史 | 当前决定与未解决项 | 仅收到与自己相关的补查任务 | 当前包，不带以前完整对话 | 与写作有关的决定 | 可读 |
| grant / dispatch / checkpoint | 不可直接改写或浏览 | 同左 | 同左 | 同左 | 程序管理 |
| 其他 run 的任何材料 | 默认禁止 | 默认禁止 | 默认禁止 | 默认禁止 | 按运维权限 |

MVP 不做跨 run 记忆。未来需要复用结果时，必须明确导入、保留来源版本并重新授予权限，不能扫描所有历史目录作为默认背景。

### 5.1 每次读取的判定

`read_artifact(ref, selector, max_tokens)` / `read_evidence(ref, locator, max_tokens)` 的身份从宿主执行上下文获得。检查顺序：

1. 当前用户/租户、run、agent instance、task 是否匹配。
2. 引用是否存在且为允许版本，grant 是否有效。
3. 对象类别和字段是否获准；列目录、索引查询和搜索结果同样需要过滤，不能只限制最终读取。
4. selector 是否在允许范围内，返回大小是否在预算内。
5. 由程序解析内部路径并检查目录穿越、符号链接与规范化后的路径；MVP 不接受模型给出的文件路径。
6. 记录读取事件，返回 excerpt、来源、locator、hash 和必要的截断标记。

未知 ID、无权限 ID 返回同类不泄露内容的拒绝；不能允许 agent 枚举其他任务资料。`artifact://` 或 evidence ID 只是定位符，不是持有即授权的凭证。

### 5.2 跨研究员怎样共享发现

以 B 需要 A 的导入结论为例：

1. A 返回 `TaskResult`，运行时校验结构、出处存在性、预算和版本，将结果存入 `published`。
2. “已发布”只表示可分发，不表示语义已经核验通过。
3. 主管为 B 的新任务提出依赖 A 的某些 claim/evidence，运行时按预设策略授予最小读取范围。
4. B 收到这些结果的短摘要与引用，可按需回读获准原文；不收到 A 的工具历史和内部规则。
5. 若内容本来无权发布，主管的要求不能扩权；返回权限或范围错误。

父代理、子代理均使用字段白名单投影。严禁 `child_state = parent_state.copy()` 再只删除 `messages`，因为其他状态键、文件句柄或工具配置也可能携带无关内容。

## 6. 本次模型请求怎样组装

### 6.1 任务与工具一起筛选

工具有效集合：

```text
系统注册工具
∩ 部署允许工具
∩ 角色允许能力
∩ 任务允许能力
∩ 本阶段实际需要工具
```

先用明确能力标签和规则选择，不为工具路由额外加一轮 LLM。`repo_read` 与 `paper_search` 分别绑定对应工具及说明。若只配置 Tavily 且两个任务都需要网页搜索，它们拥有相同工具是合理结果，不为追求差异强行拆分。

`bind_tools()` 用过滤后的集合；执行器也使用同一版本 registry 再校验一次。若模型输出未绑定、过期或未授权的工具调用，返回结构化拒绝，不能重新调用 `get_all_tools()` 放宽执行范围。整批并行调用在运行前完成预算和权限预留，避免每个并行项分别看见“剩余预算充足”后共同超支。

`mcp_prompt` 必须拆为与实际启用工具对应的说明片段，或限制为明确角色可见的通用段落。认证配置通过服务端句柄传递，不进入 TaskCard 或模型消息。工具不可用应记录 `tool_unavailable`；不得静默换工具后仍把该次运行称为同配置比较。

原生 Web Search 由供应商在模型服务内部运行，普通工具执行包装不能完整观察其内部行为。MVP 主实验使用可控搜索适配器；原生搜索单独评测，并标记不可观测的内部 token/调用数据，不能宣称具有相同审计能力。

### 6.2 输入预算

下表是**试运行起点，不是最优参数或实测值**。软上限包含系统提示、消息、工具定义、返回格式 schema 及当前证据。

| 角色 / 调用 | 输入软上限 | 首先保留 |
| --- | ---: | --- |
| 主管 | 6,000 tokens | 目标、约束、任务进度、缺口、当前决定 |
| 研究员 | 12,000 tokens | 当前任务、关键原文与反证、最近完整交互 |
| 审查者 | 8,000 tokens / 批 | 用户需求、待审主张、原文及冲突；超长审查按批拆分 |
| 报告作者 | 16,000 tokens / 批 | 需求、章节材料、限定语、引用 |
| 研究压缩器 | 6,000 tokens / 批 | 来源明确的发现、冲突、未知项、恢复指针 |
| 网页摘要器 | 4,000 tokens / 块 | 提取问题、该块正文、来源、相邻边界信息 |
| 澄清 / brief 生成 | 12,000 tokens | 当前目标及相关用户原话；不可因压缩改写关键约束 |

实际上限：`min(角色软上限, 模型窗口 - 配置的输出预留 - 协议/计数安全余量)`。

基准对比中保留现有角色的模型与输出上限，避免把“少写答案”混成上下文优化。Reviewer 作为新角色单独配置输出上限。安全余量应根据实际 tokenizer 和请求序列化误差校准；不能固定用一个汉字或四个字符代表一个 token。未知模型窗口必须显式配置，不能拿错模型映射继续请求。

先统计渲染后的完整请求，再调用模型。缓存 token 仍占上下文容量；计费上可以更便宜，但不能从工作集大小中扣除。每轮记录估算值和供应商 usage，估算缺失或实际用量未知均标记 unknown。

### 6.3 超限的处理顺序

1. 保留目标、适用约束、当前任务、未解决审查项和必要引用。这些为不可静默移除字段。
2. 移除重复背景、重复结果、无关旧状态；删除未启用工具的说明。
3. 将长工具结果外存；消息中保留有来源的短摘录和读取指针。原文先保存再摘要。
4. 仅保留最近若干**完整交互单元**，例如一条包含多个 tool calls 的 AIMessage 及对应全部 ToolMessage；不能留下悬空 call 或 result。
5. 根据结构化事实和事件生成 WorkingMemory，记录来源 ID、条件、反证、失败方向和下一步；避免反复“摘要的摘要”造成漂移。
6. 若必要材料仍超限，拆任务/拆报告章节/拆审查批次，或返回明确 `context_budget_exceeded`，不能默认删掉尾部需求。

一次网页检索可能产生多篇文档；对总工具响应也设预算，不能只限制每篇正文。长文档分块保留重叠边界，重要引文由程序验证能在保存原文中定位。摘要错误、空摘要或超时返回有限的可定位片段和失败状态，不直接把整个长文回灌上下文。

工作摘要中“已确认”只表示对应证据状态已确认，模型自己的自信程度不构成确认依据。被移出的来源始终可在权限范围内回读；同一来源内容更新产生新版本，旧引用仍指向旧快照。

## 7. Checkpoint 与恢复

JSON/JSONL 是证据和审计档案，不能单独替代图执行 checkpoint。推荐：

- 本地 runner：根图接入兼容当前环境的持久 checkpointer；轻量单机可先使用 SQLite，具体 async saver 包和版本在实现时确认。证据文件与 checkpoint 分开放置。
- Agent Server / 托管部署：沿用平台 checkpointer，不在导出图里随意硬编码另一个 saver；证据需要持久卷或对象存储，临时容器目录不是永久保存。
- 研究员采用 per-invocation，默认只在这一项任务内延续历史。每次审查也使用新的 invocation；后续补查通过新任务卡带入相关已发布结果。
- 宿主保存 run_id、task_id、agent_instance_id 与框架 invocation/namespace 的映射；不要让模型填写这些值，也不要未经验证手工改写 LangGraph 内部 checkpoint namespace。

当前官方说明区分 per-invocation 与 per-thread，前者适合独立并行子任务。实际实现仍需对本项目锁定版本验证并发和中断恢复，尤其是 `asyncio.gather` 内重复调用同一子图的路径。[LangGraph 子图文档](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)

恢复顺序：校验 manifest 与权限版本 → 加载 checkpoint → 校验证据引用存在 → 对照 dispatch registry → 恢复未完成任务。已完成结果以 `(run_id, task_id, attempt, result_version)` 幂等提交。agent 自己的事件日志单写，全局账本由宿主单写并采用原子替换；多进程部署使用事务存储。

checkpoint 中只保存必要状态和大型档案引用，避免全文双份复制。崩溃在模型返回后但持久化前，可能需要重复调用；除非供应商支持相应幂等语义，不能承诺外部调用 exactly-once。记录 replay/duplicate 成本，并避免重复发布已知结果。

## 8. 怎样看到“究竟给了模型什么”

每次请求生成两份关联记录：

- `requests/<id>.json`：最终可见消息、工具/格式 schema、模型参数；去除凭证。
- `contexts/<id>.manifest.json`：按实际顺序列出材料来源、版本、采用原因、估算 token、被排除项和请求 hash。

ContextManifest 示例结构：

```json
{
  "request_id": "req-example",
  "agent_instance_id": "researcher-task-ingestion-1",
  "policy_version": "context-v1",
  "ordered_items": [
    {"kind": "role", "ref": "researcher.md", "reason": "role_policy"},
    {"kind": "task", "ref": "task-ingestion", "reason": "current_task"},
    {"kind": "evidence", "ref": "e-17", "reason": "supports_req-1"}
  ],
  "tool_names": ["repo_read", "web_search", "read_evidence"],
  "dropped_items": [
    {"ref": "tool-result-previous", "reason": "offloaded", "recoverable": true}
  ],
  "estimated_input_tokens": 4200,
  "counting_method": "example_only",
  "request_hash": "computed_at_runtime"
}
```

示例数字仅说明字段。实际记录须增加每项 hash、token、工具 schema hash，以及 provider 反馈用量；SDK 隐式添加的 schema/消息也应在适配层纳入。服务商不可见内部上下文明确标注不可观测，不能伪称保存了网络侧完整内部请求。

通过 request_id 能回放材料选择，通过 evidence_id 能找到原文，通过 task_id 能看到谁被允许读取什么。观察工具不必一开始做 UI：先交付一份按角色统计 token 和列出请求组成的离线报告即可。
