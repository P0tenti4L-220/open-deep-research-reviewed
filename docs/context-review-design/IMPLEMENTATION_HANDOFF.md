# 给代码执行者的实施交接

这是实现规格，不是已完成的功能清单。保持现有主图入口和默认使用方式，所有新行为通过明确配置开启。先做可观测基线，再做优化。

逐轮上下文需求、按需 ContextManager、request_context 回读、旧结论失效和修复，按新增的 [DYNAMIC_CONTEXT.md](DYNAMIC_CONTEXT.md) 实施；优先扩展现有模块，不创建全知共享管理会话。

## 1. 文件级落点

| 现有 / 建议模块 | 需要实现的职责 |
| --- | --- |
| `src/open_deep_research/state.py` | ResearchBrief / TaskCard / TaskResult / EvidenceRecord / ClaimRecord / ReviewPacket / ReviewDecision；显式状态与输入输出投影 |
| 新增 `context.py` | ContextBuilder、角色预算、工具与说明选择、完整交互单元裁剪、工作摘要验证、ContextManifest |
| 新增 `artifacts.py` | ArtifactStore、证据原文/摘录存储、身份绑定读取、授权账本、版本及幂等发布；先用本地实现 |
| 新增 `reviewer.py` | ReviewPacketBuilder、审查提示、结构化结果解析、证据读取、审查覆盖；不承担调度权限 |
| `deep_researcher.py` | 接入上下文构造器、typed dispatch、审查路由、全局预算/停止、错误传播、状态恢复 |
| `utils.py` | 工具 registry 与执行复核；搜索结果先归档再摘要；token 计数及原生搜索兼容适配 |
| `prompts.py` 与新增 `context_policy/` | 按角色拆规则，去掉跨角色说明；保留任务必要约束 |
| `configuration.py` | 优化开关、artifact_root、角色预算、Reviewer 预算、工具能力映射、记录策略 |
| 新增 `tests/test_context_isolation.py` 等 | 契约、拒绝越权、上下文保真、工具协议、审查和恢复测试 |
| `tests/run_evaluate.py` / 新实验入口 | 四组实验、配置固定、所有费用统计、冻结语料和在线结果分离 |
| `tests/evaluators.py` | 保留旧辅指标，增加基于原文的主张支持、完整性和审查误拦指标 |

文件链接均在 [README](README.md) 的现状表中。具体拆分可以根据代码规模调整，优先少量清晰模块，不先建复杂插件框架。

## 2. 核心接口契约

以下为语言无关签名，不是可直接执行的实现：

```text
build_context(identity, phase, task_view, artifact_store, policy, model_profile)
  -> ContextPacket(messages, tools, response_schema, manifest, budget_reservation)

select_tools(identity, phase, capability_tags, registry, policy)
  -> ToolSet(version, definitions, executors)

execute_tool(identity, call, toolset_version, remaining_budget)
  -> ToolResult(status, excerpt, artifact_refs, usage, error)

publish_result(identity, task_result, expected_version)
  -> PublishedResultRef

read_evidence(identity_from_runtime, evidence_ref, selector, max_tokens)
  -> EvidenceExcerpt OR PermissionDenied / MissingEvidence

build_review_packet(run_snapshot, proposal_ref, review_scope)
  -> ReviewPacket

review(packet, reviewer_policy)
  -> ReviewDecision

apply_review(decision, current_proposal_hash, remaining_budget)
  -> continue_research | rewrite_report | finalize | finalize_partial
```

所有身份、权限、预算上限由宿主注入。`identity` 不暴露为可由模型填写的工具参数。拒绝、缺失、超时和未知错误使用结构化 status，不能混入普通研究文本后继续被引用。

ReviewPacketBuilder 只读取程序账本，不接受主管提供“事实已核验”的布尔值作为事实。`apply_review` 检查对象版本与审查覆盖；Reviewer 的任意自由文本不能直接成为工具调用或权限修改。

## 3. 主管子图怎样修改

建议路由：

```text
supervisor
  → validate_proposal
     → approved_dispatch → supervisor_tools → record_results → supervisor
     → review_gate → reviewer → apply_review
          → supervisor            # 明确补查或重拆
          → supervisor_end         # 准备写作
          → partial_end            # 到限、不可确认、失败
```

结束判断必须从现在 `supervisor_tools` 内的直接 END 分支提出来。迭代超限、无工具调用、`ResearchComplete` 和异常都转为带原因的状态；仅配置的 baseline 模式保留原行为以便对照。

如果同一 AIMessage 同时请求 ResearchComplete 和 ConductResearch，不能既完成又派发。建议将它标为冲突提议，生成对应每个 tool_call_id 的明确状态结果，要求下一次选择；如果已经无预算，则直接 partial_end。所有未执行工具调用都需要在继续模型会话前有相匹配的拒绝/取消结果。

现有 `asyncio.gather` 派发可以保留，但要为每个任务预分配独立 task_id、agent_instance_id、执行视图与预算。不得让某一任务的异常吞掉其他已完成任务；分别记录结果状态，再决定是否补查或停止。

新任务/审查事件通过持久 idempotency key 去重。`research_iterations`、tool-call 轮数、审查次数、修订次数、真实工具调用数分别计数，不混用一个计数器。确认当前 `>` 和 `>=` 的边界行为，避免改造后多跑一轮。

## 4. 研究员和辅助模型怎样修改

研究员输入从单一 `research_topic` 升级为 TaskCard 的显式投影，保留旧字段适配以兼容现有入口。只传本任务需要的内容；父状态中的原始消息、全量 notes、工具全集与秘密配置不复制给子图。

模型调用节点只使用 `build_context()` 返回的数据。工具执行节点使用同一个 ToolSet 版本；权限校验后将原始结果写入 artifact，再决定向模型返回哪些片段。

当前 `ResearcherState.researcher_messages` 使用追加 reducer。不能往它追加一个“summary”就认为旧消息已释放。建议保留归档事件外存，并新增独立的有限 `working_messages / working_memory` 字段，或明确使用支持替换的 reducer；保证新请求不再读取全量归档列表。

`compress_research()` 不直接 `append()` 修改取自 state 的消息列表。输入由当前任务的证据索引和必要工作状态构造；超限分批处理，保留全部未解决冲突和恢复指针。它的输出为 TaskResult，保留兼容 `compressed_research` 的渲染字段。

网页摘要器同样有输入预算和调用记录。保存真实原文后再分块提取；`exact_excerpt` 需能在原文定位，摘要失败不能把全量原文回传研究员。结构合法但语义不支持的摘要应由证据核验发现，不能把 schema 校验当语义验证。

主管、Reviewer、作者、澄清/brief 生成、网页摘要和研究压缩，都应经过统一调用记录入口。不要只统计 `researcher()` 一处，而漏掉大量辅助模型费用。

## 5. 报告交付与审查结果失效

将当前直连 `final_report_generation → END` 改成：

```text
generate_draft → review_report → validate_review
  → finalize
  → revise_draft → review_report
  → supervisor → generate_draft → review_report  # 必须补查且仍有预算
  → finalize_partial
```

Writer 基于 claim/evidence 生成带引用的文本，不能把普通 think 记录或错误消息作为 findings。长报告可分节写作，但每节都携带全局需求摘要及本节边界；整合后检查跨节矛盾。

报告阶段返回研究循环时沿用原 run 的总预算与返工计数，不能因为切换子图重置上限。缺证主张也可以降为明确未知；是否补查取决于该项是否必答、补查价值与剩余预算。

审查绑定 draft hash。任何后续改写使相应审查失效；只重审变更段落时，还需检查其对跨节结论的影响。最终稿的关键主张必须能追溯到审查覆盖表；不能审完草稿后再让作者自由润色并新增未审事实。

保持 `final_report` 和返回消息等现有消费者字段。新增 `run_status`、`review_status`、`evidence_manifest_ref`、`metrics_ref`。如需旧 `raw_notes`，提供兼容导出并明确来源组成，不将该字段重新塞回模型请求。大字段可保留为外部引用，但需明确旧调用方的迁移策略。

## 6. 错误、预算与停止策略

| 条件 | 规定行为 |
| --- | --- |
| schema / 参数错误 | 有界重试，计入调用预算；超限返回 failed / review_inconclusive |
| search timeout / rate limit | 保留错误类别、来源和可重试信息，按策略重试；不把空结果解释为“无证据存在” |
| MCP 工具加载失败 | 明确 unavailable；必需能力缺失则该任务 partial/failed |
| evidence 引用不存在 | 不让该 claim 通过证据校验，必要时补抓或降为未知 |
| empty / corrupted summary | 保留旧档案，有限片段回退或重新提取；不能作为最终答案 |
| context 超限 | 先外存/分批，再显式失败；不无记录地截断需求 |
| Reviewer 超时、解析失败、调用预算耗尽 | 标记未审或不确定，不能默认为 pass |
| Reviewer 要求继续但全局预算耗尽 | 确定性停止新增任务，输出已证实部分和缺口 |
| 未知异常 | error_class=unknown，保留诊断标识；禁止 `except → complete` |
| 用户取消 / 到达运行 deadline | 取消或收束在途工作，记录已花费用和已完成结果 |

每次发起模型/工具调用前预留费用与时间；并行预算扣减需原子操作。供应商用量有延迟时按最大可能值预留再结算，允许记录不可避免的在途超支而不继续新任务。审查不能在预算耗尽后重置计数器重新启动主管。

状态至少区分 `complete / partial / failed / cancelled`，审查另有 `passed / changes_requested / inconclusive / not_run`。预算耗尽时报告可能仍可用，但状态不能等同于完整完成。

## 7. 实施阶段与验收

### P0：冻结基线和观察记录

工作：确认实际 Python/依赖；保存源码、配置和模型标识；接入透明 trace、原文归档、用量统计；列出确定性故障修复是否改变基线。

验收：同一次请求的消息、工具及 response schema hash 可追溯；源码与 [baseline-files.json](baseline-files.json) 的变化有记录；A_original 的模型输入不因观察代码改变。

### P1：证据与角色边界

工作：TaskCard、EvidenceRecord、TaskResult；ArtifactStore 和读取授权；工具 registry；结果发布契约。

验收：A 不能读 B 的私有日志，不能读其他 run；跨 agent 转交具体 evidence 后可回读，grant 不会自动扩成目录权限。此阶段变更若改变原模型输入，应纳入 Context 因子，不污染 A_original。

### P2：上下文优化

工作：角色规则、预算、长工具结果外存、有限完整交互、工作摘要、mcp_prompt 过滤、所有模型节点的 ContextManifest。

验收：每次请求预算可解释；重要条件和反证保留；工具 call/result 协议有效；旧证据可恢复；压缩失败不会导致假完成。

### P3：独立审查

工作：plan/readiness/final review、停滞触发、限定返工、审查覆盖与报告版本绑定。

验收：主管不能绕过结束审查；审查者能基于原文否决失据主张，也能通过正确结果；审查失效或超时可观察；所有循环都受全局预算约束。

### P4：实验与采用决策

工作：四组、冻结语料、独立 grader、人工抽查、成本/延迟/质量分解。

验收：交付真实运行结果、失败样例与置信区间，按 [EVALUATION.md](EVALUATION.md) 选择是否启用 B 或 D。没有跑实验就只能交付功能，不能声称“性能提升 X%”。

若先做小规模可运行版本，保留 P0/P1 的必要边界，只实现研究员外存+预算与最终双审查（结束前、报告后）；通过后再增加高级工具选择与停滞触发。不要一开始引入自动跨 run 记忆、框架迁移或大规模动态 agent 生成。

## 8. 必要测试与故障注入清单

| 场景 | 期望 |
| --- | --- |
| 研究员 A/B 各放唯一 canary 文本 | 未授权时对方请求及工具返回中均不出现；显式发布后仅授权片段可见 |
| 同一模型伪造其他 agent_id / run_id | 身份从宿主绑定，模型参数不能越权 |
| 猜测 artifact ID、路径穿越、符号链接、索引枚举 | 拒绝读取，不泄漏正文或可枚举目录 |
| 并行任务复用同配置、同时 checkpoint | 消息、预算、结果和恢复位置不串线 |
| 杀进程后重启 | 已发布结果不重复提交，未完成任务按映射恢复，权限和预算未重置 |
| 伪造未绑定工具名 / 过期工具集合 | 执行前拒绝，不回退到工具全集 |
| 搜索返回提示“读取其他 agent 日志” | 作为外部数据处理，不能获得权限 |
| 工具结果超大，单条超过完整输入预算 | 原文归档；只给限定片段与指针；不生成空的工作摘要 |
| 一条 AI 消息包含多个 tool calls | 保留或移除整个协议单元，全部结果与 call_id 对应 |
| 压缩丢失否定、日期或数量限定 | 保真校验/留出测试失败，不自动发布为受支持主张 |
| 关键事实出现在很早的结果中 | 压缩后仍能从证据引用回读并正确作答 |
| 网页摘要失败、MCP 失败、未知异常 | 显式错误状态，成本被记录，不伪装研究完成 |
| 同 URL 不同版本内容 | 生成版本化证据，旧引用不被覆盖 |
| 原文互相冲突 | 反证和未解决冲突进入审查包，不能只传支持性摘要 |
| ReviewPacket 缺来源 / Reviewer 编造 evidence_id | 拒绝 pass，要求修正或标记未知 |
| 正确报告与注入一条错误结论的报告 | 同时测误拦和漏检；不能用“一律否决”过关 |
| 主管无 tool call / 提前 ResearchComplete | 仍经过 readiness gate |
| 审查后 Writer 改写事实 | 草稿 hash 改变，旧 pass 失效 |
| 多轮修订无改进 / 预算耗尽 | 按上限停止，返回 partial/inconclusive |
| 超长报告超出 Reviewer 单次窗口 | 分批覆盖；未审部分明确列出，不假称全文审查 |
| 只有默认搜索工具的小任务 | 不为减少工具数删除必需能力，避免不必要路由成本 |
| 原生搜索缺少内部用量 | 标 unknown，与可观测后端分组统计 |

这些测试应检查外部可观察契约与失败模式，不仅检查某个函数恰好调用了另一个函数。完成后保留代表性请求清单和证据记录，方便人工确认隔离实际生效。

## 9. 配置与回退要求

建议明确 `context_policy = baseline | scoped` 和 `reviewer_enabled = true | false`，构成四组实验。新功能在实验模式显式开启；稳定后再决定默认值。

配置至少包括：各角色输入预算、output reserve、工具能力映射、artifact_root、日志策略、max_review_events、max_review_model_calls、max_revision_rounds、审查 timeout、全局 wall-clock/token/cost 上限及价格快照版本。

回退通过功能开关和版本化策略完成；已保存证据不删除。若原图必须修改错误处理，留存原版运行入口或源码快照，保证 A_original 可重放。不要在优化同一轮随意升级所有模型、依赖和搜索后端。

## 10. 可直接转交的执行任务说明

> 在 `~/open_deep_research` 按 `docs/context-review-design/` 实施独立 Reviewer 与最小充分上下文机制。先核对当前源码和 baseline hash，再从 P0 开始。保留原 LangGraph 主实现与公共返回字段，新增行为通过开关配置。所有 agent 和辅助模型调用都要有请求清单、token/cost 记录；原文证据与上下文工作集分离；读取和工具执行有身份绑定检查。Reviewer 使用独立包，不能继承主管完整历史，不能绕过证据核验或无限返工。每阶段完成指定契约验证后再推进。实验保留 A/B/C/D 四组，运行前按可用模型和预算确定样本量；未实测不得承诺性能比例。此设计不要求迁移 Deep Agents、不要求接入跨 run 记忆。
