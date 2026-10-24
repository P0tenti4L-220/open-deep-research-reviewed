# 第一轮阻塞项修复记录（2026-10-23，第二轮复核前）

对应复核报告：[ROUND1_ACCEPTANCE_REVIEW_2026-10-23.md](ROUND1_ACCEPTANCE_REVIEW_2026-10-23.md) 的 R1–R5。

本次只修这 5 项阻塞问题，没有推翻架构：路由、父图/子图共享状态、reducer 去重
全部保留。修复后又发现了 3 个复核报告未列出的缺陷，一并修掉（见 §2）。

验证方式与复核一致：离线测试 + 同一份复现探针，不请求真实模型/搜索 API。

## 1. 五项阻塞问题的处理

| 项 | 问题 | 修改后行为 | 关键位置 |
| --- | --- | --- | --- |
| R1 | 部分核验 / 引用不存在的证据仍可 `passed → finalize` | 报告阶段“完整通过”由程序判定，任一条件不满足即为 `inconclusive` | [reviewer.py](../../src/open_deep_research/reviewer.py) `validate_decision` |
| R2 | 长草稿被截断（每节 6000 字符、最多 24 节），尾部未审仍全文交付 | 草稿按连续段落切成 segment，拼接后与原文完全一致；整份进入审查，超预算则分批审查 | `reviewer.py` `_segments` / `run_review` / `trim_packet_for_budget` |
| R3 | 总调用预算没有在统一入口执行 | 每次尝试（含每次重试）先在运行级、原子地预留额度；耗尽即受控收尾，不再发请求 | [context.py](../../src/open_deep_research/context.py) `ModelCallBudget` / `assemble`；[artifacts.py](../../src/open_deep_research/artifacts.py) `reserve_model_call` |
| R4 | 重试覆盖角色计数；超时调用计数为 0 | 账本按 attempt 记账（`attempt_id`），超时/取消记一次“用量未知”，取消异常继续向上传播 | `assemble` / `_record_attempt`；`UsageRecord.attempt_id` |
| R5 | `langgraph.json` 公开入口缺少 `__odr_ctx__`，直接 `RuntimeError` | 公开入口改为宿主适配层：宿主按调用方 thread 绑定身份、store 与图，纯 JSON 输入可用 | [run.py](../../src/open_deep_research/run.py) `PublicEntry`；`deep_researcher.py` 导出 `deep_researcher` |

### R1：报告阶段的交付门禁

`build_review_packet` 现在由程序算出交付需要覆盖的范围，`validate_decision` 逐项核对，
只有全部满足才允许 `passed`（否则 `inconclusive`，并给出可执行 finding）：

- `delivery_required_claim_ids`：报告阶段要求全部 claim 被检查；
- `missing_evidence_ids`：claim 引用了从未归档的证据 id → 直接阻断；
- `coverage_matrix` 加 `available_evidence_ids`：强制要求必须由**实际存在**的证据覆盖，
  或由审查者用 `acknowledged_unknown_requirement_ids` 明确承认为“未知”；
- `unbound_assertions`：段落有事实断言却不引用任何 claim 或 `[E<n>]` → 阻断；
- `scope_issues`：报告无 claim 清单、草稿为空 → 阻断（“没有 claim”不等于“无需审查”）；
- 未读 segment / 未核 requirement 直接阻断。

`apply_review` 另做一层防御：即使 verdict 为 `pass`，只要 decision 仍记录未覆盖范围，
也只允许 `finalize_partial`，不得 `finalize`。

Writer 的输入也加了归档状态行（本次归档了多少条证据、多少条 claim，必须用真实的
`[E<n>]` 引用，否则写“Unknown: …”），让“无证据的报告”在生成阶段就暴露。

### R2：全文覆盖

- `_segments()` 按段落边界切分，`"".join(seg.text) == draft` 由测试断言；
  超过 segment 上限时仍然输出全部 segment（标 `within_budget=False`），不丢字符；
- 审查包以 `<DraftSections><Segment id="S1">…` 呈现整份草稿；
- 一次请求装不下时分批审查，合并时取保守结论；审批要求**每个** segment 都被读过；
- `trim_packet_for_budget` 只压缩证据摘录，草稿 segment 若被移出必须按 id 列出并记为未读。

### R3：预算在统一入口执行

- `ModelCallBudget.reserve()` → `ArtifactStore.reserve_model_call()`：在同一个锁内完成
  “检查 + 写入”，并行研究员 / 重试 / 辅助调用不能同时花掉最后一笔额度；
- 授权即计数：超时或崩溃的在途请求也算一次（不会“免费”）；
- 未知用量按“保守输出预留”计入 token 上限；
- 预算耗尽抛 `BudgetExhausted`，各节点转为受控收尾（`budget_stop` 节点），
  主管派发不会伪造工具结果；
- 辅助调用（澄清、需求抽取）只在还有 3 次以上余量时运行，并走同样的入口；
- 显式 `0` 视为硬上限（“一次都不许调用”），空字符串视为未设置。

### R4：按尝试记账

- 账本键为 `attempt_id`（`req-…#a1`、`#a2`…），`calls_by_role()` 与全局计数一致；
- 超时/取消记录 `counting_method="timed_out"`、`unknown_usage_calls` 递增，
  `asyncio.CancelledError` 原样抛出，不再被 `except Exception` 吞掉；
- `metrics/usage.json` 持久化逐次尝试与预留，重启后角色计数一致（有测试）。

### R5：公开入口

- `langgraph.json` 仍指向 `deep_researcher.py:deep_researcher`，但导出对象改为
  `run.PublicEntry` 适配层；
- 宿主在一次调用内完成 `run.make_config()`：由调用方 `thread_id` 推导 run id、创建
  store、绑定图与身份，然后调用对应编译图；
- 客户端在 `configurable` 里伪造的 `__odr_ctx__` / `__odr_graph__` 一律忽略；
- run 的图绑定写入 `control/graph_binding.json`，恢复同一 thread 时沿用同一张图，
  预算和身份不重置；
- 显式 `0` 额度会在构造任何 provider 客户端之前停下。

## 2. 修复过程中新发现并修复的问题

1. **研究阶段审批永远无法放行**（`deep_researcher.py` `apply_review_node`）：
   原实现传 `current_hash=""`，而“文本未变”判断是 `decision.draft_hash and current_hash and ...`，
   空哈希使条件恒假，于是 `pass` 被判为过期 → 一直 `continue_research` → 最终
   `finalize_partial`。修复后使用被审文本的哈希，正常 `pass` 可进入报告阶段。
   **这也意味着复核报告里“39 passed”时，审批放行路径实际上没有被任何测试覆盖。**
2. **研究阶段的 Reviewer 看不到任何 claim**：`reviewer` 节点用证据的 `_owner_agent`
   规则过滤 claim，但 `write_claims` 写入的是研究者实例 id，永远匹配不上。
3. **公开入口的 state 更新被整包替换**：适配层原先以 `dict` 作 state schema，内层节点
   返回的更新会替换整个 state（只保留最后一次更新）。现在按 `AgentState` 的 channel 声明
   `EntryState`，恢复合并语义。
4. **环境变量空字符串被当作 0/False**：`Configuration.from_runnable_config` 遇到 `.env`
   中空值会把 `max_total_model_calls` 变成 `0`、`reviewer_enabled` 变成 `False`。
   现在空串视为未设置。

## 3. 测试

```text
.venv/bin/python -m pytest tests/odr -q          → 66 passed, 1 skipped
.venv/bin/python -m pytest -q                    → 66 passed, 1 skipped, 1 error（legacy fixture，历史问题）
```

新增/扩充的回归测试：

| 文件 | 覆盖 |
| --- | --- |
| `tests/odr/test_delivery_gate.py` | R1/R2 全部验收场景（部分核验、缺失证据、空 claim、未引用断言、第 6001 字符后 / 第 25 节 / Writer 新增断言、segment 完整性、伪造 id） |
| `tests/odr/test_model_boundary.py` | R3/R4（预算不超发、并发原子性、未知用量、超时计数、重启一致） |
| `tests/odr/test_public_entry.py` | R5（`langgraph.json` 入口、纯输入、Reviewer 开关、伪造上下文、续跑、0 额度、空环境变量） |
| `tests/odr/test_delivery_gate.py`（末尾） | 审查额度耗尽记为预算停止，而不是供应商失败 |

原有测试的修改理由（都是把“未核验也能通过”的期望改成新契约）：

- `test_review_flows.py`：示例草稿不引用任何已归档证据，按 R1 只能有限交付；
  重入研究改成真实断言（同 run、同 store、再次研究）。
- `test_search_pipeline.py`、`test_graph_runs.py`：审查脚本改为明确声明已核 requirement /
  segment（这正是 R1 要求审查者说清楚的）；不引用证据的示例报告改为有限交付断言。
- `test_model_boundary.py`：事件字段 `attempt` 改为 `attempt_id`。

## 4. 复现探针结果（修复后）

探针已改为“期望 / 观测 / 是否满足”的通过判定式，结果保存在
[round1_acceptance_probe_20261023_after_fix.json](round1_acceptance_probe_20261023_after_fix.json)（**6/6 通过**）。
历史文件 `round1_acceptance_probe_20261023.json` 保留修复前结果，未覆盖。

```bash
.venv/bin/python docs/context-review-design/round1_acceptance_probe_20261023.py /tmp/round1-probe-after.json
```

| 探针 | 修复前 | 修复后 |
| --- | --- | --- |
| R1：C1 已查、C2 未查且 claim 引用 E-missing | `passed` / `finalize` | `inconclusive` / `continue_research` |
| R2：第 6100 字符后的未引用断言 | 审查者看不到（6000 字符截断） | 审查者看到完整草稿；未引用断言 → 不交付全文 |
| R3/R4：剩余 2 次额度、重试 3 次 | 3 次尝试全部发出，角色计数 1 | 发送 2 次后 `BudgetExhausted (3/3)`；角色计数 2，账本 3 行 |
| R4：一次超时 | 计数 0 | 计数 1，`unknown_usage_calls=1` |
| R3：`max_total_model_calls=1` 完整图 | 3 次调用 | 1 次调用 |
| R5：公开入口普通调用 | `RuntimeError` | 正常执行并选中 reviewer 图 |

## 5. 仍未完成（不属本轮阻塞项）

- **真实付费集成没有重跑**：`RUN_REAL_INTEGRATION=1` 的冒烟测试断言仍偏弱（只断言非空和
  metrics 存在），下一轮应补“搜索事件、归档证据、claim 链、审查覆盖、目标答案”。
  本次修复后，任何报告若要 `complete`，都必须有真实归档证据与覆盖记录，这一点尚未在
  真实供应商下验证。
- **无 store 时没有运行级计费**（`artifacts_enabled=False`）：此时 `ModelCallBudget.strict`
  为 False，预算不生效，也不对外宣称强制上限。reviewer 图必须开启 store。
- **`max_segments` 之外的草稿**仍可通过（全部分批）审查，但会消耗更多审查调用；
  受 `max_review_events` / `max_review_model_calls` 约束，属于已知行为。
- legacy 测试的 fixture 错误（`--research-agent` 未注册）未修，它早于本轮改动。
- 未运行 ruff / mypy（环境未安装），只做了语法编译与测试。
