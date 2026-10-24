# 路由修复实施记录（2026-10-23）

本文件记录按 `ROUTING_AUDIT_2026-10-23.md` 的“修正版方案 1”实施的结果。原则未变：

- 先删除重复路由，保留最终交付节点的单一职责；
- **没有**给 `final_report` 加覆盖型 reducer；
- 父图持有 run 级状态，子图通过同名 channel 共享；
- 每次运行只有一个 finalizer 写 `final_report`。

## 1. 实施内容

### 1.1 删除重复边

`deep_researcher.py` 中删除了 5 条重复静态边：

```
review_report -> finalize
review_report -> report_writer
review_report -> supervisor_return
review_report -> finalize_partial
supervisor_return -> research_supervisor
```

`review_report` / `supervisor_return` 只通过 `Command(goto=...)` 路由。`final_report` 仍是 `LastValue`。

### 1.2 父子图状态与预算传递

`AgentState` 增加并由父图持有：

- `research_iterations` / `max_iterations`
- `task_ledger`
- `review_history`
- `report_version`
- `report_ready`
- `writing_constraints`

`SupervisorState` 增加 `report_ready`、`run_status`、`review_status`，使研究阶段结束时的状态能穿过子图边界回到父图。

`review_history` 由追加改为按稳定 ID 合并：

- reviews / decisions / pending：`review_id`
- revisions：`(stage, review_id)`
- proposals：`proposal_id`
- `open_issues`：用最新值替换，不做历史追加

`task_ledger` 也按任务 ID 合并。这样编译后的子图返回完整 `review_history` 时，父图不会把同一审查、返工或提案重复计数。

预算（`max_total_model_calls` / `max_total_tokens`）、研究轮数、审查事件数、返工轮数、审查模型调用数都在父图/运行级累计，子图切换不会重置。`max_review_model_calls` 现在由 `ArtifactStore.calls_by_role()` 实际执行，而不再只写进 run manifest。

### 1.3 接通重写 / 补查 / 停止分支

`apply_review` 的契约：

| 阶段 | 决定 | 路由 |
|---|---|---|
| report | pass + passed + 预算未耗尽 | `finalize` |
| report | revise | `rewrite_report` |
| report | insufficient_evidence 且有研究轮数 | `continue_research` |
| report | insufficient_evidence 且无研究轮数 | `finalize_partial` |
| report | stop | `finalize_partial` |
| report | 预算/返工上限触顶 | `finalize_partial` |
| 研究阶段 | pass + passed | `finalize` |
| 研究阶段 | revise / insufficient_evidence | `continue_research` |
| 研究阶段 | stop | `finalize_partial` |

旧 draft hash 的审查不能授权交付；超时、无输出、伪造 evidence id、未检查任何 claim 的 `pass` 都会被 `validate_decision` 降级为 `insufficient_evidence`，不能变成 `passed`。

`report_writer` 在预算耗尽时不再调用 writer 模型，直接 `finalize_partial`。`review_report` 在审查事件、审查模型调用或总预算触顶时也不再调用 reviewer。

`supervisor_return` 保留同一 run、同一 store、`research_iterations`、`max_iterations`、`task_ledger` 和已消耗的预算，只重置 `report_ready`，让补查回到同一个 supervisor 子图。

### 1.4 统一模型绑定顺序

`context.assemble` 的顺序固定为：

```
具体 chat model
  -> bind_tools(packet.tools)
  -> with_structured_output(packet.response_schema)
  -> 有界重试循环（每次失败都记一次 failed_call）
  -> ainvoke
```

`clarify_with_user`、`write_research_brief`、`supervisor`、`researcher`、`review_report` 都传入未包装的配置模型和 `retries=...`，不再在 `assemble` 前套 `.with_retry()`。`RunnableRetry` 不保证 `bind_tools` / `with_structured_output`，这是审计复现的 `TypeError` 根因。

## 2. 测试

### 2.1 离线契约与流程

```
.venv/bin/python -m pytest tests/odr -q
39 passed, 1 skipped
```

跳过的是默认不运行的真实集成测试。分文件：

| 文件 | 覆盖 | 数量 |
|---|---|---|
| `tests/odr/test_graph_runs.py` | 基线/作用域/审查 A-D 路径、静态边缺失、`final_report` 为 `LastValue`、父子 channel 存在 | 6 |
| `tests/odr/test_review_flows.py` | 改写后版本递增、返工上限、补查回到同一 run、研究 STOP、预算耗尽、审查事件/审查模型调用上限、研究轮数上限 | 8 |
| `tests/odr/test_review_contracts.py` | `apply_review` 矩阵、stale hash、超时/伪造证据/未检查 claim、`review_history` 去重与 `open_issues` 替换 | 19 |
| `tests/odr/test_model_boundary.py` | 真实 `assemble` 的绑定顺序、重试逐次计数、wrapper 拒绝、图内所有模型调用可绑定 | 4 |
| `tests/odr/test_search_pipeline.py` | mock Tavily 的 search → archive → evidence → claims → review → report 全链路；超时/空结果；未构造真实 Tavily client | 2 |

搜索测试不是 no-search 路径：它替换 `tavily_search_async`，真实运行 `evidence_aware_web_search`、归档器、抽取循环、ledger、writer 和 review packet，并断言 `evidence/records/E1.json`、归档对象、claim 文件、`task_ledger`、reviewer 上下文中的 `<Evidence>` 和最终报告。

### 2.2 真实集成（单独运行）

```
RUN_REAL_INTEGRATION=1 .venv/bin/python -m pytest tests/odr/integration -q -s
1 passed
```

该测试使用 `.env` 中的真实 Tavily 和 DeepSeek，默认被 `skipif` 跳过；离线测试不导入 provider，也不访问网络。

### 2.3 全仓测试

`pytest` 收集到 40 个 odr 测试；另有 `src/legacy/tests/test_report_quality.py` 在收集阶段报 `no option named 'research_agent'`。这是该 legacy 测试自身缺少 pytest 选项定义导致的既有问题，与本次路由改动无关。

## 3. 审计快照

- `routing_audit_20261023.json`：修复前的历史快照，已恢复，保持 323 行不变。
- `routing_audit_20261023_after_fix.json`：修复后重新生成的快照。

修复后快照的关键差异：

- `channels` 中 `research_iterations`、`max_iterations`、`review_history`、`task_ledger`、`report_ready`、`review_status`、`run_status` 均为 parent + subgraph 可见；
- `final_report_channel` 仍为 `LastValue`；
- `report_actions.insufficient_evidence` 为 `continue_research`（探针现在提供 `iterations_left`）；
- `original`、`routing-only` 均不再出现 `InvalidUpdateError`，节点顺序为 `research_supervisor → report_writer → review_report → finalize`；
- `revision-caps` 只产生 2 次 report review，draft 版本为 `[1, 2]`，最终 `finalize_partial`；
- `real_assemble_wrapped_schema` 仍按预期报 `TypeError`，因为探针故意传入 wrapper；生产调用点已改为传具体模型，并由 `test_model_boundary.py` 覆盖；
- `external_network_attempts = 0`。

重新生成快照时，脚本默认写入 `routing_audit_20261023_after_fix.json`，不会覆盖修复前的历史 JSON。也可显式传路径：

```
.venv/bin/python docs/context-review-design/routing_audit_20261023.py /tmp/audit.json
```

## 4. 变更文件

- `src/open_deep_research/state.py`
- `src/open_deep_research/deep_researcher.py`
- `src/open_deep_research/reviewer.py`
- `src/open_deep_research/context.py`
- `src/open_deep_research/artifacts.py`
- `tests/odr/conftest.py`
- `tests/odr/harness.py`
- `tests/odr/test_graph_runs.py`
- `tests/odr/test_review_flows.py`
- `tests/odr/test_review_contracts.py`
- `tests/odr/test_model_boundary.py`
- `tests/odr/test_search_pipeline.py`
- `tests/odr/integration/test_real_smoke.py`
- `docs/context-review-design/routing_audit_20261023.py`
- `docs/context-review-design/routing_audit_20261023_after_fix.json`