# LangGraph 路由、状态与测试核查

核查日期：2026-10-23（Asia/Shanghai）。目录：`~/open_deep_research`。

## 结论

选择“最终报告由父图交付节点负责”的方向，但当前报错首先应通过删除重复路由解决，尚无证据要求为了它重构预算逻辑。不要给 `final_report` 添加取最后一次更新的 reducer 来隐藏冲突。

本次只增加核查文档、诊断脚本和结果；没有修改 `src/` 或现有 `tests/`。候选路由改动仅在 Python 内存中的图构建器生效。当前目录没有 `.git`；核查结果包含相关源码 SHA-256，作为本次证据的版本边界。

> **2026-10-23 实施更新**：本文件的修复已按“修正版方案 1”执行，实施记录与测试结果见 [FIX_REPORT_2026-10-23.md](FIX_REPORT_2026-10-23.md)。本文件保留核查时的原始结论和数据；修复前快照为 `routing_audit_20261023.json`，修复后快照为 `routing_audit_20261023_after_fix.json`。

## 已实测的结果

环境：Python 3.11，LangGraph 1.2.9，langchain-core 1.4.8，pytest 9.0.3。

| 检查 | 结果 | 能证明什么 |
| --- | --- | --- |
| 原 `tests/odr/test_graph_runs.py` | 1 passed / 1 failed；`InvalidUpdateError: final_report` | 报告中的失败可以复现 |
| 在内存中移除五条重复出边，再跑同一测试文件 | 2 passed | 这些出边足以解释当前失败；不代表实现全部完成 |
| 原图更新流 | `review_report` 后出现 `finalize`、`finalize_partial`、`report_writer`、`supervisor_return` | 互斥分支被错误并行启动 |
| 仅修路由，报告连续要求两次重写；审查和修订上限均设为 1 | 实际进行了 3 次报告审查，然后交付 | 现有上限不能约束该路径 |
| 调用真实 `context.assemble()`，传入与 brief 节点同形的模型包装 | `TypeError: ContextPacket carries a response schema but the model cannot produce structured output` | 之前的模型包装修复尚未覆盖所有调用点 |
| 上述离线检查的外部网络连接尝试 | 0 | 不依赖 Tavily 或真实模型可验证这些缺陷 |

复现入口：

```bash
rtk proxy .venv/bin/python docs/context-review-design/routing_audit_20261023.py
```

脚本运行原图、内存候选图和修订上限探针，记录更新流、实际通道、版本、源码摘要和调用边界错误，输出到同目录的 `routing_audit_20261023_after_fix.json`（修复前快照保留为 `routing_audit_20261023.json`，不要覆盖）。脚本的图运行沿用当前 scripted fixture，因此其成功结果不覆盖真实模型协议；独立的 `assemble` 探针才会调用真实组装入口。

## 1. 当前写冲突的直接根因

`deep_researcher.py:1808–1814` 已通过 `Command(goto=...)` 返回四选一结果。`build_deep_researcher()` 的 2038–2041 行又给 `review_report` 注册四条无条件边。这些边会一起生效，而不会被 `Command` 取消。

因此，当前明确的两个 `final_report` 写入者是 `finalize`（1870 行）和 `finalize_partial`（1894 行）。`supervisor_return`（1817 行）本身不写这个字段。将本次冲突直接归因于“预算沿用导致父子图抢写”不符合实际更新流。

最小路由修复是删去以下五条边，保留已有 `Command` 决策：

```diff
-    builder.add_edge("review_report", "finalize")
-    builder.add_edge("review_report", "report_writer")
-    builder.add_edge("review_report", "supervisor_return")
-    builder.add_edge("review_report", "finalize_partial")
-    builder.add_edge("supervisor_return", "research_supervisor")
```

最后一条与 `supervisor_return` 的 `Command` 重复；前四条是当前不同交付分支同时执行的直接原因。保留 `START` 入口、子图返回后的单一条件路由，以及交付节点至 `END` 的静态边。

已有 `destinations=(...)` 可用于图示，不是运行时权限或路由约束。当前安装版本的 `StateGraph.add_node` 文档明确说明它只影响图渲染。不能为了画出候选路线再增加真实静态边。

`final_report: str` 当前已经使用 LangGraph 的 `LastValue` 通道；其语义是每个 super-step 最多接受一次更新，并非按完成时间“最后一个覆盖”。自定义 `lambda old, new: new` 会吞掉这个异常，但不会取消错误启动的重写、补查和另一条交付路径，也不能撤回它们已经产生的文件写入。

建议保留终态单一责任：普通 writer 只产出 draft；`finalize`/`finalize_partial` 互斥交付，或者合成一个交付节点。单纯把写入挪到新节点并不能代替删除错误出边。

## 2. 路由修好后仍存在的状态缺口

不是所有 `Command(update={...})` 键都会自动成为持久状态。以下是编译后通道的实际检查：

| 字段 | 父图 | supervisor 子图 |
| --- | --- | --- |
| `research_iterations`、`max_iterations` | 无 | 有 |
| `review_history`、`task_ledger` | 无 | 有 |
| `report_version`、`report_ready` | 无 | 无 |
| `run_status`、`review_status` | 有 | 无 |

另有 writer 使用的 `writing_constraints` 未声明在父图状态中。影响包括：

- 报告阶段读取不到研究阶段的审查历史；报告审查写入的历史也不能保留到下一轮。审查与返工上限不能按预期累计。
- `report_version` 每次读取默认值，不能可靠递增。更新流中该字段被过滤。
- `report_ready` 当前既未成为通道，`_after_supervisor()` 也没有读取它，而是恒定返回 `report_writer`；其注释描述的 approved/limited 区分没有实现。
- 原 run 的模型调用/token 统计保存在同一个 ArtifactStore 中，这是另一条状态来源，不能笼统断言所有预算都已丢失。但图内研究迭代、审查、修订等计数的跨阶段传递确实缺失。

修复建议：明确父图持有 run 级计数、审查历史、版本和交付状态；通过显式子图输入/输出 schema，或包装节点映射，传入已有计数并返回新的计数/事件。研究子图输出不包含 `final_report`。同一次 run 保持相同 store/身份，不重新初始化预算。

不要仅给父图补一个追加 reducer 就结束：当前 `supervisor_return` 会把整个 `review_history` 又放入 update，追加 reducer 可能重复计入历史。应使用稳定事件 ID 去重，或只提交本轮新增事件。`review_history_reducer` 当前对 `open_issues` 也追加，与“已解决问题移除”的注释不符，需分别定义历史与当前问题的合并规则。

## 3. 报告返回研究目前并未真正连通

`reviewer.py:478` 的 `apply_review()` 在报告阶段、预算开放时，实际映射为：

| verdict | action |
| --- | --- |
| pass + passed | finalize |
| revise | rewrite_report |
| insufficient_evidence | rewrite_report |
| stop | rewrite_report |

因此 `review_report()` 中 `action == "continue_research"` 的分支不会被该决策函数选中。原图中看见 `supervisor_return`，不能证明缺证补查逻辑已生效；它可能只是被错误静态边启动。

应按已验证 findings 和预算，明确区分：表达修改→重写；必答项缺证且有预算→补查；停止或预算耗尽→有限交付。沿用项目 `IMPLEMENTATION_HANDOFF.md` 第 5 节的要求，不必为消除本次异常放弃预算继承。允许返工和补查的工作流有环，不宜再称为 DAG。

## 4. scripted 接缝掩盖了尚未修好的模型包装

`tests/odr/conftest.py` 替换了整个 `context.assemble` 和 `deep_researcher.assemble`，并自己重写请求/用量记录。这没有执行真实的工具绑定、结构化输出绑定和模型调用代码，文件注释中“保持 structured-output handling under test”的表述不成立。

`clarify_with_user`（314 行）、`write_research_brief`（458 行）、`researcher`（1316 行）仍先调用 `with_retry()`，然后传入 `assemble()`；researcher 的 baseline 分支还会在包装层调用 `bind_tools()`。本次用实际安装库验证，该包装对象没有 `bind_tools` 或 `with_structured_output`，真实 `assemble()` 在网络调用前就失败。

推荐把绑定顺序统一到调用入口：配置原模型→绑定该角色 tools/schema→包装有界重试→调用；每次重试仍须纳入预算与观测。不要只删掉重试或仅修 supervisor 一处。

测试保留 scripted 图流程测试，但另加使用真实 `assemble` 的模型适配测试，在底层模型响应/传输位置替换外部依赖。这样既可离线运行，也能捕获包装顺序、工具协议和结构化输出问题。

## 5. 测试应达到的程度

建议“相关离线契约与流程测试全绿，真实服务 smoke test 单列”。记录失败案例是核查产物，不能代替修复主流程已知错误。至少验证：

1. A/B 无审查路径与 C/D 审查路径分别可结束；当前所谓 baseline 测试实际使用 scoped policy，只覆盖 B 的组合。
2. pass 只触发一次最终交付；revise、补查、partial 分支互斥，不能同时创建两份最终文件。
3. 报告重写版本递增、旧 draft hash 审查失效；拒绝和超时不能成为 passed。
4. 返回研究仍使用原 run_id、store 和已消费预算；全局研究轮数、审查、返工各自触顶后停止。
5. 真实 `assemble` 的结构化输出/工具绑定/重试顺序；所有实际调用计入记录。
6. mock Tavily 底层响应，跑通派发→搜索→原文归档→证据→压缩→审查→报告；再覆盖超时/空结果/预算耗尽。可用统一断网 guard 检出意外外呼。
7. 单独安排可选真实 Tavily/LLM smoke test，明确 API、费用和限额；离线通过不等于服务集成已验证。

当前 `run_graph()` 为获取更新顺序和最终结果，会用相同 config/run_id/store 完整执行两次。应改为一次流式运行后读取同一 checkpointer 的最终状态，或让两次执行使用不同 run/store 和脚本状态，避免观测步骤污染预算。

## 在线依据

以下资料用于核对框架语义，具体缺陷以本地源码与离线复现为准：

- [LangGraph 官方 Graph API：Command](https://docs.langchain.com/oss/python/langgraph/graph-api#command)：动态 `Command` 不取消静态出边。`Command.PARENT` 也是受支持的父图跳转方式，不能仅凭此前失败断言框架不支持；当前项目可以继续采用显式状态交接。
- [LangGraph issue #6571](https://github.com/langchain-ai/langgraph/issues/6571#issuecomment-3646977648)：维护者解释两条路径同时执行是预期语义，而非 Command 应覆盖静态边。
- [LangChain 论坛：Twice execution of agent when using the interrupt](https://forum.langchain.com/t/twice-execution-of-agent-when-using-the-interrupt/2964/10)：提问者确认统一 Command 路由、去掉重复路径后恢复预期行为。
- [LangGraph 官方并发更新错误说明](https://docs.langchain.com/oss/python/langgraph/errors/INVALID_CONCURRENT_GRAPH_UPDATE)：reducer 用于有意的并行合并；本例两个互斥交付分支不应靠 reducer 合并。
- [开源 langgraph-reflection 实现](https://github.com/langchain-ai/langgraph-reflection/blob/main/src/langgraph_reflection/__init__.py)：审查后通过一个条件路由选择返回原图或结束，可参考它的互斥选择结构，不必引入新依赖。
- [官方子图文档](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)：支持共享状态，也支持包装节点显式转换输入输出；后者适合控制研究子图可以写回哪些字段。
- [官方测试指南](https://docs.langchain.com/oss/python/langgraph/test)：分别测试节点、局部执行路径与新建 checkpointer 的图实例，可避免只依赖完整真实服务运行。

当前证据足以确定最小路由修复和上述后续工作，不足以声称搜索集成、全部 A/B/C/D 契约或研究质量评测已经完成。
