# 学习参考与来源核验

核验日期：2026-10-23。以作者文章、项目源代码、官方文档及问题提出者的论坛记录为主。论坛内容用作设计线索和测试场景，不当作普遍性能结论。未确认 X 原帖直链；本方案引用作者博客原文。

## 建议学习顺序

先读 S1 理解用户提出的动机，再读 S2/S3 区分消息隔离、工具配置和存储权限；之后读 S4/S5 设计审查闭环；最后用 S6–S10 设计预算、恢复和针对性验证。

| ID | 来源 | 本次确认的内容 | 本项目如何借鉴及边界 |
| --- | --- | --- | --- |
| S1 | 立行，2026-10-04，[《吃一堑，长一智：Agent 时代的两个关键词 Context 与 Loop》](https://zj1123581321.com/posts/261004-context-and-loop/) | 按角色拼接规则、限制注入量；Advisor 使用独立上下文，并由实际运行信号触发咨询 | 借鉴角色输入和干净审查包；作者的个案与体感不代表本项目效果，KB 上限也不能直接换算为统一 token 预算 |
| S2 | [Deep Agents 子代理源码](https://github.com/langchain-ai/deepagents/blob/c89da392fbfdea1df7b6ff1ea8bad030efab090d/libs/deepagents/deepagents/middleware/subagents.py) | `SubAgent` 可配置工具；isolated 模式新建任务消息，fork 模式继承父对话；isolated 路径仍会转移部分非排除状态 | 借鉴独立任务输入和显式工具集合；本项目采用字段白名单，不能把“消息独立”等同于“其他状态全部隔离” |
| S3 | [Deep Agents 文件后端源码](https://github.com/langchain-ai/deepagents/blob/c89da392fbfdea1df7b6ff1ea8bad030efab090d/libs/deepagents/deepagents/backends/filesystem.py) | 源码明确区分虚拟路径限制与进程沙箱；后者并非 `root_dir` 或 `virtual_mode` 自动提供 | MVP 使用受控 artifact ID 接口；未来若给 shell / 任意文件工具，需另做进程与文件系统隔离 |
| S4 | [LangGraph Reflection 源码](https://github.com/langchain-ai/langgraph-reflection/blob/888d007163f1479e1e7a734f66dc7dcbd12424d7/src/langgraph_reflection/__init__.py)及[项目说明](https://github.com/langchain-ai/langgraph-reflection/blob/888d007163f1479e1e7a734f66dc7dcbd12424d7/README.md) | 主图执行后进入 reflection，根据反馈决定继续；使用共享状态 schema，并有 remaining_steps 退出逻辑 | 借鉴显式反馈回路；不直接套整个 wrapper，另建 ReviewPacket，提供原文证据和明确返工上限 |
| S5 | Anthropic，[Building effective agents](https://www.anthropic.com/research/building-effective-agents)，2024-12-19 | evaluator–optimizer 模式适合有明确评判标准、反馈可带来实际改进的工作 | 把准确性、覆盖、引用支持设成可核验判据；“增加一个评价 LLM”本身不能保证减少幻觉 |
| S6 | Anthropic，[Effective context engineering for AI agents](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents)，2025-09-29 | 强调高信号且充分的上下文、清晰工具、按需取得材料 | 精简冗余同时保护任务约束与证据；不把最短上下文当作目标 |
| S7 | LangChain，[Context Management for Deep Agents](https://www.langchain.com/blog/context-management-for-deepagents) | 工具结果外存、历史压缩；专门测试压缩后的目标保持与信息恢复 | 加入“早期关键信息被移出后仍可恢复”的验收；文章中的窗口比例和示例成绩不是本项目参数依据 |
| S8 | [LangGraph Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) | 介绍显式输入输出映射、per-invocation 与 per-thread 持久化；重复并行调用同一 per-thread 子图有冲突风险 | 研究员默认使用独立 invocation；实现时对锁定依赖做并发与恢复验证，不凭文档假定环境完全一致 |
| S9 | [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | checkpointer 管线程状态，store 管图外数据；内存 saver 不跨进程重启保存，Agent Server 管理部署持久化 | 区分工作流恢复、证据文件和日志，不将 JSONL 日志冒充 checkpoint |
| S10 | LangChain Forum，[tool isolation between Skills](https://forum.langchain.com/t/tool-isolation-between-skills/3496)，2026-04-20 | 讨论 Skill 说明中的工具约束与实际工具绑定、执行限制的区别 | 用于提醒“写进提示词”不是权限实现；不把帖子对当时版本的描述推广为所有当前版本事实 |
| S11 | LangChain Forum，[单个巨大工具结果导致空摘要的报告](https://forum.langchain.com/t/deepagents-summarization-can-produce-an-empty-summary-when-a-single-tool-result-exceeds-trimtokenstosummarize/4751)，2026-10-15 | 提问者报告摘要输入被裁为空、任务未完成却输出摘要的现象，涉及 JS/Cloud 场景 | 转化为故障注入用例；本次未在 Python 本项目复现，不能声称本项目存在相同实现缺陷 |
| S12 | [Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://arxiv.org/abs/2411.06037)，ICLR 2025；[Google Research 说明](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context/) | 区分材料相关性和回答所需信息是否充分，研究自动充分性评估与适当不回答 | 为当前需求建立 covered/missing/uncertain；研究中的判别器不是全局最小集合求解器，也不能替代原始来源核验 |
| S13 | [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)，2023/2024 | 从分层存储和数据调入调出角度管理有限上下文 | 借鉴工作集与外存分离；不要求迁移框架，也不能保证自动找回所有遗漏 |
| S14 | [Deep Agents 摘要中间件源码](https://github.com/langchain-ai/deepagents/blob/c89da392fbfdea1df7b6ff1ea8bad030efab090d/libs/deepagents/deepagents/middleware/summarization.py)；[官方说明](https://www.langchain.com/blog/context-engineering-deep-agents-remember)，2026-10-14 | 每个 invocation 有历史文件标识；被摘要的历史外存，并将路径写入摘要消息供后续读取 | 借鉴可恢复移出。本项目额外要求持久化确认和受控回读；不照搬任意路径读取 |
| S15 | Letta V1 SDK 文档：[Context hierarchy](https://docs.letta.com/v1-sdk/memory/context-hierarchy)、[Archival memory](https://docs.letta.com/v1-sdk/memory/archival-memory) | 将始终可见的 memory blocks、可读取文件、可搜索档案分开 | 借鉴按需加载层次；链接为 V1/legacy 文档，只作为概念参考，不据此指定当前 SDK API 或默认值 |
| S16 | LangChain Forum，[Skills and Memory persistance across threads](https://forum.langchain.com/t/skills-and-memory-persistance-across-threads/2987)，2026-02-23 | 讨论新 thread 不会自动继承之前聊天与默认状态内文件，需要显式持久化和重载 | 用作边界提醒；本方案跨角色和跨 run 都保持明确授权，不照搬帖子里宽泛的文件根目录示例 |
| S17 | [LLMLingua-2](https://arxiv.org/abs/2403.12968)，2024 | 研究高效且保真的 prompt compression | 可以作为后续压缩器候选；压缩保真、材料是否充分、角色是否有权读取是不同问题，本阶段不新增该依赖 |

## 本次实际看的代码位置

### Deep Agents

固定阅读快照：`c89da392fbfdea1df7b6ff1ea8bad030efab090d`，通过 GitHub 连接器读取。

- `middleware/subagents.py`：`SubAgent` 定义约第 81 行、工具字段约第 112 行、mode 约第 219 行、输入准备约第 813–836 行。
- 要点：isolated 分支虽然重置 `messages`，仍从运行时状态复制部分其他字段。因此本设计采用“只准传哪些字段”，而非仅删掉 `messages`。
- `backends/filesystem.py`：约第 121–135 行的边界说明、约第 182 行的路径解析。
- 文档与源码可能在默认参数上不同步。本设计不依赖默认 `virtual_mode` 值；要求显式配置和测试访问拒绝。

### LangGraph Reflection

固定阅读快照：`888d007163f1479e1e7a734f66dc7dcbd12424d7`，最新 commit 的时间为 2025-03-18，本次可读取。

- `create_reflection_graph()`：主图与 reflection 图的连接方法。
- `end_or_reflect()`：根据剩余步骤和最后消息类型决定继续或退出。
- 它是清晰的小型参考实现；本方案不要求增加此依赖，也不将其作为当前本地 LangGraph 版本兼容性的证明。

## 本地核验边界

- 项目名与路径已明确为 `open_deep_research`。
- 当前目录无 `.git`；源文件与配置的内容散列见 [baseline-files.json](baseline-files.json)。
- `pyproject.toml` 声明 `langgraph>=0.5.4`；本次 `uv.lock` 中锁定的是 `langgraph==1.2.9`。没有据此假设当前虚拟环境一定安装了该版本，执行者应检查实际环境。
- 当前代码已有最终压缩和部分网页摘要；本方案提出的是请求级预算、可恢复外存、证据契约及审查节点，不是声称此前完全没有上下文处理。
- 本次未调用付费研究模型、未运行 benchmark、未验证性能增益。所有百分比门槛只用于后续实验决策。

## 资料如何变成设计决策

| 设计决定 | 参考线索 | 需要本项目自行验证的内容 |
| --- | --- | --- |
| Reviewer 每次从独立审查包开始 | S1、S4 | 是否降低误判、是否会因缺失必要背景增加错判 |
| 角色/任务级工具白名单，执行时再次校验 | S2、S10 | 任务成功率、工具召回率、越权拒绝 |
| 原文存储与模型工作集分离 | S3、S7、S9 | 回读正确性、延迟、存储成本、引用可追踪 |
| 硬预算、外存优先、摘要可恢复 | S6、S7、S11 | 巨大单条结果、空摘要、关键条件丢失 |
| 只读证据审查、有上限的修订循环 | S4、S5 | 漏检、误拦、成本和终止行为 |
| 运行状态恢复与信息访问权限分开设计 | S8、S9 | 并发隔离、重启恢复、幂等与跨 run 访问 |

以上是借鉴与设计推导的映射，不是外部资料对最终方案效果的背书。

## 动态选择与恢复补充核验

S14 的固定源码中，`_get_session_id()`（约 797 行）将历史标识绑定到 invocation，`_get_history_path()`（约 820 行）生成历史文件路径，`_build_new_messages_with_path()`（约 865 行）把可回读路径放入摘要消息。

`_offload_to_backend()`（约 1326 行）的注释允许写入失败后返回 None 并继续。因此，本方案提出的“确认原文保存成功后才移出”是我们的额外要求，不能说成该参考代码已无条件保证可恢复。GitHub 读取首次遇到连接传输错误，重试后成功核验了上述源码；未运行该中间件。

S12/S13/S15/S17 本次阅读的是论文摘要及官方说明/文档，未复现其模型实验。新设计的 StepDemand、按需语义 Manager、遗漏导致的结论失效与修复关系是对本项目的设计推导；没有外部证据证明这一完整组合已能达到某个性能提升比例。
