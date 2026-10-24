# 与 Open Deep Research 相关的项目与实操 issue

核查日期：2026-10-22（Asia/Shanghai）。这里将“文章、100 star”理解为带项目介绍的 GitHub 仓库。star 是当日 GitHub 页面显示的约数，不是精确 API 计数。

## 五个项目

这五个项目在本次核查时均未归档，均超过 100 stars。前四个近期有主干提交；第五个胜在实现精简，但维护节奏较慢。

| 项目与介绍入口 | Stars | 技术与 Open Deep Research 的关系 | 本次检索到的近期主干提交 | 对我们的用途 |
|---|---:|---|---|---|
| [GPT Researcher](https://github.com/assafelovic/gpt-researcher) | 约 30.2k | Python 研究代理，包含规划、搜索、抓取、上下文处理和报告生成；与原项目的产品目标非常接近 | [2026-10-10](https://github.com/assafelovic/gpt-researcher/commit/3582a6841fe7dc801e2a34e34ee408641aea9bce) | 适合迁移 Python 研究工作流知识；本轮明确的小 bug 已有 PR，暂不重复认领 |
| [DeerFlow](https://github.com/bytedance/deer-flow) | 约 84.3k | Python 后端和 TypeScript 前端；从深度研究扩展为带子代理、工具、记忆和沙箱的通用代理系统 | [2026-10-22](https://github.com/bytedance/deer-flow/commit/29ba806fc6428cab19206daf72d17bb7e31b1bef) | 适合学习完整产品，但仓库大，不宜第一项就改架构 |
| [DeepAgents](https://github.com/langchain-ai/deepagents) | 约 30.5k | LangChain/LangGraph 生态的通用代理框架，涉及规划、子代理、文件和上下文管理；不是只做报告的应用 | [2026-10-22](https://github.com/langchain-ai/deepagents/commit/b4ac8829692785a221150658401ae558f4bff3f3) | 与原项目的框架知识衔接紧；本轮几个小问题已有人推进，不作为立即认领的首选 |
| [Local Deep Research](https://github.com/LearningCircuit/local-deep-research) | 约 9.4k | Python 研究系统，支持本地/云端模型、多个搜索引擎、文献和私有文档，并有完整界面 | [2026-10-22](https://github.com/LearningCircuit/local-deep-research/commit/dbd56ead57f0713b48cc8a46e69cf639ce800db3) | **本轮最适合上手贡献**：有维护者拆好的独立 UI 小项和文档小项 |
| [dzhng/deep-research](https://github.com/dzhng/deep-research) | 约 19.8k | TypeScript 精简实现，围绕搜索、网页抓取和递归研究生成报告；便于理解核心循环 | [2026-04-11](https://github.com/dzhng/deep-research/commit/1f8f3e285bbc23e80b98a66a64effab9069f3ad4) | 适合读代码和本地练习；不应把未归档等同于维护活跃或 PR 会很快合并 |

## 最适合现在做的两个小项

### A. 有可见成果：Local Deep Research #4961 的第 3 项

[Zotero page UX: smoother collection selection and friendlier sync feedback](https://github.com/LearningCircuit/local-deep-research/issues/4961)

- 状态：开放、无 assignee；维护者添加了 `good first issue`、`help wanted`，并明确表示各小项可独立提交。
- **本次建议只做第 3 项：群组选择。** 把页面底部的群组列表改成 Library 类型附近的下拉选择，选择后同步 Library ID，并通过既有 autosave 保存。
- 当前主干仍有独立的 `zotero-groups-card`、`zotero-groups-body`，选择按钮再回填上方的 `zt-library-type` 和 `zt-library-id`。因此该条需求仍有对应代码入口。
- 已排除重叠：第 1 项集合多选已有 [PR #5254](https://github.com/LearningCircuit/local-deep-research/pull/5254)；第 4 项反馈统一已有 [PR #5269](https://github.com/LearningCircuit/local-deep-research/pull/5269)；评论中第 2 项也有人认领。
- 对第 3 项，按 Zotero、group、dropdown 和相关 issue 检索，**暂未发现同范围 PR，也未见评论认领**。
- 另检查了 [PR #6762](https://github.com/LearningCircuit/local-deep-research/pull/6762) 中 Zotero 页面的 diff：它改响应式布局、控件和连接测试反馈，没有实现群组下拉选择。它会触及同一模板，落地时仍须核对最新分支以避免冲突。

代码入口：

- [zotero.html](https://github.com/LearningCircuit/local-deep-research/blob/dbd56ead57f0713b48cc8a46e69cf639ce800db3/src/local_deep_research/web/templates/pages/zotero.html)：表单、样式、内联脚本。
- [test_zotero_integration_ci.js](https://github.com/LearningCircuit/local-deep-research/blob/dbd56ead57f0713b48cc8a46e69cf639ce800db3/tests/ui_tests/test_zotero_integration_ci.js)：维护者指定的 UI 契约测试入口。

建议验收：

1. 选择个人库时，群组选择不干扰原行为；选择群组库时，下拉框显示名称。
2. 切换群组正确更新 Library ID、保存并在重载后恢复。
3. 群组列表为空或请求失败时给出清晰反馈，保留手动 ID 输入。
4. 配置尚未加载时不自动保存；所有动态名称和 ID 使用现有安全渲染方式。
5. 用假 API 数据验证交互，不需要付费 LLM；另做浏览器显示检查。

工作量判断：小到中等。主要是一个页面和对应 UI 测试，但仍需要按仓库要求完成环境安装和验证。

### B. 最小贡献练习：Local Deep Research #7673

[docs(search_system): correct the settings-or-queue-row provenance comment](https://github.com/LearningCircuit/local-deep-research/issues/7673)

- 状态：开放、无 assignee、无评论，由维护者从合并后的代码审查中提出。
- 问题：`search_system.py` 的注释把 `strategy_name` 和 `research_context` 都说成来自 settings 或队列 JSON；维护者说明两者的真实来源并不相同。
- 当前主干仍保留该注释。按 issue 编号和 `research_context` 检索历史 PR，未发现专门修正该来源说明的 PR；已识别原始改动 [#7610](https://github.com/LearningCircuit/local-deep-research/pull/7610) 和相邻防御检查工作，不把它们当作尚无人处理的新 bug。
- 改动应只纠正注释：分别说明策略名与研究上下文的输入来源，不顺带改行为。
- 验收：跟踪真实调用点核对来源、确认 diff 只有注释，执行仓库要求的检查。无需为这类注释修改新增行为测试。
- 工作量判断：很小；适合练习 fork、分支、diff 和 PR 流程，对代理算法的学习有限。

[当前源码入口](https://github.com/LearningCircuit/local-deep-research/blob/dbd56ead57f0713b48cc8a46e69cf639ce800db3/src/local_deep_research/search_system.py#L133)

## 各项目的 issue 核查结果

| 项目 / Issue | 内容 | 判断 |
|---|---|---|
| GPT Researcher [#2214](https://github.com/assafelovic/gpt-researcher/issues/2214) | URL 的路径或参数包含 youtube.com 时，Google 检索误删非 YouTube 结果 | 小、可离线复现，但已有 [PR #2215](https://github.com/assafelovic/gpt-researcher/pull/2215)，不重复提交 |
| GPT Researcher [#2218](https://github.com/assafelovic/gpt-researcher/issues/2218) | 最终报告生成失败却返回空字符串 | 已有 [PR #2219](https://github.com/assafelovic/gpt-researcher/pull/2219) |
| GPT Researcher [#2163](https://github.com/assafelovic/gpt-researcher/issues/2163) | 空反馈绕过计划修改计数 | 已有 [PR #2164](https://github.com/assafelovic/gpt-researcher/pull/2164) |
| DeerFlow [#7184](https://github.com/bytedance/deer-flow/issues/7184) / [#7188](https://github.com/bytedance/deer-flow/issues/7188) | 等待端口脚本对非法超时/超大整数处理不当 | 适合练习 Shell 参数验证，但已有 [#7185](https://github.com/bytedance/deer-flow/pull/7185) 等 PR，不重复做 |
| DeerFlow [#7267](https://github.com/bytedance/deer-flow/issues/7267) | 音乐生成失败后仍返回成功退出码 | 已有 [PR #7268](https://github.com/bytedance/deer-flow/pull/7268) |
| DeerFlow [#6562](https://github.com/bytedance/deer-flow/issues/6562) | 搜索 provider 的 max_results 上限不一致 | 等待维护者确定策略，已有参与者表示愿意实现；不适合作为无人处理的小 bug |
| DeepAgents [#6964](https://github.com/langchain-ai/deepagents/issues/6964) | Windows 下 ripgrep UTF-8 输出按系统编码解码，出现乱码或异常 | 修复方向清楚，但报告者已明确准备好改动和回归测试，不能仅因尚无 PR 就视为无人处理 |
| DeepAgents [#7120](https://github.com/langchain-ai/deepagents/issues/7120) | reftable 仓库的分支显示为 .invalid | 报告者已有改动和测试，并说明要在 #7119 后推进；不重复认领 |
| DeepAgents [#7129](https://github.com/langchain-ai/deepagents/issues/7129) | Modal 单文件权限/目录错误导致整个传输批次中断 | 报告者已有候选补丁和测试，并请求分配；不重复认领 |
| DeepAgents [#6830](https://github.com/langchain-ai/deepagents/issues/6830) | 希望摘要输入预算随模型窗口变化 | 维护者评论指出主路径默认 trim_tokens_to_summarize=None，并不受报告描述的 4000 token 限制；不作为已确认的主路径 bug，且已有 assignee |
| Local Deep Research [#4961](https://github.com/LearningCircuit/local-deep-research/issues/4961) | Zotero 页面交互改进 | 第 3 项群组下拉选择为本轮首选；其余小项须分别去重 |
| Local Deep Research [#7673](https://github.com/LearningCircuit/local-deep-research/issues/7673) | 研究参数来源注释错误 | 本轮最小、清晰的文档贡献候选 |
| Local Deep Research [#7652](https://github.com/LearningCircuit/local-deep-research/issues/7652) | 网络错误误归为文件系统错误，影响重试冷却时间 | 有实质后端价值，但与未合并的 [PR #7645](https://github.com/LearningCircuit/local-deep-research/pull/7645) 相邻且会触及同一逻辑；不优先作为第一项 |
| dzhng/deep-research [#172](https://github.com/dzhng/deep-research/issues/172) | Dockerfile 仍使用已 EOL 的 Node 18 | 未发现专门升级基础运行时的 PR。已检查 [#137](https://github.com/dzhng/deep-research/pull/137) 的 Docker diff，它仍使用 Node 18；[#174](https://github.com/dzhng/deep-research/pull/174) 虽提及 Node 22/24 测试环境，但没有修改 Dockerfile。适合本地 Docker 练习，维护活跃度较低 |

Node 版本状态另核对了 [Node.js 官方发布表](https://nodejs.org/en/about/previous-releases)：18 为 EOL，22 和 24 为 LTS。若后续处理 #172，需验证依赖安装、容器构建与入口启动，不能只替换标签就宣布完成，也不据此声称存在可利用漏洞。

## 建议顺序

1. 想看到真实产品变化：选 **Local Deep Research #4961 的第 3 项**。
2. 想用最小改动走完贡献流程：先选 **#7673**，再做 UI 小项。
3. 想继续学习 Python 研究代理主流程：阅读 **GPT Researcher**，但本轮已找到的小 bug 不再重复提交。
4. 想练容器构建：**dzhng #172** 可做本地练习，合并时效没有保证。

## 核查范围与限制

- 查询了候选项目的 star 页面、归档状态、近期提交、开放 issue 和相关 PR；对以上推荐读取了正文、评论及相关源码，对容易重叠的 #6762 / #137 / #174 检查了相关 diff。
- PR 查询包括开放和关闭状态，不把“关闭但未合并”当作未有人提交过。
- 中途遇到 GitHub 搜索的 secondary rate limit；暂停该轮搜索、继续独立阅读后，相关候选搜索已成功重试。失败响应没有被当作“零结果”。
- 未逐一审查这些仓库的全部历史 issue/PR；“未发现同范围 PR”仅代表本次检索结果。选定后、动手前仍应检查最新状态。
- 本轮只做选题调查；未安装这些仓库、未运行候选补丁、未联系维护者，也未创建 PR。
