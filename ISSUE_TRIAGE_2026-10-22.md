# Open Deep Research issue 筛选记录

核查日期：2026-10-22（Asia/Shanghai）。目标：寻找有实际价值、工作量可控、未发现同范围历史 PR 的开放 issue。

## 结论

- 项目是 [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)。上游于 **2026-08-21 归档**，现在只读。因此，这里的候选适合本地或 fork 维护；目前无法向该上游新提 PR。
- 本次读取了 **39 个开放 issue**，检索并去重了 **199 条历史 PR 记录**，包括开放、关闭和合并记录；对候选和容易重复的修复进行了正文、评论及相关代码核对，并抽查了可能重叠的 PR diff。并非逐一审查全部 PR diff。
- **首选 #203：MCP 工具固定参数。次选 #233：主实现中的自定义报告结构。** 两者目前均开放、无 assignee、无评论；按编号及相关功能关键词检索，未发现同范围 PR。
- 没有找到值得优先处理、同时完全符合“尚未有人 PR 过”的简单确定性 bug。多数明显 bug 已有历史修复 PR，即使对应 issue 仍开放。
- “未发现同范围 PR”限于本次上游检索；不能排除未提交的分支、未明确描述的实现或其他 fork 中的工作。

上游代码核查基线：[1b7d2e80db9faa586165c60e09096dbbfd483a64](https://github.com/langchain-ai/open_deep_research/commit/1b7d2e80db9faa586165c60e09096dbbfd483a64)。本地目录没有 `.git`，不能把它当作已经同步的 Git checkout；本次没有改动程序代码。

## 1. 首选：#203，MCP 工具固定参数

[Passing pre-set arguments to MCP Tools](https://github.com/langchain-ai/open_deep_research/issues/203)

**实际需求：** 用户在界面上已经选定的参数，应由程序传给 MCP 工具，避免模型遗漏、猜测或改写。例如用户选定 `project_id` 和日期范围，模型只需要填写搜索词。

**现有代码：** `MCPConfig` 只有 `url`、`tools`、`auth_required`；`load_mcp_tools()` 负责连接、筛选工具及包装认证，没有固定业务参数的配置入口。研究工具执行仍使用模型生成的调用参数。

- [MCPConfig](https://github.com/langchain-ai/open_deep_research/blob/1b7d2e80db9faa586165c60e09096dbbfd483a64/src/open_deep_research/configuration.py#L19)
- [load_mcp_tools](https://github.com/langchain-ai/open_deep_research/blob/1b7d2e80db9faa586165c60e09096dbbfd483a64/src/open_deep_research/utils.py#L449)
- [researcher_tools](https://github.com/langchain-ai/open_deep_research/blob/1b7d2e80db9faa586165c60e09096dbbfd483a64/src/open_deep_research/deep_researcher.py#L435)

**建议最小范围（设计建议，尚未实施）：**

1. 为 MCP 配置增加按工具名设置的固定参数映射。
2. 对模型展示的工具 schema 隐藏这些参数，在实际调用前注入配置值；固定值优先，不能被模型参数覆盖。
3. 使用原工具 schema 校验最终参数；错误配置要给出明确错误。
4. 保留工具调用 ID、RunnableConfig 传递和现有认证包装行为；每次运行的参数独立。

**验收重点：** 模型省略固定参数仍能正确执行；试图覆盖时仍使用配置值；不同运行间不串值；未配置时沿用原调用方式。可以使用假 MCP 工具离线验证，无需调用付费模型。

**工作量判断：中等。** 主要涉及 `configuration.py`、`utils.py` 和针对性测试。难点是 schema 与真实调用参数一致，而不是简单地把参数补进提示词。

**重复核对：** [PR #153](https://github.com/langchain-ai/open_deep_research/pull/153) 增加直接传入自定义 Python 工具的能力，已检查 diff；它没有实现现有 MCP 工具的固定参数绑定。历史 MCP 认证、JSON 配置修复也不是同一需求。

## 2. 次选：#233，主实现中的自定义报告结构

[Support for Customizable Report Template Structure](https://github.com/langchain-ai/open_deep_research/issues/233)

**实际需求：** 用户提供子主题和对应来源 URL，研究过程按这些主题开展，最终报告遵循指定结构。

**现有代码边界：** 主实现的研究简报提示词已经要求保留用户偏好和优先来源，所以在普通输入中写要求已有一定支持。旧版 `legacy` 还存在 `report_structure` 配置。真正值得补充的是：**当前主实现中显式、可传递、可核验的结构配置**，不能仅把旧版能力再写一次或只改一处 prompt。

**建议最小范围（设计建议，尚未实施）：**

1. 增加可选的结构化报告提纲，包含章节标题、研究要求和优先 URL。
2. 在生成简报、主管分配研究和最终写作时一致传递。
3. 检查必需章节的覆盖情况并报告缺项；不能把提示词遵循等同于保证输出准确。
4. 第一版将 URL 定义为“优先参考来源”；若要强制访问每个 URL，需要另行实现抓取或检索工具支持。

**工作量判断：中等偏大。** 涉及配置、多个阶段的数据传递、提示词及结构覆盖验证；比 #203 更容易扩大范围。

**重复核对：** 按 issue 编号、`report template`、`report structure` 等检索；旧版结构配置和 [PR #210](https://github.com/langchain-ai/open_deep_research/pull/210) 的证券研究专用格式不等于主实现的通用模板接口。检查了 #210 的相关 diff，未发现与上述最小范围相同的实现。

## 3. 备选：#234，评估已有自定义报告

[Evaluating Custom Reports with run_evaluate.py](https://github.com/langchain-ai/open_deep_research/issues/234)

这是使用咨询，可转化为独立的报告评估入口：读取已有 Markdown 报告、问题和可选参考答案，复用评估器，而不重新执行研究图。

当前 `tests/run_evaluate.py` 的 `target()` 会重新运行研究图；仅修改 `dataset_name` 不能让它改为评估用户已有的报告。未发现同范围 PR，但现有评估链本身有已被历史 PR 覆盖的缺陷（如 #217），因此不建议把它作为第一个工作项。实现前应先确认需要哪些评分维度，以及是否允许依赖 LangSmith 和模型调用。

## 已排除的明显重复或已有人推进的问题

“已有 PR”包含关闭但未合并的 PR，不能因为 PR 不再开放就认为没人做过。

| Issue | 问题 | 排除证据 |
|---|---|---|
| [#283](https://github.com/langchain-ai/open_deep_research/issues/283) | 主管吞掉研究异常 | [#185](https://github.com/langchain-ai/open_deep_research/pull/185)、[#286](https://github.com/langchain-ai/open_deep_research/pull/286)、[#333](https://github.com/langchain-ai/open_deep_research/pull/333) 已提交相关修复 |
| [#284](https://github.com/langchain-ai/open_deep_research/issues/284) | 旧版并行章节提前完成 | [#299](https://github.com/langchain-ai/open_deep_research/pull/299) |
| [#252](https://github.com/langchain-ai/open_deep_research/issues/252) | 压缩时删错消息方向 | [#254](https://github.com/langchain-ai/open_deep_research/pull/254)、[#255](https://github.com/langchain-ai/open_deep_research/pull/255)、[#332](https://github.com/langchain-ai/open_deep_research/pull/332) |
| [#217](https://github.com/langchain-ai/open_deep_research/issues/217) | 正确性评估缺少实际输入 | [#245](https://github.com/langchain-ai/open_deep_research/pull/245)、[#251](https://github.com/langchain-ai/open_deep_research/pull/251)、[#300](https://github.com/langchain-ai/open_deep_research/pull/300)；已核实 #300 关闭、未合并 |
| [#213](https://github.com/langchain-ai/open_deep_research/issues/213) | MCP_CONFIG JSON 未解析 | [#302](https://github.com/langchain-ai/open_deep_research/pull/302)；已核实关闭、未合并 |
| [#232](https://github.com/langchain-ai/open_deep_research/issues/232) | Sources 格式冲突 | [#301](https://github.com/langchain-ai/open_deep_research/pull/301)；已核实关闭、未合并 |
| [#268](https://github.com/langchain-ai/open_deep_research/issues/268) | README 链接失效 | [#270](https://github.com/langchain-ai/open_deep_research/pull/270) |
| [#225](https://github.com/langchain-ai/open_deep_research/issues/225) | StateGraph 弃用参数 | [#254](https://github.com/langchain-ai/open_deep_research/pull/254)、[#303](https://github.com/langchain-ai/open_deep_research/pull/303) |
| [#218](https://github.com/langchain-ai/open_deep_research/issues/218) | 压缩阶段 function_call 序列化报错 | [#306](https://github.com/langchain-ai/open_deep_research/pull/306) 已处理相关的异常误分类；它不等于解决序列化根因，但按严格的“不与已有 PR 重叠”标准暂不首选 |
| [#269](https://github.com/langchain-ai/open_deep_research/issues/269) | 旧版计划修改循环无上限 | 未找到对应修复 PR，但评论中已有作者和另一位贡献者推进、认领；不宜当作无人处理的机会 |
| [#296](https://github.com/langchain-ai/open_deep_research/issues/296) | 搜索内容增量压缩中间件 | 提案人已表示有实现和待提交分支；且无信息损失主张尚需独立验证，不宜重复开发 |

## 39 个开放 issue 的初筛索引

此表是筛选分类，不表示每项均已复现。

| Issue | 初筛结果 |
|---|---|
| #296 | 提案人已有实现；不优先 |
| #269 | 旧版；评论已有认领 |
| #284 | 已有相关 PR #299 |
| #290 | 从 #284 评论派生的低信息条目 |
| #283 | 已有相关 PR #185 / #286 / #333 |
| #213 | 已有相关 PR #302 |
| #268 | 已有相关 PR #270 |
| #267 | 主要为截图，当前缺少可直接核验的文本复现步骤 |
| #201 | Azure 接入咨询；已有相关支持 PR #211 / #215 |
| #252 | 已有相关 PR #254 / #255 / #332 |
| #250 | 自定义搜索接入咨询；已有相关 PR #153 / #206 等 |
| #217 | 已有相关 PR #245 / #251 / #300 |
| #234 | 自定义报告评估咨询；可作为备选 |
| #233 | 主实现的通用报告结构配置；次选 |
| #232 | 已有相关 PR #301 |
| #231 | 大版本适配咨询；范围过大，不宜作为首个工作项 |
| #175 | 子图状态/流式可见性咨询；需先区分 SDK 使用方式与实际缺陷 |
| #195 | OpenAI 兼容端点接入；已有相关 PR #220 |
| #227 | 仓库描述和 topics；归档上游的维护者元数据操作 |
| #194 | tracing 空值配置；已有相关 PR #308 |
| #202 | 模型切换咨询；与 Azure 支持/配置文档重叠 |
| #226 | 模型未产生并行工具调用；尚不能据此判定框架并发代码有 bug |
| #225 | 已有相关 PR #254 / #303 |
| #221 | issue 明确链接已有 PR #220 |
| #218 | 有相关 PR #306，序列化根因仍需独立复现 |
| #208 | Exa 搜索；已有相关 PR #264 |
| #216 | 依赖版本冲突；需在明确版本环境中复现，不优先泛化升级 |
| #193 | LangSmith 数据集访问/配置问题；先验证公开数据集使用方式 |
| #214 | Python 调用示例咨询；已有相关 PR #144 |
| #196 | 自定义 base_url；已有相关 PR #220 |
| #204 | NVIDIA 托管 Notebook 连接失败；需外部环境，当前不适合直接修复 |
| #203 | MCP 固定参数；首选 |
| #198 | ResearchComplete 提示词设计咨询；非已证实 bug |
| #190 | 工具调用与兼容端点配置；需复现，已有端点支持工作 |
| #187 | 初学者使用文档；与已有示例/文档工作重叠 |
| #184 | 报告篇幅/质量诉求；需要明确验收指标，非小型确定性修复 |
| #171 | Tavily key 使用问题；需核对调用方与当前实现，未复现 |
| #183 | datetime 调用；已有相关 PR #169 / #176 / #178 |
| #177 | Gemini provider 选择诉求；需区分现有配置能力与文档问题 |

## 检索方法与限制

- issue 检索：`repo:langchain-ai/open_deep_research is:issue is:open`。
- PR 检索：开放 PR、全部状态的近期 PR、较早时间段 PR，并按编号去重。
- 候选额外检索：issue 编号、`pre-set`、`preset`、`tool arguments`、`report template`、`report structure` 等；阅读候选 issue 评论，核查可能重叠的 #153 / #200 / #210 PR diff。
- 代码核对使用 GitHub 当前主分支内容，本地源码只用于定位和交叉阅读。
- 本次属于选题调查，没有进行修改后的测试、真实 API 调用或远端写操作。
