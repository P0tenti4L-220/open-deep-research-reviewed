# 第一轮验收复核（2026-10-23）

## 复核后状态（2026-10-23 修复轮追加）

> 本节由修复轮追加。R1–R5 已在代码中修复并补充回归测试，另修复 3 个复核时未发现的
> 缺陷（研究阶段审批无法放行、研究阶段 Reviewer 看不到 claim、公开入口 state 被整包替换）。
> 详细内容见 [ROUND1_FIX_REPORT_2026-10-23.md](ROUND1_FIX_REPORT_2026-10-23.md)。
>
> 当前离线结果：`tests/odr` = **66 passed, 1 skipped**；全仓仍有 legacy fixture 的 1 个 error；
> 复现探针（改为通过判定式）= **6/6**，见 `round1_acceptance_probe_20261023_after_fix.json`。
> 真实付费集成本轮仍未重跑，因此**不能据此宣称真实运行下的质量或成本已改善**。
> 下文保留复核当时的原始判断，作为修复前快照。

## 结论与范围

**架构方向可以保留，但当前实现未通过第一轮验收。先修复本文 5 项阻塞问题，再冻结第一轮版本并上传 GitHub；第二轮能力扩展另开分支。**

本次对照 `IMPLEMENTATION_HANDOFF.md`、设计 README 和 `FIX_REPORT_2026-10-23.md`，检查当前本地源码、运行已有离线测试，并增加不调用真实供应商的复现探针。没有修改生产代码、原测试、配置或历史审计快照，没有初始化 Git，也没有上传远程。

当前目录 `~/open_deep_research` 没有 `.git`，本次结论只针对当前文件，不能给出相对上游某个 commit 的 diff 结论。此前报告的真实集成 `1 passed` 本次没有重跑；本次也没有重新评估答案质量、成本收益或完整权限隔离。

## 已确认有效的修复

- `review_report`、`supervisor_return` 没有重复静态出边，动态路由使用 `Command(goto=...)`。
- `final_report` 仍是 `LastValue`；现有流程测试的交付路径没有复现原来的并发写入错误。
- 父图/主管子图共享研究轮数、任务账本、审查历史等运行级状态；现有补查测试确认同一 run 继续。
- 审查历史按稳定 ID 去重，当前 `open_issues` 使用新值替换。
- 模型工具/结构化输出绑定先于外层重试，原来的 wrapper 绑定问题已覆盖。

这些结果证明本轮路由修复有效，但不等于所有审查与预算契约都已经成立。

## 必须修复的问题

### R1 · P1：部分核验或缺失证据仍能授权完整交付

> **状态：已修复**（报告阶段交付门禁由程序判定；见修复记录 §1）。


定位：`src/open_deep_research/reviewer.py:279`、`:298`、`:303`、`:513`。

`validate_decision()` 计算了 `unchecked_claim_ids`，但只在一个 claim 都没检查时拒绝 `pass`。只要检查过一个，就可能保留 `passed`。同时，证据检查以 `packet.evidence_ids` 中声明过的 ID 为准，不保证它实际存在于已读取的证据中；`build_review_packet()` 记录的缺失证据问题没有成为确定性阻断条件。

本次探针：审查包有 C1、C2，引用未归档的 E-missing；模型只报告检查了 C1。

```text
verdict = pass
status = passed
unchecked_claim_ids = [C2]
open_issues = [Referenced evidence not archived: E-missing]
apply_review(...) = finalize
```

影响：关键主张未审或来源不存在，报告仍可标记为完整通过。当前对“伪造证据”的测试只覆盖 Reviewer 在 findings 里引入未知 ID，没有覆盖 claim 原有引用缺失。

修复要求：按阶段定义交付必需的 claim/requirement 集合；报告交付时必须核对覆盖、实际可用来源和未解决阻塞项。计划阶段可以没有 claim；报告阶段不能把“没有建立 claim 清单”自动视为无需事实审查。空清单或明确未知项要有相应的内容/范围判定。

验收：两条必审 claim 只检查一条、声明了不存在的来源、空 claim 清单但报告含事实断言，均不得完整通过；完整核验且证据充分的正例仍能通过。

### R2 · P1：长草稿被静默截断，最终却交付完整草稿

> **状态：已修复（草稿切段完整覆盖，分批审查，超限记为未读）**（见修复记录 §1）。

定位：`src/open_deep_research/reviewer.py:168`（尤其 `:175`、`:181`、`:182`），以及 `deep_researcher.py:1922`。

`_split_sections()` 每节只保留前 6000 字符，并只保留前 24 节。没有记录这次截断产生的未审范围。最终 `finalize()` 交付的是完整 `final_report_draft`，草稿 hash 一致并不能证明其全部内容进入过审查。

本次探针：创建真实本地证据 E1 和 C1/C2，模型检查两个已登记 claim；把额外断言标记放在同一节第 6100 字符之后。

```text
marker_visible_to_reviewer = false
section_chars = 6000
open_issues = []
status = passed
apply_review(...) = finalize
```

影响：报告尾部新增的错误主张可能根本没被 Reviewer 看到，仍以全文通过交付。后面的 `trim_packet_for_budget()` 也只加文字提示，没有把未审范围转为机器可检查的覆盖状态。

修复要求：分批覆盖全部草稿并合并段落/claim 覆盖，或在超限时明确返回 partial/inconclusive；完整交付必须绑定实际审查范围和最终草稿。登记 claim 应与最终草稿对应，而不只是读取研究阶段的全部 claim。

验收：错误分别放在第 6001 字符后、第 25 节和 Writer 新增断言中，均不能在未审情况下获得全文通过。

### R3 · P1：总预算没有在统一调用入口执行

> **状态：已修复（每次尝试原子预留；耗尽即受控收尾）**（见修复记录 §1）。

定位：`src/open_deep_research/context.py:860`；`deep_researcher.py:232`、`:543`。

现有预算检查位于部分审查/写作/停止节点，`_call_model()` 和 `assemble()` 在发出请求前没有统一预算检查，也没有并行调用的原子预留。`assemble()` 会按 retries 继续发起尝试，不重新检查剩余总额度。

本次完整图探针使用现有 ScriptedRuntime，设置 `max_total_model_calls=1`：

```text
actual_model_calls = 3
roles = [brief_writer, brief_writer, supervisor]
run_status = partial
```

另一个使用真实 `assemble()` 的本地假模型在总调用上限为 1 时，仍执行 3 次重试尝试。

影响：最终返回 partial 不代表执行期间遵守了预算；研究员并行派发和重试会进一步扩大问题。

修复要求：所有模型调用（含辅助模型与每次重试）经过同一运行级预算预留/结算入口；并行预留要原子化，预算耗尽时转入受控收尾。未知 token 用量应保留 unknown，并按约定的保守预留执行上限，不能按免费调用处理。

验收：在 1 次剩余额度、多个并发申请和多次重试情况下，实际获准的新请求数不得超限；预留与结算过程可追溯。

### R4 · P1：重试覆盖角色用量记录，超时调用未计数

> **状态：已修复（按 attempt 记账；超时计数并传播取消）**（见修复记录 §1）。

定位：`src/open_deep_research/artifacts.py:827`、`:858`；`context.py:864`；`reviewer.py:450`。

每次重试沿用同一个 `request_id`。`record_usage()` 虽然递增 `model_calls`，但会覆盖 `self._meta.usage[request_id]`；`calls_by_role()` 又按该字典计数。因此 3 次尝试在审查角色计数中变成 1 次，`max_review_model_calls` 会漏计重试。

`asyncio.wait_for()` 超时取消内部调用时，`assemble()` 的 `except Exception` 没有记录这次取消的在途请求。探针结果：

```text
重试：actual_attempts=3, total_calls=3, reviewer_calls=1
超时：error="review timed out", total_calls=0, reviewer_calls=0
```

影响：审查调用上限、角色成本统计与总调用统计不一致；超时调用可能已经消耗供应商资源，却完全不进入预算账本。

修复要求：区分逻辑请求 ID 与 attempt ID，每次尝试单独结算；超时/取消记录调用已启动及用量未知，并正确传播取消。不能简单吞掉取消异常。

验收：3 次尝试的全局与角色调用数均为 3，明细保留 3 行；一次已开始的超时请求计为一次尝试且用量未知，重启读取持久化账本后统计一致。

### R5 · P1：README 对应的公开图入口失效

> **状态：已修复（公开入口改为宿主适配层）**（见修复记录 §1）。

定位：`langgraph.json:4`、`src/open_deep_research/deep_researcher.py:136`、`:2130`；新入口在 `run.py:34`。

`langgraph.json` 仍加载导出的 `deep_researcher` 编译图，但节点要求 config 预先含有 Python 对象 `__odr_ctx__`。只有新 `run.make_config()` 创建它。直接按照原图接口调用会在首次模型请求前报错：

```text
RuntimeError: ODR runtime context missing: graph nodes must not be called without the host-bound identity
```

本次通过普通 `deep_researcher.ainvoke(messages, config)` 复现，没有启动服务器或调用真实模型。README 仍引导用户进入 Studio；该配置入口未接入新的宿主初始化。此外，导出图固定为 baseline，单独在 Studio 设置 reviewer 开关也不会自动选中另一张图。

修复要求：为默认服务入口接入受控、运行级的身份/store/图选择初始化，或明确修改公开入口与文档。原实施规格要求保持默认使用方式，因此优先补兼容入口。不要让客户端构造可伪造的身份对象来绕过宿主边界。

验收：使用普通 JSON 输入和默认配置从实际公开入口启动，在离线假模型下可以走完流程；Reviewer 开关与选择的图拓扑一致；恢复同一 run 时身份和预算不重置。

## 测试结果及其边界

| 本次实际执行 | 结果 |
| --- | --- |
| `rtk proxy .venv/bin/python -m pytest tests/odr -q` | 39 passed, 1 skipped |
| `rtk proxy .venv/bin/python -m pytest --collect-only -q --disable-warnings` | 41 tests collected |
| `rtk proxy .venv/bin/python -m pytest -q --disable-warnings` | 39 passed, 1 skipped, 1 error |
| 新增离线验收探针 | 上述边界行为全部复现，结果保存在同目录 JSON |

全仓错误发生在 legacy 测试的 **fixture setup 阶段**，不是 collection 阶段：`src/legacy/tests/test_report_quality.py:86` 读取未注册的 `--research-agent` 选项。本次确认了当前根因；没有 Git 历史，未独立验证“该错误早于本轮改动”的历史归属。

现有测试为什么仍通过：图流程测试替换了 `assemble()`，会主动填入用量；模型边界测试只检查全局 `model_calls` 和日志行数，没有检查角色计数、取消和预算拦截；审查测试主要检查零 claim 核验与 Reviewer 编造 findings 引用，没有检查部分覆盖和长草稿完整性。

真实集成测试仅断言报告非空、状态为 complete/partial、metrics 文件存在（`tests/odr/integration/test_real_smoke.py:43`）。因此此前的 `1 passed` 可以作为一次冒烟结果，不能单独证明 Tavily 搜索、证据归档、事实核验和正确答案全部完成。应另外断言实际搜索事件、归档证据、claim 链、审查覆盖及目标答案/来源。

复现文件：

- [round1_acceptance_probe_20261023.py](round1_acceptance_probe_20261023.py)
- [round1_acceptance_probe_20261023.json](round1_acceptance_probe_20261023.json)

从仓库根目录重跑并保存新结果（不要覆盖本次历史 JSON）：

```bash
rtk proxy .venv/bin/python docs/context-review-design/round1_acceptance_probe_20261023.py /tmp/round1-probe-new.json
```

探针使用临时 ArtifactStore、本地假模型和已有离线 harness；不读取 `.env`，不请求真实搜索或模型 API。它展示行为，不是通过/失败式回归测试；修复时应把上述预期转成正式测试。

## 第一轮收尾与第二轮分界

先修 R1–R5 并补对应回归测试，保留本轮已生效的路由、状态和 reducer 修复。长草稿完整覆盖、调用上限和公开入口属于第一轮既定承诺，不宜归到第二轮新功能。

同步修正文档状态：设计目录 README 仍写“尚未实现”，根 README 仍主要介绍上游；应清晰列出本项目增加的能力、启动方式、实测范围和未完成事项。未经独立实验，不沿用上游榜单成绩作为修改后版本成绩，不声称成本或质量已改善。

## 修复后上传到个人 GitHub 的步骤

此节是操作说明，本次没有执行 Git 初始化、提交或上传。

1. 在个人账户下创建一个空仓库，例如 `open-deep-research-reviewed`。不要自动生成 README、LICENSE 或 `.gitignore`，本地已有这些文件。保留上游 MIT LICENSE，在 README 写明基于 `langchain-ai/open_deep_research`，并列出自己的新增内容。当前副本没有 Git 历史，不能伪称保留了上游 commit 链。
2. 在现有 `.gitignore` 基础上补充 `.odr_runs/`、`.venv/`。`.env` 已在忽略规则中；当前虚拟环境也有内部忽略文件，但项目级规则应明确。运行归档默认可能包含用户输入、搜索内容和请求记录，不应随源码一起暂存。保留不含凭证的 `.env.example`。
3. 在 `~/open_deep_research` 打开终端，初始化并预览将被纳入的文件：

```bash
rtk git init -b main
rtk git status --short --untracked-files=all
rtk git add --dry-run .
```

4. 确认预览只包含准备发布的源码、测试、文档和必要资源；检查笔记/审计材料是否要公开，然后提交：

```bash
rtk git add .
rtk git diff --cached --stat
rtk git diff --cached --check
rtk git diff --cached
rtk git commit -m "feat: establish round-one context and review workflow"
```

5. 将下列占位符替换为刚创建仓库的实际 URL，再推送：

```bash
rtk git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
rtk git remote -v
rtk git push -u origin main
rtk git tag -a v0.1.0 -m "Round one accepted baseline"
rtk git push origin v0.1.0
```

GitHub 的本地源码上传与空仓库流程见 [官方说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。`v0.1.0` 仅在上述问题修复并验收后使用；如果只保存当前未通过版本，应明确标记为 WIP。

6. 要让别人从个人主页看到项目，使用公开仓库，然后到个人主页的 **Customize your pins** 勾选该仓库并保存。无需使用 GitHub Pages。见 [置顶仓库官方说明](https://docs.github.com/en/account-and-profile/how-tos/profile-customization/pinning-items-to-your-profile)。
7. 第一轮 tag 推送后，再创建第二轮开发分支：

```bash
rtk git switch -c feat/round2
```

这样第一轮保留可回到的版本标记，第二轮修改可以通过独立 commit/PR 审查。
