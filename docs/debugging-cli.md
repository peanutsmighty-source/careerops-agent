# CareerOps 命令行复现机制

## 它解决什么问题

网页适合观察流程，pytest 适合防止已知问题再次出现，但它们不能直接回答：

> 数据库里的 ToolCall #17 当时用了什么参数、获得了什么结果，现在重新执行是否还会出现同一个问题？

`careerops-debug` 提供四个命令，把查看现场、保存现场和执行复现分开。

## 使用流程

先列出最近的工具调用：

```powershell
python -m app.debug_cli calls --limit 10
```

查看一个调用的任务、参数、结果和全部相关 Trace：

```powershell
python -m app.debug_cli show 3
```

把当前 SQLite 数据库一致地复制为复现快照，并生成带 SHA-256 的 JSON 清单：

```powershell
python -m app.debug_cli snapshot 3 --output .debug\tool-call-3.db
```

在临时数据库副本里重新执行保存的工具和参数：

```powershell
python -m app.debug_cli replay 3
```

也可以从之前保存的快照复现：

```powershell
python -m app.debug_cli replay 3 --snapshot .debug\tool-call-3.db
```

安装项目后，以上命令也可以写成 `careerops-debug ...`。

## 重放完整场景

Tool Replay 只能重新执行一个工具。跨 API、数据库和 LangGraph 的问题使用 Scenario Runner：

```powershell
python -m app.debug_cli scenarios
python -m app.debug_cli scenario graph-no-skills
python -m app.debug_cli scenario jd-url-dedupe
python -m app.debug_cli scenario graph-human-rejection
```

每个场景都会启动新的 Python 进程，并在导入 FastAPI、SQLAlchemy 和 LangGraph 之前设置独立的数据库与 checkpoint 路径。这一点很重要：如果只在当前进程里临时替换环境变量，已经创建的 SQLAlchemy Engine 和 LangGraph checkpointer 不会跟着切换。

当前场景覆盖：

- 没有 JD 技能时，Graph 必须走 `validation_failed -> block_run`。
- 同一 JD URL 的 tracking 参数不同，仍只能保存一份 Archive。
- 有效学习计划等待人工审批后，被拒绝必须走 `human_review -> block_run`。

## 分支覆盖率

项目使用 Coverage.py 的 branch mode，同时统计某行是否运行以及 `if/else` 的不同出口是否执行：

```powershell
coverage run -m pytest
coverage report -m
```

配置位于 `pyproject.toml`。覆盖率用于发现测试盲区，不作为“代码没有缺陷”的证明。

当前基线（2026-09-01）：

```text
TOTAL branch coverage: 89%
learning_graph:         100%
tool_recovery:          100%
tool_runtime:            92%
tools:                   90%
ingestion:               91%
```

`fail_under = 80` 已作为回归底线。低于 80% 时 `coverage report` 返回失败，但提高门槛仍应依据风险，而不是为了数字给简单属性和声明堆测试。

第一次启用 branch coverage 时，HTML 抓取场景发现了一个真实缺陷：`<script>` 虽然被过滤，但所有块级标签也被压成同一行，导致公司字段吞掉职位和正文。现在 `TextExtractor` 会为标题、段落、列表等块级标签保留换行，同时继续忽略 script/style/noscript。

## Replay 实际做了什么

以 `update_plan_step` 为例：

1. 从真实数据库读取 ToolCall #3 保存的工具名、参数和原始结果。
2. 使用 SQLite backup API 创建一致的临时数据库副本。
3. 在副本里找到同一个 AgentTask 和 PlanStep。
4. 读取副本中的任务策略和历史授权，构造受控执行上下文后重新调用工具。
5. 比较原始与重放的 status、output 和 error。
6. 回滚副本事务并删除临时目录。

真实 `careerops.db` 不会成为重放执行目标。`external_write` 工具也会被拒绝，因为数据库副本隔离不了 GitHub、邮件或云服务上的副作用。

## 当前限制

- 快照是“当前数据库状态”，不是 ToolCall 执行前的历史状态。失败调用通常没有提交业务写入，因此已经足够复现；对成功写入的精确历史复现，之后需要 before-state fixture 或事件溯源。
- 快照包含完整开发数据库，可能含个人数据，不应直接上传到 Issue 或公开仓库。
- 当前快照与 replay 只支持开发环境的文件型 SQLite；未来切换 PostgreSQL 后需要独立的测试库/事务沙箱。
- Replay 只能复现确定性工具层问题。模型为什么选择某个工具，还需要保存模型输入、输出、模型版本和采样参数后才能复现。

## 与主流 Agent 调试的关系

可靠 Agent 通常同时保留三类调试材料：

- Trace：模型和工具按什么顺序执行。
- Checkpoint：工作流执行到了哪里。
- Replay fixture：让某个工具调用在隔离环境里重新运行。

CareerOps 当前已经具备 ToolCall/Trace、LangGraph Checkpoint、工具 Replay 和隔离 Scenario Runner。模型级 replay 会在 Agent Loop 阶段加入。
