# CareerOps Agent Loop 与 Workflow Graph

## 1. 两层架构

CareerOps 将业务工作流和自主执行循环分开：

```text
LangGraph Workflow
load_context -> run_agent -> evaluate_result -> complete / block
                   |
                   +-- AgentLoopEngine
                       model -> tool -> observation -> model
```

Graph 决定任务目前处于哪个业务阶段。`run_agent` Node 只调用 `AgentLoopEngine`，不实现循环细节。Engine 是普通 Python 服务，不导入 LangGraph，可以由 API、CLI、测试或其他编排框架直接调用。

## 2. 一个具体运行

用户任务是“根据真实 JD 选择下一个 Agent 学习任务”。外层 Graph 首先进入 `load_context`，然后在 `run_agent` Node 内启动 AgentRun：

```text
AgentRun Step 1: model -> search_jobs
                 tool  -> observation: 3 jobs

AgentRun Step 2: model -> get_skill_demand
                 tool  -> observation: 5 skills

AgentRun Step 3: model -> final_answer
```

Agent Loop 完成后，Graph 离开 `run_agent`，进入 `evaluate_result`。有最终答案则走 `complete_workflow`，达到上限或失败则走 `block_workflow`。

## 3. AgentLoopEngine 的边界

Engine 负责：

- 组织当前任务、成功标准、可用工具和历史 observations。
- 调用 Model Adapter 获得 `tool_call` 或 `final_answer`。
- 将工具提议交给可靠 Tool Runtime。
- 把工具结果追加为下一轮 observation。
- 执行服务器配置的最大步数。
- 保存 AgentRun、AgentRunStep 和 ExecutionTrace。

Engine 不负责 Graph 的业务阶段、人工审批流程和 Node 路由。它也不能绕过 Tool Registry、参数校验、TaskPolicy、ToolAuthorization 和幂等规则。

## 4. Graph State 与 AgentRun 数据

外层 Graph State 只保存跨 Node 需要的数据：

```text
task_id, run_id, agent_status, final_answer,
evaluation_errors, workflow_status, node_history
```

AgentRunStep 保存 Node 内部每次模型决定和 observation。Graph State 不复制完整 ToolCall 历史，而是通过 `run_id` 引用它。

两种持久化解决不同问题：

- LangGraph checkpoint：工作流到了哪个 Node，下一步是什么。
- AgentRun/AgentRunStep：`run_agent` Node 内部执行了哪些模型和工具步骤。

可以通过 `GET /agent/tasks/{task_id}/agent-runs/{run_id}/checkpoints` 查看外层 Workflow checkpoint。

## 5. 为什么不能只用 Graph

如果模型调用和工具执行直接写死为 LangGraph Node：

- Agent Loop 无法脱离 LangGraph 独立测试。
- 更换编排框架需要重写循环。
- 业务阶段和模型内部迭代混在同一张图里。
- checkpoint 容易同时承担业务恢复和模型步骤审计两种职责。

现在 `start_agent_run()` 可以直接运行 Engine；`start_agent_workflow()` 则创建外层 Graph，并在 `run_agent` Node 中调用相同 Engine。

## 6. 最大步数

如果模型持续执行“搜索职位 -> 再搜索”，普通无限循环会持续消耗时间和费用。`max_steps` 是 Engine 的服务器配置，不在模型控制范围内。

达到上限后，AgentRun 保存：

```json
{
  "status": "max_steps",
  "stop_reason": "max_steps_reached"
}
```

外层 `evaluate_result` 会发现没有最终答案，并把 Workflow 路由到 `block_workflow`。

## 7. Demo 与真实模型

`demo` provider 是确定性策略，用于在没有 API Key 时观察完整 Loop 和稳定测试。它不是 LLM。

`openai` provider 使用 Responses API。无论使用哪个模型，模型都只能返回结构化动作，权限、工具执行和停止条件由 AgentLoopEngine 与 Tool Runtime 控制。

## 8. 当前恢复边界

外层 Graph 会在 Node 边界保存 checkpoint。AgentLoopEngine 会在每次模型决定和工具结果后提交数据库记录。

如果进程在 `run_agent` Node 内突然崩溃：

- 外层 checkpoint 仍可能显示下一步是 `run_agent`。
- AgentRun 可能停在 `running`。
- 已提交的 AgentRunStep 和 ToolCallRecord 会保留。

AgentRun Recovery 现在会根据最后一个完整 Step、ToolCall 状态和外层 checkpoint 判断继续执行、恢复 observation 或转人工处理。成功恢复内部 Loop 后，还会继续同一个外层 Workflow thread。

## 9. 分阶段耗时怎样定位慢请求

每个 AgentRunStep 的 timing_json 会记录：

- context_assembly_ms：筛选和装配本轮 Memory Context。
- request_build_ms：读取任务、策略、工具和 Context，形成完整模型请求。
- model_call_ms：等待模型返回并解析结果。
- tool_call_ms：从进入 Tool Runtime 到获得 observation 的总时间。
- step_total_ms：这个 Step 已知阶段的总时间。

AgentRun 的 timing_json 会汇总所有 Step，并增加 memory_finalize_ms、wall_clock_ms 和 unattributed_ms。wall-clock 包含中断和等待，unattributed 是 wall-clock 减去已知阶段后的剩余时间。如果 model_call_ms 很大，应检查供应商延迟、推理配置和 token 数；如果 tool_call_ms 很大而 ToolCall.duration_ms 很小，应检查鉴权、事务或连接；如果 unattributed_ms 很大，应检查调度、恢复间隔或缺少埋点的阶段。

失败 Run 还会记录 failed_phase。它不能保证在设备瞬间断电时写入，因为进程已经没有机会执行写库；这种情况由仍处于 running 的 Run、最后一个持久化 Step 和恢复扫描共同判断。

## 10. 当前限制

- Agent Loop 仍是同步执行，长耗时模型调用之后应移入 worker。
- Recovery 目前手动触发，尚无启动扫描、worker lease 和多实例互斥。
- Graph 还没有版本管理，修改 Node/Edge 后恢复旧 checkpoint 存在兼容风险。
- 当前把完整 observation 传给下一轮；Context Compaction 阶段会控制长度。
- GoalContract 和经过筛选的长期 Memory 已装配进模型上下文；RAG 和 context compaction 尚未实现。
