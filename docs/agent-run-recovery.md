# CareerOps AgentRun 中断恢复

AgentRun 恢复的目标不是“重新运行一次 Agent”，而是从最后一个已经提交的可靠边界继续，同时避免重复工具副作用。

## 持久化顺序

一次工具步骤按下面的顺序落库：

```text
1. 保存 AgentRunStep 中的模型决定
2. 创建 ToolCallRecord，并在同一次事务中写入 step.tool_call_id
3. Tool Runtime 授权并执行 handler
4. 保存 ToolCall 最终结果
5. 把 ToolCall 结果写回 AgentRunStep.observation
6. 下一轮模型调用
```

第 2 步的原子绑定很重要。它保证只要数据库里存在 ToolCall，就一定能从 AgentRunStep 找到它；如果 Step 没有 `tool_call_id`，则 ToolCall 尚未提交，handler 也还没有开始。

## 恢复分类

Recovery Manager 没有操作系统提供的“断电行号”。它通过查询最后一次成功提交后留下的数据组合来反推边界：

```text
AgentRun.status
  + 最后一个 AgentRunStep.action_type
  + AgentRunStep.tool_call_id 是否存在
  + AgentRunStep.observation 是否存在
  + ToolCallRecord.status / effect
  + 外层 LangGraph checkpoint.next_nodes
= 恢复动作
```

数据库事务是这套推断可信的前提。例如 ToolCall 创建和 `step.tool_call_id` 绑定在同一事务中，因此数据库不会合法地留下“ToolCall 已提交但 Step 没有关联”的中间状态。`tool_call_id` 为空时，Recovery 才能据此断定 handler 尚未开始。对于外部写工具的 `executing`，现有证据不足以判断成功或失败，所以结论必须是 `needs_review`，而不是猜测断电位置。

| 断电后留下的数据 | 恢复动作 | 为什么安全 |
| --- | --- | --- |
| Run 为 `running`，没有 Step | 重新调用模型 | 尚无已提交动作和工具副作用 |
| 已保存 `final_answer` Step | 直接完成 Run | 不再次调用模型 |
| Step 已有 observation | 从下一轮模型调用继续 | 上一步已经完整结束 |
| Tool Step 没有 `tool_call_id` | 执行已保存的工具决定一次 | ToolCall 创建和 Step 绑定是原子事务，handler 尚未开始 |
| ToolCall 已是 succeeded/failed/denied | 恢复 observation 后继续 | 使用持久化结果，不重跑 handler |
| read/internal_write 停在 authorized/executing | 交给 Tool Recovery | 使用原参数、最新授权和已有事务/幂等策略恢复 |
| external_write 停在 executing | Run 进入 needs_review | 外部操作可能已经成功，不能盲目重发 |

成功恢复 AgentRun 后，Recovery Manager 还会检查对应的外层 LangGraph thread。如果 Workflow checkpoint 仍停在 `run_agent` Node，则继续同一个 thread，让它执行 `evaluate_result` 和最终路由。

## API

```text
POST /agent/tasks/{task_id}/recover-agent-runs?stale_after_seconds=60
```

只扫描超过阈值且仍为 `running` 的 Run。返回每个 Run 的恢复动作、当前状态和原因。

## 当前限制

- 恢复由 API 或控制台手动触发，尚未在服务启动时自动扫描。
- 没有 worker lease；多个服务实例同时恢复同一 Run 时仍需要数据库锁或租约。
- 外部工具还没有 reconciliation adapter，因此未知结果只能进入人工处理。
- Graph 尚未版本化；代码升级后恢复旧 checkpoint 需要版本兼容策略。
