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

## Worker 与 lease

`execution_mode: worker` 先持久化 AgentRun，再把 Run ID 投递给进程内线程池。队列消息本身不携带完整 State；worker 重新读取数据库，并用单条条件 `UPDATE` 抢占 `agent_runs` 上的 `lease_owner`、`lease_expires_at` 和 `lease_heartbeat_at`。只有 status 仍为 `running`，且 lease 为空、已过期或属于同一 owner 时才能成功。

worker 在租约周期的三分之一间隔续租。正常完成、失败或进程内异常都会释放；进程崩溃无法执行 finally，其他实例只能等 `lease_expires_at` 后接管。重复投递使用不同 owner，因此同一时刻只有一个任务能进入 Agent workflow。同步 API、直接 AgentLoop helper、手动恢复和启动恢复也使用同一租约协议。

服务启动时扫描超过 `CAREEROPS_STARTUP_RECOVERY_SECONDS`（默认 60 秒）的 `running` Run，并异步投递恢复。恢复分类仍依据持久化 Step、ToolCall 和 checkpoint；lease 只决定谁有权执行，不会把不安全的 external-write 未知结果变成可重试。

## 当前限制

- worker 是进程内线程池，不是 Redis/Kafka 等持久队列；崩溃后的 Run 依靠数据库扫描，而不是消息确认机制。
- heartbeat 失败不能强制中断一个已经阻塞在第三方 SDK 内的 Python 调用。生产实现需要调用超时、协作取消和 fencing token，防止失去租约的旧 worker 提交迟到结果。
- SQLite 可验证条件更新语义，但高并发生产部署仍应使用 PostgreSQL，并结合锁等待、隔离级别和监控压测。
- 外部工具还没有 reconciliation adapter，因此未知结果只能进入人工处理。
- 启动扫描当前只在进程启动时执行；没有常驻调度器周期性重扫。
