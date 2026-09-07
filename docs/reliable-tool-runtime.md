# CareerOps 可靠 Tool Runtime V2

## 1. 为什么不能只执行函数并写 Trace

V1 的流程是：校验参数、执行 handler、写入 `ExecutionTrace`。它能说明过去发生过什么，但不能可靠回答“这次 ToolCall 现在是什么状态”。进程在 handler 前后中断时，Trace 可能还没有写入；收到网络重试时，也没有稳定对象可以判断是否已经执行。

V2 新增 `ToolCallRecord`，把一次工具调用变成拥有身份和生命周期的持久化对象：

```text
proposed -> authorized -> executing -> succeeded / failed
                  \-> denied
```

- `ToolCallRecord` 保存当前结论、参数、key、attempt、output 和 error。
- `ExecutionTrace` 保存不可变事件历史，包括真实执行和幂等重放。
- LangGraph State 可以引用 `tool_call_id`，但不需要复制所有执行细节。

如果没有独立 ToolCall 状态表，恢复进程只能从零散 Trace 推测状态；如果直接修改旧 Trace，又会破坏审计历史。

## 2. 为什么每个工具声明不同策略

Registry 现在声明：

```text
effect               read / internal_write / external_write
idempotency_mode     none / operation_key
repeat_policy        always_allow / naturally_idempotent / business_unique
```

当前配置：

| Tool | Effect | Idempotency | Repeat policy |
| --- | --- | --- | --- |
| `search_jobs` | read | none | always_allow |
| `get_skill_demand` | read | none | always_allow |
| `get_job` | read | none | always_allow |
| `update_plan_step` | internal_write | operation_key | naturally_idempotent |

相同参数的两次搜索可能看到不同数据，所以应该产生两个 ToolCall。设置相同 PlanStep 状态的网络重试不应该再次执行外围副作用，所以使用稳定 operation key。

如果所有工具都强制按参数去重，合法的重复查询、提醒和同内容笔记会被错误拦截；如果所有工具都允许重复，写工具在超时重试时可能重复发送、扣费或追加数据。

## 3. Operation key 与请求指纹

两者解决不同问题：

- `idempotency_key` 标识“同一次业务操作”。
- `request_fingerprint` 是规范化 tool name、arguments 和 plan step 后的 SHA-256，用来检查重试有没有偷偷改变请求。

判断规则：

```text
新 key                         -> 创建并执行新 ToolCall
已有 key + 相同 fingerprint   -> 返回原 ToolCall，不重跑 handler
已有 key + 不同 fingerprint   -> 409 Conflict
```

不能只用参数哈希作为 key，因为两个参数完全相同的新业务意图可能都合法。Runtime 在 ToolCall 首次创建时生成 operation key；调用方重试时复用这个 key，新意图则创建新 key。

数据库使用 `(task_id, idempotency_key)` 唯一约束。应用层预检查提供清晰响应，数据库约束处理两个并发请求同时通过预检查的竞争窗口。

## 4. 重放为什么仍然写事件

幂等重放不会再次调用 handler，也不会增加 `attempt_count`，但会：

- 增加 `replay_count`。
- 写入状态为 `replayed` 的 `ExecutionTrace`。
- 返回原 ToolCall 的 output 和 trace identity。

完全不记录重放会让运维人员无法解释客户端为什么多次收到同一结果；把重放当成真实 attempt 又会误导重试指标。分开记录 attempt 和 replay 可以同时保持安全与可观察性。

## 5. 为什么重放也要重新鉴权

已有结果不等于当前调用方有权读取结果。Runtime 在比较 fingerprint 和返回 output 前先检查当前权限。否则知道 operation key 的低权限调用方可能读取历史写工具结果，或者利用不同的错误响应推测敏感参数。

当前 Runtime 会读取服务端 `TaskPolicy`，并为首次执行和每次重放分别写入 `ToolAuthorization`。因此客户端即使伪造 `granted_permissions`，请求也会因为未知字段而被拒绝。策略被撤销后，同一个 operation key 也不能借历史结果绕过权限。

## 6. 事务边界

ToolCall 先以 `proposed` 持久化，再转为 `authorized` 和 `executing`。内部数据库 handler、最终 ToolCall 结果和执行 Trace 在同一个提交边界完成。

这样进程中断时：

- handler 提交前中断：业务修改回滚，ToolCall 保留在可诊断状态。
- handler 成功并提交：ToolCall 结果与执行 Trace 一起存在。
- 同 key 重试：读取已有记录，不再次执行。

外部 API 副作用仍不能由本地事务回滚。后续需要 `outcome_unknown`、外部 operation ID、状态查询和对账 worker。

## 7. 常见 Agent 怎么处理

多数 Agent SDK 或编排框架会提供工具 schema、模型产生 tool call、执行回调、消息/事件记录，有些还提供 checkpoint 和人工审批。它们通常不会自动理解每个业务工具的唯一约束，也不能替外部 API 创造幂等能力。

生产系统通常在 SDK 外围增加 Harness 层：

- 持久化 run、step 和 tool call identity。
- 服务端 authorization policy。
- 唯一 idempotency key 和请求指纹。
- attempt、retry 和 replay 事件。
- 数据库事务或 outbox。
- 外部操作对账与人工处理队列。

框架负责“如何编排调用”，业务 Runtime 负责“这次调用是否安全、是否重复、是否完成”。CareerOps 当前正在学习和实现后者。

## 8. 当前边界与下一步

当前版本已完成持久化 ToolCall、按工具策略、operation key、fingerprint、重放、冲突、服务端 TaskPolicy、逐次 ToolAuthorization 和 UI 策略控制。

尚未完成：

1. 基于登录用户身份的策略管理，以及外部写操作的一次性人工审批。
2. 为外部工具注册 reconciliation adapter，让 `outcome_unknown` 可以自动查询外部结果。

## V3：中断恢复

`POST /agent/tasks/{task_id}/recover-tool-calls` 会扫描超过 60 秒仍停在中间状态的调用。恢复不是统一重跑，而是先看副作用类型：

| 已保存状态/工具类型 | 恢复动作 | 原因 |
| --- | --- | --- |
| `proposed` | `needs_review` | 尚未完成授权，运行时不能替用户授权 |
| `authorized` / `executing` + `read` | 使用原参数重试 | 读取不会产生业务副作用 |
| `authorized` / `executing` + `internal_write` | 在新事务中重试 | 上一次业务写入与成功记录在同一事务提交；中途崩溃时二者一起回滚 |
| `executing` + `external_write` | `outcome_unknown` | 外部操作可能成功，不能根据本地超时直接重发 |
| `outcome_unknown` 且没有对账适配器 | `needs_review` | 需要用户或外部查询接口确认真实结果 |

每次恢复决定都会生成一条带 `recovery: true` 的 `ExecutionTrace`。这让恢复过程可审计，也让 Agent Loop 之后只能消费已经明确的工具结果。
3. 对 schema 变更使用正式数据库迁移工具。
4. LLM 根据 observation 决定下一动作的 Agent Loop。
