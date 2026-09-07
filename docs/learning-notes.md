# CareerOps Agent 学习笔记

本文记录截至当前阶段已经接触和实现的核心知识点，后续开发继续追加。

## 1. 项目定位

- 最终目标是实现一个可运行、可观察、可恢复的 Agent/Harness。
- CareerOps 是真实业务场景：采集 Agent 岗位 JD，分析技能需求，生成学习任务并积累求职证据。
- 项目同时服务于学习和求职，因此每个功能都应对应 Agent 能力、可运行代码、测试和可解释的设计选择。

## 2. Agent Runtime

- `AgentTask` 保存用户目标、约束、成功标准和任务状态。
- `PlanStep` 将任务拆成有顺序的执行步骤，并要求每一步关联成功标准。
- `ExecutionTrace` 记录用户输入、规划、工具调用、评估和未来的上下文压缩事件。
- Runtime 不只是调用模型，还负责状态、权限、工具执行、错误处理、恢复和审计。

## 3. LangGraph

- **State**：一次工作流当前携带的结构化数据，例如目标、技能排名、学习计划和审批结果。
- **Node**：读取或修改 State 的一个执行步骤，例如 `build_plan`。
- **Edge**：Node 之间的转移规则，决定下一步执行哪个 Node。
- **Checkpoint**：Node 执行边界上的 State 快照，用于检查历史和恢复任务。
- **Interrupt/Resume**：Graph 可以在人工审批处暂停，稍后从同一个 thread 和 checkpoint 继续。
- Checkpoint 保存的是工作流状态；Coding Agent 的任务恢复还可能包括会话历史、工作区状态、终端状态和运行中的工具信息，范围更大。

## 4. 目标与记忆

- `GoalContract` 保存长期目标、范围边界和成功标准，防止长任务在上下文变化后偏离。
- working memory 保存当前工作信息，episodic memory 保存发生过的事件，fact memory 保存稳定事实。
- 上下文压缩是从完整历史生成较短的工作上下文，不应取代原始 Trace。
- 压缩时必须保留用户目标、重要约束、已完成事项、未完成事项、关键决定和证据来源。

## 5. LangGraph 与 AutoGen 的理解

- LangGraph 主要用显式 State、Node 和 Edge 表达可恢复工作流。
- AutoGen 更关注消息驱动的 Agent/组件协作和对话式编排。
- Planner、Executor、Reviewer 是职责，不一定都必须实现成独立 Agent。
- Coding Agent 中的 planner 通常可以是 Runtime 内的一种规划能力，不必是另一个拥有独立身份和上下文的 Agent。
- 只有当角色需要独立上下文、工具权限、生命周期或并行工作时，拆成多个 Agent 才可能有价值。

## 6. Tool Calling

- Tool 不只是 Python 函数，还需要稳定名称、描述、参数 JSON Schema、权限等级、标准化结果和调用 Trace。
- `Tool Registry` 保存可用工具定义；`Tool Executor` 负责查找、校验、授权、执行和记录结果。
- Pydantic 在执行前校验模型生成的 arguments，未知字段和非法范围不能进入 handler。
- `success`、`error` 和 `denied` 都是 Agent 可以读取的 observation，不应让预期内的工具错误直接弄崩 Runtime。
- 当前已实现读取 JD、读取技能需求、读取单个职位和更新 PlanStep 等工具。
- `ToolCallRecord` 保存一次调用的当前生命周期；`ExecutionTrace` 保存执行和重放事件历史。
- 工具声明 `effect`、`idempotency_mode` 和 `repeat_policy`，避免给读取与写入工具套用相同重复规则。

## 7. 工具权限

- LLM 只能提出工具调用，不能给自己授权。
- 权限应由服务端根据用户身份、Task Policy、工具风险、资源范围和人工审批计算。
- read 工具通常可以自动允许；内部 write 工具需要更严格策略；投递简历、发送邮件等外部动作应逐次审批。
- 高风险授权应绑定 task、tool、资源、参数摘要、有效期和最大使用次数，而不是只授予宽泛的 `write`。
- 当前 ToolCall 请求不能携带 `granted_permissions`。服务端读取版本化 `TaskPolicy`，并为每次执行或重放保存 `ToolAuthorization` 判定。
- `TaskPolicy` 回答“这个任务原则上可用哪些工具”；`ToolAuthorization` 回答“这一次调用为什么被允许或拒绝”；`ToolCallRecord` 回答“这次操作执行到哪一步、结果是什么”。

## 8. 事务与脏数据

- 数据库事务不是 Tool Calling 专属机制，但 Tool Executor 可以用它保证业务修改和 Trace 一起提交或一起回滚。
- 当前数据库 handler 使用 savepoint；handler 失败时回滚内部修改，再返回错误 observation。
- 数据库事务无法撤销已经发生的外部副作用，例如简历已经发送或邮件已经送达。

## 9. 幂等与未知结果

- 幂等表示相同业务操作重复请求多次，最终只产生一次业务副作用。
- 常见方法是为调用生成稳定的 `idempotency_key`，重复请求直接返回第一次结果。
- `request_fingerprint` 检查相同 key 的重试是否改变了工具、参数或 PlanStep；改变时返回冲突。
- 幂等重放不增加真实执行 attempt，但单独记录 replay 次数和审计事件。
- timeout 只代表调用方没有收到响应，不代表外部操作失败；此时状态应为 `outcome_unknown`。
- 发生未知结果时，应通过状态查询、外部 operation ID、Webhook 或人工确认进行对账，不能盲目重试。
- `status()` 用来确认真实结果；`compensate()` 用来尽可能撤销已发生的操作；它们与幂等相关，但不是幂等本身。
- 无法确认结果且外部系统不支持幂等时，应进入 `needs_review`，由用户检查后决定。

## 10. 当前进度

已经完成：

- 真实公开 JD 归档、去重和结构化解析。
- JD 技能统计和学习路线生成。
- Goal Contract、分层 Memory 和目标对齐记录。
- AgentTask、PlanStep 和 ExecutionTrace。
- LangGraph 工作流、checkpoint、人工中断与恢复。
- Tool Registry、参数校验、read/write 权限演示、事务保护和 Tool Call Trace。
- 持久化 ToolCall 状态、按工具幂等策略、operation key、请求指纹与重放检查。
- 服务端 Task Policy、逐次 Tool Authorization，以及策略撤销后的重放再鉴权。
- 独立 AgentLoopEngine：模型动作、工具 observation、最终答案、最大步数和停止原因均持久化，不依赖 LangGraph。
- 外层 LangGraph Workflow：`run_agent` Node 调用 AgentLoopEngine，再由评估 Node 决定完成或阻塞。
- AgentRun Recovery：原子绑定 Step/ToolCall，按断电位置恢复模型步骤、工具结果和外层 Workflow；外部未知结果进入人工处理。
- 可视化 Runtime Console。

尚未完成：

- 登录用户身份、外部写操作的一次性人工审批。
- 未知结果对账和外部副作用的可靠重试。
- AgentRun 的中断恢复 API，以及异步 worker 执行。
- RAG、上下文压缩执行器、评估系统、MCP 和 Multi-Agent 实验。

## 11. 下一学习阶段

可靠 Tool Execution、受限 Agent Loop 和 Memory Runtime v1 已经完成：

1. 在每次模型调用前装配 GoalContract、任务状态和相关 Memory。
2. working、episodic、fact memory 按类型、重要性、相关性和过期时间筛选。
3. Run 成功后使用稳定 key 幂等记录 episodic outcome。

Memory 是持久化候选信息，Context 是某次模型调用临时选择出的输入，两者不是同一个东西。下一步是为 Context Compaction 明确必须保留的信息、可压缩的信息和可丢弃的信息。

## 命令行调试与复现

- `show` 读取 ToolCall 和 Trace，相当于查看故障现场。
- `snapshot` 使用 SQLite backup 保存一致的数据库副本和哈希清单。
- `replay` 在临时副本中用原工具参数重新执行；服务端根据副本中的 TaskPolicy/ToolAuthorization 恢复执行上下文，并比较 status、output、error。
- 数据库隔离只能隔离本地副作用，不能隔离 GitHub、邮件等外部副作用，因此 CLI 拒绝重放 `external_write`。
- 当前是工具级确定性复现；模型级复现还需要记录 prompt、context、模型版本和推理配置。
- Scenario Runner 在新进程中配置临时业务库和 checkpoint 库，可以复现跨 API、SQLAlchemy、LangGraph 的完整路径。
- Branch coverage 不只检查某行是否执行，还检查条件的不同出口；第一次运行就发现 HTML 块级标签被错误压成一行的解析缺陷。
4. 引入正式数据库迁移，替代仅依赖 `create_all` 的 schema 创建方式。

## 12. Memory Gate v2 学习心得

这次实现把“模型提出一条记忆”和“系统确认并保存一条记忆”明确拆开：

- `MemoryCandidate` 是待审核的信息，不等于长期 Memory。
- `decision` 回答内容是否可信和值得保存，分为 `accept`、`reject` 和 `needs_review`。
- `storage_action` 回答本次是否真的写入数据库，分为 `stored`、`already_stored` 和 `not_stored`。
- 将判断和写入动作分开后，同一 Run 恢复时可以得到 `accept + already_stored`，既表达内容可信，也保证幂等。
- provenance 不能因为模型填写了 ID 就被信任。Runtime 必须查询数据库，核对 Run、Step、ToolCall、Task 和工具名称之间的真实关联。
- 伪造的来源声明可以保留在原始 JSON 中供审计，但不能写入代表可信关系的外键字段。
- Candidate Journal 保存接受、拒绝和待复核结果，让误存和误拒都能被观察、测试和后续评估。

本阶段没有使用模型 Evaluator。最小信息量、敏感值、精确重复、作用域和 provenance 校验均由确定性规则完成，因此可以稳定测试且不产生额外 token。后续只应把语义重复、事实冲突和长期价值等模糊问题交给可选模型 Evaluator。

真实运行验证中，第一次 Demo Run 产生 episodic 和 fact 两条 `accept + stored` Candidate；第二次相同运行产生两条 `reject + not_stored` Candidate，原因为 `duplicate_content`，且没有新增 Memory。这说明 Candidate 审计、内容去重和持久化写入已经形成闭环。

## 13. 发布仓库时的工程习惯

- 初始化仓库前先检查工作目录中的数据库、缓存、日志、密钥和无关样本，不能依赖“看起来像源码”来判断是否适合提交。
- SQLite 在服务运行时除了主数据库，还可能生成 `-wal` 和 `-shm` 文件；忽略 `*.db` 并不会自动忽略这些旁路文件。
- 从其他项目复用 Git 身份和远程命名习惯即可，不应复制其他项目的 `.git` 目录。
- 首次推送前先用 `git status` 和待提交文件列表检查提交边界，再创建远程仓库。

## 14. 可选语义 Evaluator 学习心得

- 确定性规则适合回答“必须满足什么”：来源存在、作用域正确、不能含密钥、不能精确重复。这些规则便宜、稳定、可复现。
- 模型适合回答“看起来意味着什么”：是否值得长期保存、是否是语义重复、是否与已有事实冲突。这类判断不能靠简单字符串比较覆盖。
- 两者不是二选一。正确顺序是先用硬规则缩小范围，再让模型处理模糊候选，最后由 Runtime 校验模型输出。
- Pydantic 结构化输出把自由文本回答限制成固定字段，但它只保证形状正确，不保证事实正确。模型返回的 Memory ID 仍要回数据库核对。
- Evaluator 失败不能拖垮主 AgentRun。fail-closed 会保留 Candidate 并标记 `needs_review`，而不是误存或静默丢弃。
- 模型判断会增加 token 成本，所以 Candidate Journal 要记录调用次数、输入 token、输出 token 和模型版本，为后续 evals 比较质量与成本提供数据。
