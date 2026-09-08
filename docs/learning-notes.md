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

## 15. 语义去重学习心得

- “相似”不等于“重复”。Embedding 只能找出可能相关的候选，最终仍需规则或模型判断两句话表达的是同一事实、互相冲突，还是仅仅主题相近。
- 去重适合使用漏斗架构：便宜的规范化匹配先处理明显重复，Embedding 从大量 Memory 中召回少量候选，模型只比较边界案例。
- Candidate 与旧 Memory 的比较必须遵守 GoalContract、类型和作用域。否则一个任务的临时偏好可能错误删除另一个任务的独立记录。
- 模型只能引用 Runtime 提供的候选 ID。即使结构化输出合法，引用不在候选集合中的 ID 仍是无效判断。
- 重复和冲突的处理不同。重复 Candidate 不落库并指向已有 Memory；冲突 Candidate 保持待复核，直到版本与 supersede 规则决定哪条是当前事实。
- SQLAlchemy `commit` 后对象属性可能过期。测试如果关闭 Session 后仍读取 ORM 对象，应提前保存稳定 ID 或重新查询，避免 `DetachedInstanceError`。

## 16. DeepSeek 模型适配学习心得

- OpenAI 兼容表示请求协议相近，不表示每个能力完全相同。DeepSeek 可以通过 OpenAI SDK 调用 Chat Completions、Tool Calls 和 JSON Output，但不能据此假设它提供 Embeddings。
- Agent Harness 应依赖自己的 `AgentModel` 和 `SemanticMemoryEvaluator` 接口。供应商适配器只负责把统一请求翻译成特定 API，再把结果翻译回来。
- JSON Output 保证返回 JSON，Pydantic 再验证字段、枚举和 ID 类型；Runtime 仍负责核对 ID 和 provenance 是否真实。
- 密钥只从环境变量或被 Git 忽略的本地文件读取。它不能进入 prompt、Trace、数据库、测试快照或异常消息。
- 单元测试应模拟 SDK 响应，验证协议转换而不产生费用；真实连通测试单独执行，并且不能打印请求头或密钥。
- 最小真实连通测试已创建 `AgentRun #14`：模型返回了 `get_skill_demand` 工具调用，并记录 token 用量。它证明网络、鉴权、协议转换和结果解析可以协同工作，但不代表完整业务链路已经验证。
- “允许使用某个模型 API”和“允许把哪些项目数据发送给该模型”是两种权限。完整运行会外发 task goal、Memory Context 和 tool observations，因此 Harness 还需要显式的数据出站策略，而不能只检查 API key 是否存在。

## 17. Agent Runtime 分阶段耗时

- Run 总耗时只能说明“整轮慢”，不能说明慢在模型、Context、工具还是 Memory Evaluator。性能定位需要分阶段计时并使用同一个 run ID、step ID 关联。
- 每个 AgentRunStep 记录 Context 装配、请求构造、模型调用、工具调用和 Step 总耗时；AgentRun 汇总所有 Step，并额外记录 Memory 收尾和无法归因的 wall-clock 时间。
- unattributed_ms 不是错误，它包含数据库提交、调度、恢复间隔以及尚未单独埋点的代码。该值异常大时，说明应该继续增加 span，而不是武断地归因给模型。
- 失败 Run 记录 failed_phase。例如 model_call 表示异常发生在等待或解析模型响应期间，而不是工具 handler。

## 18. 事实版本、Supersede 与 Candidate 审计

以同一个学习任务的技能统计为例：第一次工具观察到“LangGraph 出现在 3 个 JD”，后来重新统计得到“出现在 5 个 JD”。这不是两条可以并列进入 Context 的事实，而是同一个稳定事实在不同时刻的两个版本。

- `AgentMemory` 保存当前值和当前 `version`，供 Context 读取。
- `AgentMemoryRevision` 保存每个版本的内容、来源、provenance、变更原因和有效时间区间，供审计。
- supersede 在同一事务中关闭旧 revision、递增 Memory version、更新当前值并创建新 revision；缺少任一步都会产生“当前值”和历史不一致的半完成状态。
- 旧 Memory 在启动迁移时立即补一条 `legacy_backfill` v1 revision。若等到第一次 supersede 才补，尚未发生变化的旧 Memory 会在 revision API 中没有历史，审计语义不完整。
- 自动 supersede 只接受来源链已经由 Runtime 核验的工具事实。用户偏好和模型判断可能涉及语义误解或高风险覆盖，必须继续拒绝或进入 `needs_review`。
- Candidate Journal 记录的是每一次判断事件，不能按稳定 key 做 upsert。“3 个 JD”的 `stored` 和“5 个 JD”的 `superseded` 都必须保留，否则事后只能看到结果，无法解释系统何时、为何改变事实。

主流 Agent 系统通常也会把当前视图与历史事件分开：当前状态服务低成本读取，revision/event log 服务审计、回放和纠错；模型只提出候选，确定性 Runtime 控制版本号、作用域、来源校验和写入事务。

本次实现的难点是兼容旧 SQLite：删除唯一约束不能用普通 `ALTER TABLE`，因此启动兼容逻辑需要在事务中创建 append-only 新表、复制旧 Journal、替换旧表并恢复索引。测试同时验证旧行保留、重复稳定键可以追加以及迁移可安全重复运行。

面试时可以继续追问：

1. 为什么不能让旧事实和新事实同时进入模型 Context？
2. revision table、事件溯源和普通审计日志有什么区别？
3. 如何处理两个 worker 并发 supersede 同一个 Memory 的版本竞争？
4. 为什么工具事实可以自动更新，而用户偏好通常需要人工确认？
5. 数据库事务能保证哪些一致性，又不能解决哪些外部系统问题？
