# CareerOps Memory Runtime

## Memory、State 和 Context 的区别

以任务“根据真实 JD 选择下一个学习项目”为例：

- State 是这一次 AgentRun 的精确进度，例如已经调用了 `search_jobs`，下一步要调用模型。
- Memory 是跨 Run 保存的信息，例如用户目标岗位是 Agent Engineer，或者上次 Run 得出了什么结论。
- Context 是调用模型前临时装配的输入，其中包含任务、GoalContract、选中的 Memory、工具列表和本次 Run 的 observations。

数据库里的所有 Memory 不会直接全部塞给模型。Memory Runtime 负责从持久化记忆构造本轮 Context。

## Memory 是怎样记录的

当前有两条写入路径：

1. 用户或应用通过 `POST /agent/memories` 明确写入。调用者提供类型、作用域、key、内容、重要性和可选过期时间，Runtime 验证作用域后保存。
2. 任务创建或后续 `user_input` Trace 写入时，free-text Candidate Builder 从用户明确表达中提取 preference、fact、correction 和 learning episode，再交给 Memory Evaluator。
3. AgentRun 完成后，Run Candidate Builder 从最终答案提出 episodic 候选，Memory Evaluator 审核后决定保存或拒绝。模型不能直接绕过审核写数据库。

手动入口不是为了让用户维护所有记忆，而是控制面：用户可以明确保存关键事实、纠正 Agent，测试作用域，并审计自动写入策略。正常运行的目标仍然是由事件和 Evaluator 自动创建大部分 Memory。

三种类型表达“这条信息是什么”：

| 类型 | 含义 | 例子 | 生命周期 |
| --- | --- | --- | --- |
| `fact` | 相对稳定、后续仍成立的信息 | 目标岗位是 Agent Engineer | 长期，直到修改或失效 |
| `episodic` | 某次任务发生过的结果或经验 | Tool Runtime 阶段验证了断电恢复 | 长期经历，可被后续任务参考 |
| `working` | 仍在处理的草稿、假设或下一步 | 正在比较两种 compaction 策略 | 仅 task/run 内可见，任务结束后应清理或过期 |

作用域表达“谁可以看到”：

| 作用域 | 可见范围 | 允许类型 |
| --- | --- | --- |
| `contract` | 同一 GoalContract 下的所有任务 | fact、episodic |
| `task` | 只有指定 AgentTask | fact、episodic、working |
| `run` | 只有指定 AgentRun | working |

类型和作用域是两个维度。`working` 不是因为优先级低，而是因为它尚未成为稳定知识；因此系统禁止 contract-scoped working memory。

## 生命周期

Memory 有 `active` 和 `retired` 两个状态。Context 只召回 active Memory。

- 成功的 `get_skill_demand` 工具结果会先形成 run-scoped working memory，供同一 Run 的后续模型步骤使用。
- AgentRun 成功完成时，白名单 promotion policy 把这条 provenance 已验证的摘要提升为 task-scoped fact，然后退休原 working memory。
- AgentRun 失败或达到 step limit 时，该 Run 的 working memory 只退休，不晋升。
- AgentTask 完成或取消时，该 Task 下仍 active 的 working memory 自动退休。
- 用户可以通过 `POST /agent/memories/{memory_id}/retire` 明确撤回一条 Memory。
- `expires_at` 只是适合时间敏感信息的辅助边界。到期记录不会进入 Context，但事实是否仍成立不能只靠时间判断。

退休采用软删除：内容、来源、退休时间和原因继续保留，便于审计，不再影响模型。当前晋升仅支持 `verified_skill_demand_fact_v1` 白名单策略；通用 working-memory 总结和后台物理清理尚未实现。

## Working Memory 总结与晋升

例如 `get_skill_demand` 返回结构化技能统计后，Runtime 不直接把整个 tool output 当长期事实。它先生成一条内容受控的 working Candidate，只保留前五项技能、job count 和 requirement count，并绑定真实 Run、Step、ToolCall 和工具名。通过 Gate 后，这条 Memory 使用 run scope，因此下一模型步骤可以引用，但其他 Run/Task 不可见。

只有 Run 返回 final answer 并进入 `completed`，同一确定性摘要才生成 task-scoped fact Candidate；provenance 中记录 `promotion_policy` 和 `promoted_from_working_memory_id`，再次经过 Gate 后写入。随后原 working memory 被软退休。若 Run 失败或达到最大 step，结束路径不会生成 promotion Candidate，只记录退休原因。

当前没有让模型自由决定 `promotion_eligible=true`，因为模型可以把假设标成“已验证”。晋升资格由 Runtime 的工具白名单、成功 observation、真实 provenance 和成功终态共同决定。这个 v1 故意只覆盖 `get_skill_demand`，证明生命周期闭环而不把任意工具输出长期化。

Task completion 目前有两条来源：控制面显式把 Task PATCH 为 `completed`，或者 deterministic learning graph 返回 completed。两者都会在提交 Task 状态的同一事务中退休 task-scoped working memory。当前 API 尚未逐条验证 `success_criteria` 的证据，因此 `completed` 仍是受信控制面声明，而不是通用自动验收结论。

事实变化使用版本覆盖，而不是让同一稳定 key 的旧值与新值同时保持 active。例如 `task:12:skill-demand` 从“LangGraph 出现在 3 个 JD”变为“出现在 5 个 JD”时，`AgentMemory` 原地更新为 version 2，version 1 和 version 2 都写入 `AgentMemoryRevision`。旧 revision 的 `valid_to` 被关闭，只有 `AgentMemory` 的当前内容进入 Context。

自动 supersede 被刻意限制为 provenance 已验证的 `tool:*` fact，并且新旧记录必须属于同一 scope、task 和 run。用户事实、模型推断和模糊冲突仍然拒绝覆盖或进入 `needs_review`，不能因为模型声称“这是更新值”就退休旧事实。

## 读取流程

每次模型调用前，`assemble_memory_context` 执行以下步骤：

```text
当前 AgentTask
  -> 找到创建任务时绑定的 GoalContract 版本
  -> 读取 contract Memory、当前 task Memory 和当前 run Memory
  -> 排除 expires_at 已到期的记录
  -> 根据 importance、memory_type 和任务关键词相关性排序
  -> 应用 memory_limit 和 memory_token_budget
  -> 把结果放入 AgentModelRequest.memory_context
```

模型请求随后保存在 `AgentRunStep.model_request`。因此调试时可以查看模型当时实际获得了哪些记忆，而不是根据当前数据库内容猜测。

## 写入流程：Memory Evaluator

### Free-text Candidate Builder

例如用户输入“我的目标岗位是 Agent Engineer”，系统先创建真实的 `user_input` ExecutionTrace，然后由确定性双语规则生成结构化 proposal：类别为 `fact`、Memory 类型为 `fact`、稳定 key 为 `user-target-role`，并保存命中的原文、规则版本和 Trace ID。目标岗位使用稳定 key，后续纠正才有机会与同一事实建立冲突或版本关系；其他内容使用规范化文本哈希，避免暴露原文到 key。

Builder 只负责召回可能值得记忆的内容，不负责判定是否应成为 Durable Memory。proposal 会转换成 `source=user_input` 的 MemoryCandidate，并经过既有 provenance、敏感信息、最小信息量、重复和语义审核。当前默认没有自动注入 Semantic Evaluator，所以有效的用户输入 Candidate 会进入 `needs_review + not_stored`，可在 Candidate Journal 审阅，不会静默写入长期记忆。

当前规则只接受明确的第一人称表达，并让 correction 规则优先于嵌套的普通 fact 规则。这样牺牲隐式偏好召回，换取更低的误报和零额外模型调用。`python -m app.memory_benchmark` 使用中英文正例及第三方/命令式负例报告 extraction precision、recall 和 false-positive rate。未来可以把主 Agent 的结构化输出作为 piggyback proposal，或后台批处理模糊文本，但仍必须通过同一个 Gate。

AgentRun 成功产生最终答案后，不再直接写入长期 Memory，而是先生成候选并审核：

```text
AgentRun final answer
  -> Candidate Builder：生成 episodic 候选和 provenance
  -> Deterministic Evaluator：核验真实来源链，再检查信息量、敏感值和重复内容
  -> accept：允许成为 Memory
     reject：确定不应保存
     needs_review：证据不足，等待人或后续模型复核
  -> storage_action：stored / superseded / already_stored / not_stored
  -> 每种结果都写入 MemoryCandidateRecord 和 ExecutionTrace
```

`memory_key = agent-run:{run_id}:outcome` 在同一个 GoalContract 和 memory type 下唯一，防止同一 Run 在恢复时重复写入。Evaluator 还会比较内容，防止不同 Run 把相同结论反复写入。

`provenance` 不是 Candidate 自己声称“我有来源”就算有效。`agent_runtime` 来源会查询真实 AgentRun 并核对 task、provider 和 model；`tool:*` 来源会继续核对 AgentRunStep 和 ToolCall 是否存在、是否属于同一个 task、工具名是否一致。模型给出的未知 ID 会被拒绝，原始声明仍留在 provenance 供审计，但不会写进可信外键列。来源类型暂时无法验证时，Candidate 进入 `needs_review`。

Candidate Journal 把“是否值得记住”和“本次是否执行写入”分开。例如恢复同一个 Run 时，判断仍是 `accept`，但写入动作是 `already_stored`；可信工具事实变化时则是 `accept + superseded`。这样既不重复插入，也不会误称这条内容被拒绝。

Journal 自身是 append-only 审计记录。同一稳定 key 和同一 Evaluator 版本的后续评估会新增行，而不是更新旧行。因此“3 个 JD”的首次 `stored` 决策和“5 个 JD”的后续 `superseded` 决策可以同时解释。旧 SQLite 开发库启动时会移除历史 upsert 唯一约束并保留已有行。

审核先执行确定性规则，因此敏感信息、无效作用域、精确重复和伪造来源不会进入模型判断。对于来源已经由真实 `user_input` Trace 和原文证据验证、但长期价值仍不明确的 Candidate，可选 Semantic Evaluator 才会被调用。

Semantic Evaluator 使用 Pydantic 结构化输出，同时给出类型、长期价值、语义重复 Memory ID、冲突 Memory ID 和结论。Runtime 会再次验证模型引用的 ID 是否确实来自本次提供的 Memory 列表；模型不能虚构 ID，也不能推翻确定性拒绝。模型超时或解析失败时采用 fail-closed：Candidate 保持 `needs_review`，不会写入 Memory。

每次语义判断将 Evaluator 版本、模型调用次数、输入 token 和输出 token 写入 Candidate Journal。默认自动运行路径仍不调用语义模型；只有显式注入 Evaluator 且规则给出 `semantic_evaluation_required` 时才付出这部分 token 成本。

语义去重不会把全部 Memory 直接塞给模型。系统先按 GoalContract、类型、active 状态和作用域筛选，再执行以下漏斗：

    规范化内容相等
      -> 直接 reject，不调用 Embedding 和模型
    Embedding 或词法相似度召回
      -> 最多选择 8 条可能相关的 Memory
    模型比较 Candidate 与召回结果
      -> duplicate：reject，并记录已有 Memory ID
      -> conflict：needs_review，不覆盖旧事实
      -> related but different：继续按长期价值决定

Embedding 适配器是可选依赖。没有配置时使用确定性词法 Jaccard 召回；配置 OpenAI Embedding 后会记录 `embedding_calls`、`embedding_tokens`、模型版本、候选 ID 和相似度。Embedding 只负责缩小比较范围，不能独立删除 Memory。召回失败时不会继续让模型凭空判断。

Semantic Evaluator 同时支持 OpenAI Structured Outputs 和 DeepSeek JSON Output。DeepSeek 的 JSON 会在本地再次经过 Pydantic 校验；DeepSeek Chat API 不承担 Embedding 召回。

Run 完成、审核结果、可选的 episodic memory 和 memory trace 在同一个数据库事务中提交。不会出现 Run 已标记完成，但审核记录丢失的半完成状态。

## 为什么不能读取全部记忆

不做筛选会带来四类问题：

1. 过期计划会误导模型。
2. 无关内容占用模型上下文。
3. 重要目标可能被大量低价值历史淹没。
4. 同一次任务在数据增长后得到不可预测的输入规模。

当前使用确定性词法相关性，优点是便于理解、复现和测试。以后接入 embedding/RAG 时，向量召回替换候选排序，但过期过滤、预算、GoalContract 隔离和审计结构继续保留。

## API 和控制台

```text
GET /agent/tasks/{task_id}/memory-context
```

可选参数：

- `memory_limit`：最多选择多少条记忆。
- `memory_token_budget`：选中 Memory 序列可占用的最大估算 token 数。

Runtime Console 的 `MEMORY RUNTIME` 区域显示同一结果。它是预览，不会调用模型，也不会修改记忆。

`MEMORY STORE` 区域用于创建 Memory，并立即查看完整记录、作用域和生命周期状态。这和 Memory Context 是两个视图：Store 展示持久化记录，Context 展示下一次模型调用实际选中的子集。

`CANDIDATE JOURNAL` 展示自动路径产生的每条 Candidate，包括完整内容、provenance、`accept/reject/needs_review`、原因、Evaluator 版本、token 用量和最终 Memory ID。可用 `GET /agent/memory-candidates?task_id=...` 查询同样的数据。

新建成功后，控制台会持续显示“新 Memory 已创建”、高亮新记录，并在右侧展示整条持久化内容。这不只是短暂的 toast，直到用户选择其他 Memory 才消失。

## Memory 创建策略

项目运行时以自动创建为主，手动创建是控制面：

1. 自动路径：AgentRun 或其他可观测事件产生 Memory Candidate，经 Evaluator 审核后写入。当前已实现成功 Run 自动产生 episodic candidate，并从 `get_skill_demand` 的结构化工具结果生成 fact candidate。
2. 手动路径：用户明确指定长期目标、纠正错误事实，或开发者调试 Memory Runtime 时使用。来自页面的记录会标记 `source=runtime_console`。
3. 最终目标：自动 Candidate Builder 提取 episodic/fact/working 候选，确定性规则负责隐私、重复、作用域和冲突校验，人只处理纠错和高风险冲突。

## 当前限制

- 相关性还是词法匹配，没有 embedding 和混合检索。
- token 预算使用 LangChain 的本地近似计数器，适合模型调用前的确定性预检；供应商返回的真实 usage 会在调用后单独记录，两者可能因 tokenizer 和消息包装不同而有偏差。
- working memory 已有 task/run 作用域，并会在对应 Run/Task 终止时软退休；尚未实现自动总结和晋升。
- Candidate Builder 目前支持 Run outcome 的 episodic 候选和结构化技能需求的 fact 候选；还没有模型驱动的自由文本 fact/preference 提取。
- 当前审核已检查来源/作用域完整性、最小信息量、敏感值、规范化重复和可选 Embedding/模型语义重复；provenance 已验证的同 scope 工具事实支持自动 supersede，用户事实和模型提出的模糊冲突仍进入拒绝或人工复核，尚未实现人工冲突处理动作。
- Context Compaction 已在 AgentLoop 模型调用前接入：observation 超过所分配的 token 预算后使用 LangChain `trim_messages` 保留近期内容，并用确定性摘要替代较旧 observation；GoalContract、任务约束、完成动作、未解决 blocker、运行 ID 和下一动作是受保护字段。
- Runtime Console 和 `GET /agent/tasks/{task_id}/agent-runs/{run_id}/context-compaction` 同时展示原始输入、压缩结果、保留项、移除项和原因；触发时完整审计还会持久化到 AgentRunStep 与 Trace。
- 尚未实现后台 consolidation、遗忘和分层压缩。

## Token-aware Context 预算

AgentLoop 在调用模型前使用同一近似计数器拆分并记录四个输入分区：

```text
模型 Context 总预算
  - 输出预留
  = 输入预算
      ├── prompt：系统指令、目标、约束、成功标准和 execution_context
      ├── memory：GoalContract、检索元数据和选中 Memory
      ├── tools：当前允许工具的完整 schema
      └── observations：近期原文和旧历史的确定性摘要
```

默认总预算为 8192 tokens，输出预留 1024。可用 `CAREEROPS_MODEL_CONTEXT_TOKENS` 和 `CAREEROPS_RESERVED_OUTPUT_TOKENS` 配置。Runtime 先计算不可压缩的 prompt、空 Memory 外壳和工具 schema；如果这些受保护输入已经超过预算，Run 会在模型调用前失败。剩余空间中最多三分之一且不超过默认 768 tokens 分配给 Memory，其余交给 observations。最终请求必须满足 `estimated_input_tokens <= input_token_budget`。

`AgentRunStep.model_request.context_token_budget` 保存本步预算、各分区估算、剩余空间和 compaction 节省量。Run 完成后，`timing_json.token_usage` 汇总估算 Context、供应商实际报告的 Agent 模型输入/输出、compaction 输入/输出/节省，以及独立的 Memory Evaluator/Embedding usage。控制台在选中 Run 时展示最近一步的预算。

这仍是预检而非供应商 tokenizer 的绝对保证：近似计数器不完全了解不同 API 的消息包装。输出预留会作为 OpenAI `max_output_tokens` 或 DeepSeek `max_tokens` 传入；后续发现估算长期偏低时，应根据 provider usage 做安全余量校准。

## 最小质量基线

`python -m app.memory_benchmark` 在隔离的内存数据库中运行真实 Evaluator 和 Context assembler，不调用外部模型。当前标注覆盖可信 Runtime 候选、敏感值、信息不足、伪造 provenance、未知来源、需要语义复核的用户输入，以及相关、无关和过期 Memory。

报告包含 Candidate precision/recall、三分类准确率、retrieval recall，以及压缩前后的关键约束保留率。两项 retention 使用同一标签，验证关键约束通过 compactor 后仍完整存在。小型确定性样本得到 1.0 只能证明回归规则符合标签，不能代表真实分布上的泛化质量。
