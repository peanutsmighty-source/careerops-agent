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
2. AgentRun 完成后，Candidate Builder 从最终答案提出 episodic 候选，Memory Evaluator 审核后决定保存或拒绝。模型不能直接绕过审核写数据库。

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

- AgentRun 完成、失败或达到 step limit 时，该 Run 的 working memory 自动退休。
- AgentTask 完成或取消时，该 Task 下仍 active 的 working memory 自动退休。
- 用户可以通过 `POST /agent/memories/{memory_id}/retire` 明确撤回一条 Memory。
- `expires_at` 只是适合时间敏感信息的辅助边界。到期记录不会进入 Context，但事实是否仍成立不能只靠时间判断。

退休采用软删除：内容、来源、退休时间和原因继续保留，便于审计，不再影响模型。尚未实现 working memory 晋升、事实 supersede 链和后台物理清理。

## 读取流程

每次模型调用前，`assemble_memory_context` 执行以下步骤：

```text
当前 AgentTask
  -> 找到创建任务时绑定的 GoalContract 版本
  -> 读取 contract Memory、当前 task Memory 和当前 run Memory
  -> 排除 expires_at 已到期的记录
  -> 根据 importance、memory_type 和任务关键词相关性排序
  -> 应用 memory_limit 和 memory_char_budget
  -> 把结果放入 AgentModelRequest.memory_context
```

模型请求随后保存在 `AgentRunStep.model_request`。因此调试时可以查看模型当时实际获得了哪些记忆，而不是根据当前数据库内容猜测。

## 写入流程：Memory Evaluator

AgentRun 成功产生最终答案后，不再直接写入长期 Memory，而是先生成候选并审核：

```text
AgentRun final answer
  -> Candidate Builder：生成 episodic 候选和 provenance
  -> Deterministic Evaluator：核验真实来源链，再检查信息量、敏感值和重复内容
  -> accept：允许成为 Memory
     reject：确定不应保存
     needs_review：证据不足，等待人或后续模型复核
  -> storage_action：stored / already_stored / not_stored
  -> 每种结果都写入 MemoryCandidateRecord 和 ExecutionTrace
```

`memory_key = agent-run:{run_id}:outcome` 在同一个 GoalContract 和 memory type 下唯一，防止同一 Run 在恢复时重复写入。Evaluator 还会比较内容，防止不同 Run 把相同结论反复写入。

`provenance` 不是 Candidate 自己声称“我有来源”就算有效。`agent_runtime` 来源会查询真实 AgentRun 并核对 task、provider 和 model；`tool:*` 来源会继续核对 AgentRunStep 和 ToolCall 是否存在、是否属于同一个 task、工具名是否一致。模型给出的未知 ID 会被拒绝，原始声明仍留在 provenance 供审计，但不会写进可信外键列。来源类型暂时无法验证时，Candidate 进入 `needs_review`。

Candidate Journal 把“是否值得记住”和“本次是否执行写入”分开。例如恢复同一个 Run 时，判断仍是 `accept`，但写入动作是 `already_stored`。这样既不重复插入，也不会误称这条内容被拒绝。

审核先执行确定性规则，因此敏感信息、无效作用域、精确重复和伪造来源不会进入模型判断。对于来源已经由真实 `user_input` Trace 和原文证据验证、但长期价值仍不明确的 Candidate，可选 Semantic Evaluator 才会被调用。

Semantic Evaluator 使用 Pydantic 结构化输出，同时给出类型、长期价值、语义重复 Memory ID、冲突 Memory ID 和结论。Runtime 会再次验证模型引用的 ID 是否确实来自本次提供的 Memory 列表；模型不能虚构 ID，也不能推翻确定性拒绝。模型超时或解析失败时采用 fail-closed：Candidate 保持 `needs_review`，不会写入 Memory。

每次语义判断将 Evaluator 版本、模型调用次数、输入 token 和输出 token 写入 Candidate Journal。默认自动运行路径仍不调用语义模型；只有显式注入 Evaluator 且规则给出 `semantic_evaluation_required` 时才付出这部分 token 成本。

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
- `memory_char_budget`：记忆 key 和 content 可占用的最大字符数。

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
- 字符预算只是 token 预算的确定性近似。
- working memory 已有 task/run 作用域，并会在对应 Run/Task 终止时软退休；尚未实现自动总结和晋升。
- Candidate Builder 目前支持 Run outcome 的 episodic 候选和结构化技能需求的 fact 候选；还没有模型驱动的自由文本 fact/preference 提取。
- 当前审核已检查来源/作用域完整性、最小信息量、敏感值和精确重复，并提供可选语义重复、长期价值和冲突判断；尚未实现 `needs_review` 的人工处理动作和事实 supersede 链。
- Context Compaction 尚未实现；实现后 Runtime Console 必须同时展示压缩前输入、压缩后 Context、保留项和丢弃/摘要原因。
- 尚未实现合并、冲突检测、遗忘和 context compaction。
