# Agent Governance

CareerOps is an Agent/Harness project. Its job-intelligence features are a real-world setting for learning and demonstrating Agent engineering, not an independent CRUD roadmap.

## Goal contract

The active `GoalContract` is persistent, versioned state. It is separate from conversation summaries so a compaction step cannot silently replace the north-star goal with the latest implementation task.

It contains:

- A north-star goal: build a runnable, observable, and evaluable CareerOps Agent Harness.
- Product context: CareerOps turns job data and personal evidence into career-learning work.
- A learning contract: each phase explains the concept, tradeoff, exercise, and difficulty. After substantial technical work, the explanation also includes likely interview questions and concise answer guidance; trivial changes do not generate filler questions.
- Scope guardrails: product work must enable Agent tools, memory, retrieval, evaluation, or observable state.
- Success criteria: the durable checks that each proposal must reference.

Creating a new contract through `POST /agent/goal-contracts` creates the next version and makes it active. The old version stays in the database so past decisions remain explainable.

## Layered memory

`AgentMemory` has three memory types. The type is intentionally explicit because a single free-form summary has no reliable retention policy.

| Type | Purpose | Example |
| --- | --- | --- |
| `working` | Temporary state for the current plan or execution | Current task, tool result, open blocker |
| `episodic` | Durable record of a decision or outcome | A roadmap was deprioritized because it did not advance the Agent Harness |
| `fact` | Stable domain or user facts that can be retrieved later | User target role, JD source trust level |

Memories have a key, source, importance, optional expiry, and contract version. Future context compaction should retain the goal contract first, then select memories by type, relevance, importance, and expiry. It should not summarize all prior turns into one unstructured paragraph.

## Alignment gate

Before the Agent performs a development action, it creates an `AgentAlignment` record. The proposal must name:

- An Agent capability such as `tool_calling`, `memory`, or `evaluation`.
- One exact success criterion from the active contract.
- A plain-language connection to the north-star goal.
- What the user will learn and reproduce.

The current deterministic gate checks the structure of this mapping. It returns `defer` when the capability is unsupported, the criterion is not part of the active contract, or a required explanation is empty. This is deliberately narrow: keyword matching cannot prove that a proposal genuinely advances the goal.

When the LLM runtime is introduced, an evaluator should add a second semantic review: inspect the proposed action and its evidence, then score whether the declared mapping is credible. That evaluator becomes an Agent eval case, not an invisible prompt rule.

## User task runtime

The project goal contract governs development of CareerOps. It is not a substitute for the goal of a real CareerOps user. `AgentTask` is the user-level contract for a long-running request.

An Agent task stores a user goal, constraints, success criteria, status, and the project goal-contract version that governed its creation. `PlanStep` is an ordered unit of work. Every step must reference one exact success criterion from its parent task, name an Agent capability, explain its connection to the user goal, and state a learning outcome.

`ExecutionTrace` records user input, plan creation, future tool calls, evaluation, and context compaction. The trace is the source of truth for an Agent run; a compacted context is a derived view and must never become the only record of what happened.

## API

- `GET /agent/goal-contract`: read the active contract.
- `POST /agent/goal-contracts`: create a new active contract version.
- `POST /agent/memories`: store a typed memory under the active contract.
- `GET /agent/memories`: list active-contract memory, optionally filtered by `memory_type`.
- `POST /agent/alignments`: evaluate and record a proposed action before execution.
- `POST /agent/tasks`: create a user-level task contract and its initial trace event.
- `GET /agent/tasks` and `GET /agent/tasks/{task_id}`: retrieve durable task state.
- `POST /agent/tasks/{task_id}/plan`: validate and save ordered plan steps.
- `POST /agent/tasks/{task_id}/traces`: append a runtime trace event.
