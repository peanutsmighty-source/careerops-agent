# CareerOps Agent

CareerOps Agent is a project for designing a runnable, observable, and evaluable AI Agent Harness. Job intelligence and career planning are its real-world setting: the Agent will use public JD data and personal project evidence to help a user learn for and pursue AI Agent engineering roles.

See [TODO.md](TODO.md) for the prioritized capability backlog, source of each requirement, and acceptance criteria.

The first release builds trustworthy Agent foundations: job data, structured skill requirements, project evidence, learning tasks, a persistent goal contract, layered memory, and a deterministic goal-alignment gate. Stage 2 adds public job-source ingestion, JD archiving, duplicate detection, and deterministic structured parsing. Later stages add the LLM runtime, tool calling, retrieval, context compaction, evaluation, and workflow orchestration.

## Scope

### In scope now

- Job postings and their skill requirements.
- Canonical skills and aliases.
- Project evidence that supports a skill claim.
- Gap tasks with explicit acceptance criteria.
- Public job-source registration.
- JD archiving with canonical URL and content-hash deduplication.
- Rule-based JD parsing for AI application and Agent engineering skills.
- A documented REST API and automated tests.
- A versioned goal contract that protects the Agent/Harness north-star goal across long-running work.
- Typed Agent memory and a pre-execution alignment record for every proposed development action.

### Not in the first release

- Automated job applications or resume submission.
- Unattended crawling of restricted job platforms.
- Fully autonomous career recommendations without user review.
- An interview simulator.
- Scraping behind login walls or sources that disallow automated collection.

## Architecture

```text
FastAPI API -> SQLAlchemy models -> SQLite (development) / PostgreSQL (later)
```

The data model deliberately keeps source text and evidence links. Matching and AI-generated recommendations in later phases must remain explainable and reviewable.

## Run locally

Requires Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the visual Runtime Console. Use
`http://127.0.0.1:8000/docs` when you want to inspect or call the raw API.

## Test

```powershell
pytest
python -m coverage run -m pytest
python -m coverage report -m
```

## Debug a persisted tool call

```powershell
python -m app.debug_cli calls --limit 10
python -m app.debug_cli show 3
python -m app.debug_cli replay 3
python -m app.debug_cli scenarios
python -m app.debug_cli scenario graph-no-skills
```

Replay runs against a temporary SQLite copy and refuses external-write tools. Read [the command-line debugging guide](docs/debugging-cli.md) for snapshots, safety boundaries, and current limitations.

## Data model

- `JobSource`: a public source to monitor, such as a company careers page or public job board.
- `JobArchive`: the raw JD text, source URL, normalized URL, content hash, parse status, and structured parse result.
- `Job`: a job posting and its public source URL.
- `Skill`: a canonical, categorized skill.
- `JobRequirement`: a skill requirement extracted from one job, including the supporting JD sentence.
- `Project`: a project used as career evidence.
- `ProfileEvidence`: a concrete project, code, demo, or document that supports a skill.
- `GapTask`: a manually reviewed action that improves a skill or its evidence.
- `GoalContract`: versioned north-star goal, scope guardrails, and success criteria that compaction must preserve.
- `AgentMemory`: typed working, episodic, or fact memory with contract/task/run visibility scope.
- `AgentAlignment`: an auditable decision that connects proposed work to an Agent capability, success criterion, and learning outcome.
- `AgentTask`: a user-level goal contract with constraints, success criteria, and runtime status.
- `PlanStep`: ordered work that must map to an `AgentTask` success criterion.
- `ExecutionTrace`: immutable-style event history for user input, plans, tool calls, evaluation, and context compaction.
- `ToolCallRecord`: durable current state for one tool operation, including strategy, idempotency identity, attempts, replays, and result.
- `TaskPolicy`: server-side allowlist and risk rules for tools available to one task.
- `ToolAuthorization`: an auditable allow, deny, or approval-required decision for one ToolCall under one policy version.
- `AgentRun`: one bounded model/tool execution with a final status, answer, and stop reason.
- `AgentRunStep`: one persisted model action and its optional ToolCall observation.

## Stage 2 ingestion API

- `POST /job-sources`: register a public source.
- `GET /job-sources`: list registered sources.
- `POST /job-archives`: archive supplied JD text, deduplicate it, optionally parse it, and optionally create a `Job`.
- `POST /job-archives/fetch`: fetch a public URL, archive the resulting text, deduplicate it, and optionally create a `Job`.
- `GET /job-archives`: list archived JDs.
- `POST /job-archives/parse`: preview deterministic JD parsing without persisting it.
- `GET /analytics/skills`: rank extracted skills by job demand.
- `GET /analytics/learning-roadmap`: turn high-frequency skills into a learning roadmap with project ideas and evidence targets.
- `GET /learning/tasks`: break the roadmap into project and evidence tasks with acceptance criteria.
- `GET /agent/goal-contract`: retrieve the active Agent goal contract.
- `POST /agent/goal-contracts`: create a new active goal contract version.
- `POST /agent/memories` and `GET /agent/memories`: store and retrieve typed Agent memory.
- `POST /agent/alignments`: record a goal-alignment decision before a development action.
- `POST /agent/tasks`, `GET /agent/tasks`, and `GET /agent/tasks/{task_id}`: create and retrieve user task contracts.
- `POST /agent/tasks/{task_id}/plan`: validate and persist ordered plan steps against the user task contract.
- `POST /agent/tasks/{task_id}/traces` and `GET /agent/tasks/{task_id}/traces`: append and inspect Agent runtime trace events.
- `POST /agent/tasks/{task_id}/run-learning-graph`: run the deterministic LangGraph skill-to-plan workflow.
- `POST /agent/tasks/{task_id}/resume-learning-graph`: approve or reject the interrupted plan and resume the same graph thread.
- `GET /agent/tasks/{task_id}/graph-checkpoints`: inspect every persisted State snapshot and next Node.
- `GET /agent/tools`: inspect registered tool descriptions, permissions, and JSON argument schemas.
- `GET /agent/tasks/{task_id}/tool-policy`: inspect the server-side tool policy for one task.
- `PUT /agent/tasks/{task_id}/tool-policy`: update the task's tool allowlist from the control plane.
- `POST /agent/tasks/{task_id}/tool-calls`: validate, authorize, execute, and trace one tool invocation.
- `GET /agent/tasks/{task_id}/tool-calls`: inspect durable ToolCall state and idempotency metadata.
- `POST /agent/tasks/{task_id}/agent-runs`: run the outer LangGraph workflow whose `run_agent` node invokes the framework-independent AgentLoopEngine.
- `GET /agent/tasks/{task_id}/agent-runs`: inspect model steps, observations, and stop reasons.
- `GET /agent/tasks/{task_id}/agent-runs/{run_id}/checkpoints`: inspect the outer workflow's Node-level State history.
- `POST /agent/tasks/{task_id}/recover-agent-runs`: classify and recover stale AgentRuns without blindly repeating ambiguous tools.
- `GET /agent/tasks/{task_id}/memory-context`: preview the GoalContract, selected non-expired memories, and context budget for the next model call.
- `GET /agent/memory-candidates`: inspect accepted, rejected, and review-required Candidate decisions with provenance and storage outcomes.

The parser is intentionally deterministic for now. It extracts company, title, location, and a first set of AI/Agent engineering skills such as Python, FastAPI, LangGraph, RAG, Docker, Kubernetes, and evaluation-related requirements.

Read [the Agent governance design](docs/agent-governance.md) for the goal-contract, memory, and alignment model.

Read [the LangGraph learning notes](docs/langgraph-learning.md) for a concrete State, Node, Edge, and checkpoint walkthrough. The current Windows development environment uses Python 3.14 successfully for the tests, but LangChain Core emits a Pydantic V1 compatibility warning; Python 3.11-3.13 is the cleaner learning environment until that compatibility layer is removed upstream.

Read [the Tool Calling learning notes](docs/tool-calling-learning.md) for a concrete registry, JSON Schema, permission gate, observation, and trace walkthrough.

Read [the cumulative learning notes](docs/learning-notes.md) for a concise Chinese summary of the Agent and Harness concepts covered so far.

Read [the reliable Tool Runtime V2 design](docs/reliable-tool-runtime.md) for the rationale behind durable ToolCall state, per-tool idempotency policy, request fingerprints, replay auditing, and transaction boundaries.

Read [the command-line debugging guide](docs/debugging-cli.md) for inspecting, snapshotting, and safely reproducing persisted tool calls.

Read [the bounded Agent Loop learning notes](docs/agent-loop-learning.md) for the concrete model/tool cycle, server-side step limit, persistence model, and current recovery boundary.

Read [the AgentRun recovery notes](docs/agent-run-recovery.md) for crash windows, atomic Step/ToolCall linking, observation restoration, and outer Workflow resumption.

Read [the Memory Runtime notes](docs/memory-runtime.md) for the difference between durable memory, run state, and assembled model context, plus retrieval and write policies.
