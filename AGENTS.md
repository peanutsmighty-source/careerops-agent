# CareerOps Agent Instructions

## Purpose

CareerOps is a learning-first AI Agent Harness project. Public JD intelligence is its real-world scenario, not the final architectural goal.

The project must both produce a credible Agent-engineering portfolio and teach the user how Agent systems work. Do not drift into unrelated CRUD, UI polish, or premature multi-agent work.

## Start Here

Before changing code:

1. Read `docs/HANDOFF.md` for the current breakpoint and exact next action.
2. Read the relevant item in `TODO.md` and its acceptance criteria.
3. Use `README.md` for the stable product/API overview.
4. Open a focused document under `docs/` only when working on that subsystem.

Do not load the whole `docs/` directory into context.

## Architecture Boundaries

- `GoalContract` is the durable north star shared across tasks and runs.
- LangGraph owns outer workflow routing, interrupts, and checkpoints.
- `AgentLoopEngine` owns the bounded model/tool loop and remains framework-independent.
- Tool Runtime owns validation, policy, authorization, idempotency, execution state, and recovery.
- Memory Runtime owns Candidates, evaluation, lifecycle, retrieval, and Context assembly.
- State/checkpoint, Trace, Memory, Context, and transcript have different responsibilities.
- LLM output is an untrusted proposal. Deterministic code controls permissions, IDs, scope, provenance, secrets, and invariants.

## Learning Contract

After meaningful work, teach the user the newly introduced Agent-engineering concepts in concrete Chinese, with enough depth that the user can explain the design independently rather than merely recognize terminology.

- Start with one concrete CareerOps example, then explain the problem being solved and what would fail without the mechanism.
- Walk through how it works here, including the important data flow, control boundary, persistence or recovery behavior, and the reason for the chosen tradeoff. Connect explanations to relevant files or tests when useful.
- Compare the implementation with common production Agent patterns, and clearly distinguish the current minimal implementation from capabilities that remain unfinished.
- Describe the most important implementation difficulty, failure discovered, or engineering lesson. Do not hide limitations behind generic summaries.
- End with at least one realistic interviewer question about the new work and a concise answer outline showing the reasoning the interviewer expects.

Keep the report focused on material introduced by the current work. Do not repeat the project background or previously explained architecture unless it is necessary to understand a new relationship. Prefer depth and teaching value over a fixed bullet limit; shorten only genuinely repetitive content, and expand further when the user asks.

## Safety and Delivery

- Inspect and preserve the dirty worktree before editing.
- Distinguish implemented, wired, tested, committed, and pushed.
- Finish and verify the current TODO before starting the next major capability.
- Never commit credentials, local databases, caches, `.test-tmp`, or `.codex-*.patch` files.
- Never print `ds_key.txt` or `DEEPSEEK_API_KEY`.
- Do not send project goals, Memory Context, or tool observations to external models without explicit user authorization.
