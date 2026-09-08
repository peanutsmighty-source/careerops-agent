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

After meaningful work, add a brief Chinese learning note covering only new concepts or tradeoffs introduced by that work. Prefer one concrete example, then summarize why it matters, how CareerOps implements it, and the main pitfall in at most five bullets. Do not repeat project background, previously explained architecture, generic industry patterns, or interview questions unless they add material new insight or the user asks for them. Expand only on request.

## Safety and Delivery

- Inspect and preserve the dirty worktree before editing.
- Distinguish implemented, wired, tested, committed, and pushed.
- Finish and verify the current TODO before starting the next major capability.
- Never commit credentials, local databases, caches, `.test-tmp`, or `.codex-*.patch` files.
- Never print `ds_key.txt` or `DEEPSEEK_API_KEY`.
- Do not send project goals, Memory Context, or tool observations to external models without explicit user authorization.
