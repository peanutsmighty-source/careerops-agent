# CareerOps Current Handoff

Updated: 2026-09-08

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `97d1683 Add deterministic memory benchmark baseline`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete observable T10 Context Compaction and close the T18 post-compaction metric.

Implemented, verified, and committed locally; pending push authorization:

- AgentLoop compacts long observation history before a model call while protecting GoalContract, constraints, execution progress, blockers, IDs, and next action.
- LangChain `trim_messages` and approximate token counting select recent observations; old observations receive a deterministic summary, so no external model sees project context.
- Raw and compacted Context, retained/removed items, and reasons are persisted in AgentRunStep and a dedicated Trace when compaction triggers.
- Runtime Console and a read-only preview endpoint expose the same audit shape.
- The deterministic Memory benchmark now reports critical-constraint retention both before and after compaction, completing T18 together with T10.

Expected changed files: Context Compaction service, AgentLoop/API/schema/UI wiring, focused tests, benchmark, README, TODO, Memory docs, learning notes, and this handoff.

## Verification

- Focused tests: Context Compaction and Memory benchmark both pass.
- Benchmark: 7 labeled Candidate cases; all metrics, including pre/post-compaction retention, report 1.0.
- Full suite: 72 passed.
- Branch coverage: 87%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Do not confuse the current character trigger plus observation trimming with T11's full model token budget.
- Full raw Context is intentionally persisted for audit and excluded from the model-facing request payload.
- Deterministic summarization is intentional until an external summarizer has a separate data-egress authorization path.
- Treat the current 1.0 scores as a deterministic regression baseline, not real-world quality evidence.
- Add mislabeled or production-derived cases as failures are discovered; do not tune only to synthetic examples.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Push the local `Add observable context compaction` commit only with explicit authorization.
2. Next major capability is T11 token-aware budgeting; do not begin it as part of this handoff.

After T10/T18: token-aware budgeting. Do not start RAG or multi-agent work yet.
