# CareerOps Current Handoff

Updated: 2026-09-09

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `97d1683 Add deterministic memory benchmark baseline`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T11 token-aware budgeting across the model Context and runtime usage audit.

Implemented, verified, and committed locally; pending push authorization:

- Memory selection now uses `memory_token_budget`; character limits no longer control Context assembly.
- AgentLoop measures prompt, Memory, full tool schemas, and observations before every model call, reserves output tokens, and rejects protected input that cannot fit.
- Remaining input capacity is divided between bounded Memory and observations; token-aware compaction progressively reduces old observation detail until the configured budget is met.
- Each AgentRunStep persists the estimator, configured limits, per-section usage, remaining tokens, and compaction savings. Runtime Console shows the latest budget.
- Run totals keep estimated Context, provider-reported Agent model usage, compaction savings, and Evaluator/Embedding usage separate.
- OpenAI and DeepSeek adapters pass the reserved output budget to their provider request and record provider usage when returned.

Expected changed files: token budget service/tests, Memory Runtime, Context Compaction, AgentLoop/provider adapters, API/schema/UI, benchmark/tests, README, TODO, Memory docs, learning notes, and this handoff.

## Verification

- Full suite: 75 passed after final review.
- Branch coverage: 87%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- The local estimator is deterministic and provider-agnostic, not an exact provider tokenizer. Compare its estimate with provider usage and keep safety margin.
- `model_context_tokens` is a configured admission-control budget, not automatically discovered from the selected model.
- Protected prompt and tool schemas are never truncated; an impossible configured budget fails before calling the model.
- Memory receives at most one third of capacity remaining after protected base input, capped by `memory_token_budget`; this is an explicit initial allocation policy to revisit with measurements.
- T10's raw compaction audit remains local and excluded from the model-facing payload.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Push the two local commits only with explicit authorization.
2. The recommended next capability is T03 free-text Candidate Builder, then T07/T09; do not start it in this handoff.

Do not start RAG or multi-agent work yet.
