# CareerOps Current Handoff

Updated: 2026-09-14

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `97d1683 Add deterministic memory benchmark baseline`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T07 working-Memory summarization and controlled promotion.

Implemented, verified, and committed in the latest local commit; not pushed yet:

- Successful `get_skill_demand` observations are deterministically summarized into run-scoped working Memory for subsequent model steps.
- The Builder requires non-empty structured evidence and verified Run/Step/ToolCall provenance; other tools are not automatically captured.
- A completed Run promotes the summary through a new task-scoped fact Candidate and records the source working Memory ID.
- Failed and step-limited Runs retire working Memory without promotion; temporary data cannot enter later tasks.
- Candidate Journal and AgentRun Trace expose working capture, promotion policy, promoted IDs, and retired IDs.

Expected changed files: AgentLoop, Memory Evaluator, lifecycle tests, README, TODO, Memory docs, learning notes, and this handoff.

## Verification

- Targeted lifecycle suite: 3 passed.
- Full suite: 80 passed.
- Branch coverage: 88%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Promotion is controlled by the Runtime allowlist, not by a model-proposed eligibility flag.
- `verified_skill_demand_fact_v1` is intentionally the only promotion policy in this minimal implementation.
- Run completion permits promotion; failed/max-step outcomes only retire working Memory.
- Explicit Task completion still trusts the control plane and does not yet evaluate evidence for every success criterion.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Push the four local commits only with explicit authorization.
2. The recommended next capability is T09 consolidation/forgetting; do not start it in this handoff.

Do not start RAG or multi-agent work yet.
