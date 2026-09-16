# CareerOps Current Handoff

Updated: 2026-09-16

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Delivery baseline: T15 is complete; use `git log -1 --oneline` for the current immutable commit ID.
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T15 external-write reconciliation adapters.

Implemented and verified:

- External handlers can report an uncertain outcome together with a provider operation ID.
- `ToolCallRecord` persists provider operation identity, reconciliation state, and reconciliation time.
- Each external tool may register a deterministic reconciliation adapter.
- Provider-confirmed success restores the result without a duplicate external write.
- Pending/unknown provider status remains `outcome_unknown`; provider-confirmed absence is re-authorized before retry.
- Missing operation identity or adapter remains a `needs_review` boundary.

See `app/services/tools.py`, `app/services/tool_recovery.py`, and `tests/test_tool_reconciliation.py`.

## Verification

- Targeted reconciliation and compatibility suites: 9 passed.
- Full suite: 98 passed (81.93 seconds under branch coverage).
- Coverage with branch measurement: 88%; reconciliation module: 92%, Tool Runtime: 91%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- A provider operation ID is required for deterministic reconciliation; without it the Runtime cannot infer an outcome.
- Adapter status is provider evidence, not an LLM judgment.
- There is no default real external-write provider or periodic reconciliation worker yet.
- `not_found` permits a retry path only after current TaskPolicy authorization; T16 will add attributable one-time approval.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. T16 authenticated identity and one-time external-write approval is next.
2. T12 remains deferred by the user's RAG restriction; do not begin RAG or multi-agent work early.

Do not start RAG or multi-agent work yet.
