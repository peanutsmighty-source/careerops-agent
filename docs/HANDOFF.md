# CareerOps Current Handoff

Updated: 2026-09-16

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Delivery remote/upstream: `private-origin` (`https://github.com/peanutsmighty-source/careerops-agent-private.git`)
- Public remote: `origin` (`https://github.com/peanutsmighty-source/careerops-agent.git`), currently at the T14 baseline; do not publish private changes there without explicit destination-specific authorization.
- Branch: `main`
- Delivery baseline: T15 is on `private-origin/main`; T16 is committed locally and one commit ahead of that upstream. Use `git log -1 --oneline` for the current immutable commit ID.
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T16 authenticated identity and one-time external-write approval.

Implemented and verified:

- External writes pause at `awaiting_approval` without executing the handler.
- Approval endpoints require a server-configured Bearer credential; actor identity is not client supplied.
- Approval scope binds task, ToolCall, tool, operation key, and request fingerprint.
- Expired approvals cannot execute; expired slots can be replaced without rewriting history.
- Atomic one-time consumption prevents concurrent or cross-operation reuse.
- Consuming actor and approval IDs are persisted in ToolAuthorization and execution Trace.

See `app/services/identity.py`, `app/services/tool_approval.py`, and `tests/test_tool_approval.py`.

## Verification

- Targeted approval, reconciliation, and compatibility suites: 13 passed.
- Full suite: 102 passed (95.01 seconds under branch coverage).
- Coverage with branch measurement: 88%; approval module: 84%, identity: 91%, Tool Runtime: 92%.
- T16 file diff check: passed. The unrelated trailing whitespace in `agent_run_lease.py` remains intentionally untouched.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- The Bearer credential is a minimal single-operator identity mechanism, not production OIDC/RBAC.
- There is no default real external-write provider; tests use fake handlers to prove approval semantics.
- Approval consumption is intentionally separate from result replay: replaying a completed call does not spend another write approval.
- The unrelated trailing-whitespace worktree edit in `agent_run_lease.py` is preserved and excluded from T16 delivery.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. T17 historical before-state replay or event sourcing is next.
2. T12 remains deferred by the user's RAG restriction; do not begin RAG or multi-agent work early.

Do not start RAG or multi-agent work yet.
