# CareerOps Current Handoff

Updated: 2026-09-16

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Delivery baseline: T13 and its promotion-policy correction are committed and pushed; use `git log -1 --oneline` for the current immutable commit ID.
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T14 asynchronous AgentRun workers, startup recovery, and leases.

Implemented and verified:

- `execution_mode: worker` persists and asynchronously executes AgentRuns; inline remains default.
- All execution/recovery entry points use atomic AgentRun leases with heartbeat and expiry.
- Duplicate delivery and concurrent manual recovery cannot execute a currently leased Run.
- Startup scans and schedules stale running Runs automatically.
- Existing recovery classification and T13 version preflight remain authoritative after lease acquisition.

See `app/services/agent_run_lease.py`, `app/services/agent_run_worker.py`, and `tests/test_agent_run_workers.py`.

## Verification

- Targeted worker/lease and compatibility suites pass.
- Full suite: 94 passed (78.36 seconds under coverage).
- Coverage with branch measurement: 88%; lease module: 80%, worker module: 82%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Lease ownership is execution authority, not a recovery decision; unsafe tool outcomes still need review.
- The in-process executor is not a durable queue; restart scanning recovers database-backed work.
- Lease heartbeat cannot forcibly cancel a blocked SDK call; fencing tokens remain unfinished.
- Startup recovery is one scan per process start, not a periodic scheduler.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Finish full verification, commit T14, and push using the user's existing push authorization.
2. T15 external-write reconciliation is next. T12 remains deferred by the user's RAG restriction.

Do not start RAG or multi-agent work yet.
