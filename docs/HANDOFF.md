# CareerOps Current Handoff

Updated: 2026-09-17

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Delivery remote/upstream: `private-origin` (`https://github.com/peanutsmighty-source/careerops-agent-private.git`)
- Public remote: `origin` (`https://github.com/peanutsmighty-source/careerops-agent.git`), currently at the T14 baseline; do not publish private changes there without explicit destination-specific authorization.
- Branch: `main`
- Delivery baseline: T15 is on `private-origin/main`; T16 and T17 are committed locally and two commits ahead of that upstream after this handoff is finalized. Use `git log -2 --oneline` for immutable commit IDs.
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T17 historical before-state replay for development SQLite.

Implemented and verified:

- Every file-backed SQLite `internal_write` attempt captures an online-backup fixture immediately before the handler runs.
- `ToolReplayFixture` binds the fixture path and SHA-256 to an exact ToolCall attempt.
- `careerops-debug replay <id> --before-attempt <n>` selects an exact pre-call state and fails on missing or modified fixtures.
- Historical state supplies replay inputs while the durable final ToolCall supplies the expected outcome.
- Replays execute on a second temporary copy, preserving both the live database and the historical fixture.
- External writes remain non-replayable and must use reconciliation.

See `app/services/replay_fixture.py`, `app/debug_cli.py`, and the debug CLI test in `tests/test_api.py`.

## Verification

- Targeted replay, internal-write recovery, and reconciliation suites: 5 passed.
- Full suite: 102 passed (95.23 seconds under branch coverage).
- Coverage with branch measurement: 88%; replay fixture module: 92%, Tool Runtime: 92%.
- T17 file diff check must exclude the unrelated trailing whitespace in `agent_run_lease.py`.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- The implementation copies the whole SQLite database per internal-write attempt; fixture retention and byte quotas are not implemented.
- PostgreSQL needs PITR, event sourcing, or tool-specific fixtures rather than this SQLite mechanism.
- Fixture files can contain sensitive development data and must remain local.
- The unrelated trailing-whitespace worktree edit in `agent_run_lease.py` is preserved and excluded from T17 delivery.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. T12 is the only open tracked capability, but remains deferred by the user's RAG restriction.
2. Do not invent another major capability or begin RAG/multi-agent work without user direction.

Do not start RAG or multi-agent work yet.
