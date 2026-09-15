# CareerOps Current Handoff

Updated: 2026-09-15

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `97d1683 Add deterministic memory benchmark baseline`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T09 deterministic consolidation, forgetting, and retired payload cleanup.

Implemented and verified:

- Successful Runs consolidate exact Runtime episode duplicates and cap active task episodes at eight in their completion transaction.
- Task-local expired memories retire; facts and user episodes are excluded from capacity eviction.
- Maintenance API defaults to dry-run. Explicit purge removes old working/episodic revisions and payloads after retention, preserving identity and cleanup audit.
- Candidate decisions remain inspectable; payload retention is an explicit exception to append-only content history.
- Startup backfill and Candidate replay respect cleanup tombstones.
- Business databases were not purged. No external model calls were used.

See `app/services/memory_maintenance.py` and `tests/test_memory_maintenance.py`.

## Verification

- Targeted maintenance suite: 4 passed.
- Full suite: 84 passed (80.24 seconds under coverage).
- Coverage with branch measurement: 89%; maintenance module: 97%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Cleanup retains tombstone IDs/keys so replay cannot recreate active Memory.
- Cleanup removes Memory revision rows and selected payload columns, not all original copies in Trace/checkpoints/provenance/backups.
- Metadata still grows. Global byte quotas, archive tiers, physical SQLite file compaction and a cleanup scheduler are unfinished.
- T07 promotion still permits only verified skill-demand facts on successful completion; failure/max-step paths only retire working Memory.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Commit T09 and push pending commits using the user's existing push authorization; confirm actual remote status before reporting delivery.
2. T09 is complete at the documented minimal scope. T12 is next in the backlog, but the user's RAG restriction remains in force; do not start it automatically.

Do not start RAG or multi-agent work yet.
