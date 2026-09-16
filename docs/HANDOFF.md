# CareerOps Current Handoff

Updated: 2026-09-16

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `54927f9 Bound task memory and add retention cleanup`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T13 Graph/checkpoint versioning and explicit migration.

Implemented and verified:

- New learning and outer Agent workflow checkpoints persist graph versions.
- History APIs expose stored/runtime versions and compatibility.
- Resume and stale-Run recovery fail before Node/model/tool execution on incompatible checkpoints.
- Explicit APIs migrate only supported unversioned checkpoints at known Node boundaries and record audit Traces.
- Unknown versions and boundaries remain blocked.

See `app/services/checkpoint_versioning.py` and `tests/test_checkpoint_versioning.py`.

## Verification

- Targeted checkpoint/version suite: 8 passed; promotion-policy suite: 3 passed.
- Full suite: 89 passed (70.56 seconds under coverage).
- Coverage with branch measurement: 89%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- A graph version describes control-flow compatibility, not database schema compatibility.
- Only `unversioned -> v1` migration is implemented; it deliberately rejects unknown versions.
- Learning migration re-enters only the side-effect-free human-review interrupt.
- Agent recovery performs version preflight before inspecting or executing pending work.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Finish full verification, commit T13, and push using the user's existing push authorization.
2. T14 is the next allowed Runtime task. T12 remains deferred by the user's RAG restriction.

Do not start RAG or multi-agent work yet.
