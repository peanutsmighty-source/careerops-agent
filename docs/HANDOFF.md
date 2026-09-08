# CareerOps Current Handoff

Updated: 2026-09-08

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `ec9d8a0 Version and supersede trusted memory facts`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Establish the minimal T18 Memory benchmark before Context Compaction.

Implemented locally and verified, pending commit/push:

- `python -m app.memory_benchmark` runs deterministic labeled cases without an external model.
- Candidate labels cover accept, reject, and needs_review outcomes.
- Metrics report Candidate precision/recall, exact decision accuracy, and retrieval recall.
- Critical-constraint retention is explicitly named as a pre-compaction baseline.
- T18 remains open until T10 measures the same labels after compaction.
- `AGENTS.md` Learning Contract now requests only concise, materially new learning notes.

Expected changed files: `AGENTS.md`, `app/memory_benchmark.py`, `tests/test_memory_benchmark.py`, `pyproject.toml`, `README.md`, `TODO.md`, `docs/memory-runtime.md`, `docs/learning-notes.md`, and this handoff.

## Verification

- Benchmark: 7 labeled Candidate cases; all reported metrics are 1.0.
- Full suite: 71 passed.
- Coverage: 87%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Keep T18 unchecked until post-compaction critical-constraint retention is measured.
- Treat the current 1.0 scores as a deterministic regression baseline, not real-world quality evidence.
- Add mislabeled or production-derived cases as failures are discovered; do not tune only to synthetic examples.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them. The Windows sandbox repeatedly returned `helper_unknown_error: setup refresh had errors`, so stop and report it if normal editing remains unavailable.

## Next Steps

1. Clean `.test-tmp` and review the benchmark diff.
2. Commit the benchmark baseline and concise Learning Contract.
3. Push only with explicit authorization for the new commit.
4. Next major capability: T10 observable Context Compaction, reusing the benchmark's critical-constraint label and metric.

After the benchmark baseline: observable Context Compaction, then token-aware budgeting. Do not start RAG or multi-agent work yet.
