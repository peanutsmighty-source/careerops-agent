# CareerOps Current Handoff

Updated: 2026-09-10

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Remote: `https://github.com/peanutsmighty-source/careerops-agent`
- Branch: `main`
- Latest pushed commit: `97d1683 Add deterministic memory benchmark baseline`
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T03 free-text Candidate Builder without adding a model call per user turn.

Implemented, verified, and committed in the latest local commit; not pushed yet:

- Explicit English and Chinese preferences, target-role facts, corrections, and learning episodes become structured proposals.
- Task creation and subsequent `user_input` Trace creation both invoke the Builder inline with no extra model or external-data call.
- Every proposal binds to a real user-input Trace and passes through the existing Memory Gate; default user-input Candidates remain `needs_review + not_stored`.
- Candidate Journal and Trace metadata expose evidence, extraction rule/version, confidence, category, and generated record IDs.
- The deterministic benchmark now reports labeled extraction precision, recall, and false-positive rate in addition to Gate/retrieval/compaction metrics.

Expected changed files: free-text Candidate Builder/tests, API wiring, benchmark/tests, README, TODO, Memory docs, learning notes, and this handoff.

## Verification

- Targeted suite: 4 passed after fixing sentence boundaries and rule precedence.
- Full suite: 78 passed.
- Branch coverage: 88%.
- `git diff --check`: passed.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- The v1 rules intentionally prefer precision over recall and only recognize explicit first-person phrasing.
- Correction rules must run before nested fact patterns; ordinary English rules are anchored at sentence start.
- Extraction quality is measured on a small deterministic baseline, not yet a representative production corpus.
- Model-based or piggyback extraction remains a future option and must not bypass provenance validation or the Memory Gate.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. Push the three local commits only with explicit authorization.
2. The recommended next capability after T03 is T07 working-Memory summarization/promotion, then T09; do not start it in this handoff.

Do not start RAG or multi-agent work yet.
