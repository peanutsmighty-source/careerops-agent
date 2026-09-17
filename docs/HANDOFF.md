# CareerOps Current Handoff

Updated: 2026-09-17

This file contains only volatile development state. Stable architecture is in `README.md`; priorities are in `TODO.md`; explanations are in focused `docs/` notes.

## Repository

- Workspace: `E:\tmp\careerops-agent`
- Delivery remote/upstream: `private-origin` (`https://github.com/peanutsmighty-source/careerops-agent-private.git`)
- Public remote: `origin` (`https://github.com/peanutsmighty-source/careerops-agent.git`), currently at the T14 baseline; do not publish private changes there without explicit destination-specific authorization.
- Branch: `main`
- Delivery baseline: T15 is on `private-origin/main`; T16, T17, and T12 are committed locally and three commits ahead of that upstream after this handoff is finalized. Use `git log -3 --oneline` for immutable commit IDs.
- Never print or commit `ds_key.txt` or environment API keys.

## Current Task

Complete T12 scope-first hybrid retrieval and RAG.

Implemented and verified:

- Memory eligibility is filtered by GoalContract, active lifecycle, expiry, and contract/task/run scope before ranking.
- Memory and public JD requirements use separate retrieval channels and share the Context token budget.
- Lexical, optional embedding, and deterministic Memory-prior ranks are fused with RRF.
- JD evidence retains requirement/job IDs and source URL and is labeled untrusted evidence-only content.
- AgentLoop stores compact retrieval metadata in model Context and full query/filter/score/provider audit in AgentRunStep plus a retrieval Trace.
- External embeddings are disabled by default and require both provider selection and explicit data-egress acknowledgement.
- Embedding errors fall back to lexical retrieval with an auditable failure type.

See `app/services/hybrid_retrieval.py`, `app/services/memory_runtime.py`, `docs/rag-retrieval.md`, and `tests/test_hybrid_retrieval.py`.

## Verification

- Targeted retrieval, benchmark, token-budget, and compaction suites: passed.
- Full suite: 105 passed (97.27 seconds under branch coverage).
- Coverage with branch measurement: 88%; hybrid retrieval module: 92%, Memory Runtime: 87%.
- Deterministic semantic benchmark: lexical recall 0.0, hybrid recall 1.0, cross-contract leakage 0.
- T12 file diff check must exclude the unrelated trailing whitespace in `agent_run_lease.py`.

Use:

```powershell
python -m pytest -q --basetemp=.test-tmp -p no:cacheprovider
```

## Review Before Completion

- Embeddings are computed over eligible rows at request time; there is no persistent vector index or content-hash cache.
- JD RAG indexes structured requirement evidence rather than arbitrary raw-archive chunks.
- RRF weights and semantic threshold are fixed; the benchmark is a small deterministic regression set, not production-quality ranking evidence.
- The unrelated trailing-whitespace worktree edit in `agent_run_lease.py` is preserved and excluded from T12 delivery.

## Present but Not Default-Wired

- DeepSeek semantic Memory evaluation exists and is tested, but automatic AgentRun does not inject it.
- Default runs use deterministic Memory rules; ambiguous Candidates remain `needs_review` without evaluator token cost.
- Optional OpenAI embeddings exist. Local BGE-M3 remains a future option because no official DeepSeek embedding endpoint was found in the docs checked earlier.

## Cleanup

Run `git status --short`. Remove `.codex-*.patch` and `.test-tmp` artifacts if present; never stage them.

## Next Steps

1. All 18 tracked capabilities are complete at their documented minimal scope.
2. Do not begin multi-agent work without a concrete scenario and explicit user direction.

Do not start multi-agent work yet.
