# CareerOps Agent Backlog

This backlog tracks product capabilities, not individual code edits. Completed foundations remain documented in README.md and the learning notes.

## Source labels

- `user-direct`: explicitly identified through the user's scenarios or requested behavior.
- `user-derived`: exposed by following the user's questions to an architectural consequence.
- `roadmap`: already implied by the original Agent Harness roadmap.

Tracked count: **18 capabilities**: 6 complete and 12 open. Sources: 8 `user-direct`, 6 `user-derived`, and 4 `roadmap`.

## P0 - Memory Gate v2

- [x] **T01 Enforce provenance, source, and scope integrity** (`user-derived`)
  - Verify referenced task, run, step, tool call, or user message exists and belongs to the same GoalContract.
  - Never accept model-invented provenance identifiers.
  - Acceptance: invalid or cross-scope provenance is rejected; every decision is traced.

- [x] **T02 Add `needs_review` to Memory Candidate decisions** (`user-direct`)
  - Separate `accept`, `reject`, and ambiguous/high-risk review outcomes.
  - Acceptance: unresolved conflicts and sensitive user facts are not silently stored or discarded.

- [x] **T08 Persist and inspect every Candidate decision** (`user-direct`)
  - Keep accepted, rejected, and review-required candidates with content, evidence, reason, evaluator version, and token usage.
  - Show newly created and rejected candidates in Runtime Console.
  - Acceptance: a user can explain why a Candidate did or did not become Memory.

- [x] **T04 Add an optional semantic/model Evaluator** (`user-derived`)
  - Use structured output to classify type, long-term value, semantic duplication, and possible conflicts.
  - Only call it after deterministic rules leave an ambiguous decision.
  - Acceptance: deterministic rules remain authoritative and model calls are metered.

- [x] **T05 Detect semantic duplicates** (`user-derived`)
  - Combine normalized matching, embeddings, and model review for borderline cases.
  - Acceptance: paraphrases do not create redundant active Memory.

- [x] **T06 Add fact conflict, version, and supersede handling** (`user-derived`)
  - Preserve history while ensuring only the current fact enters Context.
  - Acceptance: changed JD counts or user preferences retire/supersede older facts instead of silently coexisting.

## P1 - Context and Retrieval

- [ ] **T10 Implement observable Context Compaction** (`user-direct`)
  - Reuse LangGraph/LangChain message trimming and summarization primitives.
  - CareerOps owns protected fields, trigger policy, persistence, and audit UI.
  - Show raw input, compacted Context, retained items, removed items, and reasons.
  - Acceptance: GoalContract, constraints, completed actions, unresolved blockers, IDs, and next action survive compaction tests.

- [ ] **T11 Replace character budget with token-aware budgeting** (`roadmap`)
  - Track prompt, memory, evaluator, and compaction token usage separately.
  - Acceptance: context assembly stays below a configured model budget.

- [ ] **T12 Add embedding/hybrid retrieval and RAG** (`roadmap`)
  - Preserve scope, lifecycle, evidence, and deterministic filters around vector retrieval.
  - Acceptance: retrieval improves relevant Memory/JD recall without leaking across contracts.

- [ ] **T03 Add free-text Candidate Builder** (`roadmap`)
  - Extract explicit user facts, preferences, corrections, and reusable episodes with structured output.
  - Prefer piggybacking on the main model response or background batching over one extra call per turn.
  - Acceptance: labeled examples measure extraction recall and false-positive rate.

- [ ] **T07 Summarize and promote working Memory** (`user-direct`)
  - Retire temporary state at Run/Task boundaries and promote only verified reusable outcomes.
  - Acceptance: temporary hypotheses do not leak into later tasks.

- [ ] **T09 Add consolidation, forgetting, and physical cleanup** (`roadmap`)
  - Keep audit history while bounding active and retained storage.
  - Acceptance: repeated runs cannot grow active Context without limit.

- [ ] **T18 Build Memory precision/recall benchmarks** (`user-derived`)
  - Label what should be stored, rejected, reviewed, retrieved, and preserved through compaction.
  - Acceptance: CI reports Candidate precision/recall and critical-constraint retention.

## P2 - Runtime Reliability

- [ ] **T13 Version Graphs and migrate old checkpoints** (`user-direct`)
  - Acceptance: a checkpoint records graph version and incompatible resumes fail safely or migrate explicitly.

- [ ] **T14 Add asynchronous workers, startup recovery, leases, and multi-instance exclusion** (`user-derived`)
  - Acceptance: only one worker owns a Run and stale work is recovered automatically.

- [ ] **T15 Add external-write reconciliation adapters** (`user-direct`)
  - Query operation status by provider operation ID before retrying an unknown outcome.
  - Acceptance: timeout-after-success does not duplicate an external side effect.

- [ ] **T16 Add authenticated identity and one-time external-write approval** (`user-direct`)
  - Acceptance: approval is attributable, scoped, expiring, and cannot be reused for another operation.

- [ ] **T17 Add historical before-state replay or event sourcing** (`user-direct`)
  - Acceptance: CLI reproduction can recreate a chosen pre-call state, not only copy the current database.

## Recommended sequence

1. T01 -> T02 -> T08: make Candidate decisions trustworthy and visible.
2. T04 -> T05 -> T06 -> T18: add semantic judgment with measurable quality.
3. T10 -> T11: implement compaction with protected fields and token accounting.
4. T03 -> T07 -> T09 -> T12: broaden automatic memory and retrieval after the gate is reliable.
5. T13 -> T17: harden long-running and external-effect execution.
