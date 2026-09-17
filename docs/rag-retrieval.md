# CareerOps Hybrid Retrieval and RAG

## Purpose

T12 assembles a small, governed evidence pack for each model step. It has two channels:

- Memory: accepted, active facts and episodes visible to the current GoalContract, Task, and Run.
- JD knowledge: public `JobRequirement` evidence with job identity and source URL.

The channels share a token budget but keep distinct payloads and provenance. JD text is marked `untrusted_public_evidence` and `evidence_only`; the Agent system prompt prohibits following instructions found inside it.

## Retrieval order

Memory eligibility is decided before similarity:

1. Match `goal_contract_id`.
2. Require `status=active`.
3. Enforce contract/task/run visibility.
4. Exclude expired records.
5. Rank only the remaining records.

This prevents vector similarity from granting visibility. JD requirements are public corpus evidence and do not inherit private Memory scope.

Each channel uses Reciprocal Rank Fusion over:

- lexical Jaccard rank;
- optional cosine-similarity rank;
- a deterministic Memory prior based on importance and Memory type.

Semantic results below the configured threshold do not enter the semantic rank. An embedding error records only the exception type and falls back to lexical retrieval. It never asks the Agent model to guess missing retrieval results.

## Persistent embedding cache

When semantic retrieval is enabled, `CachedRetrievalEmbeddingProvider` looks up each query and eligible document by `(provider_version, content_sha256)` before calling the configured provider. Cache misses are embedded in one batch and stored in `retrieval_embedding_cache`; cache hits reuse the stored vector. The table stores the hash and vector, not a duplicate of the Memory or JD text.

The content hash makes invalidation automatic: editing a Memory or JD requirement produces a new key. The provider version isolates vectors from different models or model revisions, so an old vector is never treated as output from a newly selected model. `embedding_calls`, `embedding_tokens`, `cache_hits`, and `cache_misses` are included in retrieval audit metadata.

The cache reduces repeated provider work but is not a vector index. Scope and lifecycle filters still run first, and the runtime still loads and ranks every eligible row in Python. A production-scale corpus would normally add retention for stale cache entries and an approximate-nearest-neighbor index after enforcing the same authorization filters.

## Context and audit

`assemble_memory_context` returns:

- `memories`: governed Memory records;
- `knowledge`: JD evidence with requirement/job IDs and source URL;
- `retrieval`: query, deterministic filters, selected scores, method, provider version, usage, failure mode, candidate counts, and token consumption.

The two result queues are selected in alternating order so one channel cannot automatically consume every slot. The total serialized evidence must fit `memory_token_budget`. AgentLoop persists the same retrieval metadata in an `ExecutionTrace(event_type="retrieval")` before the model decision is committed.

## Embedding authorization

Default runtime behavior is lexical and makes no external call. Tests inject deterministic fake embeddings.

OpenAI embeddings require both settings:

```powershell
$env:CAREEROPS_RETRIEVAL_EMBEDDING_PROVIDER = "openai"
$env:CAREEROPS_ALLOW_EXTERNAL_RETRIEVAL = "true"
```

The second flag is an explicit data-egress acknowledgement because eligible Memory and public JD evidence are sent to the provider. `OPENAI_API_KEY` must also be configured. Unknown providers and an OpenAI selection without the acknowledgement fail before an AgentRun model call.

## Quality checks

The deterministic benchmark now reports lexical semantic recall, hybrid semantic recall, and cross-contract leakage. Its labeled synonym case improves from lexical recall `0.0` to hybrid recall `1.0`, while leakage remains `0`. The cache regression fixture records 5 unique misses and 2 provider batches on the first two-channel retrieval, then 6 hits and 0 provider calls on an identical repeat. It also verifies that editing one document recomputes only that document and changing the provider version creates an isolated cache namespace. API/AgentLoop tests additionally verify JD evidence retrieval, token budgeting, trace persistence, external-provider authorization, and lexical fallback.

These are small regression fixtures, not evidence of production ranking quality. A production corpus needs larger labeled Recall@K/nDCG and citation-grounding evaluations.

## Current limits

- Embeddings are cached persistently by provider version and content hash, but there is no ANN vector index; ranking still scans eligible rows at request time.
- Cache rows do not yet have eviction or orphan cleanup, so obsolete content hashes remain until maintenance is added.
- SQLite and Python ranking are suitable for the portfolio dataset, not a large JD corpus.
- JD retrieval currently uses structured `JobRequirement.evidence_text`, not arbitrary raw-archive chunking.
- RRF weights and semantic threshold are fixed rather than calibrated on real user judgments.
- There is no model reranker. Deterministic filters, hybrid scores, and token budget remain authoritative.
