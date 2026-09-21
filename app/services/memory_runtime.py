from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import AgentMemory, AgentRun, AgentTask, GoalContract, JobRequirement
from app.services.hybrid_retrieval import (
    HybridRetrievalResult,
    RankedRetrievalDocument,
    RetrievalDocument,
    rank_hybrid_documents,
)
from app.services.memory_similarity import MemoryEmbeddingProvider
from app.services.retrieval_embedding_cache import CachedRetrievalEmbeddingProvider
from app.services.retrieval_router import choose_retrieval_route
from app.services.token_budget import estimate_tokens


DEFAULT_MEMORY_LIMIT = 8
DEFAULT_MEMORY_TOKEN_BUDGET = 768
TYPE_PRIORITY = {"working": 30, "fact": 20, "episodic": 10}


@dataclass(frozen=True)
class MemoryContext:
    goal_contract: dict
    memories: list[dict]
    knowledge: list[dict]
    candidate_count: int
    knowledge_candidate_count: int
    excluded_expired_count: int
    token_budget: int
    tokens_used: int
    retrieval_method: str
    retrieval_route: str
    retrieval_route_reason: str
    retrieval_route_signals: tuple[str, ...]
    embedding_provider_version: str | None
    embedding_usage: dict[str, int]
    embedding_error: str | None
    selected_scores: list[dict]
    query: str
    filters: dict

    def as_dict(self, *, include_audit: bool = False) -> dict:
        retrieval = {
            "candidate_count": self.candidate_count,
            "knowledge_candidate_count": self.knowledge_candidate_count,
            "selected_count": len(self.memories),
            "knowledge_selected_count": len(self.knowledge),
            "excluded_expired_count": self.excluded_expired_count,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "token_estimator": "langchain_count_tokens_approximately_v1",
            "method": self.retrieval_method,
            "route": self.retrieval_route,
        }
        if include_audit:
            retrieval.update(self.audit_dict())
        return {
            "goal_contract": self.goal_contract,
            "memories": self.memories,
            "knowledge": self.knowledge,
            "retrieval": retrieval,
        }

    def audit_dict(self) -> dict:
        return {
            "embedding_provider_version": self.embedding_provider_version,
            "retrieval_route_reason": self.retrieval_route_reason,
            "retrieval_route_signals": list(self.retrieval_route_signals),
            "embedding_usage": self.embedding_usage,
            "embedding_error": self.embedding_error,
            "selected_scores": self.selected_scores,
            "query": self.query,
            "filters": self.filters,
        }


def assemble_memory_context(
    session: Session,
    task: AgentTask,
    *,
    run_id: int | None = None,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    knowledge_limit: int = 4,
    memory_token_budget: int = DEFAULT_MEMORY_TOKEN_BUDGET,
    embedding_provider: MemoryEmbeddingProvider | None = None,
    now: datetime | None = None,
) -> MemoryContext:
    if memory_limit < 0 or knowledge_limit < 0 or memory_token_budget < 0:
        raise ValueError("memory limits cannot be negative")

    current_time = now or datetime.utcnow()
    contract = session.get(GoalContract, task.goal_contract_id)
    if not contract:
        raise ValueError("task goal contract no longer exists")

    all_memories = list(
        session.scalars(
            select(AgentMemory)
            .where(
                AgentMemory.goal_contract_id == contract.id,
                AgentMemory.status == "active",
                or_(
                    AgentMemory.scope_type == "contract",
                    and_(
                        AgentMemory.scope_type == "task",
                        AgentMemory.task_id == task.id,
                    ),
                    and_(
                        run_id is not None,
                        AgentMemory.scope_type == "run",
                        AgentMemory.run_id == run_id,
                    ),
                ),
            )
            .order_by(AgentMemory.id)
        )
    )
    active_memories = [
        memory
        for memory in all_memories
        if memory.expires_at is None or memory.expires_at > current_time
    ]
    query = " ".join(
        [
            task.title,
            task.user_goal,
            *task.constraints,
            *task.success_criteria,
            *(
                f"{step.title} {step.success_criterion} {step.goal_connection}"
                for step in task.plan_steps
                if step.status in {"pending", "in_progress"}
            ),
        ]
    )
    memory_documents = [
        RetrievalDocument(
            key=f"memory:{memory.id}",
            source_type="memory",
            content=f"{memory.memory_type} {memory.memory_key} {memory.content}",
            prior_score=(
                memory.importance * 100 + TYPE_PRIORITY.get(memory.memory_type, 0)
            ),
            payload={
                "id": memory.id,
                "memory_type": memory.memory_type,
                "scope_type": memory.scope_type,
                "task_id": memory.task_id,
                "run_id": memory.run_id,
                "memory_key": memory.memory_key,
                "content": memory.content,
                "source": memory.source,
                "importance": memory.importance,
            },
        )
        for memory in active_memories
    ]
    requirements = list(
        session.scalars(select(JobRequirement).order_by(JobRequirement.id))
    )
    knowledge_documents = [
        RetrievalDocument(
            key=f"job_requirement:{requirement.id}",
            source_type="job_requirement",
            content=" ".join(
                [
                    requirement.job.company,
                    requirement.job.title,
                    requirement.skill.name,
                    requirement.evidence_text,
                ]
            ),
            payload={
                "source_type": "job_requirement",
                "requirement_id": requirement.id,
                "job_id": requirement.job_id,
                "company": requirement.job.company,
                "title": requirement.job.title,
                "skill": requirement.skill.name,
                "evidence_text": requirement.evidence_text,
                "source_url": requirement.job.source_url,
                "trust": "untrusted_public_evidence",
                "instruction_policy": "evidence_only",
            },
        )
        for requirement in requirements
    ]
    cached_embedding_provider = (
        CachedRetrievalEmbeddingProvider(session.get_bind(), embedding_provider)
        if embedding_provider
        else None
    )
    route = choose_retrieval_route(
        query,
        embedding_available=cached_embedding_provider is not None,
    )
    routed_embedding_provider = (
        cached_embedding_provider if route.strategy == "hybrid" else None
    )
    memory_result, knowledge_result = _rank_channels(
        query,
        memory_documents if memory_limit else [],
        knowledge_documents if knowledge_limit else [],
        embedding_provider=routed_embedding_provider,
        enabled=memory_token_budget > 0,
    )
    selected, knowledge, tokens_used, selected_scores = _select_under_budget(
        memory_result.candidates,
        knowledge_result.candidates,
        memory_limit=memory_limit,
        knowledge_limit=knowledge_limit,
        token_budget=memory_token_budget,
    )
    embedding_usage = {
        key: memory_result.usage.get(key, 0) + knowledge_result.usage.get(key, 0)
        for key in _zero_embedding_usage()
    }
    methods = {memory_result.method, knowledge_result.method} - {"none"}
    retrieval_method = (
        "not_run_budget_zero"
        if memory_token_budget == 0
        else "hybrid"
        if "hybrid" in methods
        else "lexical_fallback"
        if "lexical_fallback" in methods
        else "lexical"
    )
    embedding_error = memory_result.embedding_error or knowledge_result.embedding_error

    return MemoryContext(
        goal_contract={
            "id": contract.id,
            "version": contract.version,
            "north_star_goal": contract.north_star_goal,
            "product_context": contract.product_context,
            "learning_contract": contract.learning_contract,
            "scope_guardrails": contract.scope_guardrails,
            "success_criteria": contract.success_criteria,
        },
        memories=selected,
        knowledge=knowledge,
        candidate_count=len(active_memories),
        knowledge_candidate_count=len(knowledge_documents),
        excluded_expired_count=len(all_memories) - len(active_memories),
        token_budget=memory_token_budget,
        tokens_used=tokens_used,
        retrieval_method=retrieval_method,
        retrieval_route=route.strategy,
        retrieval_route_reason=route.reason,
        retrieval_route_signals=route.signals,
        embedding_provider_version=(
            embedding_provider.provider_version if embedding_provider else None
        ),
        embedding_usage=embedding_usage,
        embedding_error=embedding_error,
        selected_scores=selected_scores,
        query=query,
        filters={
            "goal_contract_id": contract.id,
            "memory_status": "active",
            "exclude_expired": True,
            "visible_scopes": ["contract", "task"]
            + (["run"] if run_id is not None else []),
            "task_id": task.id,
            "run_id": run_id,
            "knowledge_source": "public_job_requirements",
        },
    )


def _rank_channels(
    query: str,
    memories: list[RetrievalDocument],
    knowledge: list[RetrievalDocument],
    *,
    embedding_provider: MemoryEmbeddingProvider | None,
    enabled: bool,
) -> tuple[HybridRetrievalResult, HybridRetrievalResult]:
    if not enabled:
        empty = HybridRetrievalResult((), "none", None, _zero_embedding_usage())
        return empty, empty
    return (
        rank_hybrid_documents(
            query,
            memories,
            embedding_provider=embedding_provider,
            limit=max(len(memories), 1),
        ),
        rank_hybrid_documents(
            query,
            knowledge,
            embedding_provider=embedding_provider,
            limit=max(len(knowledge), 1),
        ),
    )


def _select_under_budget(
    memory_candidates: tuple[RankedRetrievalDocument, ...],
    knowledge_candidates: tuple[RankedRetrievalDocument, ...],
    *,
    memory_limit: int,
    knowledge_limit: int,
    token_budget: int,
) -> tuple[list[dict], list[dict], int, list[dict]]:
    selected: list[dict] = []
    knowledge: list[dict] = []
    token_payloads: list[dict] = []
    scores: list[dict] = []
    queues = (
        ("memory", memory_candidates, memory_limit, selected),
        ("knowledge", knowledge_candidates, knowledge_limit, knowledge),
    )
    positions = {"memory": 0, "knowledge": 0}
    while True:
        progressed = False
        for channel, candidates, limit, destination in queues:
            while positions[channel] < len(candidates) and len(destination) < limit:
                candidate = candidates[positions[channel]]
                positions[channel] += 1
                payload = candidate.document.payload
                candidate_tokens = estimate_tokens([*token_payloads, payload])
                if candidate_tokens > token_budget:
                    continue
                destination.append(payload)
                token_payloads.append(payload)
                scores.append(
                    {
                        "key": candidate.document.key,
                        "channel": channel,
                        "lexical_score": round(candidate.lexical_score, 6),
                        "semantic_score": (
                            round(candidate.semantic_score, 6)
                            if candidate.semantic_score is not None
                            else None
                        ),
                        "fusion_score": round(candidate.fusion_score, 6),
                    }
                )
                progressed = True
                break
        if not progressed:
            break
    return selected, knowledge, estimate_tokens(token_payloads), scores


def _zero_embedding_usage() -> dict[str, int]:
    return {
        "embedding_calls": 0,
        "embedding_tokens": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }


def resolve_memory_scope(
    session: Session,
    *,
    goal_contract_id: int,
    memory_type: str,
    scope_type: str,
    task_id: int | None,
    run_id: int | None,
) -> tuple[int | None, int | None]:
    """Validate a memory's lifetime boundary and return canonical scope ids."""
    if scope_type == "contract":
        if memory_type == "working":
            raise ValueError("working memory must be scoped to a task or run")
        if task_id is not None or run_id is not None:
            raise ValueError("contract-scoped memory cannot include task_id or run_id")
        return None, None

    if scope_type == "task":
        if task_id is None or run_id is not None:
            raise ValueError("task-scoped memory requires task_id and cannot include run_id")
        task = session.get(AgentTask, task_id)
        if not task or task.goal_contract_id != goal_contract_id:
            raise ValueError("memory task does not belong to the active goal contract")
        return task.id, None

    if scope_type == "run":
        if memory_type != "working":
            raise ValueError("run-scoped memory is reserved for working memory")
        if run_id is None:
            raise ValueError("run-scoped memory requires run_id")
        run = session.get(AgentRun, run_id)
        task = session.get(AgentTask, run.task_id) if run else None
        if not run or not task or task.goal_contract_id != goal_contract_id:
            raise ValueError("memory run does not belong to the active goal contract")
        if task_id is not None and task_id != task.id:
            raise ValueError("memory task_id does not match the run")
        return task.id, run.id

    raise ValueError(f"unsupported memory scope: {scope_type}")
