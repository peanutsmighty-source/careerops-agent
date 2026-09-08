from __future__ import annotations

import re
from dataclasses import dataclass, replace

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AgentMemory,
    AgentRun,
    AgentRunStep,
    AgentTask,
    ExecutionTrace,
    MemoryCandidateRecord,
    ToolCallRecord,
)
from app.services.semantic_memory_evaluator import (
    SemanticEvaluationResult,
    SemanticMemoryEvaluator,
)
from app.services.memory_similarity import (
    MemoryEmbeddingProvider,
    SimilaritySearchResult,
    rank_similar_memories,
)
from app.services.memory_versioning import record_initial_memory_version, supersede_memory


MIN_OUTCOME_LENGTH = 24
EVALUATOR_VERSION = "deterministic-v2"
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(
        r"\b(?:api[_ -]?key|access[_ -]?token|password|secret)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: str
    scope_type: str
    task_id: int | None
    run_id: int | None
    memory_key: str
    content: str
    source: str
    importance: int
    relevance_text: str
    provenance: dict


@dataclass(frozen=True)
class MemoryEvaluation:
    decision: str
    reasons: tuple[str, ...]
    candidate: MemoryCandidate
    storage_action: str = "not_stored"
    memory: AgentMemory | None = None
    candidate_record: MemoryCandidateRecord | None = None
    duplicate_memory_id: int | None = None
    evaluator_version: str = EVALUATOR_VERSION
    evaluator_usage: dict[str, int] | None = None
    evaluator_output: dict | None = None


def evaluate_and_store_run_outcome(
    session: Session, run: AgentRun, *, answer: str
) -> MemoryEvaluation:
    """Compatibility wrapper for callers that only need the outcome episode."""
    task = session.get(AgentTask, run.task_id)
    if not task:
        raise ValueError("agent task no longer exists")
    return evaluate_and_store_candidate(
        session, task, build_run_outcome_candidate(task, run, answer=answer)
    )


def evaluate_and_store_run_memories(
    session: Session, run: AgentRun, *, answer: str
) -> tuple[MemoryEvaluation, ...]:
    """Build candidates from a completed run and evaluate each one independently."""
    task = session.get(AgentTask, run.task_id)
    if not task:
        raise ValueError("agent task no longer exists")
    return tuple(
        evaluate_and_store_candidate(session, task, candidate)
        for candidate in build_run_memory_candidates(task, run, answer=answer)
    )


def evaluate_and_store_candidate(
    session: Session,
    task: AgentTask,
    candidate: MemoryCandidate,
    *,
    semantic_evaluator: SemanticMemoryEvaluator | None = None,
    embedding_provider: MemoryEmbeddingProvider | None = None,
) -> MemoryEvaluation:
    existing = session.scalar(
        select(AgentMemory).where(
            AgentMemory.goal_contract_id == task.goal_contract_id,
            AgentMemory.memory_type == candidate.memory_type,
            AgentMemory.memory_key == candidate.memory_key,
        )
    )
    rule_evaluation = _evaluate_rules(session, task, candidate, check_duplicate=not existing)
    if (
        rule_evaluation.decision == "needs_review"
        and rule_evaluation.reasons == ("semantic_evaluation_required",)
        and semantic_evaluator is not None
    ):
        rule_evaluation = _evaluate_semantics(
            session,
            task,
            rule_evaluation,
            semantic_evaluator,
            embedding_provider=embedding_provider,
        )

    if rule_evaluation.decision == "accept" and existing:
        if _normalized(existing.content) == _normalized(candidate.content):
            evaluation = replace(
                rule_evaluation,
                reasons=("stable_memory_key", "provenance_verified"),
                storage_action="already_stored",
                memory=existing,
            )
        else:
            if _can_auto_supersede(existing, candidate):
                memory = supersede_memory(
                    session,
                    existing,
                    content=candidate.content,
                    source=candidate.source,
                    importance=candidate.importance,
                    provenance=candidate.provenance,
                    reason="trusted_fact_changed",
                )
                evaluation = replace(
                    rule_evaluation,
                    reasons=("trusted_fact_superseded", "provenance_verified"),
                    storage_action="superseded",
                    memory=memory,
                )
            else:
                evaluation = replace(
                    rule_evaluation, decision="reject", reasons=("memory_key_conflict",)
                )
    elif rule_evaluation.decision == "accept":
        memory = AgentMemory(
            goal_contract_id=task.goal_contract_id,
            scope_type=candidate.scope_type,
            task_id=candidate.task_id,
            run_id=candidate.run_id,
            memory_type=candidate.memory_type,
            memory_key=candidate.memory_key,
            content=candidate.content,
            source=candidate.source,
            importance=candidate.importance,
        )
        session.add(memory)
        session.flush()
        record_initial_memory_version(session, memory, provenance=candidate.provenance)
        evaluation = replace(rule_evaluation, storage_action="stored", memory=memory)
    else:
        evaluation = rule_evaluation

    record = _record_evaluation(session, task, evaluation)
    return replace(evaluation, candidate_record=record)


def build_run_memory_candidates(
    task: AgentTask, run: AgentRun, *, answer: str
) -> tuple[MemoryCandidate, ...]:
    candidates = [build_run_outcome_candidate(task, run, answer=answer)]
    skill_demand = build_skill_demand_candidate(task, run)
    if skill_demand:
        candidates.append(skill_demand)
    return tuple(candidates)


def build_run_outcome_candidate(
    task: AgentTask, run: AgentRun, *, answer: str
) -> MemoryCandidate:
    normalized_answer = WHITESPACE_PATTERN.sub(" ", answer).strip()
    return MemoryCandidate(
        memory_type="episodic",
        scope_type="task",
        task_id=task.id,
        run_id=None,
        memory_key=f"agent-run:{run.id}:outcome",
        content=f"Outcome for task '{task.title}': {normalized_answer}",
        source="agent_runtime",
        importance=3,
        relevance_text=normalized_answer,
        provenance={
            "agent_run_id": run.id,
            "task_id": task.id,
            "provider": run.provider,
            "model": run.model,
        },
    )


def build_skill_demand_candidate(
    task: AgentTask, run: AgentRun
) -> MemoryCandidate | None:
    """Extract a fact candidate from the structured get_skill_demand observation."""
    for step in reversed(run.steps):
        response = step.model_response or {}
        observation = step.observation or {}
        if response.get("tool_name") != "get_skill_demand":
            continue
        if observation.get("status") != "succeeded":
            continue
        output = observation.get("output") or {}
        skills = output.get("skills") or []
        if not skills:
            continue
        entries = [
            (
                f"{skill.get('name', 'unknown')} "
                f"({skill.get('job_count', 0)} jobs, "
                f"{skill.get('requirement_count', 0)} requirements)"
            )
            for skill in skills[:5]
        ]
        evidence = "; ".join(entries)
        return MemoryCandidate(
            memory_type="fact",
            scope_type="task",
            task_id=task.id,
            run_id=None,
            memory_key=f"task:{task.id}:skill-demand",
            content=f"Observed JD skill demand: {evidence}.",
            source="tool:get_skill_demand",
            importance=4,
            relevance_text=evidence,
            provenance={
                "agent_run_id": run.id,
                "task_id": task.id,
                "agent_run_step_id": step.id,
                "tool_call_id": step.tool_call_id,
                "tool_name": "get_skill_demand",
            },
        )
    return None


def evaluate_memory_candidate(
    session: Session, task: AgentTask, candidate: MemoryCandidate
) -> MemoryEvaluation:
    return _evaluate_rules(session, task, candidate, check_duplicate=True)


def _evaluate_rules(
    session: Session,
    task: AgentTask,
    candidate: MemoryCandidate,
    *,
    check_duplicate: bool,
) -> MemoryEvaluation:
    reject_reasons = _scope_reasons(session, task, candidate)
    provenance_decision, provenance_reasons = _validate_provenance(session, task, candidate)
    if provenance_decision == "reject":
        reject_reasons.extend(provenance_reasons)

    content = candidate.relevance_text.strip()
    if len(content) < MIN_OUTCOME_LENGTH:
        reject_reasons.append("insufficient_information")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        reject_reasons.append("contains_sensitive_value")

    duplicate = _find_duplicate(session, task, candidate) if check_duplicate else None
    if duplicate:
        reject_reasons.append("duplicate_content")
    if reject_reasons:
        return MemoryEvaluation(
            decision="reject",
            reasons=tuple(dict.fromkeys(reject_reasons)),
            candidate=candidate,
            duplicate_memory_id=duplicate.id if duplicate else None,
        )
    if provenance_decision == "needs_review":
        return MemoryEvaluation(
            decision="needs_review",
            reasons=tuple(provenance_reasons),
            candidate=candidate,
        )
    return MemoryEvaluation(
        decision="accept",
        reasons=("informative", "novel", "provenance_verified"),
        candidate=candidate,
    )


def _scope_reasons(
    session: Session, task: AgentTask, candidate: MemoryCandidate
) -> list[str]:
    if candidate.scope_type == "contract":
        return ["invalid_contract_scope"] if candidate.task_id or candidate.run_id else []
    if candidate.scope_type == "task":
        return [] if candidate.task_id == task.id and candidate.run_id is None else ["invalid_task_scope"]
    if candidate.scope_type == "run":
        run = session.get(AgentRun, candidate.run_id) if candidate.run_id else None
        return [] if run and run.task_id == task.id and candidate.task_id is None else ["invalid_run_scope"]
    return ["unknown_scope_type"]


def _validate_provenance(
    session: Session, task: AgentTask, candidate: MemoryCandidate
) -> tuple[str, list[str]]:
    provenance = candidate.provenance or {}
    if candidate.source == "agent_runtime":
        required = ("agent_run_id", "task_id", "provider", "model")
        if any(provenance.get(field) is None for field in required):
            return "reject", ["missing_runtime_provenance"]
        run = session.get(AgentRun, provenance["agent_run_id"])
        if not run:
            return "reject", ["unknown_agent_run_provenance"]
        if (
            run.task_id != task.id
            or provenance["task_id"] != task.id
            or run.provider != provenance["provider"]
            or run.model != provenance["model"]
        ):
            return "reject", ["runtime_provenance_mismatch"]
        return "accept", ["provenance_verified"]

    if candidate.source.startswith("tool:"):
        required = (
            "agent_run_id",
            "task_id",
            "agent_run_step_id",
            "tool_call_id",
            "tool_name",
        )
        if any(provenance.get(field) is None for field in required):
            return "reject", ["missing_tool_provenance"]
        run = session.get(AgentRun, provenance["agent_run_id"])
        step = session.get(AgentRunStep, provenance["agent_run_step_id"])
        tool_call = session.get(ToolCallRecord, provenance["tool_call_id"])
        source_tool = candidate.source.removeprefix("tool:")
        if not run or not step or not tool_call:
            return "reject", ["unknown_tool_provenance"]
        if (
            run.task_id != task.id
            or provenance["task_id"] != task.id
            or step.run_id != run.id
            or step.task_id != task.id
            or step.tool_call_id != tool_call.id
            or tool_call.task_id != task.id
            or tool_call.tool_name != provenance["tool_name"]
            or source_tool != provenance["tool_name"]
        ):
            return "reject", ["tool_provenance_mismatch"]
        return "accept", ["provenance_verified"]

    if candidate.source == "user_input":
        required = ("execution_trace_id", "task_id", "evidence_text")
        if any(provenance.get(field) is None for field in required):
            return "reject", ["missing_user_input_provenance"]
        trace = session.get(ExecutionTrace, provenance["execution_trace_id"])
        evidence = _normalized(str(provenance["evidence_text"]))
        if not trace:
            return "reject", ["unknown_user_input_provenance"]
        if (
            trace.task_id != task.id
            or provenance["task_id"] != task.id
            or trace.event_type != "user_input"
            or not evidence
            or evidence not in _normalized(trace.input_summary)
        ):
            return "reject", ["user_input_provenance_mismatch"]
        return "needs_review", ["semantic_evaluation_required"]

    return "needs_review", ["untrusted_provenance_source"]


def _evaluate_semantics(
    session: Session,
    task: AgentTask,
    rule_evaluation: MemoryEvaluation,
    evaluator: SemanticMemoryEvaluator,
    *,
    embedding_provider: MemoryEmbeddingProvider | None,
) -> MemoryEvaluation:
    candidate = rule_evaluation.candidate
    active_memories = list(
        session.scalars(
            select(AgentMemory).where(
                AgentMemory.goal_contract_id == task.goal_contract_id,
                AgentMemory.memory_type == candidate.memory_type,
                AgentMemory.status == "active",
            )
        )
    )
    memories = [
        memory for memory in active_memories if _scopes_overlap(candidate, memory)
    ]
    try:
        similarity = rank_similar_memories(
            candidate.content,
            memories,
            embedding_provider=embedding_provider,
        )
    except Exception:
        provider_version = (
            embedding_provider.provider_version if embedding_provider else "lexical"
        )
        return replace(
            rule_evaluation,
            reasons=("similarity_search_failed",),
            evaluator_version=f"{provider_version}:failed",
            evaluator_usage={
                "model_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "embedding_calls": 1 if embedding_provider else 0,
                "embedding_tokens": 0,
            },
            evaluator_output={"status": "similarity_search_failed"},
        )
    try:
        result = evaluator.evaluate(
            candidate={
                "memory_type": candidate.memory_type,
                "scope_type": candidate.scope_type,
                "content": candidate.content,
                "source": candidate.source,
                "importance": candidate.importance,
            },
            existing_memories=similarity.contexts(),
        )
    except Exception:
        usage = dict(similarity.usage)
        usage.update(
            {
                "model_calls": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        )
        return replace(
            rule_evaluation,
            reasons=("semantic_evaluator_failed",),
            evaluator_version=f"{evaluator.evaluator_version}:failed",
            evaluator_usage=usage,
            evaluator_output={"status": "failed"},
        )
    return _apply_semantic_result(rule_evaluation, result, similarity)


def _apply_semantic_result(
    rule_evaluation: MemoryEvaluation,
    result: SemanticEvaluationResult,
    similarity: SimilaritySearchResult,
) -> MemoryEvaluation:
    candidate = rule_evaluation.candidate
    judgment = result.judgment
    known_ids = {item.memory.id for item in similarity.candidates}
    referenced_ids = set(judgment.conflict_memory_ids)
    if judgment.semantic_duplicate_memory_id:
        referenced_ids.add(judgment.semantic_duplicate_memory_id)
    usage = dict(similarity.usage)
    for key, value in result.usage.items():
        usage[key] = usage.get(key, 0) + value
    common = {
        "evaluator_version": result.evaluator_version,
        "evaluator_usage": usage,
        "evaluator_output": {
            **judgment.model_dump(),
            "similarity_search": {
                "method": similarity.method,
                "provider_version": similarity.provider_version,
                "candidates": similarity.contexts(),
            },
        },
    }
    if not referenced_ids.issubset(known_ids):
        return replace(
            rule_evaluation,
            reasons=("invalid_model_memory_reference",),
            **common,
        )
    if judgment.memory_type != candidate.memory_type:
        return replace(rule_evaluation, reasons=("model_memory_type_mismatch",), **common)
    if judgment.conflict_memory_ids:
        return replace(rule_evaluation, reasons=("semantic_conflict",), **common)
    if judgment.semantic_duplicate_memory_id:
        return replace(
            rule_evaluation,
            decision="reject",
            reasons=("semantic_duplicate",),
            duplicate_memory_id=judgment.semantic_duplicate_memory_id,
            **common,
        )
    if judgment.long_term_value == "low" or judgment.decision == "reject":
        return replace(
            rule_evaluation,
            decision="reject",
            reasons=("low_long_term_value",),
            **common,
        )
    if judgment.decision == "accept":
        return replace(
            rule_evaluation,
            decision="accept",
            reasons=("semantic_value_verified", "provenance_verified"),
            **common,
        )
    return replace(rule_evaluation, reasons=("model_requested_review",), **common)


def _can_auto_supersede(existing: AgentMemory, candidate: MemoryCandidate) -> bool:
    return (
        existing.status == "active"
        and candidate.memory_type == "fact"
        and candidate.source.startswith("tool:")
        and existing.scope_type == candidate.scope_type
        and existing.task_id == candidate.task_id
        and existing.run_id == candidate.run_id
    )


def _record_evaluation(
    session: Session, task: AgentTask, evaluation: MemoryEvaluation
) -> MemoryCandidateRecord:
    candidate = evaluation.candidate
    claimed_run_id = candidate.provenance.get("agent_run_id")
    source_run = session.get(AgentRun, claimed_run_id) if claimed_run_id else None
    record = MemoryCandidateRecord(
        goal_contract_id=task.goal_contract_id,
        task_id=task.id,
        source_run_id=(
            source_run.id if source_run and source_run.task_id == task.id else None
        ),
        scope_run_id=candidate.run_id,
        memory_id=evaluation.memory.id if evaluation.memory else None,
        memory_type=candidate.memory_type,
        scope_type=candidate.scope_type,
        memory_key=candidate.memory_key,
        content=candidate.content,
        source=candidate.source,
        importance=candidate.importance,
        provenance=candidate.provenance,
        decision=evaluation.decision,
        storage_action=evaluation.storage_action,
        reasons=list(evaluation.reasons),
        evaluator_version=evaluation.evaluator_version,
        evaluator_usage=evaluation.evaluator_usage
        or {"model_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
        evaluator_output=evaluation.evaluator_output,
    )
    session.add(record)
    session.flush()
    return record


def _find_duplicate(
    session: Session, task: AgentTask, candidate: MemoryCandidate
) -> AgentMemory | None:
    candidate_answer = _normalized(candidate.relevance_text)
    candidate_canonical = _canonical(candidate.content)
    memories = session.scalars(
        select(AgentMemory).where(
            AgentMemory.goal_contract_id == task.goal_contract_id,
            AgentMemory.memory_type == candidate.memory_type,
            AgentMemory.scope_type == candidate.scope_type,
            AgentMemory.task_id == candidate.task_id,
            AgentMemory.run_id == candidate.run_id,
        )
    )
    for memory in memories:
        existing = _normalized(memory.content)
        if (
            existing == _normalized(candidate.content)
            or existing.endswith(candidate_answer)
            or (
                candidate_canonical
                and _canonical(memory.content) == candidate_canonical
            )
        ):
            return memory
    return None


def _normalized(value: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", value).strip().casefold()


def _canonical(value: str) -> str:
    return "".join(character for character in _normalized(value) if character.isalnum())


def _scopes_overlap(candidate: MemoryCandidate, memory: AgentMemory) -> bool:
    if candidate.scope_type == "contract":
        return memory.scope_type == "contract"
    if candidate.scope_type == "task":
        return memory.scope_type == "contract" or (
            memory.scope_type == "task" and memory.task_id == candidate.task_id
        )
    return memory.scope_type == "run" and memory.run_id == candidate.run_id
