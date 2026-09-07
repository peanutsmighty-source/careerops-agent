from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.models import AgentMemory, AgentRun, AgentTask, GoalContract


DEFAULT_MEMORY_LIMIT = 8
DEFAULT_MEMORY_CHAR_BUDGET = 3000
TYPE_PRIORITY = {"working": 30, "fact": 20, "episodic": 10}
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


@dataclass(frozen=True)
class MemoryContext:
    goal_contract: dict
    memories: list[dict]
    candidate_count: int
    excluded_expired_count: int
    char_budget: int
    chars_used: int

    def as_dict(self) -> dict:
        return {
            "goal_contract": self.goal_contract,
            "memories": self.memories,
            "retrieval": {
                "candidate_count": self.candidate_count,
                "selected_count": len(self.memories),
                "excluded_expired_count": self.excluded_expired_count,
                "char_budget": self.char_budget,
                "chars_used": self.chars_used,
            },
        }


def assemble_memory_context(
    session: Session,
    task: AgentTask,
    *,
    run_id: int | None = None,
    memory_limit: int = DEFAULT_MEMORY_LIMIT,
    memory_char_budget: int = DEFAULT_MEMORY_CHAR_BUDGET,
    now: datetime | None = None,
) -> MemoryContext:
    if memory_limit < 0 or memory_char_budget < 0:
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
    query_tokens = _tokens(
        " ".join(
            [
                task.title,
                task.user_goal,
                *task.constraints,
                *task.success_criteria,
                contract.north_star_goal,
            ]
        )
    )
    ranked = sorted(
        active_memories,
        key=lambda memory: (
            -_memory_score(memory, query_tokens),
            -memory.importance,
            memory.id,
        ),
    )

    selected: list[dict] = []
    chars_used = 0
    for memory in ranked:
        if len(selected) >= memory_limit:
            break
        size = len(memory.memory_key) + len(memory.content)
        if chars_used + size > memory_char_budget:
            continue
        selected.append(
            {
                "id": memory.id,
                "memory_type": memory.memory_type,
                "scope_type": memory.scope_type,
                "task_id": memory.task_id,
                "run_id": memory.run_id,
                "memory_key": memory.memory_key,
                "content": memory.content,
                "source": memory.source,
                "importance": memory.importance,
                "relevance_score": _memory_score(memory, query_tokens),
            }
        )
        chars_used += size

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
        candidate_count=len(active_memories),
        excluded_expired_count=len(all_memories) - len(active_memories),
        char_budget=memory_char_budget,
        chars_used=chars_used,
    )


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


def _memory_score(memory: AgentMemory, query_tokens: set[str]) -> int:
    memory_tokens = _tokens(
        f"{memory.memory_type} {memory.memory_key} {memory.content}"
    )
    overlap = len(query_tokens & memory_tokens)
    return memory.importance * 100 + TYPE_PRIORITY.get(memory.memory_type, 0) + min(overlap, 20)


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}
