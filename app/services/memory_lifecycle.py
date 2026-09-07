from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentMemory


def retire_memory(memory: AgentMemory, *, reason: str) -> bool:
    if memory.status == "retired":
        return False
    memory.status = "retired"
    memory.retired_at = datetime.utcnow()
    memory.retirement_reason = reason
    return True


def retire_run_working_memories(
    session: Session, *, run_id: int, reason: str
) -> list[int]:
    memories = list(
        session.scalars(
            select(AgentMemory).where(
                AgentMemory.run_id == run_id,
                AgentMemory.memory_type == "working",
                AgentMemory.status == "active",
            )
        )
    )
    return [memory.id for memory in memories if retire_memory(memory, reason=reason)]


def retire_task_working_memories(
    session: Session, *, task_id: int, reason: str
) -> list[int]:
    memories = list(
        session.scalars(
            select(AgentMemory).where(
                AgentMemory.task_id == task_id,
                AgentMemory.memory_type == "working",
                AgentMemory.status == "active",
            )
        )
    )
    return [memory.id for memory in memories if retire_memory(memory, reason=reason)]
