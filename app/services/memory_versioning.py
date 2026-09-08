from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentMemory, AgentMemoryRevision


def record_initial_memory_version(
    session: Session, memory: AgentMemory, *, provenance: dict | None = None
) -> AgentMemoryRevision:
    revision = AgentMemoryRevision(
        memory_id=memory.id,
        version=memory.version,
        content=memory.content,
        source=memory.source,
        importance=memory.importance,
        provenance=provenance,
        change_reason="created",
        valid_from=memory.created_at,
    )
    session.add(revision)
    session.flush()
    return revision


def supersede_memory(
    session: Session,
    memory: AgentMemory,
    *,
    content: str,
    source: str,
    importance: int,
    provenance: dict | None,
    reason: str,
) -> AgentMemory:
    now = datetime.utcnow()
    current = session.scalar(
        select(AgentMemoryRevision).where(
            AgentMemoryRevision.memory_id == memory.id,
            AgentMemoryRevision.version == memory.version,
        )
    )
    if current is None:
        current = record_initial_memory_version(session, memory)
    current.valid_to = now

    memory.version += 1
    memory.content = content
    memory.source = source
    memory.importance = importance
    memory.updated_at = now
    session.add(
        AgentMemoryRevision(
            memory_id=memory.id,
            version=memory.version,
            content=content,
            source=source,
            importance=importance,
            provenance=provenance,
            change_reason=reason,
            valid_from=now,
        )
    )
    session.flush()
    return memory
