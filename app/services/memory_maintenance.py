"""Deterministic, task-local retention; caller owns the transaction.

Consolidation keeps a representative of exact duplicate runtime episodes.
It does not infer new facts or grant broader visibility.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    AgentMemory, AgentTask, ExecutionTrace, MemoryCandidateRecord, MemoryCleanupRecord,
)


def maintain_task_memories(
    session: Session,
    task: AgentTask,
    *,
    dry_run: bool = True,
    purge: bool = False,
    episode_limit: int = 8,
    retention_days: int = 30,
    now: datetime | None = None,
) -> dict:
    if episode_limit < 1 or retention_days < 1:
        raise ValueError("memory retention limits must be positive")
    current = now or datetime.utcnow()
    session.flush()
    memories = list(session.scalars(select(AgentMemory).where(
        AgentMemory.goal_contract_id == task.goal_contract_id,
        AgentMemory.task_id == task.id,
        AgentMemory.scope_type.in_(("task", "run")),
    )))
    retire = {}
    representatives = {}
    episodes = []
    for memory in sorted(memories, key=lambda m: (-m.importance, -m.id)):
        if memory.status != "active":
            continue
        if memory.expires_at is not None and memory.expires_at <= current:
            retire[memory.id] = {"reason": "expired"}
        elif (memory.memory_type == "episodic" and memory.source == "agent_runtime"
              and memory.scope_type == "task"):
            # Exact text only: similar wording is not proof of semantic equivalence.
            representative = representatives.get(memory.content)
            if representative is not None:
                retire[memory.id] = {
                    "reason": "duplicate_episode", "representative_id": representative,
                }
            else:
                representatives[memory.content] = memory.id
                episodes.append(memory)
    for memory in episodes[episode_limit:]:
        retire[memory.id] = {"reason": "episode_capacity"}

    cleaned_ids = set(session.scalars(select(MemoryCleanupRecord.memory_id).join(
        AgentMemory, AgentMemory.id == MemoryCleanupRecord.memory_id
    ).where(AgentMemory.task_id == task.id)))
    cutoff = current - timedelta(days=retention_days)
    cleanup = [m for m in memories if purge and m.status == "retired"
               and m.retired_at is not None and m.retired_at <= cutoff
               and m.memory_type in ("working", "episodic") and m.id not in cleaned_ids]
    report = {
        "policy": "task_memory_retention_v1", "task_id": task.id,
        "dry_run": dry_run, "episode_limit": episode_limit,
        "retention_days": retention_days,
        "retire": [{"memory_id": key, **value} for key, value in retire.items()],
        "purge_memory_ids": [m.id for m in cleanup],
    }
    if dry_run:
        return report
    for memory in memories:
        if memory.id in retire:
            memory.status = "retired"
            memory.retired_at = current
            memory.retirement_reason = retire[memory.id]["reason"]
    for memory in cleanup:
        session.add(MemoryCleanupRecord(
            memory_id=memory.id,
            content_sha256=sha256(memory.content.encode("utf-8")).hexdigest(),
            removed_revision_count=len(memory.revisions), cleaned_at=current,
        ))
        memory.content = "[retired payload purged]"
        memory.revisions.clear()
        for candidate in session.scalars(select(MemoryCandidateRecord).where(
            MemoryCandidateRecord.memory_id == memory.id
        )):
            candidate.content = "[retired payload purged]"
            candidate.evaluator_output = None
    if retire or cleanup:
        session.add(ExecutionTrace(
            task_id=task.id, event_type="memory_maintenance", status="completed",
            input_summary="Apply task-local Memory retention policy.",
            output_summary=f"Retired {len(retire)}; purged {len(cleanup)} payloads.",
            metadata_json=report,
        ))
    session.flush()
    return report
