from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import engine
from app.models import (
    AgentMemory, AgentRun, AgentTask, ExecutionTrace, MemoryCandidateRecord,
    MemoryCleanupRecord,
)
from app.services.agent_loop import AgentDecision, AgentLoopEngine, create_agent_run
from app.services.memory_evaluator import evaluate_and_store_run_outcome
from app.services.memory_maintenance import maintain_task_memories
from app.services.memory_versioning import record_initial_memory_version
from app.schema_compat import ensure_memory_revision_history


def make_task(client):
    return client.post("/agent/tasks", json={
        "title": "Memory retention", "user_goal": "Learn bounded Agent memory.",
        "success_criteria": ["Old episodes cannot crowd out new evidence."],
    }).json()["id"]


def add_memory(session, task, key, **overrides):
    values = dict(goal_contract_id=task.goal_contract_id, task_id=task.id,
                  scope_type="task", memory_type="episodic", source="agent_runtime",
                  memory_key=key, content=key, importance=3)
    values.update(overrides)
    memory = AgentMemory(**values)
    session.add(memory)
    session.flush()
    record_initial_memory_version(session, memory)
    return memory


def test_retention_dry_run_scope_duplicates_expiry_and_rollback(client):
    task_id, other_id = make_task(client), make_task(client)
    now = datetime.utcnow()
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        memories = [add_memory(session, task, f"episode-{i}") for i in range(12)]
        duplicate = add_memory(session, task, "duplicate", content="episode-11")
        expired = add_memory(session, task, "expired", memory_type="working",
                             expires_at=now - timedelta(seconds=1))
        fact = add_memory(session, task, "fact", memory_type="fact")
        other = add_memory(session, session.get(AgentTask, other_id), "other")
        session.commit()
        report = maintain_task_memories(session, task, now=now)
        assert len(report["retire"]) == 6  # one duplicate, four overflow, one expired
        assert all(m.status == "active" for m in [*memories, duplicate, expired])
        assert session.scalar(select(ExecutionTrace).where(
            ExecutionTrace.event_type == "memory_maintenance")) is None
        applied = maintain_task_memories(session, task, dry_run=False, now=now)
        assert applied["retire"] == report["retire"]
        assert fact.status == other.status == "active"
        assert len([m for m in [*memories, duplicate] if m.status == "active"]) == 8
        assert maintain_task_memories(session, task, dry_run=False, now=now)["retire"] == []
        session.rollback()
        assert all(m.status == "active" for m in memories)


def test_cleanup_retention_tombstone_and_revision_deletion(client):
    task_id = make_task(client)
    now = datetime.utcnow()
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        old = add_memory(session, task, "old", status="retired",
                         retired_at=now-timedelta(days=31))
        recent = add_memory(session, task, "recent", status="retired", retired_at=now)
        fact = add_memory(session, task, "fact", memory_type="fact", status="retired",
                          retired_at=now-timedelta(days=31))
        session.commit()
        assert maintain_task_memories(session, task, purge=True)["purge_memory_ids"] == [old.id]
        assert old.content == "old" and len(old.revisions) == 1
        maintain_task_memories(session, task, purge=True, dry_run=False)
        session.commit()
        assert old.content == "[retired payload purged]" and old.revisions == []
        assert old.memory_key == "old" and old.status == "retired"
        assert recent.content == "recent" and fact.content == "fact"
        assert session.scalar(select(MemoryCleanupRecord)).removed_revision_count == 1
        assert maintain_task_memories(session, task, purge=True)["purge_memory_ids"] == []
    ensure_memory_revision_history(engine)
    with Session(engine) as session:
        assert session.scalar(select(AgentMemory).where(
            AgentMemory.memory_key == "old")).revisions == []


def test_run_completion_automatically_bounds_episodes(client):
    task_id = make_task(client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        for i in range(12):
            add_memory(session, task, f"old-runtime-episode-{i}")
        session.commit()
    response = client.post(f"/agent/tasks/{task_id}/agent-runs",
                           json={"provider": "demo", "max_steps": 4})
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    with Session(engine) as session:
        active = list(session.scalars(select(AgentMemory).where(
            AgentMemory.task_id == task_id, AgentMemory.memory_type == "episodic",
            AgentMemory.status == "active")))
        assert len(active) == 8
    endpoint = f"/agent/tasks/{task_id}/memory-maintenance"
    assert client.post(endpoint, json={}).json()["dry_run"] is True
    assert client.post(endpoint, json={"retention_days": 0}).status_code == 422
    assert client.post("/agent/tasks/999999/memory-maintenance", json={}).status_code == 404


class AnswerModel:
    provider = "demo"
    model = "retention-test"

    def __init__(self, answer):
        self.answer = answer

    def decide(self, request):
        return AgentDecision(action_type="final_answer", content=self.answer)


def test_repeated_runs_cleanup_journal_and_replay(client):
    task_id = make_task(client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        runs = []
        for i in range(11):
            model = AnswerModel(f"Verified distinct Agent retention learning result number {i}.")
            run = create_agent_run(session, task=task, model=model, max_steps=1)
            runs.append(run.id)
            AgentLoopEngine(model).run(run.id)
        session.expire_all()
        episodes = list(session.scalars(select(AgentMemory).where(
            AgentMemory.task_id == task_id, AgentMemory.memory_type == "episodic"
        ).order_by(AgentMemory.id)))
        assert len(episodes) == 11
        assert sum(m.status == "active" for m in episodes) == 8
        old = episodes[0]
        old.retired_at = datetime.utcnow() - timedelta(days=31)
        session.commit()
        maintain_task_memories(session, task, dry_run=False, purge=True)
        session.commit()
        journal = session.scalar(select(MemoryCandidateRecord).where(
            MemoryCandidateRecord.memory_id == old.id))
        assert journal.content == "[retired payload purged]"
        assert journal.decision == "accept" and journal.source_run_id == runs[0]
        run = session.get(AgentRun, runs[0])
        replay = evaluate_and_store_run_outcome(session, run, answer=run.final_answer)
        session.commit()
        assert replay.decision == "reject"
        assert replay.candidate_record.content == "[retired payload purged]"
        assert old.status == "retired" and old.revisions == []
