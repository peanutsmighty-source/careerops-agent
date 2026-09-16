from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database import engine
from app.main import app
from app.models import AgentRun, AgentTask
from app.services.agent_loop import create_agent_model, create_agent_run
from app.services.agent_run_lease import (
    acquire_agent_run_lease,
    release_agent_run_lease,
    renew_agent_run_lease,
)
from app.services.agent_run_worker import agent_run_workers


def create_task(client):
    return client.post("/agent/tasks", json={
        "title": "Run work under a lease",
        "user_goal": "Prevent two workers from advancing the same Run.",
        "success_criteria": ["Only one worker owns the Run."],
    }).json()


def test_agent_run_lease_is_atomic_and_expired_lease_can_be_taken_over(client):
    task_data = create_task(client)
    now = datetime.utcnow()
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        run = create_agent_run(session, task=task, model=create_agent_model("demo"), max_steps=2)
        assert acquire_agent_run_lease(
            session, run_id=run.id, owner="worker-a", now=now, lease_seconds=30
        ) is True
        assert acquire_agent_run_lease(
            session, run_id=run.id, owner="worker-b", now=now, lease_seconds=30
        ) is False
        assert renew_agent_run_lease(
            session, run_id=run.id, owner="worker-b", now=now, lease_seconds=30
        ) is False
        later = now + timedelta(seconds=31)
        assert acquire_agent_run_lease(
            session, run_id=run.id, owner="worker-b", now=later, lease_seconds=30
        ) is True
        assert release_agent_run_lease(
            session, run_id=run.id, owner="worker-a"
        ) is False
        assert release_agent_run_lease(
            session, run_id=run.id, owner="worker-b"
        ) is True


def test_worker_mode_returns_and_executes_one_leased_run(client):
    task = create_task(client)
    response = client.post(
        f"/agent/tasks/{task['id']}/agent-runs",
        json={"provider": "demo", "max_steps": 4, "execution_mode": "worker"},
    )
    assert response.status_code == 200
    run_id = response.json()["id"]
    # Duplicate delivery is allowed; the database lease admits only one execution.
    agent_run_workers.submit(run_id)
    agent_run_workers.wait_for_idle()
    run = client.get(f"/agent/tasks/{task['id']}/agent-runs").json()[0]
    assert run["id"] == run_id
    assert run["status"] == "completed"
    assert run["step_count"] == 3
    assert run["lease_owner"] is None
    assert run["lease_expires_at"] is None


def test_manual_recovery_reports_lease_conflict(client):
    task_data = create_task(client)
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        run = create_agent_run(session, task=task, model=create_agent_model("demo"), max_steps=2)
        run.started_at = old
        session.commit()
        run_id = run.id
        assert acquire_agent_run_lease(
            session, run_id=run.id, owner="other-instance", lease_seconds=30
        )
    report = client.post(
        f"/agent/tasks/{task_data['id']}/recover-agent-runs?stale_after_seconds=0"
    )
    assert report.status_code == 200
    assert report.json()["decisions"][0]["action"] == "lease_conflict"
    assert report.json()["recovered_count"] == 0
    with Session(engine) as session:
        release_agent_run_lease(session, run_id=run_id, owner="other-instance")


def test_application_startup_schedules_stale_run_recovery():
    with TestClient(app) as first_client:
        task_data = create_task(first_client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        run = create_agent_run(session, task=task, model=create_agent_model("demo"), max_steps=1)
        run.started_at = datetime.utcnow() - timedelta(minutes=5)
        session.commit()
        run_id = run.id
    with TestClient(app) as restarted_client:
        agent_run_workers.wait_for_idle()
        runs = restarted_client.get(
            f"/agent/tasks/{task_data['id']}/agent-runs"
        ).json()
        recovered = next(run for run in runs if run["id"] == run_id)
        assert recovered["status"] == "max_steps"
        assert recovered["lease_owner"] is None
