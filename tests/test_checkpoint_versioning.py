from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentTask
from app.services.agent_loop import create_agent_model, create_agent_run
from app.services.agent_workflow import (
    AGENT_WORKFLOW_VERSION,
    agent_workflow_thread_id,
    build_agent_workflow,
)
from app.services.learning_graph import (
    LEARNING_GRAPH_VERSION,
    _graph_config,
    learning_graph,
)


def create_task(client):
    return client.post("/agent/tasks", json={
        "title": "Version a resumable graph",
        "user_goal": "Resume only compatible workflow state.",
        "success_criteria": ["An incompatible checkpoint fails before execution."],
    }).json()


def seed_skill(client):
    response = client.post("/job-archives", json={
        "source_url": "https://example.com/jobs/versioned-graph",
        "raw_content": "Company: Example\nTitle: Agent Engineer\nBuild an Agent Harness.",
    })
    assert response.status_code == 201


def test_new_learning_checkpoint_records_graph_version(client):
    seed_skill(client)
    task = create_task(client)
    result = client.post(f"/agent/tasks/{task['id']}/run-learning-graph")
    assert result.status_code == 200
    assert result.json()["graph_version"] == LEARNING_GRAPH_VERSION
    assert result.json()["checkpoint_compatible"] is True
    checkpoint = client.get(f"/agent/tasks/{task['id']}/graph-checkpoints").json()[0]
    assert checkpoint["state"]["graph_version"] == LEARNING_GRAPH_VERSION
    assert checkpoint["checkpoint_compatible"] is True


def test_unversioned_learning_checkpoint_fails_then_migrates_explicitly(client):
    seed_skill(client)
    task_data = create_task(client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        learning_graph.invoke({
            "task_id": task.id, "user_goal": task.user_goal,
            "success_criteria": task.success_criteria, "top_skills": [],
            "learning_plan": [], "validation_errors": [], "approved": None,
            "approval_comment": None, "status": "started", "node_history": [],
        }, config=_graph_config(task))

    resume = client.post(
        f"/agent/tasks/{task_data['id']}/resume-learning-graph",
        json={"approved": True},
    )
    assert resume.status_code == 409
    assert "unversioned" in resume.json()["detail"]
    assert client.get(
        f"/agent/tasks/{task_data['id']}/graph-checkpoints"
    ).json()[0]["checkpoint_compatible"] is False

    wrong = client.post(
        f"/agent/tasks/{task_data['id']}/migrate-learning-graph-checkpoint",
        json={"source_version": "learning-plan-v0"},
    )
    assert wrong.status_code == 409
    migrated = client.post(
        f"/agent/tasks/{task_data['id']}/migrate-learning-graph-checkpoint",
        json={"source_version": "unversioned"},
    )
    assert migrated.status_code == 200
    assert migrated.json()["graph_version"] == LEARNING_GRAPH_VERSION
    assert migrated.json()["awaiting_approval"] is True
    completed = client.post(
        f"/agent/tasks/{task_data['id']}/resume-learning-graph",
        json={"approved": True},
    )
    assert completed.status_code == 200
    assert completed.json()["state"]["status"] == "completed"


def test_unversioned_agent_checkpoint_requires_explicit_migration(client):
    task_data = create_task(client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        model = create_agent_model("demo")
        run = create_agent_run(session, task=task, model=model, max_steps=2)
        run_id = run.id
        graph = build_agent_workflow(model)
        config = {"configurable": {"thread_id": agent_workflow_thread_id(run)}}
        graph.update_state(config, {
            "task_id": task.id, "run_id": run.id, "provider": model.provider,
            "model": model.model, "max_steps": 2, "context_ready": True,
            "agent_status": "running", "evaluation_errors": [],
            "workflow_status": "context_loaded", "node_history": ["load_context"],
        }, as_node="load_context")
        session.commit()

    history = client.get(
        f"/agent/tasks/{task_data['id']}/agent-runs/{run_id}/checkpoints"
    ).json()
    assert history[0]["graph_version"] == "unversioned"
    assert history[0]["checkpoint_compatible"] is False
    wrong = client.post(
        f"/agent/tasks/{task_data['id']}/agent-runs/{run_id}/migrate-checkpoint",
        json={"source_version": "agent-workflow-v0"},
    )
    assert wrong.status_code == 409
    migrated = client.post(
        f"/agent/tasks/{task_data['id']}/agent-runs/{run_id}/migrate-checkpoint",
        json={"source_version": "unversioned"},
    )
    assert migrated.status_code == 200
    assert migrated.json()["graph_version"] == AGENT_WORKFLOW_VERSION
    assert migrated.json()["next_nodes"] == ["run_agent"]


def test_recovery_blocks_incompatible_checkpoint_before_agent_execution(client):
    task_data = create_task(client)
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        model = create_agent_model("demo")
        run = create_agent_run(session, task=task, model=model, max_steps=2)
        run_id = run.id
        graph = build_agent_workflow(model)
        graph.update_state(
            {"configurable": {"thread_id": agent_workflow_thread_id(run)}},
            {
                "task_id": task.id, "run_id": run.id, "provider": model.provider,
                "model": model.model, "max_steps": 2, "context_ready": True,
                "agent_status": "running", "evaluation_errors": [],
                "workflow_status": "context_loaded", "node_history": ["load_context"],
            },
            as_node="load_context",
        )
        session.commit()

    recovered = client.post(
        f"/agent/tasks/{task_data['id']}/recover-agent-runs?stale_after_seconds=0"
    )
    assert recovered.status_code == 200
    decision = recovered.json()["decisions"][0]
    assert decision["action"] == "needs_review"
    assert "unversioned" in decision["reason"]
    run = client.get(f"/agent/tasks/{task_data['id']}/agent-runs").json()[0]
    assert run["id"] == run_id and run["status"] == "needs_review"
    assert run["steps"] == []
