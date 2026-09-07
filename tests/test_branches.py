from datetime import datetime, timedelta
from email.message import Message

import pytest
from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentTask, GoalContract, ToolCallRecord
from app.scenario_runner import SCENARIOS
from app.services.governance import evaluate_alignment
from app.services.learning_graph import route_after_review, route_after_validation, validate_plan
from app.services.runtime import validate_plan_step
from app.services.tool_runtime import request_fingerprint
from app.services.agent_loop import AgentDecision, start_agent_run
from app.services.agent_workflow import agent_workflow_checkpoint_history


@pytest.mark.parametrize("scenario_name", list(SCENARIOS))
def test_scenario_definition_exercises_expected_cross_module_path(client, scenario_name):
    result = SCENARIOS[scenario_name].run(client)

    assert result


def test_graph_and_goal_validators_report_every_missing_mapping():
    graph_result = validate_plan(
        {"top_skills": [], "learning_plan": [], "success_criteria": [], "node_history": []}
    )
    assert graph_result["status"] == "validation_failed"
    assert len(graph_result["validation_errors"]) == 3
    assert route_after_validation(graph_result) == "block_run"
    assert route_after_validation({"validation_errors": []}) == "human_review"
    assert route_after_review({"approved": True}) == "complete_run"
    assert route_after_review({"approved": False}) == "block_run"

    contract = GoalContract(success_criteria=["known criterion"])
    alignment = evaluate_alignment(
        contract,
        agent_capability="unsupported",
        success_criterion="unknown criterion",
        goal_connection=" ",
        learning_outcome=" ",
    )
    assert alignment.decision == "defer"
    assert len(alignment.notes) == 4

    task = AgentTask(success_criteria=["known criterion"])
    plan = validate_plan_step(
        task,
        agent_capability="unsupported",
        success_criterion="unknown criterion",
        goal_connection=" ",
        learning_outcome=" ",
    )
    assert plan.is_valid is False
    assert len(plan.notes) == 4


def test_tool_runtime_covers_unknown_tool_replay_permission_and_read_tool_errors(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Exercise tool error branches",
            "user_goal": "Observe tool registry and permission failures.",
            "success_criteria": ["Each unsafe branch returns an explicit result."],
        },
    ).json()
    step = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Persist one write call",
                    "agent_capability": "tool_calling",
                    "success_criterion": "Each unsafe branch returns an explicit result.",
                    "goal_connection": "The task tests runtime boundaries.",
                    "learning_outcome": "See how replay permission is checked.",
                }
            ]
        },
    ).json()[0]
    arguments = {"plan_step_id": step["id"], "status": "completed"}

    unknown = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "missing_tool", "arguments": {}},
    )
    missing_job = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "get_job", "arguments": {"job_id": 999}},
    )
    self_granted = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "search_jobs",
            "arguments": {},
            "granted_permissions": ["read", "write"],
        },
    )
    policy = client.get(f"/agent/tasks/{task['id']}/tool-policy").json()
    client.put(
        f"/agent/tasks/{task['id']}/tool-policy",
        json={"allowed_tools": [*policy["allowed_tools"], "update_plan_step"]},
    )
    succeeded = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": arguments,
            "idempotency_key": "branch-replay-write",
            "plan_step_id": step["id"],
        },
    )
    client.put(
        f"/agent/tasks/{task['id']}/tool-policy",
        json={"allowed_tools": policy["allowed_tools"]},
    )
    denied_replay = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": arguments,
            "idempotency_key": "branch-replay-write",
            "plan_step_id": step["id"],
        },
    )

    assert unknown.status_code == 404
    assert self_granted.status_code == 422
    assert missing_job.json()["status"] == "failed"
    assert missing_job.json()["error"] == "job not found"
    assert succeeded.json()["status"] == "succeeded"
    assert denied_replay.status_code == 403


def test_recovery_marks_unapproved_and_unregistered_calls_for_review(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Classify ambiguous recovery records",
            "user_goal": "Do not execute unapproved or removed tools.",
            "success_criteria": ["Both records require review."],
        },
    ).json()
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        session.add_all(
            [
                ToolCallRecord(
                    task_id=task["id"],
                    tool_name="update_plan_step",
                    permission="write",
                    effect="internal_write",
                    idempotency_mode="operation_key",
                    repeat_policy="naturally_idempotent",
                    idempotency_key="stopped-before-authorization",
                    request_fingerprint=request_fingerprint("update_plan_step", {}),
                    arguments_json={},
                    status="proposed",
                    updated_at=old,
                ),
                ToolCallRecord(
                    task_id=task["id"],
                    tool_name="removed_internal_tool",
                    permission="write",
                    effect="internal_write",
                    idempotency_mode="operation_key",
                    repeat_policy="business_unique",
                    idempotency_key="removed-tool-call",
                    request_fingerprint=request_fingerprint("removed_internal_tool", {}),
                    arguments_json={},
                    status="authorized",
                    updated_at=old,
                ),
            ]
        )
        session.commit()

    report = client.post(
        f"/agent/tasks/{task['id']}/recover-tool-calls?stale_after_seconds=60"
    ).json()

    assert report["scanned_count"] == 2
    assert [decision["action"] for decision in report["decisions"]] == [
        "needs_review",
        "needs_review",
    ]
    assert "before authorization" in report["decisions"][0]["reason"]
    assert "no longer registered" in report["decisions"][1]["reason"]


def test_agent_loop_records_invalid_model_action_as_failed_run(client):
    class InvalidActionModel:
        provider = "demo"
        model = "invalid-action-model"

        def decide(self, request):
            return AgentDecision(action_type="delegate_everything")

    task_id = client.post(
        "/agent/tasks",
        json={
            "title": "Reject invalid model actions",
            "user_goal": "Keep model output inside the Agent protocol.",
            "success_criteria": ["Persist a failed run."],
        },
    ).json()["id"]
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        run = start_agent_run(
            session,
            task=task,
            provider="demo",
            model_name=None,
            max_steps=2,
            model_override=InvalidActionModel(),
        )

        assert run.status == "failed"
        assert run.stop_reason == "runtime_error"
        assert "unsupported model action" in run.error
        assert run.timing_json["failed_phase"] == "decision_validation"
        assert run.timing_json["wall_clock_ms"] >= 0


def test_agent_loop_returns_unknown_tool_error_to_the_model_as_observation(client):
    class HallucinatedToolModel:
        provider = "demo"
        model = "hallucinated-tool-model"

        def decide(self, request):
            if not request.observations:
                return AgentDecision(
                    action_type="tool_call",
                    tool_name="invented_job_tool",
                    arguments={},
                    call_id="invented-1",
                )
            assert request.observations[0]["status"] == "error"
            return AgentDecision(
                action_type="final_answer",
                content="The requested tool is unavailable, so I stopped safely.",
            )

    task_id = client.post(
        "/agent/tasks",
        json={
            "title": "Handle a hallucinated tool",
            "user_goal": "Do not crash when a model invents a tool.",
            "success_criteria": ["Return a final answer after the error observation."],
        },
    ).json()["id"]
    with Session(engine) as session:
        task = session.get(AgentTask, task_id)
        run = start_agent_run(
            session,
            task=task,
            provider="demo",
            model_name=None,
            max_steps=3,
            model_override=HallucinatedToolModel(),
        )

        assert run.status == "completed"
        assert run.step_count == 2
        assert run.steps[0].observation["status"] == "error"
        assert run.steps[0].tool_call_id is None
        assert agent_workflow_checkpoint_history(run) == []


def test_fetch_archive_converts_html_and_ignores_script_content(client, monkeypatch):
    html = """
    <html><body>
      <h1>Company: HTML AI</h1>
      <p>Title: Agent Engineer</p>
      <p>Use Python and FastAPI.</p>
      <script>SecretFakeSkill LangGraph</script>
    </body></html>
    """
    class FakeResponse:
        headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return html.encode("utf-8")

    monkeypatch.setattr("app.services.ingestion.urlopen", lambda request, timeout: FakeResponse())

    response = client.post(
        "/job-archives/fetch",
        json={"source_url": "https://example.com/jobs/html-agent"},
    )

    assert response.status_code == 201
    archive = response.json()
    assert "SecretFakeSkill" not in archive["raw_content"]
    assert archive["structured_data"]["company"] == "HTML AI"
    assert {item["name"] for item in archive["structured_data"]["requirements"]} == {
        "Python",
        "FastAPI",
    }
