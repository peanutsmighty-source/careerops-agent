from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentRunStep, AgentTask, PlanStep
from app.services.agent_loop import (
    AgentDecision,
    AgentLoopEngine,
    create_agent_run,
)


class CompactionFinalModel:
    provider = "demo"
    model = "compaction-final-model"

    def decide(self, request):
        return AgentDecision(
            action_type="final_answer",
            content="Context compaction preserved the protected execution state.",
        )


def test_context_compaction_is_observable_and_preserves_protected_fields(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Compact a long Agent context",
            "user_goal": "Keep the goal and current execution state while reducing history.",
            "constraints": ["Never remove the explicit authorization boundary."],
            "success_criteria": ["The next model request retains protected fields."],
        },
    ).json()
    model = CompactionFinalModel()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        steps = [
            PlanStep(
                task_id=task.id,
                sequence=1,
                title="Record the completed action",
                agent_capability="context_management",
                success_criterion=task.success_criteria[0],
                goal_connection="Preserve completed work.",
                learning_outcome="Understand protected progress.",
                status="completed",
                result_summary="The durable state was recorded.",
            ),
            PlanStep(
                task_id=task.id,
                sequence=2,
                title="Resolve the blocked approval",
                agent_capability="user_interaction",
                success_criterion=task.success_criteria[0],
                goal_connection="Keep unresolved blockers visible.",
                learning_outcome="Understand approval boundaries.",
                status="blocked",
                result_summary="Explicit authorization is still required.",
            ),
            PlanStep(
                task_id=task.id,
                sequence=3,
                title="Run the next compaction check",
                agent_capability="context_management",
                success_criterion=task.success_criteria[0],
                goal_connection="Continue from the correct next action.",
                learning_outcome="Understand resumable context.",
                status="pending",
            ),
        ]
        session.add_all(steps)
        session.commit()
        run = create_agent_run(session, task=task, model=model, max_steps=8)
        for sequence in range(1, 5):
            session.add(
                AgentRunStep(
                    run_id=run.id,
                    task_id=task.id,
                    sequence=sequence,
                    action_type="tool_call",
                    model_request={"step_number": sequence},
                    model_response={
                        "action_type": "tool_call",
                        "tool_name": "benchmark_tool",
                    },
                    observation={
                        "tool_call_id": 100 + sequence,
                        "trace_id": 200 + sequence,
                        "task_id": task.id,
                        "plan_step_id": steps[min(sequence - 1, 2)].id,
                        "tool_name": "benchmark_tool",
                        "status": "succeeded",
                        "output": {
                            "sequence": sequence,
                            "detail": "historical observation " * 40,
                        },
                    },
                )
            )
        run.step_count = 4
        session.commit()
        run_id = run.id
        completed_step_id = steps[0].id
        blocked_step_id = steps[1].id
        next_step_id = steps[2].id

    preview = client.get(
        f"/agent/tasks/{task_payload['id']}/agent-runs/{run_id}/context-compaction",
        params={"char_threshold": 800, "recent_observation_tokens": 120},
    )
    assert preview.status_code == 200
    audit = preview.json()
    assert audit["triggered"] is True
    assert audit["raw_char_count"] > audit["compacted_char_count"]
    assert audit["raw_input"]["goal_contract"] == audit["compacted_context"][
        "goal_contract"
    ]
    assert audit["raw_input"]["task"] == audit["compacted_context"]["task"]
    execution = audit["compacted_context"]["execution_context"]
    assert execution["task_id"] == task_payload["id"]
    assert execution["run_id"] == run_id
    assert execution["completed_actions"][0]["plan_step_id"] == completed_step_id
    assert execution["unresolved_blockers"][0]["plan_step_id"] == blocked_step_id
    assert execution["next_action"]["plan_step_id"] == next_step_id
    assert audit["removed_items"]
    assert {item["reason"] for item in audit["removed_items"]} == {
        "summarized_old_observation"
    }
    assert any(
        item["reason"] == "protected_constraints"
        for item in audit["retained_items"]
    )

    AgentLoopEngine(model, compaction_char_threshold=800).run(run_id)

    run = client.get(f"/agent/tasks/{task_payload['id']}/agent-runs").json()[0]
    persisted = run["steps"][-1]["model_request"]
    assert persisted["context_compaction"]["triggered"] is True
    assert persisted["execution_context"]["next_action"]["plan_step_id"] == next_step_id
    assert persisted["observations"][0]["kind"] == "compacted_observation_summary"
    traces = client.get(f"/agent/tasks/{task_payload['id']}/traces").json()
    compaction_trace = next(
        trace for trace in traces if trace["event_type"] == "context_compaction"
    )
    assert compaction_trace["status"] == "applied"
    assert compaction_trace["metadata_json"]["raw_input"]["task"]["id"] == task_payload["id"]
    assert compaction_trace["metadata_json"]["removed_items"]
