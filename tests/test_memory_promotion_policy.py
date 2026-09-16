from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentMemory, AgentRun, AgentRunStep, AgentTask, ToolCallRecord
from app.services.memory_evaluator import build_skill_demand_candidate
from app.services.memory_versioning import record_initial_memory_version
from app.services.tool_runtime import request_fingerprint


def test_promotion_requires_allowlisted_working_memory_provenance(client):
    task_data = client.post("/agent/tasks", json={
        "title": "Verify promotion evidence",
        "user_goal": "Promote only the captured working result.",
        "success_criteria": ["Promotion is linked to its working Memory."],
    }).json()
    with Session(engine) as session:
        task = session.get(AgentTask, task_data["id"])
        run = AgentRun(task_id=task.id, provider="demo", model="promotion-test",
                       status="completed", max_steps=2, step_count=1)
        session.add(run)
        session.flush()
        call = ToolCallRecord(
            task_id=task.id, tool_name="get_skill_demand", permission="read",
            effect="read", idempotency_mode="none", repeat_policy="always_allow",
            idempotency_key="promotion-policy", arguments_json={}, status="succeeded",
            request_fingerprint=request_fingerprint("get_skill_demand", {}),
            output_json={"skills": [{"name": "LangGraph", "job_count": 3,
                                     "requirement_count": 4}]},
        )
        session.add(call)
        session.flush()
        step = AgentRunStep(
            run_id=run.id, task_id=task.id, sequence=1, action_type="tool_call",
            model_request={}, model_response={"tool_name": "get_skill_demand"},
            tool_call_id=call.id,
            observation={"status": "succeeded", "output": call.output_json},
        )
        session.add(step)
        session.flush()
        assert build_skill_demand_candidate(session, task, run) is None

        working = AgentMemory(
            goal_contract_id=task.goal_contract_id, scope_type="run", run_id=run.id,
            memory_type="working",
            memory_key=f"agent-run:{run.id}:working:skill-demand",
            content="LangGraph (3 jobs, 4 requirements)",
            source="tool:get_skill_demand", importance=4,
        )
        session.add(working)
        session.flush()
        record_initial_memory_version(
            session, working,
            provenance={"promotion_policy": "different-policy"},
        )
        assert build_skill_demand_candidate(session, task, run) is None

        working.revisions[0].provenance = {
            "agent_run_id": run.id, "task_id": task.id,
            "agent_run_step_id": step.id, "tool_call_id": call.id,
            "tool_name": "get_skill_demand",
            "promotion_policy": "verified_skill_demand_fact_v1",
        }
        promoted = build_skill_demand_candidate(session, task, run)
        assert promoted is not None
        assert promoted.provenance["promoted_from_working_memory_id"] == working.id
