from types import SimpleNamespace

from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentMemory, AgentRunStep, AgentTask
from app.services.agent_loop import (
    AgentDecision,
    AgentLoopEngine,
    AgentModelRequest,
    OpenAIResponsesAgentModel,
    create_agent_run,
)
from app.services.memory_versioning import record_initial_memory_version


class BudgetAwareFinalModel:
    provider = "demo"
    model = "budget-aware-final"

    def __init__(self):
        self.call_count = 0
        self.request = None

    def decide(self, request):
        self.call_count += 1
        self.request = request
        return AgentDecision(
            action_type="final_answer",
            content="The budgeted request retained the protected task state.",
            provider_metadata={"prompt_tokens": 111, "completion_tokens": 12},
        )


def test_openai_adapter_applies_output_reserve_and_records_provider_usage():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="response-1",
            usage=SimpleNamespace(input_tokens=91, output_tokens=17),
            output=[],
            output_text="Budget respected.",
        )

    model = OpenAIResponsesAgentModel.__new__(OpenAIResponsesAgentModel)
    model.model = "openai-test"
    model._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    request = AgentModelRequest(
        task_id=1,
        user_goal="Respect the budget",
        constraints=[],
        success_criteria=["Record provider usage"],
        memory_context={},
        observations=[],
        tools=[],
        step_number=1,
        max_steps=2,
        context_token_budget={"reserved_output_tokens": 64},
    )

    decision = model.decide(request)

    assert captured["max_output_tokens"] == 64
    assert decision.provider_metadata == {
        "response_id": "response-1",
        "prompt_tokens": 91,
        "completion_tokens": 17,
    }


def test_agent_context_is_partitioned_and_kept_within_token_budget(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Budget a long Agent request",
            "user_goal": "Keep the request below its configured model budget.",
            "constraints": ["Never remove the authorization boundary."],
            "success_criteria": ["Every Context section reports token usage."],
        },
    ).json()
    model = BudgetAwareFinalModel()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        for index in range(3):
            memory = AgentMemory(
                goal_contract_id=task.goal_contract_id,
                task_id=task.id,
                scope_type="task",
                memory_type="fact",
                memory_key=f"budget-memory-{index}",
                content="Relevant durable evidence " * 30,
                source="test_fixture",
                importance=5 - index,
            )
            session.add(memory)
            session.flush()
            record_initial_memory_version(session, memory)
        run = create_agent_run(session, task=task, model=model, max_steps=8)
        for sequence in range(1, 5):
            session.add(
                AgentRunStep(
                    run_id=run.id,
                    task_id=task.id,
                    sequence=sequence,
                    action_type="tool_call",
                    model_request={"step_number": sequence},
                    model_response={"action_type": "tool_call"},
                    observation={
                        "tool_call_id": 100 + sequence,
                        "trace_id": 200 + sequence,
                        "tool_name": "budget_tool",
                        "status": "succeeded",
                        "output": {"detail": "large tool result " * 100},
                    },
                )
            )
        run.step_count = 4
        session.commit()
        run_id = run.id

    AgentLoopEngine(
        model,
        model_context_tokens=1400,
        reserved_output_tokens=200,
        memory_token_budget=240,
    ).run(run_id)

    assert model.call_count == 1
    request_budget = model.request.context_token_budget
    assert request_budget["within_budget"] is True
    assert request_budget["estimated_input_tokens"] <= 1200
    assert request_budget["remaining_input_tokens"] >= 0
    assert set(request_budget["sections"]) == {
        "prompt_tokens",
        "memory_tokens",
        "tool_tokens",
        "observation_tokens",
    }
    assert all(value > 0 for value in request_budget["sections"].values())
    assert request_budget["compaction"]["saved_tokens"] > 0
    retrieval = model.request.memory_context["retrieval"]
    assert retrieval["tokens_used"] <= retrieval["token_budget"] <= 240

    run = client.get(f"/agent/tasks/{task_payload['id']}/agent-runs").json()[0]
    usage = run["timing_json"]["token_usage"]
    assert usage["context"]["estimated_input_tokens"] <= 1200
    assert usage["agent_model"] == {
        "provider_prompt_tokens": 111,
        "provider_completion_tokens": 12,
    }
    assert usage["compaction"]["saved_tokens"] > 0
    assert usage["evaluator"]["prompt_tokens"] == 0
    assert usage["evaluator"]["completion_tokens"] == 0


def test_agent_fails_before_model_call_when_protected_context_exceeds_budget(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Reject an impossible Context budget",
            "user_goal": "Do not call the model if protected input cannot fit.",
            "constraints": ["This protected constraint must not be truncated."],
            "success_criteria": ["Fail before the model call."],
        },
    ).json()
    model = BudgetAwareFinalModel()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        run = create_agent_run(session, task=task, model=model, max_steps=2)
        run_id = run.id

    AgentLoopEngine(
        model,
        model_context_tokens=80,
        reserved_output_tokens=20,
        memory_token_budget=0,
    ).run(run_id)

    assert model.call_count == 0
    run = client.get(f"/agent/tasks/{task_payload['id']}/agent-runs").json()[0]
    assert run["status"] == "failed"
    assert run["timing_json"]["failed_phase"] == "request_build"
    assert "protected prompt and tool schemas exceed" in run["error"]
