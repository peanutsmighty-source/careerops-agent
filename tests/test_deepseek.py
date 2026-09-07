import json
from types import SimpleNamespace

import pytest

from app.schemas import AgentRunCreate
from app.services.agent_loop import (
    AgentModelRequest,
    DeepSeekChatAgentModel,
)
from app.services.model_credentials import load_deepseek_api_key
from app.services.semantic_memory_evaluator import (
    DeepSeekSemanticMemoryEvaluator,
)


def test_deepseek_key_loader_prefers_environment_and_supports_ignored_file(
    monkeypatch, tmp_path
):
    key_file = tmp_path / "ds_key.txt"
    key_file.write_text("file-secret", encoding="utf-8")
    monkeypatch.setenv("CAREEROPS_DEEPSEEK_KEY_FILE", str(key_file))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert load_deepseek_api_key() == "file-secret"

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-secret")
    assert load_deepseek_api_key() == "environment-secret"


def test_empty_deepseek_key_file_is_rejected(monkeypatch, tmp_path):
    key_file = tmp_path / "ds_key.txt"
    key_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("CAREEROPS_DEEPSEEK_KEY_FILE", str(key_file))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(ValueError, match="key file is empty"):
        load_deepseek_api_key()


def test_deepseek_agent_model_converts_tool_call_without_exposing_credentials():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            id="completion-1",
            usage=SimpleNamespace(prompt_tokens=41, completion_tokens=9),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="search_jobs",
                                    arguments=json.dumps({"query": "Agent", "limit": 3}),
                                ),
                            )
                        ],
                    )
                )
            ],
        )

    model = DeepSeekChatAgentModel.__new__(DeepSeekChatAgentModel)
    model.model = "deepseek-test"
    model._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    request = AgentModelRequest(
        task_id=1,
        user_goal="Find Agent roles",
        constraints=[],
        success_criteria=["Use JD evidence"],
        memory_context={},
        observations=[],
        tools=[
            {
                "name": "search_jobs",
                "description": "Search archived jobs.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
        step_number=1,
        max_steps=4,
    )

    decision = model.decide(request)

    assert decision.action_type == "tool_call"
    assert decision.tool_name == "search_jobs"
    assert decision.arguments == {"query": "Agent", "limit": 3}
    assert captured["model"] == "deepseek-test"
    assert "api_key" not in json.dumps(captured)


def test_deepseek_semantic_evaluator_validates_json_and_records_usage():
    judgment = {
        "decision": "reject",
        "memory_type": "fact",
        "long_term_value": "high",
        "semantic_duplicate_memory_id": 7,
        "conflict_memory_ids": [],
        "rationale": "The candidate restates memory 7.",
    }

    def create(**kwargs):
        assert kwargs["response_format"] == {"type": "json_object"}
        return SimpleNamespace(
            usage=SimpleNamespace(prompt_tokens=73, completion_tokens=22),
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(judgment))
                )
            ],
        )

    evaluator = DeepSeekSemanticMemoryEvaluator.__new__(
        DeepSeekSemanticMemoryEvaluator
    )
    evaluator.model = "deepseek-test"
    evaluator.evaluator_version = "deepseek:deepseek-test"
    evaluator._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    result = evaluator.evaluate(
        candidate={"memory_type": "fact", "content": "Use examples."},
        existing_memories=[{"id": 7, "content": "Explain with examples."}],
    )

    assert result.judgment.semantic_duplicate_memory_id == 7
    assert result.usage == {
        "model_calls": 1,
        "prompt_tokens": 73,
        "completion_tokens": 22,
    }
    assert AgentRunCreate(provider="deepseek").provider == "deepseek"
