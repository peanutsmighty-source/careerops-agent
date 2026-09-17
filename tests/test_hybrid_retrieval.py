import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import engine
from app.models import (
    AgentMemory,
    AgentTask,
    ExecutionTrace,
    GoalContract,
    Job,
    JobRequirement,
    RetrievalEmbeddingCacheEntry,
    Skill,
)
from app.services.agent_loop import AgentDecision, AgentLoopEngine, create_agent_run
from app.services.memory_runtime import assemble_memory_context
from app.services.memory_similarity import (
    EmbeddingBatch,
    create_configured_retrieval_embedding_provider,
)


class SemanticTestEmbeddingProvider:
    provider_version = "semantic-test-v1"

    def __init__(self):
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            related = any(
                phrase in lowered
                for phrase in (
                    "asynchronous",
                    "process interruption",
                    "worker crash",
                    "durable lease",
                    "fault-tolerant",
                )
            )
            vectors.append([1.0, 0.0] if related else [0.0, 1.0])
        return EmbeddingBatch(
            vectors=vectors,
            provider_version=self.provider_version,
            usage={"embedding_calls": 1, "embedding_tokens": len(texts) * 5},
        )


class FailingEmbeddingProvider:
    provider_version = "failing-test-v1"

    def embed(self, texts):
        raise RuntimeError("embedding unavailable")


class CapturingAnswerModel:
    provider = "demo"
    model = "retrieval-capture"

    def __init__(self):
        self.request = None

    def decide(self, request):
        self.request = request
        return AgentDecision(
            action_type="final_answer",
            content="Retrieved evidence was assembled with provenance.",
        )


def _contract(version: int, goal: str) -> GoalContract:
    return GoalContract(
        version=version,
        north_star_goal=goal,
        product_context="Test hybrid retrieval.",
        learning_contract="Keep retrieval observable.",
        scope_guardrails=["Do not leak across contracts."],
        success_criteria=["Retrieve relevant evidence."],
        is_active=version == 1,
    )


def test_hybrid_retrieval_improves_semantic_recall_without_contract_leakage():
    provider = SemanticTestEmbeddingProvider()
    with Session(engine) as session:
        contract = _contract(1, "Build reliable asynchronous Agent runtimes.")
        other_contract = _contract(2, "Unrelated private career plan.")
        task = AgentTask(
            goal_contract=contract,
            title="Recover asynchronous work",
            user_goal="Recover work after process interruption.",
            constraints=["Do not duplicate execution."],
            success_criteria=["Select evidence about resilient workers."],
        )
        session.add_all([contract, other_contract, task])
        session.flush()
        session.add_all(
            [
                AgentMemory(
                    goal_contract_id=contract.id,
                    memory_type="fact",
                    memory_key="worker-recovery",
                    content="Worker crash recovery uses a durable lease.",
                    source="runtime",
                    importance=2,
                ),
                AgentMemory(
                    goal_contract_id=contract.id,
                    memory_type="fact",
                    memory_key="visual-preference",
                    content="The dashboard uses a blue visual palette.",
                    source="user",
                    importance=5,
                ),
                AgentMemory(
                    goal_contract_id=other_contract.id,
                    memory_type="fact",
                    memory_key="other-contract-secret",
                    content="Worker crash recovery uses a durable lease and a private token.",
                    source="user",
                    importance=5,
                ),
            ]
        )
        related_skill = Skill(name="Distributed Jobs", category="agent_runtime")
        unrelated_skill = Skill(name="CSS", category="frontend")
        related_job = Job(
            company="Reliable AI",
            title="Agent Runtime Engineer",
            source_url="https://example.com/jobs/runtime",
            description="Build resilient runtimes.",
        )
        unrelated_job = Job(
            company="Design AI",
            title="Frontend Engineer",
            source_url="https://example.com/jobs/frontend",
            description="Build visual interfaces.",
        )
        related_job.requirements.append(
            JobRequirement(
                skill=related_skill,
                evidence_text="Design fault-tolerant background processing.",
            )
        )
        unrelated_job.requirements.append(
            JobRequirement(
                skill=unrelated_skill,
                evidence_text="Create polished responsive layouts.",
            )
        )
        session.add_all([related_job, unrelated_job])
        session.commit()

        lexical = assemble_memory_context(
            session,
            task,
            memory_limit=1,
            knowledge_limit=1,
            memory_token_budget=1200,
        )
        hybrid = assemble_memory_context(
            session,
            task,
            memory_limit=1,
            knowledge_limit=1,
            memory_token_budget=1200,
            embedding_provider=provider,
        )

        assert [item["memory_key"] for item in lexical.memories] == [
            "visual-preference"
        ]
        assert lexical.knowledge == []
        assert [item["memory_key"] for item in hybrid.memories] == [
            "worker-recovery"
        ]
        assert [item["skill"] for item in hybrid.knowledge] == ["Distributed Jobs"]
        assert hybrid.candidate_count == 2
        assert hybrid.retrieval_method == "hybrid"
        assert hybrid.embedding_usage["embedding_calls"] == 2
        assert hybrid.embedding_usage["cache_hits"] == 1
        assert hybrid.embedding_usage["cache_misses"] == 5
        assert hybrid.knowledge[0]["trust"] == "untrusted_public_evidence"
        assert provider.call_count == 2
        assert session.scalar(select(func.count(RetrievalEmbeddingCacheEntry.id))) == 5

        cached = assemble_memory_context(
            session,
            task,
            memory_limit=1,
            knowledge_limit=1,
            memory_token_budget=1200,
            embedding_provider=provider,
        )
        assert cached.embedding_usage == {
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "cache_hits": 6,
            "cache_misses": 0,
        }
        assert provider.call_count == 2

        worker_memory = session.scalar(
            select(AgentMemory).where(AgentMemory.memory_key == "worker-recovery")
        )
        worker_memory.content += " It supports fault-tolerant asynchronous work."
        session.commit()
        refreshed = assemble_memory_context(
            session,
            task,
            memory_limit=1,
            knowledge_limit=1,
            memory_token_budget=1200,
            embedding_provider=provider,
        )
        assert refreshed.embedding_usage["embedding_calls"] == 1
        assert refreshed.embedding_usage["cache_hits"] == 5
        assert refreshed.embedding_usage["cache_misses"] == 1
        assert provider.call_count == 3

        other_version = SemanticTestEmbeddingProvider()
        other_version.provider_version = "semantic-test-v2"
        version_isolated = assemble_memory_context(
            session,
            task,
            memory_limit=1,
            knowledge_limit=1,
            memory_token_budget=1200,
            embedding_provider=other_version,
        )
        assert version_isolated.embedding_usage["embedding_calls"] == 2
        assert version_isolated.embedding_usage["cache_hits"] == 1
        assert version_isolated.embedding_usage["cache_misses"] == 5
        assert other_version.call_count == 2

        model = CapturingAnswerModel()
        run = create_agent_run(session, task=task, model=model, max_steps=1)
        run_id = run.id
        task_id = task.id

    AgentLoopEngine(model, retrieval_embedding_provider=provider).run(run_id)

    assert model.request.memory_context["retrieval"]["method"] == "hybrid"
    assert model.request.memory_context["knowledge"][0]["skill"] == "Distributed Jobs"
    with Session(engine) as session:
        trace = session.scalar(
            select(ExecutionTrace).where(
                ExecutionTrace.task_id == task_id,
                ExecutionTrace.event_type == "retrieval",
            )
        )
        assert trace.status == "assembled"
        assert trace.metadata_json["embedding_provider_version"] == "semantic-test-v1"
        assert trace.metadata_json["knowledge_selected_count"] == 1


def test_embedding_failure_falls_back_to_lexical_retrieval():
    with Session(engine) as session:
        contract = _contract(1, "Build observable Agent systems.")
        task = AgentTask(
            goal_contract=contract,
            title="Inspect Agent traces",
            user_goal="Use an Agent trace to explain execution.",
            constraints=[],
            success_criteria=["Retrieve trace evidence."],
        )
        memory = AgentMemory(
            goal_contract=contract,
            memory_type="fact",
            memory_key="trace-evidence",
            content="Agent trace evidence explains execution decisions.",
            source="runtime",
            importance=4,
        )
        session.add_all([contract, task, memory])
        session.commit()

        context = assemble_memory_context(
            session,
            task,
            embedding_provider=FailingEmbeddingProvider(),
        )

        assert context.retrieval_method == "lexical_fallback"
        assert context.embedding_error == "RuntimeError"
        assert context.embedding_usage == {
            "embedding_calls": 0,
            "embedding_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        assert [item["memory_key"] for item in context.memories] == [
            "trace-evidence"
        ]


def test_external_retrieval_embedding_requires_explicit_data_egress(monkeypatch):
    monkeypatch.setenv("CAREEROPS_RETRIEVAL_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("CAREEROPS_ALLOW_EXTERNAL_RETRIEVAL", "false")

    with pytest.raises(ValueError, match="require.*ALLOW_EXTERNAL_RETRIEVAL"):
        create_configured_retrieval_embedding_provider()
