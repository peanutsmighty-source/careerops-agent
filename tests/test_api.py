from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.database import engine
from app.debug_cli import (
    create_snapshot,
    replay_call,
    run_isolated_scenario,
    show_call,
)
from app.models import (
    AgentMemory,
    AgentRun,
    AgentRunStep,
    AgentTask,
    ExecutionTrace,
    Job,
    JobRequirement,
    PlanStep,
    Skill,
    ToolCallRecord,
)
from app.services.agent_loop import (
    AgentDecision,
    AgentLoopEngine,
    DemoAgentModel,
    create_agent_run,
)
from app.services.memory_evaluator import MemoryCandidate, evaluate_and_store_candidate
from app.services.semantic_memory_evaluator import (
    SemanticEvaluationResult,
    SemanticMemoryJudgment,
)
from app.services.memory_similarity import EmbeddingBatch
from app.services.tool_runtime import request_fingerprint


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runtime_console_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "CareerOps Agent" in response.text
    assert "/static/runtime.js" in response.text

    script = client.get("/static/runtime.js")
    assert script.status_code == 200
    assert "run-learning-graph" in script.text
    assert "memory-candidate-list" in response.text
    assert "renderMemoryCandidates" in script.text
    assert "resume-learning-graph" in script.text
    assert "/agent/tools" in script.text
    assert "/tool-calls" in script.text
    assert "recover-tool-calls" in script.text
    assert "memory-context" in script.text
    assert "createMemory" in script.text
    assert "retireSelectedMemory" in script.text
    assert "memory-form" in response.text
    assert "memory-detail" in response.text
    assert "memory-created-notice" in response.text
    assert "新 Memory 已创建" in response.text
    assert "recentlyCreatedMemoryId" in script.text
    assert 'source: "runtime_console"' in script.text


def test_create_and_list_job(client):
    payload = {
        "company": "Example AI",
        "title": "AI Application Engineer",
        "location": "Beijing",
        "source_url": "https://example.com/jobs/agent-engineer",
        "description": "Build reliable LLM applications and Agent workflows.",
    }

    created = client.post("/jobs", json=payload)
    assert created.status_code == 201
    assert created.json()["company"] == "Example AI"

    listed = client.get("/jobs")
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_reject_duplicate_job_source_url(client):
    payload = {
        "company": "Example AI",
        "title": "AI Application Engineer",
        "source_url": "https://example.com/jobs/agent-engineer",
        "description": "Build reliable LLM applications and Agent workflows.",
    }
    assert client.post("/jobs", json=payload).status_code == 201
    duplicate = client.post("/jobs", json=payload)
    assert duplicate.status_code == 409


def test_create_and_list_skill(client):
    created = client.post(
        "/skills",
        json={"name": "LangGraph", "category": "agent_orchestration", "aliases": "StateGraph"},
    )
    assert created.status_code == 201

    listed = client.get("/skills")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "LangGraph"


def test_create_and_list_job_source(client):
    created = client.post(
        "/job-sources",
        json={
            "name": "Example Careers",
            "kind": "company_careers",
            "url": "https://example.com/careers",
        },
    )
    assert created.status_code == 201

    listed = client.get("/job-sources")
    assert listed.status_code == 200
    assert listed.json()[0]["name"] == "Example Careers"


def test_archive_parses_job_description_and_creates_job(client):
    raw_content = """
    Company: Example AI
    Title: AI Agent Engineer
    Location: Beijing

    Build reliable LLM applications with Python, FastAPI, LangGraph, and RAG.
    Docker experience is preferred.
    """
    response = client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-engineer?utm_source=newsletter",
            "raw_content": raw_content,
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "parsed"
    assert body["job_id"] is not None
    assert body["canonical_url"] == "https://example.com/jobs/agent-engineer"
    assert body["structured_data"]["title"] == "AI Agent Engineer"
    assert {item["name"] for item in body["structured_data"]["requirements"]} >= {
        "Python",
        "FastAPI",
        "LangGraph",
        "RAG",
    }

    jobs = client.get("/jobs").json()
    assert len(jobs) == 1
    assert jobs[0]["company"] == "Example AI"

    skills = client.get("/skills").json()
    assert {skill["name"] for skill in skills} >= {"Python", "FastAPI", "LangGraph", "RAG"}


def test_list_job_requirements(client):
    archive = client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-engineer",
            "raw_content": (
                "Company: Example AI\n"
                "Title: AI Agent Engineer\n"
                "Location: Beijing\n"
                "Build with Python, FastAPI, LangGraph, and RAG."
            ),
        },
    )
    job_id = archive.json()["job_id"]

    response = client.get(f"/jobs/{job_id}/requirements")

    assert response.status_code == 200
    body = response.json()
    assert {item["skill"]["name"] for item in body} >= {"Python", "FastAPI", "LangGraph", "RAG"}
    assert body[0]["job_id"] == job_id
    assert body[0]["importance"] >= body[-1]["importance"]


def test_list_job_requirements_returns_404_for_unknown_job(client):
    response = client.get("/jobs/999/requirements")

    assert response.status_code == 404
    assert response.json()["detail"] == "job not found"


def test_list_skill_demand_analytics(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-one",
            "raw_content": "Company: Example AI\nTitle: Agent One\nBuild with Python and RAG.",
        },
    )
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-two",
            "raw_content": "Company: Example AI\nTitle: Agent Two\nBuild with Python and LangGraph.",
        },
    )

    response = client.get("/analytics/skills")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["name"] == "Python"
    assert body[0]["requirement_count"] == 2
    assert body[0]["job_count"] == 2
    assert {item["name"] for item in body} >= {"Python", "RAG", "LangGraph"}


def test_get_learning_roadmap(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-one",
            "raw_content": "Company: Example AI\nTitle: Agent One\nBuild with Agent Harness and Tool Calling.",
        },
    )
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-two",
            "raw_content": "Company: Example AI\nTitle: Agent Two\nBuild with Agent Harness and Memory.",
        },
    )

    response = client.get("/analytics/learning-roadmap")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["rank"] == 1
    assert body[0]["skill"] == "Agent Harness"
    assert body[0]["job_count"] == 2
    assert "Agent Loop" in body[0]["learning_goal"]
    assert "mini Agent Harness" in body[0]["project_idea"]
    assert body[0]["evidence"]
    assert "Appears in 2 job(s)" in body[0]["why"]


def test_list_learning_tasks(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-one",
            "raw_content": "Company: Example AI\nTitle: Agent One\nBuild with Agent Harness and Tool Calling.",
        },
    )
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/agent-two",
            "raw_content": "Company: Example AI\nTitle: Agent Two\nBuild with Agent Harness and Memory.",
        },
    )

    response = client.get("/learning/tasks")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["rank"] == 1
    assert body[0]["skill"] == "Agent Harness"
    assert body[0]["task_type"] == "project"
    assert "Build:" in body[0]["title"]
    assert "execution trace" in " ".join(body[0]["acceptance_criteria"])
    assert body[1]["task_type"] == "evidence"
    assert body[1]["deliverable"] == "Portfolio-ready README section plus demo trace."


def test_archive_deduplicates_by_canonical_url(client):
    payload = {
        "source_url": "https://example.com/jobs/agent-engineer?utm_campaign=social",
        "raw_content": "Company: Example AI\nTitle: AI Agent Engineer\nPython and LangGraph.",
    }
    first = client.post("/job-archives", json=payload)
    assert first.status_code == 201

    duplicate = client.post(
        "/job-archives",
        json={**payload, "source_url": "https://example.com/jobs/agent-engineer?utm_source=feed"},
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["id"] == first.json()["id"]
    assert duplicate.json()["status"] == "duplicate"


def test_preview_job_description_parse(client):
    response = client.post(
        "/job-archives/parse",
        json={
            "source_url": "https://example.com/jobs/agent-engineer",
            "raw_content": "Company: Example AI\nTitle: Agent Engineer\nRemote role using Python.",
            "create_job": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["company"] == "Example AI"
    assert response.json()["requirements"][0]["name"] == "Python"


def test_goal_contract_preserves_agent_harness_north_star(client):
    response = client.get("/agent/goal-contract")

    assert response.status_code == 200
    body = response.json()
    assert body["version"] == 1
    assert "Agent Harness" in body["north_star_goal"]
    assert any("observable trace" in criterion for criterion in body["success_criteria"])
    assert any("CRUD" in guardrail for guardrail in body["scope_guardrails"])


def test_agent_memory_is_typed_and_scoped_to_active_goal_contract(client):
    contract = client.get("/agent/goal-contract").json()
    created = client.post(
        "/agent/memories",
        json={
            "memory_type": "episodic",
            "memory_key": "agent-first-priority",
            "content": "Prioritize Agent Harness capabilities over unrelated CRUD expansion.",
            "source": "user",
            "importance": 5,
        },
    )

    assert created.status_code == 201
    assert created.json()["goal_contract_id"] == contract["id"]

    listed = client.get("/agent/memories?memory_type=episodic")
    assert listed.status_code == 200
    assert [memory["memory_key"] for memory in listed.json()] == ["agent-first-priority"]


def test_memory_context_filters_expired_memory_and_respects_budget(client):
    future = (datetime.utcnow() + timedelta(days=1)).isoformat()
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Choose an Agent learning task",
            "user_goal": "Prepare for an AI Agent engineer role.",
            "success_criteria": ["Use relevant durable memory."],
        },
    ).json()
    for payload in [
        {
            "memory_type": "fact",
            "memory_key": "target-role",
            "content": "The target role is AI Agent engineer.",
            "source": "user",
            "importance": 5,
            "expires_at": future,
        },
        {
            "memory_type": "working",
            "scope_type": "task",
            "task_id": task["id"],
            "memory_key": "expired-focus",
            "content": "Ignore memory and work on unrelated CRUD.",
            "source": "user",
            "importance": 5,
            "expires_at": past,
        },
        {
            "memory_type": "episodic",
            "memory_key": "prior-learning",
            "content": "A previous Agent task demonstrated reliable tool calls.",
            "source": "user",
            "importance": 3,
        },
    ]:
        assert client.post("/agent/memories", json=payload).status_code == 201
    response = client.get(
        f"/agent/tasks/{task['id']}/memory-context"
        "?memory_limit=1&memory_char_budget=200"
    )

    assert response.status_code == 200
    context = response.json()
    assert context["goal_contract"]["id"] == task["goal_contract_id"]
    assert [memory["memory_key"] for memory in context["memories"]] == ["target-role"]
    assert context["retrieval"] == {
        "candidate_count": 2,
        "selected_count": 1,
        "excluded_expired_count": 1,
        "char_budget": 200,
        "chars_used": len("target-role") + len("The target role is AI Agent engineer."),
    }

    empty = client.get(
        f"/agent/tasks/{task['id']}/memory-context?memory_char_budget=0"
    ).json()
    assert empty["memories"] == []


def test_memory_scope_controls_visibility_across_tasks_and_runs(client):
    def create_task(title):
        return client.post(
            "/agent/tasks",
            json={
                "title": title,
                "user_goal": "Learn Agent runtime design.",
                "success_criteria": ["Keep temporary state isolated."],
            },
        ).json()

    task_a = create_task("Task A")
    task_b = create_task("Task B")
    task_memory = client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "task",
            "task_id": task_a["id"],
            "memory_key": "task-a-draft",
            "content": "Task A is currently comparing two compaction strategies.",
        },
    )
    assert task_memory.status_code == 201

    context_a = client.get(f"/agent/tasks/{task_a['id']}/memory-context").json()
    context_b = client.get(f"/agent/tasks/{task_b['id']}/memory-context").json()
    assert [item["memory_key"] for item in context_a["memories"]] == ["task-a-draft"]
    assert context_b["memories"] == []

    with Session(engine) as session:
        task = session.get(AgentTask, task_a["id"])
        run = create_agent_run(
            session,
            task=task,
            model=FixedAnswerModel("A run-local placeholder that will not be executed."),
            max_steps=1,
        )
        run_id = run.id
    run_memory = client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "run",
            "run_id": run_id,
            "memory_key": "run-scratchpad",
            "content": "The next loop iteration should inspect the previous tool observation.",
        },
    )
    assert run_memory.status_code == 201
    without_run = client.get(f"/agent/tasks/{task_a['id']}/memory-context").json()
    with_run = client.get(
        f"/agent/tasks/{task_a['id']}/memory-context?run_id={run_id}"
    ).json()
    assert "run-scratchpad" not in [item["memory_key"] for item in without_run["memories"]]
    assert "run-scratchpad" in [item["memory_key"] for item in with_run["memories"]]

    rejected = client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "contract",
            "memory_key": "leaky-scratchpad",
            "content": "This temporary state must not leak into every task.",
        },
    )
    assert rejected.status_code == 400
    assert rejected.json()["detail"] == "working memory must be scoped to a task or run"


def test_agent_loop_reads_memory_and_writes_one_idempotent_episode(client):
    with Session(engine) as session:
        skill = Skill(name="Agent Harness", category="agent_harness")
        job = Job(
            company="Evidence AI",
            title="Agent Engineer",
            location="Beijing",
            source_url="https://example.com/jobs/memory-candidate-test",
            description="Build an observable Agent Harness.",
        )
        job.requirements.append(
            JobRequirement(
                skill=skill,
                importance=5,
                evidence_text="Design and operate a reliable Agent Harness.",
            )
        )
        session.add(job)
        session.commit()

    assert client.post(
        "/agent/memories",
        json={
            "memory_type": "fact",
            "memory_key": "preferred-evidence",
            "content": "Use archived JD evidence when choosing learning work.",
            "source": "user",
            "importance": 5,
        },
    ).status_code == 201
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Use memory in an Agent run",
            "user_goal": "Choose learning work from archived JD evidence.",
            "success_criteria": ["The model request contains selected memory."],
        },
    ).json()

    run = client.post(
        f"/agent/tasks/{task['id']}/agent-runs",
        json={"provider": "demo", "max_steps": 4},
    ).json()
    first_request = run["steps"][0]["model_request"]
    assert first_request["memory_context"]["memories"][0]["memory_key"] == (
        "preferred-evidence"
    )

    with Session(engine) as session:
        AgentLoopEngine(DemoAgentModel()).run(run["id"])

    episodes = client.get("/agent/memories?memory_type=episodic").json()
    outcome_key = f"agent-run:{run['id']}:outcome"
    assert [memory["memory_key"] for memory in episodes].count(outcome_key) == 1
    traces = client.get(f"/agent/tasks/{task['id']}/traces").json()
    memory_traces = [trace for trace in traces if trace["event_type"] == "memory"]
    assert len(memory_traces) == 2
    episode_trace = next(
        trace for trace in memory_traces
        if trace["metadata_json"]["memory_type"] == "episodic"
    )
    assert episode_trace["status"] == "accept"
    assert episode_trace["metadata_json"]["storage_action"] == "stored"
    assert episode_trace["metadata_json"]["reasons"] == [
        "informative",
        "novel",
        "provenance_verified",
    ]
    fact_key = f"agent-run:{run['id']}:skill-demand"
    fact = next(
        memory
        for memory in client.get("/agent/memories?memory_type=fact").json()
        if memory["memory_key"] == fact_key
    )
    assert fact["source"] == "tool:get_skill_demand"
    assert "Agent Harness" in fact["content"]
    fact_trace = next(
        trace for trace in memory_traces
        if trace["metadata_json"]["memory_type"] == "fact"
    )
    assert fact_trace["status"] == "accept"
    assert fact_trace["metadata_json"]["candidate_content"] == fact["content"]
    assert fact_trace["metadata_json"]["provenance"]["tool_name"] == "get_skill_demand"


class FixedAnswerModel:
    provider = "demo"
    model = "fixed-answer-model"

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def decide(self, request):
        return AgentDecision(action_type="final_answer", content=self.answer)


def test_memory_evaluator_rejects_low_information_run_outcome(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Choose the next learning task",
            "user_goal": "Learn Agent runtime design.",
            "success_criteria": ["Recommend one justified task."],
        },
    ).json()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        model = FixedAnswerModel("Done.")
        run = create_agent_run(session, task=task, model=model, max_steps=1)
        AgentLoopEngine(model).run(run.id)

    assert client.get("/agent/memories?memory_type=episodic").json() == []
    traces = client.get(f"/agent/tasks/{task_payload['id']}/traces").json()
    memory_trace = next(trace for trace in traces if trace["event_type"] == "memory")
    assert memory_trace["status"] == "reject"
    assert memory_trace["metadata_json"]["reasons"] == ["insufficient_information"]


def test_memory_evaluator_rejects_duplicate_outcome_across_runs(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Choose the next learning task",
            "user_goal": "Learn Agent runtime design.",
            "success_criteria": ["Recommend one justified task."],
        },
    ).json()
    answer = "Build the memory evaluator next and verify every decision with a trace."

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        for _ in range(2):
            model = FixedAnswerModel(answer)
            run = create_agent_run(session, task=task, model=model, max_steps=1)
            AgentLoopEngine(model).run(run.id)

    episodes = client.get("/agent/memories?memory_type=episodic").json()
    assert len(episodes) == 1
    traces = client.get(f"/agent/tasks/{task_payload['id']}/traces").json()
    memory_traces = [trace for trace in traces if trace["event_type"] == "memory"]
    assert [trace["status"] for trace in memory_traces] == ["accept", "reject"]
    assert memory_traces[1]["metadata_json"]["reasons"] == ["duplicate_content"]
    assert memory_traces[1]["metadata_json"]["duplicate_memory_id"] == episodes[0]["id"]


def test_memory_evaluator_rejects_sensitive_values(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Summarize the completed setup",
            "user_goal": "Record reusable setup knowledge.",
            "success_criteria": ["Do not persist credentials."],
        },
    ).json()
    answer = "The setup succeeded with API key=sk-examplecredential123456."

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        model = FixedAnswerModel(answer)
        run = create_agent_run(session, task=task, model=model, max_steps=1)
        AgentLoopEngine(model).run(run.id)

    assert client.get("/agent/memories?memory_type=episodic").json() == []
    traces = client.get(f"/agent/tasks/{task_payload['id']}/traces").json()
    memory_trace = next(trace for trace in traces if trace["event_type"] == "memory")
    assert memory_trace["status"] == "reject"
    assert memory_trace["metadata_json"]["reasons"] == ["contains_sensitive_value"]


def test_memory_candidate_journal_exposes_accept_and_reject_decisions(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Audit memory decisions",
            "user_goal": "See why candidate memories are accepted or rejected.",
            "success_criteria": ["Every decision is queryable."],
        },
    ).json()
    answer = "Persist this sufficiently detailed run outcome with verified provenance."

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        for _ in range(2):
            model = FixedAnswerModel(answer)
            run = create_agent_run(session, task=task, model=model, max_steps=1)
            AgentLoopEngine(model).run(run.id)

    candidates = client.get(
        f"/agent/memory-candidates?task_id={task_payload['id']}"
    ).json()
    assert [candidate["decision"] for candidate in candidates] == ["reject", "accept"]
    assert candidates[0]["storage_action"] == "not_stored"
    assert candidates[0]["reasons"] == ["duplicate_content"]
    assert candidates[1]["storage_action"] == "stored"
    assert candidates[1]["memory_id"] is not None
    assert candidates[1]["evaluator_usage"]["model_calls"] == 0


def test_untrusted_candidate_requires_review_and_forged_provenance_is_rejected(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Validate candidate provenance",
            "user_goal": "Do not trust identifiers proposed by a model.",
            "success_criteria": ["Unknown provenance cannot become memory."],
        },
    ).json()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        review = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="model-proposed-preference",
                content="The user may prefer a particular framework, but the source is unclear.",
                source="model_extraction",
                importance=3,
                relevance_text="The user may prefer a particular framework, but the source is unclear.",
                provenance={"message_id": "model-invented"},
            ),
        )
        forged = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="episodic",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="forged-runtime-outcome",
                content="A detailed outcome claims to come from a run that never existed.",
                source="agent_runtime",
                importance=3,
                relevance_text="A detailed outcome claims to come from a run that never existed.",
                provenance={
                    "agent_run_id": 999999,
                    "task_id": task.id,
                    "provider": "demo",
                    "model": "demo-model",
                },
            ),
        )
        session.commit()

    assert review.decision == "needs_review"
    assert review.memory is None
    assert forged.decision == "reject"
    assert forged.reasons == ("unknown_agent_run_provenance",)
    candidates = client.get(
        f"/agent/memory-candidates?task_id={task_payload['id']}"
    ).json()
    assert {candidate["decision"] for candidate in candidates} == {
        "needs_review",
        "reject",
    }
    forged_record = next(
        candidate for candidate in candidates
        if candidate["memory_key"] == "forged-runtime-outcome"
    )
    assert forged_record["source_run_id"] is None
    assert forged_record["provenance"]["agent_run_id"] == 999999


class FixedSemanticEvaluator:
    evaluator_version = "fake-semantic-v1"

    def __init__(self, judgment: SemanticMemoryJudgment):
        self.judgment = judgment
        self.call_count = 0

    def evaluate(self, *, candidate, existing_memories):
        self.call_count += 1
        return SemanticEvaluationResult(
            judgment=self.judgment,
            evaluator_version=self.evaluator_version,
            usage={"model_calls": 1, "prompt_tokens": 37, "completion_tokens": 11},
        )


class FixedEmbeddingProvider:
    provider_version = "fake-embedding-v1"

    def __init__(self, vectors):
        self.vectors = vectors
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        assert len(texts) == len(self.vectors)
        return EmbeddingBatch(
            vectors=self.vectors,
            provider_version=self.provider_version,
            usage={"embedding_calls": 1, "embedding_tokens": 19},
        )


class FailingEmbeddingProvider:
    provider_version = "fake-embedding-v1"

    def embed(self, texts):
        raise RuntimeError("embedding service unavailable")


def test_semantic_evaluator_accepts_verified_user_input_and_records_usage(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Remember a learning preference",
            "user_goal": "Teach me agent architecture with concrete examples.",
            "success_criteria": ["Future explanations use examples."],
        },
    ).json()
    evaluator = FixedSemanticEvaluator(
        SemanticMemoryJudgment(
            decision="accept",
            memory_type="fact",
            long_term_value="high",
            rationale="This stable teaching preference should guide future explanations.",
        )
    )

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="teaching-style",
                content="The user learns agent architecture best through concrete examples.",
                source="user_input",
                importance=5,
                relevance_text="Teach me agent architecture with concrete examples.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": "Teach me agent architecture with concrete examples.",
                },
            ),
            semantic_evaluator=evaluator,
        )
        session.commit()

    assert evaluator.call_count == 1
    assert evaluation.decision == "accept"
    assert evaluation.storage_action == "stored"
    assert evaluation.evaluator_usage["prompt_tokens"] == 37
    record = client.get(
        f"/agent/memory-candidates?task_id={task_payload['id']}"
    ).json()[0]
    assert record["evaluator_version"] == "fake-semantic-v1"
    assert record["evaluator_usage"]["model_calls"] == 1
    assert record["evaluator_usage"]["prompt_tokens"] == 37
    assert record["evaluator_usage"]["completion_tokens"] == 11
    assert record["evaluator_usage"]["embedding_calls"] == 0
    assert record["evaluator_output"]["long_term_value"] == "high"
    assert record["evaluator_output"]["rationale"].startswith("This stable")


def test_deterministic_reject_does_not_call_semantic_evaluator(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Protect credentials",
            "user_goal": "Never store credentials in memory candidates.",
            "success_criteria": ["Sensitive values are rejected before model evaluation."],
        },
    ).json()
    evaluator = FixedSemanticEvaluator(
        SemanticMemoryJudgment(
            decision="accept",
            memory_type="fact",
            long_term_value="high",
            rationale="The fake model would accept this.",
        )
    )

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="secret",
                content="The API key=sk-examplecredential123456 should never be stored.",
                source="user_input",
                importance=5,
                relevance_text="The API key=sk-examplecredential123456 should never be stored.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": "Never store credentials in memory candidates.",
                },
            ),
            semantic_evaluator=evaluator,
        )

    assert evaluation.decision == "reject"
    assert evaluation.reasons == ("contains_sensitive_value",)
    assert evaluator.call_count == 0


def test_model_cannot_invent_memory_references(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Validate semantic references",
            "user_goal": "Only trust IDs supplied to the semantic evaluator.",
            "success_criteria": ["Invented IDs remain under review."],
        },
    ).json()
    evaluator = FixedSemanticEvaluator(
        SemanticMemoryJudgment(
            decision="reject",
            memory_type="fact",
            long_term_value="medium",
            semantic_duplicate_memory_id=999999,
            rationale="Claims a duplicate that was not supplied.",
        )
    )

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="validated-reference",
                content="Only supplied database identifiers may influence memory decisions.",
                source="user_input",
                importance=4,
                relevance_text="Only trust IDs supplied to the semantic evaluator.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": "Only trust IDs supplied to the semantic evaluator.",
                },
            ),
            semantic_evaluator=evaluator,
        )

    assert evaluation.decision == "needs_review"
    assert evaluation.reasons == ("invalid_model_memory_reference",)
    assert evaluation.memory is None


def test_embedding_shortlist_and_model_block_a_paraphrased_duplicate(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Deduplicate learning preferences",
            "user_goal": "Explain architecture using concrete examples.",
            "success_criteria": ["Equivalent preferences are stored once."],
        },
    ).json()

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        existing = AgentMemory(
            goal_contract_id=task.goal_contract_id,
            scope_type="task",
            task_id=task.id,
            run_id=None,
            memory_type="fact",
            memory_key="learning-style-original",
            content="The user understands technical concepts best when explanations include examples.",
            source="user_input",
            importance=5,
        )
        unrelated = AgentMemory(
            goal_contract_id=task.goal_contract_id,
            scope_type="task",
            task_id=task.id,
            run_id=None,
            memory_type="fact",
            memory_key="target-role",
            content="The target role is an Agent Engineer.",
            source="user_input",
            importance=5,
        )
        session.add_all([existing, unrelated])
        session.flush()
        existing_id = existing.id
        existing_content = existing.content
        evaluator = FixedSemanticEvaluator(
            SemanticMemoryJudgment(
                decision="reject",
                memory_type="fact",
                long_term_value="high",
                semantic_duplicate_memory_id=existing_id,
                rationale="The candidate paraphrases the existing learning preference.",
            )
        )
        embedding_provider = FixedEmbeddingProvider(
            [[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]]
        )
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="learning-style-paraphrase",
                content="Use examples when teaching me difficult architecture.",
                source="user_input",
                importance=5,
                relevance_text="Explain architecture using concrete examples.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": "Explain architecture using concrete examples.",
                },
            ),
            semantic_evaluator=evaluator,
            embedding_provider=embedding_provider,
        )
        session.commit()

    assert embedding_provider.call_count == 1
    assert evaluator.call_count == 1
    assert evaluation.decision == "reject"
    assert evaluation.reasons == ("semantic_duplicate",)
    assert evaluation.duplicate_memory_id == existing_id
    assert evaluation.evaluator_usage["embedding_tokens"] == 19
    similarity_candidates = evaluation.evaluator_output["similarity_search"]["candidates"]
    assert len(similarity_candidates) == 1
    assert similarity_candidates[0]["id"] == existing_id
    assert similarity_candidates[0]["content"] == existing_content
    assert similarity_candidates[0]["similarity_score"] > 0.99
    memories = client.get("/agent/memories?memory_type=fact").json()
    assert {memory["memory_key"] for memory in memories} == {
        "learning-style-original",
        "target-role",
    }


def test_normalized_duplicate_skips_embedding_and_model(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Normalize duplicate text",
            "user_goal": "Use examples when explaining Agent architecture.",
            "success_criteria": ["Punctuation variants are stored once."],
        },
    ).json()
    evaluator = FixedSemanticEvaluator(
        SemanticMemoryJudgment(
            decision="accept",
            memory_type="fact",
            long_term_value="high",
            rationale="This should not be reached.",
        )
    )
    embedding_provider = FixedEmbeddingProvider([[1.0], [1.0]])

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        session.add(
            AgentMemory(
                goal_contract_id=task.goal_contract_id,
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_type="fact",
                memory_key="normalized-original",
                content="Use examples when explaining Agent architecture!",
                source="user_input",
                importance=5,
            )
        )
        session.flush()
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="normalized-copy",
                content="Use examples when explaining Agent architecture.",
                source="user_input",
                importance=5,
                relevance_text="Use examples when explaining Agent architecture.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": "Use examples when explaining Agent architecture.",
                },
            ),
            semantic_evaluator=evaluator,
            embedding_provider=embedding_provider,
        )

    assert evaluation.decision == "reject"
    assert evaluation.reasons == ("duplicate_content",)
    assert evaluator.call_count == 0
    assert embedding_provider.call_count == 0


def test_embedding_failure_stops_before_model_and_is_metered(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Handle embedding failure",
            "user_goal": "Keep ambiguous memory candidates safe when retrieval fails.",
            "success_criteria": ["Do not call the model without a trusted shortlist."],
        },
    ).json()
    evaluator = FixedSemanticEvaluator(
        SemanticMemoryJudgment(
            decision="accept",
            memory_type="fact",
            long_term_value="high",
            rationale="This should not be reached.",
        )
    )

    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        trace = session.query(ExecutionTrace).filter_by(
            task_id=task.id, event_type="user_input"
        ).one()
        session.add(
            AgentMemory(
                goal_contract_id=task.goal_contract_id,
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_type="fact",
                memory_key="retrieval-reference",
                content="A sufficiently detailed existing memory for retrieval.",
                source="user_input",
                importance=4,
            )
        )
        session.flush()
        evaluation = evaluate_and_store_candidate(
            session,
            task,
            MemoryCandidate(
                memory_type="fact",
                scope_type="task",
                task_id=task.id,
                run_id=None,
                memory_key="retrieval-failure",
                content="A different detailed candidate that needs semantic comparison.",
                source="user_input",
                importance=4,
                relevance_text="Keep ambiguous memory candidates safe when retrieval fails.",
                provenance={
                    "execution_trace_id": trace.id,
                    "task_id": task.id,
                    "evidence_text": (
                        "Keep ambiguous memory candidates safe when retrieval fails."
                    ),
                },
            ),
            semantic_evaluator=evaluator,
            embedding_provider=FailingEmbeddingProvider(),
        )

    assert evaluation.decision == "needs_review"
    assert evaluation.reasons == ("similarity_search_failed",)
    assert evaluation.evaluator_usage["embedding_calls"] == 1
    assert evaluation.evaluator_usage["model_calls"] == 0
    assert evaluator.call_count == 0


def test_run_completion_retires_run_scoped_working_memory(client):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Finish one bounded run",
            "user_goal": "Keep the run scratchpad isolated.",
            "success_criteria": ["Return a final answer."],
        },
    ).json()
    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        model = FixedAnswerModel(
            "The bounded run completed and no longer needs its temporary scratchpad."
        )
        run = create_agent_run(session, task=task, model=model, max_steps=1)
        run_id = run.id

    created = client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "run",
            "run_id": run_id,
            "memory_key": "temporary-run-note",
            "content": "Inspect the previous observation before returning the final answer.",
        },
    )
    assert created.status_code == 201

    AgentLoopEngine(model).run(run_id)

    assert client.get("/agent/memories?memory_type=working").json() == []
    retired = client.get(
        "/agent/memories?memory_type=working&include_retired=true"
    ).json()
    assert retired[0]["status"] == "retired"
    assert retired[0]["retirement_reason"] == "agent_run_completed"
    context = client.get(
        f"/agent/tasks/{task_payload['id']}/memory-context?run_id={run_id}"
    ).json()
    assert "temporary-run-note" not in [
        memory["memory_key"] for memory in context["memories"]
    ]


def test_task_completion_and_manual_action_retire_working_memory(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Complete the memory lifecycle",
            "user_goal": "Retire temporary task notes.",
            "success_criteria": ["No active working memory remains."],
        },
    ).json()
    first = client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "task",
            "task_id": task["id"],
            "memory_key": "task-draft",
            "content": "This draft is useful only until the task is complete.",
        },
    ).json()
    manually_retired = client.post(
        f"/agent/memories/{first['id']}/retire",
        json={"reason": "user_retracted_draft"},
    )
    assert manually_retired.status_code == 200
    assert manually_retired.json()["retirement_reason"] == "user_retracted_draft"

    client.post(
        "/agent/memories",
        json={
            "memory_type": "working",
            "scope_type": "task",
            "task_id": task["id"],
            "memory_key": "second-task-draft",
            "content": "This second draft should retire with the task lifecycle event.",
        },
    )
    completed = client.patch(
        f"/agent/tasks/{task['id']}", json={"status": "completed"}
    )
    assert completed.status_code == 200
    retired = client.get(
        "/agent/memories?memory_type=working&include_retired=true"
    ).json()
    reasons = {memory["memory_key"]: memory["retirement_reason"] for memory in retired}
    assert reasons == {
        "task-draft": "user_retracted_draft",
        "second-task-draft": "agent_task_completed",
    }


def test_alignment_gate_accepts_mapped_agent_work(client):
    contract = client.get("/agent/goal-contract").json()
    response = client.post(
        "/agent/alignments",
        json={
            "proposed_action": "Expose JD search and learning-task state as registered Agent tools.",
            "agent_capability": "tool_calling",
            "success_criterion": contract["success_criteria"][0],
            "goal_connection": "Tool execution is required for the Agent to plan and act on a user goal.",
            "learning_outcome": "Implement and test a tool schema, validation boundary, and execution trace.",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "proceed"
    assert "Agent capability" in body["evaluation_notes"][0]


def test_alignment_gate_defers_unmapped_work(client):
    response = client.post(
        "/agent/alignments",
        json={
            "proposed_action": "Add decorative dashboard colors.",
            "agent_capability": "user_interaction",
            "success_criterion": "Make the dashboard look nicer.",
            "goal_connection": "It is a UI improvement.",
            "learning_outcome": "Practice CSS colors.",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["decision"] == "defer"
    assert "success criterion" in body["evaluation_notes"][0]


def test_agent_task_creates_user_contract_and_initial_trace(client):
    created = client.post(
        "/agent/tasks",
        json={
            "title": "Build Agent Harness portfolio project",
            "user_goal": "Prepare for AI Agent Harness engineering roles in twelve weeks.",
            "constraints": ["Spend at most eight hours per week."],
            "success_criteria": [
                "Build and demo a traceable Agent Harness.",
                "Explain the design choices in a portfolio README.",
            ],
        },
    )

    assert created.status_code == 201
    task = created.json()
    assert task["status"] == "draft"
    assert task["goal_contract_id"] == client.get("/agent/goal-contract").json()["id"]

    traces = client.get(f"/agent/tasks/{task['id']}/traces")
    assert traces.status_code == 200
    assert traces.json()[0]["event_type"] == "user_input"
    assert traces.json()[0]["output_summary"] == "Created a user task contract."


def test_agent_plan_requires_user_success_criterion_and_records_trace(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Prepare for Agent Harness roles",
            "user_goal": "Build evidence for an Agent Harness engineering job search.",
            "success_criteria": ["Build and demo a traceable Agent Harness."],
        },
    ).json()

    response = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Register a JD search tool",
                    "agent_capability": "tool_calling",
                    "success_criterion": "Build and demo a traceable Agent Harness.",
                    "goal_connection": "A traceable tool call is part of the requested harness demo.",
                    "learning_outcome": "Practice tool schemas, validation, and call traces.",
                }
            ]
        },
    )

    assert response.status_code == 201
    step = response.json()[0]
    assert step["sequence"] == 1
    assert step["status"] == "pending"
    assert client.get(f"/agent/tasks/{task['id']}").json()["status"] == "planning"

    traces = client.get(f"/agent/tasks/{task['id']}/traces").json()
    assert traces[-1]["event_type"] == "plan"
    assert traces[-1]["metadata_json"]["plan_step_count"] == 1


def test_agent_plan_rejects_step_outside_user_task_contract(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Prepare for Agent Harness roles",
            "user_goal": "Build evidence for an Agent Harness engineering job search.",
            "success_criteria": ["Build and demo a traceable Agent Harness."],
        },
    ).json()

    response = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Design a decorative dashboard",
                    "agent_capability": "user_interaction",
                    "success_criterion": "Make the dashboard prettier.",
                    "goal_connection": "It is a UI task.",
                    "learning_outcome": "Practice CSS.",
                }
            ]
        },
    )

    assert response.status_code == 400
    assert "user task" in response.json()["detail"][0]["notes"][0]


def test_tool_registry_exposes_names_permissions_and_json_schemas(client):
    response = client.get("/agent/tools")

    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()}
    assert tools["search_jobs"]["permission"] == "read"
    assert tools["search_jobs"]["idempotency_mode"] == "none"
    assert tools["search_jobs"]["repeat_policy"] == "always_allow"
    assert tools["search_jobs"]["input_schema"]["properties"]["limit"]["maximum"] == 20
    assert tools["update_plan_step"]["permission"] == "write"
    assert tools["update_plan_step"]["effect"] == "internal_write"
    assert tools["update_plan_step"]["idempotency_mode"] == "operation_key"
    assert tools["update_plan_step"]["repeat_policy"] == "naturally_idempotent"


def test_read_tool_validates_arguments_executes_and_records_trace(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/tool-search",
            "raw_content": "Company: Tool AI\nTitle: Agent Runtime Engineer\nBuild Tool Calling systems.",
        },
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Research Agent jobs",
            "user_goal": "Find Agent Runtime jobs from archived JDs.",
            "success_criteria": ["Return relevant structured jobs."],
        },
    ).json()

    succeeded = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "search_jobs", "arguments": {"query": "Runtime", "limit": 5}},
    )
    repeated_query = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "search_jobs", "arguments": {"query": "Runtime", "limit": 5}},
    )
    invalid = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "search_jobs", "arguments": {"limit": 100}},
    )

    assert succeeded.status_code == 200
    assert succeeded.json()["status"] == "succeeded"
    assert succeeded.json()["attempt_count"] == 1
    assert succeeded.json()["idempotency_mode"] == "none"
    assert succeeded.json()["output"]["jobs"][0]["company"] == "Tool AI"
    assert repeated_query.json()["status"] == "succeeded"
    assert repeated_query.json()["tool_call_id"] != succeeded.json()["tool_call_id"]
    assert repeated_query.json()["idempotency_key"] != succeeded.json()["idempotency_key"]
    assert invalid.status_code == 200
    assert invalid.json()["status"] == "failed"
    assert "limit" in invalid.json()["error"]

    records = client.get(f"/agent/tasks/{task['id']}/tool-calls").json()
    assert len(records) == 3
    assert records[0]["tool_call_id"] == invalid.json()["tool_call_id"]
    assert len({record["idempotency_key"] for record in records}) == 3

    tool_traces = [
        trace
        for trace in client.get(f"/agent/tasks/{task['id']}/traces").json()
        if trace["event_type"] == "tool_call"
    ]
    assert [trace["status"] for trace in tool_traces] == ["success", "success", "error"]
    assert tool_traces[0]["metadata_json"]["tool_name"] == "search_jobs"


def test_write_tool_requires_permission_and_can_only_update_its_task_plan(client):
    criterion = "Show a permission-checked tool execution trace."
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Execute a tool-backed plan",
            "user_goal": "Learn how a Harness controls write tools.",
            "success_criteria": [criterion],
        },
    ).json()
    step = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Run a permission-checked write tool",
                    "agent_capability": "tool_calling",
                    "success_criterion": criterion,
                    "goal_connection": "The requested Harness demo needs a controlled tool call.",
                    "learning_outcome": "Observe permission denial and successful execution.",
                }
            ]
        },
    ).json()[0]
    arguments = {
        "plan_step_id": step["id"],
        "status": "completed",
        "result_summary": "Permission gate and trace verified.",
    }

    denied = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={"tool_name": "update_plan_step", "arguments": arguments},
    )
    policy = client.get(f"/agent/tasks/{task['id']}/tool-policy").json()
    policy_update = client.put(
        f"/agent/tasks/{task['id']}/tool-policy",
        json={"allowed_tools": [*policy["allowed_tools"], "update_plan_step"]},
    )
    succeeded = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": arguments,
            "idempotency_key": "complete-plan-step-once",
            "plan_step_id": step["id"],
        },
    )
    replayed = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": arguments,
            "idempotency_key": "complete-plan-step-once",
            "plan_step_id": step["id"],
        },
    )
    conflict = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": {**arguments, "result_summary": "Changed after the first call."},
            "idempotency_key": "complete-plan-step-once",
            "plan_step_id": step["id"],
        },
    )

    assert denied.json()["status"] == "denied"
    assert "not allowed by TaskPolicy" in denied.json()["error"]
    assert denied.json()["authorization_decision"] == "denied"
    assert policy_update.json()["version"] == policy["version"] + 1
    assert succeeded.json()["status"] == "succeeded"
    assert succeeded.json()["authorization_decision"] == "allowed"
    assert succeeded.json()["output"]["plan_step"]["status"] == "completed"
    assert replayed.status_code == 200
    assert replayed.json()["replayed"] is True
    assert replayed.json()["tool_call_id"] == succeeded.json()["tool_call_id"]
    assert replayed.json()["trace_id"] == succeeded.json()["trace_id"]
    assert replayed.json()["attempt_count"] == 1
    assert replayed.json()["replay_count"] == 1
    assert conflict.status_code == 409
    assert "different tool arguments" in conflict.json()["detail"]
    assert client.get(f"/agent/tasks/{task['id']}/plan").json()[0]["status"] == "completed"

    records = client.get(f"/agent/tasks/{task['id']}/tool-calls").json()
    assert len(records) == 2
    tool_traces = [
        trace
        for trace in client.get(f"/agent/tasks/{task['id']}/traces").json()
        if trace["event_type"] == "tool_call"
    ]
    assert len(tool_traces) == 3
    assert tool_traces[-1]["status"] == "replayed"


def test_demo_agent_loop_calls_tools_and_finishes_with_a_persisted_answer(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Inspect Agent job evidence",
            "user_goal": "Use real JD evidence to choose the next Agent learning task.",
            "constraints": ["Use only allowed read tools."],
            "success_criteria": ["Return an evidence-based next action."],
        },
    ).json()

    response = client.post(
        f"/agent/tasks/{task['id']}/agent-runs",
        json={"provider": "demo", "max_steps": 4},
    )

    assert response.status_code == 200
    run = response.json()
    assert run["provider"] == "demo"
    assert run["status"] == "completed", run["error"]
    assert run["stop_reason"] == "final_answer"
    assert run["step_count"] == 3
    assert [step["action_type"] for step in run["steps"]] == [
        "tool_call",
        "tool_call",
        "final_answer",
    ]
    assert run["steps"][0]["observation"]["status"] == "succeeded"
    assert run["steps"][1]["observation"]["status"] == "succeeded"
    assert run["steps"][2]["observation"] is None
    assert "Reviewed" in run["final_answer"]

    listed = client.get(f"/agent/tasks/{task['id']}/agent-runs").json()
    assert listed[0]["id"] == run["id"]
    assert len(client.get(f"/agent/tasks/{task['id']}/tool-calls").json()) == 2
    traces = client.get(f"/agent/tasks/{task['id']}/traces").json()
    assert len([trace for trace in traces if trace["event_type"] == "model_call"]) == 3
    checkpoints = client.get(
        f"/agent/tasks/{task['id']}/agent-runs/{run['id']}/checkpoints"
    ).json()
    assert checkpoints[0]["state"]["node_history"] == [
        "load_context",
        "run_agent",
        "evaluate_result",
        "complete_workflow",
    ]
    assert "model_step" not in {
        node
        for checkpoint in checkpoints
        for node in checkpoint["state"].get("node_history", [])
    }


def test_agent_loop_stops_at_the_server_side_step_limit(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Bound an Agent run",
            "user_goal": "Inspect jobs without running forever.",
            "constraints": [],
            "success_criteria": ["Stop at the configured limit."],
        },
    ).json()

    response = client.post(
        f"/agent/tasks/{task['id']}/agent-runs",
        json={"provider": "demo", "max_steps": 1},
    )

    assert response.status_code == 200
    run = response.json()
    assert run["status"] == "max_steps", run["error"]
    assert run["stop_reason"] == "max_steps_reached"
    assert run["step_count"] == 1
    assert len(run["steps"]) == 1
    assert run["steps"][0]["tool_call_id"] is not None
    checkpoints = client.get(
        f"/agent/tasks/{task['id']}/agent-runs/{run['id']}/checkpoints"
    ).json()
    assert checkpoints[0]["state"]["workflow_status"] == "blocked"


def test_agent_run_recovery_executes_a_saved_but_not_started_tool_step(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Recover before ToolCall creation",
            "user_goal": "Continue a saved Agent decision safely.",
            "success_criteria": ["Complete without losing the saved decision."],
        },
    ).json()
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        run = AgentRun(
            task_id=task["id"],
            provider="demo",
            model="careerops-demo-policy-v1",
            status="running",
            max_steps=4,
            step_count=1,
            started_at=old,
        )
        session.add(run)
        session.flush()
        session.add(
            AgentRunStep(
                run_id=run.id,
                task_id=task["id"],
                sequence=1,
                action_type="tool_call",
                model_request={"step_number": 1},
                model_response={
                    "action_type": "tool_call",
                    "tool_name": "search_jobs",
                    "arguments": {"query": "Agent", "limit": 3},
                    "call_id": "saved-before-tool",
                    "provider_metadata": {"deterministic": True},
                },
                created_at=old,
            )
        )
        session.commit()
        run_id = run.id

    report = client.post(
        f"/agent/tasks/{task['id']}/recover-agent-runs?stale_after_seconds=60"
    ).json()

    assert report["recovered_count"] == 1
    assert report["decisions"][0]["action"] == "executed_pending_tool"
    recovered = client.get(f"/agent/tasks/{task['id']}/agent-runs").json()[0]
    assert recovered["id"] == run_id
    assert recovered["status"] == "completed"
    assert recovered["step_count"] == 3
    assert recovered["steps"][0]["tool_call_id"] is not None


def test_agent_run_recovery_restores_a_terminal_tool_result_before_continuing(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Recover after a committed tool result",
            "user_goal": "Reuse the durable ToolCall result.",
            "success_criteria": ["Do not execute the completed call again."],
        },
    ).json()
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        run = AgentRun(
            task_id=task["id"],
            provider="demo",
            model="careerops-demo-policy-v1",
            status="running",
            max_steps=4,
            step_count=1,
            started_at=old,
        )
        session.add(run)
        session.flush()
        tool_call = ToolCallRecord(
            task_id=task["id"],
            tool_name="search_jobs",
            permission="read",
            effect="read",
            idempotency_mode="none",
            repeat_policy="always_allow",
            idempotency_key="finished-before-observation",
            request_fingerprint=request_fingerprint(
                "search_jobs", {"query": "Agent", "limit": 3}
            ),
            arguments_json={"query": "Agent", "limit": 3},
            status="succeeded",
            attempt_count=1,
            output_json={"count": 0, "jobs": []},
            started_at=old,
            completed_at=old,
            created_at=old,
            updated_at=old,
        )
        session.add(tool_call)
        session.flush()
        session.add(
            AgentRunStep(
                run_id=run.id,
                task_id=task["id"],
                sequence=1,
                action_type="tool_call",
                model_request={"step_number": 1},
                model_response={
                    "action_type": "tool_call",
                    "tool_name": "search_jobs",
                    "arguments": {"query": "Agent", "limit": 3},
                    "call_id": "finished-before-observation",
                },
                tool_call_id=tool_call.id,
                created_at=old,
            )
        )
        session.commit()
        run_id = run.id
        tool_call_id = tool_call.id

    report = client.post(
        f"/agent/tasks/{task['id']}/recover-agent-runs?stale_after_seconds=60"
    ).json()

    assert report["decisions"][0]["action"] == "restored_observation"
    recovered = client.get(f"/agent/tasks/{task['id']}/agent-runs").json()[0]
    assert recovered["id"] == run_id
    assert recovered["status"] == "completed"
    assert recovered["steps"][0]["observation"]["tool_call_id"] == tool_call_id
    assert recovered["steps"][0]["observation"]["attempt_count"] == 1


def test_agent_run_recovery_escalates_an_ambiguous_external_write(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Do not repeat an external write",
            "user_goal": "Escalate an ambiguous external operation.",
            "success_criteria": ["Require reconciliation instead of retrying."],
        },
    ).json()
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        run = AgentRun(
            task_id=task["id"],
            provider="demo",
            model="careerops-demo-policy-v1",
            status="running",
            max_steps=4,
            step_count=1,
            started_at=old,
        )
        session.add(run)
        session.flush()
        tool_call = ToolCallRecord(
            task_id=task["id"],
            tool_name="send_application",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            idempotency_key="ambiguous-application",
            request_fingerprint=request_fingerprint("send_application", {}),
            arguments_json={},
            status="executing",
            attempt_count=1,
            started_at=old,
            created_at=old,
            updated_at=old,
        )
        session.add(tool_call)
        session.flush()
        session.add(
            AgentRunStep(
                run_id=run.id,
                task_id=task["id"],
                sequence=1,
                action_type="tool_call",
                model_request={"step_number": 1},
                model_response={
                    "action_type": "tool_call",
                    "tool_name": "send_application",
                    "arguments": {},
                    "call_id": "ambiguous-application",
                },
                tool_call_id=tool_call.id,
                created_at=old,
            )
        )
        session.commit()

    report = client.post(
        f"/agent/tasks/{task['id']}/recover-agent-runs?stale_after_seconds=60"
    ).json()

    assert report["recovered_count"] == 0
    assert report["decisions"][0]["action"] == "needs_review"
    recovered = client.get(f"/agent/tasks/{task['id']}/agent-runs").json()[0]
    assert recovered["status"] == "needs_review"
    assert recovered["stop_reason"] == "recovery_required"
    assert "Reconcile" in recovered["error"]


def test_debug_cli_inspects_snapshots_and_replays_without_changing_source(client, tmp_path):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Reproduce a tool call safely",
            "user_goal": "Debug a saved write call without changing production data.",
            "success_criteria": ["Replay against an isolated database copy."],
        },
    ).json()
    step = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Create a reproducible write",
                    "agent_capability": "tool_calling",
                    "success_criterion": "Replay against an isolated database copy.",
                    "goal_connection": "The debugging workflow needs a saved ToolCall.",
                    "learning_outcome": "Compare recorded and replayed observations.",
                }
            ]
        },
    ).json()[0]
    arguments = {
        "plan_step_id": step["id"],
        "status": "completed",
        "result_summary": "Original tool output.",
    }
    policy = client.get(f"/agent/tasks/{task['id']}/tool-policy").json()
    client.put(
        f"/agent/tasks/{task['id']}/tool-policy",
        json={"allowed_tools": [*policy["allowed_tools"], "update_plan_step"]},
    )
    call = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "update_plan_step",
            "arguments": arguments,
            "idempotency_key": "debug-cli-write-once",
            "plan_step_id": step["id"],
        },
    ).json()
    with Session(engine) as session:
        source_step = session.get(PlanStep, step["id"])
        source_step.status = "blocked"
        source_step.result_summary = "Changed after the recorded call."
        session.commit()

    inspected = show_call("sqlite:///./test_careerops.db", call["tool_call_id"])
    snapshot = create_snapshot(
        "sqlite:///./test_careerops.db",
        call["tool_call_id"],
        tmp_path / "tool-call.db",
    )
    replayed = replay_call("sqlite:///./test_careerops.db", call["tool_call_id"])

    assert inspected["tool_call"]["arguments"] == arguments
    assert len(inspected["traces"]) == 1
    assert (tmp_path / "tool-call.db").is_file()
    assert (tmp_path / "tool-call.json").is_file()
    assert len(snapshot["database_sha256"]) == 64
    assert replayed.database_isolated is True
    assert replayed.replay_status == "success"
    assert replayed.output_matches is True
    with Session(engine) as session:
        source_step = session.get(PlanStep, step["id"])
        assert source_step.status == "blocked"
        assert source_step.result_summary == "Changed after the recorded call."


def test_scenario_runner_uses_isolated_database_and_checkpoint_process():
    result = run_isolated_scenario("graph-no-skills")

    assert result["status"] == "passed"
    assert result["database_isolated"] is True
    assert result["checkpoint_isolated"] is True
    assert result["result"]["final_status"] == "blocked"
    assert result["result"]["node_history"][-2:] == ["validate_plan", "block_run"]


def test_recovery_retries_stale_internal_write_with_saved_arguments(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Recover an interrupted internal write",
            "user_goal": "Resume a safe CareerOps database operation.",
            "success_criteria": ["The saved plan update completes once."],
        },
    ).json()
    step = client.post(
        f"/agent/tasks/{task['id']}/plan",
        json={
            "steps": [
                {
                    "title": "Complete after recovery",
                    "agent_capability": "tool_calling",
                    "success_criterion": "The saved plan update completes once.",
                    "goal_connection": "Recovery must finish the interrupted operation.",
                    "learning_outcome": "Observe a safe internal-write retry.",
                }
            ]
        },
    ).json()[0]
    arguments = {
        "plan_step_id": step["id"],
        "status": "completed",
        "result_summary": "Recovered from the saved request.",
    }
    policy = client.get(f"/agent/tasks/{task['id']}/tool-policy").json()
    client.put(
        f"/agent/tasks/{task['id']}/tool-policy",
        json={"allowed_tools": [*policy["allowed_tools"], "update_plan_step"]},
    )
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        session.add(
            ToolCallRecord(
                task_id=task["id"],
                plan_step_id=step["id"],
                tool_name="update_plan_step",
                permission="write",
                effect="internal_write",
                idempotency_mode="operation_key",
                repeat_policy="naturally_idempotent",
                idempotency_key="recover-plan-step-once",
                request_fingerprint=request_fingerprint(
                    "update_plan_step", arguments, plan_step_id=step["id"]
                ),
                arguments_json=arguments,
                status="executing",
                attempt_count=1,
                started_at=old,
                updated_at=old,
            )
        )
        session.commit()

    response = client.post(
        f"/agent/tasks/{task['id']}/recover-tool-calls?stale_after_seconds=60"
    )

    assert response.status_code == 200
    report = response.json()
    assert report["scanned_count"] == 1
    assert report["recovered_count"] == 1
    assert report["decisions"][0]["action"] == "retried"
    record = client.get(f"/agent/tasks/{task['id']}/tool-calls").json()[0]
    assert record["status"] == "succeeded"
    assert record["attempt_count"] == 2
    assert record["output"]["plan_step"]["status"] == "completed"
    assert client.get(f"/agent/tasks/{task['id']}/plan").json()[0]["status"] == "completed"


def test_recovery_does_not_blindly_retry_stale_external_write(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Recover an interrupted external write",
            "user_goal": "Avoid creating the same external issue twice.",
            "success_criteria": ["Unknown outcomes are reviewed before retry."],
        },
    ).json()
    arguments = {"repo": "careerops", "title": "Recovery design"}
    old = datetime.utcnow() - timedelta(minutes=5)
    with Session(engine) as session:
        session.add(
            ToolCallRecord(
                task_id=task["id"],
                tool_name="github_create_issue",
                permission="write",
                effect="external_write",
                idempotency_mode="operation_key",
                repeat_policy="business_unique",
                idempotency_key="create-recovery-issue-once",
                request_fingerprint=request_fingerprint(
                    "github_create_issue", arguments
                ),
                arguments_json=arguments,
                status="executing",
                attempt_count=1,
                started_at=old,
                updated_at=old,
            )
        )
        session.commit()

    first = client.post(
        f"/agent/tasks/{task['id']}/recover-tool-calls?stale_after_seconds=60"
    )
    assert first.status_code == 200
    assert first.json()["recovered_count"] == 0
    assert first.json()["decisions"][0]["action"] == "marked_outcome_unknown"
    assert client.get(f"/agent/tasks/{task['id']}/tool-calls").json()[0]["status"] == "outcome_unknown"

    second = client.post(
        f"/agent/tasks/{task['id']}/recover-tool-calls?stale_after_seconds=0"
    )
    assert second.json()["decisions"][0]["action"] == "needs_review"
    record = client.get(f"/agent/tasks/{task['id']}/tool-calls").json()[0]
    assert record["status"] == "needs_review"
    assert record["attempt_count"] == 1


def test_learning_graph_runs_nodes_and_exposes_checkpoints(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/graph-agent-one",
            "raw_content": "Company: Example AI\nTitle: Agent One\nBuild with Agent Harness and Tool Calling.",
        },
    )
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/graph-agent-two",
            "raw_content": "Company: Example AI\nTitle: Agent Two\nBuild with Agent Harness and RAG.",
        },
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Generate a JD-driven learning plan",
            "user_goal": "Use real JD demand to choose the next three Agent skills.",
            "success_criteria": ["Return a validated learning plan based on parsed JD skills."],
        },
    ).json()

    run = client.post(f"/agent/tasks/{task['id']}/run-learning-graph")

    assert run.status_code == 200
    body = run.json()
    assert body["state"]["status"] == "plan_validated"
    assert body["awaiting_approval"] is True
    assert body["interrupt"]["question"] == "Do you approve this JD-driven learning plan?"
    assert body["next_nodes"] == ["human_review"]
    assert body["state"]["node_history"] == [
        "load_skill_demand",
        "build_plan",
        "validate_plan",
    ]
    assert body["state"]["top_skills"][0]["name"] == "Agent Harness"
    assert len(body["state"]["learning_plan"]) == 3
    assert body["checkpoint_count"] >= 5

    checkpoints = client.get(f"/agent/tasks/{task['id']}/graph-checkpoints")
    assert checkpoints.status_code == 200
    snapshots = checkpoints.json()
    assert snapshots[0]["state"]["status"] == "plan_validated"
    assert snapshots[0]["next_nodes"] == ["human_review"]

    resumed = client.post(
        f"/agent/tasks/{task['id']}/resume-learning-graph",
        json={"approved": True, "comment": "The priorities match the JD evidence."},
    )
    assert resumed.status_code == 200
    resumed_body = resumed.json()
    assert resumed_body["awaiting_approval"] is False
    assert resumed_body["state"]["status"] == "completed"
    assert resumed_body["state"]["approved"] is True
    assert resumed_body["state"]["approval_comment"] == "The priorities match the JD evidence."
    assert resumed_body["state"]["node_history"] == [
        "load_skill_demand",
        "build_plan",
        "validate_plan",
        "human_review",
        "complete_run",
    ]
    assert client.get(f"/agent/tasks/{task['id']}").json()["status"] == "completed"

    second_run = client.post(f"/agent/tasks/{task['id']}/run-learning-graph")
    assert second_run.status_code == 200
    assert second_run.json()["checkpoint_count"] == resumed_body["checkpoint_count"]


def test_learning_graph_rejection_takes_blocked_edge(client):
    client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/rejected-plan-agent",
            "raw_content": "Company: Example AI\nTitle: Agent Engineer\nBuild with Agent Harness.",
        },
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Review a generated plan",
            "user_goal": "Generate and review a JD-driven plan.",
            "success_criteria": ["The user explicitly approves the plan."],
        },
    ).json()
    started = client.post(f"/agent/tasks/{task['id']}/run-learning-graph")
    assert started.json()["awaiting_approval"] is True

    rejected = client.post(
        f"/agent/tasks/{task['id']}/resume-learning-graph",
        json={"approved": False, "comment": "The plan is too broad."},
    )

    assert rejected.status_code == 200
    body = rejected.json()
    assert body["state"]["status"] == "blocked"
    assert body["state"]["approved"] is False
    assert body["state"]["node_history"][-2:] == ["human_review", "block_run"]
    assert client.get(f"/agent/tasks/{task['id']}").json()["status"] == "blocked"


def test_learning_graph_cannot_resume_without_interrupt(client):
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Unstarted graph",
            "user_goal": "Do not resume before the graph starts.",
            "success_criteria": ["The runtime rejects an invalid resume."],
        },
    ).json()

    response = client.post(
        f"/agent/tasks/{task['id']}/resume-learning-graph",
        json={"approved": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "learning graph is not waiting for approval"
