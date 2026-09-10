from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.models import AgentMemory, AgentRun, AgentTask, ExecutionTrace, GoalContract
from app.services.memory_evaluator import MemoryCandidate, evaluate_and_store_candidate
from app.services.memory_runtime import assemble_memory_context
from app.services.memory_versioning import record_initial_memory_version
from app.services.context_compaction import compact_agent_context
from app.services.free_text_candidate_builder import extract_free_text_candidate_proposals


@dataclass(frozen=True)
class CandidateCaseResult:
    case_id: str
    expected_decision: str
    actual_decision: str


@dataclass(frozen=True)
class MemoryBenchmarkReport:
    candidate_case_count: int
    candidate_precision: float
    candidate_recall: float
    decision_accuracy: float
    extraction_case_count: int
    extraction_precision: float
    extraction_recall: float
    extraction_false_positive_rate: float
    retrieval_recall: float
    pre_compaction_critical_constraint_retention: float
    post_compaction_critical_constraint_retention: float
    candidate_cases: tuple[CandidateCaseResult, ...]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_cases"] = [asdict(case) for case in self.candidate_cases]
        return payload


def run_memory_benchmark() -> MemoryBenchmarkReport:
    """Run deterministic labeled Memory Gate and retrieval cases in isolation."""
    benchmark_now = datetime(2026, 1, 1, 12, 0, 0)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        contract = GoalContract(
            version=1,
            north_star_goal="Build a trustworthy Agent Harness.",
            product_context="CareerOps uses public JD intelligence as its scenario.",
            learning_contract="Explain new Agent engineering tradeoffs concisely.",
            scope_guardrails=["Do not send project data to external models."],
            success_criteria=["Memory decisions are measurable and auditable."],
            is_active=True,
        )
        session.add(contract)
        session.flush()
        task = AgentTask(
            goal_contract_id=contract.id,
            title="Learn LangGraph memory retrieval",
            user_goal="Use Agent memory safely when prioritizing LangGraph learning.",
            constraints=["Never expose project goals without explicit authorization."],
            success_criteria=["Retrieve current LangGraph evidence."],
            status="in_progress",
        )
        session.add(task)
        session.flush()
        run = AgentRun(
            task_id=task.id,
            provider="benchmark",
            model="deterministic",
            status="completed",
            max_steps=1,
            step_count=1,
        )
        session.add(run)
        user_trace = ExecutionTrace(
            task_id=task.id,
            event_type="user_input",
            status="recorded",
            input_summary="Please explain Memory Runtime with a concrete example.",
            output_summary="Input captured for benchmark provenance.",
        )
        session.add(user_trace)
        session.flush()

        runtime_provenance = {
            "agent_run_id": run.id,
            "task_id": task.id,
            "provider": run.provider,
            "model": run.model,
        }
        labeled_candidates = (
            (
                "verified_runtime_outcome",
                "accept",
                _candidate(
                    task,
                    key="benchmark-runtime-outcome",
                    content="The Memory Runtime benchmark completed with auditable results.",
                    source="agent_runtime",
                    provenance=runtime_provenance,
                ),
            ),
            (
                "verified_runtime_learning",
                "accept",
                _candidate(
                    task,
                    key="benchmark-runtime-learning",
                    content="The benchmark distinguishes persistent Memory from assembled Context.",
                    source="agent_runtime",
                    provenance=runtime_provenance,
                ),
            ),
            (
                "sensitive_value",
                "reject",
                _candidate(
                    task,
                    key="benchmark-secret",
                    content="The API key=sk-benchmarkcredential12345 must remain private.",
                    source="agent_runtime",
                    provenance=runtime_provenance,
                ),
            ),
            (
                "insufficient_information",
                "reject",
                _candidate(
                    task,
                    key="benchmark-short",
                    content="Too short.",
                    source="agent_runtime",
                    provenance=runtime_provenance,
                ),
            ),
            (
                "forged_runtime_provenance",
                "reject",
                _candidate(
                    task,
                    key="benchmark-forged",
                    content="This candidate claims a runtime source that does not exist.",
                    source="agent_runtime",
                    provenance={**runtime_provenance, "agent_run_id": 999999},
                ),
            ),
            (
                "untrusted_model_source",
                "needs_review",
                _candidate(
                    task,
                    key="benchmark-model-proposal",
                    content="The model proposes a durable preference without verified evidence.",
                    source="model_proposal",
                    provenance={"task_id": task.id},
                ),
            ),
            (
                "verified_user_input_requires_semantics",
                "needs_review",
                _candidate(
                    task,
                    key="benchmark-user-preference",
                    content="The user learns Memory Runtime best through concrete examples.",
                    source="user_input",
                    provenance={
                        "execution_trace_id": user_trace.id,
                        "task_id": task.id,
                        "evidence_text": "explain Memory Runtime with a concrete example",
                    },
                ),
            ),
        )
        results = tuple(
            CandidateCaseResult(
                case_id=case_id,
                expected_decision=expected,
                actual_decision=evaluate_and_store_candidate(
                    session, task, candidate
                ).decision,
            )
            for case_id, expected, candidate in labeled_candidates
        )

        critical_key = "external-model-authorization"
        relevant_keys = {critical_key, "current-langgraph-demand"}
        _add_memory(
            session,
            contract,
            memory_key=critical_key,
            content="Never send project goals to external models without explicit authorization.",
            importance=5,
        )
        _add_memory(
            session,
            contract,
            task=task,
            memory_key="current-langgraph-demand",
            content="LangGraph currently appears in five relevant job descriptions.",
            importance=4,
        )
        _add_memory(
            session,
            contract,
            task=task,
            memory_key="unrelated-docker-note",
            content="Docker image cleanup is scheduled for a later maintenance task.",
            importance=4,
        )
        _add_memory(
            session,
            contract,
            task=task,
            memory_key="expired-langgraph-note",
            content="An expired LangGraph observation should not be retrieved.",
            importance=5,
            expires_at=benchmark_now - timedelta(days=1),
        )
        session.flush()
        context = assemble_memory_context(
            session,
            task,
            memory_limit=2,
            memory_token_budget=256,
            now=benchmark_now,
        )
        retrieved_keys = {memory["memory_key"] for memory in context.memories}
        compacted = compact_agent_context(
            task,
            run_id=run.id,
            step_number=2,
            max_steps=run.max_steps,
            memory_context=context.as_dict(),
            observations=[
                {
                    "tool_call_id": 1,
                    "trace_id": 1,
                    "tool_name": "benchmark_tool",
                    "status": "succeeded",
                    "output": {"detail": "old observation " * 200},
                }
            ],
            observation_token_budget=1,
        )
        compacted_keys = {
            memory["memory_key"]
            for memory in compacted.compacted_context["memory_context"]["memories"]
        }
        protected_constraint_retained = (
            critical_key in compacted_keys
            and compacted.compacted_context["task"]["constraints"] == task.constraints
        )

    expected_positive = [case.expected_decision == "accept" for case in results]
    actual_positive = [case.actual_decision == "accept" for case in results]
    true_positive = sum(
        expected and actual
        for expected, actual in zip(expected_positive, actual_positive, strict=True)
    )
    predicted_positive = sum(actual_positive)
    positive_count = sum(expected_positive)
    extraction = _free_text_extraction_metrics()
    return MemoryBenchmarkReport(
        candidate_case_count=len(results),
        candidate_precision=_ratio(true_positive, predicted_positive),
        candidate_recall=_ratio(true_positive, positive_count),
        decision_accuracy=_ratio(
            sum(
                case.expected_decision == case.actual_decision
                for case in results
            ),
            len(results),
        ),
        extraction_case_count=extraction["case_count"],
        extraction_precision=extraction["precision"],
        extraction_recall=extraction["recall"],
        extraction_false_positive_rate=extraction["false_positive_rate"],
        retrieval_recall=_ratio(len(relevant_keys & retrieved_keys), len(relevant_keys)),
        pre_compaction_critical_constraint_retention=float(critical_key in retrieved_keys),
        post_compaction_critical_constraint_retention=float(
            protected_constraint_retained
        ),
        candidate_cases=results,
    )


def _free_text_extraction_metrics() -> dict[str, int | float]:
    labeled_cases = (
        ("I prefer explanations with concrete examples.", {"preference"}),
        ("My target role is Agent Engineer.", {"fact"}),
        ("Correction: my target role is Platform Engineer.", {"correction"}),
        ("I learned that checkpoint versions prevent unsafe resumes.", {"episode"}),
        ("我更喜欢先看具体例子。", {"preference"}),
        ("我的目标岗位是 Agent Engineer。", {"fact"}),
        ("更正：我的目标岗位是 Platform Engineer。", {"correction"}),
        ("我发现分离状态和记忆可以减少错误恢复。", {"episode"}),
        ("Please explain token budgeting.", set()),
        ("We prefer PostgreSQL in production.", set()),
        ("The target role is unclear.", set()),
    )
    true_positive = false_positive = false_negative = true_negative = 0
    for text, expected_categories in labeled_cases:
        actual_categories = {
            proposal.category
            for proposal in extract_free_text_candidate_proposals(text)
        }
        true_positive += len(expected_categories & actual_categories)
        false_positive += len(actual_categories - expected_categories)
        false_negative += len(expected_categories - actual_categories)
        true_negative += int(not expected_categories and not actual_categories)
    return {
        "case_count": len(labeled_cases),
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
        "false_positive_rate": _ratio(
            false_positive,
            false_positive + true_negative,
        ),
    }


def _candidate(
    task: AgentTask,
    *,
    key: str,
    content: str,
    source: str,
    provenance: dict,
) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type="episodic",
        scope_type="task",
        task_id=task.id,
        run_id=None,
        memory_key=key,
        content=content,
        source=source,
        importance=3,
        relevance_text=content,
        provenance=provenance,
    )


def _add_memory(
    session: Session,
    contract: GoalContract,
    *,
    memory_key: str,
    content: str,
    importance: int,
    task: AgentTask | None = None,
    expires_at: datetime | None = None,
) -> None:
    memory = AgentMemory(
        goal_contract_id=contract.id,
        scope_type="task" if task else "contract",
        task_id=task.id if task else None,
        memory_type="fact",
        memory_key=memory_key,
        content=content,
        source="benchmark_fixture",
        importance=importance,
        expires_at=expires_at,
    )
    session.add(memory)
    session.flush()
    record_initial_memory_version(session, memory)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def main() -> None:
    print(json.dumps(run_memory_benchmark().as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
