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
from app.services.memory_similarity import EmbeddingBatch
from app.services.memory_versioning import record_initial_memory_version
from app.services.context_compaction import compact_agent_context
from app.services.free_text_candidate_builder import extract_free_text_candidate_proposals
from app.services.hybrid_retrieval import (
    SEMANTIC_THRESHOLD,
    RetrievalDocument,
    rank_hybrid_documents,
)
from app.services.retrieval_router import choose_retrieval_route


@dataclass(frozen=True)
class CandidateCaseResult:
    case_id: str
    expected_decision: str
    actual_decision: str


@dataclass(frozen=True)
class RetrievalCaseResult:
    case_id: str
    expected_key: str
    lexical_top_key: str | None
    semantic_top_key: str | None
    hybrid_top_key: str | None
    routed_top_key: str | None
    route_strategy: str
    route_reason: str
    routed_embedding_calls: int


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
    lexical_semantic_retrieval_recall: float
    hybrid_semantic_retrieval_recall: float
    retrieval_case_count: int
    lexical_retrieval_recall_at_1: float
    semantic_retrieval_recall_at_1: float
    hybrid_retrieval_recall_at_1: float
    routed_retrieval_recall_at_1: float
    routed_retrieval_embedding_calls: int
    hybrid_retrieval_regression_count: int
    retrieval_scope_leakage_count: int
    pre_compaction_critical_constraint_retention: float
    post_compaction_critical_constraint_retention: float
    candidate_cases: tuple[CandidateCaseResult, ...]
    retrieval_cases: tuple[RetrievalCaseResult, ...]

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_cases"] = [asdict(case) for case in self.candidate_cases]
        payload["retrieval_cases"] = [asdict(case) for case in self.retrieval_cases]
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
        semantic_metrics = _semantic_retrieval_metrics(session, contract)
        retrieval_cases = _retrieval_case_results()

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
        lexical_semantic_retrieval_recall=semantic_metrics["lexical_recall"],
        hybrid_semantic_retrieval_recall=semantic_metrics["hybrid_recall"],
        retrieval_case_count=len(retrieval_cases),
        lexical_retrieval_recall_at_1=_retrieval_recall_at_1(
            retrieval_cases, "lexical_top_key"
        ),
        semantic_retrieval_recall_at_1=_retrieval_recall_at_1(
            retrieval_cases, "semantic_top_key"
        ),
        hybrid_retrieval_recall_at_1=_retrieval_recall_at_1(
            retrieval_cases, "hybrid_top_key"
        ),
        routed_retrieval_recall_at_1=_retrieval_recall_at_1(
            retrieval_cases, "routed_top_key"
        ),
        routed_retrieval_embedding_calls=sum(
            case.routed_embedding_calls for case in retrieval_cases
        ),
        hybrid_retrieval_regression_count=sum(
            case.hybrid_top_key != case.expected_key
            and (
                case.lexical_top_key == case.expected_key
                or case.semantic_top_key == case.expected_key
            )
            for case in retrieval_cases
        ),
        retrieval_scope_leakage_count=semantic_metrics["scope_leakage_count"],
        pre_compaction_critical_constraint_retention=float(critical_key in retrieved_keys),
        post_compaction_critical_constraint_retention=float(
            protected_constraint_retained
        ),
        candidate_cases=results,
        retrieval_cases=retrieval_cases,
    )


class _BenchmarkEmbeddingProvider:
    provider_version = "benchmark-semantic-v1"

    def __init__(self):
        self.call_count = 0

    def embed(self, texts):
        self.call_count += 1
        vectors = []
        for text in texts:
            lowered = text.lower()
            recovery_related = any(
                phrase in lowered
                for phrase in (
                    "asynchronous",
                    "process interruption",
                    "worker crash",
                    "durable lease",
                    "进程崩溃",
                    "恢复后台任务",
                )
            )
            tool_operation_related = any(
                phrase in lowered
                for phrase in (
                    "toolcall 1842",
                    "external operation outcome unknown",
                )
            )
            if recovery_related:
                vectors.append([1.0, 0.0, 0.0])
            elif tool_operation_related:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return EmbeddingBatch(
            vectors=vectors,
            provider_version=self.provider_version,
            usage={"embedding_calls": 1, "embedding_tokens": len(texts) * 5},
        )


def _retrieval_case_results() -> tuple[RetrievalCaseResult, ...]:
    cases = (
        (
            "exact-identifier",
            "ToolCall 1842 timeout",
            "z-toolcall-1842",
            (
                RetrievalDocument(
                    key="z-toolcall-1842",
                    source_type="trace",
                    content="ToolCall 1842 timeout",
                    payload={},
                ),
                RetrievalDocument(
                    key="a-operation-unknown",
                    source_type="trace",
                    content="External operation outcome unknown",
                    payload={},
                ),
            ),
        ),
        (
            "semantic-paraphrase",
            "Resume asynchronous work after process interruption",
            "worker-recovery",
            (
                RetrievalDocument(
                    key="worker-recovery",
                    source_type="memory",
                    content="Worker crash recovery uses a durable lease",
                    payload={},
                ),
                RetrievalDocument(
                    key="visual-theme",
                    source_type="memory",
                    content="The dashboard uses a blue visual palette",
                    payload={},
                ),
            ),
        ),
        (
            "cross-language",
            "进程崩溃后如何恢复后台任务",
            "worker-recovery-english",
            (
                RetrievalDocument(
                    key="worker-recovery-english",
                    source_type="memory",
                    content="Worker crash recovery uses a durable lease",
                    payload={},
                ),
                RetrievalDocument(
                    key="frontend-layout",
                    source_type="memory",
                    content="Build responsive frontend layouts",
                    payload={},
                ),
            ),
        ),
    )
    provider = _BenchmarkEmbeddingProvider()
    results = []
    for case_id, query, expected_key, documents in cases:
        lexical = rank_hybrid_documents(query, documents, limit=len(documents))
        hybrid = rank_hybrid_documents(
            query,
            documents,
            embedding_provider=provider,
            limit=len(documents),
        )
        route = choose_retrieval_route(query, embedding_available=True)
        routed_provider = _BenchmarkEmbeddingProvider()
        routed = rank_hybrid_documents(
            query,
            documents,
            embedding_provider=(
                routed_provider if route.strategy == "hybrid" else None
            ),
            limit=len(documents),
        )
        semantic_candidates = sorted(
            (
                candidate
                for candidate in hybrid.candidates
                if candidate.semantic_score is not None
                and candidate.semantic_score >= SEMANTIC_THRESHOLD
            ),
            key=lambda candidate: (
                -candidate.semantic_score,
                candidate.document.key,
            ),
        )
        results.append(
            RetrievalCaseResult(
                case_id=case_id,
                expected_key=expected_key,
                lexical_top_key=(
                    lexical.candidates[0].document.key if lexical.candidates else None
                ),
                semantic_top_key=(
                    semantic_candidates[0].document.key
                    if semantic_candidates
                    else None
                ),
                hybrid_top_key=(
                    hybrid.candidates[0].document.key if hybrid.candidates else None
                ),
                routed_top_key=(
                    routed.candidates[0].document.key if routed.candidates else None
                ),
                route_strategy=route.strategy,
                route_reason=route.reason,
                routed_embedding_calls=routed_provider.call_count,
            )
        )
    return tuple(results)


def _retrieval_recall_at_1(
    cases: tuple[RetrievalCaseResult, ...], field: str
) -> float:
    return _ratio(
        sum(getattr(case, field) == case.expected_key for case in cases),
        len(cases),
    )


def _semantic_retrieval_metrics(
    session: Session, contract: GoalContract
) -> dict[str, int | float]:
    task = AgentTask(
        goal_contract_id=contract.id,
        title="Recover asynchronous work",
        user_goal="Recover work after process interruption.",
        constraints=["Do not duplicate execution."],
        success_criteria=["Retrieve resilient execution evidence."],
        status="in_progress",
    )
    other_contract = GoalContract(
        version=2,
        north_star_goal="Keep another user's facts isolated.",
        product_context="Scope-leakage benchmark.",
        learning_contract="Measure isolation.",
        scope_guardrails=["Do not cross contracts."],
        success_criteria=["Leakage remains zero."],
        is_active=False,
    )
    session.add_all([task, other_contract])
    session.flush()
    session.add_all(
        [
            AgentMemory(
                goal_contract_id=contract.id,
                memory_type="fact",
                memory_key="semantic-worker-recovery",
                content="Worker crash recovery uses a durable lease.",
                source="benchmark",
                importance=2,
            ),
            AgentMemory(
                goal_contract_id=contract.id,
                memory_type="fact",
                memory_key="semantic-distractor",
                content="The dashboard uses a blue visual palette.",
                source="benchmark",
                importance=5,
            ),
            AgentMemory(
                goal_contract_id=other_contract.id,
                memory_type="fact",
                memory_key="cross-contract-worker-secret",
                content="Worker crash recovery uses a private durable lease token.",
                source="benchmark",
                importance=5,
            ),
        ]
    )
    session.flush()
    lexical = assemble_memory_context(
        session,
        task,
        memory_limit=1,
        knowledge_limit=0,
        memory_token_budget=256,
    )
    hybrid = assemble_memory_context(
        session,
        task,
        memory_limit=1,
        knowledge_limit=0,
        memory_token_budget=256,
        embedding_provider=_BenchmarkEmbeddingProvider(),
    )
    expected = {"semantic-worker-recovery"}
    lexical_keys = {item["memory_key"] for item in lexical.memories}
    hybrid_keys = {item["memory_key"] for item in hybrid.memories}
    return {
        "lexical_recall": _ratio(len(expected & lexical_keys), len(expected)),
        "hybrid_recall": _ratio(len(expected & hybrid_keys), len(expected)),
        "scope_leakage_count": int("cross-contract-worker-secret" in hybrid_keys),
    }


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
