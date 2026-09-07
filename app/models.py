from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class RequirementType(StrEnum):
    REQUIRED = "required"
    PREFERRED = "preferred"


class EvidenceType(StrEnum):
    CODE = "code"
    DEMO = "demo"
    DOCUMENT = "document"
    EXPERIENCE = "experience"


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class SourceKind(StrEnum):
    COMPANY_CAREERS = "company_careers"
    PUBLIC_JOB_BOARD = "public_job_board"
    RSS = "rss"
    MANUAL = "manual"


class ArchiveStatus(StrEnum):
    ARCHIVED = "archived"
    DUPLICATE = "duplicate"
    PARSED = "parsed"


class MemoryType(StrEnum):
    WORKING = "working"
    EPISODIC = "episodic"
    FACT = "fact"


class MemoryScope(StrEnum):
    CONTRACT = "contract"
    TASK = "task"
    RUN = "run"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"


class AgentCapability(StrEnum):
    AGENT_LOOP = "agent_loop"
    TOOL_CALLING = "tool_calling"
    MEMORY = "memory"
    CONTEXT_MANAGEMENT = "context_management"
    RETRIEVAL = "retrieval"
    EVALUATION = "evaluation"
    USER_INTERACTION = "user_interaction"
    AGENT_FOUNDATION = "agent_foundation"


class AlignmentDecision(StrEnum):
    PROCEED = "proceed"
    DEFER = "defer"


class AgentTaskStatus(StrEnum):
    DRAFT = "draft"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TraceEventType(StrEnum):
    PLAN = "plan"
    TOOL_CALL = "tool_call"
    EVALUATION = "evaluation"
    CONTEXT_COMPACTION = "context_compaction"
    USER_INPUT = "user_input"
    MODEL_CALL = "model_call"
    AGENT_RUN = "agent_run"
    MEMORY = "memory"


class ToolCallStatus(StrEnum):
    PROPOSED = "proposed"
    AUTHORIZED = "authorized"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    OUTCOME_UNKNOWN = "outcome_unknown"
    NEEDS_REVIEW = "needs_review"


class AgentRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"


class JobSource(Base):
    __tablename__ = "job_sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(40), default=SourceKind.PUBLIC_JOB_BOARD)
    url: Mapped[str] = mapped_column(String(1000), unique=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    archives: Mapped[list[JobArchive]] = relationship(back_populates="source")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    company: Mapped[str] = mapped_column(String(200), index=True)
    title: Mapped[str] = mapped_column(String(200), index=True)
    location: Mapped[str] = mapped_column(String(100), default="Beijing")
    source_url: Mapped[str] = mapped_column(String(1000), unique=True)
    description: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    requirements: Mapped[list[JobRequirement]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    archives: Mapped[list[JobArchive]] = relationship(back_populates="job")


class JobArchive(Base):
    __tablename__ = "job_archives"
    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_job_archive_canonical_url"),
        UniqueConstraint("content_hash", name="uq_job_archive_content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("job_sources.id"), nullable=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    source_url: Mapped[str] = mapped_column(String(1000), index=True)
    canonical_url: Mapped[str] = mapped_column(String(1000))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    raw_content: Mapped[str] = mapped_column(Text)
    structured_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=ArchiveStatus.ARCHIVED)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    source: Mapped[JobSource | None] = relationship(back_populates="archives")
    job: Mapped[Job | None] = relationship(back_populates="archives")


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    category: Mapped[str] = mapped_column(String(120), index=True)
    aliases: Mapped[str | None] = mapped_column(Text, nullable=True)

    requirements: Mapped[list[JobRequirement]] = relationship(back_populates="skill")
    evidence: Mapped[list[ProfileEvidence]] = relationship(back_populates="skill")
    gap_tasks: Mapped[list[GapTask]] = relationship(back_populates="skill")


class JobRequirement(Base):
    __tablename__ = "job_requirements"
    __table_args__ = (UniqueConstraint("job_id", "skill_id", name="uq_job_skill_requirement"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"))
    importance: Mapped[int] = mapped_column(Integer, default=3)
    requirement_type: Mapped[str] = mapped_column(String(20), default=RequirementType.REQUIRED)
    evidence_text: Mapped[str] = mapped_column(Text)

    job: Mapped[Job] = relationship(back_populates="requirements")
    skill: Mapped[Skill] = relationship(back_populates="requirements")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str] = mapped_column(Text)
    repository_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    demo_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    evidence: Mapped[list[ProfileEvidence]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProfileEvidence(Base):
    __tablename__ = "profile_evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"))
    evidence_type: Mapped[str] = mapped_column(String(20), default=EvidenceType.CODE)
    strength: Mapped[int] = mapped_column(Integer, default=1)
    description: Mapped[str] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    project: Mapped[Project] = relationship(back_populates="evidence")
    skill: Mapped[Skill] = relationship(back_populates="evidence")


class GapTask(Base):
    __tablename__ = "gap_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id"))
    title: Mapped[str] = mapped_column(String(240))
    acceptance_criteria: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(20), default=TaskStatus.TODO)

    skill: Mapped[Skill] = relationship(back_populates="gap_tasks")


class GoalContract(Base):
    __tablename__ = "goal_contracts"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    north_star_goal: Mapped[str] = mapped_column(Text)
    product_context: Mapped[str] = mapped_column(Text)
    learning_contract: Mapped[str] = mapped_column(Text)
    scope_guardrails: Mapped[list[str]] = mapped_column(JSON)
    success_criteria: Mapped[list[str]] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    memories: Mapped[list[AgentMemory]] = relationship(
        back_populates="goal_contract", cascade="all, delete-orphan"
    )
    alignments: Mapped[list[AgentAlignment]] = relationship(
        back_populates="goal_contract", cascade="all, delete-orphan"
    )
    tasks: Mapped[list[AgentTask]] = relationship(
        back_populates="goal_contract", cascade="all, delete-orphan"
    )


class AgentMemory(Base):
    __tablename__ = "agent_memories"
    __table_args__ = (
        UniqueConstraint("goal_contract_id", "memory_type", "memory_key", name="uq_agent_memory_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_contract_id: Mapped[int] = mapped_column(ForeignKey("goal_contracts.id"), index=True)
    scope_type: Mapped[str] = mapped_column(String(20), default=MemoryScope.CONTRACT, index=True)
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_tasks.id"), nullable=True, index=True
    )
    run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    memory_type: Mapped[str] = mapped_column(String(30), index=True)
    memory_key: Mapped[str] = mapped_column(String(160))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(80), default="user")
    importance: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(20), default=MemoryStatus.ACTIVE, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retirement_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    goal_contract: Mapped[GoalContract] = relationship(back_populates="memories")


class MemoryCandidateRecord(Base):
    __tablename__ = "memory_candidates"
    __table_args__ = (
        UniqueConstraint(
            "goal_contract_id",
            "memory_type",
            "memory_key",
            "evaluator_version",
            name="uq_memory_candidate_evaluation",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_contract_id: Mapped[int] = mapped_column(ForeignKey("goal_contracts.id"), index=True)
    task_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_tasks.id"), nullable=True, index=True
    )
    source_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    scope_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_runs.id"), nullable=True
    )
    memory_id: Mapped[int | None] = mapped_column(
        ForeignKey("agent_memories.id"), nullable=True, index=True
    )
    memory_type: Mapped[str] = mapped_column(String(30), index=True)
    scope_type: Mapped[str] = mapped_column(String(20), index=True)
    memory_key: Mapped[str] = mapped_column(String(160))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(80), index=True)
    importance: Mapped[int] = mapped_column(Integer, default=3)
    provenance: Mapped[dict] = mapped_column(JSON)
    decision: Mapped[str] = mapped_column(String(30), index=True)
    storage_action: Mapped[str] = mapped_column(String(30))
    reasons: Mapped[list[str]] = mapped_column(JSON)
    evaluator_version: Mapped[str] = mapped_column(String(60))
    evaluator_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evaluator_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )


class AgentAlignment(Base):
    __tablename__ = "agent_alignments"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_contract_id: Mapped[int] = mapped_column(ForeignKey("goal_contracts.id"), index=True)
    proposed_action: Mapped[str] = mapped_column(Text)
    agent_capability: Mapped[str] = mapped_column(String(40), index=True)
    success_criterion: Mapped[str] = mapped_column(Text)
    goal_connection: Mapped[str] = mapped_column(Text)
    learning_outcome: Mapped[str] = mapped_column(Text)
    decision: Mapped[str] = mapped_column(String(20), default=AlignmentDecision.PROCEED)
    evaluation_notes: Mapped[list[str]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    goal_contract: Mapped[GoalContract] = relationship(back_populates="alignments")


class AgentTask(Base):
    __tablename__ = "agent_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_contract_id: Mapped[int] = mapped_column(ForeignKey("goal_contracts.id"), index=True)
    title: Mapped[str] = mapped_column(String(240), index=True)
    user_goal: Mapped[str] = mapped_column(Text)
    constraints: Mapped[list[str]] = mapped_column(JSON)
    success_criteria: Mapped[list[str]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(30), default=AgentTaskStatus.DRAFT, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    goal_contract: Mapped[GoalContract] = relationship(back_populates="tasks")
    plan_steps: Mapped[list[PlanStep]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="PlanStep.sequence"
    )
    traces: Mapped[list[ExecutionTrace]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ExecutionTrace.id"
    )
    tool_calls: Mapped[list[ToolCallRecord]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ToolCallRecord.id"
    )
    tool_policy: Mapped[TaskPolicy | None] = relationship(
        back_populates="task", cascade="all, delete-orphan", uselist=False
    )
    tool_authorizations: Mapped[list[ToolAuthorization]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ToolAuthorization.id"
    )
    agent_runs: Mapped[list[AgentRun]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="AgentRun.id"
    )


class PlanStep(Base):
    __tablename__ = "plan_steps"
    __table_args__ = (UniqueConstraint("task_id", "sequence", name="uq_plan_step_sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(240))
    agent_capability: Mapped[str] = mapped_column(String(40), index=True)
    success_criterion: Mapped[str] = mapped_column(Text)
    goal_connection: Mapped[str] = mapped_column(Text)
    learning_outcome: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default=PlanStepStatus.PENDING, index=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    task: Mapped[AgentTask] = relationship(back_populates="plan_steps")
    traces: Mapped[list[ExecutionTrace]] = relationship(back_populates="plan_step")


class ExecutionTrace(Base):
    __tablename__ = "execution_traces"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    plan_step_id: Mapped[int | None] = mapped_column(ForeignKey("plan_steps.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(30), default="recorded")
    input_summary: Mapped[str] = mapped_column(Text)
    output_summary: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    task: Mapped[AgentTask] = relationship(back_populates="traces")
    plan_step: Mapped[PlanStep | None] = relationship(back_populates="traces")


class ToolCallRecord(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (
        UniqueConstraint("task_id", "idempotency_key", name="uq_task_tool_idempotency_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    plan_step_id: Mapped[int | None] = mapped_column(ForeignKey("plan_steps.id"), nullable=True)
    trace_id: Mapped[int | None] = mapped_column(ForeignKey("execution_traces.id"), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    permission: Mapped[str] = mapped_column(String(20))
    effect: Mapped[str] = mapped_column(String(30))
    idempotency_mode: Mapped[str] = mapped_column(String(30))
    repeat_policy: Mapped[str] = mapped_column(String(40))
    idempotency_key: Mapped[str] = mapped_column(String(240))
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    arguments_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(
        String(30), default=ToolCallStatus.PROPOSED, index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    replay_count: Mapped[int] = mapped_column(Integer, default=0)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped[AgentTask] = relationship(back_populates="tool_calls")
    authorizations: Mapped[list[ToolAuthorization]] = relationship(
        back_populates="tool_call",
        cascade="all, delete-orphan",
        order_by="ToolAuthorization.id",
    )


class TaskPolicy(Base):
    __tablename__ = "task_policies"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    allowed_tools: Mapped[list[str]] = mapped_column(JSON)
    external_writes_require_approval: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    task: Mapped[AgentTask] = relationship(back_populates="tool_policy")


class ToolAuthorization(Base):
    __tablename__ = "tool_authorizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    tool_call_id: Mapped[int] = mapped_column(ForeignKey("tool_calls.id"), index=True)
    policy_id: Mapped[int] = mapped_column(ForeignKey("task_policies.id"), index=True)
    policy_version: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    permission: Mapped[str] = mapped_column(String(20))
    effect: Mapped[str] = mapped_column(String(30))
    decision: Mapped[str] = mapped_column(String(30), index=True)
    reason: Mapped[str] = mapped_column(Text)
    policy_snapshot: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    task: Mapped[AgentTask] = relationship(back_populates="tool_authorizations")
    tool_call: Mapped[ToolCallRecord] = relationship(back_populates="authorizations")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    provider: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(120))
    status: Mapped[str] = mapped_column(String(30), default=AgentRunStatus.RUNNING, index=True)
    max_steps: Mapped[int] = mapped_column(Integer)
    step_count: Mapped[int] = mapped_column(Integer, default=0)
    final_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    stop_reason: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    timing_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    task: Mapped[AgentTask] = relationship(back_populates="agent_runs")
    steps: Mapped[list[AgentRunStep]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="AgentRunStep.sequence"
    )


class AgentRunStep(Base):
    __tablename__ = "agent_run_steps"
    __table_args__ = (UniqueConstraint("run_id", "sequence", name="uq_agent_run_step_sequence"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("agent_tasks.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    action_type: Mapped[str] = mapped_column(String(30))
    model_request: Mapped[dict] = mapped_column(JSON)
    model_response: Mapped[dict] = mapped_column(JSON)
    tool_call_id: Mapped[int | None] = mapped_column(
        ForeignKey("tool_calls.id"), nullable=True, index=True
    )
    observation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timing_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="steps")
