from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RequirementRead(BaseModel):
    name: str
    category: str
    importance: int
    requirement_type: str
    evidence_text: str


class JobCreate(BaseModel):
    company: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    location: str = Field(default="Beijing", max_length=100)
    source_url: HttpUrl
    description: str = Field(min_length=1)
    published_at: datetime | None = None


class JobRead(JobCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class SkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    category: str = Field(min_length=1, max_length=120)
    aliases: str | None = None


class SkillRead(SkillCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int


class JobRequirementRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    skill: SkillRead
    importance: int
    requirement_type: str
    evidence_text: str


class SkillDemandRead(BaseModel):
    skill_id: int
    name: str
    category: str
    requirement_count: int
    job_count: int


class LearningRoadmapItemRead(BaseModel):
    rank: int
    skill_id: int
    skill: str
    category: str
    job_count: int
    requirement_count: int
    why: str
    learning_goal: str
    project_idea: str
    evidence: str


class LearningTaskRead(BaseModel):
    rank: int
    skill_id: int
    skill: str
    task_type: str
    title: str
    why: str
    acceptance_criteria: list[str]
    deliverable: str


class GoalContractCreate(BaseModel):
    north_star_goal: str = Field(min_length=1)
    product_context: str = Field(min_length=1)
    learning_contract: str = Field(min_length=1)
    scope_guardrails: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(min_length=1)


class GoalContractRead(GoalContractCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    version: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


class AgentMemoryCreate(BaseModel):
    memory_type: Literal["working", "episodic", "fact"]
    scope_type: Literal["contract", "task", "run"] = "contract"
    task_id: int | None = Field(default=None, ge=1)
    run_id: int | None = Field(default=None, ge=1)
    memory_key: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1)
    source: str = Field(default="user", min_length=1, max_length=80)
    importance: int = Field(default=3, ge=1, le=5)
    expires_at: datetime | None = None


class AgentMemoryRead(AgentMemoryCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    goal_contract_id: int
    status: Literal["active", "retired"]
    retired_at: datetime | None
    retirement_reason: str | None
    created_at: datetime
    updated_at: datetime


class AgentMemoryRetire(BaseModel):
    reason: str = Field(min_length=1, max_length=240)


class MemoryCandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    goal_contract_id: int
    task_id: int | None
    source_run_id: int | None
    scope_run_id: int | None
    memory_id: int | None
    memory_type: str
    scope_type: str
    memory_key: str
    content: str
    source: str
    importance: int
    provenance: dict
    decision: Literal["accept", "reject", "needs_review"]
    storage_action: Literal["stored", "already_stored", "not_stored"]
    reasons: list[str]
    evaluator_version: str
    evaluator_usage: dict | None
    created_at: datetime
    updated_at: datetime


class AgentAlignmentCreate(BaseModel):
    proposed_action: str = Field(min_length=1)
    agent_capability: Literal[
        "agent_loop",
        "tool_calling",
        "memory",
        "context_management",
        "retrieval",
        "evaluation",
        "user_interaction",
        "agent_foundation",
    ]
    success_criterion: str = Field(min_length=1)
    goal_connection: str = Field(min_length=1)
    learning_outcome: str = Field(min_length=1)


class AgentAlignmentRead(AgentAlignmentCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    goal_contract_id: int
    decision: Literal["proceed", "defer"]
    evaluation_notes: list[str]
    created_at: datetime


class AgentTaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    user_goal: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(min_length=1)


class AgentTaskRead(AgentTaskCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    goal_contract_id: int
    status: Literal[
        "draft",
        "planning",
        "in_progress",
        "waiting_for_approval",
        "blocked",
        "completed",
        "cancelled",
    ]
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


class AgentTaskUpdate(BaseModel):
    status: Literal[
        "draft",
        "planning",
        "in_progress",
        "waiting_for_approval",
        "blocked",
        "completed",
        "cancelled",
    ]


class PlanStepCreate(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    agent_capability: Literal[
        "agent_loop",
        "tool_calling",
        "memory",
        "context_management",
        "retrieval",
        "evaluation",
        "user_interaction",
        "agent_foundation",
    ]
    success_criterion: str = Field(min_length=1)
    goal_connection: str = Field(min_length=1)
    learning_outcome: str = Field(min_length=1)


class PlanCreate(BaseModel):
    steps: list[PlanStepCreate] = Field(min_length=1)


class PlanStepRead(PlanStepCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    sequence: int
    status: Literal["pending", "in_progress", "completed", "blocked", "skipped"]
    result_summary: str | None
    created_at: datetime
    updated_at: datetime


class ExecutionTraceCreate(BaseModel):
    plan_step_id: int | None = None
    event_type: Literal[
        "plan",
        "tool_call",
        "evaluation",
        "context_compaction",
        "user_input",
        "model_call",
        "agent_run",
        "memory",
    ]
    status: str = Field(default="recorded", min_length=1, max_length=30)
    input_summary: str = Field(min_length=1)
    output_summary: str = Field(min_length=1)
    metadata_json: dict | None = None


class ExecutionTraceRead(ExecutionTraceCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    created_at: datetime


class ToolDefinitionRead(BaseModel):
    name: str
    description: str
    permission: Literal["read", "write"]
    effect: Literal["read", "internal_write", "external_write"]
    idempotency_mode: Literal["none", "operation_key"]
    repeat_policy: Literal["always_allow", "naturally_idempotent", "business_unique"]
    input_schema: dict


class ToolCallCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1, max_length=100)
    arguments: dict = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=240)
    plan_step_id: int | None = None


class TaskPolicyUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed_tools: list[str]


class TaskPolicyRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    version: int
    allowed_tools: list[str]
    external_writes_require_approval: bool
    created_at: datetime
    updated_at: datetime


class ToolCallRead(BaseModel):
    tool_call_id: int
    trace_id: int | None
    task_id: int
    plan_step_id: int | None
    tool_name: str
    permission: Literal["read", "write"]
    effect: Literal["read", "internal_write", "external_write"]
    idempotency_mode: Literal["none", "operation_key"]
    repeat_policy: Literal["always_allow", "naturally_idempotent", "business_unique"]
    idempotency_key: str
    request_fingerprint: str
    status: Literal[
        "proposed",
        "authorized",
        "executing",
        "succeeded",
        "failed",
        "denied",
        "outcome_unknown",
        "needs_review",
    ]
    attempt_count: int
    replay_count: int
    arguments: dict
    output: dict | None
    error: str | None
    duration_ms: float | None
    replayed: bool = False
    authorization_id: int | None
    authorization_decision: Literal["allowed", "denied", "requires_approval"] | None
    authorization_reason: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class ToolRecoveryDecisionRead(BaseModel):
    tool_call_id: int
    previous_status: str
    status: str
    action: Literal["retried", "marked_outcome_unknown", "needs_review"]
    reason: str


class ToolRecoveryRunRead(BaseModel):
    task_id: int
    stale_before: datetime
    scanned_count: int
    recovered_count: int
    decisions: list[ToolRecoveryDecisionRead]


class AgentRunCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["demo", "openai"] = "demo"
    model: str | None = Field(default=None, min_length=1, max_length=120)
    max_steps: int = Field(default=4, ge=1, le=8)


class MemoryContextRead(BaseModel):
    goal_contract: dict
    memories: list[dict]
    retrieval: dict


class AgentRunStepRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    run_id: int
    task_id: int
    sequence: int
    action_type: Literal["tool_call", "final_answer"]
    model_request: dict
    model_response: dict
    tool_call_id: int | None
    observation: dict | None
    created_at: datetime


class AgentRunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    task_id: int
    provider: Literal["demo", "openai"]
    model: str
    status: Literal["running", "completed", "max_steps", "failed", "needs_review"]
    max_steps: int
    step_count: int
    final_answer: str | None
    stop_reason: str | None
    error: str | None
    steps: list[AgentRunStepRead]
    created_at: datetime
    started_at: datetime
    completed_at: datetime | None


class AgentRunRecoveryDecisionRead(BaseModel):
    run_id: int
    previous_status: str
    status: str
    action: Literal[
        "resumed",
        "completed_from_saved_answer",
        "restored_observation",
        "executed_pending_tool",
        "needs_review",
    ]
    reason: str


class AgentRunRecoveryRead(BaseModel):
    task_id: int
    stale_before: datetime
    scanned_count: int
    recovered_count: int
    decisions: list[AgentRunRecoveryDecisionRead]


class LearningGraphRunRead(BaseModel):
    thread_id: str
    state: dict
    checkpoint_count: int
    awaiting_approval: bool
    interrupt: dict | None
    next_nodes: list[str]


class LearningGraphResumeCreate(BaseModel):
    approved: bool
    comment: str | None = None


class GraphCheckpointRead(BaseModel):
    checkpoint_id: str | None
    step: int | None
    next_nodes: list[str]
    state: dict


class JobSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(default="public_job_board", max_length=40)
    url: HttpUrl
    is_active: bool = True


class JobSourceRead(JobSourceCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class JobArchiveCreate(BaseModel):
    source_id: int | None = None
    source_url: HttpUrl
    raw_content: str = Field(min_length=1)
    parse: bool = True
    create_job: bool = True


class JobArchiveRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_id: int | None
    job_id: int | None
    source_url: str
    canonical_url: str
    content_hash: str
    raw_content: str
    structured_data: dict | None
    status: str
    fetched_at: datetime
    parsed_at: datetime | None


class JobFetchCreate(BaseModel):
    source_id: int | None = None
    source_url: HttpUrl
    parse: bool = True
    create_job: bool = True


class ParsedJobRead(BaseModel):
    company: str
    title: str
    location: str
    description: str
    published_at: str | None
    requirements: list[RequirementRead]
