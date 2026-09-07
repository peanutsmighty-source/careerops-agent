from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import AgentTask, Job, PlanStep
from app.services.analytics import skill_demand_rows


ToolPermission = Literal["read", "write"]
ToolEffect = Literal["read", "internal_write", "external_write"]
IdempotencyMode = Literal["none", "operation_key"]
RepeatPolicy = Literal["always_allow", "naturally_idempotent", "business_unique"]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchJobsArguments(ToolArguments):
    query: str = Field(default="", max_length=100)
    limit: int = Field(default=10, ge=1, le=20)


class SkillDemandArguments(ToolArguments):
    limit: int = Field(default=10, ge=1, le=20)


class GetJobArguments(ToolArguments):
    job_id: int = Field(gt=0)


class UpdatePlanStepArguments(ToolArguments):
    plan_step_id: int = Field(gt=0)
    status: Literal["in_progress", "completed", "blocked"]
    result_summary: str | None = Field(default=None, max_length=2000)


@dataclass(frozen=True)
class ToolContext:
    session: Session
    task: AgentTask


ToolHandler = Callable[[ToolContext, ToolArguments], dict]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    permission: ToolPermission
    effect: ToolEffect
    idempotency_mode: IdempotencyMode
    repeat_policy: RepeatPolicy
    arguments_model: type[ToolArguments]
    handler: ToolHandler

    def public_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "permission": self.permission,
            "effect": self.effect,
            "idempotency_mode": self.idempotency_mode,
            "repeat_policy": self.repeat_policy,
            "input_schema": self.arguments_model.model_json_schema(),
        }


@dataclass(frozen=True)
class ToolExecution:
    tool_name: str
    permission: str | None
    status: Literal["success", "error", "denied"]
    arguments: dict
    output: dict | None
    error: str | None
    duration_ms: float


def _search_jobs(context: ToolContext, arguments: ToolArguments) -> dict:
    args = SearchJobsArguments.model_validate(arguments)
    statement = select(Job).order_by(Job.created_at.desc(), Job.id.desc()).limit(args.limit)
    if args.query.strip():
        pattern = f"%{args.query.strip()}%"
        statement = statement.where(
            or_(
                Job.company.ilike(pattern),
                Job.title.ilike(pattern),
                Job.description.ilike(pattern),
            )
        )
    jobs = list(context.session.scalars(statement))
    return {
        "count": len(jobs),
        "jobs": [
            {
                "id": job.id,
                "company": job.company,
                "title": job.title,
                "location": job.location,
                "source_url": job.source_url,
            }
            for job in jobs
        ],
    }


def _get_skill_demand(context: ToolContext, arguments: ToolArguments) -> dict:
    args = SkillDemandArguments.model_validate(arguments)
    rows = skill_demand_rows(context.session)[: args.limit]
    return {"count": len(rows), "skills": rows}


def _get_job(context: ToolContext, arguments: ToolArguments) -> dict:
    args = GetJobArguments.model_validate(arguments)
    job = context.session.get(Job, args.job_id)
    if not job:
        raise ValueError("job not found")
    return {
        "job": {
            "id": job.id,
            "company": job.company,
            "title": job.title,
            "location": job.location,
            "description": job.description,
            "source_url": job.source_url,
        }
    }


def _update_plan_step(context: ToolContext, arguments: ToolArguments) -> dict:
    args = UpdatePlanStepArguments.model_validate(arguments)
    step = context.session.get(PlanStep, args.plan_step_id)
    if not step or step.task_id != context.task.id:
        raise ValueError("plan step does not belong to this task")
    step.status = args.status
    step.result_summary = args.result_summary
    context.session.flush()
    return {
        "plan_step": {
            "id": step.id,
            "sequence": step.sequence,
            "status": step.status,
            "result_summary": step.result_summary,
        }
    }


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    definition.name: definition
    for definition in [
        ToolDefinition(
            name="search_jobs",
            description="Search archived CareerOps jobs by company, title, or JD text.",
            permission="read",
            effect="read",
            idempotency_mode="none",
            repeat_policy="always_allow",
            arguments_model=SearchJobsArguments,
            handler=_search_jobs,
        ),
        ToolDefinition(
            name="get_skill_demand",
            description="Return skills ranked by demand across parsed job descriptions.",
            permission="read",
            effect="read",
            idempotency_mode="none",
            repeat_policy="always_allow",
            arguments_model=SkillDemandArguments,
            handler=_get_skill_demand,
        ),
        ToolDefinition(
            name="get_job",
            description="Read one structured job and its full archived description.",
            permission="read",
            effect="read",
            idempotency_mode="none",
            repeat_policy="always_allow",
            arguments_model=GetJobArguments,
            handler=_get_job,
        ),
        ToolDefinition(
            name="update_plan_step",
            description="Update the execution status and result of one plan step owned by this task.",
            permission="write",
            effect="internal_write",
            idempotency_mode="operation_key",
            repeat_policy="naturally_idempotent",
            arguments_model=UpdatePlanStepArguments,
            handler=_update_plan_step,
        ),
    ]
}


def list_tools() -> list[dict]:
    return [definition.public_schema() for definition in TOOL_REGISTRY.values()]


def get_tool(tool_name: str) -> ToolDefinition | None:
    return TOOL_REGISTRY.get(tool_name)


def execute_tool(
    context: ToolContext,
    *,
    tool_name: str,
    arguments: dict,
    granted_permissions: set[str],
) -> ToolExecution:
    started_at = perf_counter()
    definition = TOOL_REGISTRY.get(tool_name)
    if not definition:
        return _result(started_at, tool_name, None, "error", arguments, error="unknown tool")
    if definition.permission not in granted_permissions:
        return _result(
            started_at,
            tool_name,
            definition.permission,
            "denied",
            arguments,
            error=f"{definition.permission} permission is required",
        )
    try:
        validated = definition.arguments_model.model_validate(arguments)
    except ValidationError as exc:
        return _result(
            started_at,
            tool_name,
            definition.permission,
            "error",
            arguments,
            error=_validation_message(exc),
        )
    try:
        with context.session.begin_nested():
            output = definition.handler(context, validated)
    except ValueError as exc:
        return _result(
            started_at,
            tool_name,
            definition.permission,
            "error",
            validated.model_dump(),
            error=str(exc),
        )
    except Exception as exc:
        return _result(
            started_at,
            tool_name,
            definition.permission,
            "error",
            validated.model_dump(),
            error=f"tool execution failed ({type(exc).__name__})",
        )
    return _result(
        started_at,
        tool_name,
        definition.permission,
        "success",
        validated.model_dump(),
        output=output,
    )


def _result(
    started_at: float,
    tool_name: str,
    permission: str | None,
    status: Literal["success", "error", "denied"],
    arguments: dict,
    *,
    output: dict | None = None,
    error: str | None = None,
) -> ToolExecution:
    return ToolExecution(
        tool_name=tool_name,
        permission=permission,
        status=status,
        arguments=arguments,
        output=output,
        error=error,
        duration_ms=round((perf_counter() - started_at) * 1000, 3),
    )


def _validation_message(exc: ValidationError) -> str:
    issue = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in issue["loc"]) or "arguments"
    return f"{location}: {issue['msg']}"
