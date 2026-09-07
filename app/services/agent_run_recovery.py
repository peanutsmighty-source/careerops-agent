from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models import AgentRun, AgentRunStep, AgentTask, ExecutionTrace, ToolCallRecord
from app.services.agent_loop import (
    AgentLoopEngine,
    create_agent_model,
    get_agent_run,
    json_safe,
)
from app.services.tool_recovery import recover_tool_call
from app.services.tool_runtime import serialize_tool_call
from app.services.agent_workflow import resume_agent_workflow_if_interrupted


TERMINAL_TOOL_STATUSES = {"succeeded", "failed", "denied"}


@dataclass(frozen=True)
class AgentRunRecoveryDecision:
    run_id: int
    previous_status: str
    status: str
    action: str
    reason: str


@dataclass(frozen=True)
class AgentRunRecoveryReport:
    task_id: int
    stale_before: datetime
    scanned_count: int
    recovered_count: int
    decisions: list[AgentRunRecoveryDecision]


def recover_stale_agent_runs(
    session: Session,
    *,
    task: AgentTask,
    stale_after_seconds: int = 60,
) -> AgentRunRecoveryReport:
    stale_before = datetime.utcnow() - timedelta(seconds=stale_after_seconds)
    candidates = list(
        session.scalars(
            select(AgentRun)
            .options(selectinload(AgentRun.steps))
            .where(AgentRun.task_id == task.id, AgentRun.status == "running")
            .order_by(AgentRun.id)
        )
    )
    stale_runs = [
        run for run in candidates if _last_activity_at(run) <= stale_before
    ]
    decisions = [_recover_run(session, task, run) for run in stale_runs]
    return AgentRunRecoveryReport(
        task_id=task.id,
        stale_before=stale_before,
        scanned_count=len(stale_runs),
        recovered_count=sum(
            decision.action != "needs_review" for decision in decisions
        ),
        decisions=decisions,
    )


def _last_activity_at(run: AgentRun) -> datetime:
    if run.steps:
        return run.steps[-1].created_at
    return run.started_at


def _recover_run(
    session: Session, task: AgentTask, run: AgentRun
) -> AgentRunRecoveryDecision:
    previous_status = run.status
    try:
        model = create_agent_model(run.provider, run.model)
    except ValueError as exc:
        return _mark_run_for_review(session, run, previous_status, str(exc))

    engine = AgentLoopEngine(model)
    if not run.steps:
        engine.run(run.id)
        _resume_outer_workflow(session, run.id, model)
        return _finished_decision(
            session,
            run.id,
            previous_status,
            "resumed",
            "No model step had been committed, so the Agent loop restarted safely.",
        )

    latest = run.steps[-1]
    if latest.action_type == "final_answer":
        engine.run(run.id)
        _resume_outer_workflow(session, run.id, model)
        return _finished_decision(
            session,
            run.id,
            previous_status,
            "completed_from_saved_answer",
            "The saved final answer was used without calling the model again.",
        )

    if latest.observation is not None:
        engine.run(run.id)
        _resume_outer_workflow(session, run.id, model)
        return _finished_decision(
            session,
            run.id,
            previous_status,
            "resumed",
            "The last observation was complete, so the next model step could run.",
        )

    if latest.tool_call_id is None:
        engine.execute_pending_tool_step(run.id, latest.id)
        engine.run(run.id)
        _resume_outer_workflow(session, run.id, model)
        return _finished_decision(
            session,
            run.id,
            previous_status,
            "executed_pending_tool",
            "The model decision was saved before any ToolCall was committed; the tool ran once.",
        )

    tool_call = session.get(ToolCallRecord, latest.tool_call_id)
    if not tool_call:
        return _mark_run_for_review(
            session,
            run,
            previous_status,
            "The Agent step references a ToolCall that no longer exists.",
        )

    if tool_call.status not in TERMINAL_TOOL_STATUSES:
        recover_tool_call(session, task=task, record=tool_call)
        session.expire_all()
        tool_call = session.get(ToolCallRecord, latest.tool_call_id)

    if tool_call and tool_call.status in TERMINAL_TOOL_STATUSES:
        _restore_observation(session, latest.id, tool_call)
        engine.run(run.id)
        _resume_outer_workflow(session, run.id, model)
        return _finished_decision(
            session,
            run.id,
            previous_status,
            "restored_observation",
            "The durable ToolCall result was restored into the Agent step before continuing.",
        )

    reason = (
        tool_call.error
        if tool_call and tool_call.error
        else "The ToolCall result is not safe to resume automatically."
    )
    session.expire_all()
    run = session.get(AgentRun, run.id)
    return _mark_run_for_review(session, run, previous_status, reason)


def _restore_observation(
    session: Session, step_id: int, tool_call: ToolCallRecord
) -> None:
    step = session.get(AgentRunStep, step_id)
    if not step:
        raise ValueError("agent run step no longer exists")
    step.observation = json_safe(serialize_tool_call(tool_call))
    session.commit()


def _resume_outer_workflow(session: Session, run_id: int, model) -> None:
    run = get_agent_run(session, run_id)
    resume_agent_workflow_if_interrupted(run, model)


def _finished_decision(
    session: Session,
    run_id: int,
    previous_status: str,
    action: str,
    reason: str,
) -> AgentRunRecoveryDecision:
    recovered = get_agent_run(session, run_id)
    return AgentRunRecoveryDecision(
        run_id=recovered.id,
        previous_status=previous_status,
        status=recovered.status,
        action=action,
        reason=reason,
    )


def _mark_run_for_review(
    session: Session,
    run: AgentRun,
    previous_status: str,
    reason: str,
) -> AgentRunRecoveryDecision:
    run.status = "needs_review"
    run.stop_reason = "recovery_required"
    run.error = reason
    session.add(
        ExecutionTrace(
            task_id=run.task_id,
            event_type="agent_run",
            status="needs_review",
            input_summary=f"Recover stale Agent run {run.id}.",
            output_summary=reason,
            metadata_json={
                "agent_run_id": run.id,
                "previous_status": previous_status,
                "recovery": True,
            },
        )
    )
    session.commit()
    return AgentRunRecoveryDecision(
        run_id=run.id,
        previous_status=previous_status,
        status=run.status,
        action="needs_review",
        reason=reason,
    )
