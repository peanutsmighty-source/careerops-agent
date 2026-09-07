from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentTask, ExecutionTrace, ToolCallRecord
from app.services.authorization import authorize_tool_call
from app.services.tool_runtime import retry_persisted_tool_call
from app.services.tools import get_tool


RECOVERABLE_STATUSES = ("proposed", "authorized", "executing", "outcome_unknown")


@dataclass(frozen=True)
class RecoveryDecision:
    tool_call_id: int
    previous_status: str
    status: str
    action: str
    reason: str


@dataclass(frozen=True)
class RecoveryReport:
    task_id: int
    stale_before: datetime
    scanned_count: int
    recovered_count: int
    decisions: list[RecoveryDecision]


def recover_stale_tool_calls(
    session: Session,
    *,
    task: AgentTask,
    stale_after_seconds: int = 60,
) -> RecoveryReport:
    stale_before = datetime.utcnow() - timedelta(seconds=stale_after_seconds)
    records = list(
        session.scalars(
            select(ToolCallRecord)
            .where(
                ToolCallRecord.task_id == task.id,
                ToolCallRecord.status.in_(RECOVERABLE_STATUSES),
                ToolCallRecord.updated_at <= stale_before,
            )
            .order_by(ToolCallRecord.id)
        )
    )
    decisions = [_recover_record(session, task, record) for record in records]
    return RecoveryReport(
        task_id=task.id,
        stale_before=stale_before,
        scanned_count=len(records),
        recovered_count=sum(decision.action == "retried" for decision in decisions),
        decisions=decisions,
    )


def _recover_record(
    session: Session, task: AgentTask, record: ToolCallRecord
) -> RecoveryDecision:
    previous_status = record.status

    if previous_status == "proposed":
        return _mark_for_review(
            session,
            record,
            previous_status,
            "The process stopped before authorization; the tool was not executed.",
        )

    if previous_status == "outcome_unknown":
        return _mark_for_review(
            session,
            record,
            previous_status,
            "No external reconciliation adapter is registered for this tool.",
        )

    if record.effect == "external_write":
        record.status = "outcome_unknown"
        record.error = (
            "The external operation may have succeeded before the response was stored. "
            "Reconcile it before retrying."
        )
        _add_recovery_trace(
            session,
            record,
            status="outcome_unknown",
            output_summary="Marked the external write outcome unknown; no retry was attempted.",
            previous_status=previous_status,
        )
        session.commit()
        return RecoveryDecision(
            tool_call_id=record.id,
            previous_status=previous_status,
            status=record.status,
            action="marked_outcome_unknown",
            reason=record.error,
        )

    definition = get_tool(record.tool_name)
    if not definition:
        return _mark_for_review(
            session,
            record,
            previous_status,
            "The tool is no longer registered, so its saved request cannot be retried.",
        )

    authorization = authorize_tool_call(
        session,
        task=task,
        record=record,
        definition=definition,
    )
    if not authorization.allowed:
        return _mark_for_review(
            session,
            record,
            previous_status,
            f"Recovery authorization failed: {authorization.authorization.reason}",
        )

    retry_persisted_tool_call(session, task=task, record=record)
    return RecoveryDecision(
        tool_call_id=record.id,
        previous_status=previous_status,
        status=record.status,
        action="retried",
        reason=(
            "The read or internal database operation is safe to retry after its previous "
            "uncommitted transaction was rolled back."
        ),
    )


def recover_tool_call(
    session: Session, *, task: AgentTask, record: ToolCallRecord
) -> RecoveryDecision:
    """Recover one known ToolCall for an owning AgentRun recovery operation."""
    return _recover_record(session, task, record)


def _mark_for_review(
    session: Session,
    record: ToolCallRecord,
    previous_status: str,
    reason: str,
) -> RecoveryDecision:
    record.status = "needs_review"
    record.error = reason
    _add_recovery_trace(
        session,
        record,
        status="needs_review",
        output_summary=reason,
        previous_status=previous_status,
    )
    session.commit()
    return RecoveryDecision(
        tool_call_id=record.id,
        previous_status=previous_status,
        status=record.status,
        action="needs_review",
        reason=reason,
    )


def _add_recovery_trace(
    session: Session,
    record: ToolCallRecord,
    *,
    status: str,
    output_summary: str,
    previous_status: str,
) -> None:
    session.add(
        ExecutionTrace(
            task_id=record.task_id,
            plan_step_id=record.plan_step_id,
            event_type="tool_call",
            status=status,
            input_summary=f"Recover stale tool call {record.id}.",
            output_summary=output_summary,
            metadata_json={
                "tool_call_id": record.id,
                "tool_name": record.tool_name,
                "effect": record.effect,
                "previous_status": previous_status,
                "recovery": True,
            },
        )
    )
