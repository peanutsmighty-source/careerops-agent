from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AgentTask, ExecutionTrace, ToolApproval, ToolCallRecord
from app.services.identity import AuthenticatedActor
from app.services.tools import get_tool


class ToolApprovalError(ValueError):
    pass


def create_tool_approval(
    session: Session,
    *,
    task: AgentTask,
    record: ToolCallRecord,
    actor: AuthenticatedActor,
    expires_in_seconds: int,
) -> ToolApproval:
    definition = get_tool(record.tool_name)
    if not definition or definition.effect != "external_write":
        raise ToolApprovalError("only a registered external-write tool can be approved")
    if record.task_id != task.id:
        raise ToolApprovalError("tool call does not belong to this task")
    if record.status not in {"awaiting_approval", "denied"}:
        raise ToolApprovalError("tool call is not waiting for external-write approval")
    latest_authorization = record.authorizations[-1] if record.authorizations else None
    if not latest_authorization or latest_authorization.decision != "requires_approval":
        raise ToolApprovalError("the latest authorization did not request human approval")

    now = datetime.utcnow()
    active = session.scalar(
        select(ToolApproval).where(ToolApproval.active_slot == record.id)
    )
    if active and active.expires_at > now:
        raise ToolApprovalError("an active approval already exists for this tool call")
    if active:
        active.active_slot = None
        active.revoked_at = now
        session.flush()

    approval = ToolApproval(
        task_id=task.id,
        tool_call_id=record.id,
        active_slot=record.id,
        actor_id=actor.actor_id,
        tool_name=record.tool_name,
        idempotency_key=record.idempotency_key,
        request_fingerprint=record.request_fingerprint,
        expires_at=now + timedelta(seconds=expires_in_seconds),
    )
    record.status = "awaiting_approval"
    session.add(approval)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise ToolApprovalError("an active approval already exists for this tool call") from exc
    session.add(
        ExecutionTrace(
            task_id=task.id,
            plan_step_id=record.plan_step_id,
            event_type="tool_call",
            status="approval_granted",
            input_summary=f"Approve external tool call {record.id}.",
            output_summary=(
                f"Actor {actor.actor_id} granted a one-time approval that expires at "
                f"{approval.expires_at.isoformat()}."
            ),
            metadata_json={
                "tool_call_id": record.id,
                "approval_id": approval.id,
                "actor_id": actor.actor_id,
                "tool_name": record.tool_name,
                "idempotency_key": record.idempotency_key,
                "request_fingerprint": record.request_fingerprint,
                "expires_at": approval.expires_at.isoformat(),
            },
        )
    )
    session.commit()
    session.refresh(approval)
    return approval


def consume_matching_tool_approval(
    session: Session, *, record: ToolCallRecord
) -> ToolApproval | None:
    now = datetime.utcnow()
    approval = session.scalar(
        select(ToolApproval).where(
            ToolApproval.active_slot == record.id,
            ToolApproval.task_id == record.task_id,
            ToolApproval.tool_call_id == record.id,
            ToolApproval.tool_name == record.tool_name,
            ToolApproval.idempotency_key == record.idempotency_key,
            ToolApproval.request_fingerprint == record.request_fingerprint,
            ToolApproval.consumed_at.is_(None),
            ToolApproval.revoked_at.is_(None),
            ToolApproval.expires_at > now,
        )
    )
    if not approval:
        return None
    result = session.execute(
        update(ToolApproval)
        .where(
            ToolApproval.id == approval.id,
            ToolApproval.active_slot == record.id,
            ToolApproval.consumed_at.is_(None),
            ToolApproval.revoked_at.is_(None),
            ToolApproval.expires_at > now,
        )
        .values(active_slot=None, consumed_at=now)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    if result.rowcount != 1:
        return None
    session.refresh(approval)
    return approval


def list_tool_approvals(session: Session, tool_call_id: int) -> list[ToolApproval]:
    return list(
        session.scalars(
            select(ToolApproval)
            .where(ToolApproval.tool_call_id == tool_call_id)
            .order_by(ToolApproval.id)
        )
    )
