from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import AgentTask, ExecutionTrace, ToolCallRecord
from app.services.authorization import authorize_tool_call
from app.services.tools import ToolContext, ToolDefinition, execute_tool, get_tool


class UnknownToolError(ValueError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class ToolReplayPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class ToolCallOutcome:
    record: ToolCallRecord
    replayed: bool


RecordLinker = Callable[[Session, ToolCallRecord], None]


def run_tool_call(
    session: Session,
    *,
    task: AgentTask,
    tool_name: str,
    arguments: dict,
    requested_idempotency_key: str | None,
    plan_step_id: int | None,
    record_linker: RecordLinker | None = None,
) -> ToolCallOutcome:
    definition = get_tool(tool_name)
    if not definition:
        raise UnknownToolError("unknown tool")

    fingerprint = request_fingerprint(tool_name, arguments, plan_step_id=plan_step_id)
    idempotency_key = _resolve_idempotency_key(definition, requested_idempotency_key)
    existing = _find_tool_call(session, task.id, idempotency_key)
    if existing:
        _link_record(session, existing, record_linker)
        return _replay_or_conflict(session, task, existing, fingerprint)

    record = ToolCallRecord(
        task_id=task.id,
        plan_step_id=plan_step_id,
        tool_name=definition.name,
        permission=definition.permission,
        effect=definition.effect,
        idempotency_mode=definition.idempotency_mode,
        repeat_policy=definition.repeat_policy,
        idempotency_key=idempotency_key,
        request_fingerprint=fingerprint,
        arguments_json=arguments,
        status="proposed",
    )
    session.add(record)
    try:
        session.flush()
        _link_record(session, record, record_linker, commit=False)
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = _find_tool_call(session, task.id, idempotency_key)
        if not existing:
            raise
        _link_record(session, existing, record_linker)
        return _replay_or_conflict(session, task, existing, fingerprint)
    session.refresh(record)

    authorization = authorize_tool_call(
        session,
        task=task,
        record=record,
        definition=definition,
    )
    if not authorization.allowed:
        record.status = "denied"
        record.error = authorization.authorization.reason
        record.completed_at = datetime.utcnow()
        trace = ExecutionTrace(
            task_id=task.id,
            plan_step_id=plan_step_id,
            event_type="tool_call",
            status="denied",
            input_summary=f"Authorize tool {definition.name}.",
            output_summary=authorization.authorization.reason,
            metadata_json={
                "tool_call_id": record.id,
                "tool_name": definition.name,
                "authorization_id": authorization.authorization.id,
                "authorization_decision": authorization.authorization.decision,
                "policy_version": authorization.authorization.policy_version,
                "arguments": arguments,
            },
        )
        session.add(trace)
        session.flush()
        record.trace_id = trace.id
        session.commit()
        session.refresh(record)
        return ToolCallOutcome(record=record, replayed=False)

    record.status = "authorized"
    session.commit()
    record.status = "executing"
    record.attempt_count += 1
    record.started_at = datetime.utcnow()
    session.commit()

    execution = execute_tool(
        ToolContext(session=session, task=task),
        tool_name=definition.name,
        arguments=arguments,
        granted_permissions={definition.permission},
    )
    record.status = {
        "success": "succeeded",
        "error": "failed",
        "denied": "denied",
    }[execution.status]
    record.arguments_json = execution.arguments
    record.output_json = execution.output
    record.error = execution.error
    record.duration_ms = execution.duration_ms
    record.completed_at = datetime.utcnow()

    trace = ExecutionTrace(
        task_id=task.id,
        plan_step_id=plan_step_id,
        event_type="tool_call",
        status=execution.status,
        input_summary=f"Call tool {execution.tool_name}.",
        output_summary=(
            f"Tool {execution.tool_name} returned successfully."
            if execution.status == "success"
            else f"Tool {execution.tool_name} {execution.status}: {execution.error}"
        ),
        metadata_json={
            "tool_call_id": record.id,
            "tool_name": execution.tool_name,
            "permission": execution.permission,
            "effect": definition.effect,
            "idempotency_mode": definition.idempotency_mode,
            "repeat_policy": definition.repeat_policy,
            "idempotency_key": record.idempotency_key,
            "request_fingerprint": record.request_fingerprint,
            "authorization_id": authorization.authorization.id,
            "authorization_decision": authorization.authorization.decision,
            "authorization_reason": authorization.authorization.reason,
            "policy_version": authorization.authorization.policy_version,
            "arguments": execution.arguments,
            "output": execution.output,
            "error": execution.error,
            "duration_ms": execution.duration_ms,
        },
    )
    session.add(trace)
    session.flush()
    record.trace_id = trace.id
    session.commit()
    session.refresh(record)
    return ToolCallOutcome(record=record, replayed=False)


def list_task_tool_calls(session: Session, task_id: int) -> list[ToolCallRecord]:
    return list(
        session.scalars(
            select(ToolCallRecord)
            .where(ToolCallRecord.task_id == task_id)
            .order_by(ToolCallRecord.id.desc())
        )
    )


def retry_persisted_tool_call(
    session: Session,
    *,
    task: AgentTask,
    record: ToolCallRecord,
) -> ToolCallRecord:
    """Retry a stale read/internal-write call whose prior DB transaction did not commit."""
    definition = get_tool(record.tool_name)
    if not definition:
        raise UnknownToolError("unknown tool")

    record.status = "executing"
    record.attempt_count += 1
    record.started_at = datetime.utcnow()
    record.completed_at = None
    record.error = None
    session.commit()

    execution = execute_tool(
        ToolContext(session=session, task=task),
        tool_name=definition.name,
        arguments=record.arguments_json,
        granted_permissions={definition.permission},
    )
    record.status = {
        "success": "succeeded",
        "error": "failed",
        "denied": "denied",
    }[execution.status]
    record.arguments_json = execution.arguments
    record.output_json = execution.output
    record.error = execution.error
    record.duration_ms = execution.duration_ms
    record.completed_at = datetime.utcnow()
    trace = ExecutionTrace(
        task_id=task.id,
        plan_step_id=record.plan_step_id,
        event_type="tool_call",
        status="recovered" if execution.status == "success" else execution.status,
        input_summary=f"Retry stale tool call {record.id} with its saved arguments.",
        output_summary=(
            f"Recovered tool {record.tool_name} successfully."
            if execution.status == "success"
            else f"Recovery attempt {execution.status}: {execution.error}"
        ),
        metadata_json={
            "tool_call_id": record.id,
            "tool_name": record.tool_name,
            "effect": record.effect,
            "idempotency_key": record.idempotency_key,
            "attempt_count": record.attempt_count,
            "recovery": True,
            "arguments": execution.arguments,
            "output": execution.output,
            "error": execution.error,
        },
    )
    session.add(trace)
    session.flush()
    record.trace_id = trace.id
    session.commit()
    session.refresh(record)
    return record


def request_fingerprint(
    tool_name: str, arguments: dict, *, plan_step_id: int | None = None
) -> str:
    canonical_request = json.dumps(
        {
            "tool_name": tool_name,
            "arguments": arguments,
            "plan_step_id": plan_step_id,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()


def serialize_tool_call(record: ToolCallRecord, *, replayed: bool = False) -> dict:
    authorization = record.authorizations[-1] if record.authorizations else None
    return {
        "tool_call_id": record.id,
        "trace_id": record.trace_id,
        "task_id": record.task_id,
        "plan_step_id": record.plan_step_id,
        "tool_name": record.tool_name,
        "permission": record.permission,
        "effect": record.effect,
        "idempotency_mode": record.idempotency_mode,
        "repeat_policy": record.repeat_policy,
        "idempotency_key": record.idempotency_key,
        "request_fingerprint": record.request_fingerprint,
        "status": record.status,
        "attempt_count": record.attempt_count,
        "replay_count": record.replay_count,
        "arguments": record.arguments_json,
        "output": record.output_json,
        "error": record.error,
        "duration_ms": record.duration_ms,
        "replayed": replayed,
        "authorization_id": authorization.id if authorization else None,
        "authorization_decision": authorization.decision if authorization else None,
        "authorization_reason": authorization.reason if authorization else None,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
    }


def _resolve_idempotency_key(
    definition: ToolDefinition, requested_idempotency_key: str | None
) -> str:
    if definition.idempotency_mode == "none":
        return f"call:{uuid4().hex}"
    if requested_idempotency_key and requested_idempotency_key.strip():
        return requested_idempotency_key.strip()
    return f"operation:{uuid4().hex}"


def _find_tool_call(
    session: Session, task_id: int, idempotency_key: str
) -> ToolCallRecord | None:
    return session.scalar(
        select(ToolCallRecord).where(
            ToolCallRecord.task_id == task_id,
            ToolCallRecord.idempotency_key == idempotency_key,
        )
    )


def _link_record(
    session: Session,
    record: ToolCallRecord,
    record_linker: RecordLinker | None,
    *,
    commit: bool = True,
) -> None:
    if not record_linker:
        return
    record_linker(session, record)
    if commit:
        session.commit()


def _replay_or_conflict(
    session: Session,
    task: AgentTask,
    existing: ToolCallRecord,
    fingerprint: str,
) -> ToolCallOutcome:
    if existing.request_fingerprint != fingerprint:
        raise IdempotencyConflictError(
            "idempotency key was already used with different tool arguments"
        )
    definition = get_tool(existing.tool_name)
    if not definition:
        raise UnknownToolError("unknown tool")
    authorization = authorize_tool_call(
        session,
        task=task,
        record=existing,
        definition=definition,
    )
    if not authorization.allowed:
        raise ToolReplayPermissionError(authorization.authorization.reason)
    existing.replay_count += 1
    session.add(
        ExecutionTrace(
            task_id=existing.task_id,
            plan_step_id=existing.plan_step_id,
            event_type="tool_call",
            status="replayed",
            input_summary=f"Replay tool call {existing.id}.",
            output_summary=(
                "Returned the existing ToolCall result without executing the handler again."
            ),
            metadata_json={
                "tool_call_id": existing.id,
                "tool_name": existing.tool_name,
                "idempotency_key": existing.idempotency_key,
                "request_fingerprint": existing.request_fingerprint,
                "replayed_status": existing.status,
                "authorization_id": authorization.authorization.id,
                "authorization_decision": authorization.authorization.decision,
                "policy_version": authorization.authorization.policy_version,
            },
        )
    )
    session.commit()
    session.refresh(existing)
    return ToolCallOutcome(record=existing, replayed=True)
