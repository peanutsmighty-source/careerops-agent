from datetime import datetime, timedelta

from pydantic import Field
from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentTask, ToolApproval, ToolCallRecord
from app.services.authorization import get_or_create_task_policy
from app.services.tool_runtime import request_fingerprint
from app.services.tools import (
    ReconciliationResult,
    TOOL_REGISTRY,
    ToolArguments,
    ToolDefinition,
)


class ExternalMessageArguments(ToolArguments):
    message: str = Field(min_length=1)


def test_authenticated_one_time_approval_is_attributable_scoped_and_consumed(
    client, monkeypatch
):
    monkeypatch.setenv("CAREEROPS_OPERATOR_TOKEN", "test-operator-secret")
    monkeypatch.setenv("CAREEROPS_OPERATOR_ID", "operator:alice")
    writes = []

    def send_message(context, arguments):
        writes.append(arguments.message)
        return {"provider_message_id": f"message-{len(writes)}"}

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "send_external_message",
        ToolDefinition(
            name="send_external_message",
            description="Send one fake external message.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ExternalMessageArguments,
            handler=send_message,
        ),
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Approve one external message",
            "user_goal": "Attribute and scope a one-time write approval.",
            "success_criteria": ["Only the approved operation executes."],
        },
    ).json()
    with Session(engine) as session:
        task_record = session.get(AgentTask, task["id"])
        policy = get_or_create_task_policy(session, task_record)
        policy.allowed_tools = [*policy.allowed_tools, "send_external_message"]
        session.commit()

    first_payload = {
        "tool_name": "send_external_message",
        "arguments": {"message": "first"},
        "idempotency_key": "external-message-first",
    }
    proposed = client.post(
        f"/agent/tasks/{task['id']}/tool-calls", json=first_payload
    ).json()
    assert proposed["status"] == "awaiting_approval"
    assert proposed["authorization_decision"] == "requires_approval"
    assert proposed["attempt_count"] == 0
    assert writes == []

    approval_url = (
        f"/agent/tasks/{task['id']}/tool-calls/"
        f"{proposed['tool_call_id']}/approvals"
    )
    assert client.post(approval_url, json={}).status_code == 401
    assert client.post(
        approval_url,
        json={},
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401
    granted = client.post(
        approval_url,
        json={"expires_in_seconds": 300},
        headers={"Authorization": "Bearer test-operator-secret"},
    )
    assert granted.status_code == 201
    approval = granted.json()
    assert approval["actor_id"] == "operator:alice"
    assert approval["tool_call_id"] == proposed["tool_call_id"]
    assert approval["consumed_at"] is None
    duplicate = client.post(
        approval_url,
        json={},
        headers={"Authorization": "Bearer test-operator-secret"},
    )
    assert duplicate.status_code == 409
    assert "active approval" in duplicate.json()["detail"]

    executed = client.post(
        f"/agent/tasks/{task['id']}/tool-calls", json=first_payload
    ).json()
    assert executed["status"] == "succeeded"
    assert executed["attempt_count"] == 1
    assert executed["authorization_actor_id"] == "operator:alice"
    assert executed["approval_id"] == approval["id"]
    assert writes == ["first"]
    already_finished = client.post(
        approval_url,
        json={},
        headers={"Authorization": "Bearer test-operator-secret"},
    )
    assert already_finished.status_code == 409
    assert "not waiting" in already_finished.json()["detail"]

    approvals = client.get(
        approval_url,
        headers={"Authorization": "Bearer test-operator-secret"},
    ).json()
    assert approvals[0]["consumed_at"] is not None

    replayed = client.post(
        f"/agent/tasks/{task['id']}/tool-calls", json=first_payload
    ).json()
    assert replayed["replayed"] is True
    assert replayed["attempt_count"] == 1
    assert writes == ["first"]

    second = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "send_external_message",
            "arguments": {"message": "second"},
            "idempotency_key": "external-message-second",
        },
    ).json()
    assert second["status"] == "awaiting_approval"
    denied_retry = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "send_external_message",
            "arguments": {"message": "second"},
            "idempotency_key": "external-message-second",
        },
    )
    assert denied_retry.status_code == 403
    assert writes == ["first"]


def test_expired_approval_cannot_execute_and_can_be_replaced(client, monkeypatch):
    monkeypatch.setenv("CAREEROPS_OPERATOR_TOKEN", "expiry-secret")
    monkeypatch.setenv("CAREEROPS_OPERATOR_ID", "operator:bob")
    writes = []

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "expire_external_message",
        ToolDefinition(
            name="expire_external_message",
            description="Fake external write with expiring approval.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ExternalMessageArguments,
            handler=lambda context, arguments: writes.append(arguments.message) or {"ok": True},
        ),
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Expire an approval",
            "user_goal": "Reject an expired write approval.",
            "success_criteria": ["Expired approval does not execute."],
        },
    ).json()
    with Session(engine) as session:
        task_record = session.get(AgentTask, task["id"])
        policy = get_or_create_task_policy(session, task_record)
        policy.allowed_tools = [*policy.allowed_tools, "expire_external_message"]
        session.commit()
    payload = {
        "tool_name": "expire_external_message",
        "arguments": {"message": "expires"},
        "idempotency_key": "expiring-message",
    }
    call = client.post(f"/agent/tasks/{task['id']}/tool-calls", json=payload).json()
    approval_url = (
        f"/agent/tasks/{task['id']}/tool-calls/{call['tool_call_id']}/approvals"
    )
    created = client.post(
        approval_url,
        json={"expires_in_seconds": 1},
        headers={"Authorization": "Bearer expiry-secret"},
    ).json()
    with Session(engine) as session:
        approval = session.get(ToolApproval, created["id"])
        approval.expires_at = datetime.utcnow() - timedelta(seconds=1)
        session.commit()

    denied = client.post(f"/agent/tasks/{task['id']}/tool-calls", json=payload)
    assert denied.status_code == 403
    assert writes == []

    replacement = client.post(
        approval_url,
        json={"expires_in_seconds": 300},
        headers={"Authorization": "Bearer expiry-secret"},
    )
    assert replacement.status_code == 201
    assert replacement.json()["id"] != created["id"]
    executed = client.post(f"/agent/tasks/{task['id']}/tool-calls", json=payload)
    assert executed.status_code == 200
    assert executed.json()["status"] == "succeeded"
    assert writes == ["expires"]


def test_reconciled_not_found_requires_a_fresh_approval_before_retry(
    client, monkeypatch
):
    monkeypatch.setenv("CAREEROPS_OPERATOR_TOKEN", "retry-secret")
    monkeypatch.setenv("CAREEROPS_OPERATOR_ID", "operator:retry-reviewer")
    writes = []

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "retry_after_reconciliation",
        ToolDefinition(
            name="retry_after_reconciliation",
            description="Retry only after reconciliation and fresh approval.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ExternalMessageArguments,
            handler=lambda context, arguments: writes.append(arguments.message) or {"ok": True},
            reconciler=lambda context, operation_id: ReconciliationResult(
                status="not_found"
            ),
        ),
    )
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Approve a reconciled retry",
            "user_goal": "Require a new approval after the provider reports not found.",
            "success_criteria": ["No automatic external retry."],
        },
    ).json()
    arguments = {"message": "retry after reconciliation"}
    with Session(engine) as session:
        task_record = session.get(AgentTask, task["id"])
        policy = get_or_create_task_policy(session, task_record)
        policy.allowed_tools = [*policy.allowed_tools, "retry_after_reconciliation"]
        session.add(
            ToolCallRecord(
                task_id=task["id"],
                tool_name="retry_after_reconciliation",
                permission="write",
                effect="external_write",
                idempotency_mode="operation_key",
                repeat_policy="business_unique",
                idempotency_key="reconciled-retry-once",
                request_fingerprint=request_fingerprint(
                    "retry_after_reconciliation", arguments
                ),
                arguments_json=arguments,
                status="outcome_unknown",
                attempt_count=1,
                provider_operation_id="provider-missing-retry",
            )
        )
        session.commit()

    recovery = client.post(
        f"/agent/tasks/{task['id']}/recover-tool-calls?stale_after_seconds=0"
    ).json()
    assert recovery["decisions"][0]["action"] == "awaiting_approval"
    assert writes == []
    record = client.get(f"/agent/tasks/{task['id']}/tool-calls").json()[0]
    approval_url = (
        f"/agent/tasks/{task['id']}/tool-calls/{record['tool_call_id']}/approvals"
    )
    approved = client.post(
        approval_url,
        json={},
        headers={"Authorization": "Bearer retry-secret"},
    )
    assert approved.status_code == 201
    executed = client.post(
        f"/agent/tasks/{task['id']}/tool-calls",
        json={
            "tool_name": "retry_after_reconciliation",
            "arguments": arguments,
            "idempotency_key": "reconciled-retry-once",
        },
    )
    assert executed.status_code == 200
    assert executed.json()["status"] == "succeeded"
    assert executed.json()["attempt_count"] == 2
    assert writes == ["retry after reconciliation"]
