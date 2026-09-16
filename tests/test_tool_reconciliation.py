from sqlalchemy.orm import Session

from app.database import engine
from app.models import AgentTask, ToolCallRecord
from app.services.authorization import get_or_create_task_policy
from app.services.tool_runtime import request_fingerprint, run_tool_call
from app.services.tools import (
    ExternalOperationUncertainError,
    ReconciliationResult,
    TOOL_REGISTRY,
    ToolArguments,
    ToolDefinition,
)


def test_timeout_after_provider_success_is_reconciled_without_duplicate_write(
    client, monkeypatch
):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Reconcile a provider timeout",
            "user_goal": "Do not create the external record twice.",
            "success_criteria": ["Recover the provider result by operation ID."],
        },
    ).json()
    calls = {"write": 0, "query": 0}

    def external_handler(context, arguments):
        calls["write"] += 1
        raise ExternalOperationUncertainError(
            "provider-op-42", "provider accepted the write, but its response timed out"
        )

    def reconcile(context, provider_operation_id):
        calls["query"] += 1
        assert provider_operation_id == "provider-op-42"
        return ReconciliationResult(
            status="succeeded",
            output={"external_record_id": "record-42"},
            detail="Provider confirmed the original write succeeded.",
        )

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "create_external_record",
        ToolDefinition(
            name="create_external_record",
            description="Create one fake provider record for reconciliation tests.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ToolArguments,
            handler=external_handler,
            reconciler=reconcile,
        ),
    )
    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        policy = get_or_create_task_policy(session, task)
        policy.allowed_tools = [*policy.allowed_tools, "create_external_record"]
        policy.external_writes_require_approval = False
        session.commit()
        outcome = run_tool_call(
            session,
            task=task,
            tool_name="create_external_record",
            arguments={},
            requested_idempotency_key="create-external-record-once",
            plan_step_id=None,
        )
        assert outcome.record.status == "outcome_unknown"
        assert outcome.record.provider_operation_id == "provider-op-42"
        assert outcome.record.completed_at is None

    response = client.post(
        f"/agent/tasks/{task_payload['id']}/recover-tool-calls?stale_after_seconds=0"
    )

    assert response.status_code == 200
    report = response.json()
    assert report["recovered_count"] == 1
    assert report["decisions"][0]["action"] == "reconciled_succeeded"
    record = client.get(
        f"/agent/tasks/{task_payload['id']}/tool-calls"
    ).json()[0]
    assert record["status"] == "succeeded"
    assert record["attempt_count"] == 1
    assert record["provider_operation_id"] == "provider-op-42"
    assert record["reconciliation_status"] == "succeeded"
    assert record["output"] == {"external_record_id": "record-42"}
    assert calls == {"write": 1, "query": 1}


def test_non_terminal_provider_status_stays_unknown_without_retry(client, monkeypatch):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Wait for a provider operation",
            "user_goal": "Do not retry while the provider is still processing.",
            "success_criteria": ["Keep the operation pending."],
        },
    ).json()
    calls = {"write": 0}

    def external_handler(context, arguments):
        calls["write"] += 1
        return {"unexpected": True}

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "pending_external_record",
        ToolDefinition(
            name="pending_external_record",
            description="Fake pending provider operation.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ToolArguments,
            handler=external_handler,
            reconciler=lambda context, operation_id: ReconciliationResult(
                status="pending", detail="Provider is still processing the original request."
            ),
        ),
    )
    with Session(engine) as session:
        session.add(
            ToolCallRecord(
                task_id=task_payload["id"],
                tool_name="pending_external_record",
                permission="write",
                effect="external_write",
                idempotency_mode="operation_key",
                repeat_policy="business_unique",
                idempotency_key="pending-record-once",
                request_fingerprint=request_fingerprint("pending_external_record", {}),
                arguments_json={},
                status="outcome_unknown",
                attempt_count=1,
                provider_operation_id="provider-pending-7",
            )
        )
        session.commit()

    report = client.post(
        f"/agent/tasks/{task_payload['id']}/recover-tool-calls?stale_after_seconds=0"
    ).json()

    assert report["recovered_count"] == 0
    assert report["decisions"][0]["action"] == "reconciliation_pending"
    record = client.get(
        f"/agent/tasks/{task_payload['id']}/tool-calls"
    ).json()[0]
    assert record["status"] == "outcome_unknown"
    assert record["reconciliation_status"] == "pending"
    assert record["attempt_count"] == 1
    assert calls["write"] == 0


def test_provider_not_found_is_queried_before_authorized_retry(client, monkeypatch):
    task_payload = client.post(
        "/agent/tasks",
        json={
            "title": "Retry only after provider reconciliation",
            "user_goal": "Retry a write only when the original operation does not exist.",
            "success_criteria": ["Query before retrying."],
        },
    ).json()
    events = []

    def external_handler(context, arguments):
        events.append("write")
        return {"external_record_id": "record-after-retry"}

    def reconcile(context, operation_id):
        events.append(f"query:{operation_id}")
        return ReconciliationResult(status="not_found")

    monkeypatch.setitem(
        TOOL_REGISTRY,
        "retry_external_record",
        ToolDefinition(
            name="retry_external_record",
            description="Fake provider operation that is safe to retry after reconciliation.",
            permission="write",
            effect="external_write",
            idempotency_mode="operation_key",
            repeat_policy="business_unique",
            arguments_model=ToolArguments,
            handler=external_handler,
            reconciler=reconcile,
        ),
    )
    with Session(engine) as session:
        task = session.get(AgentTask, task_payload["id"])
        policy = get_or_create_task_policy(session, task)
        policy.allowed_tools = [*policy.allowed_tools, "retry_external_record"]
        policy.external_writes_require_approval = False
        session.add(
            ToolCallRecord(
                task_id=task.id,
                tool_name="retry_external_record",
                permission="write",
                effect="external_write",
                idempotency_mode="operation_key",
                repeat_policy="business_unique",
                idempotency_key="retry-record-once",
                request_fingerprint=request_fingerprint("retry_external_record", {}),
                arguments_json={},
                status="outcome_unknown",
                attempt_count=1,
                provider_operation_id="provider-missing-9",
            )
        )
        session.commit()

    report = client.post(
        f"/agent/tasks/{task_payload['id']}/recover-tool-calls?stale_after_seconds=0"
    ).json()

    assert report["recovered_count"] == 1
    assert report["decisions"][0]["action"] == "retried_after_reconciliation"
    record = client.get(
        f"/agent/tasks/{task_payload['id']}/tool-calls"
    ).json()[0]
    assert record["status"] == "succeeded"
    assert record["attempt_count"] == 2
    assert record["output"] == {"external_record_id": "record-after-retry"}
    assert events == ["query:provider-missing-9", "write"]
