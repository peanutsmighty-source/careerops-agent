from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AgentTask, TaskPolicy, ToolAuthorization, ToolCallRecord
from app.services.tools import ToolDefinition, get_tool, list_tools


class InvalidTaskPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class AuthorizationDecision:
    authorization: ToolAuthorization

    @property
    def allowed(self) -> bool:
        return self.authorization.decision == "allowed"


def default_allowed_tools() -> list[str]:
    return [tool["name"] for tool in list_tools() if tool["permission"] == "read"]


def get_or_create_task_policy(session: Session, task: AgentTask) -> TaskPolicy:
    policy = session.scalar(select(TaskPolicy).where(TaskPolicy.task_id == task.id))
    if policy:
        return policy
    policy = TaskPolicy(task_id=task.id, allowed_tools=default_allowed_tools())
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def update_task_policy(
    session: Session, task: AgentTask, *, allowed_tools: list[str]
) -> TaskPolicy:
    normalized = list(dict.fromkeys(name.strip() for name in allowed_tools if name.strip()))
    unknown = [name for name in normalized if not get_tool(name)]
    if unknown:
        raise InvalidTaskPolicyError(f"unknown tools in task policy: {', '.join(unknown)}")
    policy = get_or_create_task_policy(session, task)
    policy.allowed_tools = normalized
    policy.version += 1
    session.commit()
    session.refresh(policy)
    return policy


def authorize_tool_call(
    session: Session,
    *,
    task: AgentTask,
    record: ToolCallRecord,
    definition: ToolDefinition,
) -> AuthorizationDecision:
    policy = get_or_create_task_policy(session, task)
    if definition.name not in policy.allowed_tools:
        decision = "denied"
        reason = f"Tool '{definition.name}' is not allowed by TaskPolicy v{policy.version}."
    elif definition.effect == "external_write" and policy.external_writes_require_approval:
        decision = "requires_approval"
        reason = f"External-write tool '{definition.name}' requires explicit human approval."
    else:
        decision = "allowed"
        reason = f"TaskPolicy v{policy.version} allows tool '{definition.name}'."

    authorization = ToolAuthorization(
        task_id=task.id,
        tool_call_id=record.id,
        policy_id=policy.id,
        policy_version=policy.version,
        tool_name=definition.name,
        permission=definition.permission,
        effect=definition.effect,
        decision=decision,
        reason=reason,
        policy_snapshot={
            "allowed_tools": policy.allowed_tools,
            "external_writes_require_approval": policy.external_writes_require_approval,
        },
    )
    session.add(authorization)
    session.commit()
    session.refresh(authorization)
    return AuthorizationDecision(authorization=authorization)
