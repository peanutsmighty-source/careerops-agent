from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.models import AgentTask


DEFAULT_COMPACTION_CHAR_THRESHOLD = 6000
DEFAULT_RECENT_OBSERVATION_TOKENS = 384


@dataclass(frozen=True)
class ContextCompactionResult:
    triggered: bool
    trigger_reason: str
    raw_context: dict
    compacted_context: dict
    retained_items: list[dict]
    removed_items: list[dict]
    raw_char_count: int
    compacted_char_count: int

    @property
    def model_observations(self) -> list[dict]:
        return self.compacted_context["observations"]

    @property
    def execution_context(self) -> dict:
        return self.compacted_context["execution_context"]

    def as_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "trigger_reason": self.trigger_reason,
            "raw_char_count": self.raw_char_count,
            "compacted_char_count": self.compacted_char_count,
            "raw_input": self.raw_context,
            "compacted_context": self.compacted_context,
            "retained_items": self.retained_items,
            "removed_items": self.removed_items,
        }


def compact_agent_context(
    task: AgentTask,
    *,
    run_id: int,
    step_number: int,
    max_steps: int,
    memory_context: dict,
    observations: list[dict],
    char_threshold: int = DEFAULT_COMPACTION_CHAR_THRESHOLD,
    recent_observation_tokens: int = DEFAULT_RECENT_OBSERVATION_TOKENS,
) -> ContextCompactionResult:
    if char_threshold < 0 or recent_observation_tokens < 1:
        raise ValueError(
            "char_threshold must be non-negative and recent_observation_tokens positive"
        )

    execution_context = _execution_context(
        task, run_id=run_id, step_number=step_number, max_steps=max_steps
    )
    raw_context = {
        "goal_contract": memory_context["goal_contract"],
        "task": {
            "id": task.id,
            "title": task.title,
            "user_goal": task.user_goal,
            "constraints": task.constraints,
            "success_criteria": task.success_criteria,
        },
        "execution_context": execution_context,
        "memory_context": memory_context,
        "observations": observations,
    }
    raw_char_count = _json_chars(raw_context)
    protected_items = _protected_items(raw_context)
    if raw_char_count <= char_threshold or not observations:
        return ContextCompactionResult(
            triggered=False,
            trigger_reason="below_char_threshold" if observations else "no_observations",
            raw_context=raw_context,
            compacted_context=raw_context,
            retained_items=[
                *protected_items,
                *[
                    {
                        "kind": "observation",
                        "index": index,
                        "reason": "below_char_threshold",
                    }
                    for index in range(len(observations))
                ],
            ],
            removed_items=[],
            raw_char_count=raw_char_count,
            compacted_char_count=raw_char_count,
        )

    messages = [
        HumanMessage(
            id=f"observation:{index}",
            content=json.dumps(observation, ensure_ascii=False, sort_keys=True),
        )
        for index, observation in enumerate(observations)
    ]
    trimmed = trim_messages(
        messages,
        max_tokens=recent_observation_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        allow_partial=False,
    )
    retained_indexes = {
        int(message.id.split(":", 1)[1])
        for message in trimmed
        if message.id and message.id.startswith("observation:")
    }
    removed_indexes = [
        index for index in range(len(observations)) if index not in retained_indexes
    ]
    compacted_observations = (
        [
            {
                "kind": "compacted_observation_summary",
                "summary": _summarize_observations(
                    [observations[index] for index in removed_indexes]
                ),
            }
        ]
        if removed_indexes
        else []
    )
    compacted_observations.extend(
        observations[index] for index in sorted(retained_indexes)
    )
    compacted_context = {
        **raw_context,
        "observations": compacted_observations,
    }
    return ContextCompactionResult(
        triggered=bool(removed_indexes),
        trigger_reason=(
            "char_threshold_exceeded" if removed_indexes else "nothing_trimmed"
        ),
        raw_context=raw_context,
        compacted_context=compacted_context,
        retained_items=[
            *protected_items,
            *[
                {
                    "kind": "observation",
                    "index": index,
                    "reason": "recent_observation",
                }
                for index in sorted(retained_indexes)
            ],
        ],
        removed_items=[
            {
                "kind": "observation",
                "index": index,
                "reason": "summarized_old_observation",
                "content": observations[index],
            }
            for index in removed_indexes
        ],
        raw_char_count=raw_char_count,
        compacted_char_count=_json_chars(compacted_context),
    )


def _execution_context(
    task: AgentTask, *, run_id: int, step_number: int, max_steps: int
) -> dict:
    completed_actions = [
        _plan_step_snapshot(step)
        for step in task.plan_steps
        if step.status == "completed"
    ]
    unresolved_blockers = [
        _plan_step_snapshot(step)
        for step in task.plan_steps
        if step.status == "blocked"
    ]
    next_step = next(
        (
            step
            for step in task.plan_steps
            if step.status in {"in_progress", "pending"}
        ),
        None,
    )
    return {
        "task_id": task.id,
        "run_id": run_id,
        "step_number": step_number,
        "max_steps": max_steps,
        "completed_actions": completed_actions,
        "unresolved_blockers": unresolved_blockers,
        "next_action": _plan_step_snapshot(next_step) if next_step else None,
    }


def _plan_step_snapshot(step) -> dict:
    return {
        "plan_step_id": step.id,
        "sequence": step.sequence,
        "title": step.title,
        "status": step.status,
        "success_criterion": step.success_criterion,
        "result_summary": step.result_summary,
    }


def _protected_items(raw_context: dict) -> list[dict]:
    return [
        {"kind": "goal_contract", "reason": "protected_goal_contract"},
        {"kind": "task_goal", "reason": "protected_user_goal"},
        {"kind": "constraints", "reason": "protected_constraints"},
        {"kind": "success_criteria", "reason": "protected_success_criteria"},
        {"kind": "completed_actions", "reason": "protected_execution_state"},
        {"kind": "unresolved_blockers", "reason": "protected_execution_state"},
        {"kind": "identifiers", "reason": "protected_runtime_identifiers"},
        {"kind": "next_action", "reason": "protected_next_action"},
        {"kind": "memory_context", "reason": "bounded_memory_runtime_output"},
    ]


def _summarize_observations(observations: list[dict]) -> dict:
    return {
        "observation_count": len(observations),
        "items": [
            {
                "tool_call_id": observation.get("tool_call_id"),
                "trace_id": observation.get("trace_id"),
                "plan_step_id": observation.get("plan_step_id"),
                "tool_name": observation.get("tool_name"),
                "status": observation.get("status"),
                "outcome_preview": _preview(
                    observation.get("output")
                    if observation.get("output") is not None
                    else observation.get("error")
                ),
            }
            for observation in observations
        ],
    }


def _preview(value, limit: int = 160) -> str | None:
    if value is None:
        return None
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}…"


def _json_chars(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))
