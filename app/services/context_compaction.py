from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.models import AgentTask
from app.services.token_budget import estimate_tokens


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
    raw_token_count: int
    compacted_token_count: int
    raw_observation_tokens: int
    observation_token_budget: int

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
            "raw_token_count": self.raw_token_count,
            "compacted_token_count": self.compacted_token_count,
            "raw_observation_tokens": self.raw_observation_tokens,
            "observation_token_budget": self.observation_token_budget,
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
    observation_token_budget: int = DEFAULT_RECENT_OBSERVATION_TOKENS,
) -> ContextCompactionResult:
    if observation_token_budget < 0:
        raise ValueError("observation_token_budget cannot be negative")

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
    raw_token_count = estimate_tokens(raw_context)
    raw_observation_tokens = estimate_tokens(observations)
    protected_items = _protected_items(raw_context)
    if raw_observation_tokens <= observation_token_budget or not observations:
        return ContextCompactionResult(
            triggered=False,
            trigger_reason="within_observation_token_budget" if observations else "no_observations",
            raw_context=raw_context,
            compacted_context=raw_context,
            retained_items=[
                *protected_items,
                *[
                    {
                        "kind": "observation",
                        "index": index,
                        "reason": "within_observation_token_budget",
                    }
                    for index in range(len(observations))
                ],
            ],
            removed_items=[],
            raw_char_count=raw_char_count,
            compacted_char_count=raw_char_count,
            raw_token_count=raw_token_count,
            compacted_token_count=raw_token_count,
            raw_observation_tokens=raw_observation_tokens,
            observation_token_budget=observation_token_budget,
        )

    compacted_observations, retained_indexes, removed_indexes = _compact_observations(
        observations, observation_token_budget
    )
    compacted_context = {
        **raw_context,
        "observations": compacted_observations,
    }
    summary_created = bool(
        compacted_observations
        and compacted_observations[0].get("kind") == "compacted_observation_summary"
    )
    return ContextCompactionResult(
        triggered=True,
        trigger_reason="observation_token_budget_exceeded",
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
                for index in retained_indexes
            ],
        ],
        removed_items=[
            {
                "kind": "observation",
                "index": index,
                "reason": (
                    "summarized_old_observation"
                    if summary_created
                    else "removed_to_fit_observation_budget"
                ),
                "content": observations[index],
            }
            for index in removed_indexes
        ],
        raw_char_count=raw_char_count,
        compacted_char_count=_json_chars(compacted_context),
        raw_token_count=raw_token_count,
        compacted_token_count=estimate_tokens(compacted_context),
        raw_observation_tokens=raw_observation_tokens,
        observation_token_budget=observation_token_budget,
    )


def build_execution_context(
    task: AgentTask, *, run_id: int, step_number: int, max_steps: int
) -> dict:
    return _execution_context(
        task, run_id=run_id, step_number=step_number, max_steps=max_steps
    )


def _compact_observations(
    observations: list[dict], token_budget: int
) -> tuple[list[dict], list[int], list[int]]:
    messages = [
        HumanMessage(
            id=f"observation:{index}",
            content=json.dumps(observation, ensure_ascii=False, sort_keys=True),
        )
        for index, observation in enumerate(observations)
    ]
    trimmed = (
        trim_messages(
            messages,
            max_tokens=token_budget,
            token_counter=count_tokens_approximately,
            strategy="last",
            allow_partial=False,
        )
        if token_budget
        else []
    )
    retained_indexes = sorted({
        int(message.id.split(":", 1)[1])
        for message in trimmed
        if message.id and message.id.startswith("observation:")
    })
    removed_indexes = sorted(
        index for index in range(len(observations)) if index not in retained_indexes
    )
    include_preview = True
    summary_item_limit = len(removed_indexes)

    while True:
        compacted = _build_compacted_observations(
            observations,
            retained_indexes,
            removed_indexes,
            include_preview=include_preview,
            summary_item_limit=summary_item_limit,
        )
        if estimate_tokens(compacted) <= token_budget:
            return compacted, retained_indexes, removed_indexes
        if retained_indexes:
            removed_indexes.append(retained_indexes.pop(0))
            removed_indexes.sort()
            summary_item_limit = len(removed_indexes)
            continue
        if include_preview:
            include_preview = False
            continue
        if summary_item_limit:
            summary_item_limit -= 1
            continue
        return [], retained_indexes, removed_indexes


def _build_compacted_observations(
    observations: list[dict],
    retained_indexes: list[int],
    removed_indexes: list[int],
    *,
    include_preview: bool,
    summary_item_limit: int,
) -> list[dict]:
    compacted = []
    if removed_indexes:
        compacted.append(
            {
                "kind": "compacted_observation_summary",
                "summary": _summarize_observations(
                    [observations[index] for index in removed_indexes],
                    include_preview=include_preview,
                    item_limit=summary_item_limit,
                ),
            }
        )
    compacted.extend(observations[index] for index in retained_indexes)
    return compacted


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


def _summarize_observations(
    observations: list[dict], *, include_preview: bool, item_limit: int
) -> dict:
    summarized = observations[-item_limit:] if item_limit else []
    return {
        "observation_count": len(observations),
        "omitted_detail_count": len(observations) - len(summarized),
        "items": [
            {
                "tool_call_id": observation.get("tool_call_id"),
                "trace_id": observation.get("trace_id"),
                "plan_step_id": observation.get("plan_step_id"),
                "tool_name": observation.get("tool_name"),
                "status": observation.get("status"),
                **(
                    {
                        "outcome_preview": _preview(
                            observation.get("output")
                            if observation.get("output") is not None
                            else observation.get("error")
                        )
                    }
                    if include_preview
                    else {}
                ),
            }
            for observation in summarized
        ],
    }


def _preview(value, limit: int = 160) -> str | None:
    if value is None:
        return None
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}…"


def _json_chars(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))
