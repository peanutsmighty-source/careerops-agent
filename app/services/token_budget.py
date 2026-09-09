from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately


TOKEN_ESTIMATOR = "langchain_count_tokens_approximately_v1"


def estimate_tokens(value) -> int:
    """Estimate serialized input tokens without calling an external model."""
    if value in (None, "", [], {}):
        return 0
    content = value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True
    )
    return count_tokens_approximately([HumanMessage(content=content)])


@dataclass(frozen=True)
class ContextTokenBudget:
    model_context_tokens: int
    reserved_output_tokens: int
    input_token_budget: int
    prompt_tokens: int
    memory_tokens: int
    tool_tokens: int
    observation_tokens: int
    estimated_input_tokens: int
    remaining_input_tokens: int
    within_budget: bool
    compaction_input_tokens: int
    compaction_output_tokens: int

    @property
    def compaction_saved_tokens(self) -> int:
        return max(self.compaction_input_tokens - self.compaction_output_tokens, 0)

    def as_dict(self) -> dict:
        return {
            "estimator": TOKEN_ESTIMATOR,
            "model_context_tokens": self.model_context_tokens,
            "reserved_output_tokens": self.reserved_output_tokens,
            "input_token_budget": self.input_token_budget,
            "estimated_input_tokens": self.estimated_input_tokens,
            "remaining_input_tokens": self.remaining_input_tokens,
            "within_budget": self.within_budget,
            "sections": {
                "prompt_tokens": self.prompt_tokens,
                "memory_tokens": self.memory_tokens,
                "tool_tokens": self.tool_tokens,
                "observation_tokens": self.observation_tokens,
            },
            "compaction": {
                "input_tokens": self.compaction_input_tokens,
                "output_tokens": self.compaction_output_tokens,
                "saved_tokens": self.compaction_saved_tokens,
            },
        }


def measure_context_budget(
    *,
    model_context_tokens: int,
    reserved_output_tokens: int,
    prompt,
    memory_context: dict,
    tools: list[dict],
    observations: list[dict],
    raw_observations: list[dict] | None = None,
) -> ContextTokenBudget:
    if model_context_tokens < 1:
        raise ValueError("model_context_tokens must be positive")
    if reserved_output_tokens < 1 or reserved_output_tokens >= model_context_tokens:
        raise ValueError(
            "reserved_output_tokens must be positive and smaller than the context budget"
        )
    input_token_budget = model_context_tokens - reserved_output_tokens
    prompt_tokens = estimate_tokens(prompt)
    memory_tokens = estimate_tokens(memory_context)
    tool_tokens = estimate_tokens(tools)
    observation_tokens = estimate_tokens(observations)
    estimated_input_tokens = (
        prompt_tokens + memory_tokens + tool_tokens + observation_tokens
    )
    compaction_input_tokens = estimate_tokens(
        observations if raw_observations is None else raw_observations
    )
    return ContextTokenBudget(
        model_context_tokens=model_context_tokens,
        reserved_output_tokens=reserved_output_tokens,
        input_token_budget=input_token_budget,
        prompt_tokens=prompt_tokens,
        memory_tokens=memory_tokens,
        tool_tokens=tool_tokens,
        observation_tokens=observation_tokens,
        estimated_input_tokens=estimated_input_tokens,
        remaining_input_tokens=max(input_token_budget - estimated_input_tokens, 0),
        within_budget=estimated_input_tokens <= input_token_budget,
        compaction_input_tokens=compaction_input_tokens,
        compaction_output_tokens=observation_tokens,
    )
