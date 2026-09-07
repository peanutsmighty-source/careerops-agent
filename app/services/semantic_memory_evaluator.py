from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from app.models import AgentMemory


class SemanticMemoryJudgment(BaseModel):
    decision: Literal["accept", "reject", "needs_review"]
    memory_type: Literal["working", "episodic", "fact"]
    long_term_value: Literal["low", "medium", "high"]
    semantic_duplicate_memory_id: int | None = Field(default=None, ge=1)
    conflict_memory_ids: list[int] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=500)


@dataclass(frozen=True)
class SemanticEvaluationResult:
    judgment: SemanticMemoryJudgment
    evaluator_version: str
    usage: dict[str, int]


class SemanticMemoryEvaluator(Protocol):
    evaluator_version: str

    def evaluate(
        self,
        *,
        candidate: dict,
        existing_memories: Sequence[dict],
    ) -> SemanticEvaluationResult: ...


class OpenAISemanticMemoryEvaluator:
    """Optional OpenAI adapter; the deterministic gate remains the final authority."""

    def __init__(self, model: str = "gpt-5-mini") -> None:
        from openai import OpenAI

        self.model = model
        self.evaluator_version = f"openai:{self.model}"
        self._client = OpenAI()

    def evaluate(
        self,
        *,
        candidate: dict,
        existing_memories: Sequence[dict],
    ) -> SemanticEvaluationResult:
        response = self._client.responses.parse(
            model=self.model,
            instructions=(
                "You evaluate a proposed CareerOps memory. Classify its memory type and "
                "long-term value. Compare it only with the supplied existing memories. "
                "Never invent memory IDs. Reject low-value or semantically duplicate content. "
                "Use needs_review for plausible conflicts or uncertainty."
            ),
            input=json.dumps(
                {"candidate": candidate, "existing_memories": list(existing_memories)},
                ensure_ascii=False,
            ),
            text_format=SemanticMemoryJudgment,
        )
        judgment = response.output_parsed
        if judgment is None:
            raise ValueError("semantic evaluator returned no structured judgment")
        usage = response.usage
        return SemanticEvaluationResult(
            judgment=judgment,
            evaluator_version=self.evaluator_version,
            usage={
                "model_calls": 1,
                "prompt_tokens": usage.input_tokens if usage else 0,
                "completion_tokens": usage.output_tokens if usage else 0,
            },
        )


def memory_context(memory: AgentMemory) -> dict:
    return {
        "id": memory.id,
        "memory_type": memory.memory_type,
        "scope_type": memory.scope_type,
        "content": memory.content,
        "source": memory.source,
    }
