from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from app.models import AgentMemory


DEFAULT_SIMILARITY_LIMIT = 8
EMBEDDING_CANDIDATE_THRESHOLD = 0.55
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    provider_version: str
    usage: dict[str, int]


class MemoryEmbeddingProvider(Protocol):
    provider_version: str

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


@dataclass(frozen=True)
class SimilarityCandidate:
    memory: AgentMemory
    score: float


@dataclass(frozen=True)
class SimilaritySearchResult:
    candidates: tuple[SimilarityCandidate, ...]
    method: str
    provider_version: str | None
    usage: dict[str, int]

    def contexts(self) -> list[dict]:
        return [
            {
                "id": item.memory.id,
                "memory_type": item.memory.memory_type,
                "scope_type": item.memory.scope_type,
                "content": item.memory.content,
                "source": item.memory.source,
                "similarity_score": round(item.score, 6),
            }
            for item in self.candidates
        ]


class OpenAIMemoryEmbeddingProvider:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.model = model
        self.provider_version = f"openai:{model}"
        self._client = OpenAI()

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return EmbeddingBatch(
            vectors=[item.embedding for item in response.data],
            provider_version=self.provider_version,
            usage={
                "embedding_calls": 1,
                "embedding_tokens": response.usage.total_tokens,
            },
        )


def rank_similar_memories(
    candidate_content: str,
    memories: Sequence[AgentMemory],
    *,
    embedding_provider: MemoryEmbeddingProvider | None = None,
    limit: int = DEFAULT_SIMILARITY_LIMIT,
) -> SimilaritySearchResult:
    if limit < 1:
        raise ValueError("similarity limit must be positive")
    if not memories:
        return SimilaritySearchResult((), "none", None, _zero_usage())

    if embedding_provider is None:
        candidate_tokens = _tokens(candidate_content)
        ranked = sorted(
            (
                SimilarityCandidate(memory, _jaccard(candidate_tokens, _tokens(memory.content)))
                for memory in memories
            ),
            key=lambda item: (-item.score, item.memory.id),
        )
        selected = tuple(item for item in ranked if item.score > 0)[:limit]
        return SimilaritySearchResult(selected, "lexical", None, _zero_usage())

    batch = embedding_provider.embed(
        [candidate_content, *(memory.content for memory in memories)]
    )
    if len(batch.vectors) != len(memories) + 1:
        raise ValueError("embedding provider returned the wrong number of vectors")
    candidate_vector = batch.vectors[0]
    ranked = sorted(
        (
            SimilarityCandidate(memory, _cosine(candidate_vector, vector))
            for memory, vector in zip(memories, batch.vectors[1:], strict=True)
        ),
        key=lambda item: (-item.score, item.memory.id),
    )
    selected = tuple(
        item for item in ranked if item.score >= EMBEDDING_CANDIDATE_THRESHOLD
    )[:limit]
    return SimilaritySearchResult(
        selected,
        "embedding",
        batch.provider_version,
        batch.usage,
    )


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(value)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must have equal non-zero dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("embedding vectors must have non-zero norms")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _zero_usage() -> dict[str, int]:
    return {"embedding_calls": 0, "embedding_tokens": 0}
