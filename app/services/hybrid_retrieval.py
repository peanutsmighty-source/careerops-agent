from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from app.services.memory_similarity import MemoryEmbeddingProvider


RRF_K = 60
SEMANTIC_THRESHOLD = 0.55
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)


@dataclass(frozen=True)
class RetrievalDocument:
    key: str
    source_type: str
    content: str
    payload: dict
    prior_score: float = 0.0


@dataclass(frozen=True)
class RankedRetrievalDocument:
    document: RetrievalDocument
    lexical_score: float
    semantic_score: float | None
    fusion_score: float


@dataclass(frozen=True)
class HybridRetrievalResult:
    candidates: tuple[RankedRetrievalDocument, ...]
    method: str
    provider_version: str | None
    usage: dict[str, int]
    embedding_error: str | None = None


def rank_hybrid_documents(
    query: str,
    documents: Sequence[RetrievalDocument],
    *,
    embedding_provider: MemoryEmbeddingProvider | None = None,
    limit: int = 20,
) -> HybridRetrievalResult:
    """Fuse lexical, optional semantic, and deterministic policy-prior ranks."""
    if limit < 1:
        raise ValueError("retrieval limit must be positive")
    if not documents:
        return HybridRetrievalResult((), "none", None, _zero_usage())

    query_tokens = _tokens(query)
    lexical_scores = {
        document.key: _jaccard(query_tokens, _tokens(document.content))
        for document in documents
    }
    lexical_rank = [
        item.key
        for item in sorted(
            documents,
            key=lambda item: (-lexical_scores[item.key], item.key),
        )
        if lexical_scores[item.key] > 0
    ]
    prior_rank = [
        item.key
        for item in sorted(documents, key=lambda item: (-item.prior_score, item.key))
        if item.prior_score > 0
    ]

    semantic_scores: dict[str, float] = {}
    semantic_rank: list[str] = []
    provider_version = None
    usage = _zero_usage()
    embedding_error = None
    if embedding_provider is not None:
        provider_version = embedding_provider.provider_version
        try:
            batch = embedding_provider.embed([query, *(item.content for item in documents)])
            if len(batch.vectors) != len(documents) + 1:
                raise ValueError("embedding provider returned the wrong number of vectors")
            semantic_scores = {
                document.key: _cosine(batch.vectors[0], vector)
                for document, vector in zip(
                    documents, batch.vectors[1:], strict=True
                )
            }
            semantic_rank = [
                item.key
                for item in sorted(
                    documents,
                    key=lambda item: (-semantic_scores[item.key], item.key),
                )
                if semantic_scores[item.key] >= SEMANTIC_THRESHOLD
            ]
            usage = batch.usage
        except Exception as exc:
            embedding_error = type(exc).__name__

    fused: dict[str, float] = {}
    for ranking in (lexical_rank, semantic_rank, prior_rank):
        for position, key in enumerate(ranking, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + position)

    by_key = {document.key: document for document in documents}
    ranked = sorted(fused, key=lambda key: (-fused[key], key))[:limit]
    method = "hybrid" if embedding_provider is not None else "lexical"
    if embedding_error:
        method = "lexical_fallback"
    return HybridRetrievalResult(
        candidates=tuple(
            RankedRetrievalDocument(
                document=by_key[key],
                lexical_score=lexical_scores[key],
                semantic_score=semantic_scores.get(key),
                fusion_score=fused[key],
            )
            for key in ranked
        ),
        method=method,
        provider_version=provider_version,
        usage=usage,
        embedding_error=embedding_error,
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
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _zero_usage() -> dict[str, int]:
    return {"embedding_calls": 0, "embedding_tokens": 0}
