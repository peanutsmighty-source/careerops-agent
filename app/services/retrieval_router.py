from __future__ import annotations

import re
from dataclasses import dataclass


MAX_IDENTIFIER_LOOKUP_TOKENS = 8
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(
    r"(?:\b(?:tool\s*call|toolcall|agent\s*run|run|task|memory|trace|job)"
    r"\s*[#:]?\s*\d+\b|(?:工具调用|运行|任务|记忆|追踪|岗位)\s*[#:]?\s*\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RetrievalRouteDecision:
    strategy: str
    reason: str
    signals: tuple[str, ...]


def choose_retrieval_route(
    query: str, *, embedding_available: bool
) -> RetrievalRouteDecision:
    """Choose a conservative retrieval route with an auditable reason."""
    if not embedding_available:
        return RetrievalRouteDecision(
            strategy="lexical",
            reason="embedding_provider_unavailable",
            signals=(),
        )

    identifiers = tuple(match.group(0) for match in IDENTIFIER_PATTERN.finditer(query))
    token_count = len(TOKEN_PATTERN.findall(query))
    if identifiers and token_count <= MAX_IDENTIFIER_LOOKUP_TOKENS:
        return RetrievalRouteDecision(
            strategy="lexical",
            reason="short_exact_identifier_lookup",
            signals=identifiers,
        )

    return RetrievalRouteDecision(
        strategy="hybrid",
        reason="default_recall_route",
        signals=identifiers,
    )
