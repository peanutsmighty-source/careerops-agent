from __future__ import annotations

import hashlib
import re
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import AgentTask, ExecutionTrace
from app.services.memory_evaluator import (
    MemoryCandidate,
    MemoryEvaluation,
    evaluate_and_store_candidate,
)


BUILDER_VERSION = "explicit-free-text-v1"
SENTENCE_SPLIT = re.compile(r"(?:[\n。！？!?;；]+|(?<!\d)\.(?:\s+|$))")
WHITESPACE = re.compile(r"\s+")


class FreeTextCandidateProposal(BaseModel):
    category: Literal["preference", "fact", "correction", "episode"]
    memory_type: Literal["fact", "episodic"]
    memory_key: str
    content: str
    evidence_text: str
    extraction_rule: str
    confidence: float


PATTERNS = (
    (
        "correction",
        "fact",
        "english_correction",
        re.compile(r"^correction\s*:\s*(.{3,200})$", re.IGNORECASE),
    ),
    (
        "correction",
        "fact",
        "chinese_correction",
        re.compile(r"^(?:更正|纠正)[：:]\s*(.{2,200})$"),
    ),
    (
        "preference",
        "fact",
        "english_preference",
        re.compile(r"^I\s+prefer\s+(.{3,200})$", re.IGNORECASE),
    ),
    (
        "preference",
        "fact",
        "chinese_preference",
        re.compile(r"^(?:我偏好|我更喜欢)(.{2,200})$"),
    ),
    (
        "fact",
        "fact",
        "english_target_role",
        re.compile(r"^my\s+(?:target|desired)\s+role\s+is\s+(.{2,120})$", re.IGNORECASE),
    ),
    (
        "fact",
        "fact",
        "chinese_target_role",
        re.compile(r"^我的目标(?:岗位|职位)是(.{2,120})$"),
    ),
    (
        "episode",
        "episodic",
        "english_learning_episode",
        re.compile(r"^I\s+(?:learned|discovered)\s+that\s+(.{3,200})$", re.IGNORECASE),
    ),
    (
        "episode",
        "episodic",
        "chinese_learning_episode",
        re.compile(r"^我(?:学到|发现)(?:了|的是|[：:])?\s*(.{3,200})$"),
    ),
)


def extract_free_text_candidate_proposals(text: str) -> tuple[FreeTextCandidateProposal, ...]:
    proposals: list[FreeTextCandidateProposal] = []
    seen: set[tuple[str, str]] = set()
    for raw_sentence in SENTENCE_SPLIT.split(text):
        sentence = WHITESPACE.sub(" ", raw_sentence).strip(" .，,")
        if not sentence:
            continue
        for category, memory_type, rule, pattern in PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            detail = WHITESPACE.sub(" ", match.group(1)).strip(" .，,")
            identity = (category, detail.casefold())
            if identity in seen:
                break
            seen.add(identity)
            proposals.append(
                FreeTextCandidateProposal(
                    category=category,
                    memory_type=memory_type,
                    memory_key=_memory_key(category, detail, rule),
                    content=_candidate_content(category, detail),
                    evidence_text=sentence,
                    extraction_rule=rule,
                    confidence=1.0,
                )
            )
            break
    return tuple(proposals)


def evaluate_free_text_candidates(
    session: Session,
    task: AgentTask,
    trace: ExecutionTrace,
    *,
    text: str,
) -> tuple[MemoryEvaluation, ...]:
    proposals = extract_free_text_candidate_proposals(text)
    evaluations = []
    for proposal in proposals:
        candidate = MemoryCandidate(
            memory_type=proposal.memory_type,
            scope_type="task",
            task_id=task.id,
            run_id=None,
            memory_key=proposal.memory_key,
            content=proposal.content,
            source="user_input",
            importance=5 if proposal.category != "episode" else 3,
            relevance_text=proposal.content,
            provenance={
                "execution_trace_id": trace.id,
                "task_id": task.id,
                "evidence_text": proposal.evidence_text,
                "candidate_builder": BUILDER_VERSION,
                "candidate_category": proposal.category,
                "extraction_rule": proposal.extraction_rule,
                "confidence": proposal.confidence,
            },
        )
        evaluations.append(evaluate_and_store_candidate(session, task, candidate))
    return tuple(evaluations)


def _memory_key(category: str, detail: str, rule: str) -> str:
    target_role_detail = re.match(
        r"^(?:my\s+(?:target|desired)\s+role\s+is|我的目标(?:岗位|职位)是)",
        detail,
        re.IGNORECASE,
    )
    if (
        category == "fact"
        and rule in {"english_target_role", "chinese_target_role"}
    ) or (category == "correction" and target_role_detail):
        return "user-target-role"
    digest = hashlib.sha256(detail.casefold().encode("utf-8")).hexdigest()[:12]
    return f"user-{category}:{digest}"


def _candidate_content(category: str, detail: str) -> str:
    labels = {
        "preference": "User preference",
        "fact": "User-stated fact",
        "correction": "User correction",
        "episode": "User-reported learning episode",
    }
    return f"{labels[category]}: {detail}"
