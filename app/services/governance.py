from __future__ import annotations

from dataclasses import dataclass

from app.models import AgentCapability, AlignmentDecision, GoalContract


DEFAULT_GOAL_CONTRACT = {
    "north_star_goal": "Design and implement a runnable, observable, and evaluable CareerOps AI Agent Harness.",
    "product_context": (
        "CareerOps is the real-world setting: it uses job descriptions, learning evidence, "
        "and career goals to help a user prepare for AI Agent and Harness roles."
    ),
    "learning_contract": (
        "Every development phase must explain the relevant Agent concept, engineering tradeoff, "
        "reproducible exercise, and observed difficulty so the user learns by building."
    ),
    "scope_guardrails": [
        "Prioritize capabilities that make CareerOps a reliable Agent Harness.",
        "Build product CRUD only when it enables an Agent tool, memory, retrieval, evaluation, or observable state.",
        "Do not let a local implementation task replace the north-star goal.",
    ],
    "success_criteria": [
        "A user goal can be planned, executed with tools, and verified through an observable trace.",
        "The agent preserves explicit goals and constraints across long-running work.",
        "Each implemented capability includes a reproducible learning exercise and automated validation.",
    ],
}


@dataclass(frozen=True)
class AlignmentEvaluation:
    decision: AlignmentDecision
    notes: list[str]


def evaluate_alignment(
    contract: GoalContract,
    *,
    agent_capability: str,
    success_criterion: str,
    goal_connection: str,
    learning_outcome: str,
) -> AlignmentEvaluation:
    """Make the required goal mapping explicit before an Agent takes a development action."""
    notes: list[str] = []
    if agent_capability not in {capability.value for capability in AgentCapability}:
        notes.append("The proposed work does not name a supported Agent capability.")
    if success_criterion not in contract.success_criteria:
        notes.append("The proposed work is not linked to an active goal-contract success criterion.")
    if not goal_connection.strip():
        notes.append("The proposed work does not explain its connection to the north-star goal.")
    if not learning_outcome.strip():
        notes.append("The proposed work does not state what the user will learn or reproduce.")

    if notes:
        return AlignmentEvaluation(decision=AlignmentDecision.DEFER, notes=notes)

    return AlignmentEvaluation(
        decision=AlignmentDecision.PROCEED,
        notes=["The action is linked to an Agent capability, success criterion, and learning outcome."],
    )
