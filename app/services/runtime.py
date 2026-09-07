from __future__ import annotations

from dataclasses import dataclass

from app.models import AgentCapability, AgentTask


@dataclass(frozen=True)
class PlanValidation:
    notes: list[str]

    @property
    def is_valid(self) -> bool:
        return not self.notes


def validate_plan_step(
    task: AgentTask,
    *,
    agent_capability: str,
    success_criterion: str,
    goal_connection: str,
    learning_outcome: str,
) -> PlanValidation:
    """Require each runtime plan step to map back to the user's task contract."""
    notes: list[str] = []
    if agent_capability not in {capability.value for capability in AgentCapability}:
        notes.append("The plan step does not name a supported Agent capability.")
    if success_criterion not in task.success_criteria:
        notes.append("The plan step is not linked to a success criterion from this user task.")
    if not goal_connection.strip():
        notes.append("The plan step does not explain how it advances the user goal.")
    if not learning_outcome.strip():
        notes.append("The plan step does not state a learning outcome.")
    return PlanValidation(notes=notes)
