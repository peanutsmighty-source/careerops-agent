from __future__ import annotations

import os
import sqlite3
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.database import SessionLocal
from app.models import AgentTask
from app.services.analytics import skill_demand_rows
from app.services.roadmap import build_learning_roadmap


class CareerOpsState(TypedDict, total=False):
    task_id: int
    user_goal: str
    success_criteria: list[str]
    top_skills: list[dict]
    learning_plan: list[dict]
    validation_errors: list[str]
    approved: bool | None
    approval_comment: str | None
    status: str
    node_history: list[str]


def load_skill_demand(state: CareerOpsState) -> CareerOpsState:
    """Node 1: read deterministic skill demand from the CareerOps database."""
    with SessionLocal() as session:
        top_skills = skill_demand_rows(session)[:3]
    return {
        "top_skills": top_skills,
        "status": "skills_loaded",
        "node_history": [*state.get("node_history", []), "load_skill_demand"],
    }


def build_plan(state: CareerOpsState) -> CareerOpsState:
    """Node 2: turn the selected skills into a deterministic learning plan."""
    learning_plan = build_learning_roadmap(state.get("top_skills", []), limit=3)
    return {
        "learning_plan": learning_plan,
        "status": "plan_built",
        "node_history": [*state.get("node_history", []), "build_plan"],
    }


def validate_plan(state: CareerOpsState) -> CareerOpsState:
    """Node 3: decide whether the graph may complete or must take the blocked edge."""
    errors: list[str] = []
    if not state.get("top_skills"):
        errors.append("No parsed JD skills are available.")
    if not state.get("learning_plan"):
        errors.append("No learning plan was generated.")
    if not state.get("success_criteria"):
        errors.append("The Agent task has no success criteria.")
    return {
        "validation_errors": errors,
        "status": "plan_validated" if not errors else "validation_failed",
        "node_history": [*state.get("node_history", []), "validate_plan"],
    }


def complete_run(state: CareerOpsState) -> CareerOpsState:
    """Node 4a: terminal node for a valid plan."""
    return {
        "status": "completed",
        "node_history": [*state.get("node_history", []), "complete_run"],
    }


def human_review(state: CareerOpsState) -> CareerOpsState:
    """Node 4: pause until a user approves or rejects the generated learning plan."""
    decision = interrupt(
        {
            "question": "Do you approve this JD-driven learning plan?",
            "task_id": state["task_id"],
            "learning_plan": state.get("learning_plan", []),
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    comment = decision.get("comment") if isinstance(decision, dict) else None
    return {
        "approved": approved,
        "approval_comment": comment,
        "status": "approved" if approved else "rejected",
        "node_history": [*state.get("node_history", []), "human_review"],
    }


def block_run(state: CareerOpsState) -> CareerOpsState:
    """Node 4b: terminal node reached through the validation failure edge."""
    return {
        "status": "blocked",
        "node_history": [*state.get("node_history", []), "block_run"],
    }


def route_after_validation(state: CareerOpsState) -> str:
    return "block_run" if state.get("validation_errors") else "human_review"


def route_after_review(state: CareerOpsState) -> str:
    return "complete_run" if state.get("approved") else "block_run"


def build_graph(checkpointer: SqliteSaver):
    builder = StateGraph(CareerOpsState)
    builder.add_node("load_skill_demand", load_skill_demand)
    builder.add_node("build_plan", build_plan)
    builder.add_node("validate_plan", validate_plan)
    builder.add_node("human_review", human_review)
    builder.add_node("complete_run", complete_run)
    builder.add_node("block_run", block_run)

    builder.add_edge(START, "load_skill_demand")
    builder.add_edge("load_skill_demand", "build_plan")
    builder.add_edge("build_plan", "validate_plan")
    builder.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {"human_review": "human_review", "block_run": "block_run"},
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"complete_run": "complete_run", "block_run": "block_run"},
    )
    builder.add_edge("complete_run", END)
    builder.add_edge("block_run", END)
    return builder.compile(checkpointer=checkpointer)


CHECKPOINT_DB_PATH = os.getenv("CAREEROPS_CHECKPOINT_DB", "careerops_checkpoints.db")
_checkpoint_connection = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
checkpoint_saver = SqliteSaver(_checkpoint_connection)
learning_graph = build_graph(checkpoint_saver)


def graph_thread_id(task: AgentTask) -> str:
    created_at = task.created_at.isoformat() if task.created_at else "new"
    return f"careerops-task-{task.id}-{created_at}"


def _graph_config(task: AgentTask) -> dict:
    return {"configurable": {"thread_id": graph_thread_id(task)}}


def _interrupt_payload(snapshot) -> dict | None:
    if not snapshot.interrupts:
        return None
    value = snapshot.interrupts[0].value
    return value if isinstance(value, dict) else {"value": value}


def _graph_result(task: AgentTask) -> dict:
    snapshot = learning_graph.get_state(_graph_config(task))
    return {
        "thread_id": graph_thread_id(task),
        "state": dict(snapshot.values),
        "awaiting_approval": bool(snapshot.interrupts),
        "interrupt": _interrupt_payload(snapshot),
        "next_nodes": list(snapshot.next),
    }


def start_learning_graph(task: AgentTask) -> dict:
    config = _graph_config(task)
    current = learning_graph.get_state(config)
    if current.values:
        return _graph_result(task)
    learning_graph.invoke(
        {
            "task_id": task.id,
            "user_goal": task.user_goal,
            "success_criteria": task.success_criteria,
            "top_skills": [],
            "learning_plan": [],
            "validation_errors": [],
            "approved": None,
            "approval_comment": None,
            "status": "started",
            "node_history": [],
        },
        config=config,
    )
    return _graph_result(task)


def resume_learning_graph(task: AgentTask, *, approved: bool, comment: str | None) -> dict:
    config = _graph_config(task)
    current = learning_graph.get_state(config)
    if not current.interrupts:
        raise ValueError("learning graph is not waiting for approval")
    learning_graph.invoke(
        Command(resume={"approved": approved, "comment": comment}),
        config=config,
    )
    return _graph_result(task)


def checkpoint_history(task: AgentTask) -> list[dict]:
    config = _graph_config(task)
    snapshots = learning_graph.get_state_history(config)
    return [
        {
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "step": snapshot.metadata.get("step"),
            "next_nodes": list(snapshot.next),
            "state": dict(snapshot.values),
        }
        for snapshot in snapshots
    ]
