from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import AgentRun, AgentTask
from app.services.agent_loop import (
    AgentLoopEngine,
    AgentModel,
    create_agent_model,
    create_agent_run,
    get_agent_run,
)
from app.services.learning_graph import checkpoint_saver


class AgentWorkflowState(TypedDict, total=False):
    task_id: int
    run_id: int
    provider: str
    model: str
    max_steps: int
    context_ready: bool
    agent_status: str
    final_answer: str | None
    stop_reason: str | None
    evaluation_errors: list[str]
    workflow_status: str
    node_history: list[str]


def load_context_node(state: AgentWorkflowState) -> AgentWorkflowState:
    return {
        "context_ready": True,
        "workflow_status": "context_loaded",
        "node_history": [*state.get("node_history", []), "load_context"],
    }


def _run_agent_node(model: AgentModel):
    def run_agent(state: AgentWorkflowState) -> AgentWorkflowState:
        AgentLoopEngine(model).run(state["run_id"])
        with SessionLocal() as session:
            run = get_agent_run(session, state["run_id"])
            return {
                "agent_status": run.status,
                "final_answer": run.final_answer,
                "stop_reason": run.stop_reason,
                "workflow_status": "agent_finished",
                "node_history": [*state.get("node_history", []), "run_agent"],
            }

    return run_agent


def evaluate_result_node(state: AgentWorkflowState) -> AgentWorkflowState:
    errors: list[str] = []
    if state.get("agent_status") != "completed":
        errors.append(f"Agent stopped with status {state.get('agent_status')}.")
    if not state.get("final_answer"):
        errors.append("Agent did not return a final answer.")
    return {
        "evaluation_errors": errors,
        "workflow_status": "evaluation_failed" if errors else "evaluation_passed",
        "node_history": [*state.get("node_history", []), "evaluate_result"],
    }


def complete_workflow_node(state: AgentWorkflowState) -> AgentWorkflowState:
    return {
        "workflow_status": "completed",
        "node_history": [*state.get("node_history", []), "complete_workflow"],
    }


def block_workflow_node(state: AgentWorkflowState) -> AgentWorkflowState:
    return {
        "workflow_status": "blocked",
        "node_history": [*state.get("node_history", []), "block_workflow"],
    }


def route_after_evaluation(state: AgentWorkflowState) -> str:
    return "block_workflow" if state.get("evaluation_errors") else "complete_workflow"


def build_agent_workflow(model: AgentModel):
    builder = StateGraph(AgentWorkflowState)
    builder.add_node("load_context", load_context_node)
    builder.add_node("run_agent", _run_agent_node(model))
    builder.add_node("evaluate_result", evaluate_result_node)
    builder.add_node("complete_workflow", complete_workflow_node)
    builder.add_node("block_workflow", block_workflow_node)
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "run_agent")
    builder.add_edge("run_agent", "evaluate_result")
    builder.add_conditional_edges(
        "evaluate_result",
        route_after_evaluation,
        {
            "complete_workflow": "complete_workflow",
            "block_workflow": "block_workflow",
        },
    )
    builder.add_edge("complete_workflow", END)
    builder.add_edge("block_workflow", END)
    return builder.compile(checkpointer=checkpoint_saver)


def agent_workflow_thread_id(run: AgentRun) -> str:
    created_at = run.created_at.isoformat() if run.created_at else "new"
    return f"careerops-agent-workflow-{run.id}-{created_at}"


def start_agent_workflow(
    session: Session,
    *,
    task: AgentTask,
    provider: str,
    model_name: str | None,
    max_steps: int,
    model_override: AgentModel | None = None,
) -> AgentRun:
    model = model_override or create_agent_model(provider, model_name)
    run = create_agent_run(session, task=task, model=model, max_steps=max_steps)
    graph = build_agent_workflow(model)
    graph.invoke(
        {
            "task_id": task.id,
            "run_id": run.id,
            "provider": model.provider,
            "model": model.model,
            "max_steps": max_steps,
            "context_ready": False,
            "agent_status": "running",
            "final_answer": None,
            "stop_reason": None,
            "evaluation_errors": [],
            "workflow_status": "started",
            "node_history": [],
        },
        config={"configurable": {"thread_id": agent_workflow_thread_id(run)}},
    )
    return get_agent_run(session, run.id)


def resume_agent_workflow_if_interrupted(run: AgentRun, model: AgentModel) -> bool:
    graph = build_agent_workflow(model)
    config = {"configurable": {"thread_id": agent_workflow_thread_id(run)}}
    snapshot = graph.get_state(config)
    if not snapshot.values or not snapshot.next:
        return False
    graph.invoke(None, config=config)
    return True


def agent_workflow_checkpoint_history(run: AgentRun) -> list[dict]:
    graph = build_agent_workflow(create_agent_model("demo"))
    snapshots = graph.get_state_history(
        {"configurable": {"thread_id": agent_workflow_thread_id(run)}}
    )
    return [
        {
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
            "step": snapshot.metadata.get("step"),
            "next_nodes": list(snapshot.next),
            "state": dict(snapshot.values),
        }
        for snapshot in snapshots
    ]
