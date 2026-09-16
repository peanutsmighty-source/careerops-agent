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
from app.services.checkpoint_versioning import (
    UNVERSIONED_CHECKPOINT,
    checkpoint_version,
    require_checkpoint_version,
    require_explicit_source_version,
)


AGENT_WORKFLOW_VERSION = "agent-workflow-v1"


class AgentWorkflowState(TypedDict, total=False):
    graph_version: str
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
            "graph_version": AGENT_WORKFLOW_VERSION,
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
    require_checkpoint_version(
        dict(snapshot.values),
        graph_name="agent workflow",
        expected_version=AGENT_WORKFLOW_VERSION,
    )
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
            "graph_version": checkpoint_version(dict(snapshot.values)),
            "runtime_graph_version": AGENT_WORKFLOW_VERSION,
            "checkpoint_compatible": (
                checkpoint_version(dict(snapshot.values)) == AGENT_WORKFLOW_VERSION
            ),
            "state": dict(snapshot.values),
        }
        for snapshot in snapshots
    ]


def require_agent_workflow_checkpoint_compatible(run: AgentRun, model: AgentModel) -> None:
    graph = build_agent_workflow(model)
    snapshot = graph.get_state(
        {"configurable": {"thread_id": agent_workflow_thread_id(run)}}
    )
    if snapshot.values and snapshot.next:
        require_checkpoint_version(
            dict(snapshot.values),
            graph_name="agent workflow",
            expected_version=AGENT_WORKFLOW_VERSION,
        )


def migrate_agent_workflow_checkpoint(
    run: AgentRun, model: AgentModel, *, source_version: str
) -> dict:
    graph = build_agent_workflow(model)
    config = {"configurable": {"thread_id": agent_workflow_thread_id(run)}}
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise ValueError("agent workflow has no checkpoint to migrate")
    require_explicit_source_version(dict(snapshot.values), source_version=source_version)
    if source_version != UNVERSIONED_CHECKPOINT:
        raise ValueError(f"no agent workflow migration exists from '{source_version}'")
    next_nodes = list(snapshot.next)
    predecessor = {"run_agent": "load_context", "evaluate_result": "run_agent",
                   "complete_workflow": "evaluate_result",
                   "block_workflow": "evaluate_result"}.get(
                       next_nodes[0] if len(next_nodes) == 1 else ""
                   )
    if predecessor is None:
        raise ValueError("agent workflow checkpoint is not at a migratable boundary")
    graph.update_state(config, {"graph_version": AGENT_WORKFLOW_VERSION}, as_node=predecessor)
    migrated = graph.get_state(config)
    require_checkpoint_version(
        dict(migrated.values), graph_name="agent workflow",
        expected_version=AGENT_WORKFLOW_VERSION,
    )
    return {
        "thread_id": agent_workflow_thread_id(run),
        "graph_version": checkpoint_version(dict(migrated.values)),
        "runtime_graph_version": AGENT_WORKFLOW_VERSION,
        "checkpoint_compatible": True,
        "next_nodes": list(migrated.next),
        "state": dict(migrated.values),
    }
