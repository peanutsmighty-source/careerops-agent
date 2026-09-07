from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database import Base, engine, get_session
from app.schema_compat import (
    ensure_agent_timing_columns,
    ensure_memory_candidate_columns,
    ensure_memory_scope_columns,
)
from app.models import (
    AgentAlignment,
    AgentMemory,
    AgentRun,
    AgentTask,
    ArchiveStatus,
    ExecutionTrace,
    GoalContract,
    Job,
    JobArchive,
    JobRequirement,
    JobSource,
    MemoryCandidateRecord,
    PlanStep,
    Skill,
)
from app.schemas import (
    AgentAlignmentCreate,
    AgentAlignmentRead,
    AgentMemoryCreate,
    AgentMemoryRead,
    AgentMemoryRetire,
    AgentTaskCreate,
    AgentTaskRead,
    AgentTaskUpdate,
    AgentRunCreate,
    AgentRunRead,
    AgentRunRecoveryRead,
    ExecutionTraceCreate,
    ExecutionTraceRead,
    GraphCheckpointRead,
    GoalContractCreate,
    GoalContractRead,
    JobArchiveCreate,
    JobArchiveRead,
    JobCreate,
    JobFetchCreate,
    JobRequirementRead,
    JobRead,
    JobSourceCreate,
    JobSourceRead,
    LearningRoadmapItemRead,
    LearningGraphRunRead,
    LearningGraphResumeCreate,
    LearningTaskRead,
    MemoryContextRead,
    MemoryCandidateRead,
    PlanCreate,
    PlanStepRead,
    ParsedJobRead,
    SkillDemandRead,
    SkillCreate,
    SkillRead,
    TaskPolicyRead,
    TaskPolicyUpdate,
    ToolCallCreate,
    ToolCallRead,
    ToolDefinitionRead,
    ToolRecoveryRunRead,
)
from app.services.governance import DEFAULT_GOAL_CONTRACT, evaluate_alignment
from app.services.analytics import skill_demand_rows
from app.services.ingestion import (
    canonicalize_url,
    content_fingerprint,
    fetch_public_text,
    parse_job_description,
)
from app.services.learning_graph import (
    checkpoint_history,
    graph_thread_id,
    resume_learning_graph,
    start_learning_graph,
)
from app.services.roadmap import build_learning_roadmap, build_learning_tasks
from app.services.runtime import validate_plan_step
from app.services.tool_runtime import (
    IdempotencyConflictError,
    ToolReplayPermissionError,
    UnknownToolError,
    list_task_tool_calls,
    run_tool_call,
    serialize_tool_call,
)
from app.services.tool_recovery import recover_stale_tool_calls
from app.services.authorization import (
    InvalidTaskPolicyError,
    get_or_create_task_policy,
    update_task_policy,
)
from app.services.tools import list_tools
from app.services.agent_loop import get_agent_run, list_agent_runs
from app.services.agent_workflow import (
    agent_workflow_checkpoint_history,
    start_agent_workflow,
)
from app.services.agent_run_recovery import recover_stale_agent_runs
from app.services.memory_runtime import assemble_memory_context, resolve_memory_scope
from app.services.memory_lifecycle import retire_memory, retire_task_working_memories


app = FastAPI(title="CareerOps Agent", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_memory_scope_columns(engine)
    ensure_memory_candidate_columns(engine)
    ensure_agent_timing_columns(engine)
    with Session(engine) as session:
        _get_or_create_active_goal_contract(session)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def runtime_console() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/agent/goal-contract", response_model=GoalContractRead)
def get_active_goal_contract(session: Session = Depends(get_session)) -> GoalContract:
    return _get_or_create_active_goal_contract(session)


@app.post("/agent/goal-contracts", response_model=GoalContractRead, status_code=status.HTTP_201_CREATED)
def create_goal_contract(
    payload: GoalContractCreate, session: Session = Depends(get_session)
) -> GoalContract:
    active_contracts = list(session.scalars(select(GoalContract).where(GoalContract.is_active)))
    for contract in active_contracts:
        contract.is_active = False
    latest_version = session.scalar(select(func.max(GoalContract.version))) or 0
    contract = GoalContract(version=latest_version + 1, **payload.model_dump())
    session.add(contract)
    session.commit()
    session.refresh(contract)
    return contract


@app.post("/agent/memories", response_model=AgentMemoryRead, status_code=status.HTTP_201_CREATED)
def create_agent_memory(
    payload: AgentMemoryCreate, session: Session = Depends(get_session)
) -> AgentMemory:
    contract = _get_or_create_active_goal_contract(session)
    values = payload.model_dump()
    try:
        values["task_id"], values["run_id"] = resolve_memory_scope(
            session,
            goal_contract_id=contract.id,
            memory_type=payload.memory_type,
            scope_type=payload.scope_type,
            task_id=payload.task_id,
            run_id=payload.run_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    memory = AgentMemory(goal_contract_id=contract.id, **values)
    session.add(memory)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="memory key already exists for this memory type") from exc
    session.refresh(memory)
    return memory


@app.get("/agent/memories", response_model=list[AgentMemoryRead])
def list_agent_memories(
    memory_type: str | None = None,
    include_retired: bool = False,
    session: Session = Depends(get_session),
) -> list[AgentMemory]:
    contract = _get_or_create_active_goal_contract(session)
    statement = select(AgentMemory).where(AgentMemory.goal_contract_id == contract.id)
    if memory_type:
        statement = statement.where(AgentMemory.memory_type == memory_type)
    if not include_retired:
        statement = statement.where(AgentMemory.status == "active")
    return list(session.scalars(statement.order_by(AgentMemory.importance.desc(), AgentMemory.id)))


@app.get("/agent/memory-candidates", response_model=list[MemoryCandidateRead])
def list_memory_candidates(
    task_id: int | None = Query(default=None, ge=1),
    run_id: int | None = Query(default=None, ge=1),
    decision: str | None = Query(default=None, pattern="^(accept|reject|needs_review)$"),
    session: Session = Depends(get_session),
) -> list[MemoryCandidateRecord]:
    contract = _get_or_create_active_goal_contract(session)
    statement = select(MemoryCandidateRecord).where(
        MemoryCandidateRecord.goal_contract_id == contract.id
    )
    if task_id is not None:
        statement = statement.where(MemoryCandidateRecord.task_id == task_id)
    if run_id is not None:
        statement = statement.where(MemoryCandidateRecord.source_run_id == run_id)
    if decision is not None:
        statement = statement.where(MemoryCandidateRecord.decision == decision)
    return list(session.scalars(statement.order_by(MemoryCandidateRecord.id.desc())))


@app.post("/agent/memories/{memory_id}/retire", response_model=AgentMemoryRead)
def retire_agent_memory(
    memory_id: int,
    payload: AgentMemoryRetire,
    session: Session = Depends(get_session),
) -> AgentMemory:
    contract = _get_or_create_active_goal_contract(session)
    memory = session.get(AgentMemory, memory_id)
    if not memory or memory.goal_contract_id != contract.id:
        raise HTTPException(status_code=404, detail="memory not found")
    retire_memory(memory, reason=payload.reason)
    session.commit()
    session.refresh(memory)
    return memory


@app.post("/agent/alignments", response_model=AgentAlignmentRead, status_code=status.HTTP_201_CREATED)
def check_goal_alignment(
    payload: AgentAlignmentCreate, session: Session = Depends(get_session)
) -> AgentAlignment:
    contract = _get_or_create_active_goal_contract(session)
    evaluation = evaluate_alignment(
        contract,
        agent_capability=payload.agent_capability,
        success_criterion=payload.success_criterion,
        goal_connection=payload.goal_connection,
        learning_outcome=payload.learning_outcome,
    )
    alignment = AgentAlignment(
        goal_contract_id=contract.id,
        **payload.model_dump(),
        decision=evaluation.decision,
        evaluation_notes=evaluation.notes,
    )
    session.add(alignment)
    session.commit()
    session.refresh(alignment)
    return alignment


@app.post("/agent/tasks", response_model=AgentTaskRead, status_code=status.HTTP_201_CREATED)
def create_agent_task(payload: AgentTaskCreate, session: Session = Depends(get_session)) -> AgentTask:
    contract = _get_or_create_active_goal_contract(session)
    task = AgentTask(goal_contract_id=contract.id, **payload.model_dump())
    session.add(task)
    session.flush()
    session.add(
        ExecutionTrace(
            task_id=task.id,
            event_type="user_input",
            input_summary=task.user_goal,
            output_summary="Created a user task contract.",
            metadata_json={
                "constraints": task.constraints,
                "success_criteria": task.success_criteria,
            },
        )
    )
    get_or_create_task_policy(session, task)
    session.refresh(task)
    return task


@app.get("/agent/tasks", response_model=list[AgentTaskRead])
def list_agent_tasks(session: Session = Depends(get_session)) -> list[AgentTask]:
    return list(session.scalars(select(AgentTask).order_by(AgentTask.updated_at.desc(), AgentTask.id)))


@app.get("/agent/tasks/{task_id}", response_model=AgentTaskRead)
def get_agent_task(task_id: int, session: Session = Depends(get_session)) -> AgentTask:
    return _get_agent_task_or_404(session, task_id)


@app.patch("/agent/tasks/{task_id}", response_model=AgentTaskRead)
def update_agent_task(
    task_id: int, payload: AgentTaskUpdate, session: Session = Depends(get_session)
) -> AgentTask:
    task = _get_agent_task_or_404(session, task_id)
    task.status = payload.status
    task.completed_at = datetime.utcnow() if payload.status == "completed" else None
    if payload.status in {"completed", "cancelled"}:
        retire_task_working_memories(
            session,
            task_id=task.id,
            reason=f"agent_task_{payload.status}",
        )
    session.commit()
    session.refresh(task)
    return task


@app.post(
    "/agent/tasks/{task_id}/plan", response_model=list[PlanStepRead], status_code=status.HTTP_201_CREATED
)
def create_agent_plan(
    task_id: int, payload: PlanCreate, session: Session = Depends(get_session)
) -> list[PlanStep]:
    task = _get_agent_task_or_404(session, task_id)
    if task.plan_steps:
        raise HTTPException(status_code=409, detail="task already has a plan")

    validation_errors: list[dict] = []
    for sequence, step in enumerate(payload.steps, start=1):
        validation = validate_plan_step(
            task,
            agent_capability=step.agent_capability,
            success_criterion=step.success_criterion,
            goal_connection=step.goal_connection,
            learning_outcome=step.learning_outcome,
        )
        if not validation.is_valid:
            validation_errors.append({"sequence": sequence, "notes": validation.notes})
    if validation_errors:
        raise HTTPException(status_code=400, detail=validation_errors)

    plan_steps = [
        PlanStep(task_id=task.id, sequence=sequence, **step.model_dump())
        for sequence, step in enumerate(payload.steps, start=1)
    ]
    task.status = "planning"
    session.add_all(plan_steps)
    session.add(
        ExecutionTrace(
            task_id=task.id,
            event_type="plan",
            input_summary="Validated and stored a task plan.",
            output_summary=f"Created {len(plan_steps)} ordered plan step(s).",
            metadata_json={"plan_step_count": len(plan_steps)},
        )
    )
    session.commit()
    for step in plan_steps:
        session.refresh(step)
    return plan_steps


@app.get("/agent/tasks/{task_id}/plan", response_model=list[PlanStepRead])
def get_agent_plan(task_id: int, session: Session = Depends(get_session)) -> list[PlanStep]:
    _get_agent_task_or_404(session, task_id)
    return list(
        session.scalars(select(PlanStep).where(PlanStep.task_id == task_id).order_by(PlanStep.sequence))
    )


@app.post(
    "/agent/tasks/{task_id}/traces",
    response_model=ExecutionTraceRead,
    status_code=status.HTTP_201_CREATED,
)
def create_execution_trace(
    task_id: int, payload: ExecutionTraceCreate, session: Session = Depends(get_session)
) -> ExecutionTrace:
    _get_agent_task_or_404(session, task_id)
    if payload.plan_step_id:
        step = session.get(PlanStep, payload.plan_step_id)
        if not step or step.task_id != task_id:
            raise HTTPException(status_code=400, detail="plan step does not belong to this task")
    trace = ExecutionTrace(task_id=task_id, **payload.model_dump())
    session.add(trace)
    session.commit()
    session.refresh(trace)
    return trace


@app.get("/agent/tasks/{task_id}/traces", response_model=list[ExecutionTraceRead])
def list_execution_traces(task_id: int, session: Session = Depends(get_session)) -> list[ExecutionTrace]:
    _get_agent_task_or_404(session, task_id)
    return list(
        session.scalars(
            select(ExecutionTrace)
            .where(ExecutionTrace.task_id == task_id)
            .order_by(ExecutionTrace.id)
        )
    )


@app.post("/agent/tasks/{task_id}/agent-runs", response_model=AgentRunRead)
def run_bounded_agent_loop(
    task_id: int,
    payload: AgentRunCreate,
    session: Session = Depends(get_session),
):
    task = _get_agent_task_or_404(session, task_id)
    try:
        return start_agent_workflow(
            session,
            task=task,
            provider=payload.provider,
            model_name=payload.model,
            max_steps=payload.max_steps,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/agent/tasks/{task_id}/agent-runs", response_model=list[AgentRunRead])
def get_bounded_agent_runs(
    task_id: int, session: Session = Depends(get_session)
):
    _get_agent_task_or_404(session, task_id)
    return list_agent_runs(session, task_id)


@app.get(
    "/agent/tasks/{task_id}/memory-context",
    response_model=MemoryContextRead,
)
def preview_agent_task_memory_context(
    task_id: int,
    run_id: int | None = Query(default=None, ge=1),
    memory_limit: int = Query(default=8, ge=0, le=50),
    memory_char_budget: int = Query(default=3000, ge=0, le=50000),
    session: Session = Depends(get_session),
):
    task = _get_agent_task_or_404(session, task_id)
    if run_id is not None:
        run = session.get(AgentRun, run_id)
        if not run or run.task_id != task.id:
            raise HTTPException(status_code=404, detail="agent run not found for this task")
    return assemble_memory_context(
        session,
        task,
        run_id=run_id,
        memory_limit=memory_limit,
        memory_char_budget=memory_char_budget,
    ).as_dict()


@app.get(
    "/agent/tasks/{task_id}/agent-runs/{run_id}/checkpoints",
    response_model=list[GraphCheckpointRead],
)
def get_agent_run_workflow_checkpoints(
    task_id: int, run_id: int, session: Session = Depends(get_session)
):
    _get_agent_task_or_404(session, task_id)
    try:
        run = get_agent_run(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if run.task_id != task_id:
        raise HTTPException(status_code=404, detail="agent run not found")
    return agent_workflow_checkpoint_history(run)


@app.post(
    "/agent/tasks/{task_id}/recover-agent-runs",
    response_model=AgentRunRecoveryRead,
)
def recover_agent_runs(
    task_id: int,
    stale_after_seconds: int = Query(default=60, ge=0, le=86400),
    session: Session = Depends(get_session),
):
    task = _get_agent_task_or_404(session, task_id)
    report = recover_stale_agent_runs(
        session,
        task=task,
        stale_after_seconds=stale_after_seconds,
    )
    return {
        "task_id": report.task_id,
        "stale_before": report.stale_before,
        "scanned_count": report.scanned_count,
        "recovered_count": report.recovered_count,
        "decisions": [decision.__dict__ for decision in report.decisions],
    }


@app.get("/agent/tools", response_model=list[ToolDefinitionRead])
def get_registered_tools() -> list[dict]:
    return list_tools()


@app.get("/agent/tasks/{task_id}/tool-policy", response_model=TaskPolicyRead)
def get_agent_task_tool_policy(
    task_id: int, session: Session = Depends(get_session)
):
    task = _get_agent_task_or_404(session, task_id)
    return get_or_create_task_policy(session, task)


@app.put("/agent/tasks/{task_id}/tool-policy", response_model=TaskPolicyRead)
def put_agent_task_tool_policy(
    task_id: int,
    payload: TaskPolicyUpdate,
    session: Session = Depends(get_session),
):
    task = _get_agent_task_or_404(session, task_id)
    try:
        return update_task_policy(session, task, allowed_tools=payload.allowed_tools)
    except InvalidTaskPolicyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/agent/tasks/{task_id}/tool-calls", response_model=ToolCallRead)
def call_agent_tool(
    task_id: int,
    payload: ToolCallCreate,
    session: Session = Depends(get_session),
) -> dict:
    task = _get_agent_task_or_404(session, task_id)
    if payload.plan_step_id:
        step = session.get(PlanStep, payload.plan_step_id)
        if not step or step.task_id != task.id:
            raise HTTPException(status_code=400, detail="plan step does not belong to this task")

    try:
        outcome = run_tool_call(
            session,
            task=task,
            tool_name=payload.tool_name,
            arguments=payload.arguments,
            requested_idempotency_key=payload.idempotency_key,
            plan_step_id=payload.plan_step_id,
        )
    except UnknownToolError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ToolReplayPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return serialize_tool_call(outcome.record, replayed=outcome.replayed)


@app.get("/agent/tasks/{task_id}/tool-calls", response_model=list[ToolCallRead])
def get_agent_tool_calls(
    task_id: int, session: Session = Depends(get_session)
) -> list[dict]:
    _get_agent_task_or_404(session, task_id)
    return [serialize_tool_call(record) for record in list_task_tool_calls(session, task_id)]


@app.post(
    "/agent/tasks/{task_id}/recover-tool-calls",
    response_model=ToolRecoveryRunRead,
)
def recover_agent_tool_calls(
    task_id: int,
    stale_after_seconds: int = Query(default=60, ge=0, le=86400),
    session: Session = Depends(get_session),
) -> dict:
    task = _get_agent_task_or_404(session, task_id)
    report = recover_stale_tool_calls(
        session,
        task=task,
        stale_after_seconds=stale_after_seconds,
    )
    return {
        "task_id": report.task_id,
        "stale_before": report.stale_before,
        "scanned_count": report.scanned_count,
        "recovered_count": report.recovered_count,
        "decisions": [decision.__dict__ for decision in report.decisions],
    }


@app.post("/agent/tasks/{task_id}/run-learning-graph", response_model=LearningGraphRunRead)
def run_agent_learning_graph(
    task_id: int, session: Session = Depends(get_session)
) -> dict:
    task = _get_agent_task_or_404(session, task_id)
    result = start_learning_graph(task)
    state = result["state"]
    history = checkpoint_history(task)
    if result["awaiting_approval"]:
        task.status = "waiting_for_approval"
    else:
        task.status = "completed" if state.get("status") == "completed" else "blocked"
    task.completed_at = datetime.utcnow() if task.status == "completed" else None
    if task.status == "completed":
        retire_task_working_memories(
            session,
            task_id=task.id,
            reason="agent_task_completed",
        )
    session.add(
        ExecutionTrace(
            task_id=task.id,
            event_type="evaluation",
            input_summary="Run the deterministic LangGraph learning-plan workflow.",
            output_summary=(
                "Graph paused for user approval."
                if result["awaiting_approval"]
                else f"Graph finished with status {state.get('status')}."
            ),
            metadata_json={
                "thread_id": graph_thread_id(task),
                "node_history": state.get("node_history", []),
                "checkpoint_count": len(history),
            },
        )
    )
    session.commit()
    return {**result, "checkpoint_count": len(history)}


@app.post(
    "/agent/tasks/{task_id}/resume-learning-graph",
    response_model=LearningGraphRunRead,
)
def resume_agent_learning_graph(
    task_id: int,
    payload: LearningGraphResumeCreate,
    session: Session = Depends(get_session),
) -> dict:
    task = _get_agent_task_or_404(session, task_id)
    try:
        result = resume_learning_graph(
            task,
            approved=payload.approved,
            comment=payload.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    state = result["state"]
    history = checkpoint_history(task)
    task.status = "completed" if state.get("status") == "completed" else "blocked"
    task.completed_at = datetime.utcnow() if task.status == "completed" else None
    if task.status == "completed":
        retire_task_working_memories(
            session,
            task_id=task.id,
            reason="agent_task_completed",
        )
    session.add(
        ExecutionTrace(
            task_id=task.id,
            event_type="user_input",
            input_summary=(
                "User approved the generated learning plan."
                if payload.approved
                else "User rejected the generated learning plan."
            ),
            output_summary=f"Graph resumed and finished with status {state.get('status')}.",
            metadata_json={
                "thread_id": graph_thread_id(task),
                "approved": payload.approved,
                "comment": payload.comment,
                "checkpoint_count": len(history),
            },
        )
    )
    session.commit()
    return {**result, "checkpoint_count": len(history)}


@app.get(
    "/agent/tasks/{task_id}/graph-checkpoints",
    response_model=list[GraphCheckpointRead],
)
def list_graph_checkpoints(
    task_id: int, session: Session = Depends(get_session)
) -> list[dict]:
    task = _get_agent_task_or_404(session, task_id)
    return checkpoint_history(task)


@app.post("/jobs", response_model=JobRead, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreate, session: Session = Depends(get_session)) -> Job:
    values = payload.model_dump()
    values["source_url"] = str(values["source_url"])
    job = Job(**values)
    session.add(job)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="source_url already exists") from exc
    session.refresh(job)
    return job


@app.get("/jobs", response_model=list[JobRead])
def list_jobs(session: Session = Depends(get_session)) -> list[Job]:
    return list(session.scalars(select(Job).order_by(Job.created_at.desc())))


@app.get("/jobs/{job_id}/requirements", response_model=list[JobRequirementRead])
def list_job_requirements(
    job_id: int, session: Session = Depends(get_session)
) -> list[JobRequirement]:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return list(
        session.scalars(
            select(JobRequirement)
            .where(JobRequirement.job_id == job_id)
            .order_by(JobRequirement.importance.desc(), JobRequirement.id)
        )
    )


@app.post("/skills", response_model=SkillRead, status_code=status.HTTP_201_CREATED)
def create_skill(payload: SkillCreate, session: Session = Depends(get_session)) -> Skill:
    skill = Skill(**payload.model_dump())
    session.add(skill)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="skill name already exists") from exc
    session.refresh(skill)
    return skill


@app.get("/skills", response_model=list[SkillRead])
def list_skills(session: Session = Depends(get_session)) -> list[Skill]:
    return list(session.scalars(select(Skill).order_by(Skill.name)))


@app.get("/analytics/skills", response_model=list[SkillDemandRead])
def list_skill_demand(session: Session = Depends(get_session)) -> list[dict]:
    return skill_demand_rows(session)


@app.get("/analytics/learning-roadmap", response_model=list[LearningRoadmapItemRead])
def get_learning_roadmap(session: Session = Depends(get_session)) -> list[dict]:
    return build_learning_roadmap(skill_demand_rows(session))


@app.get("/learning/tasks", response_model=list[LearningTaskRead])
def list_learning_tasks(session: Session = Depends(get_session)) -> list[dict]:
    return build_learning_tasks(skill_demand_rows(session))


@app.post("/job-sources", response_model=JobSourceRead, status_code=status.HTTP_201_CREATED)
def create_job_source(
    payload: JobSourceCreate, session: Session = Depends(get_session)
) -> JobSource:
    values = payload.model_dump()
    values["url"] = str(values["url"])
    source = JobSource(**values)
    session.add(source)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="job source already exists") from exc
    session.refresh(source)
    return source


@app.get("/job-sources", response_model=list[JobSourceRead])
def list_job_sources(session: Session = Depends(get_session)) -> list[JobSource]:
    return list(session.scalars(select(JobSource).order_by(JobSource.name)))


@app.post("/job-archives", response_model=JobArchiveRead, status_code=status.HTTP_201_CREATED)
def archive_job_description(
    payload: JobArchiveCreate, session: Session = Depends(get_session)
) -> JobArchive:
    return _archive_raw_content(
        session=session,
        source_url=str(payload.source_url),
        raw_content=payload.raw_content,
        source_id=payload.source_id,
        should_parse=payload.parse,
        should_create_job=payload.create_job,
    )


@app.post("/job-archives/fetch", response_model=JobArchiveRead, status_code=status.HTTP_201_CREATED)
def fetch_and_archive_job_description(
    payload: JobFetchCreate, session: Session = Depends(get_session)
) -> JobArchive:
    raw_content = fetch_public_text(str(payload.source_url))
    return _archive_raw_content(
        session=session,
        source_url=str(payload.source_url),
        raw_content=raw_content,
        source_id=payload.source_id,
        should_parse=payload.parse,
        should_create_job=payload.create_job,
    )


@app.get("/job-archives", response_model=list[JobArchiveRead])
def list_job_archives(session: Session = Depends(get_session)) -> list[JobArchive]:
    return list(session.scalars(select(JobArchive).order_by(JobArchive.fetched_at.desc())))


@app.post("/job-archives/parse", response_model=ParsedJobRead)
def preview_job_description_parse(payload: JobArchiveCreate) -> dict:
    parsed = parse_job_description(payload.raw_content, str(payload.source_url))
    return parsed.as_dict()


def _archive_raw_content(
    session: Session,
    source_url: str,
    raw_content: str,
    source_id: int | None,
    should_parse: bool,
    should_create_job: bool,
) -> JobArchive:
    canonical_url = canonicalize_url(source_url)
    fingerprint = content_fingerprint(raw_content)
    duplicate = session.scalar(
        select(JobArchive).where(
            (JobArchive.canonical_url == canonical_url) | (JobArchive.content_hash == fingerprint)
        )
    )
    if duplicate:
        duplicate.status = ArchiveStatus.DUPLICATE
        session.commit()
        session.refresh(duplicate)
        return duplicate

    archive = JobArchive(
        source_id=source_id,
        source_url=source_url,
        canonical_url=canonical_url,
        content_hash=fingerprint,
        raw_content=raw_content,
    )
    if should_parse:
        parsed = parse_job_description(raw_content, source_url)
        archive.structured_data = parsed.as_dict()
        archive.parsed_at = datetime.utcnow()
        archive.status = ArchiveStatus.PARSED
        if should_create_job:
            archive.job = _job_from_parsed_description(session, source_url, parsed)

    session.add(archive)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="job archive already exists") from exc
    session.refresh(archive)
    return archive


def _job_from_parsed_description(session: Session, source_url: str, parsed) -> Job:
    existing_job = session.scalar(select(Job).where(Job.source_url == source_url))
    if existing_job:
        return existing_job

    job = Job(
        company=parsed.company,
        title=parsed.title,
        location=parsed.location,
        source_url=source_url,
        description=parsed.description,
        published_at=parsed.published_at,
    )
    for requirement in parsed.requirements:
        skill = _get_or_create_skill(session, requirement.name, requirement.category)
        job.requirements.append(
            JobRequirement(
                skill=skill,
                importance=requirement.importance,
                requirement_type=requirement.requirement_type,
                evidence_text=requirement.evidence_text,
            )
        )
    return job


def _get_or_create_skill(session: Session, name: str, category: str) -> Skill:
    skill = session.scalar(select(Skill).where(Skill.name == name))
    if skill:
        return skill
    skill = Skill(name=name, category=category)
    session.add(skill)
    return skill


def _get_or_create_active_goal_contract(session: Session) -> GoalContract:
    contract = session.scalar(
        select(GoalContract)
        .where(GoalContract.is_active)
        .order_by(GoalContract.version.desc())
    )
    if contract:
        return contract

    contract = GoalContract(version=1, **DEFAULT_GOAL_CONTRACT)
    session.add(contract)
    session.commit()
    session.refresh(contract)
    return contract


def _get_agent_task_or_404(session: Session, task_id: int) -> AgentTask:
    task = session.get(AgentTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="agent task not found")
    return task
