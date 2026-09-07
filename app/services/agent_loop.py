from __future__ import annotations

import json
import os
from time import perf_counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Protocol
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.database import SessionLocal
from app.models import AgentRun, AgentRunStep, AgentTask, ExecutionTrace
from app.services.authorization import get_or_create_task_policy
from app.services.memory_evaluator import evaluate_and_store_run_memories
from app.services.memory_lifecycle import retire_run_working_memories
from app.services.memory_runtime import assemble_memory_context
from app.services.model_credentials import load_deepseek_api_key
from app.services.tool_runtime import run_tool_call, serialize_tool_call
from app.services.tools import list_tools


DEFAULT_OPENAI_MODEL = os.getenv("CAREEROPS_OPENAI_MODEL", "gpt-5-mini")
DEFAULT_DEEPSEEK_MODEL = os.getenv("CAREEROPS_DEEPSEEK_MODEL", "deepseek-v4-flash")


@dataclass(frozen=True)
class AgentDecision:
    action_type: str
    content: str | None = None
    tool_name: str | None = None
    arguments: dict | None = None
    call_id: str | None = None
    provider_metadata: dict | None = None

    def as_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "content": self.content,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_id": self.call_id,
            "provider_metadata": self.provider_metadata or {},
        }


@dataclass(frozen=True)
class AgentModelRequest:
    task_id: int
    user_goal: str
    constraints: list[str]
    success_criteria: list[str]
    memory_context: dict
    observations: list[dict]
    tools: list[dict]
    step_number: int
    max_steps: int

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_goal": self.user_goal,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "memory_context": self.memory_context,
            "observations": self.observations,
            "tools": [tool["name"] for tool in self.tools],
            "step_number": self.step_number,
            "max_steps": self.max_steps,
        }


class AgentModel(Protocol):
    provider: str
    model: str

    def decide(self, request: AgentModelRequest) -> AgentDecision: ...


class DemoAgentModel:
    provider = "demo"
    model = "careerops-demo-policy-v1"

    def decide(self, request: AgentModelRequest) -> AgentDecision:
        if not request.observations:
            return AgentDecision(
                action_type="tool_call",
                tool_name="search_jobs",
                arguments={"query": "Agent", "limit": 3},
                call_id=f"demo-{request.step_number}",
                provider_metadata={"deterministic": True},
            )
        if len(request.observations) == 1:
            return AgentDecision(
                action_type="tool_call",
                tool_name="get_skill_demand",
                arguments={"limit": 5},
                call_id=f"demo-{request.step_number}",
                provider_metadata={"deterministic": True},
            )
        jobs = request.observations[0].get("output") or {}
        skills = request.observations[1].get("output") or {}
        return AgentDecision(
            action_type="final_answer",
            content=(
                f"Reviewed {jobs.get('count', 0)} matching jobs and "
                f"{len(skills.get('skills', []))} demanded skills. "
                "Use the JD evidence to choose the next learning task."
            ),
            provider_metadata={"deterministic": True},
        )


class OpenAIResponsesAgentModel:
    provider = "openai"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or DEFAULT_OPENAI_MODEL
        self._client = OpenAI()

    def decide(self, request: AgentModelRequest) -> AgentDecision:
        response = self._client.responses.create(
            model=self.model,
            instructions=(
                "You are the CareerOps single-agent planner. Choose at most one tool per turn. "
                "Use tools only when their observations are needed. When the task can be answered, "
                "return a concise final answer. Never claim permissions or invent tool results."
            ),
            input=json.dumps(request.as_dict(), ensure_ascii=False),
            tools=[
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                    "strict": True,
                }
                for tool in request.tools
            ],
        )
        for item in response.output:
            if item.type == "function_call":
                return AgentDecision(
                    action_type="tool_call",
                    tool_name=item.name,
                    arguments=json.loads(item.arguments),
                    call_id=item.call_id,
                    provider_metadata={"response_id": response.id},
                )
        if response.output_text:
            return AgentDecision(
                action_type="final_answer",
                content=response.output_text,
                provider_metadata={"response_id": response.id},
            )
        raise ValueError("model returned neither a tool call nor a final answer")


class DeepSeekChatAgentModel:
    provider = "deepseek"

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        self.model = model or DEFAULT_DEEPSEEK_MODEL
        self._client = OpenAI(
            api_key=load_deepseek_api_key(),
            base_url="https://api.deepseek.com",
        )

    def decide(self, request: AgentModelRequest) -> AgentDecision:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the CareerOps single-agent planner. Choose at most one "
                        "tool per turn. Use tools only when their observations are needed. "
                        "When the task can be answered, return a concise final answer. "
                        "Never claim permissions or invent tool results."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(request.as_dict(), ensure_ascii=False),
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"],
                    },
                }
                for tool in request.tools
            ],
        )
        message = response.choices[0].message
        usage = response.usage
        metadata = {
            "completion_id": response.id,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
        }
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            return AgentDecision(
                action_type="tool_call",
                tool_name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments),
                call_id=tool_call.id,
                provider_metadata=metadata,
            )
        if message.content:
            return AgentDecision(
                action_type="final_answer",
                content=message.content,
                provider_metadata=metadata,
            )
        raise ValueError("DeepSeek returned neither a tool call nor a final answer")


def create_agent_model(provider: str, model: str | None = None) -> AgentModel:
    if provider == "demo":
        return DemoAgentModel()
    if provider == "openai":
        try:
            return OpenAIResponsesAgentModel(model)
        except Exception as exc:
            raise ValueError(
                "OpenAI provider is not configured; set OPENAI_API_KEY before starting the run"
            ) from exc
    if provider == "deepseek":
        try:
            return DeepSeekChatAgentModel(model)
        except Exception as exc:
            raise ValueError(
                "DeepSeek provider is not configured; set DEEPSEEK_API_KEY "
                "or provide a non-empty ds_key.txt"
            ) from exc
    raise ValueError(f"unknown agent model provider: {provider}")


class AgentLoopEngine:
    """Framework-independent model/tool loop with durable step boundaries."""

    def __init__(
        self,
        model: AgentModel,
        *,
        session_factory: Callable = SessionLocal,
        tool_runner: Callable = run_tool_call,
    ) -> None:
        self.model = model
        self.session_factory = session_factory
        self.tool_runner = tool_runner
        self._active_phase = "initialization"

    def run(self, run_id: int) -> None:
        try:
            self._run(run_id)
        except Exception as exc:
            self._mark_failed(run_id, exc)

    def execute_pending_tool_step(self, run_id: int, step_id: int) -> None:
        with self.session_factory() as session:
            step = session.get(AgentRunStep, step_id)
            if not step or step.run_id != run_id or step.action_type != "tool_call":
                raise ValueError("pending agent tool step not found")
            if step.observation is not None or step.tool_call_id is not None:
                raise ValueError("agent tool step is not pending execution")
            response = step.model_response
            decision = AgentDecision(
                action_type="tool_call",
                tool_name=response.get("tool_name"),
                arguments=response.get("arguments"),
                call_id=response.get("call_id"),
                provider_metadata=response.get("provider_metadata"),
            )
            self._validate_decision(decision)
        self._execute_tool(run_id, step_id, decision)

    def _run(self, run_id: int) -> None:
        if self._finish_committed_boundary(run_id):
            return
        if self._step_limit_reached(run_id):
            self._stop_at_limit(run_id)
            return
        while True:
            self._active_phase = "request_build"
            built = self._build_request(run_id)
            if built is None:
                return
            request, timing = built
            self._active_phase = "model_call"
            model_started = perf_counter()
            decision = self.model.decide(request)
            timing["model_call_ms"] = _elapsed_ms(model_started)
            self._active_phase = "decision_validation"
            self._validate_decision(decision)
            step_id = self._record_model_decision(
                run_id, request, decision, timing=timing
            )
            if decision.action_type == "final_answer":
                self._active_phase = "memory_finalize"
                self._complete_run(run_id, decision.content or "")
                return
            self._active_phase = "tool_call"
            self._execute_tool(run_id, step_id, decision)
            if self._step_limit_reached(run_id):
                self._stop_at_limit(run_id)
                return

    def _finish_committed_boundary(self, run_id: int) -> bool:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                raise ValueError("agent run no longer exists")
            if run.status != "running" or not run.steps:
                return run.status != "running"
            latest = run.steps[-1]
            if latest.action_type == "final_answer":
                answer = latest.model_response.get("content") or ""
            elif latest.observation is None:
                raise RuntimeError(
                    "agent run has an incomplete tool step; recovery is required"
                )
            else:
                return False
        self._complete_run(run_id, answer)
        return True

    def _build_request(
        self, run_id: int
    ) -> tuple[AgentModelRequest, dict[str, float]] | None:
        request_started = perf_counter()
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                raise ValueError("agent run no longer exists")
            if run.status != "running":
                return None
            task = session.get(AgentTask, run.task_id)
            if not task:
                raise ValueError("agent task no longer exists")
            observations = [step.observation for step in run.steps if step.observation is not None]
            policy = get_or_create_task_policy(session, task)
            context_started = perf_counter()
            memory_context = assemble_memory_context(session, task, run_id=run.id)
            context_assembly_ms = _elapsed_ms(context_started)
            allowed = set(policy.allowed_tools)
            tools = [tool for tool in list_tools() if tool["name"] in allowed]
            request = AgentModelRequest(
                task_id=task.id,
                user_goal=task.user_goal,
                constraints=task.constraints,
                success_criteria=task.success_criteria,
                memory_context=memory_context.as_dict(),
                observations=observations,
                tools=tools,
                step_number=run.step_count + 1,
                max_steps=run.max_steps,
            )
            return request, {
                "context_assembly_ms": context_assembly_ms,
                "request_build_ms": _elapsed_ms(request_started),
            }

    @staticmethod
    def _validate_decision(decision: AgentDecision) -> None:
        if decision.action_type not in {"tool_call", "final_answer"}:
            raise ValueError(f"unsupported model action: {decision.action_type}")
        if decision.action_type == "tool_call" and (
            not decision.tool_name or decision.arguments is None
        ):
            raise ValueError("tool_call action requires tool_name and arguments")

    def _record_model_decision(
        self,
        run_id: int,
        request: AgentModelRequest,
        decision: AgentDecision,
        *,
        timing: dict[str, float],
    ) -> int:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run or run.status != "running":
                raise ValueError("agent run is not running")
            sequence = run.step_count + 1
            step_timing = {
                **timing,
                "step_total_ms": round(
                    timing["request_build_ms"] + timing["model_call_ms"], 3
                ),
            }
            step = AgentRunStep(
                run_id=run.id,
                task_id=run.task_id,
                sequence=sequence,
                action_type=decision.action_type,
                model_request=request.as_dict(),
                model_response=decision.as_dict(),
                timing_json=step_timing,
            )
            run.step_count = sequence
            session.add(step)
            session.add(
                ExecutionTrace(
                    task_id=run.task_id,
                    event_type="model_call",
                    status=decision.action_type,
                    input_summary=f"Agent run {run.id}, model step {sequence}.",
                    output_summary=(
                        f"Model proposed tool {decision.tool_name}."
                        if decision.action_type == "tool_call"
                        else "Model returned a final answer."
                    ),
                    metadata_json={
                        "agent_run_id": run.id,
                        "sequence": sequence,
                        "provider": run.provider,
                        "model": run.model,
                        "action": decision.as_dict(),
                        "timing": step_timing,
                    },
                )
            )
            session.commit()
            session.refresh(step)
            return step.id

    def _execute_tool(self, run_id: int, step_id: int, decision: AgentDecision) -> None:
        tool_started = perf_counter()
        tool_trace_id = None
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            step = session.get(AgentRunStep, step_id)
            task = session.get(AgentTask, run.task_id) if run else None
            if not run or not task or not step:
                raise ValueError("agent run state is incomplete")
            try:
                outcome = self.tool_runner(
                    session,
                    task=task,
                    tool_name=decision.tool_name,
                    arguments=decision.arguments,
                    requested_idempotency_key=(
                        f"agent-run:{run.id}:step:{step.sequence}:"
                        f"{decision.call_id or uuid4().hex}"
                    ),
                    plan_step_id=None,
                    record_linker=lambda active_session, record: setattr(
                        active_session.get(AgentRunStep, step_id),
                        "tool_call_id",
                        record.id,
                    ),
                )
                observation = json_safe(
                    serialize_tool_call(outcome.record, replayed=outcome.replayed)
                )
                step.tool_call_id = outcome.record.id
                tool_trace_id = outcome.record.trace_id
            except Exception as exc:
                session.rollback()
                step = session.get(AgentRunStep, step_id)
                observation = {
                    "tool_name": decision.tool_name,
                    "status": "error",
                    "output": None,
                    "error": str(exc),
                }
            step.observation = observation
            timing = dict(step.timing_json or {})
            timing["tool_call_ms"] = _elapsed_ms(tool_started)
            timing["step_total_ms"] = round(
                timing.get("request_build_ms", 0)
                + timing.get("model_call_ms", 0)
                + timing["tool_call_ms"],
                3,
            )
            step.timing_json = timing
            if tool_trace_id:
                trace = session.get(ExecutionTrace, tool_trace_id)
                if trace:
                    trace.metadata_json = {
                        **(trace.metadata_json or {}),
                        "agent_run_id": run.id,
                        "agent_run_step_id": step.id,
                        "sequence": step.sequence,
                    }
            session.commit()

    def _step_limit_reached(self, run_id: int) -> bool:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                raise ValueError("agent run no longer exists")
            return run.step_count >= run.max_steps

    def _complete_run(self, run_id: int, answer: str) -> None:
        finalize_started = perf_counter()
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                raise ValueError("agent run no longer exists")
            run.status = "completed"
            run.final_answer = answer
            run.stop_reason = "final_answer"
            memory_evaluations = evaluate_and_store_run_memories(session, run, answer=answer)
            memories = [evaluation.memory for evaluation in memory_evaluations if evaluation.memory]
            episodic_memory = next(
                (
                    evaluation.memory
                    for evaluation in memory_evaluations
                    if evaluation.candidate.memory_type == "episodic" and evaluation.memory
                ),
                None,
            )
            retired_memory_ids = retire_run_working_memories(
                session,
                run_id=run.id,
                reason="agent_run_completed",
            )
            memory_finalize_ms = _elapsed_ms(finalize_started)
            _finalize_run_timing(run, memory_finalize_ms=memory_finalize_ms)
            for evaluation in memory_evaluations:
                memory = evaluation.memory
                candidate = evaluation.candidate
                session.add(
                    ExecutionTrace(
                        task_id=run.task_id,
                        event_type="memory",
                        status=evaluation.decision,
                        input_summary=(
                            f"Evaluate {candidate.memory_type} candidate from Agent run {run.id}."
                        ),
                        output_summary=(
                            f"Accepted {candidate.memory_type} candidate; "
                            f"storage action: {evaluation.storage_action}."
                            if evaluation.decision == "accept"
                            else f"Candidate decision: {evaluation.decision}."
                        ),
                        metadata_json={
                            "agent_run_id": run.id,
                            "decision": evaluation.decision,
                            "storage_action": evaluation.storage_action,
                            "reasons": list(evaluation.reasons),
                            "candidate_record_id": (
                                evaluation.candidate_record.id
                                if evaluation.candidate_record
                                else None
                            ),
                            "memory_id": memory.id if memory else None,
                            "memory_type": candidate.memory_type,
                            "memory_key": candidate.memory_key,
                            "candidate_content": candidate.content,
                            "candidate_source": candidate.source,
                            "duplicate_memory_id": evaluation.duplicate_memory_id,
                            "provenance": candidate.provenance,
                        },
                    )
                )
            session.add(
                ExecutionTrace(
                    task_id=run.task_id,
                    event_type="agent_run",
                    status="completed",
                    input_summary=f"Agent run {run.id} returned a final answer.",
                    output_summary=answer,
                    metadata_json={
                        "agent_run_id": run.id,
                        "episodic_memory_id": episodic_memory.id if episodic_memory else None,
                        "memory_ids": [memory.id for memory in memories],
                        "memory_decisions": [
                            {
                                "memory_type": evaluation.candidate.memory_type,
                                "decision": evaluation.decision,
                                "storage_action": evaluation.storage_action,
                                "reasons": list(evaluation.reasons),
                            }
                            for evaluation in memory_evaluations
                        ],
                        "retired_working_memory_ids": retired_memory_ids,
                        "timing": run.timing_json,
                    },
                )
            )
            session.commit()

    def _stop_at_limit(self, run_id: int) -> None:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                raise ValueError("agent run no longer exists")
            run.status = "max_steps"
            run.stop_reason = "max_steps_reached"
            run.completed_at = datetime.utcnow()
            retired_memory_ids = retire_run_working_memories(
                session,
                run_id=run.id,
                reason="agent_run_max_steps",
            )
            _finalize_run_timing(run)
            session.add(
                ExecutionTrace(
                    task_id=run.task_id,
                    event_type="agent_run",
                    status="max_steps",
                    input_summary=f"Agent run {run.id} reached its execution limit.",
                    output_summary=f"Stopped after {run.step_count} model step(s).",
                    metadata_json={
                        "agent_run_id": run.id,
                        "max_steps": run.max_steps,
                        "retired_working_memory_ids": retired_memory_ids,
                        "timing": run.timing_json,
                    },
                )
            )
            session.commit()

    def _mark_failed(self, run_id: int, exc: Exception) -> None:
        with self.session_factory() as session:
            run = session.get(AgentRun, run_id)
            if not run:
                return
            run.status = "failed"
            run.stop_reason = "runtime_error"
            run.error = f"{type(exc).__name__}: {exc}"
            retired_memory_ids = retire_run_working_memories(
                session,
                run_id=run.id,
                reason="agent_run_failed",
            )
            _finalize_run_timing(run, failed_phase=self._active_phase)
            session.add(
                ExecutionTrace(
                    task_id=run.task_id,
                    event_type="agent_run",
                    status="failed",
                    input_summary=f"Agent run {run.id} failed.",
                    output_summary=run.error,
                    metadata_json={
                        "agent_run_id": run.id,
                        "retired_working_memory_ids": retired_memory_ids,
                        "timing": run.timing_json,
                    },
                )
            )
            session.commit()


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _finalize_run_timing(
    run: AgentRun,
    *,
    memory_finalize_ms: float = 0,
    failed_phase: str | None = None,
) -> None:
    run.completed_at = datetime.utcnow()
    step_timings = [step.timing_json or {} for step in run.steps]
    request_build_ms = round(
        sum(item.get("request_build_ms", 0) for item in step_timings), 3
    )
    context_assembly_ms = round(
        sum(item.get("context_assembly_ms", 0) for item in step_timings), 3
    )
    model_call_ms = round(
        sum(item.get("model_call_ms", 0) for item in step_timings), 3
    )
    tool_call_ms = round(
        sum(item.get("tool_call_ms", 0) for item in step_timings), 3
    )
    wall_clock_ms = round(
        max((run.completed_at - run.started_at).total_seconds() * 1000, 0), 3
    )
    attributed_ms = request_build_ms + model_call_ms + tool_call_ms + memory_finalize_ms
    timing = {
        "wall_clock_ms": wall_clock_ms,
        "context_assembly_ms": context_assembly_ms,
        "request_build_ms": request_build_ms,
        "model_call_ms": model_call_ms,
        "tool_call_ms": tool_call_ms,
        "memory_finalize_ms": memory_finalize_ms,
        "unattributed_ms": round(max(wall_clock_ms - attributed_ms, 0), 3),
    }
    if failed_phase:
        timing["failed_phase"] = failed_phase
    run.timing_json = timing


def json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def create_agent_run(
    session: Session, *, task: AgentTask, model: AgentModel, max_steps: int
) -> AgentRun:
    run = AgentRun(
        task_id=task.id,
        provider=model.provider,
        model=model.model,
        status="running",
        max_steps=max_steps,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def start_agent_run(
    session: Session,
    *,
    task: AgentTask,
    provider: str,
    model_name: str | None,
    max_steps: int,
    model_override: AgentModel | None = None,
) -> AgentRun:
    """Run the engine directly, without a workflow framework."""
    model = model_override or create_agent_model(provider, model_name)
    run = create_agent_run(session, task=task, model=model, max_steps=max_steps)
    AgentLoopEngine(model).run(run.id)
    return get_agent_run(session, run.id)


def get_agent_run(session: Session, run_id: int) -> AgentRun:
    session.expire_all()
    run = session.scalar(
        select(AgentRun)
        .options(selectinload(AgentRun.steps))
        .where(AgentRun.id == run_id)
    )
    if not run:
        raise ValueError("agent run not found")
    return run


def list_agent_runs(session: Session, task_id: int) -> list[AgentRun]:
    return list(
        session.scalars(
            select(AgentRun)
            .options(selectinload(AgentRun.steps))
            .where(AgentRun.task_id == task_id)
            .order_by(AgentRun.id.desc())
        )
    )
