from __future__ import annotations

import os
import socket
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import SessionLocal
from app.models import AgentRun, AgentTask, ExecutionTrace
from app.services.agent_run_lease import claimed_agent_run_lease
from app.services.agent_run_recovery import recover_agent_run
from app.services.agent_workflow import execute_agent_workflow_run


DEFAULT_WORKER_COUNT = int(os.getenv("CAREEROPS_AGENT_WORKERS", "2"))
DEFAULT_STARTUP_STALE_SECONDS = int(
    os.getenv("CAREEROPS_STARTUP_RECOVERY_SECONDS", "60")
)


class AgentRunWorkerPool:
    def __init__(self) -> None:
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4()}"
        self._executor: ThreadPoolExecutor | None = None
        self._futures: set[Future] = set()
        self._lock = Lock()

    def start(self) -> None:
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=DEFAULT_WORKER_COUNT,
                    thread_name_prefix="careerops-agent",
                )

    def shutdown(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def submit(self, run_id: int, *, recovery: bool = False) -> bool:
        self.start()
        with self._lock:
            assert self._executor is not None
            future = self._executor.submit(self._execute, run_id, recovery)
            self._futures.add(future)
        future.add_done_callback(self._discard)
        return True

    def schedule_startup_recovery(
        self, *, stale_after_seconds: int = DEFAULT_STARTUP_STALE_SECONDS
    ) -> list[int]:
        stale_before = datetime.utcnow() - timedelta(seconds=stale_after_seconds)
        with SessionLocal() as session:
            candidates = list(session.scalars(
                select(AgentRun).options(selectinload(AgentRun.steps)).where(
                    AgentRun.status == "running"
                ).order_by(AgentRun.id)
            ))
            run_ids = [
                run.id for run in candidates
                if (run.steps[-1].created_at if run.steps else run.started_at) <= stale_before
            ]
        for run_id in run_ids:
            self.submit(run_id, recovery=True)
        return run_ids

    def wait_for_idle(self, timeout: float = 10) -> None:
        with self._lock:
            futures = list(self._futures)
        for future in futures:
            future.result(timeout=timeout)

    def _discard(self, future: Future) -> None:
        with self._lock:
            self._futures.discard(future)

    def _execute(self, run_id: int, recovery: bool) -> None:
        owner = f"{self.worker_id}:run:{run_id}:{uuid4()}"
        with claimed_agent_run_lease(
            SessionLocal, run_id=run_id, owner=owner
        ) as lease:
            if not lease.acquired:
                return
            try:
                if recovery:
                    with SessionLocal() as session:
                        run = session.get(AgentRun, run_id)
                        if not run or run.status != "running":
                            return
                        task = session.get(AgentTask, run.task_id)
                        if not task:
                            raise ValueError("AgentRun task no longer exists")
                        recover_agent_run(session, task, run)
                else:
                    execute_agent_workflow_run(run_id, lease_owner=owner)
            except Exception as exc:
                self._record_failure(run_id, owner, str(exc))

    @staticmethod
    def _record_failure(run_id: int, owner: str, reason: str) -> None:
        with SessionLocal() as session:
            run = session.get(AgentRun, run_id)
            if (
                not run
                or run.status != "running"
                or run.lease_owner != owner
            ):
                return
            run.status = "failed"
            run.stop_reason = "worker_failed"
            run.error = reason
            run.completed_at = datetime.utcnow()
            session.add(ExecutionTrace(
                task_id=run.task_id,
                event_type="agent_run",
                status="failed",
                input_summary=f"Execute AgentRun {run.id} in a background worker.",
                output_summary=reason,
                metadata_json={"agent_run_id": run.id, "worker_failure": True},
            ))
            session.commit()


agent_run_workers = AgentRunWorkerPool()
