from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event, Thread
from typing import Callable, Iterator

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from app.models import AgentRun


DEFAULT_LEASE_SECONDS = 30


def acquire_agent_run_lease(
    session: Session,
    *,
    run_id: int,
    owner: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> bool:
    if not owner or lease_seconds < 3:
        raise ValueError("lease owner is required and lease_seconds must be at least 3")
    current = now or datetime.utcnow()
    result = session.execute(
        update(AgentRun)
        .where(
            AgentRun.id == run_id,
            AgentRun.status == "running",
            or_(
                AgentRun.lease_owner.is_(None),
                AgentRun.lease_expires_at.is_(None),
                AgentRun.lease_expires_at <= current,
                AgentRun.lease_owner == owner,
            ),
        )
        .values(
            lease_owner=owner,
            lease_expires_at=current + timedelta(seconds=lease_seconds),
            lease_heartbeat_at=current,
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount == 1


def renew_agent_run_lease(
    session: Session,
    *,
    run_id: int,
    owner: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.utcnow()
    result = session.execute(
        update(AgentRun)
        .where(
            AgentRun.id == run_id,
            AgentRun.status == "running",
            AgentRun.lease_owner == owner,
            AgentRun.lease_expires_at > current,
        )
        .values(
            lease_expires_at=current + timedelta(seconds=lease_seconds),
            lease_heartbeat_at=current,
        )
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount == 1


def release_agent_run_lease(session: Session, *, run_id: int, owner: str) -> bool:
    result = session.execute(
        update(AgentRun)
        .where(AgentRun.id == run_id, AgentRun.lease_owner == owner)
        .values(lease_owner=None, lease_expires_at=None, lease_heartbeat_at=None)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount == 1


@dataclass
class LeaseHandle:
    acquired: bool
    lost: Event


@contextmanager
def claimed_agent_run_lease(
    session_factory: Callable[[], Session],
    *,
    run_id: int,
    owner: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> Iterator[LeaseHandle]:
    with session_factory() as session:
        acquired = acquire_agent_run_lease(
            session, run_id=run_id, owner=owner, lease_seconds=lease_seconds
        )
    lost = Event()
    if not acquired:
        yield LeaseHandle(acquired=False, lost=lost)
        return

    stop = Event()

    def heartbeat() -> None:
        while not stop.wait(max(1.0, lease_seconds / 3)):
            try:
                with session_factory() as session:
                    if not renew_agent_run_lease(
                        session, run_id=run_id, owner=owner,
                        lease_seconds=lease_seconds,
                    ):
                        lost.set()
                        return
            except Exception:
                lost.set()
                return

    thread = Thread(target=heartbeat, name=f"agent-run-lease-{run_id}", daemon=True)
    thread.start()
    try:
        yield LeaseHandle(acquired=True, lost=lost)
    finally:
        stop.set()
        thread.join(timeout=2)
        with session_factory() as session:
            release_agent_run_lease(session, run_id=run_id, owner=owner)
