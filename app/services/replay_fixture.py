from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session

from app.models import ToolCallRecord, ToolReplayFixture


def capture_tool_before_state(
    session: Session, *, record: ToolCallRecord
) -> ToolReplayFixture | None:
    """Capture the committed SQLite state immediately before an internal write."""
    if record.effect != "internal_write":
        return None
    bind = session.get_bind()
    if not isinstance(bind, Engine) or bind.dialect.name != "sqlite":
        return None
    database = make_url(str(bind.url)).database
    if not database or database == ":memory:":
        return None

    source_path = Path(database).resolve()
    configured_root = os.getenv("CAREEROPS_REPLAY_FIXTURE_DIR", "").strip()
    fixture_root = (
        Path(configured_root).resolve()
        if configured_root
        else source_path.parent / ".careerops-replay"
    )
    fixture_root.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_root / (
        f"tool-call-{record.id}-attempt-{record.attempt_count}-{uuid4().hex}.db"
    )
    _backup_sqlite(bind, fixture_path)
    fixture = ToolReplayFixture(
        task_id=record.task_id,
        tool_call_id=record.id,
        attempt_number=record.attempt_count,
        database_path=str(fixture_path),
        database_sha256=_sha256_file(fixture_path),
    )
    session.add(fixture)
    session.commit()
    session.refresh(fixture)
    return fixture


def _backup_sqlite(engine: Engine, output: Path) -> None:
    raw_connection = engine.raw_connection()
    destination = sqlite3.connect(output)
    try:
        raw_connection.driver_connection.backup(destination)
    finally:
        destination.close()
        raw_connection.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
