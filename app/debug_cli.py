from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session

from app.models import AgentTask, ExecutionTrace, ToolCallRecord
from app.scenario_runner import scenario_catalog
from app.services.tool_runtime import serialize_tool_call
from app.services.tools import ToolContext, execute_tool, get_tool


DEFAULT_DATABASE_URL = "sqlite:///./careerops.db"


class DebugCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayResult:
    tool_call_id: int
    tool_name: str
    source_status: str
    replay_status: str
    status_matches: bool
    output_matches: bool
    error_matches: bool
    replay_output: dict | None
    replay_error: str | None
    database_isolated: bool = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="careerops-debug",
        description="Inspect and safely replay persisted CareerOps tool calls.",
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="Source SQLAlchemy database URL (default: sqlite:///./careerops.db).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calls = subparsers.add_parser("calls", help="List recent persisted tool calls.")
    calls.add_argument("--task-id", type=int)
    calls.add_argument("--limit", type=int, default=20)

    show = subparsers.add_parser("show", help="Show one tool call and related traces.")
    show.add_argument("tool_call_id", type=int)

    snapshot = subparsers.add_parser(
        "snapshot", help="Create a consistent SQLite database copy and JSON manifest."
    )
    snapshot.add_argument("tool_call_id", type=int)
    snapshot.add_argument("--output", type=Path, required=True)

    replay = subparsers.add_parser(
        "replay", help="Execute saved arguments against an isolated SQLite copy."
    )
    replay.add_argument("tool_call_id", type=int)
    replay.add_argument(
        "--snapshot",
        type=Path,
        help="Replay from an existing snapshot instead of a temporary current-state copy.",
    )

    subparsers.add_parser("scenarios", help="List isolated cross-module scenarios.")
    scenario = subparsers.add_parser(
        "scenario", help="Run one scenario in a fresh process with temporary databases."
    )
    scenario.add_argument("name", choices=[item["name"] for item in scenario_catalog()])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "calls":
            _print_json(list_calls(args.database_url, args.task_id, args.limit))
        elif args.command == "show":
            _print_json(show_call(args.database_url, args.tool_call_id))
        elif args.command == "snapshot":
            _print_json(create_snapshot(args.database_url, args.tool_call_id, args.output))
        elif args.command == "replay":
            _print_json(
                asdict(
                    replay_call(
                        args.database_url,
                        args.tool_call_id,
                        snapshot_path=args.snapshot,
                    )
                )
            )
        elif args.command == "scenarios":
            _print_json(scenario_catalog())
        elif args.command == "scenario":
            _print_json(run_isolated_scenario(args.name))
    except DebugCliError as exc:
        print(f"careerops-debug: {exc}", file=sys.stderr)
        return 2
    return 0


def run_isolated_scenario(name: str) -> dict:
    known_names = {item["name"] for item in scenario_catalog()}
    if name not in known_names:
        raise DebugCliError(f"unknown scenario: {name}")
    with tempfile.TemporaryDirectory(prefix=f"careerops-{name}-") as directory:
        root = Path(directory)
        env = os.environ.copy()
        env.update(
            {
                "CAREEROPS_DATABASE_URL": f"sqlite:///{(root / 'scenario.db').as_posix()}",
                "CAREEROPS_CHECKPOINT_DB": str(root / "checkpoints.db"),
                "PYTHONIOENCODING": "utf-8",
            }
        )
        completed = subprocess.run(
            [sys.executable, "-m", "app.scenario_runner", name],
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise DebugCliError(
                f"scenario process returned invalid output: {completed.stdout[-500:]}"
            ) from exc
        if completed.returncode != 0:
            raise DebugCliError(result.get("error", "scenario failed"))
        result["database_isolated"] = True
        result["checkpoint_isolated"] = True
        return result


def list_calls(database_url: str, task_id: int | None, limit: int) -> list[dict]:
    if limit < 1 or limit > 200:
        raise DebugCliError("--limit must be between 1 and 200")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            statement = select(ToolCallRecord).order_by(ToolCallRecord.id.desc()).limit(limit)
            if task_id is not None:
                statement = statement.where(ToolCallRecord.task_id == task_id)
            return [_call_summary(record) for record in session.scalars(statement)]
    finally:
        engine.dispose()


def show_call(database_url: str, tool_call_id: int) -> dict:
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            record = _get_call(session, tool_call_id)
            task = session.get(AgentTask, record.task_id)
            traces = list(
                session.scalars(
                    select(ExecutionTrace)
                    .where(
                        ExecutionTrace.task_id == record.task_id,
                        ExecutionTrace.metadata_json["tool_call_id"].as_integer()
                        == record.id,
                    )
                    .order_by(ExecutionTrace.id)
                )
            )
            return {
                "task": {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status,
                }
                if task
                else None,
                "tool_call": serialize_tool_call(record),
                "traces": [_trace_data(trace) for trace in traces],
            }
    finally:
        engine.dispose()


def create_snapshot(database_url: str, tool_call_id: int, output: Path) -> dict:
    source_engine = create_engine(database_url)
    try:
        _require_sqlite(source_engine)
        with Session(source_engine) as session:
            record = _get_call(session, tool_call_id)
            summary = _call_summary(record)

        output = output.resolve()
        if output.suffix.lower() != ".db":
            raise DebugCliError("snapshot --output must end with .db")
        output.parent.mkdir(parents=True, exist_ok=True)
        source_path = _sqlite_path(source_engine).resolve()
        if output == source_path:
            raise DebugCliError("snapshot output must not overwrite the source database")
        _backup_sqlite(source_engine, output)
        manifest_path = output.with_suffix(".json")
        manifest = {
            "format": "careerops-tool-repro-v1",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "database_file": output.name,
            "database_sha256": _sha256_file(output),
            "tool_call": summary,
            "warning": "This snapshot contains a copy of CareerOps development data.",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return {
            "snapshot": str(output),
            "manifest": str(manifest_path),
            **manifest,
        }
    finally:
        source_engine.dispose()


def replay_call(
    database_url: str,
    tool_call_id: int,
    *,
    snapshot_path: Path | None = None,
) -> ReplayResult:
    if snapshot_path:
        snapshot = snapshot_path.resolve()
        if not snapshot.is_file():
            raise DebugCliError(f"snapshot does not exist: {snapshot}")
        return _replay_from_database(snapshot, tool_call_id)

    source_engine = create_engine(database_url)
    try:
        _require_sqlite(source_engine)
        with tempfile.TemporaryDirectory(prefix="careerops-replay-") as directory:
            snapshot = Path(directory) / "replay.db"
            _backup_sqlite(source_engine, snapshot)
            return _replay_from_database(snapshot, tool_call_id)
    finally:
        source_engine.dispose()


def _replay_from_database(database_path: Path, tool_call_id: int) -> ReplayResult:
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    try:
        with Session(engine) as session:
            record = _get_call(session, tool_call_id)
            task = session.get(AgentTask, record.task_id)
            if not task:
                raise DebugCliError(f"task {record.task_id} no longer exists")
            if record.effect == "external_write":
                raise DebugCliError(
                    "external_write tools are never replayed by the debug CLI; reconcile them instead"
                )
            definition = get_tool(record.tool_name)
            if not definition:
                raise DebugCliError(f"tool is not registered: {record.tool_name}")

            granted_permissions = _recorded_permissions(session, record)
            execution = execute_tool(
                ToolContext(session=session, task=task),
                tool_name=record.tool_name,
                arguments=record.arguments_json,
                granted_permissions=granted_permissions,
            )
            session.rollback()
            source_status = {
                "succeeded": "success",
                "failed": "error",
                "denied": "denied",
            }.get(record.status, record.status)
            return ReplayResult(
                tool_call_id=record.id,
                tool_name=record.tool_name,
                source_status=source_status,
                replay_status=execution.status,
                status_matches=source_status == execution.status,
                output_matches=record.output_json == execution.output,
                error_matches=record.error == execution.error,
                replay_output=execution.output,
                replay_error=execution.error,
            )
    finally:
        engine.dispose()


def _recorded_permissions(session: Session, record: ToolCallRecord) -> set[str]:
    trace = session.get(ExecutionTrace, record.trace_id) if record.trace_id else None
    metadata = trace.metadata_json if trace and trace.metadata_json else {}
    permissions = metadata.get("granted_permissions")
    if isinstance(permissions, list):
        return {str(permission) for permission in permissions}
    if record.authorizations and record.authorizations[-1].decision == "allowed":
        return {record.permission}
    return {"read"}


def _get_call(session: Session, tool_call_id: int) -> ToolCallRecord:
    record = session.get(ToolCallRecord, tool_call_id)
    if not record:
        raise DebugCliError(f"tool call {tool_call_id} was not found")
    return record


def _call_summary(record: ToolCallRecord) -> dict:
    return {
        "tool_call_id": record.id,
        "task_id": record.task_id,
        "tool_name": record.tool_name,
        "status": record.status,
        "effect": record.effect,
        "idempotency_key": record.idempotency_key,
        "request_fingerprint": record.request_fingerprint,
        "attempt_count": record.attempt_count,
        "replay_count": record.replay_count,
        "arguments": record.arguments_json,
        "error": record.error,
        "created_at": record.created_at,
    }


def _trace_data(trace: ExecutionTrace) -> dict:
    return {
        "id": trace.id,
        "status": trace.status,
        "input_summary": trace.input_summary,
        "output_summary": trace.output_summary,
        "metadata": trace.metadata_json,
        "created_at": trace.created_at,
    }


def _require_sqlite(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        raise DebugCliError("snapshot and replay currently support the development SQLite database")


def _sqlite_path(engine: Engine) -> Path:
    database = make_url(str(engine.url)).database
    if not database or database == ":memory:":
        raise DebugCliError("a file-backed SQLite database is required")
    return Path(database)


def _backup_sqlite(source_engine: Engine, output: Path) -> None:
    output.unlink(missing_ok=True)
    raw_connection = source_engine.raw_connection()
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


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
