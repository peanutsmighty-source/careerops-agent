from __future__ import annotations

from sqlalchemy import Engine, inspect, text


def ensure_memory_scope_columns(engine: Engine) -> None:
    """Upgrade pre-scope SQLite development databases without losing local data."""
    columns = {column["name"] for column in inspect(engine).get_columns("agent_memories")}
    missing = {
        "scope_type",
        "task_id",
        "run_id",
        "status",
        "retired_at",
        "retirement_reason",
    } - columns
    if missing and engine.dialect.name != "sqlite":
        raise RuntimeError("agent_memories requires a database migration for memory scopes")
    if engine.dialect.name != "sqlite":
        return

    statements = {
        "scope_type": (
            "ALTER TABLE agent_memories ADD COLUMN scope_type VARCHAR(20) "
            "NOT NULL DEFAULT 'contract'"
        ),
        "task_id": "ALTER TABLE agent_memories ADD COLUMN task_id INTEGER",
        "run_id": "ALTER TABLE agent_memories ADD COLUMN run_id INTEGER",
        "status": (
            "ALTER TABLE agent_memories ADD COLUMN status VARCHAR(20) "
            "NOT NULL DEFAULT 'active'"
        ),
        "retired_at": "ALTER TABLE agent_memories ADD COLUMN retired_at DATETIME",
        "retirement_reason": "ALTER TABLE agent_memories ADD COLUMN retirement_reason TEXT",
    }
    with engine.begin() as connection:
        for column in (
            "scope_type",
            "task_id",
            "run_id",
            "status",
            "retired_at",
            "retirement_reason",
        ):
            if column in missing:
                connection.execute(text(statements[column]))
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_agent_memories_scope_type "
                "ON agent_memories (scope_type)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_agent_memories_task_id "
                "ON agent_memories (task_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_agent_memories_run_id "
                "ON agent_memories (run_id)"
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_agent_memories_status "
                "ON agent_memories (status)"
            )
        )
        connection.exec_driver_sql(
            "UPDATE agent_memories SET scope_type = 'task', "
            "task_id = (SELECT agent_runs.task_id FROM agent_runs "
            "WHERE agent_memories.memory_key = "
            "'agent-run:' || agent_runs.id || ':outcome') "
            "WHERE scope_type = 'contract' AND memory_type = 'episodic' "
            "AND source = 'agent_runtime' AND EXISTS "
            "(SELECT 1 FROM agent_runs WHERE agent_memories.memory_key = "
            "'agent-run:' || agent_runs.id || ':outcome')"
        )


def ensure_memory_candidate_columns(engine: Engine) -> None:
    """Upgrade SQLite candidate journals created before semantic evaluation."""
    columns = {
        column["name"] for column in inspect(engine).get_columns("memory_candidates")
    }
    if "evaluator_output" in columns:
        return
    if engine.dialect.name != "sqlite":
        raise RuntimeError(
            "memory_candidates requires a database migration for evaluator_output"
        )
    with engine.begin() as connection:
        connection.execute(
            text("ALTER TABLE memory_candidates ADD COLUMN evaluator_output JSON")
        )


def ensure_agent_timing_columns(engine: Engine) -> None:
    """Add runtime timing payloads to existing SQLite development databases."""
    missing_by_table = {}
    for table_name in ("agent_runs", "agent_run_steps"):
        columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
        if "timing_json" not in columns:
            missing_by_table[table_name] = "timing_json"
    if not missing_by_table:
        return
    if engine.dialect.name != "sqlite":
        raise RuntimeError("agent runtime timing columns require a database migration")
    with engine.begin() as connection:
        for table_name in missing_by_table:
            connection.execute(
                text(f"ALTER TABLE {table_name} ADD COLUMN timing_json JSON")
            )
