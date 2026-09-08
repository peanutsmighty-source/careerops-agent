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
        "version",
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
        "version": (
            "ALTER TABLE agent_memories ADD COLUMN version INTEGER "
            "NOT NULL DEFAULT 1"
        ),
    }
    with engine.begin() as connection:
        for column in (
            "scope_type",
            "task_id",
            "run_id",
            "status",
            "retired_at",
            "retirement_reason",
            "version",
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
    """Upgrade SQLite candidate journals and remove the old upsert constraint."""
    inspector = inspect(engine)
    columns = {
        column["name"] for column in inspector.get_columns("memory_candidates")
    }
    unique_constraints = {
        constraint["name"]
        for constraint in inspector.get_unique_constraints("memory_candidates")
    }
    needs_output = "evaluator_output" not in columns
    needs_append_only = "uq_memory_candidate_evaluation" in unique_constraints
    if not needs_output and not needs_append_only:
        return
    if engine.dialect.name != "sqlite":
        raise RuntimeError("memory_candidates requires a database migration")
    with engine.begin() as connection:
        if needs_output:
            connection.execute(
                text("ALTER TABLE memory_candidates ADD COLUMN evaluator_output JSON")
            )
        if needs_append_only:
            _rebuild_memory_candidate_journal(connection)


def ensure_memory_revision_history(engine: Engine) -> None:
    """Give legacy SQLite memories an auditable first revision eagerly."""
    inspector = inspect(engine)
    if not inspector.has_table("agent_memories") or not inspector.has_table(
        "agent_memory_revisions"
    ):
        return
    if engine.dialect.name != "sqlite":
        return
    memory_columns = {
        column["name"] for column in inspector.get_columns("agent_memories")
    }
    valid_from = "memory.created_at" if "created_at" in memory_columns else "CURRENT_TIMESTAMP"
    valid_to = (
        "CASE WHEN memory.status = 'retired' THEN memory.retired_at END"
        if {"status", "retired_at"} <= memory_columns
        else "NULL"
    )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO agent_memory_revisions "
            "(memory_id, version, content, source, importance, provenance, "
            "change_reason, valid_from, valid_to) "
            "SELECT memory.id, memory.version, memory.content, memory.source, "
            "memory.importance, NULL, 'legacy_backfill', "
            f"{valid_from}, {valid_to} FROM agent_memories AS memory "
            "WHERE NOT EXISTS (SELECT 1 FROM agent_memory_revisions AS revision "
            "WHERE revision.memory_id = memory.id)"
        )


def _rebuild_memory_candidate_journal(connection) -> None:
    connection.exec_driver_sql(
        "CREATE TABLE memory_candidates_append_only ("
        "id INTEGER NOT NULL PRIMARY KEY, "
        "goal_contract_id INTEGER NOT NULL REFERENCES goal_contracts (id), "
        "task_id INTEGER REFERENCES agent_tasks (id), "
        "source_run_id INTEGER REFERENCES agent_runs (id), "
        "scope_run_id INTEGER REFERENCES agent_runs (id), "
        "memory_id INTEGER REFERENCES agent_memories (id), "
        "memory_type VARCHAR(30) NOT NULL, scope_type VARCHAR(20) NOT NULL, "
        "memory_key VARCHAR(160) NOT NULL, content TEXT NOT NULL, "
        "source VARCHAR(80) NOT NULL, importance INTEGER NOT NULL, "
        "provenance JSON NOT NULL, decision VARCHAR(30) NOT NULL, "
        "storage_action VARCHAR(30) NOT NULL, reasons JSON NOT NULL, "
        "evaluator_version VARCHAR(60) NOT NULL, evaluator_usage JSON, "
        "evaluator_output JSON, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
    )
    columns = (
        "id, goal_contract_id, task_id, source_run_id, scope_run_id, memory_id, "
        "memory_type, scope_type, memory_key, content, source, importance, provenance, "
        "decision, storage_action, reasons, evaluator_version, evaluator_usage, "
        "evaluator_output, created_at, updated_at"
    )
    connection.exec_driver_sql(
        f"INSERT INTO memory_candidates_append_only ({columns}) "
        f"SELECT {columns} FROM memory_candidates"
    )
    connection.exec_driver_sql("DROP TABLE memory_candidates")
    connection.exec_driver_sql(
        "ALTER TABLE memory_candidates_append_only RENAME TO memory_candidates"
    )
    for column in (
        "goal_contract_id",
        "task_id",
        "source_run_id",
        "memory_id",
        "memory_type",
        "scope_type",
        "source",
        "decision",
    ):
        connection.exec_driver_sql(
            f"CREATE INDEX ix_memory_candidates_{column} "
            f"ON memory_candidates ({column})"
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
