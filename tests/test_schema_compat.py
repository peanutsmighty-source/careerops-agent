from sqlalchemy import create_engine, inspect

from app.schema_compat import (
    ensure_agent_timing_columns,
    ensure_memory_candidate_columns,
    ensure_memory_revision_history,
    ensure_memory_scope_columns,
)


def test_memory_scope_upgrade_preserves_and_backfills_old_sqlite_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE agent_runs (id INTEGER PRIMARY KEY, task_id INTEGER)")
        connection.exec_driver_sql(
            "CREATE TABLE agent_memories ("
            "id INTEGER PRIMARY KEY, goal_contract_id INTEGER NOT NULL, "
            "memory_type VARCHAR(30) NOT NULL, memory_key VARCHAR(160) NOT NULL, "
            "content TEXT NOT NULL, source VARCHAR(80) NOT NULL, "
            "importance INTEGER NOT NULL)"
        )
        connection.exec_driver_sql("INSERT INTO agent_runs (id, task_id) VALUES (5, 9)")
        connection.exec_driver_sql(
            "INSERT INTO agent_memories "
            "(id, goal_contract_id, memory_type, memory_key, content, source, importance) "
            "VALUES (1, 1, 'episodic', 'agent-run:5:outcome', "
            "'legacy outcome', 'agent_runtime', 3)"
        )

    ensure_memory_scope_columns(engine)

    assert {column["name"] for column in inspect(engine).get_columns("agent_memories")} >= {
        "scope_type",
        "task_id",
        "run_id",
        "status",
        "retired_at",
        "retirement_reason",
        "version",
    }
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT content, scope_type, task_id, run_id, status, retired_at, "
            "retirement_reason, version FROM agent_memories WHERE id = 1"
        ).one()
    assert row == ("legacy outcome", "task", 9, None, "active", None, None, 1)


def test_memory_revision_upgrade_eagerly_backfills_legacy_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-revisions.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE agent_memories ("
            "id INTEGER PRIMARY KEY, content TEXT NOT NULL, source VARCHAR(80) NOT NULL, "
            "importance INTEGER NOT NULL, version INTEGER NOT NULL, "
            "status VARCHAR(20) NOT NULL, retired_at DATETIME, created_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE agent_memory_revisions ("
            "id INTEGER PRIMARY KEY, memory_id INTEGER NOT NULL, version INTEGER NOT NULL, "
            "content TEXT NOT NULL, source VARCHAR(80) NOT NULL, importance INTEGER NOT NULL, "
            "provenance JSON, change_reason VARCHAR(120) NOT NULL, "
            "valid_from DATETIME NOT NULL, valid_to DATETIME)"
        )
        connection.exec_driver_sql(
            "INSERT INTO agent_memories VALUES "
            "(1, 'legacy fact', 'user', 4, 1, 'active', NULL, '2026-01-02 03:04:05')"
        )

    ensure_memory_revision_history(engine)
    ensure_memory_revision_history(engine)

    with engine.connect() as connection:
        rows = connection.exec_driver_sql(
            "SELECT memory_id, version, content, change_reason, valid_from, valid_to "
            "FROM agent_memory_revisions"
        ).all()
    assert rows == [(1, 1, "legacy fact", "legacy_backfill", "2026-01-02 03:04:05", None)]


def test_memory_candidate_upgrade_removes_upsert_constraint_and_preserves_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-candidates.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE memory_candidates ("
            "id INTEGER PRIMARY KEY, goal_contract_id INTEGER NOT NULL, task_id INTEGER, "
            "source_run_id INTEGER, scope_run_id INTEGER, memory_id INTEGER, "
            "memory_type VARCHAR(30) NOT NULL, scope_type VARCHAR(20) NOT NULL, "
            "memory_key VARCHAR(160) NOT NULL, content TEXT NOT NULL, "
            "source VARCHAR(80) NOT NULL, importance INTEGER NOT NULL, provenance JSON NOT NULL, "
            "decision VARCHAR(30) NOT NULL, storage_action VARCHAR(30) NOT NULL, "
            "reasons JSON NOT NULL, evaluator_version VARCHAR(60) NOT NULL, "
            "evaluator_usage JSON, evaluator_output JSON, "
            "created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL, "
            "CONSTRAINT uq_memory_candidate_evaluation UNIQUE "
            "(goal_contract_id, memory_type, memory_key, evaluator_version))"
        )
        connection.exec_driver_sql(
            "INSERT INTO memory_candidates VALUES "
            "(1, 1, 2, NULL, NULL, NULL, 'fact', 'task', 'skill-demand', "
            "'3 JDs', 'tool:test', 4, '{}', 'accept', 'stored', '[]', "
            "'deterministic-v2', '{}', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        )

    ensure_memory_candidate_columns(engine)
    ensure_memory_candidate_columns(engine)

    assert "uq_memory_candidate_evaluation" not in {
        constraint["name"]
        for constraint in inspect(engine).get_unique_constraints("memory_candidates")
    }
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO memory_candidates "
            "SELECT 2, goal_contract_id, task_id, source_run_id, scope_run_id, memory_id, "
            "memory_type, scope_type, memory_key, '5 JDs', source, importance, provenance, "
            "decision, 'superseded', reasons, evaluator_version, evaluator_usage, "
            "evaluator_output, created_at, updated_at FROM memory_candidates WHERE id = 1"
        )
        assert connection.exec_driver_sql(
            "SELECT content FROM memory_candidates ORDER BY id"
        ).scalars().all() == ["3 JDs", "5 JDs"]


def test_agent_timing_upgrade_preserves_existing_runtime_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy-timing.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE agent_runs (id INTEGER PRIMARY KEY, task_id INTEGER)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE agent_run_steps (id INTEGER PRIMARY KEY, run_id INTEGER)"
        )
        connection.exec_driver_sql(
            "INSERT INTO agent_runs (id, task_id) VALUES (3, 7)"
        )
        connection.exec_driver_sql(
            "INSERT INTO agent_run_steps (id, run_id) VALUES (4, 3)"
        )

    ensure_agent_timing_columns(engine)

    assert "timing_json" in {
        column["name"] for column in inspect(engine).get_columns("agent_runs")
    }
    assert "timing_json" in {
        column["name"] for column in inspect(engine).get_columns("agent_run_steps")
    }
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT id, task_id, timing_json FROM agent_runs WHERE id = 3"
        ).one() == (3, 7, None)
