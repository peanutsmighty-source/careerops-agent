from sqlalchemy import create_engine, inspect

from app.schema_compat import ensure_agent_timing_columns, ensure_memory_scope_columns


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
    }
    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT content, scope_type, task_id, run_id, status, retired_at, "
            "retirement_reason FROM agent_memories WHERE id = 1"
        ).one()
    assert row == ("legacy outcome", "task", 9, None, "active", None, None)


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
