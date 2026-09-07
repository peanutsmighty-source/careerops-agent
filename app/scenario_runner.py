from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    description: str
    run: Callable[["TestClient"], dict]


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _graph_no_skills(client: TestClient) -> dict:
    task_response = client.post(
        "/agent/tasks",
        json={
            "title": "Graph without parsed JD skills",
            "user_goal": "Build a JD-driven plan when no JD has been imported.",
            "success_criteria": ["Block instead of inventing unsupported skills."],
        },
    )
    _expect(task_response.status_code == 201, "AgentTask creation failed")
    task = task_response.json()

    run_response = client.post(f"/agent/tasks/{task['id']}/run-learning-graph")
    _expect(run_response.status_code == 200, "learning graph request failed")
    result = run_response.json()
    _expect(result["state"]["status"] == "blocked", "graph did not enter blocked state")
    _expect(result["awaiting_approval"] is False, "invalid plan requested approval")
    _expect(
        "No parsed JD skills are available." in result["state"]["validation_errors"],
        "missing-skill validation error was not recorded",
    )
    _expect(
        result["state"]["node_history"][-2:] == ["validate_plan", "block_run"],
        "graph took the wrong conditional edge",
    )
    stored_task = client.get(f"/agent/tasks/{task['id']}").json()
    _expect(stored_task["status"] == "blocked", "task status was not synchronized")
    return {
        "task_id": task["id"],
        "final_status": result["state"]["status"],
        "node_history": result["state"]["node_history"],
        "validation_errors": result["state"]["validation_errors"],
        "checkpoint_count": result["checkpoint_count"],
    }


def _jd_url_dedupe(client: TestClient) -> dict:
    payload = {
        "source_url": "https://example.com/jobs/runtime?utm_campaign=first",
        "raw_content": "Company: Scenario AI\nTitle: Agent Runtime Engineer\nPython and LangGraph.",
    }
    first_response = client.post("/job-archives", json=payload)
    second_response = client.post(
        "/job-archives",
        json={
            **payload,
            "source_url": "https://example.com/jobs/runtime?utm_source=second",
        },
    )
    _expect(first_response.status_code == 201, "first JD archive failed")
    _expect(second_response.status_code == 201, "duplicate JD archive request failed")
    first = first_response.json()
    second = second_response.json()
    _expect(first["id"] == second["id"], "canonical URLs created two archive rows")
    _expect(second["status"] == "duplicate", "duplicate status was not returned")
    archives = client.get("/job-archives").json()
    _expect(len(archives) == 1, "duplicate request persisted an extra archive")
    return {
        "archive_id": first["id"],
        "canonical_url": first["canonical_url"],
        "second_status": second["status"],
        "persisted_archive_count": len(archives),
    }


def _graph_human_rejection(client: TestClient) -> dict:
    archive_response = client.post(
        "/job-archives",
        json={
            "source_url": "https://example.com/jobs/rejection-scenario",
            "raw_content": "Company: Scenario AI\nTitle: Agent Engineer\nBuild Agent Harness systems.",
        },
    )
    _expect(archive_response.status_code == 201, "JD setup failed")
    task = client.post(
        "/agent/tasks",
        json={
            "title": "Reject a generated plan",
            "user_goal": "Review a JD-driven learning plan.",
            "success_criteria": ["Record the user's explicit review decision."],
        },
    ).json()
    started = client.post(f"/agent/tasks/{task['id']}/run-learning-graph").json()
    _expect(started["awaiting_approval"] is True, "graph did not pause for review")
    resumed = client.post(
        f"/agent/tasks/{task['id']}/resume-learning-graph",
        json={"approved": False, "comment": "The plan is too broad."},
    )
    _expect(resumed.status_code == 200, "graph resume failed")
    result = resumed.json()
    _expect(result["state"]["status"] == "blocked", "rejection did not block the graph")
    _expect(result["state"]["approved"] is False, "review decision was not persisted")
    return {
        "task_id": task["id"],
        "final_status": result["state"]["status"],
        "approved": result["state"]["approved"],
        "approval_comment": result["state"]["approval_comment"],
        "node_history": result["state"]["node_history"],
        "checkpoint_count": result["checkpoint_count"],
    }


SCENARIOS = {
    definition.name: definition
    for definition in [
        ScenarioDefinition(
            "graph-no-skills",
            "Run the learning graph with no imported JD skills and verify the blocked edge.",
            _graph_no_skills,
        ),
        ScenarioDefinition(
            "jd-url-dedupe",
            "Archive tracking variants of one JD URL and verify one persisted row.",
            _jd_url_dedupe,
        ),
        ScenarioDefinition(
            "graph-human-rejection",
            "Pause a valid graph for review, reject it, and verify the blocked edge.",
            _graph_human_rejection,
        ),
    ]
}


def scenario_catalog() -> list[dict]:
    return [
        {"name": definition.name, "description": definition.description}
        for definition in SCENARIOS.values()
    ]


def run_scenario(name: str) -> dict:
    definition = SCENARIOS.get(name)
    if not definition:
        raise KeyError(name)
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        result = definition.run(client)
    return {
        "scenario": definition.name,
        "description": definition.description,
        "status": "passed",
        "result": result,
    }


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in SCENARIOS:
        print("usage: python -m app.scenario_runner <scenario-name>", file=sys.stderr)
        return 2
    try:
        print(json.dumps(run_scenario(sys.argv[1]), ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "scenario": sys.argv[1],
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
