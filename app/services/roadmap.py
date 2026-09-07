from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadmapTemplate:
    learning_goal: str
    project_idea: str
    evidence: str


DEFAULT_TEMPLATE = RoadmapTemplate(
    learning_goal="Build enough working knowledge to explain the concept and use it in a small Agent project.",
    project_idea="Create a focused demo that uses this skill in a realistic Agent workflow.",
    evidence="GitHub repository with README, tests, and a short demo trace.",
)


ROADMAP_TEMPLATES: dict[str, RoadmapTemplate] = {
    "Agent Harness": RoadmapTemplate(
        learning_goal="Understand Agent Loop, state management, tool execution, retries, and long-running task control.",
        project_idea="Build a mini Agent Harness with task state, tool calls, retry handling, and execution logs.",
        evidence="Repo with a runnable harness demo, trace examples, and tests for failure recovery.",
    ),
    "Tool Calling": RoadmapTemplate(
        learning_goal="Implement tool registration, argument validation, permission checks, and call logging.",
        project_idea="Build a Tool Registry that exposes search, read, and write tools to an Agent loop.",
        evidence="Unit tests for valid calls, invalid arguments, permission denial, and call history.",
    ),
    "Context Engineering": RoadmapTemplate(
        learning_goal="Learn how to preserve task intent, compress history, and hand off context across long tasks.",
        project_idea="Build a context compaction module that turns a long Agent trace into a concise task summary.",
        evidence="Before/after trace examples plus tests that important constraints survive compaction.",
    ),
    "LLM Evaluation": RoadmapTemplate(
        learning_goal="Design repeatable checks for task success, latency, cost, and regression behavior.",
        project_idea="Build a small Agent Eval harness with golden tasks, scoring rules, and a regression report.",
        evidence="Evaluation report generated from several sample Agent runs.",
    ),
    "Memory": RoadmapTemplate(
        learning_goal="Understand short-term state, long-term memory, retrieval, and when memory should be updated.",
        project_idea="Add memory to the mini Agent Harness with explicit read/write rules.",
        evidence="Demo showing memory creation, retrieval, update, and refusal to store unsafe or irrelevant data.",
    ),
    "Multi-agent Systems": RoadmapTemplate(
        learning_goal="Learn task decomposition, role boundaries, handoff protocol, and shared state risks.",
        project_idea="Build a two-agent workflow where a planner creates tasks and an executor runs tool calls.",
        evidence="Trace showing planner/executor handoff, shared context, and final verification.",
    ),
    "MCP": RoadmapTemplate(
        learning_goal="Understand how tools and resources are exposed to Agents through a standard protocol.",
        project_idea="Build a tiny MCP-style server that exposes a job database search tool.",
        evidence="Server code, client call examples, and tests for tool schema validation.",
    ),
    "RAG": RoadmapTemplate(
        learning_goal="Learn retrieval, chunking, relevance, citations, and how RAG combines with Agent workflows.",
        project_idea="Build a JD knowledge search feature that retrieves matching requirements for a target skill.",
        evidence="Search examples with retrieved snippets and a short explanation of ranking behavior.",
    ),
    "LangChain": RoadmapTemplate(
        learning_goal="Understand chains, tools, retrievers, and where framework abstractions help or get in the way.",
        project_idea="Rebuild one mini Agent workflow with LangChain and compare it with your hand-written version.",
        evidence="Comparison notes plus tests that both versions complete the same task.",
    ),
    "LangGraph": RoadmapTemplate(
        learning_goal="Model Agent workflows as explicit state graphs with branching, retries, and checkpoints.",
        project_idea="Build a LangGraph workflow for JD ingestion: parse, validate, dedupe, and save.",
        evidence="Graph diagram, runnable workflow, and tests for retry/error paths.",
    ),
    "Go": RoadmapTemplate(
        learning_goal="Gain enough Go backend fluency to read and implement service code in Agent infrastructure teams.",
        project_idea="Build a small Go HTTP service for tool execution logs.",
        evidence="Go service repo with endpoints, tests, and a short load test result.",
    ),
    "Python": RoadmapTemplate(
        learning_goal="Strengthen Python backend, testing, typing, and data processing for Agent applications.",
        project_idea="Extend CareerOps with one well-tested ingestion or analytics feature.",
        evidence="Merged feature with pytest coverage and a clear README example.",
    ),
}


def build_learning_roadmap(skill_demand: list[dict], limit: int = 8) -> list[dict]:
    roadmap = []
    for rank, skill in enumerate(skill_demand[:limit], start=1):
        template = ROADMAP_TEMPLATES.get(skill["name"], DEFAULT_TEMPLATE)
        roadmap.append(
            {
                "rank": rank,
                "skill_id": skill["skill_id"],
                "skill": skill["name"],
                "category": skill["category"],
                "job_count": skill["job_count"],
                "requirement_count": skill["requirement_count"],
                "why": (
                    f"Appears in {skill['job_count']} job(s) and "
                    f"{skill['requirement_count']} extracted requirement(s)."
                ),
                "learning_goal": template.learning_goal,
                "project_idea": template.project_idea,
                "evidence": template.evidence,
            }
        )
    return roadmap


def build_learning_tasks(skill_demand: list[dict], limit: int = 5) -> list[dict]:
    tasks = []
    roadmap = build_learning_roadmap(skill_demand, limit=limit)
    for item in roadmap:
        tasks.extend(
            [
                {
                    "rank": len(tasks) + 1,
                    "skill_id": item["skill_id"],
                    "skill": item["skill"],
                    "task_type": "project",
                    "title": f"Build: {item['project_idea']}",
                    "why": item["why"],
                    "acceptance_criteria": _project_acceptance_criteria(item["skill"]),
                    "deliverable": item["evidence"],
                },
                {
                    "rank": len(tasks) + 2,
                    "skill_id": item["skill_id"],
                    "skill": item["skill"],
                    "task_type": "evidence",
                    "title": f"Document evidence for {item['skill']}",
                    "why": "Job applications need visible proof, not only private learning.",
                    "acceptance_criteria": [
                        "README explains the job requirement this project targets.",
                        "README includes setup and run instructions.",
                        "A short demo trace or screenshot shows the feature working.",
                        "Tests or validation steps are documented.",
                    ],
                    "deliverable": "Portfolio-ready README section plus demo trace.",
                },
            ]
        )
    return tasks


def _project_acceptance_criteria(skill: str) -> list[str]:
    criteria_by_skill = {
        "Agent Harness": [
            "A user goal can be represented as a task with explicit state.",
            "The harness can call at least one registered tool.",
            "Every step is recorded in an execution trace.",
            "A failed tool call is retried or converted into a clear error state.",
            "At least one pytest test verifies the happy path.",
        ],
        "Tool Calling": [
            "Tools are registered with names, descriptions, and argument schemas.",
            "Invalid arguments are rejected before execution.",
            "Every tool call records input, output, and status.",
            "At least one permission or safety boundary is enforced.",
            "Tests cover success, validation failure, and blocked calls.",
        ],
        "Context Engineering": [
            "A long trace can be compacted into a shorter context summary.",
            "The summary preserves user goal, constraints, completed steps, and open issues.",
            "The compactor has tests for losing critical constraints.",
            "Before and after context examples are included.",
        ],
        "LLM Evaluation": [
            "At least three golden tasks are defined.",
            "Each task has an expected outcome and scoring rule.",
            "The evaluation runner produces a pass/fail report.",
            "A regression example is documented.",
        ],
        "Memory": [
            "Memory entries have clear read and write rules.",
            "The demo shows memory retrieval changing a later action.",
            "Irrelevant or unsafe memory writes can be rejected.",
            "Tests cover create, retrieve, update, and reject paths.",
        ],
    }
    return criteria_by_skill.get(
        skill,
        [
            "A small working demo uses the skill in an Agent-related workflow.",
            "The implementation includes at least one automated test.",
            "The README explains what was learned and what tradeoffs were made.",
            "A demo trace or sample output is included.",
        ],
    )
