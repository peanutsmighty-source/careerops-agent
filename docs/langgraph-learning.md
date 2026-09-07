# LangGraph Learning Notes

This phase runs a deterministic CareerOps workflow before adding an LLM. Keeping the first graph deterministic makes State, Node, Edge, and checkpoint behavior visible without model variability.

## Concrete task

The graph handles one task: read parsed JD demand, select the top three skills, build a learning plan, validate it, and finish.

```text
START
  -> load_skill_demand
  -> build_plan
  -> validate_plan
       -> human_review -- approve --> complete_run -> END
                       -- reject  --> block_run    -> END
       -> block_run on validation failure         -> END
```

## State

`CareerOpsState` is the graph's current task snapshot. It is not the SQLAlchemy `AgentTask` row and it is not long-term memory. It contains the data needed to continue this one execution:

```json
{
  "task_id": 1,
  "user_goal": "Use real JD demand to choose the next three Agent skills.",
  "top_skills": [],
  "learning_plan": [],
  "validation_errors": [],
  "status": "started",
  "node_history": []
}
```

Each node returns only the fields it changes. LangGraph merges those updates into the next State snapshot.

## Nodes and edges

A Node is an executable step. In this graph, `load_skill_demand` queries SQLite and `build_plan` calls the existing deterministic roadmap service. A Node can later contain an LLM call or a tool call, but being a Node does not make it an Agent.

An Edge is a transition rule. The normal edge after `build_plan` always goes to `validate_plan`. The conditional edge after validation inspects `validation_errors`: an empty list goes to `complete_run`; any error goes to `block_run`.

This separation matters: nodes perform work; edges control what is allowed to run next.

## What a checkpoint looks like

After `build_plan` finishes, the SQLite checkpointer stores a snapshot similar to:

```json
{
  "checkpoint_id": "1f...",
  "step": 2,
  "next_nodes": ["validate_plan"],
  "state": {
    "task_id": 1,
    "status": "plan_built",
    "top_skills": [
      {"name": "Agent Harness", "job_count": 2}
    ],
    "learning_plan": [
      {"skill": "Agent Harness", "learning_goal": "..."}
    ],
    "node_history": ["load_skill_demand", "build_plan"]
  }
}
```

The important fields are:

- `state`: the data already produced.
- `next_nodes`: where execution should continue.
- `checkpoint_id`: the identity of this historical snapshot.
- `step`: its position in the graph execution.

`thread_id` groups all checkpoints for one Agent task. CareerOps derives it from the task ID and creation time. Calling the completed graph again with the same thread returns the final State instead of repeating the database query and plan generation.

## Interrupt and resume

`human_review` calls `interrupt()` before accepting the generated plan. The first graph request stops there and returns:

```json
{
  "awaiting_approval": true,
  "next_nodes": ["human_review"],
  "interrupt": {
    "question": "Do you approve this JD-driven learning plan?",
    "learning_plan": ["..."]
  },
  "state": {
    "status": "plan_validated",
    "approved": null
  }
}
```

The checkpoint says `next_nodes: ["human_review"]` because the node has started but has not completed. Its returned State update does not exist yet.

Approval is submitted through the resume endpoint. CareerOps invokes the same graph thread with `Command(resume={"approved": true, ...})`. LangGraph enters `human_review` again from the start; this time `interrupt()` returns the supplied dictionary. The node writes `approved: true`, and the conditional edge selects `complete_run`.

Because an interrupted node starts again, code before `interrupt()` can run more than once. External writes before that point must be absent or idempotent. This demo performs no write before the interrupt.

The business database and checkpoint database are separate:

- `careerops.db` contains jobs, skills, Agent tasks, and audit traces.
- `careerops_checkpoints.db` contains LangGraph execution snapshots.

## Try it through the API

1. Create an `AgentTask` whose success criteria require a JD-driven learning plan.
2. Call `POST /agent/tasks/{task_id}/run-learning-graph` and observe `awaiting_approval: true`.
3. Call `GET /agent/tasks/{task_id}/graph-checkpoints` and find the snapshot whose next node is `human_review`.
4. Call `POST /agent/tasks/{task_id}/resume-learning-graph` with `{"approved": true}` or `{"approved": false}`.
5. Compare the final `node_history`: approval ends in `complete_run`; rejection ends in `block_run`.
