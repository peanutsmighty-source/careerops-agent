# Tool Calling in CareerOps

## 1. What was added

CareerOps now has a small Tool Runtime with four registered tools:

| Tool | Permission | Purpose |
| --- | --- | --- |
| `search_jobs` | read | Search real archived and parsed jobs. |
| `get_skill_demand` | read | Rank skills found across JDs. |
| `get_job` | read | Read one structured job and its full JD. |
| `update_plan_step` | write | Update a plan step owned by the current task. |

The registry is code, while every invocation is data. Definitions live in
`app/services/tools.py`; invocation history lives in `ExecutionTrace`.

## 2. One concrete call

Imagine a model returns this tool request:

```json
{
  "tool_name": "search_jobs",
  "arguments": {"query": "Agent", "limit": 5}
}
```

CareerOps processes it in this order:

1. **Registry lookup** finds the `search_jobs` definition. An unknown name becomes an error observation.
2. **Permission gate** loads the task's server-side `TaskPolicy` and records a `ToolAuthorization` decision.
3. **Argument validation** uses `SearchJobsArguments`. For example, `limit: 100` fails because the maximum is 20.
4. **Handler execution** queries the CareerOps database through SQLAlchemy.
5. **Observation** returns structured JSON such as `{"count": 2, "jobs": [...]}` to the caller.
6. **Trace recording** stores the name, validated arguments, output or error, permission decision, and duration.

The result is an observation, not another model answer. A future Agent Loop will append this observation to model context and let the model decide its next action.

## 3. Why a tool is more than a Python function

A normal Python function only has inputs and an output. An Agent tool also needs a public contract and a control boundary:

- A stable name and description so a model can select it.
- A JSON Schema so arguments can be generated and validated.
- A permission level so model intent is not treated as authority.
- A normalized success, error, or denied result.
- A trace so humans and evaluations can reconstruct what happened.

That surrounding machinery is part of the Harness.

## 4. Success, error, and denied are all observations

The API intentionally returns a structured Tool Call result for expected runtime failures:

```json
{
  "tool_name": "update_plan_step",
  "status": "denied",
  "output": null,
  "error": "tool 'update_plan_step' is not allowed by TaskPolicy v1"
}
```

The Agent Loop should be able to inspect that result and choose whether to ask for approval, repair its arguments, choose another tool, or stop. Throwing away the result as an unhandled server exception would remove useful reasoning evidence.

## 5. Current authorization boundary

The ToolCall request contains only the model's proposal: tool name, arguments, and operation identity. It cannot include `granted_permissions`; unknown request fields are rejected.

The server loads `TaskPolicy`, checks whether the tool is allowlisted, evaluates its effect, and persists one `ToolAuthorization` row before any handler runs. A replay is authorized again under the current policy before its saved result can be returned.

This is a real control-plane/data-plane split, but it is not yet a complete production security system. The local app does not authenticate different users, and an `external_write` decision that requires human approval is classified but does not yet have an approval UI and one-time grant.

## 6. Exercise in the Runtime Console

1. Select task `#1` and choose `search_jobs`.
2. Call it with `{"query":"Agent","limit":5}` and inspect the success trace.
3. Change `limit` to `100` and inspect the validation error trace.
4. Choose `update_plan_step` while its TaskPolicy toggle is off and inspect the denied authorization.
5. Enable the TaskPolicy toggle, make a new operation, and inspect the allowed authorization and successful result.
6. Compare the records under **Recent calls**.

This demonstrates the core Tool Calling path before an LLM is connected: selection contract, validation, authorization, execution, observation, and tracing.
