const state = {
  tasks: [],
  selectedTaskId: null,
  task: null,
  checkpoints: [],
  selectedCheckpoint: null,
  tools: [],
  toolCalls: [],
  selectedToolCall: null,
  taskPolicy: null,
  agentRuns: [],
  selectedAgentRun: null,
  memoryContext: null,
  memories: [],
  selectedMemory: null,
  recentlyCreatedMemoryId: null,
  memoryCandidates: [],
  selectedMemoryCandidate: null,
};

const graphNodes = [
  ["load_skill_demand", "Load skills", "查询 JD 技能"],
  ["build_plan", "Build plan", "生成学习计划"],
  ["validate_plan", "Validate", "校验成功标准"],
  ["human_review", "Human review", "等待人工审批"],
  ["complete_run", "Complete", "完成或阻塞"],
];

const elements = Object.fromEntries([
  "server-status", "task-list", "empty-state", "task-view", "task-title", "task-goal",
  "task-status", "run-button", "graph-flow", "approval-bar", "approval-question",
  "approve-button", "reject-button", "skill-strip", "learning-plan", "checkpoint-count",
  "checkpoint-list", "state-json", "selected-step", "task-dialog", "task-form", "form-error",
  "new-task-button", "empty-create-button", "close-dialog", "cancel-dialog", "toast",
  "tool-form", "tool-select", "tool-description", "tool-arguments", "tool-permission",
  "write-permission-row", "grant-write", "tool-error", "call-tool-button", "tool-call-count",
  "tool-call-list", "tool-result", "tool-policy", "idempotency-row", "idempotency-key",
  "new-operation-key", "recover-tools-button",
  "agent-provider", "agent-model-row", "agent-model", "agent-max-steps",
  "run-agent-button", "agent-run-count", "agent-run-list", "agent-run-result",
  "recover-agent-runs-button",
  "refresh-memory-context-button", "memory-context-result",
  "memory-form", "memory-type", "memory-scope", "memory-importance", "memory-key",
  "memory-content", "memory-expires-at", "memory-error", "create-memory-button",
  "memory-count", "memory-list", "memory-created-notice", "memory-created-summary",
  "memory-detail", "retire-memory-button",
  "memory-candidate-count", "memory-candidate-list", "memory-candidate-detail",
].map((id) => [id, document.getElementById(id)]));

const toolArgumentExamples = {
  search_jobs: { query: "Agent", limit: 5 },
  get_skill_demand: { limit: 5 },
  get_job: { job_id: 1 },
  update_plan_step: { plan_step_id: 1, status: "in_progress", result_summary: "Started by Tool Executor." },
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `请求失败 (${response.status})`);
  }
  return response.json();
}

function statusText(status) {
  return {
    draft: "草稿",
    planning: "规划中",
    in_progress: "执行中",
    waiting_for_approval: "等待审批",
    blocked: "已阻塞",
    completed: "已完成",
    cancelled: "已取消",
  }[status] || status;
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("is-visible");
  window.setTimeout(() => elements.toast.classList.remove("is-visible"), 2200);
}

function openTaskDialog() {
  elements["form-error"].textContent = "";
  elements["task-dialog"].showModal();
}

function renderTaskList() {
  elements["task-list"].replaceChildren();
  if (!state.tasks.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "还没有任务。创建后即可运行 LangGraph。";
    elements["task-list"].append(message);
    return;
  }
  for (const task of state.tasks) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `task-item${task.id === state.selectedTaskId ? " is-active" : ""}`;
    button.innerHTML = `<strong>${escapeHtml(task.title)}</strong><span>#${task.id} · ${escapeHtml(statusText(task.status))}</span>`;
    button.addEventListener("click", () => selectTask(task.id));
    elements["task-list"].append(button);
  }
}

function renderGraph(snapshot) {
  const nodeHistory = snapshot?.state?.node_history || [];
  const nextNodes = snapshot?.next_nodes || [];
  const isBlocked = snapshot?.state?.status === "blocked";
  elements["graph-flow"].replaceChildren();

  graphNodes.forEach(([key, label, detail], index) => {
    const node = document.createElement("div");
    let className = "graph-node";
    if (nodeHistory.includes(key)) className += " is-complete";
    if (nextNodes.includes(key)) className += " is-current";
    if (key === "complete_run" && isBlocked) className += " is-blocked";
    node.className = className;
    const displayLabel = key === "complete_run" && isBlocked ? "Blocked" : label;
    node.innerHTML = `<strong>${displayLabel}</strong><span>${detail}</span>`;
    node.dataset.index = String(index);
    elements["graph-flow"].append(node);
  });
}

function renderResults(snapshot) {
  const currentState = snapshot?.state || {};
  elements["skill-strip"].replaceChildren();
  for (const skill of currentState.top_skills || []) {
    const chip = document.createElement("span");
    chip.className = "skill-chip";
    chip.textContent = `${skill.name} · ${skill.job_count} JD`;
    elements["skill-strip"].append(chip);
  }

  elements["learning-plan"].replaceChildren();
  const plan = currentState.learning_plan || [];
  if (!plan.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "运行 Graph 后生成学习计划。";
    elements["learning-plan"].append(message);
  }
  for (const item of plan) {
    const row = document.createElement("div");
    row.className = "plan-row";
    row.innerHTML = `<strong>${item.rank}. ${escapeHtml(item.skill)}</strong><p>${escapeHtml(item.learning_goal)}</p>`;
    elements["learning-plan"].append(row);
  }
}

function checkpointLabel(snapshot) {
  const history = snapshot.state?.node_history || [];
  if (!history.length) return "初始 State";
  return history[history.length - 1];
}

function renderCheckpoints() {
  elements["checkpoint-count"].textContent = String(state.checkpoints.length);
  elements["checkpoint-list"].replaceChildren();
  for (const snapshot of state.checkpoints) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `checkpoint-item${snapshot.checkpoint_id === state.selectedCheckpoint?.checkpoint_id ? " is-active" : ""}`;
    const next = snapshot.next_nodes.length ? snapshot.next_nodes.join(", ") : "END";
    button.innerHTML = `<strong>Step ${snapshot.step ?? "-"} · ${escapeHtml(checkpointLabel(snapshot))}</strong><span>next: ${escapeHtml(next)}</span>`;
    button.addEventListener("click", () => {
      state.selectedCheckpoint = snapshot;
      renderCheckpoints();
      renderInspector();
    });
    elements["checkpoint-list"].append(button);
  }
  if (!state.checkpoints.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "运行 Graph 后显示每一步 checkpoint。";
    elements["checkpoint-list"].append(message);
  }
}

function renderInspector() {
  const snapshot = state.selectedCheckpoint;
  if (!snapshot) {
    elements["selected-step"].textContent = "";
    elements["state-json"].textContent = "运行 Graph 后选择 checkpoint";
    return;
  }
  elements["selected-step"].textContent = `Step ${snapshot.step ?? "-"} · next: ${snapshot.next_nodes.join(", ") || "END"}`;
  elements["state-json"].textContent = JSON.stringify(snapshot.state, null, 2);
}

function selectedTool() {
  return state.tools.find((tool) => tool.name === elements["tool-select"].value) || state.tools[0] || null;
}

function newOperationKey() {
  const taskId = state.task?.id || "task";
  const toolName = selectedTool()?.name || "tool";
  elements["idempotency-key"].value = `${taskId}:${toolName}:${crypto.randomUUID()}`;
}

function renderToolSelection(resetArguments = false, resetKey = false) {
  const tool = selectedTool();
  if (!tool) return;
  elements["tool-description"].textContent = tool.description;
  const policyAllowed = Boolean(state.taskPolicy?.allowed_tools?.includes(tool.name));
  elements["tool-policy"].textContent = `${tool.effect} · ${tool.idempotency_mode} · ${tool.repeat_policy} · policy ${policyAllowed ? "allowed" : "denied"}`;
  elements["tool-permission"].textContent = `${tool.permission} permission`;
  elements["tool-permission"].dataset.permission = tool.permission;
  elements["write-permission-row"].hidden = tool.permission !== "write";
  elements["idempotency-row"].hidden = tool.idempotency_mode === "none";
  elements["grant-write"].checked = tool.permission === "write" && policyAllowed;
  if (tool.idempotency_mode === "none") elements["idempotency-key"].value = "";
  else if (resetKey || !elements["idempotency-key"].value) newOperationKey();
  if (resetArguments) {
    elements["tool-arguments"].value = JSON.stringify(toolArgumentExamples[tool.name] || {}, null, 2);
  }
}

function renderTools() {
  const currentName = elements["tool-select"].value;
  elements["tool-select"].replaceChildren();
  for (const tool of state.tools) {
    const option = document.createElement("option");
    option.value = tool.name;
    option.textContent = tool.name;
    elements["tool-select"].append(option);
  }
  if (state.tools.some((tool) => tool.name === currentName)) elements["tool-select"].value = currentName;
  renderToolSelection(!elements["tool-arguments"].value);
}

function renderToolCalls() {
  const calls = state.toolCalls;
  elements["tool-call-count"].textContent = `${calls.length} records`;
  elements["tool-call-list"].replaceChildren();
  for (const call of calls) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `tool-call-item${call.tool_call_id === state.selectedToolCall?.tool_call_id ? " is-active" : ""}`;
    button.dataset.status = call.status;
    const replayLabel = call.replay_count ? ` · replay ${call.replay_count}` : "";
    button.innerHTML = `<strong>#${call.tool_call_id} · ${escapeHtml(call.tool_name)}</strong><span>${escapeHtml(call.status)}${escapeHtml(replayLabel)}</span>`;
    button.addEventListener("click", () => {
      state.selectedToolCall = call;
      renderToolCalls();
    });
    elements["tool-call-list"].append(button);
  }
  if (!calls.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "还没有持久化 ToolCall。";
    elements["tool-call-list"].append(message);
  }
  elements["tool-result"].textContent = state.selectedToolCall
    ? JSON.stringify(state.selectedToolCall, null, 2)
    : "调用工具后查看 observation";
}

function renderAgentRuns() {
  elements["agent-run-count"].textContent = `${state.agentRuns.length} records`;
  elements["agent-run-list"].replaceChildren();
  for (const run of state.agentRuns) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `agent-run-item${run.id === state.selectedAgentRun?.id ? " is-active" : ""}`;
    button.innerHTML = `<strong>#${run.id} · ${escapeHtml(run.provider)} / ${escapeHtml(run.model)}</strong><span>${escapeHtml(run.status)} · ${run.step_count}/${run.max_steps} steps</span>`;
    button.addEventListener("click", () => selectAgentRun(run));
    elements["agent-run-list"].append(button);
  }
  if (!state.agentRuns.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "还没有 AgentRun。";
    elements["agent-run-list"].append(message);
  }
  elements["agent-run-result"].textContent = state.selectedAgentRun
    ? JSON.stringify(state.selectedAgentRun, null, 2)
    : "运行后查看模型动作、observation 和停止原因";
}

function renderMemoryContext() {
  elements["memory-context-result"].textContent = state.memoryContext
    ? JSON.stringify(state.memoryContext, null, 2)
    : "选择任务后查看 GoalContract、选中记忆和上下文预算";
}

function taskMemories() {
  if (!state.task) return [];
  const runIds = new Set(state.agentRuns.map((run) => run.id));
  return state.memories.filter((memory) => (
    memory.scope_type === "contract"
    || memory.task_id === state.task.id
    || (memory.run_id && runIds.has(memory.run_id))
  ));
}

function renderMemories() {
  const memories = taskMemories();
  elements["memory-count"].textContent = `${memories.length} records`;
  elements["memory-list"].replaceChildren();
  for (const memory of memories) {
    const button = document.createElement("button");
    button.type = "button";
    const isNew = memory.id === state.recentlyCreatedMemoryId;
    button.className = `memory-item${memory.id === state.selectedMemory?.id ? " is-active" : ""}${isNew ? " is-new" : ""}`;
    button.dataset.status = memory.status;
    button.innerHTML = `<strong>${escapeHtml(memory.memory_key)}${isNew ? '<em>刚创建</em>' : ""}</strong><span>${escapeHtml(memory.memory_type)} · ${escapeHtml(memory.scope_type)} · ${escapeHtml(memory.status)}</span>`;
    button.addEventListener("click", () => {
      state.selectedMemory = memory;
      if (!isNew) state.recentlyCreatedMemoryId = null;
      renderMemories();
    });
    elements["memory-list"].append(button);
  }
  if (!memories.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "当前任务还没有可检查的 Memory。";
    elements["memory-list"].append(message);
  }
  elements["memory-detail"].textContent = state.selectedMemory
    ? JSON.stringify(state.selectedMemory, null, 2)
    : "创建或选择 Memory 后查看完整内容";
  const newlyCreated = state.selectedMemory?.id === state.recentlyCreatedMemoryId;
  elements["memory-created-notice"].hidden = !newlyCreated;
  elements["memory-created-summary"].textContent = newlyCreated
    ? `#${state.selectedMemory.id} · ${state.selectedMemory.memory_type} · ${state.selectedMemory.scope_type}。完整持久化内容如下。`
    : "";
  elements["retire-memory-button"].hidden = !state.selectedMemory || state.selectedMemory.status !== "active";
}

function renderMemoryCandidates() {
  elements["memory-candidate-count"].textContent = `${state.memoryCandidates.length} records`;
  elements["memory-candidate-list"].replaceChildren();
  for (const candidate of state.memoryCandidates) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `memory-item${candidate.id === state.selectedMemoryCandidate?.id ? " is-active" : ""}`;
    button.dataset.decision = candidate.decision;
    button.innerHTML = `<strong>#${candidate.id} · ${escapeHtml(candidate.memory_key)}</strong><span>${escapeHtml(candidate.decision)} · ${escapeHtml(candidate.storage_action)} · ${escapeHtml(candidate.source)}</span>`;
    button.addEventListener("click", () => {
      state.selectedMemoryCandidate = candidate;
      renderMemoryCandidates();
    });
    elements["memory-candidate-list"].append(button);
  }
  if (!state.memoryCandidates.length) {
    const message = document.createElement("p");
    message.className = "muted-message";
    message.textContent = "当前任务还没有 Candidate 决策记录。";
    elements["memory-candidate-list"].append(message);
  }
  elements["memory-candidate-detail"].textContent = state.selectedMemoryCandidate
    ? JSON.stringify(state.selectedMemoryCandidate, null, 2)
    : "运行 Agent 后查看 Candidate、来源证据和 Evaluator 判断";
}

async function refreshMemoryCandidates() {
  if (!state.task) return;
  state.memoryCandidates = await api(`/agent/memory-candidates?task_id=${state.task.id}`);
  if (state.selectedMemoryCandidate) {
    state.selectedMemoryCandidate = state.memoryCandidates.find(
      (candidate) => candidate.id === state.selectedMemoryCandidate.id
    ) || null;
  }
  if (!state.selectedMemoryCandidate) {
    state.selectedMemoryCandidate = state.memoryCandidates[0] || null;
  }
  renderMemoryCandidates();
}

function renderMemoryFormRules() {
  const type = elements["memory-type"].value;
  const scope = elements["memory-scope"];
  for (const option of scope.options) {
    option.disabled = (type === "working" && option.value === "contract")
      || (type !== "working" && option.value === "run");
  }
  if (scope.selectedOptions[0]?.disabled) scope.value = "task";
}

async function refreshMemories(highlightNew = false) {
  const previousIds = new Set(state.memories.map((memory) => memory.id));
  state.memories = await api("/agent/memories?include_retired=true");
  const newlyCreated = highlightNew
    ? taskMemories().filter((memory) => !previousIds.has(memory.id))
    : [];
  if (newlyCreated.length) {
    state.selectedMemory = newlyCreated[0];
    state.recentlyCreatedMemoryId = newlyCreated[0].id;
  }
  if (state.selectedMemory) {
    state.selectedMemory = state.memories.find(
      (memory) => memory.id === state.selectedMemory.id
    ) || null;
  }
  renderMemories();
  return newlyCreated;
}

async function createMemory(event) {
  event.preventDefault();
  if (!state.task) return;
  const scope = elements["memory-scope"].value;
  const expiresAt = elements["memory-expires-at"].value;
  const payload = {
    memory_type: elements["memory-type"].value,
    scope_type: scope,
    task_id: scope === "task" ? state.task.id : null,
    run_id: scope === "run" ? state.selectedAgentRun?.id || null : null,
    memory_key: elements["memory-key"].value.trim(),
    content: elements["memory-content"].value.trim(),
    source: "runtime_console",
    importance: Number(elements["memory-importance"].value),
    expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
  };
  if (scope === "run" && !payload.run_id) {
    elements["memory-error"].textContent = "Run 作用域需要先选择一个 AgentRun。";
    return;
  }
  setBusy(elements["create-memory-button"], true, "创建中");
  elements["memory-error"].textContent = "";
  try {
    const createdMemory = await api("/agent/memories", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedMemory = createdMemory;
    state.recentlyCreatedMemoryId = createdMemory.id;
    elements["memory-key"].value = "";
    elements["memory-content"].value = "";
    await refreshMemories();
    await refreshMemoryContext();
    showToast(`新 Memory #${state.selectedMemory.id} 已创建并展示`);
  } catch (error) {
    elements["memory-error"].textContent = error.message;
  } finally {
    setBusy(elements["create-memory-button"], false, "创建 Memory");
  }
}

async function retireSelectedMemory() {
  if (!state.selectedMemory) return;
  try {
    await api(`/agent/memories/${state.selectedMemory.id}/retire`, {
      method: "POST",
      body: JSON.stringify({ reason: "retired_from_runtime_console" }),
    });
    await refreshMemories();
    await refreshMemoryContext();
    state.recentlyCreatedMemoryId = null;
    renderMemories();
    showToast("Memory 已退休，不再进入 Context");
  } catch (error) {
    showToast(error.message);
  }
}

async function refreshMemoryContext(showConfirmation = false) {
  if (!state.task) return;
  const button = elements["refresh-memory-context-button"];
  setBusy(button, true, "刷新中");
  try {
    const runQuery = state.selectedAgentRun ? `?run_id=${state.selectedAgentRun.id}` : "";
    state.memoryContext = await api(`/agent/tasks/${state.task.id}/memory-context${runQuery}`);
    renderMemoryContext();
    if (showConfirmation) showToast("Memory Context 已重新装配");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false, "刷新 Context");
  }
}

async function selectAgentRun(run) {
  if (!state.task) return;
  try {
    const workflowCheckpoints = await api(
      `/agent/tasks/${state.task.id}/agent-runs/${run.id}/checkpoints`
    );
    state.selectedAgentRun = { ...run, workflow_checkpoints: workflowCheckpoints };
  } catch {
    state.selectedAgentRun = run;
  }
  state.memoryContext = await api(
    `/agent/tasks/${state.task.id}/memory-context?run_id=${run.id}`
  );
  renderAgentRuns();
  renderMemoryContext();
  renderMemories();
}

function renderTask() {
  const task = state.task;
  elements["empty-state"].hidden = Boolean(task);
  elements["task-view"].hidden = !task;
  if (!task) {
    renderGraph(null);
    renderResults(null);
    renderCheckpoints();
    renderInspector();
    return;
  }

  elements["task-title"].textContent = task.title;
  elements["task-goal"].textContent = task.user_goal;
  elements["task-status"].textContent = statusText(task.status);
  elements["task-status"].dataset.status = task.status;
  elements["run-button"].disabled = task.status === "waiting_for_approval";
  elements["run-button"].textContent = task.status === "completed" || task.status === "blocked" ? "读取最终 State" : "运行 Graph";

  const latest = state.checkpoints[0] || null;
  const awaitingApproval = task.status === "waiting_for_approval";
  elements["approval-bar"].hidden = !awaitingApproval;
  elements["approval-question"].textContent = awaitingApproval ? "检查生成的技能优先级与学习计划，然后批准或拒绝。" : "";
  renderGraph(latest);
  renderResults(latest);
  renderCheckpoints();
  renderInspector();
  renderToolCalls();
  renderAgentRuns();
  renderMemories();
  renderMemoryCandidates();
}

async function loadTasks(preferredId = null) {
  state.tasks = await api("/agent/tasks");
  const nextId = preferredId || state.selectedTaskId || state.tasks[0]?.id || null;
  renderTaskList();
  if (nextId) await selectTask(nextId);
  else {
    state.task = null;
    state.checkpoints = [];
    state.selectedCheckpoint = null;
    state.toolCalls = [];
    state.selectedToolCall = null;
    state.taskPolicy = null;
    state.agentRuns = [];
    state.selectedAgentRun = null;
    state.memoryContext = null;
    state.memories = [];
    state.selectedMemory = null;
    state.recentlyCreatedMemoryId = null;
    state.memoryCandidates = [];
    state.selectedMemoryCandidate = null;
    renderTask();
  }
}

async function selectTask(taskId) {
  state.selectedTaskId = taskId;
  const [task, checkpoints, toolCalls, taskPolicy, agentRuns, memoryContext, memories, memoryCandidates] = await Promise.all([
    api(`/agent/tasks/${taskId}`),
    api(`/agent/tasks/${taskId}/graph-checkpoints`),
    api(`/agent/tasks/${taskId}/tool-calls`),
    api(`/agent/tasks/${taskId}/tool-policy`),
    api(`/agent/tasks/${taskId}/agent-runs`),
    api(`/agent/tasks/${taskId}/memory-context`),
    api("/agent/memories?include_retired=true"),
    api(`/agent/memory-candidates?task_id=${taskId}`),
  ]);
  state.task = task;
  state.checkpoints = checkpoints;
  state.selectedCheckpoint = checkpoints[0] || null;
  state.toolCalls = toolCalls;
  state.selectedToolCall = toolCalls[0] || null;
  state.taskPolicy = taskPolicy;
  state.agentRuns = agentRuns;
  state.selectedAgentRun = agentRuns[0] || null;
  state.memoryContext = memoryContext;
  state.memories = memories;
  state.selectedMemory = null;
  state.recentlyCreatedMemoryId = null;
  state.memoryCandidates = memoryCandidates;
  state.selectedMemoryCandidate = memoryCandidates[0] || null;
  renderToolSelection(false, true);
  renderTaskList();
  renderTask();
  if (state.selectedAgentRun) await selectAgentRun(state.selectedAgentRun);
}

async function runAgentLoop() {
  if (!state.task) return;
  const button = elements["run-agent-button"];
  setBusy(button, true, "运行中");
  try {
    const provider = elements["agent-provider"].value;
    const result = await api(`/agent/tasks/${state.task.id}/agent-runs`, {
      method: "POST",
      body: JSON.stringify({
        provider,
        model: provider === "openai" ? elements["agent-model"].value.trim() : null,
        max_steps: Number(elements["agent-max-steps"].value),
      }),
    });
    state.agentRuns = await api(`/agent/tasks/${state.task.id}/agent-runs`);
    state.selectedAgentRun = state.agentRuns.find((run) => run.id === result.id) || result;
    state.toolCalls = await api(`/agent/tasks/${state.task.id}/tool-calls`);
    state.selectedToolCall = state.toolCalls[0] || null;
    await selectAgentRun(state.selectedAgentRun);
    renderToolCalls();
    const createdMemories = await refreshMemories(true);
    await refreshMemoryCandidates();
    await refreshMemoryContext();
    showToast(
      createdMemories.length
        ? `Agent ${result.status} · 新建 ${createdMemories.length} 条 Memory`
        : `Agent ${result.status} · ${result.step_count} steps`
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false, "运行 Agent");
  }
}

function renderAgentProvider() {
  elements["agent-model-row"].hidden = elements["agent-provider"].value !== "openai";
}

async function recoverAgentRuns() {
  if (!state.task) return;
  const button = elements["recover-agent-runs-button"];
  setBusy(button, true, "检查中");
  try {
    const report = await api(`/agent/tasks/${state.task.id}/recover-agent-runs`, {
      method: "POST",
    });
    state.agentRuns = await api(`/agent/tasks/${state.task.id}/agent-runs`);
    state.selectedAgentRun = state.agentRuns[0] || null;
    renderAgentRuns();
    if (state.selectedAgentRun) await selectAgentRun(state.selectedAgentRun);
    const reviewCount = report.decisions.filter(
      (decision) => decision.action === "needs_review"
    ).length;
    showToast(
      report.scanned_count
        ? `恢复 ${report.recovered_count} 条，待人工 ${reviewCount} 条`
        : "没有超过 60 秒仍在运行的 AgentRun"
    );
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false, "恢复中断 Run");
  }
}

async function callTool(event) {
  event.preventDefault();
  if (!state.task) return;
  elements["tool-error"].textContent = "";
  let argumentsValue;
  try {
    argumentsValue = JSON.parse(elements["tool-arguments"].value || "{}");
  } catch {
    elements["tool-error"].textContent = "Arguments 必须是有效 JSON";
    return;
  }
  setBusy(elements["call-tool-button"], true, "调用中");
  try {
    const result = await api(`/agent/tasks/${state.task.id}/tool-calls`, {
      method: "POST",
      body: JSON.stringify({
        tool_name: elements["tool-select"].value,
        arguments: argumentsValue,
        idempotency_key: elements["idempotency-key"].value || null,
      }),
    });
    state.toolCalls = await api(`/agent/tasks/${state.task.id}/tool-calls`);
    const stored = state.toolCalls.find((call) => call.tool_call_id === result.tool_call_id);
    state.selectedToolCall = stored ? { ...stored, replayed: result.replayed } : result;
    renderToolCalls();
    showToast(result.replayed ? "返回已有 ToolCall，handler 未重跑" : `工具返回 ${result.status}`);
  } catch (error) {
    elements["tool-error"].textContent = error.message;
  } finally {
    setBusy(elements["call-tool-button"], false, "调用工具");
  }
}

async function updateSelectedToolPolicy() {
  if (!state.task || !state.taskPolicy) return;
  const tool = selectedTool();
  if (!tool || tool.permission !== "write") return;
  const allowedTools = new Set(state.taskPolicy.allowed_tools);
  if (elements["grant-write"].checked) allowedTools.add(tool.name);
  else allowedTools.delete(tool.name);
  elements["grant-write"].disabled = true;
  try {
    state.taskPolicy = await api(`/agent/tasks/${state.task.id}/tool-policy`, {
      method: "PUT",
      body: JSON.stringify({ allowed_tools: [...allowedTools] }),
    });
    renderToolSelection();
    showToast(`TaskPolicy v${state.taskPolicy.version} 已更新`);
  } catch (error) {
    elements["grant-write"].checked = !elements["grant-write"].checked;
    showToast(error.message);
  } finally {
    elements["grant-write"].disabled = false;
  }
}

async function recoverToolCalls() {
  if (!state.task) return;
  const button = elements["recover-tools-button"];
  setBusy(button, true, "检查中");
  try {
    const report = await api(`/agent/tasks/${state.task.id}/recover-tool-calls`, {
      method: "POST",
    });
    state.toolCalls = await api(`/agent/tasks/${state.task.id}/tool-calls`);
    state.selectedToolCall = state.toolCalls[0] || null;
    renderToolCalls();
    const summary = report.scanned_count
      ? `检查 ${report.scanned_count} 条，自动重试 ${report.recovered_count} 条`
      : "没有超过 60 秒仍未结束的调用";
    showToast(summary);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false, "检查中断调用");
  }
}

async function runGraph() {
  if (!state.task) return;
  setBusy(elements["run-button"], true, "运行中");
  try {
    await api(`/agent/tasks/${state.task.id}/run-learning-graph`, { method: "POST" });
    await loadTasks(state.task.id);
    showToast("Graph 已运行到当前边界");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(elements["run-button"], false, "运行 Graph");
    renderTask();
  }
}

async function reviewGraph(approved) {
  if (!state.task) return;
  const button = approved ? elements["approve-button"] : elements["reject-button"];
  setBusy(button, true, approved ? "正在批准" : "正在拒绝");
  try {
    await api(`/agent/tasks/${state.task.id}/resume-learning-graph`, {
      method: "POST",
      body: JSON.stringify({ approved, comment: approved ? "Approved in Runtime Console." : "Rejected in Runtime Console." }),
    });
    await loadTasks(state.task.id);
    showToast(approved ? "Graph 已恢复并完成" : "Graph 已进入阻塞分支");
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false, approved ? "批准并继续" : "拒绝");
  }
}

async function createTask(event) {
  event.preventDefault();
  const form = new FormData(elements["task-form"]);
  const constraint = String(form.get("constraint") || "").trim();
  const payload = {
    title: String(form.get("title") || "").trim(),
    user_goal: String(form.get("user_goal") || "").trim(),
    constraints: constraint ? [constraint] : [],
    success_criteria: [String(form.get("success_criterion") || "").trim()],
  };
  try {
    const task = await api("/agent/tasks", { method: "POST", body: JSON.stringify(payload) });
    elements["task-dialog"].close();
    await loadTasks(task.id);
    showToast("AgentTask 已创建");
  } catch (error) {
    elements["form-error"].textContent = error.message;
  }
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = label;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

elements["new-task-button"].addEventListener("click", openTaskDialog);
elements["empty-create-button"].addEventListener("click", openTaskDialog);
elements["close-dialog"].addEventListener("click", () => elements["task-dialog"].close());
elements["cancel-dialog"].addEventListener("click", () => elements["task-dialog"].close());
elements["task-form"].addEventListener("submit", createTask);
elements["run-button"].addEventListener("click", runGraph);
elements["approve-button"].addEventListener("click", () => reviewGraph(true));
elements["reject-button"].addEventListener("click", () => reviewGraph(false));
elements["tool-select"].addEventListener("change", () => renderToolSelection(true, true));
elements["new-operation-key"].addEventListener("click", newOperationKey);
elements["tool-form"].addEventListener("submit", callTool);
elements["recover-tools-button"].addEventListener("click", recoverToolCalls);
elements["grant-write"].addEventListener("change", updateSelectedToolPolicy);
elements["run-agent-button"].addEventListener("click", runAgentLoop);
elements["agent-provider"].addEventListener("change", renderAgentProvider);
elements["recover-agent-runs-button"].addEventListener("click", recoverAgentRuns);
elements["refresh-memory-context-button"].addEventListener("click", () => refreshMemoryContext(true));
elements["memory-type"].addEventListener("change", renderMemoryFormRules);
elements["memory-form"].addEventListener("submit", createMemory);
elements["retire-memory-button"].addEventListener("click", retireSelectedMemory);

async function boot() {
  try {
    await api("/health");
    elements["server-status"].classList.add("is-online");
    elements["server-status"].querySelector("span:last-child").textContent = "Runtime 在线";
    state.tools = await api("/agent/tools");
    renderMemoryFormRules();
    renderTools();
    await loadTasks();
  } catch (error) {
    elements["server-status"].classList.add("is-error");
    elements["server-status"].querySelector("span:last-child").textContent = "连接失败";
    showToast(error.message);
  }
}

boot();
