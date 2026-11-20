function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&apos;",
  }[char]));
}

const state = {
  eventSource: null,
  currentRunId: null,
  terminal: false,
  finalRendered: false,
  tokenTotal: 0,
  stepTotal: 0,
};

const els = {};

function bindElements() {
  Object.assign(els, {
    tabs: Array.from(document.querySelectorAll(".tab")),
    panes: Array.from(document.querySelectorAll(".pane")),
    runForm: document.getElementById("run-form"),
    loadForm: document.getElementById("load-form"),
    assetList: document.getElementById("asset-list"),
    addAsset: document.getElementById("add-asset"),
    question: document.getElementById("question-input"),
    maxIterations: document.getElementById("max-iterations"),
    plannerModel: document.getElementById("planner-model"),
    perceptionModel: document.getElementById("perception-model"),
    cancelBtn: document.getElementById("cancel-btn"),
    logPath: document.getElementById("log-path"),
    stream: document.getElementById("stream"),
    empty: document.getElementById("empty"),
    railCase: document.getElementById("rail-case"),
    railStage: document.getElementById("rail-stage"),
    railSuff: document.getElementById("rail-suff"),
    railNeeds: document.getElementById("rail-needs"),
    railSteps: document.getElementById("rail-steps"),
    railTok: document.getElementById("rail-tok"),
    railAtomsVal: document.getElementById("rail-atoms-val"),
    railConflictsVal: document.getElementById("rail-conflicts-val"),
    railStop: document.getElementById("rail-stop"),
    railNeedsList: document.getElementById("rail-needs-list"),
    statusPill: document.getElementById("status-pill"),
    statusText: document.getElementById("status-text"),
    verboseToggle: document.getElementById("verbose-toggle"),
    verboseSwitch: document.getElementById("verbose-switch"),
    logToggle: document.getElementById("log-toggle"),
    runlog: document.getElementById("runlog"),
  });
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function formatTokens(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue.toLocaleString("en-US") : String(value);
}

function tokenSnapshotTotals(snapshot) {
  const root = asObject(snapshot);
  const nestedTotal = asObject(root.total);
  return {
    total: root.total_tokens ?? nestedTotal.total_tokens ?? root.token_total,
    prompt: root.prompt_tokens ?? nestedTotal.prompt_tokens,
    completion: root.completion_tokens ?? nestedTotal.completion_tokens,
    calls: root.llm_calls ?? root.call_count ?? nestedTotal.llm_calls ?? nestedTotal.call_count,
  };
}

function stringifyValue(value) {
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (typeof value === "string") {
    const stripped = value.trim();
    if (stripped.startsWith("{") || stripped.startsWith("[")) {
      try {
        const parsed = JSON.parse(stripped);
        return JSON.stringify(parsed, null, 2);
      } catch (error) {
        return value;
      }
    }
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch (error) {
    return String(value);
  }
}

function previewText(value, limit = 180) {
  const text = stringifyValue(value).replace(/\s+/g, " ").trim();
  if (!text) {
    return "";
  }
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function roleColor(roleValue) {
  const role = String(roleValue || "");
  if (role.startsWith("tool:")) return "var(--r-tool)";
  if (role === "planner.decide") return "var(--r-planner)";
  if (role === "critic.check" || role === "answer_critic.check") return "var(--r-critic)";
  if (role === "setup.question_profile") return "var(--r-setup)";
  if (role === "finalizer") return "var(--r-answer)";
  if (role === "case") return "var(--r-case)";
  return "var(--muted)";
}

function evidenceDigest(record) {
  return asObject(asObject(record.extra).evidence_state_digest);
}

function verdictValue(record) {
  const extra = asObject(record.extra);
  return extra.verdict || evidenceDigest(record).sufficiency_status || "";
}

function stageValue(record) {
  return evidenceDigest(record).current_stage || asObject(record.extra).stage || "";
}

function evidenceNeeds(record) {
  return asArray(evidenceDigest(record).evidence_needs);
}

function openNeedsCount(record) {
  return evidenceNeeds(record).filter((need) => {
    const status = String(asObject(need).status || "").toLowerCase();
    return status !== "filled" && status !== "unfillable" && status !== "waived";
  }).length;
}

function subCalls(record) {
  return asArray(asObject(record.extra).sub_calls);
}

function toolCalls(record) {
  const calls = asArray(asObject(record.extra).tool_calls);
  return calls.map((call) => {
    if (typeof call === "string") return call;
    return asObject(call).name || asObject(call).tool || "";
  }).filter(Boolean);
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) {
    element.className = className;
  }
  if (text !== undefined && text !== null) {
    element.textContent = String(text);
  }
  return element;
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function appendMeta(parent, key, value) {
  if (value === null || value === undefined || value === "") {
    return;
  }
  const item = makeElement("span", "m");
  const label = makeElement("span", "k", key);
  const strong = makeElement("b", "", value);
  item.append(label, strong);
  parent.appendChild(item);
}

function createIORow(label, value) {
  const text = stringifyValue(value);
  if (!text) {
    return null;
  }
  const fragment = document.createDocumentFragment();
  const button = makeElement("button", "io-row");
  button.type = "button";
  button.append(
    makeElement("span", "chev", "›"),
    makeElement("span", "lab", label),
    makeElement("span", "prev", previewText(text))
  );
  const full = makeElement("div", "io-full");
  const pre = makeElement("pre", "", text);
  full.appendChild(pre);
  fragment.append(button, full);
  return fragment;
}

function renderEvidence(record) {
  const extra = asObject(record.extra);
  const digest = evidenceDigest(record);
  const delta = asObject(extra.evidence_delta);
  const needs = asArray(digest.evidence_needs);
  const conflictCount = digest.conflict_count;
  const stopReady = digest.stop_ready;
  const gapDiag = digest.gap_diagnosis;

  const hasDelta = Object.keys(delta).length > 0;
  const hasNeeds = needs.length > 0;
  const hasFlags = (conflictCount !== undefined && conflictCount !== null) ||
                   (stopReady !== undefined && stopReady !== null);
  const hasGap = gapDiag && typeof gapDiag === "object" && Object.keys(gapDiag).length > 0;

  if (!hasDelta && !hasNeeds && !hasFlags && !hasGap) {
    return null;
  }

  const wrap = makeElement("div", "evidence");

  // Section 1: delta atoms
  if (hasDelta) {
    const sec = makeElement("div", "ev-section");
    sec.appendChild(makeElement("div", "ev-section-head", "evidence delta"));
    const row = makeElement("div", "ev-delta-row");
    Object.entries(delta).slice(0, 12).forEach(([key, value]) => {
      const tag = makeElement("span", "etag");
      const safeKey = escapeHtml(key);
      const safeValue = escapeHtml(previewText(value, 90));
      tag.innerHTML = `<span class="plus">+</span> ${safeKey}: ${safeValue}`;
      row.appendChild(tag);
    });
    sec.appendChild(row);
    wrap.appendChild(sec);
  }

  // Section 2: evidence needs list with status badges
  if (hasNeeds) {
    const sec = makeElement("div", "ev-section");
    sec.appendChild(makeElement("div", "ev-section-head", `evidence needs · ${needs.length}`));
    const row = makeElement("div", "ev-needs-row");
    needs.forEach((need) => {
      const n = asObject(need);
      const status = String(n.status || "unfilled").toLowerCase();
      const chip = makeElement("span", "need-chip");
      const label = makeElement("span", "nl");
      label.textContent = previewText(n.need || "(need)", 72);
      const badge = makeElement("span", `ns ${status}`);
      badge.textContent = status;
      chip.append(label, badge);
      row.appendChild(chip);
    });
    sec.appendChild(row);
    wrap.appendChild(sec);
  }

  // Section 3: conflict / stop_ready flags
  if (hasFlags) {
    const sec = makeElement("div", "ev-section");
    const flagRow = makeElement("div", "ev-delta-row");
    if (conflictCount !== undefined && conflictCount !== null) {
      const flag = makeElement("span", conflictCount > 0 ? "ev-flag conflict" : "ev-flag");
      flag.textContent = `conflicts: ${conflictCount}`;
      flagRow.appendChild(flag);
    }
    if (stopReady !== undefined && stopReady !== null) {
      const flag = makeElement("span", stopReady ? "ev-flag stop" : "ev-flag");
      flag.textContent = `stop_ready: ${stopReady}`;
      flagRow.appendChild(flag);
    }
    sec.appendChild(flagRow);
    wrap.appendChild(sec);
  }

  // Section 4: gap_diagnosis (when present)
  if (hasGap) {
    const sec = makeElement("div", "ev-section");
    sec.appendChild(makeElement("div", "ev-section-head", "gap diagnosis"));
    const pre = makeElement("div", "ev-gap");
    pre.textContent = previewText(gapDiag, 300);
    sec.appendChild(pre);
    wrap.appendChild(sec);
  }

  return wrap;
}

function renderSubCalls(record) {
  const calls = subCalls(record);
  if (!calls.length) {
    return null;
  }
  const wrap = makeElement("div", "subcalls");
  wrap.appendChild(makeElement("div", "sc-head", `sub-calls · ${calls.length}`));

  calls.forEach((call, index) => {
    const normalized = asObject(call);
    const row = makeElement("div", "sc");
    row.append(
      makeElement("span", "caller", normalized.caller || `sub_call_${index + 1}`),
      makeElement("span", "", normalized.model || "model"),
      makeElement("span", "tk", `${formatTokens(normalized.prompt_tokens ?? normalized.ti)}→${formatTokens(normalized.completion_tokens ?? normalized.to)} tok`)
    );
    wrap.appendChild(row);

    const messages = asArray(normalized.messages);
    if (messages.length) {
      const verbose = makeElement("div", "verbose-block");
      const stack = makeElement("div", "msg-stack");
      messages.forEach((message) => {
        const messageObj = asObject(message);
        const role = String(messageObj.role || "message");
        const msg = makeElement("div", `msg ${role.replace(/[^a-zA-Z0-9_-]/g, "")}`);
        msg.appendChild(makeElement("div", "mh", role));
        msg.appendChild(makeElement("pre", "", stringifyValue(messageObj.content ?? messageObj.text)));
        stack.appendChild(msg);
      });
      verbose.appendChild(stack);
      wrap.appendChild(verbose);
    }
  });

  if (wrap.querySelector(".verbose-block")) {
    wrap.appendChild(makeElement("div", "vhint vhint-off", "Use full prompts to expand captured sub-call messages."));
  }
  return wrap;
}

function renderVerdict(record) {
  const verdict = verdictValue(record);
  if (!verdict) {
    return null;
  }
  const normalized = String(verdict);
  const wrap = makeElement("div", `verdict ${normalized}`);
  wrap.appendChild(makeElement("span", "v", normalized));
  const gap = asObject(record.extra).gap_diagnosis || asObject(record.extra).revision_hint || "";
  wrap.appendChild(makeElement("span", "gap", stringifyValue(gap)));
  return wrap;
}

function renderStep(record, idx, live) {
  const role = String(record.role || "");
  const event = String(record.event || asObject(record.extra).event || "");
  if (role === "case" && event === "start") {
    return null;
  }
  if (role === "case" && event === "end") {
    return renderFinal(record);
  }

  const step = makeElement("div", live ? "step live" : "step");
  step.style.setProperty("--role", roleColor(role));
  step.style.animationDelay = `${live ? 0 : Math.min(idx * 45, 260)}ms`;

  const spine = makeElement("div", "spine");
  spine.appendChild(makeElement("span", "node"));

  const card = makeElement("article", "card step-card");
  const head = makeElement("div", "card-head");
  const stepNo = record.step === null || record.step === undefined ? idx : record.step;
  head.appendChild(makeElement("span", "step-no", String(stepNo).padStart(2, "0")));

  const roleTag = makeElement("span", "role-tag");
  roleTag.appendChild(makeElement("span", "rdot"));
  roleTag.appendChild(document.createTextNode(role || "unknown"));
  head.appendChild(roleTag);

  const calls = toolCalls(record);
  if (calls.length) {
    head.appendChild(makeElement("span", "evt", `→ ${calls.join(", ")}`));
  }

  const right = makeElement("span", "head-right2");
  const status = String(record.status || "ok");
  const statusNode = makeElement("span", `status ${status}`);
  statusNode.appendChild(makeElement("span", "sdot"));
  statusNode.appendChild(document.createTextNode(status));
  right.appendChild(statusNode);
  const latency = Number(record.latency_ms ?? record.lat);
  if (Number.isFinite(latency) && latency > 0) {
    right.appendChild(makeElement("span", latency > 15000 ? "latency hot" : "latency", `${formatTokens(latency)} ms`));
  }
  head.appendChild(right);
  card.appendChild(head);

  const meta = makeElement("div", "card-meta");
  appendMeta(meta, "model", record.model);
  appendMeta(meta, "tokens", record.tokens_in !== null && record.tokens_in !== undefined ? `${formatTokens(record.tokens_in)} → ${formatTokens(record.tokens_out)}` : "");
  appendMeta(meta, "reason", record.reasoning_tokens);
  appendMeta(meta, "cached", record.cached_tokens);
  if (meta.childNodes.length) {
    card.appendChild(meta);
  }

  const verdict = renderVerdict(record);
  if (verdict) {
    card.appendChild(verdict);
  }

  const io = makeElement("div", "io");
  const inputRow = createIORow("input", record.input_preview ?? record.input);
  const outputRow = createIORow("output", record.output_preview ?? record.output ?? asObject(record.extra).response_text);
  if (inputRow) io.appendChild(inputRow);
  if (outputRow) io.appendChild(outputRow);
  if (io.childNodes.length) {
    card.appendChild(io);
  }

  const evidence = renderEvidence(record);
  if (evidence) {
    card.appendChild(evidence);
  }
  const subCallBlock = renderSubCalls(record);
  if (subCallBlock) {
    card.appendChild(subCallBlock);
  }

  step.append(spine, card);
  bindIO(step);
  return step;
}

function renderFinal(record) {
  const extra = asObject(record.extra);
  const status = String(record.final_status || extra.final_status || record.status || "");
  const isError = status === "error" || record.status === "error" || extra.error_msg;
  const final = makeElement("section", isError ? "final err final-answer" : "final final-answer");
  const header = makeElement("div", "fh");
  header.appendChild(makeElement("span", "t", isError ? "Run failed" : "Final answer"));
  header.appendChild(makeElement("span", "badge", isError ? "error" : "answered"));
  final.appendChild(header);

  const answer = record.answer_preview || record.final_answer || extra.answer_preview || extra.final_answer || extra.error_msg || record.output_preview || "";
  final.appendChild(makeElement("div", "fb", stringifyValue(answer)));

  const tokenSnapshot = tokenSnapshotTotals(record.token_snapshot || extra.token_snapshot);
  const tokens = makeElement("div", "ftoks");
  const total = record.token_total ?? tokenSnapshot.total;
  const prompt = record.tokens_in ?? tokenSnapshot.prompt;
  const completion = record.tokens_out ?? tokenSnapshot.completion;
  const calls = record.calls ?? tokenSnapshot.calls;
  [
    ["total", total],
    ["prompt", prompt],
    ["completion", completion],
    ["llm calls", calls],
  ].forEach(([label, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      const node = makeElement("span");
      node.append(document.createTextNode(`${label} `), makeElement("b", "", formatTokens(value)));
      tokens.appendChild(node);
    }
  });
  if (tokens.childNodes.length) {
    final.appendChild(tokens);
  }
  return final;
}

function updateRail(record) {
  if (record.case_id || record.case) {
    els.railCase.textContent = record.case_id || record.case;
  }
  const role = String(record.role || "");
  const event = String(record.event || asObject(record.extra).event || "");
  if (role === "case" && event === "start") {
    els.railCase.textContent = record.case_id || "case";
  }

  const stage = stageValue(record);
  if (stage) {
    els.railStage.dataset.v = stage;
    els.railStage.querySelector("b").textContent = stage;
  }
  const sufficiency = verdictValue(record);
  if (sufficiency) {
    els.railSuff.dataset.v = sufficiency;
    els.railSuff.querySelector("b").textContent = sufficiency;
  }
  const needs = evidenceNeeds(record);
  if (needs.length) {
    els.railNeeds.textContent = String(openNeedsCount(record));
  }

  const digest = evidenceDigest(record);

  // atoms count
  const atomCount = digest.evidence_atom_count;
  if (atomCount !== undefined && atomCount !== null && els.railAtomsVal) {
    els.railAtomsVal.textContent = String(atomCount);
  }

  // conflict count
  const conflictCount = digest.conflict_count;
  if (conflictCount !== undefined && conflictCount !== null && els.railConflictsVal) {
    els.railConflictsVal.textContent = String(conflictCount);
  }

  // stop_ready chip
  if (digest.stop_ready !== undefined && digest.stop_ready !== null && els.railStop) {
    const stopStr = String(digest.stop_ready);
    els.railStop.dataset.v = stopStr;
    els.railStop.querySelector("b").textContent = stopStr;
  }

  // needs list — render a need-chip per need showing truncated text + status badge
  if (els.railNeedsList) {
    const railNeeds = asArray(digest.evidence_needs);
    if (railNeeds.length) {
      clearNode(els.railNeedsList);
      railNeeds.forEach((need) => {
        const n = asObject(need);
        const status = String(n.status || "unfilled").toLowerCase();
        const chip = makeElement("span", "need-chip");
        const label = makeElement("span", "nl");
        label.textContent = previewText(n.need || "(need)", 60);
        const badge = makeElement("span", `ns ${status}`);
        badge.textContent = status;
        chip.append(label, badge);
        els.railNeedsList.appendChild(chip);
      });
    }
  }
}

function pushStep(record, idx, live) {
  updateRail(record);
  const role = String(record.role || "");
  const event = String(record.event || asObject(record.extra).event || "");
  if (role === "case" && event === "start") {
    return;
  }
  const element = renderStep(record, idx, live);
  if (!element) {
    return;
  }
  if (els.empty && els.empty.isConnected) {
    els.empty.remove();
  }
  if (live) {
    els.stream.querySelectorAll(".step.live").forEach((node) => node.classList.remove("live"));
  }
  els.stream.appendChild(element);
  if (role === "case" && event === "end") {
    state.finalRendered = true;
  }
  if (role !== "case") {
    state.stepTotal += 1;
    els.railSteps.textContent = String(state.stepTotal);
  }
  const tokenSum = Number(record.tokens_in || 0) + Number(record.tokens_out || 0);
  if (tokenSum) {
    state.tokenTotal += tokenSum;
  }
  const tokenSnapshot = tokenSnapshotTotals(record.token_snapshot || asObject(record.extra).token_snapshot);
  const totalFromSnapshot = record.token_total ?? tokenSnapshot.total;
  if (totalFromSnapshot !== null && totalFromSnapshot !== undefined && totalFromSnapshot !== "") {
    const normalizedTotal = Number(totalFromSnapshot);
    if (Number.isFinite(normalizedTotal)) {
      state.tokenTotal = normalizedTotal;
    }
  }
  els.railTok.textContent = formatTokens(state.tokenTotal || 0);
  if (live) {
    element.scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

function bindIO(element) {
  element.querySelectorAll(".io-row").forEach((button) => {
    button.addEventListener("click", () => {
      button.classList.toggle("open");
      const full = button.nextElementSibling;
      if (full && full.classList.contains("io-full")) {
        full.classList.toggle("show");
      }
    });
  });
}

function batchRender(records, batchSize = 50) {
  let index = 0;
  function step() {
    const end = Math.min(index + batchSize, records.length);
    for (; index < end; index += 1) {
      pushStep(records[index], index, false);
    }
    if (index < records.length) {
      requestAnimationFrame(step);
    }
  }
  requestAnimationFrame(step);
}

function setStatus(status, text) {
  els.statusPill.dataset.state = status;
  els.statusText.textContent = text;
}

function resetView() {
  clearNode(els.stream);
  const empty = makeElement("div", "stream-empty");
  empty.id = "empty";
  empty.append(makeElement("span", "big", "no trace loaded"), makeElement("span", "", "Run one task or load a step_trace.jsonl file."));
  els.stream.appendChild(empty);
  els.empty = empty;
  state.tokenTotal = 0;
  state.stepTotal = 0;
  state.finalRendered = false;
  els.railCase.textContent = "—";
  els.railStage.dataset.v = "";
  els.railStage.querySelector("b").textContent = "—";
  els.railSuff.dataset.v = "";
  els.railSuff.querySelector("b").textContent = "—";
  els.railNeeds.textContent = "—";
  if (els.railAtomsVal) els.railAtomsVal.textContent = "—";
  if (els.railConflictsVal) els.railConflictsVal.textContent = "—";
  if (els.railStop) { els.railStop.dataset.v = ""; els.railStop.querySelector("b").textContent = "—"; }
  if (els.railNeedsList) clearNode(els.railNeedsList);
  els.railSteps.textContent = "0";
  els.railTok.textContent = "0";
}

function clearLog() {
  clearNode(els.runlog);
}

function logLine(text, cls) {
  const line = makeElement("div", `ln ${cls || ""}`, text);
  els.runlog.appendChild(line);
  els.runlog.scrollTop = els.runlog.scrollHeight;
}

async function handleRunSubmit(event) {
  event.preventDefault();
  closeEventSource();
  resetView();
  clearLog();
  setStatus("running", "starting");
  state.terminal = false;

  const assets = Array.from(els.assetList.querySelectorAll(".asset-row"))
    .map((row) => ({
      path: row.querySelector(".asset-path").value.trim(),
      type: row.querySelector(".asset-type").value,
    }))
    .filter((asset) => asset.path);

  const payload = {
    question: els.question.value.trim(),
    assets,
    max_iterations: Number(els.maxIterations.value || 15),
  };
  const plannerModel = els.plannerModel.value.trim();
  const perceptionModel = els.perceptionModel.value.trim();
  if (plannerModel) payload.planner_model = plannerModel;
  if (perceptionModel) payload.perception_model = perceptionModel;

  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(stringifyValue(data.detail || data));
    }
    state.currentRunId = data.run_id;
    els.cancelBtn.classList.remove("hidden");
    logLine(`[run] ${data.run_id} ${data.run_dir || ""}`.trim(), "pp");
    connectSSE(data.run_id);
  } catch (error) {
    state.terminal = true;
    setStatus("error", "error");
    logLine(error.message || String(error), "err");
    els.cancelBtn.classList.add("hidden");
  }
}

function connectSSE(runId) {
  const source = new EventSource(`/api/runs/${encodeURIComponent(runId)}/events`);
  state.eventSource = source;
  setStatus("running", "running");

  source.addEventListener("step", (event) => {
    const data = JSON.parse(event.data);
    pushStep(data.record || data, state.stepTotal, true);
  });
  source.addEventListener("log_line", (event) => {
    const data = JSON.parse(event.data);
    logLine(data.text || "", data.kind || "log");
  });
  source.addEventListener("run_finished", (event) => {
    const data = JSON.parse(event.data);
    state.terminal = true;
    setStatus("finished", "finished");
    if (!state.finalRendered) {
      const final = renderFinal({
        role: "case",
        event: "end",
        status: "ok",
        final_answer: data.final_answer,
        result_kind: data.result_kind,
        token_snapshot: data.token_snapshot,
      });
      if (final) {
        if (els.empty && els.empty.isConnected) els.empty.remove();
        els.stream.appendChild(final);
        state.finalRendered = true;
      }
    }
    closeEventSource();
    els.cancelBtn.classList.add("hidden");
  });
  source.addEventListener("run_failed", (event) => {
    const data = JSON.parse(event.data);
    state.terminal = true;
    setStatus("error", "error");
    logLine(data.error || "run failed", "err");
    if (!state.finalRendered) {
      const final = renderFinal({
        role: "case",
        event: "end",
        status: "error",
        output_preview: data.traceback || data.error,
        extra: { error_msg: data.error },
      });
      if (final) {
        if (els.empty && els.empty.isConnected) els.empty.remove();
        els.stream.appendChild(final);
        state.finalRendered = true;
      }
    }
    closeEventSource();
    els.cancelBtn.classList.add("hidden");
  });
  source.onerror = () => {
    if (state.terminal) {
      closeEventSource();
      return;
    }
    setStatus("running", "reconnecting");
  };
}

function closeEventSource() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
}

async function handleCancelClick() {
  if (!state.currentRunId) {
    return;
  }
  try {
    await fetch(`/api/runs/${encodeURIComponent(state.currentRunId)}/cancel`, { method: "POST" });
    state.terminal = true;
    setStatus("idle", "cancelled");
    logLine("[run] cancellation requested", "err");
  } finally {
    closeEventSource();
    els.cancelBtn.classList.add("hidden");
  }
}

async function loadTrace(path) {
  closeEventSource();
  resetView();
  clearLog();
  setStatus("running", "loading");
  const response = await fetch("/api/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(stringifyValue(data.detail || data));
  }
  const records = asArray(data.records);
  batchRender(records);
  setStatus("finished", "loaded");
  logLine(`[load] ${path} · ${records.length} records`, "pp");
  return records;
}

async function handleLoadSubmit(event) {
  event.preventDefault();
  try {
    await loadTrace(els.logPath.value.trim());
  } catch (error) {
    state.terminal = true;
    setStatus("error", "error");
    logLine(error.message || String(error), "err");
  }
}

function addAssetRow() {
  const row = makeElement("div", "asset-row");
  const path = makeElement("input", "inp asset-path");
  path.name = "asset_path";
  path.placeholder = "/path/to/asset";
  path.type = "text";
  const type = makeElement("select", "asset-type");
  type.name = "asset_type";
  ["video", "image"].forEach((value) => {
    const option = makeElement("option", "", value);
    option.value = value;
    type.appendChild(option);
  });
  const del = makeElement("button", "asset-del", "×");
  del.type = "button";
  del.title = "remove";
  del.setAttribute("aria-label", "Remove asset");
  del.addEventListener("click", () => removeAssetRow(row));
  row.append(path, type, del);
  els.assetList.appendChild(row);
}

function removeAssetRow(row) {
  if (els.assetList.querySelectorAll(".asset-row").length > 1) {
    row.remove();
  }
}

function wireTabs() {
  els.tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      els.tabs.forEach((item) => {
        item.classList.remove("active");
        item.setAttribute("aria-selected", "false");
      });
      els.panes.forEach((pane) => pane.classList.remove("active"));
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");
      document.getElementById(`pane-${tab.dataset.pane}`).classList.add("active");
    });
  });
}

function wireUI() {
  wireTabs();
  els.runForm.addEventListener("submit", handleRunSubmit);
  els.loadForm.addEventListener("submit", handleLoadSubmit);
  els.addAsset.addEventListener("click", addAssetRow);
  els.assetList.querySelectorAll(".asset-del").forEach((button) => {
    button.addEventListener("click", (event) => removeAssetRow(event.currentTarget.closest(".asset-row")));
  });
  els.cancelBtn.addEventListener("click", handleCancelClick);
  els.logToggle.addEventListener("click", () => els.runlog.classList.toggle("open"));
  els.verboseToggle.addEventListener("click", () => {
    const enabled = !document.body.classList.contains("verbose");
    document.body.classList.toggle("verbose", enabled);
    els.verboseSwitch.classList.toggle("on", enabled);
    els.verboseToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindElements();
  wireUI();
});
