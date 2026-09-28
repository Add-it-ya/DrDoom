// The dashboard's behaviour. The rule in index.html holds for every line here: a value
// from the api is either textContent or it is sanitised. It is a file of its own so the
// Content Security Policy can refuse inline script.

"use strict";

const $ = (id) => document.getElementById(id);
let incidentId = null;

/** Append a line to the activity log. Always textContent. */
function log(message, isError) {
  const line = document.createElement("div");
  if (isError) line.className = "err";
  line.textContent = new Date().toLocaleTimeString() + "  " + message;
  $("log").appendChild(line);
}

/** Build a definition list. Values are set with textContent, never parsed as markup. */
function details(pairs) {
  const list = document.createElement("dl");
  list.className = "kv";
  for (const [label, value, pillClass] of pairs) {
    if (value === null || value === undefined || value === "") continue;
    const term = document.createElement("dt");
    term.textContent = label;
    const definition = document.createElement("dd");
    if (pillClass) {
      const pill = document.createElement("span");
      pill.className = "pill " + pillClass;
      pill.textContent = String(value);
      definition.appendChild(pill);
    } else {
      definition.textContent = String(value);
    }
    list.append(term, definition);
  }
  return list;
}

function fill(stage, node) {
  const body = $("b-" + stage);
  body.replaceChildren(node);
  $("s-" + stage).classList.add("active");
}

function reset() {
  incidentId = null;
  for (const stage of ["triage", "diagnose", "remediate", "execute"]) {
    $("b-" + stage).replaceChildren();
    $("s-" + stage).classList.remove("active");
  }
  $("gate").hidden = true;
  $("report").hidden = true;
  $("report").replaceChildren();
  $("log").replaceChildren();
}

function showTriage(data) {
  const t = data.triage || {};
  fill("triage", details([
    ["Incident", t.is_anomaly ? "detected" : "nothing detected"],
    ["Score", t.score],
    ["Threshold", t.threshold],
    ["Predicted cause", t.root_cause],
    ["Confidence", t.confidence],
  ]));
}

function showDiagnosis(data) {
  const d = data.diagnosis || {};
  const wrapper = document.createElement("div");
  wrapper.appendChild(details([
    ["Summary", d.summary],
    ["Likely cause", d.likely_cause],
    ["Confidence", d.confidence],
    ["Next action", d.next_action],
  ]));
  const cites = data.citations || [];
  if (cites.length) {
    const heading = document.createElement("div");
    heading.className = "hash";
    heading.textContent = "Sources";
    const list = document.createElement("ul");
    for (const citation of cites) {
      const item = document.createElement("li");
      if (citation.url) {
        const link = document.createElement("a");
        link.href = citation.url;           // set as a property, not interpolated markup
        link.rel = "noopener noreferrer";
        link.target = "_blank";
        link.textContent = citation.title;  // title is model-adjacent text: textContent
        item.appendChild(link);
      } else {
        item.textContent = citation.title;
      }
      list.appendChild(item);
    }
    wrapper.append(heading, list);
  }
  fill("diagnose", wrapper);
}

/** How the final risk was reached: three ratings, the most cautious stands. */
function describeRisk(risk) {
  if (!risk) return null;
  const assessor = risk.assessor || (risk.assessor_status === "not_needed"
    ? "not needed (already high)"
    : "no answer (" + risk.assessor_status + "), counted as high");
  return "author " + risk.author + ", policy floor " + risk.floor + ", independent " + assessor;
}

function showPlan(data) {
  const p = data.plan || {};
  fill("remediate", details([
    ["Immediate action", p.immediate_action],
    ["Catalogue action", p.action || "none (approving runs nothing)"],
    ["Risk", p.risk_level, p.risk_level],
    ["Rated by", describeRisk(data.risk)],
    ["Worst case", data.risk && data.risk.worst_case],
    ["Needs approval", p.requires_approval ? "yes" : "no"],
    ["Short-term fix", p.short_term_fix],
    ["Long-term fix", p.long_term_fix],
    ["Rollback", p.rollback],
  ]));
}

function showOutcome(data) {
  const execution = data.execution || {};
  fill("execute", details([
    ["Decision", data.decision],
    ["Executed", execution.executed ? "yes (dry run)" : "no"],
    ["Command", execution.command],
    ["Reason", execution.detail],
    ["Escalation", data.escalation],
  ]));
}

function showGate(pending) {
  const body = $("gateBody");
  body.replaceChildren(details([
    ["Action", pending.immediate_action],
    ["Will run", pending.would_run || "nothing (the plan names no catalogue action)"],
    ["Risk", pending.risk_level, pending.risk_level],
    ["Rated by", describeRisk(pending.risk)],
    ["Worst case", pending.risk && pending.risk.worst_case],
    ["Rollback", pending.rollback],
  ]));
  const hash = document.createElement("div");
  hash.className = "hash";
  hash.textContent = "plan " + String(pending.plan_hash || "").slice(0, 16);
  body.appendChild(hash);
  $("gate").hidden = false;
}

/**
 * The only place model output becomes markup.
 * Sanitised first; the result of marked.parse is never trusted directly.
 */
function showReport(markdown) {
  if (!markdown) return;
  // Assigned directly rather than through a variable, so the rule stays legible:
  // every innerHTML on this page has DOMPurify.sanitize on its right-hand side.
  $("report").innerHTML = DOMPurify.sanitize(
    marked.parse(markdown), { USE_PROFILES: { html: true } }
  );
  $("report").hidden = false;
}

async function investigate(anomalous) {
  reset();
  $("run").disabled = $("runCalm").disabled = true;
  log("requesting a " + (anomalous ? "disturbed" : "calm") + " window");

  try {
    const response = await fetch("/demo/window?anomalous=" + anomalous);
    const payload = await response.json();
    log("streaming the investigation");

    const stream = await fetch("/investigate/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!stream.ok) throw new Error("investigate failed: " + stream.status);

    const reader = stream.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const blocks = buffer.split("\n\n");
      buffer = blocks.pop();
      for (const block of blocks) handleEvent(block);
    }
  } catch (error) {
    log(String(error), true);
  } finally {
    $("run").disabled = $("runCalm").disabled = false;
  }
}

function handleEvent(block) {
  const nameLine = block.split("\n").find((l) => l.startsWith("event: "));
  const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
  if (!nameLine || !dataLine) return;
  const name = nameLine.slice(7).trim();
  let data;
  try { data = JSON.parse(dataLine.slice(6)); } catch { return; }

  log("stage: " + name);
  if (name === "accepted") { incidentId = data.incident_id; }
  else if (name === "triage") { showTriage(data); }
  else if (name === "diagnose") { showDiagnosis(data); }
  else if (name === "remediate" || name === "assess_risk") { showPlan(data); }
  else if (name === "awaiting_approval") { showGate(data); }
  else if (name === "execute" || name === "escalate") { showOutcome(data); }
  else if (name === "report") { showReport(data.report); }
  else if (name === "failed") { log("stopped: " + data.detail, true); }
  else if (name === "done") { log("finished: " + data.status, data.status === "failed"); }
}

async function decide(approved) {
  const key = $("key").value.trim();
  if (!key) { log("an approval key is required", true); return; }
  $("approve").disabled = $("reject").disabled = true;

  try {
    const response = await fetch("/incidents/" + encodeURIComponent(incidentId) + "/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": key },
      body: JSON.stringify({ approved }),
    });
    if (response.status === 401) throw new Error("that key was not accepted");
    if (!response.ok) throw new Error("approval failed: " + response.status);

    const data = await response.json();
    $("gate").hidden = true;
    showOutcome(data);
    showReport(data.report);
    log("decision recorded: " + data.decision);
  } catch (error) {
    log(String(error), true);
  } finally {
    $("approve").disabled = $("reject").disabled = false;
  }
}

// Where each status sits on the page's one colour scale.
const STATUS_PILL = {
  awaiting_approval: "medium", failed: "high", rejected: "high", complete: "low",
};

/** The most recent investigations. Listing needs a key; every value is textContent. */
async function loadRecent() {
  const key = $("key").value.trim();
  if (!key) { log("a key is required to list incidents", true); return; }

  try {
    const response = await fetch("/incidents?limit=20", { headers: { "X-API-Key": key } });
    if (response.status === 401) throw new Error("that key was not accepted");
    if (!response.ok) throw new Error("listing failed: " + response.status);

    const page = await response.json();
    const list = $("incidents");
    list.replaceChildren();
    for (const item of page.incidents) {
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = item.updated_at
        ? new Date(item.updated_at).toLocaleString()
        : item.incident_id;
      const state = document.createElement("span");
      state.className = "pill " + (STATUS_PILL[item.status] || "");
      state.textContent = String(item.status).replace(/_/g, " ");
      const what = document.createElement("span");
      what.className = "what";
      what.textContent = item.likely_cause || item.action
        || (item.is_anomaly ? "incident" : "nothing detected");

      const open = document.createElement("button");
      open.append(when, state, what);
      open.addEventListener("click", () => openIncident(item.incident_id));
      const entry = document.createElement("li");
      entry.appendChild(open);
      list.appendChild(entry);
    }
    log("listed " + page.incidents.length + " of " + page.total + " incidents");
  } catch (error) {
    log(String(error), true);
  }
}

/** Show a stored investigation the way the live run showed it, gate included. */
async function openIncident(id) {
  reset();
  try {
    const response = await fetch("/incidents/" + encodeURIComponent(id), {
      headers: { "X-API-Key": $("key").value.trim() },
    });
    if (response.status === 401) throw new Error("that key was not accepted");
    if (!response.ok) throw new Error("could not open " + id + ": " + response.status);

    const data = await response.json();
    incidentId = data.incident_id;
    log("opened incident " + data.incident_id + ": " + data.status);
    if (data.triage) showTriage(data);
    if (data.diagnosis) showDiagnosis(data);
    if (data.plan) showPlan(data);
    if (data.awaiting) showGate(data.awaiting);
    if (data.decision || data.escalation) showOutcome(data);
    showReport(data.report);
  } catch (error) {
    log(String(error), true);
  }
}

$("loadRecent").addEventListener("click", loadRecent);
$("run").addEventListener("click", () => investigate(true));
$("runCalm").addEventListener("click", () => investigate(false));
$("approve").addEventListener("click", () => decide(true));
$("reject").addEventListener("click", () => decide(false));
