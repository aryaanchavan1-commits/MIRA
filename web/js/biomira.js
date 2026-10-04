/* BioMIRA console views: Biological Memory + Catastrophic Forgetting Lab.
 *
 * Every number rendered here comes from the API (measured runs) or is
 * labelled "not measured yet". Nothing is computed client-side that the
 * backend did not measure.
 */
"use strict";

/* ---------- shared bits ---------- */
const bioNum = (value, digits = 3) => {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
};
const bioSigned = (value, digits = 3) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
};
const bioStateBadge = (state) => {
  const known = ["new", "candidate", "stable", "consolidated"];
  const label = known.includes(state) ? state : "new";
  return `<span class="badge badge-bio-${esc(label)}">${esc(label)}</span>`;
};

/* ---------- layer switch + stats ---------- */
async function loadBioState() {
  const stats = $("bio-stats");
  if (!stats) return;
  setBusy(stats, true);
  try {
    const data = await api("/api/biomira/state");
    $("bio-enabled").checked = Boolean(data.enabled);
    if (document.activeElement !== $("bio-kappa")) {
      $("bio-kappa").value = bioNum(data.config?.kappa_stability ?? 0, 2);
    }
    // dynamics counters are meaningless while the layer is off: printing 0.000
    // reads as "measured, nothing happening" rather than "not measured"
    const on = Boolean(data.enabled);
    const dyn = (v, digits = 3) => (on ? bioNum(v, digits) : "—");
    const cells = [
      ["Memories", data.nodes ?? "—"],
      ["Edges", data.edges ?? "—"],
      ["Active now", on ? (data.active_count ?? "—") : "—"],
      ["Mean activation", dyn(data.mean_activation)],
      ["At risk", on ? (data.at_risk_count ?? "—") : "—"],
      ["Mean uses / memory", dyn(data.mean_access_count, 2)],
      ["Replayed memories", on ? (data.replayed_nodes ?? "—") : "—"],
      ["Replay buffer", on ? `${data.replay_buffer_size ?? "—"} max` : "—"],
    ];
    stats.innerHTML = cells.map(([label, value]) =>
      `<div class="stat"><div class="v">${esc(String(value))}</div>` +
      `<div class="k">${label}</div></div>`).join("");
    const hist = data.states || {};
    const total = Object.values(hist).reduce((a, b) => a + Number(b || 0), 0) || 1;
    const bar = Object.entries(hist).map(([k, v]) =>
      `<div class="bio-hist-row"><span class="small">${bioStateBadge(k)}</span>` +
      `<span class="bio-hist-bar"><span style="width:${(100 * Number(v || 0) / total).toFixed(1)}%"></span></span>` +
      `<span class="small">${esc(String(v ?? 0))}</span></div>`).join("");
    const stale = document.getElementById("bio-hist-card");
    if (stale) stale.remove();
    stats.insertAdjacentHTML("afterend",
      `<div class="card" id="bio-hist-card"><h2>Consolidation states</h2>${bar}
        <p class="small muted">A memory walks NEW → CANDIDATE → STABLE → CONSOLIDATED as it is
        retrieved successfully and survives decay. State only advances on evidence: use it, or
        it falls back. The ladder is the mechanism; the forgetting lab below is the test of
        whether it helps.</p></div>`);
  } catch (err) {
    setStatus(stats, `Could not load BioMIRA state: ${err.message || err}`, "error");
  } finally {
    setBusy(stats, false);
  }
}

async function applyBioToggle() {
  const out = $("bio-toggle-out");
  const btn = $("bio-apply-toggle");
  setPending(btn, true);
  try {
    const res = await api("/api/biomira/enabled", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: $("bio-enabled").checked,
        kappa_stability: Number($("bio-kappa").value) || 0,
      }),
    });
    setStatus(out, `BioMIRA ${res.enabled ? "enabled" : "disabled"} · κ = ${bioNum(res.kappa_stability, 2)}. ${res.note}`, "success");
    await loadBioState();
    await loadBioMemories();
  } catch (err) {
    setStatus(out, `Could not change the layer: ${err.message || err}`, "error");
  } finally {
    setPending(btn, false);
  }
}

/* ---------- dynamics pass ---------- */
async function runBioStep(event) {
  event.preventDefault();
  const out = $("bio-step-out");
  const btn = $("bio-step-run");
  const actions = Array.from(document.querySelectorAll(".bio-action:checked"))
    .map((el) => el.value);
  if (!actions.length) {
    setStatus(out, "Pick at least one mechanism to run.", "error");
    return;
  }
  setPending(btn, true);
  setStatus(out, "Running dynamics pass…");
  try {
    const res = await api("/api/biomira/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ actions, apply: $("bio-step-apply").checked }),
    });
    const lines = [];
    if (res.decay) {
      lines.push(`decay: ${res.decay.decayed}/${res.decay.nodes} faded, mean retention ${bioNum(res.decay.retention_mean)}`);
    }
    if (res.replay) lines.push(`replay: ${res.replay.replayed ?? 0} of ${res.replay.buffer ?? "?"} memories reactivated`);
    if (res.homeostasis) lines.push(`homeostasis: cap ${bioNum(res.homeostasis.cap, 2)}, ${res.homeostasis.compressed ?? 0} compressed`);
    if (res.rings) lines.push(`radial: ${res.rings.promoted ?? 0} promoted, ${res.rings.demoted ?? 0} demoted`);
    if (res.merge) lines.push(`merge: ${res.merge.pairs ?? 0} pairs (${(res.merge.duplicates || []).length} duplicate, ${(res.merge.conflicts || []).length} conflicting)`);
    setStatus(out, `${res.applied ? "Applied" : "Dry run — nothing written"}: ${lines.join(" · ")}`, "success");
    await loadBioState();
    await loadBioMemories();
  } catch (err) {
    setStatus(out, `Dynamics pass failed: ${err.message || err}`, "error");
  } finally {
    setPending(btn, false);
  }
}

/* ---------- memory population ---------- */
async function loadBioMemories() {
  const out = $("bio-table-out");
  if (!out) return;
  setBusy(out, true);
  const state = $("bio-filter").value;
  const sort = $("bio-sort").value;
  try {
    const data = await api(
      `/api/biomira/memories?sort=${encodeURIComponent(sort)}&limit=40` +
      (state ? `&state=${encodeURIComponent(state)}` : ""));
    if (!data.enabled) {
      out.innerHTML = `<p class="small muted">BioMIRA is off — the table below shows plain MIRA
        memories with no dynamics yet. Enable the layer to accumulate stability, activation
        and consolidation state.</p>`;
      return;
    }
    const rows = (data.memories || []).map((m) => `<tr>
      <td><button type="button" class="linkish" data-bio-node="${esc(m.id)}">${esc(m.concept)}</button></td>
      <td>${bioStateBadge(m.state)}</td>
      <td>${esc(m.ring ?? "—")}</td>
      <td>${bioNum(m.activation, 2)}</td>
      <td>${bioNum(m.stability, 2)}</td>
      <td>${bioNum(m.retention, 2)}</td>
      <td>${bioNum(m.importance, 2)}</td>
      <td>${esc(String(m.access_count ?? 0))}</td>
      <td class="small muted">v${esc(String(m.version ?? 1))}</td>
    </tr>`).join("");
    out.innerHTML = `<p class="small muted" id="bio-memories-cap">${esc(String(data.count))} memories,
      sorted by ${esc(data.sort)}</p>
    <div class="table-wrap"><table class="lab-table" aria-describedby="bio-memories-cap">
      <thead><tr><th scope="col">Memory</th><th scope="col">State</th><th scope="col">Ring</th>
      <th scope="col">Activation</th><th scope="col">Stability</th><th scope="col">Retention</th>
      <th scope="col">Importance</th><th scope="col">Uses</th><th scope="col">Version</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="9" class="small muted">No memories match this filter.</td></tr>`}</tbody>
    </table></div>
    <p class="small muted">Retention below 0.5 means the memory is losing retrievability and is a
      replay candidate. Select a memory to see why it decayed or consolidated.</p>`;
  } catch (err) {
    setStatus(out, `Could not load memories: ${err.message || err}`, "error");
  } finally {
    setBusy(out, false);
  }
}

/* ---------- memory explanation (spec §19) ---------- */
async function showBioExplain(nodeId) {
  const out = $("bio-explain-out");
  if (!out) return;
  out.innerHTML = `<div class="card"><h2>Why this memory?</h2>
    <p class="small muted">Loading ${esc(nodeId)}…</p></div>`;
  try {
    const node = await api(`/api/node/${encodeURIComponent(nodeId)}`);
    const b = node.biomira || {};
    if (b.enabled === false) {
      out.innerHTML = `<div class="card"><h2>Why this memory?</h2>
        <p class="small muted">BioMIRA is off, so plain MIRA's ranking is the only story here:
        semantic + graph + hierarchy + radial + temporal. Enable the layer to see stability,
        activation and consolidation.</p></div>`;
      return;
    }
    const num = (v) => bioNum(v, 2);
    const rows = [
      ["Activation", num(b.activation)], ["Stability", num(b.stability)],
      ["Importance", num(b.importance)], ["Confidence", num(b.confidence)],
      ["Consolidation score", num(b.consolidation_score)],
      ["Retention", num(b.retention)], ["Decay rate", num(b.decay_rate)],
      ["Strength multiplier", num(b.strength_multiplier)],
      ["Retrievals", String(b.access_count ?? 0)],
      ["Retrieval failures", String(b.failures ?? 0)],
      ["Memory version", String(b.memory_version ?? 1)],
      ["Radial position", `ring ${esc(String(b.ring ?? "—"))} · sector ${esc(String(b.sector ?? "—"))}`],
    ];
    out.innerHTML = `<div class="card">
      <h2>Why this memory? ${bioStateBadge(b.state)}</h2>
      <p class="small muted">${esc(b.concept || nodeId)}</p>
      <div class="kv">${rows.map(([k, v]) =>
        `<div><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("")}</div>
      <details open><summary>Why did it decay?</summary>
        <p class="small muted">${esc(b.why_decayed || "no decay history recorded")}</p></details>
      <details><summary>Why is it consolidated at ${esc(String(b.state ?? "—"))}?</summary>
        <p class="small muted">${esc(b.why_consolidated || "not enough evidence yet")}</p></details>
      ${b.superseded_by ? `<p class="small muted">Superseded by <code>${esc(b.superseded_by)}</code> —
        kept for historical questions, not deleted.</p>` : ""}
      ${node.summary ? `<p class="small">${esc(node.summary)}</p>` : ""}
    </div>`;
  } catch (err) {
    out.innerHTML = `<div class="card"><h2>Why this memory?</h2>
      <p class="small status-error" role="alert">Could not load ${esc(nodeId)}: ${esc(err.message || err)}</p></div>`;
  }
}

/* ---------- forgetting lab ---------- */
const LAB_METRICS = [
  ["mrr", "MRR", "reciprocal rank of the first gold evidence node"],
  ["recall", "Recall", "share of gold evidence nodes in the top k"],
  ["answer_coverage", "Answer coverage", "share of the gold answer's words present in the best returned passage"],
  ["answer_found", "Answer found", "share of questions whose gold answer appears verbatim in the top k"],
];

function labChart(rows, metric = "mrr") {
  if (!rows || !rows.length) return "";
  const spec = LAB_METRICS.find((m) => m[0] === metric) || LAB_METRICS[0];
  const [key, label] = spec;
  const W = 720, H = 240, pad = 34;
  const xs = [...new Set(rows.map((r) => r.step))].sort((a, b) => a - b);
  const x = (step) => pad + (W - 2 * pad) * (xs.indexOf(step) / Math.max(1, xs.length - 1));
  // Every one of these metrics sits in a narrow band near its own mean, so a
  // fixed 0-1 axis would flatten every series into one line. Scale to the data
  // and say so on the axis, rather than quietly cropping.
  const vals = rows.map((r) => Number(r[key]) || 0);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = Math.max(hi - lo, 0.02);
  const yLo = Math.max(0, lo - span * 0.25), yHi = Math.min(1, hi + span * 0.25);
  const y = (v) => H - pad - (H - 2 * pad) *
    ((Math.min(1, Math.max(0, Number(v) || 0)) - yLo) / (yHi - yLo || 1));
  const tasks = [...new Set(rows.map((r) => r.task))].sort((a, b) => a - b);
  const colors = ["#e0703a", "#3f8f7a", "#7a6bd0", "#b0894a", "#4a7fb0",
                  "#c2557a", "#5f8f3f", "#8a6f4a"];
  const series = tasks.map((t) => {
    const pts = rows.filter((r) => r.task === t).sort((a, b) => a.step - b.step);
    const path = pts.map((p, i) =>
      `${i ? "L" : "M"}${x(p.step).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
    const dots = pts.map((p) =>
      `<circle cx="${x(p.step).toFixed(1)}" cy="${y(p[key]).toFixed(1)}" r="3.5"
        fill="${colors[t % colors.length]}"><title>Task ${String.fromCharCode(65 + t)}, after task ${p.step + 1}:
        ${label} ${bioNum(p[key])} · MRR ${bioNum(p.mrr)} · recall ${bioNum(p.recall)} · n=${esc(String(p.n ?? "?"))}</title></circle>`).join("");
    return `<path d="${path}" fill="none" stroke="${colors[t % colors.length]}" stroke-width="2"/><g>${dots}</g>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => yLo + f * (yHi - yLo));
  const grid = ticks.map((v) =>
    `<line x1="${pad}" x2="${W - pad}" y1="${y(v)}" y2="${y(v)}" class="chart-grid"/>
     <text x="${pad - 6}" y="${y(v) + 4}" class="chart-label" text-anchor="end">${v.toFixed(2)}</text>`).join("");
  const steps = xs.map((s) =>
    `<text x="${x(s)}" y="${H - pad + 16}" class="chart-label" text-anchor="middle">after ${s + 1}</text>`).join("");
  const legend = tasks.map((t) =>
    `<span><span class="dot" style="background:${colors[t % colors.length]}"></span>Task ${String.fromCharCode(65 + t)}</span>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" role="img" class="chart"
    aria-label="${esc(label)} per task across ${xs.length} learning steps, y axis zoomed to ${yLo.toFixed(2)}-${yHi.toFixed(2)}. ${esc(spec[2])}.">
    ${grid}<text x="${pad}" y="${pad - 14}" class="chart-axis">${esc(label)}</text>${steps}<g>${series}</g></svg><div class="legend">${legend}</div>`;
}

/* Interval sensitivity: does the ranking survive a different gap between
 * tasks? One table beats a paragraph of hand-waving about robustness. */
function labSweep(data) {
  const sweep = data.interval_sweep;
  if (!sweep || !Object.keys(sweep).length) return "";
  const days = Object.keys(sweep).sort((a, b) => Number(a) - Number(b));
  const variants = [...new Set(days.flatMap((d) => Object.keys(sweep[d])))];
  const cell = (v, key, signed) => {
    if (!v || v[key] == null) return "—";
    return signed ? bioSigned(v[key], 4) : bioNum(v[key]);
  };
  const body = variants.map((name) => `<tr>
    <th scope="row">${esc(name)}</th>
    ${days.map((d) => `<td>${cell(sweep[d]?.[name], "final_mrr")}</td>`).join("")}
    ${days.map((d) => `<td>${cell(sweep[d]?.[name], "final_forgetting", true)}</td>`).join("")}
  </tr>`).join("");
  return `<h3 class="subhead">Interval sensitivity</h3>
    <p class="small muted" id="lab-sweep-cap">${esc(data.meta?.sweep_note || "Final MRR at each simulated gap between task arrivals.")}</p>
    <div class="table-wrap"><table class="lab-table" aria-describedby="lab-sweep-cap">
      <thead><tr><th scope="col">Variant</th>
        ${days.map((d) => `<th scope="col">MRR @ ${esc(d)}d</th>`).join("")}
        ${days.map((d) => `<th scope="col">Forgetting @ ${esc(d)}d</th>`).join("")}
      </tr></thead><tbody>${body}</tbody></table></div>`;
}

async function loadForgettingLab() {
  const tasksOut = $("lab-tasks-out");
  const metricSel = $("lab-metric");
  const variantsOut = $("lab-variants-out");
  const chartOut = $("lab-chart-out");
  if (!tasksOut) return;
  setBusy(variantsOut, true);
  try {
    const data = await api("/api/lab/forgetting");
    const plan = data.tasks;
    const names0 = (plan?.tasks || []).map((t) => t.name);
    const intro = $("lab-intro");
    // the sequence length is data, not prose: a hardcoded "A -> D" is a lie
    // the moment the corpus is rebuilt with more shards
    if (intro && names0.length) {
      const arrow = names0.join(" → ");
      intro.innerHTML = `Sequential tasks ${esc(arrow)} (${names0.length} shards). After every task,
        <strong>all</strong> tasks learned so far are re-tested, and forgetting is measured as the
        drop from each task's own best earlier score. Scope: retrieval-level (MRR / recall) with the
        LLM frozen — this measures <strong>memory interference</strong>, not parametric forgetting.`;
    }
    tasksOut.innerHTML = plan ? `<div class="grid-2">${plan.tasks.map((t) => `
      <div class="card"><h3>Task ${esc(t.name)}</h3>
        <p class="bench-figure">${esc(String(t.questions))}</p>
        <p class="small muted">questions tested from this step on ·
          ${esc(String(t.documents))} documents + ${esc(String(t.distractors))} distractors</p>
      </div>`).join("")}</div>` : "";
    if (!data.available) {
      variantsOut.innerHTML = `<p class="small muted">Not measured yet — ${esc(data.reason || "")}.
        Until it is, this page shows nothing rather than an estimate.</p>`;
      chartOut.innerHTML = "";
      $("lab-variant-select").innerHTML = "";
      return;
    }
    const names = Object.keys(data.variants || {});
    const base = data.variants?.["B_mira"] || {};
    const sig = data.significance?.vs_baseline || {};
    const rows = names.map((name) => {
      const v = data.variants[name];
      const dMRR = Number(v.final_mrr) - Number(base.final_mrr || 0);
      const s = sig[name]?.rr;
      // a mean difference with a p-value is a claim; without one it is not.
      const verdict = name === "B_mira" ? "baseline"
        : !s ? "not tested"
          : s.p_value <= 0.05
            ? `${s.mean_diff > 0 ? "significantly better" : "significantly WORSE"} (p=${bioNum(s.p_value, 4)})`
            : `${Math.abs(dMRR) < 0.005 ? "no change" : s.mean_diff > 0 ? "trends better" : "trends worse"}, not significant`;
      return `<tr>
        <th scope="row">${esc(name)}</th>
        <td>${esc(v.spec?.kappa ? `κ=${v.spec.kappa}` : "—")}</td>
        <td>${bioNum(v.final_mrr)}</td>
        <td>${bioNum(v.final_recall)}</td>
        <td>${bioNum(v.final_answer_coverage)}</td>
        <td>${bioNum(v.final_answer_found)}</td>
        <td>${bioSigned(v.average_forgetting, 4)}</td>
        <td>${bioNum(v.mean_retention, 3)}</td>
        <td>${esc(verdict)}</td>
      </tr>`;
    }).join("");
    const nTasks = Number(data.meta?.n_tasks) || 0;
    const lastTask = nTasks ? String.fromCharCode(64 + nTasks) : "—";
    const planned = Number(data.tasks?.tasks?.length) || 0;
    const stale = planned && nTasks && planned !== nTasks;
    variantsOut.innerHTML = `<p class="small muted" id="lab-ladder-cap">Final retrieval after task
        ${esc(lastTask)} · ${esc(String(nTasks))} sequential tasks ·
        ${esc(String(data.meta?.per_task ?? ""))} questions sampled per task</p>
      <div class="table-wrap"><table class="lab-table" aria-describedby="lab-ladder-cap">
      <thead><tr><th scope="col">Variant</th><th scope="col">Stability</th><th scope="col">MRR</th>
        <th scope="col">Recall</th><th scope="col">Ans. coverage</th><th scope="col">Ans. found</th>
        <th scope="col">Avg forgetting</th><th scope="col">Retention</th>
        <th scope="col">Verdict vs MIRA</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      ${stale ? `<p class="small status-error" role="alert">The corpus on disk has ${planned} tasks
        but this table was measured on ${nTasks}. Rebuild and re-run the lab
        (scripts/build_forgetting_lab.py, then scripts/eval_forgetting.py).</p>` : ""}
      ${data.meta?.complete === false ? `<p class="small muted" role="status">This run was still
        in progress when it was written — some variants may be missing.</p>` : ""}
      <p class="small muted">${esc(data.meta?.scope || "")}
        ${data.meta?.time_model ? ` · Time model: ${esc(data.meta.time_model)}.` : ""}
        ${data.meta?.ram_rss_gb?.end ? `Runner RSS ${esc(String(data.meta.ram_rss_gb.end))} GB ·
          lab store ${esc(String(Math.round((data.meta.lab_store_bytes || 0) / 1e6)))} MB.` : ""}
        ${data.significance ? `Significance: ${esc(data.significance.test)} —
          ${esc(String(Object.values(sig)[0]?.rr?.n_pairs ?? "?"))} paired questions against
          ${esc(data.significance.baseline)}.</p>` : ""}
      ${labSweep(data)}`;
    const select = $("lab-variant-select");
    const keep = select.value || (names.includes("B_mira") ? "B_mira" : names[0]);
    select.innerHTML = names.map((n) =>
      `<option value="${esc(n)}"${n === keep ? " selected" : ""}>${esc(n)}</option>`).join("");
    const chosen = data.variants[select.value];
    const metric = $("lab-metric")?.value || "mrr";
    chartOut.innerHTML = chosen ? labChart(chosen.rows, metric)
      : `<p class="small muted">No rows for this variant.</p>`;
  } catch (err) {
    setStatus(variantsOut, `Could not load the lab: ${err.message || err}`, "error");
  } finally {
    setBusy(variantsOut, false);
  }
}

/* ---------- wiring ---------- */
(function initBioMIRA() {
  if (!$("bio-stats")) return;
  $("bio-apply-toggle").addEventListener("click", applyBioToggle);
  $("bio-step-form").addEventListener("submit", runBioStep);
  $("bio-refresh").addEventListener("click", () => { loadBioState(); loadBioMemories(); });
  $("bio-filter").addEventListener("change", loadBioMemories);
  $("bio-sort").addEventListener("change", loadBioMemories);
  document.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-bio-node]");
    if (btn) showBioExplain(btn.dataset.bioNode);
  });
  $("lab-variant-select").addEventListener("change", loadForgettingLab);
  $("#lab-metric")?.addEventListener("change", loadForgettingLab);
  document.querySelectorAll(".side-nav button").forEach((btn) => {
    if (btn.dataset.view === "biomira") btn.addEventListener("click", loadBioState);
    if (btn.dataset.view === "forgetting") btn.addEventListener("click", loadForgettingLab);
  });
  // deep links (#forgetting) activate the view without a click, so the lab has
  // to load on hashchange too — otherwise the screen renders empty
  window.addEventListener("hashchange", () => {
    const view = location.hash.slice(1);
    if (view === "forgetting") loadForgettingLab();
    if (view === "biomira") { loadBioState(); loadBioMemories(); }
  });
  loadBioState();
  loadBioMemories();
  if (location.hash.slice(1) === "forgetting") loadForgettingLab();
})();
