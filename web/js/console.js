/* MIRA console controller — talks to the FastAPI backend. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* fetch wrapper: never crashes on non-JSON error bodies (the user-facing bug:
   "Unexpected token 'I', \"Internal S...\" is not valid JSON") */
async function api(url, opts = {}) {
  const resp = await fetch(url, opts);
  const text = await resp.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* HTML/plain-text body */ }
  if (!resp.ok) {
    const detail = data && (data.detail ?? data.error);
    throw new Error(detail ? String(detail) : `${resp.status} ${resp.statusText}`);
  }
  return data;
}

/* ---------- view switching ---------- */
document.querySelectorAll(".side-nav button").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".side-nav button").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    btn.classList.add("active");
    $("view-" + btn.dataset.view).classList.add("active");
    if (btn.dataset.view === "mandala" && window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  });
});

/* ---------- overview ---------- */
let __bootTries = 0;
async function loadOverview() {
  try {
    const d = await api("/api/system");
    const w = d.workspace || {};
    $("stats").innerHTML = [
      ["Memory nodes", w.nodes ?? 0], ["Edges", w.edges ?? 0],
      ["Documents", w.documents ?? 0], ["Vectors", w.vectors ?? 0],
    ].map(([k, v]) => `<div class="stat"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
    const r = d.runtime || {};
    $("runtime").innerHTML = Object.entries(r)
      .map(([k, v]) => `<div><span class="k">${esc(k)}</span><span class="v">${esc(String(v))}</span></div>`).join("");
    const notes = [];
    if (!(d.llm || {}).available) notes.push("No local LLM loaded — answers use the extractive fallback.");
    if ((d.embeddings || {}).backend !== "st") notes.push("Embeddings on hashing fallback — retrieval quality degraded.");
    (d.warnings || []).forEach(x => notes.push(esc(x)));
    $("notices").innerHTML = notes.length
      ? notes.map(n => `<li>${n}</li>`).join("")
      : "<li>All systems nominal.</li>";
    $("side-status").innerHTML =
      `<strong>${esc(d.hardware.gpu_name || d.hardware.cpu_name || "?")}</strong><br>` +
      `tier ${esc(String(d.runtime.tier ?? d.hardware.tier ?? "?"))} · ${esc(String(d.runtime.performance_mode || ""))}<br>` +
      `${w.nodes ?? 0} nodes · ${w.documents ?? 0} docs`;
  } catch (e) {
    __bootTries++;
    $("side-status").textContent = __bootTries <= 24
      ? `Warming up… loading embedding model and LLM (${__bootTries})`
      : "API offline — is the server running?";
    if (__bootTries <= 24) setTimeout(loadOverview, 5000);
  }
}

/* ---------- chat ---------- */
$("chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("chat-q").value.trim();
  if (!q) return;
  const set = $("chat-set").value;
  $("chat-out").insertAdjacentHTML("beforeend",
    `<div class="bubble"><strong>You</strong><p>${esc(q)}</p></div>`);
  $("chat-q").value = "";
  const busy = document.createElement("div");
  busy.className = "bubble"; busy.textContent = "Retrieving and composing…";
  $("chat-out").appendChild(busy);
  try {
    const d = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, components: set, allow_web: $("chat-web").checked }),
    });
    const m = d.metrics || {};
    const rows = (d.memories || []).slice(0, 6).map((x, i) =>
      `<tr><td>${i + 1}</td><td>${esc(x.concept)}</td><td>${esc(x.type)}</td>` +
      `<td>${x.ring ?? "—"}</td><td>${(+x.score).toFixed(3)}</td></tr>`).join("");
    const src = (d.sources || []).map(s => `<li>${esc(s)}</li>`).join("");
    const amode = d.agent_mode || "memory";
    const amodeTag = amode === "memory" ? "from memory"
      : amode === "web" ? "fetched live from the web"
      : "model knowledge — ungrounded";
    busy.innerHTML = `
      <strong>Answer <span class="muted small">(${esc(d.mode)} · ${esc(amodeTag)})</span></strong>
      <p>${esc(d.answer)}</p>
      ${d.paths && d.paths.length ? `<div class="path">${d.paths.map(esc).join("<br>")}</div>` : ""}
      <div class="meta"><span>${(m.latency_ms ?? 0).toFixed ? (m.latency_ms).toFixed(0) : m.latency_ms} ms</span>
        <span>${m.n_memories ?? "—"} memories</span>
        <span>${m.context_tokens ?? "—"} ctx tokens</span>
        <span>compression ${(m.compression_ratio ?? 0) * 1 || "—"}×</span></div>
      ${rows ? `<details><summary>Retrieved memories</summary><table><tr><th>#</th><th>concept</th><th>type</th><th>ring</th><th>score</th></tr>${rows}</table></details>` : ""}
      ${src ? `<details><summary>Sources</summary><ul class="ticks small">${src}</ul></details>` : ""}`;
  } catch (err) {
    busy.innerHTML = `<p class="small" style="color:var(--warn)">Error: ${esc(err.message)}</p>`;
  }
});

/* ---------- documents ---------- */
async function loadDocs() {
  const d = await api("/api/documents");
  $("doc-list").innerHTML = (d.documents || []).map(x =>
    `<div class="doc-row"><span>${esc(x.title)} <span class="muted small">· ${x.n_chunks} chunks · ${x.file_type}</span></span>
     <button class="del" data-id="${x.id}">delete</button></div>`).join("")
    || `<p class="small muted">Nothing ingested yet.</p>`;
  document.querySelectorAll(".doc-row .del").forEach(b =>
    b.addEventListener("click", async () => {
      await fetch("/api/documents/" + b.dataset.id, { method: "DELETE" });
      loadDocs();
    }));
}
$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("file", $("doc-file").files[0]);
  fd.append("title", $("doc-title").value);
  $("upload-status").textContent = "Ingesting… (chunking, embedding, placing on the mandala)";
  try {
    const d = await api("/api/documents", { method: "POST", body: fd });
    $("upload-status").textContent =
      `Ingested: ${d.n_chunks} chunks → ${d.n_nodes} nodes, ${d.n_edges} edges.`;
    loadDocs();
  } catch (err) {
    $("upload-status").textContent = "Ingest failed: " + (err.message || err);
  }
});

/* ---------- research lab ---------- */
$("lab-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("lab-q").value.trim();
  if (!q) return;
  $("lab-out").innerHTML = `<p class="small muted">Running…</p>`;
  const d = await api("/api/lab", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: q }),
  });
  if (d.detail) { $("lab-out").innerHTML = `<p class="small" style="color:var(--warn)">${esc(d.detail)}</p>`; return; }
  let html = "";
  for (const [name, r] of Object.entries(d.results)) {
    html += `<h3 style="margin-top:1.4rem">${esc(name)} <span class="muted small">· ${r.latency_ms.toFixed(1)} ms · ${r.n_candidates} candidates</span></h3>`;
    html += `<table class="lab-table"><tr><th>#</th><th>concept</th><th>type</th><th>ring</th><th>score</th><th>hops</th></tr>`;
    r.items.slice(0, 8).forEach((it, i) => {
      html += `<tr><td>${i + 1}</td><td>${esc(it.concept)}</td><td>${esc(it.type)}</td><td>${it.ring ?? "—"}</td><td>${(+it.score).toFixed(3)}</td><td>${it.hops}</td></tr>`;
    });
    html += `</table>`;
  }
  html += `<h3 style="margin-top:1.4rem">Jaccard overlap</h3><pre class="formula">${esc(JSON.stringify(d.jaccard, null, 1))}</pre>`;
  $("lab-out").innerHTML = html;
});

/* ---------- benchmarks ---------- */
$("bench-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("bench-out").innerHTML = `<p class="small muted">Running benchmark… (this really runs every system)</p>`;
  try {
    const d = await api("/api/benchmark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_path: $("bench-ds").value, format: $("bench-fmt").value,
        limit: +$("bench-n").value, judge: $("bench-judge").checked,
      }),
    });
    const cols = ["system", "retrieval_recall", "mrr", "answer_token_f1", "context_tokens", "latency_ms"];
    let html = `<p class="small muted">${d.n_questions} questions · experiment ${esc(d.experiment_id || "not saved")}</p>`;
    if (d.judge) html += `<p class="small">judge: correctness ${d.judge.judge_correctness ?? "—"} / faithfulness ${d.judge.judge_faithfulness ?? "—"} (${d.judge.n_judged ?? 0} judged)</p>`;
    html += `<table class="bench-table"><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr>`;
    for (const row of d.table) {
      html += `<tr>${cols.map(c => `<td>${row[c] ?? "—"}</td>`).join("")}</tr>`;
    }
    html += `</table>`;
    $("bench-out").innerHTML = html;
  } catch (err) {
    $("bench-out").innerHTML = `<p class="small" style="color:var(--warn)">Error: ${esc(err.message)}</p>`;
  }
});

/* ---------- live web search ---------- */
$("ws-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("ws-q").value.trim();
  if (!q) return;
  $("ws-out").innerHTML = `<p class="small muted">Searching…</p>`;
  try {
    const d = await api("/api/websearch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, backend: $("ws-backend").value }),
    });
    if (!d.results || !d.results.length) {
      $("ws-out").innerHTML = `<p class="small" style="color:var(--warn)">${esc(d.note || "no results")}</p>`;
      return;
    }
    $("ws-out").innerHTML =
      `<p class="small muted">backend: ${esc(d.backend)} · ${d.results.length} results</p>` +
      d.results.map((r, i) => `
        <div class="doc-row">
          <span><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.title || r.url)}</a>
            <div class="muted small">${esc((r.snippet || "").slice(0, 200))}</div></span>
          <button class="btn btn-secondary ws-ingest" data-i="${i}">Ingest into memory</button>
        </div>`).join("");
    let lastResults = d.results;
    document.querySelectorAll(".ws-ingest").forEach(b =>
      b.addEventListener("click", async () => {
        const r = lastResults[+b.dataset.i];
        b.textContent = "Fetching…"; b.disabled = true;
        try {
          const d2 = await api("/api/websearch/ingest", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: r.url, title: r.title || r.url }),
          });
          const s = d2.ingested || {};
          b.textContent = `✓ ${s.n_chunks ?? "?"} chunks → ${s.n_nodes ?? "?"} nodes`;
        } catch (err) {
          b.textContent = "failed"; b.disabled = false;
        }
      }));
  } catch (err) {
    $("ws-out").innerHTML = `<p class="small" style="color:var(--warn)">Error: ${esc(err.message)}</p>`;
  }
});

/* ---------- placement strategy lab ---------- */
async function initStrategyLab() {
  try {
    const d = await api("/api/strategies");
    const sel = $("strat-select");
    sel.innerHTML = (d.strategies || []).map(s =>
      `<option value="${esc(s)}" ${s === d.current ? "selected" : ""}>${esc(s)}</option>`).join("");
  } catch (e) { /* mandala view only */ }
}
$("strat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const s = $("strat-select").value;
  $("strat-out").innerHTML = `<p class="muted">Re-placing all memories with “${esc(s)}”…</p>`;
  try {
    const d = await api("/api/strategies/" + encodeURIComponent(s) + "/apply", { method: "POST" });
    const r = d.info && d.info.radial;
    $("strat-out").innerHTML = `<p>✓ applied <strong>${esc(d.applied)}</strong>` +
      (r ? ` — radial distances recomputed for ${r.computed} nodes` : "") + `.</p>`;
    if (window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  } catch (err) {
    $("strat-out").innerHTML = `<p style="color:var(--warn)">Error: ${esc(err.message)}</p>`;
  }
});
$("strat-sweep").addEventListener("click", async () => {
  $("strat-out").innerHTML = `<p class="muted">Benchmarking all strategies (place → measure → restore)…</p>`;
  try {
    const d = await api("/api/strategies/sweep", { method: "POST" });
    let html = `<table class="bench-table"><tr><th>strategy</th><th>recall</th><th>mrr</th><th>latency ms</th></tr>`;
    for (const r of d.table) {
      html += `<tr${r.system === d.best ? " style='color:var(--gold)'" : ""}>` +
        `<td>${esc(r.system)}</td><td>${r.retrieval_recall ?? "—"}</td>` +
        `<td>${r.mrr ?? "—"}</td><td>${(r.latency_ms ?? 0).toFixed ? r.latency_ms.toFixed(1) : r.latency_ms}</td></tr>`;
    }
    html += `</table><p class="muted small">best: ${esc(d.best)} (restored: ${esc(d.restored)})</p>`;
    $("strat-out").innerHTML = html;
    if (window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  } catch (err) {
    $("strat-out").innerHTML = `<p style="color:var(--warn)">Error: ${esc(err.message)}</p>`;
  }
});

loadOverview();
loadDocs();
initStrategyLab();
