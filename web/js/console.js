/* MIRA console controller — talks to the FastAPI backend. */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const setBusy = (element, busy) => element.setAttribute("aria-busy", String(busy));
const setStatus = (element, message, state = "") => {
  element.textContent = message;
  element.classList.remove("status-error", "status-success");
  if (state) element.classList.add(`status-${state}`);
};
const setPending = (button, busy, label = "Working…") => {
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
  if (busy) {
    button.dataset.idleText = button.textContent;
    button.textContent = label;
  } else {
    button.textContent = button.dataset.idleText || button.textContent;
    delete button.dataset.idleText;
  }
};
const validateRequired = (input, error, message) => {
  const valid = input.files ? input.files.length > 0 : Boolean(input.value.trim());
  input.setAttribute("aria-invalid", String(!valid));
  error.textContent = valid ? "" : message;
  return valid;
};
const revalidateWhenFixed = (input, error, message, event = "input") =>
  input.addEventListener(event, () => {
    if (input.getAttribute("aria-invalid") === "true") validateRequired(input, error, message);
  });
const fmt = (value, digits = 3) => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
};
const boundedAffectValue = (value, bounds) => {
  if (value === null || value === "" || typeof value === "boolean") return "—";
  const number = Number(value);
  return Number.isFinite(number)
    ? Math.min(bounds[1], Math.max(bounds[0], number)).toFixed(3)
    : "—";
};
const affectText = (value, fallback, limit = 240) => {
  const text = typeof value === "string" ? value.trim() : "";
  return esc((text || fallback).slice(0, limit));
};
const affectMarkup = (snapshot) => {
  const state = snapshot && typeof snapshot === "object" && !Array.isArray(snapshot) ? snapshot : {};
  return `
    <div class="affect-readout" role="group" aria-label="Simulated affect snapshot">
      <p class="affect-title">Simulated affect snapshot</p>
      <dl class="affect-metrics" aria-label="Bounded affect values">
        <div><dt>Valence <span class="affect-range">−1 to 1</span></dt><dd>${boundedAffectValue(state.valence, [-1, 1])}</dd></div>
        <div><dt>Arousal <span class="affect-range">0 to 1</span></dt><dd>${boundedAffectValue(state.arousal, [0, 1])}</dd></div>
        <div><dt>Confidence <span class="affect-range">0 to 1</span></dt><dd>${boundedAffectValue(state.confidence, [0, 1])}</dd></div>
        <div><dt>Stress <span class="affect-range">0 to 1</span></dt><dd>${boundedAffectValue(state.stress, [0, 1])}</dd></div>
      </dl>
      <dl class="affect-details">
        <div><dt>Label</dt><dd>${affectText(state.label, "unavailable", 64)}</dd></div>
        <div><dt>Reason</dt><dd>${affectText(state.last_reason ?? state.reason, "unavailable", 160)}</dd></div>
      </dl>
      <p class="affect-disclosure">${affectText(state.disclosure, "Simulated algorithmic state, not consciousness.")}</p>
    </div>`;
};
const safeHttpUrl = (value) => {
  try {
    const url = new URL(String(value));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "#";
  } catch {
    return "#";
  }
};

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
const navButtons = document.querySelectorAll(".side-nav button");
navButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    navButtons.forEach(b => {
      const active = b === btn;
      b.classList.toggle("active", active);
      if (active) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
      $("view-" + b.dataset.view).classList.toggle("active", active);
    });
    if (btn.dataset.view === "mandala" && window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  });
});

/* ---------- overview ---------- */
let __bootTries = 0;
async function loadOverviewAffect() {
  const panel = $("affect-panel");
  const output = $("affect-output");
  setBusy(panel, true);
  setStatus(output, "Loading affect state…");
  try {
    output.innerHTML = affectMarkup(await api("/api/affect"));
  } catch (err) {
    setStatus(output, `Could not load affect state: ${err.message || err}`, "error");
  } finally {
    setBusy(panel, false);
  }
}
async function loadOverview() {
  loadOverviewAffect();
  try {
    const d = await api("/api/system");
    const w = d.workspace || {};
    const hardware = d.hardware || {};
    const runtime = d.runtime || {};
    $("stats").innerHTML = [
      ["Memory nodes", w.nodes ?? 0], ["Edges", w.edges ?? 0],
      ["Documents", w.documents ?? 0], ["Vectors", w.vectors ?? 0],
    ].map(([k, v]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`).join("");
    $("runtime").innerHTML = Object.entries(runtime)
      .map(([k, v]) => `<div><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join("");
    const notes = [];
    if (!(d.llm || {}).available) notes.push("No local LLM loaded — answers use the extractive fallback.");
    if ((d.embeddings || {}).backend !== "st") notes.push("Embeddings on hashing fallback — retrieval quality degraded.");
    (d.warnings || []).forEach(x => notes.push(esc(x)));
    $("notices").innerHTML = notes.length
      ? notes.map(n => `<li>${n}</li>`).join("")
      : "<li>All systems nominal.</li>";
    $("side-status").classList.remove("status-error", "status-success");
    $("side-status").innerHTML =
      `<strong>${esc(hardware.gpu_name || hardware.cpu_name || "?")}</strong><br>` +
      `tier ${esc(runtime.tier ?? hardware.tier ?? "?")} · ${esc(runtime.performance_mode || "")}<br>` +
      `${esc(w.nodes ?? 0)} nodes · ${esc(w.documents ?? 0)} docs`;
  } catch (e) {
    __bootTries++;
    setStatus($("side-status"), __bootTries <= 24
      ? `Warming up… loading embedding model and LLM (${__bootTries})`
      : "API offline — is the server running?", "error");
    if (__bootTries <= 24) setTimeout(loadOverview, 5000);
  }
}

/* ---------- chat ---------- */
const chatForm = $("chat-form");
const chatOutput = $("chat-out");
const chatSubmit = $("chat-submit");
revalidateWhenFixed($("chat-q"), $("chat-q-error"), "Enter a question.");
chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (chatSubmit.disabled) return;
  const input = $("chat-q");
  if (!validateRequired(input, $("chat-q-error"), "Enter a question.")) {
    input.focus();
    return;
  }
  const q = input.value.trim();
  const set = $("chat-set").value;
  chatOutput.insertAdjacentHTML("beforeend",
    `<div class="bubble"><strong>You</strong><p>${esc(q)}</p></div>`);
  input.value = "";
  const pending = document.createElement("div");
  pending.className = "bubble"; pending.textContent = "Retrieving and composing…";
  chatOutput.appendChild(pending);
  setBusy(chatForm, true); setBusy(chatOutput, true); setPending(chatSubmit, true, "Asking…");
  try {
    const d = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, components: set, allow_web: $("chat-web").checked }),
    });
    const m = d.metrics || {};
    const rows = (d.memories || []).slice(0, 6).map((x, i) =>
      `<tr><td>${i + 1}</td><th scope="row">${esc(x.concept)}</th><td>${esc(x.type)}</td>` +
      `<td>${esc(x.ring ?? "—")}</td><td>${fmt(x.score)}</td></tr>`).join("");
    const src = (d.sources || []).map(s => `<li>${esc(s)}</li>`).join("");
    const amode = d.agent_mode || "memory";
    const amodeTag = amode === "memory" ? "from memory"
      : amode === "web" ? "fetched live from the web"
      : amode === "identity" ? "project identity"
      : "model knowledge — ungrounded";
    const compression = Number(m.compression_ratio);
    const affect = affectMarkup(d.affect_snapshot);
    pending.innerHTML = `
      <strong>Answer <span class="muted small">(${esc(d.mode)} · ${esc(amodeTag)})</span></strong>
      <p>${esc(d.answer)}</p>
      <div class="meta"><span>${fmt(m.latency_ms, 0)} ms</span>
        <span>${esc(m.n_memories ?? "—")} memories</span>
        <span>${esc(m.context_tokens ?? "—")} ctx tokens</span>
        <span>compression ${Number.isFinite(compression) && compression !== 0 ? compression : "—"}×</span></div>
      ${affect}
      ${rows ? `<details><summary>Retrieved memories</summary><div class="table-wrap" role="region" aria-label="Retrieved memories" tabindex="0"><table><thead><tr><th scope="col">#</th><th scope="col">concept</th><th scope="col">type</th><th scope="col">ring</th><th scope="col">score</th></tr></thead><tbody>${rows}</tbody></table></div></details>` : ""}
      ${src ? `<details><summary>Sources</summary><ul class="ticks small">${src}</ul></details>` : ""}`;
  } catch (err) {
    pending.innerHTML = `<p class="small status-error" role="alert">Error: ${esc(err.message)}</p>`;
  } finally {
    setBusy(chatForm, false); setBusy(chatOutput, false); setPending(chatSubmit, false);
  }
});

/* ---------- live voice: speak the question, hear the answer ---------- */
(function initVoice() {
  const mic = $("chat-mic");
  const voicebar = $("chat-voicebar");
  const speakBtn = $("chat-speak");
  const stopBtn = $("chat-voice-stop");
  const voiceStatus = $("chat-voice-status");
  if (!mic || !window.MIRAVoice) return;

  const V = window.MIRAVoice;
  if (V.supported.stt) {
    mic.hidden = false;
    let finalText = "";
    mic.addEventListener("click", () => {
      if (V.isListening()) {
        V.stopListening();
        return;
      }
      finalText = "";
      mic.setAttribute("aria-pressed", "true");
      voicebar.hidden = false;
      voiceStatus.textContent = (window.MIRAI18N && MIRAI18N.t("chat.listening")) || "Listening…";
      const started = V.startListening({
        onResult: (text, isFinal) => {
          $("chat-q").value = text;
          if (isFinal) finalText = text;
        },
        onEnd: (err) => {
          mic.setAttribute("aria-pressed", "false");
          voiceStatus.textContent = err ? `mic: ${err}` : "";
          if (!err && finalText.trim()) chatForm.requestSubmit();
        },
      });
      if (!started) {
        mic.setAttribute("aria-pressed", "false");
        voiceStatus.textContent = (window.MIRAI18N && MIRAI18N.t("chat.mic.unsupported")) || "Voice input needs Chrome or Edge";
      }
    });
  }

  /* manual replay + stop; auto-speak stays off until the user opts in once */
  speakBtn.addEventListener("click", () => {
    const last = [...chatOutput.querySelectorAll(".bubble p")].reverse()
      .find(p => p.textContent.length > 40);
    if (last) V.speak(last.textContent);
  });
  stopBtn.addEventListener("click", () => {
    V.stopSpeaking();
    V.stopListening();
    voiceStatus.textContent = "";
  });
})();

/* ---------- reveal-on-scroll for landing sections ---------- */
(function initReveal() {
  const targets = document.querySelectorAll(".section, .honesty, .hero-inner");
  if (!targets.length || !("IntersectionObserver" in window) ||
      window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return;
  }
  targets.forEach(el => el.classList.add("reveal"));
  const io = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) {
        en.target.classList.add("in");
        io.unobserve(en.target);
      }
    });
  }, { threshold: 0.08 });
  targets.forEach(el => io.observe(el));
})();

/* ---------- documents ---------- */
const docList = $("doc-list");
async function loadDocs() {
  setBusy(docList, true);
  try {
    const d = await api("/api/documents");
    const documents = d.documents || [];
    docList.innerHTML = documents.map(x =>
      `<div class="doc-row"><span><span class="doc-title">${esc(x.title)}</span> <span class="muted small">· ${esc(x.n_chunks)} chunks · ${esc(x.file_type)}</span></span>
       <button class="del" type="button" data-id="${esc(x.id)}" aria-label="Delete ${esc(x.title)}">delete</button></div>`).join("")
      || `<p class="small muted">Nothing ingested yet.</p>`;
    docList.querySelectorAll(".del").forEach(button => {
      button.addEventListener("click", async () => {
        const title = button.closest(".doc-row").querySelector(".doc-title").textContent;
        if (!window.confirm(`Delete “${title}”? This removes the document and its memory nodes.`)) return;
        setPending(button, true, "Deleting…");
        setStatus($("doc-status"), `Deleting “${title}”…`);
        try {
          await api(`/api/documents/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
          setStatus($("doc-status"), `Deleted “${title}”.`, "success");
          await loadDocs();
        } catch (err) {
          setPending(button, false);
          button.textContent = "Retry delete";
          setStatus($("doc-status"), `Could not delete “${title}”: ${err.message || err}`, "error");
        }
      });
    });
  } catch (err) {
    docList.innerHTML = `<p class="small status-error" role="alert">Could not load documents: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(docList, false);
  }
}

const uploadForm = $("upload-form");
const uploadStatus = $("upload-status");
const uploadSubmit = $("upload-submit");
revalidateWhenFixed($("doc-file"), $("doc-file-error"), "Choose a document to ingest.", "change");
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (uploadSubmit.disabled) return;
  const file = $("doc-file");
  if (!validateRequired(file, $("doc-file-error"), "Choose a document to ingest.")) {
    file.focus();
    return;
  }
  const fd = new FormData();
  fd.append("file", file.files[0]);
  fd.append("title", $("doc-title").value.trim());
  setStatus(uploadStatus, "Ingesting… (chunking, embedding, placing on the mandala)");
  setBusy(uploadForm, true); setBusy(uploadStatus, true); setPending(uploadSubmit, true, "Ingesting…");
  try {
    const d = await api("/api/documents", { method: "POST", body: fd });
    setStatus(uploadStatus,
      `Ingested: ${d.n_chunks} chunks → ${d.n_nodes} nodes, ${d.n_edges} edges.`, "success");
    uploadForm.reset();
    $("doc-file").removeAttribute("aria-invalid");
    await loadDocs();
  } catch (err) {
    setStatus(uploadStatus, `Ingest failed: ${err.message || err}`, "error");
  } finally {
    setBusy(uploadForm, false); setBusy(uploadStatus, false); setPending(uploadSubmit, false);
  }
});

/* ---------- research lab ---------- */
const labForm = $("lab-form");
const labOutput = $("lab-out");
const labSubmit = $("lab-submit");
revalidateWhenFixed($("lab-q"), $("lab-q-error"), "Enter a research query.");
labForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (labSubmit.disabled) return;
  const input = $("lab-q");
  if (!validateRequired(input, $("lab-q-error"), "Enter a research query.")) {
    input.focus();
    return;
  }
  labOutput.innerHTML = `<p class="small muted">Running retrieval comparison…</p>`;
  setBusy(labForm, true); setBusy(labOutput, true); setPending(labSubmit, true, "Running…");
  try {
    const d = await api("/api/lab", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: input.value.trim() }),
    });
    let html = "";
    for (const [name, result] of Object.entries(d.results || {})) {
      html += `<h2 style="margin-top:1.4rem">${esc(name)} <span class="muted small">· ${fmt(result.latency_ms, 1)} ms · ${esc(result.n_candidates)} candidates</span></h2>`;
      html += `<div class="table-wrap" role="region" aria-label="${esc(name)} retrieval results" tabindex="0"><table class="lab-table"><thead><tr><th scope="col">#</th><th scope="col">concept</th><th scope="col">type</th><th scope="col">ring</th><th scope="col">score</th><th scope="col">hops</th></tr></thead><tbody>`;
      (result.items || []).slice(0, 8).forEach((item, i) => {
        html += `<tr><td>${i + 1}</td><th scope="row">${esc(item.concept)}</th><td>${esc(item.type)}</td><td>${esc(item.ring ?? "—")}</td><td>${fmt(item.score)}</td><td>${esc(item.hops)}</td></tr>`;
      });
      html += `</tbody></table></div>`;
    }
    html += `<h2 style="margin-top:1.4rem">Jaccard overlap</h2><pre class="formula">${esc(JSON.stringify(d.jaccard || {}, null, 1))}</pre>`;
    labOutput.innerHTML = html;
  } catch (err) {
    labOutput.innerHTML = `<p class="small status-error" role="alert">Research Lab failed: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(labForm, false); setBusy(labOutput, false); setPending(labSubmit, false);
  }
});

/* ---------- benchmarks ---------- */
const benchForm = $("bench-form");
const benchOutput = $("bench-out");
const benchSubmit = $("bench-submit");
benchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (benchSubmit.disabled) return;
  benchOutput.innerHTML = `<p class="small muted">Running benchmark… (this really runs every system)</p>`;
  setBusy(benchForm, true); setBusy(benchOutput, true); setPending(benchSubmit, true, "Running…");
  try {
    const d = await api("/api/benchmark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_path: $("bench-ds").value, format: $("bench-fmt").value,
        limit: +$("bench-n").value, judge: $("bench-judge").checked,
      }),
    });
    const cols = ["system", "retrieval_recall", "mrr", "answer_token_f1", "context_tokens", "latency_ms"];
    let html = `<p class="small muted">${esc(d.n_questions)} questions · experiment ${esc(d.experiment_id || "not saved")}</p>`;
    if (d.judge) html += `<p class="small">judge: correctness ${esc(d.judge.judge_correctness ?? "—")} / faithfulness ${esc(d.judge.judge_faithfulness ?? "—")} (${esc(d.judge.n_judged ?? 0)} judged)</p>`;
    html += `<div class="table-wrap" role="region" aria-label="Benchmark results" tabindex="0"><table class="bench-table"><thead><tr>${cols.map(c => `<th scope="col">${c}</th>`).join("")}</tr></thead><tbody>`;
    for (const row of d.table || []) {
      html += `<tr>${cols.map((c, i) => i === 0
        ? `<th scope="row">${esc(row[c] ?? "—")}</th>`
        : `<td>${esc(row[c] ?? "—")}</td>`).join("")}</tr>`;
    }
    html += `</tbody></table></div>`;
    benchOutput.innerHTML = html;
  } catch (err) {
    benchOutput.innerHTML = `<p class="small status-error" role="alert">Benchmark failed: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(benchForm, false); setBusy(benchOutput, false); setPending(benchSubmit, false);
  }
});

/* ---------- live web search ---------- */
const wsForm = $("ws-form");
const wsOutput = $("ws-out");
const wsSubmit = $("ws-submit");
revalidateWhenFixed($("ws-q"), $("ws-q-error"), "Enter a search query.");
wsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (wsSubmit.disabled) return;
  const input = $("ws-q");
  if (!validateRequired(input, $("ws-q-error"), "Enter a search query.")) {
    input.focus();
    return;
  }
  wsOutput.innerHTML = `<p class="small muted">Searching the web…</p>`;
  setBusy(wsForm, true); setBusy(wsOutput, true); setPending(wsSubmit, true, "Searching…");
  try {
    const d = await api("/api/websearch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: input.value.trim(), backend: $("ws-backend").value,
        allow_web: $("ws-consent").checked,
      }),
    });
    if (!d.results || !d.results.length) {
      wsOutput.innerHTML = `<p class="small status-error" role="alert">${esc(d.note || "No web results found.")}</p>`;
      return;
    }
    wsOutput.innerHTML =
      `<p class="small muted">backend: ${esc(d.backend)} · ${d.results.length} results</p>` +
      d.results.map((result, i) => {
        const title = result.title || result.url;
        const url = safeHttpUrl(result.url);
        return `
        <div class="doc-row">
          <span><a href="${esc(url)}" target="_blank" rel="noopener">${esc(title)}</a>
            <div class="muted small">${esc(String(result.snippet || "").slice(0, 200))}</div></span>
          <button class="btn btn-secondary ws-ingest" type="button" data-i="${i}" aria-label="Ingest ${esc(title)} into memory">Ingest into memory</button>
        </div>`;
      }).join("");
    const lastResults = d.results;
    wsOutput.querySelectorAll(".ws-ingest").forEach(button => {
      button.addEventListener("click", async () => {
        const result = lastResults[+button.dataset.i];
        button.classList.remove("status-error");
        button.setAttribute("aria-label", "Ingest this page into memory");
        setPending(button, true, "Fetching…"); setBusy(wsOutput, true);
        try {
          const ingested = await api("/api/websearch/ingest", {
            method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                url: result.url, title: result.title || result.url,
                allow_web: $("ws-consent").checked,
              }),

          });
          const stats = ingested.ingested || {};
          setPending(button, false);
          button.textContent = `Ingested: ${stats.n_chunks ?? "?"} chunks → ${stats.n_nodes ?? "?"} nodes`;
        } catch (err) {
          setPending(button, false);
          button.textContent = "Retry ingest";
          button.classList.add("status-error");
          button.setAttribute("aria-label", `Web page ingest failed: ${err.message || err}. Retry ingest.`);
        } finally {
          setBusy(wsOutput, false);
        }
      });
    });
  } catch (err) {
    wsOutput.innerHTML = `<p class="small status-error" role="alert">Web search failed: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(wsForm, false); setBusy(wsOutput, false); setPending(wsSubmit, false);
  }
});

/* ---------- placement strategy lab ---------- */
const stratForm = $("strat-form");
const stratOutput = $("strat-out");
const stratApply = $("strat-apply");
const stratSweep = $("strat-sweep");
async function initStrategyLab() {
  try {
    const d = await api("/api/strategies");
    $("strat-select").innerHTML = (d.strategies || []).map(strategy =>
      `<option value="${esc(strategy)}" ${strategy === d.current ? "selected" : ""}>${esc(strategy)}</option>`).join("");
  } catch (err) {
    $("strat-select").disabled = true;
    stratApply.disabled = true;
    stratOutput.innerHTML = `<p class="status-error" role="alert">Could not load placement strategies: ${esc(err.message || err)}</p>`;
  }
}
stratForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (stratApply.disabled) return;
  const strategy = $("strat-select").value;
  if (!strategy) return;
  stratOutput.innerHTML = `<p class="muted">Re-placing all memories with “${esc(strategy)}”…</p>`;
  setBusy(stratForm, true); setBusy(stratOutput, true); setPending(stratApply, true, "Applying…");
  stratSweep.disabled = true;
  try {
    const d = await api(`/api/strategies/${encodeURIComponent(strategy)}/apply`, { method: "POST" });
    const radial = d.info && d.info.radial;
    stratOutput.innerHTML = `<p class="status-success">Applied <strong>${esc(d.applied)}</strong>` +
      (radial ? ` — radial distances recomputed for ${esc(radial.computed)} nodes` : "") + `.</p>`;
    if (window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  } catch (err) {
    stratOutput.innerHTML = `<p class="status-error" role="alert">Could not apply strategy: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(stratForm, false); setBusy(stratOutput, false); setPending(stratApply, false);
    stratSweep.disabled = false;
  }
});
stratSweep.addEventListener("click", async () => {
  stratOutput.innerHTML = `<p class="muted">Benchmarking all strategies (place → measure → restore)…</p>`;
  setBusy(stratForm, true); setBusy(stratOutput, true); setPending(stratSweep, true, "Benchmarking…");
  stratApply.disabled = true;
  try {
    const d = await api("/api/strategies/sweep", { method: "POST" });
    let html = `<div class="table-wrap" role="region" aria-label="Placement strategy benchmark" tabindex="0"><table class="bench-table"><thead><tr><th scope="col">strategy</th><th scope="col">recall</th><th scope="col">mrr</th><th scope="col">latency ms</th></tr></thead><tbody>`;
    for (const row of d.table || []) {
      html += `<tr${row.system === d.best ? " style='color:var(--gold)'" : ""}>` +
        `<th scope="row">${esc(row.system)}</th><td>${esc(row.retrieval_recall ?? "—")}</td>` +
        `<td>${esc(row.mrr ?? "—")}</td><td>${fmt(row.latency_ms, 1)}</td></tr>`;
    }
    html += `</tbody></table></div><p class="muted small">best: ${esc(d.best)} (restored: ${esc(d.restored)})</p>`;
    stratOutput.innerHTML = html;
    if (window.__miraMandalaRefresh) window.__miraMandalaRefresh();
  } catch (err) {
    stratOutput.innerHTML = `<p class="status-error" role="alert">Strategy benchmark failed: ${esc(err.message || err)}</p>`;
  } finally {
    setBusy(stratForm, false); setBusy(stratOutput, false); setPending(stratSweep, false);
    stratApply.disabled = $("strat-select").disabled;
  }
});

loadOverview();
loadDocs();
initStrategyLab();

/* ---------- real-data research artifacts ---------- */
(async function loadRealResults() {
  const out = $("real-results-out");
  if (!out) return;
  try {
    const d = await api("/api/research/artifacts");
    const a = (d && d.artifacts) || {};
    let html = "";
    const bench = a.bench_real;
    if (bench && bench.summary) {
      const sig = (bench.significance && bench.significance.mrr) || {};
      const bt = sig.bootstrap || {};
      html += `<div class="table-wrap" role="region" aria-label="Real benchmark results" tabindex="0"><table class="bench-table"><thead><tr>` +
        `<th scope="col">system</th><th scope="col">MRR</th><th scope="col">recall@8</th><th scope="col">latency ms</th></tr></thead><tbody>`;
      for (const [name, s] of Object.entries(bench.summary)) {
        html += `<tr><th scope="row">${esc(name)}</th><td>${fmt(s.mrr)}</td>` +
          `<td>${fmt(s.retrieval_recall)}</td><td>${fmt(s.latency_ms, 1)}</td></tr>`;
      }
      html += `</tbody></table></div>`;
      if (typeof bt.mean_diff === "number" && bt.n_pairs > 0) {
        html += `<p class="small muted">mira vs flat_vector MRR: Δ=${esc(bt.mean_diff)} ` +
          `95% CI [${esc(bt.ci_low)}, ${esc(bt.ci_high)}], p≈${esc(bt.p_value)} ` +
          `(n=${esc(bt.n_pairs)} paired questions, 3 seeds)</p>`;
      }
    } else {
      html += `<p class="small muted">Full real-data benchmark: not yet run on this machine. Run <code>scripts/eval_benchmark_real.py</code>.</p>`;
    }
    const abl = a.ablation_real;
    if (abl && abl.table && abl.table.length) {
      html += `<h3 style="margin-top:1rem">Component ablation (leave-one-out)</h3>` +
        `<div class="table-wrap" role="region" aria-label="Ablation results" tabindex="0"><table class="bench-table"><thead><tr>` +
        `<th scope="col">condition</th><th scope="col">MRR</th><th scope="col">recall@8</th><th scope="col">Δ vs full</th></tr></thead><tbody>`;
      for (const row of abl.table) {
        html += `<tr><th scope="row">${esc(row.condition)}</th><td>${fmt(row.mrr)}</td>` +
          `<td>${fmt(row["recall@8"])}</td><td>${row.mrr_delta_vs_full != null ? esc(row.mrr_delta_vs_full) : "—"}</td></tr>`;
      }
      html += `</tbody></table></div>`;
    }
    const nv = a.neural_validation;
    if (nv) {
      html += `<p class="small muted">Learned scorer (document-grouped holdout): hand MRR ${esc(nv.holdout_mrr_hand)} → learned ${esc(nv.holdout_mrr_learned)} — ` +
        `<strong>${nv.verdict_enable ? "enabled" : "kept disabled"}</strong> per the significance rule.</p>`;
    }
    if (!html) html = `<p class="small muted">No artifacts yet — run the eval scripts.</p>`;
    out.innerHTML = html;
  } catch (err) {
    out.innerHTML = `<p class="small status-error" role="alert">Could not load research artifacts: ${esc(err.message || err)}</p>`;
  }
})();
