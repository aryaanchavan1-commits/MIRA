/* MIRA — ChatGPT-style conversation layer.
 * Adds on top of the existing console.js chat: persistent sessions
 * (localStorage), a welcome state with suggestion prompts, a typing
 * indicator, auto-scroll, copy buttons, and textarea auto-grow.
 * The transport still posts to /api/chat with the SAME payload shape, so
 * server behavior (grounding, provenance, affect) is unchanged.
 */
"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>\"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  /* ---------- minimal safe markdown: **bold**, *em*, `code`, lists,
     line breaks. No raw HTML ever passes through — esc() first. ---------- */
  function md(text) {
    let s = esc(text ?? "");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|\s)\*([^*\n]+)\*/g, "$1<em>$2</em>");
    s = s.replace(/^- (.+)$/gm, "<li>$1</li>");
    s = s.replace(/(<li>[\s\S]*?<\/li>)/g, "<ul>$1</ul>")
         .replace(/<\/ul>\s*<ul>/g, "");
    s = s.replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>");
    return `<p>${s}</p>`;
  }

  /* ---------- session store ---------- */
  const STORE_KEY = "mira_chat_sessions_v1";
  const sessions = { list: [], activeId: null };
  let restoreTarget = null;

  function load() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (raw && Array.isArray(raw.list)) {
        sessions.list = raw.list.slice(-30);
        sessions.activeId = raw.activeId;
      }
    } catch (e) { /* private mode / corrupt */ }
  }
  function persist() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(
        { list: sessions.list.slice(-30), activeId: sessions.activeId }));
    } catch (e) { /* quota — ignore */ }
  }
  function active() {
    return sessions.list.find(s => s.id === sessions.activeId) || null;
  }
  function newSession(activate = true) {
    const s = { id: "s" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
                title: "New chat", messages: [], created: Date.now() };
    sessions.list.push(s);
    if (activate) sessions.activeId = s.id;
    persist();
    return s;
  }
  function deleteSession(id) {
    sessions.list = sessions.list.filter(s => s.id !== id);
    if (sessions.activeId === id) sessions.activeId = sessions.list.length ? sessions.list[sessions.list.length - 1].id : null;
    persist();
  }

  /* ---------- DOM ---------- */
  let threadEl, composerEl, inputEl, sendBtn, webToggle, setSelect;
  let micBtn, voicebar, speakBtn, stopBtn, voiceStatus, sessionListEl, newChatBtn;

  function busy(b) {
    if (sendBtn) sendBtn.disabled = b;
    if (composerEl) composerEl.setAttribute("aria-busy", String(b));
  }

  function scrollEnd() {
    requestAnimationFrame(() => { threadEl.scrollTop = threadEl.scrollHeight; });
  }

  function welcomeHTML() {
    const suggestions = [
      "What is the mandala memory architecture?",
      "Summarize what my documents say about Hebbian learning",
      "How does spreading activation retrieve multi-hop evidence?",
      "What did the MuSiQue benchmark measure?",
    ];
    return `
      <div class="chat-welcome">
        <div class="chat-welcome-mark">◎</div>
        <h2>What should we explore?</h2>
        <p class="muted">Answers are grounded in your mandala memory — every claim carries its source.</p>
        <div class="chat-suggest">
          ${suggestions.map(s => `<button type="button" class="chat-suggest-btn">${esc(s)}</button>`).join("")}
        </div>
      </div>`;
  }

  function userMsgHTML(text) {
    return `<div class="chat-msg user" role="group" aria-label="Your question">
      <div class="chat-avatar user">You</div>
      <div class="chat-bubble user"><p>${esc(text)}</p></div>
    </div>`;
  }

  function metaHTML(m, d) {
    const compression = Number(m.compression_ratio);
    const amode = d.agent_mode || "memory";
    const amodeTag = amode === "memory" ? "from memory"
      : amode === "web" ? "fetched live from the web"
      : amode === "identity" ? "project identity"
      : "model knowledge — ungrounded";
    const remembered = d.remembered;
    const savedChip = remembered && remembered.document_id
      ? `<span class="chat-saved" title="Question and answer stored as mandala memory nodes">✦ saved to memory</span>`
      : "";
    return `<div class="chat-meta">
      <span class="chat-mode ${amode === "memory" ? "grounded" : ""}">${esc(amodeTag)}</span>
      <span>${Number.isFinite(m.latency_ms) ? Math.round(m.latency_ms) : "—"} ms</span>
      <span>${esc(m.n_memories ?? "—")} memories</span>
      <span>${esc(m.context_tokens ?? "—")} ctx</span>
      ${Number.isFinite(compression) && compression !== 0 ? `<span>${compression}× compressed</span>` : ""}
      ${savedChip}
    </div>`;
  }

  function assistantMsgHTML(d) {
    const m = d.metrics || {};
    const rows = (d.memories || []).slice(0, 6).map((x, i) =>
      `<tr><td>${i + 1}</td><th scope="row">${esc(x.concept)}</th><td>${esc(x.type)}</td>` +
      `<td>${esc(x.ring ?? "—")}</td><td>${Number(x.score).toFixed(3)}</td></tr>`).join("");
    const src = (d.sources || []).map(s => `<li>${esc(s)}</li>`).join("");
    const affect = (window.__affectMarkup || function(){return "";})(d.affect_snapshot);
    return `<div class="chat-msg assistant">
      <div class="chat-avatar assistant">◎</div>
      <div class="chat-bubble assistant">
        ${md(d.answer)}
        ${metaHTML(m, d)}
        ${affect}
        ${rows ? `<details><summary>Retrieved memories</summary><div class="table-wrap" tabindex="0"><table><thead><tr><th>#</th><th>concept</th><th>type</th><th>ring</th><th>score</th></tr></thead><tbody>${rows}</tbody></table></div></details>` : ""}
        ${src ? `<details><summary>Sources</summary><ul class="ticks small">${src}</ul></details>` : ""}
      </div>
    </div>`;
  }

  function errorHTML(message) {
    return `<div class="chat-msg assistant"><div class="chat-avatar assistant">◎</div>
      <div class="chat-bubble assistant"><p class="status-error" role="alert">Error: ${esc(message)}</p></div></div>`;
  }

  function renderSession() {
    const s = active();
    if (!s || !s.messages.length) {
      threadEl.innerHTML = welcomeHTML();
      bindSuggestions();
      return;
    }
    threadEl.innerHTML = s.messages.map(m =>
      m.role === "user" ? userMsgHTML(m.content)
        : m.role === "error" ? errorHTML(m.content)
        : assistantMsgHTML(m.data || { answer: m.content, metrics: {}, memories: [], sources: [] })
    ).join("");
    bindCopyButtons();
    scrollEnd();
  }

  function bindSuggestions() {
    threadEl.querySelectorAll(".chat-suggest-btn").forEach(btn => {
      btn.addEventListener("click", () => { inputEl.value = btn.textContent; send(); });
    });
  }

  function bindCopyButtons() {
    threadEl.querySelectorAll(".chat-copy").forEach(btn => {
      btn.addEventListener("click", () => {
        const text = btn.closest(".chat-bubble").querySelector("p");
        if (text && navigator.clipboard) navigator.clipboard.writeText(text.textContent).catch(() => {});
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = "Copy"; }, 1400);
      });
    });
  }

  function renderSessionList() {
    if (!sessionListEl) return;
    const items = [...sessions.list].reverse().map(s => `
      <div class="chat-session ${s.id === sessions.activeId ? "active" : ""}" data-id="${esc(s.id)}">
        <button type="button" class="chat-session-open" title="${esc(s.title)}">${esc(s.title.slice(0, 34))}</button>
        <button type="button" class="chat-session-del" aria-label="Delete conversation">×</button>
      </div>`).join("");
    sessionListEl.innerHTML = items || `<p class="small muted" style="padding:0 0.6rem">No conversations yet.</p>`;
    sessionListEl.querySelectorAll(".chat-session").forEach(row => {
      const id = row.dataset.id;
      row.querySelector(".chat-session-open").addEventListener("click", () => {
        sessions.activeId = id; persist(); renderSessionList(); renderSession();
      });
      row.querySelector(".chat-session-del").addEventListener("click", (e) => {
        e.stopPropagation();
        deleteSession(id); renderSessionList(); renderSession();
      });
    });
  }

  function updateTitle(s, text) {
    if (s && s.title === "New chat" && text) {
      s.title = text.slice(0, 40) + (text.length > 40 ? "…" : "");
      persist(); renderSessionList();
    }
  }

  let rememberToggle;

  async function send() {
    const text = (inputEl.value || "").trim();
    if (!text) return;
    let s = active();
    if (!s) { s = newSession(); renderSessionList(); }
    if (s.messages.length === 0) updateTitle(s, text);
    s.messages.push({ role: "user", content: text });

    // optimistic render: user bubble + typing indicator
    if (s.messages.length === 1) threadEl.innerHTML = "";
    threadEl.insertAdjacentHTML("beforeend", userMsgHTML(text));
    const typing = document.createElement("div");
    typing.className = "chat-msg assistant";
    typing.innerHTML = `<div class="chat-avatar assistant">◎</div>
      <div class="chat-bubble assistant"><span class="chat-typing"><i></i><i></i><i></i></span>
      <span class="muted small" style="margin-left:.6rem">Retrieving from the mandala…</span></div>`;
    threadEl.appendChild(typing);
    inputEl.value = "";
    autoGrow();
    busy(true);
    scrollEnd();

    const payload = {
      question: text,
      components: setSelect ? setSelect.value : "all",
      allow_web: webToggle ? webToggle.checked : false,
      remember: rememberToggle ? rememberToggle.checked : true,
    };
    try {
      const resp = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const raw = await resp.text();
      let d = null;
      try { d = raw ? JSON.parse(raw) : null; } catch (e) { /* non-JSON */ }
      if (!resp.ok) {
        const detail = d && (d.detail ?? d.error);
        throw new Error(detail ? String(detail) : `${resp.status} ${resp.statusText}`);
      }
      s.messages.push({ role: "assistant", data: d, content: d.answer || "" });
      typing.remove();
      threadEl.insertAdjacentHTML("beforeend", assistantMsgHTML(d));
      bindCopyButtons();
      scrollEnd();
    } catch (err) {
      s.messages.push({ role: "error", content: err.message || String(err) });
      typing.remove();
      threadEl.insertAdjacentHTML("beforeend", errorHTML(err.message || err));
      scrollEnd();
    } finally {
      persist();
      busy(false);
      inputEl.focus();
    }
  }

  function autoGrow() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 180) + "px";
  }

  function init() {
    threadEl = $("chat-thread");
    composerEl = $("chat-composer");
    inputEl = $("chat-input");
    sendBtn = $("chat-send");
    webToggle = $("chat-web");
    setSelect = $("chat-set");
    rememberToggle = $("chat-remember");
    micBtn = $("chat-mic");
    voicebar = $("chat-voicebar");
    speakBtn = $("chat-speak");
    stopBtn = $("chat-voice-stop");
    voiceStatus = $("chat-voice-status");
    sessionListEl = $("chat-sessions");
    newChatBtn = $("chat-new");
    if (!threadEl || !inputEl || !sendBtn || threadEl.dataset.chatInit) return;
    threadEl.dataset.chatInit = "1";

    load();
    if (!active()) newSession();
    renderSessionList();
    renderSession();

    sendBtn.addEventListener("click", send);
    inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
    });
    inputEl.addEventListener("input", autoGrow);

    if (newChatBtn) newChatBtn.addEventListener("click", () => {
      newSession(); renderSessionList(); renderSession();
      inputEl.focus();
    });

    /* Expose a hook so console.js voice mic can push text into the composer */
    window.__miraChat = {
      setInput(text) { inputEl.value = text; autoGrow(); inputEl.focus(); },
      send,
    };

    /* Wire the existing voice mic to the composer (same IDs console.js used) */
    if (micBtn && window.MIRAVoice) {
      const V = window.MIRAVoice;
      if (V.supported.stt) {
        micBtn.hidden = false;
        let finalText = "";
        micBtn.addEventListener("click", () => {
          if (V.isListening()) { V.stopListening(); return; }
          finalText = "";
          micBtn.setAttribute("aria-pressed", "true");
          if (voicebar) voicebar.hidden = false;
          if (voiceStatus) voiceStatus.textContent =
            (window.MIRAI18N && MIRAI18N.t("chat.listening")) || "Listening…";
          const started = V.startListening({
            onResult: (t) => { if (window.__miraChat) window.__miraChat.setInput(t); if (t) finalText = t; },
            onEnd: (err) => {
              micBtn.setAttribute("aria-pressed", "false");
              if (voiceStatus) voiceStatus.textContent = err ? `mic: ${err}` : "";
              if (!err && finalText.trim()) send();
            },
          });
          if (!started) {
            micBtn.setAttribute("aria-pressed", "false");
            if (voiceStatus) voiceStatus.textContent =
              (window.MIRAI18N && MIRAI18N.t("chat.mic.unsupported")) || "Voice input needs Chrome or Edge";
          }
        });
      }
      if (speakBtn) speakBtn.addEventListener("click", () => {
        const last = [...threadEl.querySelectorAll(".chat-bubble.assistant p")].reverse()
          .find(p => p.textContent.length > 40);
        if (last) V.speak(last.textContent);
      });
      if (stopBtn) stopBtn.addEventListener("click", () => {
        V.stopSpeaking(); V.stopListening();
        if (voiceStatus) voiceStatus.textContent = "";
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
