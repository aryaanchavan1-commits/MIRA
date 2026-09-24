/* MIRA canvas mandala — real polar layout from live data.
   Angle = sector slice, radius = ring. Nodes are clickable (console). */
"use strict";

const MIRA_TYPE_COLORS = {
  semantic: "#7a8cf0", episodic: "#e0a44a", procedural: "#58b98f",
  working: "#e06a5a", fact: "#5a9be0", entity: "#b07ae0",
  event: "#e08a4a", document: "#8a94a6", concept: "#4ab8a4",
  relation: "#d06ad0",
};

class MiraMandala {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = opts;
    this.nodes = [];
    this.edges = [];
    this.angleOf = new Map();
    this.radiusOf = new Map();
    this.rotation = opts.spin ? 0 : null;
    this.hover = null;
    this.onSelect = opts.onSelect || null;
    if (opts.interactive) {
      canvas.addEventListener("mousemove", (e) => this._move(e));
      canvas.addEventListener("click", (e) => this._click(e));
      canvas.addEventListener("mouseleave", () => { this.hover = null; this.draw(); });
    }
    window.addEventListener("resize", () => this._fit());
  }

  setData(nodes, edges, maxRings = 5) {
    this.nodes = nodes;
    this.edges = edges;
    const sectors = new Map();
    for (const n of nodes) {
      const s = n.sector || "unassigned";
      if (!sectors.has(s)) sectors.set(s, []);
      sectors.get(s).push(n);
    }
    const names = [...sectors.keys()].sort();
    const slice = (2 * Math.PI) / Math.max(1, names.length);
    const perRing = new Map(); // (sector|ring) counter for spread
    names.forEach((s, si) => {
      for (const n of sectors.get(s)) {
        const ring = Math.max(0, n.ring || 0);
        const key = s + "|" + ring;
        const i = perRing.get(key) || 0;
        perRing.set(key, i + 1);
        const inRing = sectors.get(s).filter(m => (m.ring || 0) === ring).length;
        const base = si * slice;
        const step = slice / Math.max(1, inRing + 1);
        this.angleOf.set(n.id, base + step * (i + 1));
        // small radial jitter separates stacked nodes within a ring
        const jit = ((i % 3) - 1) * 0.12;
        this.radiusOf.set(n.id, ring + 0.5 + jit);
      }
    });
    this.sectorNames = names;
    this.maxRings = maxRings;
    this._fit();
    this.draw();
  }

  _fit() {
    const r = this.canvas.getBoundingClientRect();
    if (r.width === 0) return;
    this.canvas.width = r.width * devicePixelRatio;
    this.canvas.height = r.height * devicePixelRatio;
    this.ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    if (this.nodes.length) this.draw();
  }

  _polar(n) {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const cx = w / 2, cy = h / 2;
    const maxR = Math.min(w, h) / 2 - 46;
    const ringW = maxR / Math.max(1, this.maxRings);
    const a = this.angleOf.get(n.id) + (this.rotation || 0);
    const rad = this.radiusOf.get(n.id) * ringW;
    return { x: cx + rad * Math.cos(a), y: cy + rad * Math.sin(a), cx, cy, maxR, ringW };
  }

  draw() {
    const ctx = this.ctx;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (!this.nodes.length) {
      ctx.fillStyle = "#a39c8d";
      ctx.font = "13px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("No memories yet — ingest a document first.", w / 2, h / 2);
      return;
    }
    const { cx, cy, maxR, ringW } = this._polar(this.nodes[0] || { id: "" });

    // ring guides + labels
    ctx.strokeStyle = "rgba(160,150,130,0.14)";
    ctx.fillStyle = "rgba(163,156,141,0.6)";
    ctx.font = "10px ui-monospace";
    for (let r = 0; r <= this.maxRings; r++) {
      ctx.beginPath();
      ctx.arc(cx, cy, r * ringW, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.textAlign = "left";
      ctx.fillText("R" + r, cx + 4, cy - r * ringW - 4);
    }
    // sector labels on the rim
    if (this.sectorNames) {
      ctx.textAlign = "center";
      const slice = (2 * Math.PI) / this.sectorNames.length;
      this.sectorNames.forEach((s, i) => {
        const a = i * slice + slice / 2 + (this.rotation || 0);
        const x = cx + (maxR + 24) * Math.cos(a);
        const y = cy + (maxR + 24) * Math.sin(a);
        ctx.fillStyle = "rgba(212,162,78,0.75)";
        ctx.fillText(String(s).slice(0, 14), x, y);
      });
    }
    // edges — Hebbian-strengthened edges (weight > 1) glow warm gold;
    // plain edges stay faint (w is the live consolidated weight)
    const pos = new Map(this.nodes.map(n => [n.id, this._polar(n)]));
    for (const e of this.edges) {
      const a = pos.get(e.s), b = pos.get(e.t);
      if (!a || !b) continue;
      const w = e.w ?? 1.0;
      const strengthened = Math.max(0, w - 1.0);   // > 0 after Hebbian hits
      if (strengthened > 0.05) {
        ctx.strokeStyle = `rgba(212,162,78,${Math.min(0.55, 0.25 + strengthened * 0.3)})`;
        ctx.lineWidth = Math.min(3.5, 1.2 + strengthened * 2.2);
        ctx.shadowColor = "rgba(212,162,78,0.8)";
        ctx.shadowBlur = 6;
      } else {
        ctx.strokeStyle = "rgba(160,150,130,0.10)";
        ctx.lineWidth = 1;
        ctx.shadowBlur = 0;
      }
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.shadowBlur = 0;
    // nodes (draw outer rings first) — radial proximity adds a core-ward
    // aura: the closer to the mandala's heart (radial_distance → 0), the
    // stronger the glow. Size = importance, color = memory type.
    const sorted = [...this.nodes].sort((a, b) => (b.ring || 0) - (a.ring || 0));
    for (const n of sorted) {
      const p = pos.get(n.id);
      const size = 3 + 7 * (n.importance ?? 0.5) + (n.ring === 0 ? 3 : 0);
      const color = MIRA_TYPE_COLORS[n.type] || "#8a94a6";
      const rad = n.radial;
      if (rad != null && rad < 0.6) {
        const aura = (0.6 - rad) / 0.6;      // 0..1
        ctx.globalAlpha = 1;
        ctx.fillStyle = `rgba(212,162,78,${0.10 + aura * 0.30})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, size + 4 + aura * 7, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.globalAlpha = this.hover === n.id ? 1 : 0.88;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(p.x, p.y, size, 0, 2 * Math.PI); ctx.fill();
      if (this.hover === n.id) {
        ctx.strokeStyle = "#d4a24e"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(p.x, p.y, size + 4, 0, 2 * Math.PI); ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
  }

  _nearest(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bd = 14;
    for (const n of this.nodes) {
      const p = this._polar(n);
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bd) { bd = d; best = n; }
    }
    return best;
  }
  _move(e) {
    const n = this._nearest(e);
    this.hover = n ? n.id : null;
    this.canvas.style.cursor = n ? "pointer" : "default";
    this.draw();
  }
  _click(e) {
    const n = this._nearest(e);
    if (n && this.onSelect) this.onSelect(n);
  }
}

/* ---------- hero brand animation ---------- */
function initHeroMandala() {
  const canvas = document.getElementById("hero-mandala");
  if (!canvas) return;
  const m = new MiraMandala(canvas, { interactive: false, spin: true });
  fetch("/api/mandala").then(r => r.json()).then(d => {
    m.setData(d.nodes, d.edges, d.max_rings || 5);
  }).catch(() => m.setData([], [], 5));
  (function loop() {
    if (!document.body.contains(canvas)) return;
    m.rotation += 0.0006;
    m.draw();
    requestAnimationFrame(loop);
  })();
}

/* ---------- console mandala ---------- */
function initConsoleMandala() {
  const canvas = document.getElementById("mandala-canvas");
  const panel = document.getElementById("node-panel");
  const legend = document.getElementById("legend");
  if (!canvas || canvas.dataset.init) return;
  canvas.dataset.init = "1";
  const m = new MiraMandala(canvas, { interactive: true, onSelect: select });

  async function refresh() {
    try {
      const d = await (await fetch("/api/mandala")).json();
      m.setData(d.nodes, d.edges, d.max_rings || 5);
      legend.innerHTML = Object.entries(MIRA_TYPE_COLORS)
        .filter(([t]) => d.nodes.some(n => n.type === t))
        .map(([t, c]) => `<span><span class="dot" style="background:${c}"></span>${t}</span>`)
        .join("");
    } catch (e) { panel.innerHTML = `<p class="small muted">API offline.</p>`; }
  }
  window.__miraMandalaRefresh = refresh;

  async function select(n) {
    panel.innerHTML = `<p class="small muted">Loading…</p>`;
    const d = await (await fetch("/api/node/" + n.id)).json();
    const tags = [
      d.type, "ring " + (d.ring ?? "—"), d.sector || "—",
      "conf " + (+d.confidence).toFixed(2), "imp " + (+d.importance).toFixed(2),
    ].map(t => `<span class="tag">${t}</span>`).join("");
    const prov = (d.provenance || []).map(p =>
      `<li>${p.document} p.${p.page ?? "—"} — <code>${p.chunk_id}</code></li>`).join("");
    const nb = (d.neighbors || []).slice(0, 8).map(x =>
      `<li><a href="#" data-nid="${x.id}">${x.concept}</a> · ${x.relation}</li>`).join("");
    panel.innerHTML = `
      <h4>${d.concept}</h4>${tags}
      ${d.summary ? `<p class="small" style="margin-top:.6rem">${d.summary}</p>` : ""}
      ${d.raw_text ? `<details><summary>raw text</summary><p class="small muted">${d.raw_text}</p></details>` : ""}
      ${prov ? `<p class="small" style="margin-top:.6rem"><strong>Provenance</strong></p><ul class="ticks small">${prov}</ul>` : ""}
      ${nb ? `<p class="small" style="margin-top:.6rem"><strong>Neighbors</strong></p><ul class="ticks small">${nb}</ul>` : ""}`;
    panel.querySelectorAll("[data-nid]").forEach(a =>
      a.addEventListener("click", (e) => {
        e.preventDefault();
        select({ id: a.dataset.nid });
      }));
  }
  refresh();
}

initHeroMandala();
