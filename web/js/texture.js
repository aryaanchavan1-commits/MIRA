/* MIRA — living texture engine.
 * A cursor-reactive radial lattice: breathing rings, pointer gravity,
 * ink-shimmer particles. Premium dark, minimal, never loud.
 * Respects prefers-reduced-motion (renders one static frame).
 */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function Texture(canvas, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.density = opts.density || 1;          // particle multiplier
    this.ringCount = opts.rings || 5;
    this.pointer = { x: 0.5, y: 0.5, tx: 0.5, ty: 0.5, active: false };
    this.nodes = [];
    this.t = 0;
    this._fit = this._fit.bind(this);
    this._frame = this._frame.bind(this);
    window.addEventListener("resize", this._fit);
    window.addEventListener("pointermove", this, { passive: true });
    this._fit();
    if (!reduceMotion) requestAnimationFrame(this._frame);
    else this._draw(0);
  }

  Texture.prototype.handleEvent = function (e) {
    this.pointer.tx = e.clientX / Math.max(1, window.innerWidth);
    this.pointer.ty = e.clientY / Math.max(1, window.innerHeight);
    this.pointer.active = true;
  };

  Texture.prototype._fit = function () {
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    var r = this.canvas.getBoundingClientRect();
    this.w = Math.max(320, r.width);
    this.h = Math.max(320, r.height);
    this.canvas.width = this.w * dpr;
    this.canvas.height = this.h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._seed();
    if (reduceMotion) this._draw(0);
  };

  /* Deterministic pseudo-random so the lattice is stable across frames */
  Texture.prototype._rand = function (i) {
    var x = Math.sin(i * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
  };

  Texture.prototype._seed = function () {
    this.nodes = [];
    var area = this.w * this.h;
    var count = Math.round(Math.min(140, Math.max(46, area / 16000)) * this.density);
    for (var i = 0; i < count; i++) {
      this.nodes.push({
        x: this._rand(i) * this.w,
        y: this._rand(i + 7919) * this.h,
        r: 0.6 + this._rand(i + 104729) * 1.5,
        ph: this._rand(i + 15485) * Math.PI * 2,
        sp: 0.2 + this._rand(i + 2003) * 0.5,
        drift: 6 + this._rand(i + 3571) * 14
      });
    }
  };

  Texture.prototype._frame = function (ts) {
    this.t = ts / 1000;
    /* pointer easing: the lattice leans toward the cursor, never jumps */
    this.pointer.x += (this.pointer.tx - this.pointer.x) * 0.045;
    this.pointer.y += (this.pointer.ty - this.pointer.y) * 0.045;
    this._draw(this.t);
    requestAnimationFrame(this._frame);
  };

  Texture.prototype._draw = function (t) {
    var ctx = this.ctx;
    var w = this.w, h = this.h;
    var cx = w * (0.5 + (this.pointer.x - 0.5) * 0.06);
    var cy = h * (0.42 + (this.pointer.y - 0.5) * 0.06);
    ctx.clearRect(0, 0, w, h);

    /* breathing mandala rings */
    var R = Math.min(w, h) * 0.62;
    for (var ring = 0; ring < this.ringCount; ring++) {
      var f = ring / this.ringCount;
      var rad = R * (0.25 + 0.75 * f) * (1 + Math.sin(t * 0.35 + ring * 0.9) * 0.015);
      var px = cx + (this.pointer.x - 0.5) * 26 * (1 - f);
      var py = cy + (this.pointer.y - 0.5) * 26 * (1 - f);
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(196,168,110," + (0.05 + 0.045 * (1 - f)).toFixed(3) + ")";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    /* spokes — faint, slightly rotating */
    var spokes = 24;
    var rot = t * 0.02;
    ctx.strokeStyle = "rgba(160,150,130,0.035)";
    for (var s = 0; s < spokes; s++) {
      var a = rot + (s / spokes) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * R * 1.15, cy + Math.sin(a) * R * 1.15);
      ctx.stroke();
    }

    /* ink-shimmer particles with pointer gravity */
    for (var i = 0; i < this.nodes.length; i++) {
      var p = this.nodes[i];
      var dx = this.pointer.x * w - p.x;
      var dy = this.pointer.y * h - p.y;
      var dist = Math.sqrt(dx * dx + dy * dy) || 1;
      var pull = Math.min(26, 2600 / dist);
      var ox = (dx / dist) * pull * 0.35;
      var oy = (dy / dist) * pull * 0.35;
      var wob = Math.sin(t * p.sp + p.ph) * p.drift * 0.12;
      var x = p.x + ox + wob;
      var y = p.y + oy + Math.cos(t * p.sp * 0.8 + p.ph) * p.drift * 0.1;
      var tw = 0.35 + 0.3 * Math.sin(t * 0.9 + p.ph * 2);
      ctx.beginPath();
      ctx.arc(x, y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(212,196,160," + (0.16 * tw + 0.05).toFixed(3) + ")";
      ctx.fill();
      /* connect near particles — constellation lines */
      if (i % 2 === 0) {
        var q = this.nodes[(i + 7) % this.nodes.length];
        var ddx = q.x - p.x, ddy = q.y - p.y;
        var dd = Math.sqrt(ddx * ddx + ddy * ddy);
        if (dd < 150) {
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(q.x + (this.pointer.x * w - q.x) * 0.1,
                     q.y + (this.pointer.y * h - q.y) * 0.1);
          ctx.strokeStyle = "rgba(190,175,140," + (0.05 * (1 - dd / 150)).toFixed(3) + ")";
          ctx.stroke();
        }
      }
    }
  };

  window.MIRATexture = Texture;

  function boot() {
    var hero = document.getElementById("bg-texture");
    if (hero) new Texture(hero, { rings: 6, density: 1 });
    var consoleBg = document.getElementById("bg-texture-console");
    if (consoleBg) new Texture(consoleBg, { rings: 4, density: 0.55 });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
