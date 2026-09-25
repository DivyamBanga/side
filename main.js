(() => {
  'use strict';

  const $ = s => document.querySelector(s);
  const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const DEG = Math.PI / 180;

  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fine = matchMedia('(hover: hover) and (pointer: fine)').matches;

  // Print settings. Units are mask pixels (text rasterised at FONT_PX).
  const FONT_PX = 180, FONT = `800 ${FONT_PX}px Archivo`;
  const BEAD = 3.4;          // extruded strand width
  const GAP = 8;             // lattice pitch
  const LAYERS = 6;
  const LAYER_H = 5;         // visual layer height
  const LAYER_MM = 0.32;     // what the readout claims
  const LAYER_TIME = [1.5, .9, .62, .5, .44, .44];
  const TILT = 34 * DEG;     // resting camera tilt
  const INK = '23,21,15', PINK = '#d1406a', UV = '98,70,255';

  const cv = $('#print'), ctx = cv.getContext('2d');
  const fx = $('#fx'), fctx = fx.getContext('2d');
  const stage = $('#stage'), body = document.body;
  const hudL = $('#hud-l'), hudR = $('#hud-r');

  let vw = 0, vh = 0, dpr = 1, M = null, split = false;

  /* ---------- slicing: text -> walls + lattice toolpaths ---------- */

  function build(lines) {
    const m = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
    m.font = FONT;
    const met = lines.map(t => m.measureText(t));
    const pad = 22, gap = FONT_PX * .12;
    const asc = Math.max(...met.map(x => x.actualBoundingBoxAscent));
    const desc = Math.max(...met.map(x => x.actualBoundingBoxDescent));
    const lineH = asc + desc + gap;
    const wOf = x => x.actualBoundingBoxLeft + x.actualBoundingBoxRight;
    const W = Math.ceil(Math.max(...met.map(wOf)) + pad * 2);
    const H = Math.ceil(lineH * lines.length - gap + pad * 2);
    m.canvas.width = W; m.canvas.height = H;
    m.font = FONT; m.fillStyle = '#000';
    lines.forEach((t, i) => m.fillText(t, (W - wOf(met[i])) / 2 + met[i].actualBoundingBoxLeft, pad + asc + i * lineH));

    const px = m.getImageData(0, 0, W, H).data;
    const a = new Float32Array(W * H), b = new Uint8Array(W * H);
    for (let i = 0; i < a.length; i++) { a[i] = px[i * 4 + 3] / 255; b[i] = a[i] > .5; }

    const { lab, comps } = label(b, W, H);
    const e = erode(b, W, H, 3);

    // walls, grouped by the letter they belong to
    const walls = comps.map(() => []);
    for (const loop of contours(a, W, H, 2)) {
      const id = labelNear(lab, W, H, loop[0], loop[1]);
      if (id >= 0) walls[id].push(move(loop));
    }
    const fillH = fills(e, lab, W, H, false, comps.length);
    const fillV = fills(e, lab, W, H, true, comps.length);

    // letters left to right, line by line
    const order = comps.map((c, i) => i)
      .sort((p, q) => (Math.floor(comps[p].minY / lineH) - Math.floor(comps[q].minY / lineH)) || comps[p].minX - comps[q].minX);

    const layers = [];
    let lx = null, ly = null;
    for (let L = 0; L < LAYERS; L++) {
      const fill = L % 2 ? fillV : fillH;
      const moves = [], path = new Path2D();
      let cost = 0;
      for (const id of order) {
        for (const mv of walls[id].concat(fill[id])) {
          const travel = lx === null ? 0 : Math.hypot(mv.p[0] - lx, mv.p[1] - ly) / 3;
          cost += travel;
          moves.push({ m: mv, t0: cost, travel, fx: lx ?? mv.p[0], fy: ly ?? mv.p[1] });
          cost += mv.len;
          addTo(path, mv.p);
          lx = mv.p[mv.p.length - 2]; ly = mv.p[mv.p.length - 1];
        }
      }
      layers.push({ moves, cost, path });
    }

    // soft contact shadow, pre-blurred once
    const sp = 40, sh = document.createElement('canvas');
    sh.width = W + sp * 2; sh.height = H + sp * 2;
    const s = sh.getContext('2d');
    s.shadowColor = 'rgb(60,20,35)'; s.shadowBlur = 18; s.shadowOffsetX = 10000;
    s.drawImage(m.canvas, sp - 10000, sp);

    return { W, H, Z: LAYERS * LAYER_H, layers, shadow: sh, sp };
  }

  function label(b, W, H) {
    const lab = new Int32Array(W * H).fill(-1), q = new Int32Array(W * H), comps = [];
    for (let i = 0; i < b.length; i++) {
      if (!b[i] || lab[i] >= 0) continue;
      const id = comps.length;
      let h = 0, t = 0, minX = W, minY = H;
      q[t++] = i; lab[i] = id;
      while (h < t) {
        const p = q[h++], x = p % W, y = (p - x) / W;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > 0 && b[p - 1] && lab[p - 1] < 0) { lab[p - 1] = id; q[t++] = p - 1; }
        if (x < W - 1 && b[p + 1] && lab[p + 1] < 0) { lab[p + 1] = id; q[t++] = p + 1; }
        if (y > 0 && b[p - W] && lab[p - W] < 0) { lab[p - W] = id; q[t++] = p - W; }
        if (y < H - 1 && b[p + W] && lab[p + W] < 0) { lab[p + W] = id; q[t++] = p + W; }
      }
      comps.push({ minX, minY });
    }
    return { lab, comps };
  }

  function labelNear(lab, W, H, x, y) {
    x = Math.round(x); y = Math.round(y);
    for (let r = 0; r <= 3; r++)
      for (let dy = -r; dy <= r; dy++)
        for (let dx = -r; dx <= r; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < W && yy < H && lab[yy * W + xx] >= 0) return lab[yy * W + xx];
        }
    return -1;
  }

  // square erosion so infill stays a strand-width inside the walls
  function erode(b, W, H, r) {
    const t = new Uint8Array(W * H), o = new Uint8Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = 1;
        for (let k = -r; k <= r && v; k++) { const xx = x + k; v = xx >= 0 && xx < W ? b[y * W + xx] : 0; }
        t[y * W + x] = v;
      }
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let v = 1;
        for (let k = -r; k <= r && v; k++) { const yy = y + k; v = yy >= 0 && yy < H ? t[yy * W + x] : 0; }
        o[y * W + x] = v;
      }
    return o;
  }

  // marching squares -> closed outlines (flat [x0,y0,x1,y1,...])
  function contours(a, W, H, g) {
    const cols = Math.floor((W - 1) / g), rows = Math.floor((H - 1) / g), stride = cols + 1;
    const v = (i, j) => a[j * g * W + i * g];
    const hk = (i, j) => (j * stride + i) * 2, vk = (i, j) => (j * stride + i) * 2 + 1;
    // edges: 0 top, 1 right, 2 bottom, 3 left; case = tl<<3 | tr<<2 | br<<1 | bl
    const T = [[], [[3, 2]], [[2, 1]], [[3, 1]], [[0, 1]], [[0, 1], [3, 2]], [[0, 2]], [[0, 3]],
      [[0, 3]], [[0, 2]], [[0, 3], [2, 1]], [[0, 1]], [[3, 1]], [[2, 1]], [[3, 2]], []];
    const adj = new Map();
    const link = (p, q) => {
      (adj.get(p) || adj.set(p, []).get(p)).push(q);
      (adj.get(q) || adj.set(q, []).get(q)).push(p);
    };
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < cols; i++) {
        const c = (v(i, j) > .5) << 3 | (v(i + 1, j) > .5) << 2 | (v(i + 1, j + 1) > .5) << 1 | (v(i, j + 1) > .5);
        if (!T[c].length) continue;
        const e = [hk(i, j), vk(i + 1, j), hk(i, j + 1), vk(i, j)];
        for (const [p, q] of T[c]) link(e[p], e[q]);
      }
    const pt = k => {
      const n = k >> 1, i = n % stride, j = (n - i) / stride;
      if (k & 1) { const a0 = v(i, j), a1 = v(i, j + 1); return [i * g, (j + (.5 - a0) / (a1 - a0)) * g]; }
      const a0 = v(i, j), a1 = v(i + 1, j);
      return [(i + (.5 - a0) / (a1 - a0)) * g, j * g];
    };
    const seen = new Set(), loops = [];
    for (const k of adj.keys()) {
      if (seen.has(k)) continue;
      const loop = [];
      let prev = -1, cur = k;
      while (cur !== undefined && !seen.has(cur)) {
        seen.add(cur); loop.push(pt(cur));
        const nb = adj.get(cur);
        const next = nb[0] !== prev ? nb[0] : nb[1];
        prev = cur; cur = next;
      }
      if (loop.length < 5) continue;
      loop.push(loop[0]);
      loops.push(simplify(loop, .3).flat());
    }
    return loops;
  }

  function simplify(p, eps) {
    const keep = new Uint8Array(p.length), st = [[0, p.length - 1]];
    keep[0] = keep[p.length - 1] = 1;
    while (st.length) {
      const [s, e] = st.pop();
      const [x1, y1] = p[s], [x2, y2] = p[e], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
      let md = 0, mi = -1;
      for (let i = s + 1; i < e; i++) {
        const d = L < 1e-6 ? Math.hypot(p[i][0] - x1, p[i][1] - y1) : Math.abs((p[i][0] - x1) * dy - (p[i][1] - y1) * dx) / L;
        if (d > md) { md = d; mi = i; }
      }
      if (md > eps) { keep[mi] = 1; st.push([s, mi], [mi, e]); }
    }
    return p.filter((_, i) => keep[i]);
  }

  // rectilinear infill, zig-zag per letter
  function fills(e, lab, W, H, vertical, n) {
    const out = Array.from({ length: n }, () => []), flip = new Uint8Array(n);
    const lines = vertical ? W : H, len = vertical ? H : W;
    for (let u = Math.floor(((lines - 1) % GAP) / 2); u < lines; u += GAP) {
      const row = new Map();
      let s = -1;
      for (let t = 0; t <= len; t++) {
        const idx = vertical ? t * W + u : u * W + t;
        const on = t < len && e[idx];
        if (on && s < 0) s = t;
        else if (!on && s >= 0) {
          if (t - 1 - s >= 2) {
            const id = lab[vertical ? s * W + u : u * W + s];
            (row.get(id) || row.set(id, []).get(id)).push([s, t - 1]);
          }
          s = -1;
        }
      }
      for (const [id, runs] of row) {
        if (flip[id]) { runs.reverse(); runs.forEach(r => r.reverse()); }
        flip[id] ^= 1;
        for (const [r0, r1] of runs) out[id].push(move(vertical ? [u, r0, u, r1] : [r0, u, r1, u]));
      }
    }
    return out;
  }

  function move(p) {
    const n = p.length / 2, cum = new Float32Array(n);
    for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(p[2 * i] - p[2 * i - 2], p[2 * i + 1] - p[2 * i - 1]);
    return { p, cum, len: cum[n - 1] };
  }

  function addTo(path, p) {
    path.moveTo(p[0], p[1]);
    for (let i = 2; i < p.length; i += 2) path.lineTo(p[i], p[i + 1]);
  }

  function partial(m, d) {
    const { p, cum } = m, path = new Path2D();
    path.moveTo(p[0], p[1]);
    let i = 1;
    while (i < cum.length && cum[i] <= d) { path.lineTo(p[2 * i], p[2 * i + 1]); i++; }
    let x = p[2 * i - 2], y = p[2 * i - 1];
    if (i < cum.length) {
      const k = (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
      x += (p[2 * i] - x) * k; y += (p[2 * i + 1] - y) * k;
      path.lineTo(x, y);
    }
    return { path, x, y };
  }

  /* ---------- camera (orthographic, so every layer is an affine plane) ---------- */

  function plane(cam, z) {
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), ct = Math.cos(cam.tilt), st = Math.sin(cam.tilt), s = cam.s;
    const a = s * cy, b = s * sy * ct, c = -s * sy, d = s * cy * ct;
    return [a, b, c, d, cam.x - (a * M.W + c * M.H) / 2, cam.y - (b * M.W + d * M.H) / 2 - s * st * (z - M.Z / 2)];
  }
  const setT = m => ctx.setTransform(m[0] * dpr, m[1] * dpr, m[2] * dpr, m[3] * dpr, m[4] * dpr, m[5] * dpr);
  const proj = (m, x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

  function fit(yaw, tilt, w, h) {
    const cy = Math.abs(Math.cos(yaw)), sy = Math.abs(Math.sin(yaw));
    const pw = M.W * cy + M.H * sy;
    const ph = (M.W * sy + M.H * cy) * Math.cos(tilt) + M.Z * Math.sin(tilt);
    return Math.min(w / pw, h / ph);
  }

  function introCam() {
    const yaw = lerp(-30, -12, clamp(S.clock / 6)) * DEG, tilt = 56 * DEG;
    return { yaw, tilt, s: fit(yaw, tilt, vw * (vw < 700 ? .94 : .8), vh * .6), x: vw / 2, y: vh * .48 };
  }

  const par = { yaw: 0, tilt: 0 };
  function restCam() {
    const r = stage.getBoundingClientRect();
    return { yaw: par.yaw, tilt: TILT + par.tilt, s: fit(0, TILT, r.width, r.height) * .96, x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  function camera() {
    if (!S.first || S.phase === 'idle') return restCam();
    if (S.phase === 'settle') {
      const a = introCam(), b = restCam(), k = S.settle;
      return { yaw: lerp(a.yaw, b.yaw, k), tilt: lerp(a.tilt, b.tilt, k), s: lerp(a.s, b.s, k), x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k) };
    }
    return introCam();
  }

  /* ---------- print state machine ---------- */

  const S = { phase: 'boot', t: 0, clock: 0, speed: 1, first: true, L: 0, lt: 0, mi: 0, done: 0, live: null, part: null, noz: [0, 0, 0], nozA: 0, ink: 1, cure: 0, settle: 0, dirty: true };

  function start(first) {
    const m0 = M.layers[0].moves[0].m;
    Object.assign(S, {
      phase: 'home', t: 0, speed: 1, first, L: 0, lt: 0, mi: 0, done: 0, live: new Path2D(), part: null,
      nozA: 1, ink: 1, cure: 0, settle: 0, dirty: true,
      from: [M.W * .5, M.H * .5, M.Z + 260], to: [m0.p[0], m0.p[1], LAYER_H * .5 + BEAD * .5],
    });
    if (first) S.clock = 0;
    S.noz = S.from.slice();
    body.classList.add('booted');
  }

  function finish() {
    Object.assign(S, { phase: 'idle', first: false, done: LAYERS, part: null, nozA: 0, dirty: true });
    body.classList.add('booted', 'ready');
  }

  function update(dt) {
    const d = dt * S.speed;
    S.t += d; S.clock += d;
    switch (S.phase) {
      case 'home': {
        const k = ease(clamp(S.t / .9));
        for (let i = 0; i < 3; i++) S.noz[i] = lerp(S.from[i], S.to[i], k);
        if (S.t >= .9) { S.phase = 'print'; S.t = 0; }
        break;
      }
      case 'print': {
        S.lt += d;
        const layer = M.layers[S.L], dur = LAYER_TIME[S.L];
        const cost = Math.min(1, S.lt / dur) * layer.cost;
        while (S.mi < layer.moves.length && layer.moves[S.mi].t0 + layer.moves[S.mi].m.len <= cost) addTo(S.live, layer.moves[S.mi++].m.p);
        S.part = null;
        if (S.mi < layer.moves.length) {
          const mv = layer.moves[S.mi];
          if (cost >= mv.t0) {
            S.part = partial(mv.m, cost - mv.t0);
            S.noz[0] = S.part.x; S.noz[1] = S.part.y;
          } else {
            const k = mv.travel ? clamp(1 - (mv.t0 - cost) / mv.travel) : 1;
            S.noz[0] = lerp(mv.fx, mv.m.p[0], k); S.noz[1] = lerp(mv.fy, mv.m.p[1], k);
          }
        }
        S.noz[2] = S.L * LAYER_H + LAYER_H * .5 + BEAD * .5;
        S.ink = 1 - .8 * (S.L + Math.min(1, S.lt / dur)) / LAYERS;
        if (S.lt >= dur) {
          S.done = ++S.L; S.lt -= dur; S.mi = 0; S.live = new Path2D(); S.part = null;
          if (S.L === LAYERS) { S.phase = 'lift'; S.t = 0; S.lt = 0; }
        }
        break;
      }
      case 'lift': {
        const k = clamp(S.t / .6);
        S.noz[2] = LAYERS * LAYER_H + ease(k) * 180;
        S.nozA = 1 - k;
        if (k >= 1) { S.phase = 'cure'; S.t = 0; }
        break;
      }
      case 'cure':
        S.cure = clamp(S.t / 1);
        if (S.cure >= 1) {
          S.t = 0;
          body.classList.add('ready');
          if (S.first) S.phase = 'settle'; else finish();
        }
        break;
      case 'settle':
        S.settle = ease(clamp(S.t / 1.4));
        if (S.t >= 1.4) finish();
        break;
    }
  }

  /* ---------- drawing ---------- */

  // [underside, body, highlight] per layer, uncured and cured
  const SH = [0, 1].map(cured => Array.from({ length: LAYERS }, (_, i) => {
    const f = i / (LAYERS - 1);
    return cured
      ? [`hsl(346 50% ${26 + f * 12}%)`, `hsl(346 58% ${40 + f * 12}%)`, `hsl(346 85% ${66 + f * 8}%)`]
      : [`hsl(346 45% ${62 + f * 5}%)`, `hsl(346 62% ${75 + f * 5}%)`, `hsl(346 90% ${89 + f * 3}%)`];
  }));

  function strand(cam, i, path, cured) {
    const [lo, mid, hi] = SH[+cured][i], z = i * LAYER_H + LAYER_H * .5;
    setT(plane(cam, z - LAYER_H * .22)); ctx.strokeStyle = lo; ctx.lineWidth = BEAD; ctx.stroke(path);
    setT(plane(cam, z)); ctx.strokeStyle = mid; ctx.lineWidth = BEAD * .78; ctx.stroke(path);
    setT(plane(cam, z + LAYER_H * .3)); ctx.strokeStyle = hi; ctx.lineWidth = BEAD * .3; ctx.stroke(path);
  }

  function bbox(cam) {
    let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
    for (const z of [0, M.Z]) {
      const m = plane(cam, z);
      for (const [x, y] of [[0, 0], [M.W, 0], [0, M.H], [M.W, M.H]]) {
        const [sx, sy] = proj(m, x, y);
        x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
      }
    }
    return { x0, x1, y0, y1 };
  }

  function grid(cam, alpha) {
    const m = plane(cam, 0), step = 40;
    const x0 = -M.W * .3, x1 = M.W * 1.3, y0 = -M.H * 1.4, y1 = M.H * 2.4;
    setT(m);
    ctx.beginPath();
    for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) { ctx.moveTo(x, y0); ctx.lineTo(x, y1); }
    for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) { ctx.moveTo(x0, y); ctx.lineTo(x1, y); }
    ctx.lineWidth = 1 / cam.s;
    ctx.strokeStyle = `rgba(${INK},${alpha})`;
    ctx.stroke();
    // fade the plate out radially
    const [cx, cy] = proj(m, M.W / 2, M.H / 2), R = M.W * cam.s * .62;
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr * .5, cx * dpr, cy * dpr);
    ctx.globalCompositeOperation = 'destination-in';
    const g = ctx.createRadialGradient(0, 0, 0, 0, 0, R);
    g.addColorStop(0, '#000'); g.addColorStop(.5, 'rgba(0,0,0,.55)'); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(-R, -R, R * 2, R * 2);
    ctx.restore();
  }

  function syringe(x, y, ink, alpha) {
    const k = clamp(Math.min(vh / 900, vw / 760), .5, 1.1);
    ctx.save();
    ctx.globalAlpha = alpha;
    // pneumatic line up to the gantry
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const top = y - 174 * k, ax = vw * .5 + (x - vw * .5) * .25;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.bezierCurveTo(x, top - 150 * k, ax, -60 + 120 * k, ax, -20);
    ctx.strokeStyle = `rgba(${INK},.55)`; ctx.lineWidth = 1.6; ctx.stroke();

    ctx.setTransform(dpr * k, 0, 0, dpr * k, x * dpr, y * dpr);
    ctx.strokeStyle = `rgb(${INK})`; ctx.lineJoin = 'round';
    // needle
    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(0, -34); ctx.lineWidth = 1.4; ctx.stroke();
    // luer hub
    ctx.beginPath(); ctx.moveTo(-2, -34); ctx.lineTo(2, -34); ctx.lineTo(6, -48); ctx.lineTo(-6, -48); ctx.closePath();
    ctx.fillStyle = PINK; ctx.fill(); ctx.lineWidth = 1.1; ctx.stroke();
    // barrel
    ctx.fillStyle = 'rgba(242,239,233,.92)';
    ctx.fillRect(-11, -166, 22, 118); ctx.strokeRect(-11, -166, 22, 118);
    // bioink + wiper
    const h = 104 * ink;
    ctx.fillStyle = SH[0][LAYERS - 1][1]; ctx.fillRect(-9.4, -50 - h, 18.8, h);
    ctx.fillStyle = `rgb(${INK})`; ctx.fillRect(-10.4, -54 - h, 20.8, 4);
    // graduations
    ctx.beginPath();
    for (let g = 0; g < 9; g++) { const gy = -60 - g * 12; ctx.moveTo(11, gy); ctx.lineTo(g % 2 ? 8 : 5, gy); }
    ctx.lineWidth = .8; ctx.stroke();
    // cap
    ctx.fillRect(-13.5, -174, 27, 8);
    ctx.restore();
  }

  let last = null;
  const same = (a, b) => b && Math.abs(a.yaw - b.yaw) + Math.abs(a.tilt - b.tilt) + Math.abs(a.s - b.s) * 50 + Math.abs(a.x - b.x) + Math.abs(a.y - b.y) < 1e-3;

  function draw() {
    const cam = camera();
    if (S.phase === 'idle' && !S.dirty && same(cam, last)) return;
    S.dirty = false; last = cam;

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';

    grid(cam, S.phase === 'idle' ? .06 : lerp(.1, .06, S.settle));

    const printed = S.done / LAYERS;
    if (printed > 0) {
      setT(plane(cam, 0));
      ctx.globalAlpha = .2 * printed;
      ctx.drawImage(M.shadow, -M.sp + 5, -M.sp + 14);
      ctx.globalAlpha = 1;
    }

    const cured = S.phase === 'settle' || S.phase === 'idle';
    for (let i = 0; i < S.done; i++) strand(cam, i, M.layers[i].path, cured);
    if (S.phase === 'print') {
      strand(cam, S.L, S.live, false);
      if (S.part) strand(cam, S.L, S.part.path, false);
    }

    if (S.phase === 'cure') {
      const bb = bbox(cam), x = lerp(bb.x0 - 50, bb.x1 + 50, ease(S.cure));
      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.beginPath(); ctx.rect(0, 0, x, vh); ctx.clip();
      for (let i = 0; i < LAYERS; i++) strand(cam, i, M.layers[i].path, true);
      ctx.restore();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const y0 = bb.y0 - 40, hh = bb.y1 - bb.y0 + 80;
      const g = ctx.createLinearGradient(x - 90, 0, x, 0);
      g.addColorStop(0, `rgba(${UV},0)`); g.addColorStop(1, `rgba(${UV},.16)`);
      ctx.fillStyle = g; ctx.fillRect(x - 90, y0, 90, hh);
      ctx.fillStyle = `rgba(${UV},.75)`; ctx.fillRect(x, y0, 1, hh);
      ctx.font = '500 10px "IBM Plex Mono", monospace';
      ctx.fillText('405 NM', x + 6, y0 + 10);
    }

    if (S.nozA > 0 && (S.phase === 'home' || S.phase === 'print' || S.phase === 'lift')) {
      const [sx, sy] = proj(plane(cam, S.noz[2]), S.noz[0], S.noz[1]);
      syringe(sx, sy, S.ink, S.nozA);
    }
  }

  /* ---------- printer readout ---------- */

  let hudAt = 0;
  function hud(now) {
    if (now - hudAt < 90 || S.phase === 'idle' || S.phase === 'boot') return;
    hudAt = now;
    const kpa = (24 + Math.sin(S.clock * 9) * .5 + Math.sin(S.clock * 23) * .2).toFixed(1);
    let t = '';
    if (S.phase === 'home') t = 'Homing · bed 37.0 °C · 22G needle';
    else if (S.phase === 'print') t = `Layer ${S.L + 1}/${LAYERS} · Z ${((S.L + 1) * LAYER_MM).toFixed(2)} mm · ${kpa} kPa · ${S.L % 2 ? '90' : '0'}° pass`;
    else if (S.phase === 'lift') t = `Print done · Z ${(LAYERS * LAYER_MM).toFixed(2)} mm · lifting`;
    else if (S.phase === 'cure' || S.phase === 'settle') t = `Crosslinking · 405 nm · ${Math.round(Math.min(1, S.cure) * 30)} s`;
    hudL.textContent = t;
    hudR.textContent = S.speed > 1 ? 'Fast-forward ×5' : (fine ? 'Click to fast-forward' : 'Tap to fast-forward');
  }

  /* ---------- neurons on click ---------- */

  const neurons = [];
  const GROW = .24; // px per ms
  const alphaOf = (n, now) => clamp(1 - (now - n.last - 2200) / 1000);

  function spawn(x, y) {
    const now = performance.now();
    let target = null, best = 300;
    for (const n of neurons) {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d > 50 && d < best && alphaOf(n, now) > .4) { best = d; target = n; }
    }
    const branches = [];
    const grow = (x0, y0, ang, len, w, t0, depth, aim) => {
      const p = [x0, y0];
      let px = x0, py = y0, a = ang, i = 0;
      const steps = Math.max(2, Math.round(len / 4));
      for (i = 1; i <= steps; i++) {
        if (aim) {
          if (Math.hypot(aim.x - px, aim.y - py) < 8) break;
          let da = Math.atan2(aim.y - py, aim.x - px) - a;
          da = Math.atan2(Math.sin(da), Math.cos(da));
          a += da * .2 + (Math.random() - .5) * .25;
        } else a += (Math.random() - .5) * .45;
        px += Math.cos(a) * 4; py += Math.sin(a) * 4;
        p.push(px, py);
        if (!aim && depth < 2 && i > 2 && i < steps - 2 && Math.random() < .13)
          grow(px, py, a + (Math.random() < .5 ? -1 : 1) * (.45 + Math.random() * .5), (steps - i) * 3, w * .7, t0 + i * 4 / GROW, depth + 1);
      }
      const b = { p, t0, t1: t0 + (p.length / 2 - 1) * 4 / GROW, w };
      branches.push(b);
      return b;
    };
    const n = 5 + (Math.random() * 3 | 0), base = Math.random() * Math.PI * 2;
    for (let k = 0; k < n; k++) grow(x, y, base + k * 2 * Math.PI / n + (Math.random() - .5) * .7, 16 + Math.random() * 34, 1.25, 80, 0);
    const axon = grow(x, y, target ? Math.atan2(target.y - y, target.x - x) + (Math.random() - .5) * .8 : Math.random() * Math.PI * 2,
      target ? best + 60 : 90 + Math.random() * 70, 1.05, 80, 2, target);
    if (!target) {
      const ex = axon.p[axon.p.length - 2], ey = axon.p[axon.p.length - 1];
      const ea = Math.atan2(ey - axon.p[axon.p.length - 3], ex - axon.p[axon.p.length - 4]);
      for (let k = -1; k <= 1; k++) grow(ex, ey, ea + k * .6, 10 + Math.random() * 12, .8, axon.t1, 3);
    }
    const cum = [0];
    for (let i = 2; i < axon.p.length; i += 2) cum.push(cum[cum.length - 1] + Math.hypot(axon.p[i] - axon.p[i - 2], axon.p[i + 1] - axon.p[i - 1]));
    const grown = Math.max(...branches.map(b => b.t1));
    neurons.push({ x, y, born: now, last: now + grown, branches, axon, cum, target, fireAt: now + axon.t1 + 60, fired: 0, flash: 0 });
    if (neurons.length > 14) neurons.shift();
  }

  let fxLive = false;
  function drawFx(now) {
    if (!neurons.length) {
      if (fxLive) { fctx.setTransform(1, 0, 0, 1, 0, 0); fctx.clearRect(0, 0, fx.width, fx.height); fxLive = false; }
      return;
    }
    fxLive = true;
    fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fctx.clearRect(0, 0, vw, vh);
    fctx.lineCap = 'round'; fctx.lineJoin = 'round';
    for (let i = neurons.length - 1; i >= 0; i--) {
      const n = neurons[i], age = now - n.born;

      // action potential down the axon, handed on to whatever it synapses onto
      const apDur = n.cum[n.cum.length - 1] / .75;
      if (now >= n.fireAt && n.fired !== n.fireAt && now - n.fireAt >= apDur) {
        n.fired = n.fireAt;
        if (n.target) {
          const t = n.target;
          t.flash = now; t.last = Math.max(t.last, now);
          if (t.target && neurons.includes(t.target)) t.fireAt = now + 40;
        }
      }

      const a = alphaOf(n, now);
      if (a <= 0) { neurons.splice(i, 1); continue; }
      fctx.globalAlpha = a;

      if (age < 600) {
        const k = age / 600;
        fctx.beginPath(); fctx.arc(n.x, n.y, 4 + 20 * (1 - (1 - k) ** 3), 0, Math.PI * 2);
        fctx.strokeStyle = `rgba(209,64,106,${.5 * (1 - k)})`; fctx.lineWidth = 1; fctx.stroke();
      }

      fctx.strokeStyle = `rgba(${INK},.62)`;
      for (const b of n.branches) {
        const vis = (age - b.t0) * GROW / 4;
        if (vis <= 0) continue;
        const cnt = Math.min(b.p.length / 2 - 1, vis), whole = Math.floor(cnt);
        fctx.beginPath(); fctx.moveTo(b.p[0], b.p[1]);
        for (let j = 1; j <= whole; j++) fctx.lineTo(b.p[2 * j], b.p[2 * j + 1]);
        if (whole < cnt) {
          const f = cnt - whole;
          fctx.lineTo(lerp(b.p[2 * whole], b.p[2 * whole + 2], f), lerp(b.p[2 * whole + 1], b.p[2 * whole + 3], f));
        }
        fctx.lineWidth = b.w; fctx.stroke();
      }

      if (n.target && age > n.axon.t1) {
        const p = n.axon.p;
        fctx.beginPath(); fctx.arc(p[p.length - 2], p[p.length - 1], 2, 0, Math.PI * 2);
        fctx.strokeStyle = PINK; fctx.lineWidth = 1; fctx.stroke();
      }

      const fl = n.flash ? clamp(1 - (now - n.flash) / 500) : 0;
      if (fl > 0) {
        fctx.beginPath(); fctx.arc(n.x, n.y, 4 + (1 - fl) * 14, 0, Math.PI * 2);
        fctx.strokeStyle = `rgba(209,64,106,${fl * .6})`; fctx.lineWidth = 1; fctx.stroke();
      }
      fctx.beginPath(); fctx.arc(n.x, n.y, 3.4 * clamp(age / 140) + fl * 1.5, 0, Math.PI * 2);
      fctx.fillStyle = PINK; fctx.fill();

      const apK = (now - n.fireAt) / apDur;
      if (now >= n.fireAt && apK < 1) {
        const d = apK * n.cum[n.cum.length - 1];
        let j = 1;
        while (j < n.cum.length - 1 && n.cum[j] < d) j++;
        const f = clamp((d - n.cum[j - 1]) / (n.cum[j] - n.cum[j - 1] || 1)), p = n.axon.p;
        const x = lerp(p[2 * j - 2], p[2 * j], f), y = lerp(p[2 * j - 1], p[2 * j + 1], f);
        fctx.save();
        fctx.shadowColor = PINK; fctx.shadowBlur = 10;
        fctx.beginPath(); fctx.arc(x, y, 2.4, 0, Math.PI * 2); fctx.fillStyle = PINK; fctx.fill();
        fctx.restore();
      }
    }
    fctx.globalAlpha = 1;
  }

  /* ---------- cursor: reticle + lagging gantry rails ---------- */

  const cur = $('#cursor'), coords = $('#coords'), railX = $('#rail-x'), railY = $('#rail-y');
  const mouse = { x: -100, y: -100, nx: 0, ny: 0 }, rail = { x: -100, y: -100 };
  const mm = v => (v / 3.78).toFixed(1).padStart(5, '0');

  if (fine) {
    body.classList.add('has-cursor');
    addEventListener('pointermove', e => {
      if (e.pointerType !== 'mouse') return;
      if (!body.classList.contains('cursor-on')) { rail.x = e.clientX; rail.y = e.clientY; }
      mouse.x = e.clientX; mouse.y = e.clientY;
      mouse.nx = e.clientX / vw * 2 - 1; mouse.ny = e.clientY / vh * 2 - 1;
      body.classList.add('cursor-on');
      cur.style.transform = `translate(${mouse.x}px,${mouse.y}px)`;
      cur.classList.toggle('is-link', !!e.target.closest('a,button'));
      coords.textContent = `X ${mm(mouse.x)}  Y ${mm(mouse.y)}`;
    }, { passive: true });
    document.documentElement.addEventListener('mouseleave', () => body.classList.remove('cursor-on'));
    addEventListener('pointerup', () => cur.classList.remove('is-down'));
  }

  function cursorStep() {
    if (!fine) return;
    rail.x += (mouse.x - rail.x) * .12;
    rail.y += (mouse.y - rail.y) * .12;
    railX.style.transform = `translateY(${rail.y}px)`;
    railY.style.transform = `translateX(${rail.x}px)`;
    if (!reduce) {
      par.yaw += (mouse.nx * 7 * DEG - par.yaw) * .06;
      par.tilt += (mouse.ny * 5 * DEG - par.tilt) * .06;
    }
  }

  addEventListener('pointerdown', e => {
    cur.classList.add('is-down');
    if (S.phase === 'boot') return;
    if (S.phase !== 'idle') { S.speed = 5; return; }
    if (!reduce && !e.target.closest('button')) spawn(e.clientX, e.clientY);
  });
  addEventListener('keydown', e => {
    if (S.phase !== 'idle' && S.phase !== 'boot' && !e.metaKey && !e.ctrlKey) S.speed = 5;
  });

  $('#reprint').addEventListener('click', () => {
    if (!M || S.phase !== 'idle') return;
    if (reduce) return;
    start(false);
  });

  /* ---------- clock ---------- */

  const clockEl = $('#clock');
  const tf = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Detroit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short' });
  const tick = () => { clockEl.textContent = tf.format(new Date()); };
  tick(); setInterval(tick, 15000);

  /* ---------- boot ---------- */

  const wantSplit = () => vw / vh < .8 && vw < 700;

  function resize() {
    vw = innerWidth; vh = innerHeight; dpr = Math.min(2, devicePixelRatio || 1);
    for (const c of [cv, fx]) { c.width = Math.round(vw * dpr); c.height = Math.round(vh * dpr); }
    if (M && wantSplit() !== split) {
      split = !split;
      M = build(split ? ['Diya', 'Maisuria'] : ['Diya Maisuria']);
      finish();
    }
    S.dirty = true;
  }
  addEventListener('resize', resize);

  let prev = performance.now();
  function frame(now) {
    const dt = Math.min(.05, (now - prev) / 1000);
    prev = now;
    cursorStep();
    if (M) {
      update(dt);
      draw();
      hud(now);
    }
    drawFx(now);
    requestAnimationFrame(frame);
  }

  (async () => {
    resize();
    requestAnimationFrame(frame);
    try { await Promise.race([document.fonts.load(FONT), new Promise(r => setTimeout(r, 3000))]); } catch (_) { /* fall back to whatever loaded */ }
    split = wantSplit();
    M = build(split ? ['Diya', 'Maisuria'] : ['Diya Maisuria']);
    if (reduce) finish(); else start(true);
  })();
})();
