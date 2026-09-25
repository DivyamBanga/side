// Method 2: electrospinning. A charged jet whips in a widening spiral and lays a
// single continuous fibre onto a collector patterned like her name. The whole
// fibre is simulated up front (seeded, so it's the same every visit) and then
// replayed onto an offscreen "mat" — each frame only draws the newest stretch.

import { rasterize, fit, clamp, lerp, ease, easeOut, rng } from './text.js';

export const caption = 'Name, electrospun. One polymer fibre whipped out at 15 kV onto a collector patterned with the letters.';

const LEAD = .35, SPIN = 2.9, SHEEN = .7, SETTLE = 1.0;
const STEP = 1.6;          // fibre advance per sim step (mask px)
const DENSITY = .95;       // mask area covered per unit of fibre
const CHUNK = 220;         // steps per fibre "pass" (each pass gets its own weight)

function chamfer(b, W, H, want) {
  const d = new Float32Array(W * H);
  for (let i = 0; i < d.length; i++) d[i] = b[i] === want ? 1e9 : 0;
  const D = Math.SQRT2;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!d[i]) continue;
      let v = d[i];
      if (x > 0) v = Math.min(v, d[i - 1] + 1);
      if (y > 0) {
        v = Math.min(v, d[i - W] + 1);
        if (x > 0) v = Math.min(v, d[i - W - 1] + D);
        if (x < W - 1) v = Math.min(v, d[i - W + 1] + D);
      }
      d[i] = v;
    }
  for (let y = H - 1; y >= 0; y--)
    for (let x = W - 1; x >= 0; x--) {
      const i = y * W + x;
      if (!d[i]) continue;
      let v = d[i];
      if (x < W - 1) v = Math.min(v, d[i + 1] + 1);
      if (y < H - 1) {
        v = Math.min(v, d[i + W] + 1);
        if (x < W - 1) v = Math.min(v, d[i + W + 1] + D);
        if (x > 0) v = Math.min(v, d[i + W - 1] + D);
      }
      d[i] = v;
    }
  return d;
}

// Lay the whole fibre: a curvature random walk that's steered back into the
// letters and drifts toward the least-covered part of the collector.
function spin(mask) {
  const { W, H, b } = mask;
  const din = chamfer(b, W, H, 1), dout = chamfer(b, W, H, 0);
  const sd = new Float32Array(W * H);
  for (let i = 0; i < sd.length; i++) sd[i] = b[i] ? din[i] : -dout[i];
  const at = (x, y) => sd[clamp(Math.round(y), 0, H - 1) * W + clamp(Math.round(x), 0, W - 1)];

  const C = 12, cw = Math.ceil(W / C), ch = Math.ceil(H / C);
  const cover = new Float32Array(cw * ch), cells = [];
  let area = 0;
  for (let i = 0; i < b.length; i++) area += b[i];
  for (let j = 0; j < ch; j++)
    for (let i = 0; i < cw; i++)
      if (at(i * C + C / 2, j * C + C / 2) > 2) cells.push(j * cw + i);

  const rand = rng(20260925);
  const total = Math.round(area / DENSITY / STEP);
  const pts = new Float32Array((total + 1) * 2), on = new Uint8Array(total + 1);
  const c0 = cells[Math.floor(cells.length * .1)];
  let x = (c0 % cw) * C + C / 2, y = Math.floor(c0 / cw) * C + C / 2;
  let a = rand() * Math.PI * 2, k = 0, tx = x, ty = y, since = 1e9;
  const turn = (want, amt) => { let d = want - a; d = Math.atan2(Math.sin(d), Math.cos(d)); a += d * amt; };

  pts[0] = x; pts[1] = y; on[0] = 1;
  for (let s = 1; s <= total; s++) {
    if (++since > 160 || Math.hypot(tx - x, ty - y) < 10) {
      // next target: thin spots nearby win, so the mat fills locally and then hops on
      let best = -1, bv = 1e9;
      for (let n = 0; n < 9; n++) {
        const c = cells[Math.floor(rand() * cells.length)];
        const cx = (c % cw) * C + C / 2, cy = Math.floor(c / cw) * C + C / 2;
        const v = cover[c] + Math.hypot(cx - x, cy - y) / 35 + rand() * 1.5;
        if (v < bv) { bv = v; best = c; }
      }
      tx = (best % cw) * C + C / 2; ty = Math.floor(best / cw) * C + C / 2; since = 0;
    }
    const far = Math.hypot(tx - x, ty - y) > 60;
    k += -k * (far ? .2 : .05) + (rand() - .5) * .11;
    k = clamp(k, -.5, .5);
    a += k;
    const d = at(x, y);
    if (d < 5 && !far) {
      const gx = at(x + 2, y) - at(x - 2, y), gy = at(x, y + 2) - at(x, y - 2);
      turn(Math.atan2(gy, gx), d < 0 ? .45 : .16);
    }
    // on the way to a distant spot the fibre bridges the gap, like real stray fibres do
    turn(Math.atan2(ty - y, tx - x), far ? .09 : .025);
    x += Math.cos(a) * STEP; y += Math.sin(a) * STEP;
    pts[s * 2] = x; pts[s * 2 + 1] = y;
    on[s] = at(x, y) > -1.5;
    const ci = Math.floor(y / C) * cw + Math.floor(x / C);
    if (ci >= 0 && ci < cover.length) cover[ci] += 1;
  }
  const weight = Array.from({ length: Math.ceil(total / CHUNK) + 1 }, () => [.45 + rand() * .8, .55 + rand() * .45]);
  return { pts, on, total, weight };
}

export async function create(host, opts) {
  const mask = rasterize(host.lines);
  const { W, H } = mask;
  const fibre = spin(mask);
  const spinEnd = LEAD + SPIN, sheenEnd = spinEnd + SHEEN, settleEnd = spinEnd + SETTLE;

  const el = document.createElement('div');
  el.className = 'layer';
  const cv = document.createElement('canvas');
  el.append(cv);
  host.root.append(el);
  const ctx = cv.getContext('2d');

  const mat = document.createElement('canvas'), gloss = document.createElement('canvas'), tmp = document.createElement('canvas');
  const mx = mat.getContext('2d'), gx = gloss.getContext('2d'), tx = tmp.getContext('2d');
  const pad = document.createElement('canvas'), px = pad.getContext('2d');
  let sa = 1, drawn = 0, cols = null;

  const st = { t: 0, speed: 1, first: !!opts.first, running: true, readied: false, dirty: true, lastKey: '' };

  const introRect = () => {
    const b = host.introBox(), s = fit(W, H, 0, 0, 0, b.w, b.h) * .95;
    return { x: b.cx - W * s / 2, y: b.cy - H * s / 2, s };
  };
  const restRect = () => {
    const r = host.stage, s = fit(W, H, 0, 0, 0, r.w, r.h) * .97;
    return { x: r.x + (r.w - W * s) / 2, y: r.y + (r.h - H * s) / 2, s };
  };
  function rect() {
    const t = st.t;
    if (!st.first || t >= settleEnd) return restRect();
    const a = introRect();
    if (t < spinEnd) return a;
    const b = restRect(), k = ease((t - spinEnd) / SETTLE);
    return { x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k), s: lerp(a.s, b.s, k) };
  }

  function colours() {
    const d = host.theme.dark;
    return d
      ? { fibre: w => `rgba(244,232,236,${.22 + w * .2})`, gloss: 'rgba(255,110,150,.95)', jet: 'rgba(255,120,160,', cone: host.theme.accent, pad: 'rgba(255,93,140,.09)' }
      : { fibre: w => `rgba(128,24,58,${.22 + w * .22})`, gloss: 'rgba(236,52,108,.95)', jet: 'rgba(200,50,95,', cone: host.theme.accent, pad: 'rgba(209,64,106,.075)' };
  }

  // draw fibre steps [i0, i1) onto both mats
  function lay(i0, i1) {
    const { pts, on, weight } = fibre;
    for (const [c, isGloss] of [[mx, false], [gx, true]]) {
      c.setTransform(sa, 0, 0, sa, 0, 0);
      c.lineCap = 'round'; c.lineJoin = 'round';
      for (let s = i0; s < i1;) {
        const ch = Math.floor(s / CHUNK), e = Math.min(i1, (ch + 1) * CHUNK);
        const [w, o] = weight[ch];
        c.lineWidth = w * host.dpr / sa;
        c.strokeStyle = isGloss ? cols.gloss : cols.fibre(o);
        // fibre on the patterned collector, then the few strays that land off it, much fainter
        for (const wantOn of [1, 0]) {
          c.beginPath();
          for (let j = s + 1; j <= e && j <= fibre.total; j++) {
            if (!!(on[j] && on[j - 1]) !== !!wantOn) continue;
            c.moveTo(pts[j * 2 - 2], pts[j * 2 - 1]);
            c.lineTo(pts[j * 2], pts[j * 2 + 1]);
          }
          c.globalAlpha = wantOn ? 1 : .16;
          c.stroke();
        }
        c.globalAlpha = 1;
        s = e;
      }
    }
  }

  function rebuild() {
    const s = Math.max(introRect().s, restRect().s);
    sa = Math.min(s * host.dpr, 4096 / W);
    for (const c of [mat, gloss]) { c.width = Math.ceil(W * sa); c.height = Math.ceil(H * sa); }
    cols = colours();
    // the patterned collector electrode the fibres land on
    pad.width = W; pad.height = H;
    px.fillStyle = cols.pad; px.fillRect(0, 0, W, H);
    px.globalCompositeOperation = 'destination-in';
    px.drawImage(mask.canvas, 0, 0);
    px.globalCompositeOperation = 'source-over';
    if (drawn) lay(0, drawn);
  }

  function resize() {
    cv.width = Math.round(host.vw * host.dpr); cv.height = Math.round(host.vh * host.dpr);
    rebuild();
    st.dirty = true;
  }

  function theme() { cols = colours(); rebuild(); st.dirty = true; draw(); }

  function jet(r, t) {
    const { pts } = fibre, dpr = host.dpr;
    const hx = r.x + pts[drawn * 2] * r.s, hy = r.y + pts[drawn * 2 + 1] * r.s;
    const sx = r.x + W * r.s / 2, sy = Math.max(26, r.y - Math.min(host.vh * .26, 190));
    const k = clamp(Math.min(host.vh / 900, host.vw / 760), .55, 1.1);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // spinneret: needle, hub, high-voltage lead
    ctx.strokeStyle = host.theme.ink; ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, sy - 34 * k); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx - 2, sy - 34 * k); ctx.lineTo(sx + 2, sy - 34 * k); ctx.lineTo(sx + 6, sy - 48 * k); ctx.lineTo(sx - 6, sy - 48 * k); ctx.closePath();
    ctx.fillStyle = host.theme.accent; ctx.fill(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx, sy - 48 * k); ctx.lineTo(sx, -10); ctx.strokeStyle = host.theme.dim; ctx.lineWidth = 1.6; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx, sy - 20 * k);
    for (let i = 1; i <= 6; i++) ctx.lineTo(sx + 10 * k + i * 6 * k, sy - 20 * k + (i % 2 ? -5 : 5) * k);
    ctx.lineTo(sx + 70 * k, sy - 20 * k);
    ctx.strokeStyle = host.theme.dim; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = host.theme.dim; ctx.font = `500 ${Math.round(10 * k)}px "IBM Plex Mono", monospace`;
    ctx.fillText('+15 kV', sx + 76 * k, sy - 16 * k);
    // Taylor cone
    const grow = easeOut(clamp(t / LEAD));
    ctx.beginPath();
    ctx.moveTo(sx - 4, sy);
    ctx.quadraticCurveTo(sx - 3, sy + 5 * grow, sx, sy + 11 * grow);
    ctx.quadraticCurveTo(sx + 3, sy + 5 * grow, sx + 4, sy);
    ctx.fillStyle = cols.cone; ctx.fill();
    if (t < LEAD) return;
    // straight jet, then the whipping (bending-instability) spiral down to the collector
    const jx = sx, jy = sy + 11, dx = hx - jx, dy = hy - (jy + 22), L = Math.hypot(dx, dy) || 1;
    const nx = -dy / L, ny = dx / L, Rm = Math.min(70, L * .22) * k;
    ctx.lineWidth = 1;
    let px = jx, py = jy + 22;
    ctx.beginPath(); ctx.moveTo(jx, jy); ctx.lineTo(px, py);
    ctx.strokeStyle = cols.jet + '.9)'; ctx.stroke();
    // a helix seen from the side: loops widen as the jet stretches, the near side reads brighter
    const N = 140;
    for (let i = 1; i <= N; i++) {
      const u = i / N, env = Rm * Math.pow(u, 1.1) * Math.pow(1 - u, .3) * 1.7, ph = t * 42 + u * 46;
      const x = jx + dx * u + nx * Math.cos(ph) * env + (dx / L) * Math.sin(ph) * env * .45;
      const y = jy + 22 + dy * u + ny * Math.cos(ph) * env + (dy / L) * Math.sin(ph) * env * .45;
      const near = .5 + .5 * Math.sin(ph);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y);
      ctx.strokeStyle = cols.jet + ((.35 + near * .6) * (1 - u * .35)).toFixed(3) + ')';
      ctx.lineWidth = .7 + near * .7;
      ctx.stroke();
      px = x; py = y;
    }
    ctx.beginPath(); ctx.arc(hx, hy, 2.2, 0, Math.PI * 2); ctx.fillStyle = cols.cone; ctx.fill();
  }

  // gloss: light catching the fibres, masked by a soft spot (sweep or cursor)
  function glossAt(r, cx, cy, rad, alpha) {
    const w = Math.ceil(W * r.s * host.dpr), h = Math.ceil(H * r.s * host.dpr);
    if (tmp.width !== w || tmp.height !== h) { tmp.width = w; tmp.height = h; }
    tx.globalCompositeOperation = 'source-over';
    tx.clearRect(0, 0, w, h);
    const g = tx.createRadialGradient((cx - r.x) * host.dpr, (cy - r.y) * host.dpr, 0, (cx - r.x) * host.dpr, (cy - r.y) * host.dpr, rad * host.dpr);
    g.addColorStop(0, `rgba(0,0,0,${alpha})`); g.addColorStop(1, 'rgba(0,0,0,0)');
    tx.fillStyle = g; tx.fillRect(0, 0, w, h);
    tx.globalCompositeOperation = 'source-in';
    tx.drawImage(gloss, 0, 0, w, h);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(tmp, r.x * host.dpr, r.y * host.dpr);
  }

  function draw() {
    const t = st.t, r = rect();
    const m = host.mouse;
    const key = `${r.x.toFixed(1)},${r.y.toFixed(1)},${r.s.toFixed(4)},${m.x},${m.y}`;
    if (!st.running && !st.dirty && key === st.lastKey) return;
    st.lastKey = key; st.dirty = false;

    const want = Math.floor(clamp((t - LEAD) / SPIN) * fibre.total);
    if (want > drawn) { lay(drawn, want); drawn = want; }

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.globalAlpha = clamp(t / LEAD);
    ctx.drawImage(pad, r.x * host.dpr, r.y * host.dpr, W * r.s * host.dpr, H * r.s * host.dpr);
    ctx.globalAlpha = 1;
    ctx.drawImage(mat, r.x * host.dpr, r.y * host.dpr, W * r.s * host.dpr, H * r.s * host.dpr);
    if (t < spinEnd) jet(r, t);
    else if (t < sheenEnd) {
      const k = ease((t - spinEnd) / SHEEN);
      glossAt(r, r.x + lerp(-.1, 1.1, k) * W * r.s, r.y + H * r.s / 2, W * r.s * .18, .9 * Math.sin(Math.PI * k));
    }
    if (!st.running && host.fine && m.x > -1) glossAt(r, m.x, m.y, 150, .85);
  }

  function hud(t) {
    const kv = (15 + Math.sin(t * 7) * .08 + Math.sin(t * 19) * .04).toFixed(1);
    if (t < LEAD) host.hud(`Taylor cone forming · ${kv} kV · 1.0 mL/h`);
    else if (t < spinEnd) host.hud(`Spinning · ${kv} kV · 12 cm to collector · mat ${Math.round(clamp((t - LEAD) / SPIN) * 100)}%`);
    else host.hud('Collector off · fibre mat done');
  }

  function frame(dt) {
    if (st.running) {
      st.t += dt * st.speed;
      hud(st.t);
      if (!st.readied && st.t >= spinEnd) { st.readied = true; host.ready(); }
      if (st.t >= Math.max(sheenEnd, st.first ? settleEnd : 0)) { st.running = false; st.dirty = true; }
    }
    draw();
  }

  function finish() {
    st.t = Math.max(sheenEnd, settleEnd); st.running = false; st.dirty = true;
    if (!st.readied) { st.readied = true; host.ready(); }
    draw();
  }

  resize();
  if (opts.instant) finish();
  void el.offsetWidth;
  el.classList.add('in');

  return {
    el,
    get running() { return st.running; },
    frame,
    resize,
    theme,
    skip() { st.speed = 5; },
    finish,
    replay() {
      drawn = 0;
      for (const c of [mx, gx]) { c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, mat.width, mat.height); }
      Object.assign(st, { t: 0, speed: 1, first: false, running: true, dirty: true });
    },
    pointer: () => false,
    destroy() { el.remove(); },
  };
}
