import { FONT, clamp, lerp, easeOut } from './text.js';

const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const html = document.documentElement, body = document.body;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const fine = matchMedia('(hover: hover) and (pointer: fine)').matches;

const METHODS = {
  bioprint: () => import('./bioprint.js'),
  electrospin: () => import('./electrospin.js'),
  culture: () => import('./culture.js'),
};
const WEBGL2 = (() => { try { return !!document.createElement('canvas').getContext('webgl2'); } catch (_) { return false; } })();

/* ---------- host: what every method gets to know about the page ---------- */

const stage = $('#stage');
const host = {
  root: $('#scene'),
  vw: innerWidth, vh: innerHeight, dpr: 1,
  stage: { x: 0, y: 0, w: 1, h: 1 },
  lines: ['Diya Maisuria'],
  par: { x: 0, y: 0 },
  mouse: { x: -100, y: -100 },
  theme: null,
  reduce, fine,
  introBox() {
    const narrow = host.vw < 700;
    return { cx: host.vw / 2, cy: host.vh * .47, w: host.vw * (narrow ? .94 : .82), h: host.vh * .56 };
  },
  hud(text) { hudText = text; },
  ready() { reveal(); },
};

/* ---------- theme ---------- */

let current = null, currentName = null, loading = null;

const themeBtn = $('#theme');
function readTheme() {
  const cs = getComputedStyle(html), v = n => cs.getPropertyValue(n).trim();
  return { dark: html.dataset.theme === 'dark', bg: v('--bg'), ink: v('--ink'), dim: v('--dim'), accent: v('--accent') };
}
function applyTheme(t, persist) {
  html.dataset.theme = t;
  if (persist) try { localStorage.setItem('theme', t); } catch (_) { /* private mode */ }
  $('meta[name="theme-color"]').setAttribute('content', t === 'dark' ? '#100f0d' : '#f2efe9');
  themeBtn.setAttribute('aria-label', t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
  host.theme = readTheme();
  current?.theme();
}
themeBtn.addEventListener('click', () => {
  const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
  if (!document.startViewTransition || reduce) return applyTheme(next, true);
  const r = themeBtn.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const R = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  const vt = document.startViewTransition(() => applyTheme(next, true));
  vt.ready.then(() => html.animate(
    { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${R}px at ${x}px ${y}px)`] },
    { duration: 720, easing: 'cubic-bezier(.7, 0, .2, 1)', pseudoElement: '::view-transition-new(root)' },
  )).catch(() => {});
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
  let saved = null;
  try { saved = localStorage.getItem('theme'); } catch (_) { /* ignore */ }
  if (!saved) applyTheme(e.matches ? 'dark' : 'light', false);
});
applyTheme(html.dataset.theme || 'light', false);

/* ---------- methods ---------- */

const captionEl = $('#caption');

async function use(name, { first = false, instant = false } = {}) {
  if (!WEBGL2 && name !== 'electrospin') name = 'electrospin';
  const token = loading = {};
  let inst;
  try {
    const mod = await METHODS[name]();
    if (token !== loading) return;
    const old = current;
    current = null;
    if (old) { old.el.classList.remove('in'); setTimeout(() => old.destroy(), 500); }
    inst = await mod.create(host, { first, instant: instant || reduce });
    captionEl.textContent = mod.caption;
  } catch (err) {
    if (name !== 'electrospin') return use('electrospin', { first, instant });
    throw err;
  }
  if (token !== loading) { inst.destroy(); return; }
  current = inst; currentName = name;
  $$('[data-method]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.method === name)));
  const u = new URL(location.href);
  if (name === 'bioprint') u.searchParams.delete('method'); else u.searchParams.set('method', name);
  history.replaceState(null, '', u);
}

$$('[data-method]').forEach(b => b.addEventListener('click', () => {
  if (b.dataset.method === currentName) current?.replay();
  else use(b.dataset.method);
}));
$('#replay').addEventListener('click', () => current?.replay());

/* ---------- printer readout ---------- */

const hudEl = $('#hud-l');
let hudText = '', hudShown = '', hudAt = 0;
function hudStep(now) {
  if (hudText === hudShown || now - hudAt < 80) return;
  hudShown = hudText; hudAt = now;
  hudEl.textContent = hudText;
}
$('#hud-r').textContent = fine ? 'Click to fast-forward' : 'Tap to fast-forward';

/* ---------- reveal ---------- */

let revealed = false;
function reveal() {
  if (revealed) return;
  revealed = true;
  body.classList.add('ready');
  const el = $('#gpa'), target = 4;
  if (reduce) { el.textContent = target.toFixed(1); return; }
  const t0 = performance.now() + 450;
  const tick = now => {
    const k = clamp((now - t0) / 1100);
    el.textContent = (easeOut(k) * target).toFixed(1);
    if (k < 1) requestAnimationFrame(tick);
  };
  el.textContent = '0.0';
  requestAnimationFrame(tick);
}

/* ---------- neurons on click ---------- */

const fx = $('#fx'), fctx = fx.getContext('2d');
const neurons = [];
const GROW = .26;                                   // px per ms
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
    let px = x0, py = y0, a = ang;
    const steps = Math.max(2, Math.round(len / 4));
    for (let i = 1; i <= steps; i++) {
      if (aim) {
        if (Math.hypot(aim.x - px, aim.y - py) < 8) break;
        let da = Math.atan2(aim.y - py, aim.x - px) - a;
        da = Math.atan2(Math.sin(da), Math.cos(da));
        a += da * .2 + (Math.random() - .5) * .25;
      } else a += (Math.random() - .5) * .42;
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
  for (let k = 0; k < n; k++) grow(x, y, base + k * 2 * Math.PI / n + (Math.random() - .5) * .7, 16 + Math.random() * 34, 1.3, 70, 0);
  const axon = grow(x, y, target ? Math.atan2(target.y - y, target.x - x) + (Math.random() - .5) * .8 : Math.random() * Math.PI * 2,
    target ? best + 60 : 90 + Math.random() * 70, 1.1, 70, 2, target);
  if (!target) {
    const q = axon.p, ex = q[q.length - 2], ey = q[q.length - 1], ea = Math.atan2(ey - q[q.length - 3], ex - q[q.length - 4]);
    for (let k = -1; k <= 1; k++) grow(ex, ey, ea + k * .6, 10 + Math.random() * 12, .8, axon.t1, 3);
  }
  const cum = [0];
  for (let i = 2; i < axon.p.length; i += 2) cum.push(cum[cum.length - 1] + Math.hypot(axon.p[i] - axon.p[i - 2], axon.p[i + 1] - axon.p[i - 1]));
  const grown = Math.max(...branches.map(b => b.t1));
  neurons.push({ x, y, born: now, last: now + grown, branches, axon, cum, target, fireAt: now + axon.t1 + 60, fired: 0, flash: 0 });
  if (neurons.length > 14) neurons.shift();
}

let fxLive = false;
function drawNeurons(now) {
  if (!neurons.length) {
    if (fxLive) { fctx.setTransform(1, 0, 0, 1, 0, 0); fctx.clearRect(0, 0, fx.width, fx.height); fxLive = false; }
    return;
  }
  fxLive = true;
  const th = host.theme, dpr = host.dpr;
  fctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fctx.clearRect(0, 0, host.vw, host.vh);
  fctx.lineCap = 'round'; fctx.lineJoin = 'round';
  for (let i = neurons.length - 1; i >= 0; i--) {
    const n = neurons[i], age = now - n.born;
    const apDur = n.cum[n.cum.length - 1] / .75;
    if (now >= n.fireAt && n.fired !== n.fireAt && now - n.fireAt >= apDur) {
      n.fired = n.fireAt;
      const t = n.target;
      if (t) {
        t.flash = now; t.last = Math.max(t.last, now);
        if (t.target && neurons.includes(t.target)) t.fireAt = now + 40;
      }
    }
    const a = alphaOf(n, now);
    if (a <= 0) { neurons.splice(i, 1); continue; }
    fctx.globalAlpha = a;

    if (age < 650) {
      const k = age / 650;
      fctx.beginPath(); fctx.arc(n.x, n.y, 4 + 22 * easeOut(k), 0, Math.PI * 2);
      fctx.strokeStyle = th.accent; fctx.globalAlpha = a * .55 * (1 - k); fctx.lineWidth = 1; fctx.stroke();
      fctx.globalAlpha = a;
    }

    fctx.strokeStyle = th.ink;
    for (const b of n.branches) {
      const vis = (age - b.t0) * GROW / 4;
      if (vis <= 0) continue;
      const cnt = Math.min(b.p.length / 2 - 1, vis), whole = Math.floor(cnt), p = b.p;
      // taper: thicker near the soma, finer toward the growth cone
      for (let seg = 0; seg < 3; seg++) {
        const s0 = Math.floor(whole * seg / 3), s1 = seg === 2 ? whole : Math.floor(whole * (seg + 1) / 3);
        if (s1 <= s0 && seg < 2) continue;
        fctx.beginPath(); fctx.moveTo(p[s0 * 2], p[s0 * 2 + 1]);
        for (let j = s0 + 1; j <= s1; j++) fctx.lineTo(p[j * 2], p[j * 2 + 1]);
        if (seg === 2 && whole < cnt) {
          const f = cnt - whole;
          fctx.lineTo(lerp(p[2 * whole], p[2 * whole + 2], f), lerp(p[2 * whole + 1], p[2 * whole + 3], f));
        }
        fctx.lineWidth = b.w * (1 - seg * .25);
        fctx.globalAlpha = a * (th.dark ? .85 : .62);
        fctx.stroke();
      }
    }
    fctx.globalAlpha = a;

    if (n.target && age > n.axon.t1) {
      const p = n.axon.p;
      fctx.beginPath(); fctx.arc(p[p.length - 2], p[p.length - 1], 2.2, 0, Math.PI * 2);
      fctx.strokeStyle = th.accent; fctx.lineWidth = 1.1; fctx.stroke();
    }
    const fl = n.flash ? clamp(1 - (now - n.flash) / 500) : 0;
    if (fl > 0) {
      fctx.beginPath(); fctx.arc(n.x, n.y, 4 + (1 - fl) * 16, 0, Math.PI * 2);
      fctx.globalAlpha = a * fl * .6; fctx.strokeStyle = th.accent; fctx.lineWidth = 1; fctx.stroke();
      fctx.globalAlpha = a;
    }
    fctx.beginPath(); fctx.arc(n.x, n.y, 3.5 * easeOut(clamp(age / 160)) + fl * 1.6, 0, Math.PI * 2);
    fctx.fillStyle = th.accent; fctx.fill();

    const apK = (now - n.fireAt) / apDur;
    if (now >= n.fireAt && apK < 1) {
      const d = apK * n.cum[n.cum.length - 1];
      let j = 1;
      while (j < n.cum.length - 1 && n.cum[j] < d) j++;
      const f = clamp((d - n.cum[j - 1]) / (n.cum[j] - n.cum[j - 1] || 1)), p = n.axon.p;
      const x = lerp(p[2 * j - 2], p[2 * j], f), y = lerp(p[2 * j - 1], p[2 * j + 1], f);
      fctx.save();
      fctx.shadowColor = th.accent; fctx.shadowBlur = th.dark ? 16 : 10;
      fctx.beginPath(); fctx.arc(x, y, 2.5, 0, Math.PI * 2); fctx.fillStyle = th.accent; fctx.fill();
      fctx.restore();
    }
  }
  fctx.globalAlpha = 1;
}

/* ---------- cursor: reticle, coordinates, lagging gantry rails ---------- */

const cur = $('#cursor'), coords = $('#coords'), railX = $('#rail-x'), railY = $('#rail-y');
const target = { x: -100, y: -100 }, rail = { x: -100, y: -100 };
let pointerDirty = false, linkHover = false;
const mm = v => (v / 3.78).toFixed(1).padStart(5, '0');

if (fine) {
  body.classList.add('has-cursor');
  addEventListener('pointermove', e => {
    if (e.pointerType !== 'mouse') return;
    if (!body.classList.contains('cursor-on')) { rail.x = e.clientX; rail.y = e.clientY; }
    target.x = e.clientX; target.y = e.clientY;
    linkHover = !!e.target.closest('a, button');
    body.classList.add('cursor-on');
    pointerDirty = true;
  }, { passive: true });
  html.addEventListener('mouseleave', () => { body.classList.remove('cursor-on'); host.mouse.x = host.mouse.y = -100; });
}

function cursorStep(dt) {
  if (!fine) return;
  if (pointerDirty) {
    pointerDirty = false;
    host.mouse.x = target.x; host.mouse.y = target.y;
    cur.style.transform = `translate3d(${target.x}px,${target.y}px,0)`;
    cur.classList.toggle('is-link', linkHover);
    coords.textContent = `X ${mm(target.x)}  Y ${mm(target.y)}`;
  }
  const k = 1 - Math.exp(-dt * 9);
  rail.x += (target.x - rail.x) * k; rail.y += (target.y - rail.y) * k;
  railX.style.transform = `translate3d(0,${rail.y}px,0)`;
  railY.style.transform = `translate3d(${rail.x}px,0,0)`;
  if (!reduce && target.x > -1) {
    const p = 1 - Math.exp(-dt * 3.5);
    host.par.x += (clamp(target.x / host.vw * 2 - 1, -1, 1) - host.par.x) * p;
    host.par.y += (clamp(target.y / host.vh * 2 - 1, -1, 1) - host.par.y) * p;
  }
}

// small magnetic pull on icons and buttons
$$('[data-magnet]').forEach(el => {
  el.addEventListener('pointermove', e => {
    const r = el.getBoundingClientRect();
    el.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * .25}px, ${(e.clientY - r.top - r.height / 2) * .25}px)`;
  });
  el.addEventListener('pointerleave', () => { el.style.transform = ''; });
});

addEventListener('pointerdown', e => {
  if (e.button > 0) return;
  cur.classList.add('is-down');
  if (e.target.closest('a, button')) return;
  if (current?.running) { current.skip(); return; }
  if (current?.pointer(e.clientX, e.clientY)) return;
  if (!reduce) spawn(e.clientX, e.clientY);
});
addEventListener('pointerup', () => cur.classList.remove('is-down'));
addEventListener('keydown', e => {
  if (current?.running && !e.metaKey && !e.ctrlKey && !e.altKey && e.key !== 'Tab' && e.key !== 'Shift') current.skip();
});

/* ---------- socials: links without an address yet stay visible but inert ---------- */

$$('.social a').forEach(a => {
  const href = a.getAttribute('href') || '';
  if (href && href !== 'mailto:') return;
  a.removeAttribute('href');
  a.setAttribute('aria-disabled', 'true');
  a.classList.add('pending');
  a.dataset.label += ' · soon';
});

/* ---------- clock ---------- */

const clockEl = $('#clock');
const tf = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Detroit', hour: '2-digit', minute: '2-digit', hourCycle: 'h23', timeZoneName: 'short' });
const tick = () => { clockEl.textContent = tf.format(new Date()); };
tick(); setInterval(tick, 15000);

/* ---------- layout ---------- */

const wantSplit = () => host.vw / host.vh < .8 && host.vw < 700;
function measure() {
  const r = stage.getBoundingClientRect();
  host.stage = { x: r.left, y: r.top, w: r.width, h: r.height };
}
function resize() {
  host.vw = innerWidth; host.vh = innerHeight; host.dpr = Math.min(2, devicePixelRatio || 1);
  fx.width = Math.round(host.vw * host.dpr); fx.height = Math.round(host.vh * host.dpr);
  measure();
  const lines = wantSplit() ? ['Diya', 'Maisuria'] : ['Diya Maisuria'];
  if (lines.length !== host.lines.length) {
    host.lines = lines;
    if (current) use(currentName, { instant: true });
  } else current?.resize();
}
addEventListener('resize', resize);
addEventListener('scroll', measure, { passive: true });
new ResizeObserver(measure).observe(stage);

/* ---------- loop ---------- */

let prev = performance.now();
function frame(now) {
  const dt = Math.min(.05, (now - prev) / 1000);
  prev = now;
  cursorStep(dt);
  current?.frame(dt);
  hudStep(now);
  drawNeurons(now);
  requestAnimationFrame(frame);
}

(async () => {
  resize();
  host.lines = wantSplit() ? ['Diya', 'Maisuria'] : ['Diya Maisuria'];
  requestAnimationFrame(frame);
  try { await Promise.race([document.fonts.load(FONT), new Promise(r => setTimeout(r, 2500))]); } catch (_) { /* use fallback */ }
  body.classList.add('booted');
  const q = new URLSearchParams(location.search).get('method');
  await use(METHODS[q] ? q : 'bioprint', { first: true });
})();
