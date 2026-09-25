// Method 1: extrusion bioprinting, rendered as real lit strands on the GPU.
// The name is sliced like a slicer would (walls + 0/90° lattice), every strand
// segment becomes one instanced capsule, and a single time uniform grows them.

import * as THREE from 'three';
import { rasterize, fit, clamp, lerp, ease, DEG } from './text.js';

export const caption = 'Name, bioprinted. Six layers of 0/90° lattice through a 22G needle, crosslinked at 405 nm.';

const R = 2.2;             // strand radius (mask px)
const SQUASH = .78;        // strands slump a little as they land
const GAP = 9;             // lattice pitch
const LAYERS = 6;
const LAYER_H = 3.5;
const ERODE = 4;
const TRAVEL = 3;          // travel moves are this much faster than printing
const LAYER_T = [.8, .5, .42, .36, .32, .32];
const HOME = .55, LIFT = .35, CURE = .6, SETTLE = 1.1;
const TILT = 32 * DEG, FOV = 26;
const LAYER_MM = .32;

THREE.ColorManagement.enabled = false;

/* ---------- slicing ---------- */

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

// marching squares -> closed outlines as flat [x0,y0,x1,y1,...]
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
    loops.push(simplify(loop, .45).flat());
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

// rectilinear infill, zig-zagging per letter
function fills(e, lab, W, H, vertical, n) {
  const out = Array.from({ length: n }, () => []), flip = new Uint8Array(n);
  const lines = vertical ? W : H, len = vertical ? H : W;
  for (let u = Math.floor(((lines - 1) % GAP) / 2); u < lines; u += GAP) {
    const row = new Map();
    let s = -1;
    for (let t = 0; t <= len; t++) {
      const on = t < len && e[vertical ? t * W + u : u * W + t];
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
      for (const [r0, r1] of runs) out[id].push(vertical ? [u, r0, u, r1] : [r0, u, r1, u]);
    }
  }
  return out;
}

function polyLen(p) {
  let l = 0;
  for (let i = 2; i < p.length; i += 2) l += Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
  return l;
}

// Every strand segment with the time it starts being laid down and how long it takes.
function toolpath(mask) {
  const { W, H, a, b, lineH } = mask;
  const { lab, comps } = label(b, W, H);
  const e = erode(b, W, H, ERODE);
  const walls = comps.map(() => []);
  for (const loop of contours(a, W, H, 2)) {
    const id = labelNear(lab, W, H, loop[0], loop[1]);
    if (id >= 0) walls[id].push(loop);
  }
  const fill = [fills(e, lab, W, H, false, comps.length), fills(e, lab, W, H, true, comps.length)];
  const order = comps.map((_, i) => i).sort((p, q) =>
    (Math.floor(comps[p].minY / lineH) - Math.floor(comps[q].minY / lineH)) || comps[p].minX - comps[q].minX);

  const segs = [];
  let t = HOME, lx = null, ly = null;
  for (let L = 0; L < LAYERS; L++) {
    const polys = [];
    for (const id of order) polys.push(...walls[id], ...fill[L % 2][id]);
    let cost = 0, px = lx, py = ly;
    for (const p of polys) {
      if (px !== null) cost += Math.hypot(p[0] - px, p[1] - py) / TRAVEL;
      cost += polyLen(p);
      px = p[p.length - 2]; py = p[p.length - 1];
    }
    const k = LAYER_T[L] / cost;
    let c = 0;
    for (const p of polys) {
      if (lx !== null) c += Math.hypot(p[0] - lx, p[1] - ly) / TRAVEL;
      for (let i = 2; i < p.length; i += 2) {
        const l = Math.hypot(p[i] - p[i - 2], p[i + 1] - p[i - 1]);
        if (l < 1e-3) continue;
        segs.push(p[i - 2] - W / 2, H / 2 - p[i - 1], p[i] - W / 2, H / 2 - p[i + 1], L, t + c * k, l * k);
        c += l;
      }
      lx = p[p.length - 2]; ly = p[p.length - 1];
    }
    t += LAYER_T[L];
  }
  return { segs: new Float32Array(segs), n: segs.length / 7, printEnd: t };
}

/* ---------- geometry + shaders ---------- */

// Unit capsule along +x: vertices with aEnd = 1 get pushed out by the segment length.
function capsule(radial = 8, rings = 3) {
  const pos = [], end = [], idx = [], th = [];
  for (let i = 0; i <= rings; i++) th.push([-Math.PI / 2 + Math.PI / 2 * i / rings, 0]);
  for (let i = 0; i <= rings; i++) th.push([Math.PI / 2 * i / rings, 1]);
  for (const [t, e] of th)
    for (let j = 0; j <= radial; j++) {
      const ph = j / radial * Math.PI * 2;
      pos.push(Math.sin(t), Math.cos(t) * Math.cos(ph), Math.cos(t) * Math.sin(ph));
      end.push(e);
    }
  const row = radial + 1;
  for (let i = 0; i < th.length - 1; i++)
    for (let j = 0; j < radial; j++) {
      const a = i * row + j, b = a + row;
      idx.push(a, a + 1, b, b, a + 1, b + 1);
    }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aEnd', new THREE.Float32BufferAttribute(end, 1));
  g.setIndex(idx);
  return g;
}

const strandVert = /* glsl */`
  attribute float aEnd;
  attribute vec4 aA;   // x0, y0, z, angle
  attribute vec4 aB;   // length, birth, duration, layer
  uniform float uTime, uR, uSquash;
  varying vec3 vN, vP;
  varying float vLayer, vAge;
  void main() {
    float g = clamp((uTime - aB.y) / max(aB.z, 1e-4), 0., 1.);
    vec3 p = position * uR;
    p.z *= uSquash;
    p.x += aEnd * aB.x * g;
    float c = cos(aA.w), s = sin(aA.w);
    vec3 w = vec3(aA.x + p.x * c - p.y * s, aA.y + p.x * s + p.y * c, aA.z + p.z);
    vec3 n = vec3(normal.xy, normal.z / uSquash);
    vN = normalize(vec3(n.x * c - n.y * s, n.x * s + n.y * c, n.z));
    vP = w;
    vLayer = aB.w;
    vAge = uTime - aB.y - aB.z;
    gl_Position = g > 0. ? projectionMatrix * modelViewMatrix * vec4(w, 1.) : vec4(2., 2., 2., 1.);
  }`;

const strandFrag = /* glsl */`
  uniform vec3 uCam, uLight, uRaw0, uRaw1, uCur0, uCur1, uRim, uUV;
  uniform float uCure, uCureOn, uLayers, uRimK;
  varying vec3 vN, vP;
  varying float vLayer, vAge;
  void main() {
    vec3 N = normalize(vN), V = normalize(uCam - vP), L = normalize(uLight);
    float f = vLayer / (uLayers - 1.);
    float cured = 1. - smoothstep(uCure - 10., uCure + 10., vP.x);
    vec3 base = mix(mix(uRaw0, uRaw1, f), mix(uCur0, uCur1, f), cured);
    float wet = exp(-max(vAge, 0.) * 5.);
    float dif = max(dot(N, L), 0.) * .6 + .4;
    float occ = mix(.62, 1., smoothstep(-.7, .5, N.z));
    float spec = pow(max(dot(N, normalize(L + V)), 0.), mix(24., 64., cured)) * (mix(.28, .5, cured) + wet * .7);
    float fres = pow(1. - max(dot(N, V), 0.), 3.);
    vec3 col = base * dif * occ + spec + fres * uRim * uRimK;
    col += uUV * exp(-pow((vP.x - uCure) / 16., 2.)) * uCureOn;
    gl_FragColor = vec4(col, 1.);
  }`;

const plateVert = /* glsl */`
  varying vec2 vP;
  void main() { vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`;

const plateFrag = /* glsl */`
  uniform vec3 uInk;
  uniform float uAlpha;
  uniform vec2 uSize;
  varying vec2 vP;
  void main() {
    vec2 q = vP / 40.;
    vec2 g = abs(fract(q - .5) - .5) / fwidth(q);
    float line = 1. - min(min(g.x, g.y), 1.);
    float fade = 1. - smoothstep(.3, 1., length(vP / (uSize * vec2(.75, 1.5))));
    gl_FragColor = vec4(uInk, line * fade * uAlpha);
  }`;

/* ---------- the concept ---------- */

export async function create(host, opts) {
  const mask = rasterize(host.lines);
  const { W, H } = mask;
  const Z = LAYERS * LAYER_H;
  const path = toolpath(mask);
  const { segs, n, printEnd } = path;
  const cureStart = printEnd + LIFT, cureEnd = cureStart + CURE, settleEnd = cureEnd + SETTLE;

  const el = document.createElement('div');
  el.className = 'layer';
  const gl = document.createElement('canvas'), ov = document.createElement('canvas');
  el.append(gl, ov);
  host.root.append(el);
  const octx = ov.getContext('2d');

  const renderer = new THREE.WebGLRenderer({ canvas: gl, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 1, 10);
  const root = new THREE.Group(), tiltG = new THREE.Group(), yawG = new THREE.Group(), content = new THREE.Group();
  scene.add(root); root.add(tiltG); tiltG.add(yawG); yawG.add(content);
  content.position.z = -Z / 2;

  // strands
  const geo = capsule();
  const A = new Float32Array(n * 4), B = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const o = i * 7, x0 = segs[o], y0 = segs[o + 1], x1 = segs[o + 2], y1 = segs[o + 3], L = segs[o + 4];
    A.set([x0, y0, L * LAYER_H + R * SQUASH, Math.atan2(y1 - y0, x1 - x0)], i * 4);
    B.set([Math.hypot(x1 - x0, y1 - y0), segs[o + 5], segs[o + 6], L], i * 4);
  }
  geo.setAttribute('aA', new THREE.InstancedBufferAttribute(A, 4));
  geo.setAttribute('aB', new THREE.InstancedBufferAttribute(B, 4));
  geo.instanceCount = n;
  const U = {
    uTime: { value: 0 }, uR: { value: R }, uSquash: { value: SQUASH },
    uCam: { value: new THREE.Vector3() }, uLight: { value: new THREE.Vector3(-.45, .55, 1).normalize() },
    uRaw0: { value: new THREE.Color() }, uRaw1: { value: new THREE.Color() },
    uCur0: { value: new THREE.Color() }, uCur1: { value: new THREE.Color() },
    uRim: { value: new THREE.Color() }, uUV: { value: new THREE.Color() }, uRimK: { value: .3 },
    uCure: { value: -1e4 }, uCureOn: { value: 0 }, uLayers: { value: LAYERS },
  };
  const strands = new THREE.Mesh(geo, new THREE.ShaderMaterial({ vertexShader: strandVert, fragmentShader: strandFrag, uniforms: U }));
  strands.frustumCulled = false;
  content.add(strands);

  // build plate grid
  const plateU = { uInk: { value: new THREE.Color() }, uAlpha: { value: .1 }, uSize: { value: new THREE.Vector2(W, H) } };
  const plate = new THREE.Mesh(new THREE.PlaneGeometry(W * 1.9, H * 3.4),
    new THREE.ShaderMaterial({ vertexShader: plateVert, fragmentShader: plateFrag, uniforms: plateU, transparent: true, depthWrite: false }));
  plate.position.z = -.3;
  content.add(plate);

  // soft contact shadow baked from the mask
  const sp = 44, sh = document.createElement('canvas');
  sh.width = W + sp * 2; sh.height = H + sp * 2;
  const shx = sh.getContext('2d');
  shx.shadowColor = '#000'; shx.shadowBlur = 20; shx.shadowOffsetX = 10000;
  shx.drawImage(mask.canvas, sp - 10000, sp);
  const shadowMat = new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sh), transparent: true, depthWrite: false, opacity: 0, color: 0x000000 });
  shadowMat.map.colorSpace = THREE.NoColorSpace;
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(sh.width, sh.height), shadowMat);
  shadow.position.set(5, -12, -.1);
  content.add(shadow);

  /* timeline */
  const st = { t: 0, speed: 1, first: !!opts.first, running: true, readied: false, dirty: true, last: null };

  function nozzle(t) {
    let lo = 0, hi = n - 1;
    if (t <= B[1]) return [A[0], A[1], A[2]];
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (B[m * 4 + 1] <= t) lo = m; else hi = m - 1; }
    const i = lo, a = i * 4, g = (t - B[a + 1]) / B[a + 2], len = B[a], ang = A[a + 3];
    const ex = A[a] + Math.cos(ang) * len, ey = A[a + 1] + Math.sin(ang) * len;
    if (g <= 1) return [A[a] + Math.cos(ang) * len * g, A[a + 1] + Math.sin(ang) * len * g, A[a + 2]];
    if (i + 1 >= n) return [ex, ey, A[a + 2]];
    const b = a + 4, t0 = B[a + 1] + B[a + 2], t1 = B[b + 1];
    const k = t1 > t0 ? clamp((t - t0) / (t1 - t0)) : 1;
    return [lerp(ex, A[b], k), lerp(ey, A[b + 1], k), Math.max(A[a + 2], A[b + 2])];
  }

  const introPose = t => {
    const box = host.introBox();
    const narrow = host.vw < 700;
    const yaw = lerp(narrow ? 12 : 26, narrow ? 4 : 10, clamp(t / (printEnd + 1))) * DEG, tilt = (narrow ? 52 : 58) * DEG;
    return { x: box.cx, y: box.cy, s: fit(W, H, Z, yaw, tilt, box.w, box.h) * .92, yaw, tilt };
  };
  const restPose = () => {
    const r = host.stage;
    return { x: r.x + r.w / 2, y: r.y + r.h / 2, s: fit(W, H, Z, 0, TILT, r.w, r.h) * .95, yaw: host.par.x * 7 * DEG, tilt: TILT + host.par.y * 5 * DEG };
  };
  function pose() {
    const t = st.t;
    if (!st.first || t >= settleEnd) return restPose();
    const a = introPose(t);
    if (t < cureEnd) return a;
    const b = restPose(), k = ease((t - cureEnd) / SETTLE);
    return { x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k), s: lerp(a.s, b.s, k), yaw: lerp(a.yaw, b.yaw, k), tilt: lerp(a.tilt, b.tilt, k) };
  }

  const toScreen = (x, y, z) => {
    const v = content.localToWorld(new THREE.Vector3(x, y, z)).project(camera);
    return [(v.x + 1) / 2 * host.vw, (1 - v.y) / 2 * host.vh];
  };

  function resize() {
    const { vw, vh, dpr } = host;
    renderer.setPixelRatio(dpr);
    renderer.setSize(vw, vh, false);
    ov.width = Math.round(vw * dpr); ov.height = Math.round(vh * dpr);
    const D = (vh / 2) / Math.tan(FOV * DEG / 2);
    camera.aspect = vw / vh; camera.near = D * .2; camera.far = D * 5;
    camera.position.set(0, 0, D);
    camera.updateProjectionMatrix();
    st.dirty = true;
  }

  function theme() {
    const th = host.theme, dark = th.dark;
    const set = (c, hex) => c.set(hex);
    if (dark) {
      set(U.uRaw0.value, '#6b3347'); set(U.uRaw1.value, '#b86f88');
      set(U.uCur0.value, '#8c1d42'); set(U.uCur1.value, '#ff4f82');
      set(U.uRim.value, '#ffb0c6'); U.uRimK.value = .55;
      set(U.uUV.value, '#8f7bff');
      plateU.uInk.value.set(th.ink); shadowMat.userData.k = .55;
    } else {
      set(U.uRaw0.value, '#e39cb1'); set(U.uRaw1.value, '#f6c9d5');
      set(U.uCur0.value, '#8a1c43'); set(U.uCur1.value, '#d4436e');
      set(U.uRim.value, '#ffffff'); U.uRimK.value = .3;
      set(U.uUV.value, '#6e52ff');
      plateU.uInk.value.set(th.ink); shadowMat.userData.k = .2;
    }
    st.dirty = true;
    draw();
  }

  function syringe(x, y, ink, alpha) {
    const th = host.theme, k = clamp(Math.min(host.vh / 900, host.vw / 760), .5, 1.1), dpr = host.dpr;
    const c = octx;
    c.save();
    c.globalAlpha = alpha;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    const top = y - 174 * k, ax = host.vw * .5 + (x - host.vw * .5) * .25;
    c.beginPath();
    c.moveTo(x, top);
    c.bezierCurveTo(x, top - 150 * k, ax, -60 + 120 * k, ax, -20);
    c.strokeStyle = th.dim; c.lineWidth = 1.6; c.stroke();
    c.setTransform(dpr * k, 0, 0, dpr * k, x * dpr, y * dpr);
    c.strokeStyle = th.ink; c.lineJoin = 'round';
    c.beginPath(); c.moveTo(0, 0); c.lineTo(0, -34); c.lineWidth = 1.4; c.stroke();
    c.beginPath(); c.moveTo(-2, -34); c.lineTo(2, -34); c.lineTo(6, -48); c.lineTo(-6, -48); c.closePath();
    c.fillStyle = th.accent; c.fill(); c.lineWidth = 1.1; c.stroke();
    c.fillStyle = th.bg; c.globalAlpha = alpha * .92;
    c.fillRect(-11, -166, 22, 118);
    c.globalAlpha = alpha;
    c.strokeRect(-11, -166, 22, 118);
    const h = 104 * ink;
    c.fillStyle = '#' + U.uRaw1.value.getHexString(); c.fillRect(-9.4, -50 - h, 18.8, h);
    c.fillStyle = th.ink; c.fillRect(-10.4, -54 - h, 20.8, 4);
    c.beginPath();
    for (let g = 0; g < 9; g++) { const gy = -60 - g * 12; c.moveTo(11, gy); c.lineTo(g % 2 ? 8 : 5, gy); }
    c.lineWidth = .8; c.stroke();
    c.fillRect(-13.5, -174, 27, 8);
    c.restore();
  }

  function overlay(t) {
    const dpr = host.dpr;
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.clearRect(0, 0, ov.width, ov.height);
    if (t < printEnd + LIFT) {
      let p, alpha = 1;
      if (t < HOME) {
        const q = nozzle(HOME), k = ease(t / HOME);
        p = [lerp(0, q[0], k), lerp(0, q[1], k), lerp(Z + 300, q[2], k)];
      } else if (t < printEnd) p = nozzle(t);
      else {
        const k = (t - printEnd) / LIFT;
        p = nozzle(printEnd); p[2] += ease(k) * 200; alpha = 1 - k;
      }
      const [sx, sy] = toScreen(p[0], p[1], p[2] + R * SQUASH);
      const ink = 1 - .8 * clamp((t - HOME) / (printEnd - HOME));
      syringe(sx, sy, ink, alpha);
    }
    if (t >= cureStart && t < cureEnd) {
      const x = U.uCure.value;
      const [ax, ay] = toScreen(x, -H / 2 - 30, 0), [bx, by] = toScreen(x, H / 2 + 30, 0);
      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      octx.strokeStyle = '#' + U.uUV.value.getHexString();
      octx.shadowColor = octx.strokeStyle; octx.shadowBlur = 12;
      octx.lineWidth = 1.5;
      octx.beginPath(); octx.moveTo(ax, ay); octx.lineTo(bx, by); octx.stroke();
      octx.shadowBlur = 0;
      octx.fillStyle = octx.strokeStyle;
      octx.font = '500 10px "IBM Plex Mono", monospace';
      octx.fillText('405 NM', bx + 6, by + 4);
    }
  }

  function hud(t) {
    const kpa = (24 + Math.sin(t * 9) * .5 + Math.sin(t * 23) * .2).toFixed(1);
    if (t < HOME) return host.hud('Homing · bed 37.0 °C · 22G needle');
    if (t < printEnd) {
      let L = 0, acc = HOME;
      while (L < LAYERS - 1 && t >= acc + LAYER_T[L]) acc += LAYER_T[L++];
      return host.hud(`Layer ${L + 1}/${LAYERS} · Z ${((L + 1) * LAYER_MM).toFixed(2)} mm · ${kpa} kPa · ${L % 2 ? 90 : 0}° pass`);
    }
    if (t < cureStart) return host.hud(`Print done · Z ${(LAYERS * LAYER_MM).toFixed(2)} mm`);
    host.hud(`Crosslinking · 405 nm · ${Math.round(clamp((t - cureStart) / CURE) * 30)} s`);
  }

  function draw() {
    const t = st.t, p = pose();
    const l = st.last;
    if (!st.running && !st.dirty && l && Math.abs(l.x - p.x) + Math.abs(l.y - p.y) + Math.abs(l.s - p.s) * 100 + Math.abs(l.yaw - p.yaw) * 100 + Math.abs(l.tilt - p.tilt) * 100 < .01) return;
    st.last = p; st.dirty = false;

    root.position.set(p.x - host.vw / 2, host.vh / 2 - p.y, 0);
    root.scale.setScalar(p.s);
    tiltG.rotation.x = -p.tilt;
    yawG.rotation.z = p.yaw;
    scene.updateMatrixWorld();
    U.uCam.value.copy(content.worldToLocal(camera.position.clone()));

    U.uTime.value = t;
    if (t < cureStart) { U.uCure.value = -1e4; U.uCureOn.value = 0; }
    else if (t < cureEnd) { U.uCure.value = lerp(-W / 2 - 40, W / 2 + 40, ease((t - cureStart) / CURE)); U.uCureOn.value = 1; }
    else { U.uCure.value = 1e4; U.uCureOn.value = 0; }
    shadowMat.opacity = (shadowMat.userData.k || .2) * clamp((t - HOME) / (printEnd - HOME));
    plateU.uAlpha.value = st.first && t < settleEnd ? lerp(.11, .06, clamp((t - cureEnd) / SETTLE)) : .06;

    renderer.render(scene, camera);
    overlay(t);
  }

  function frame(dt) {
    if (st.running) {
      st.t += dt * st.speed;
      hud(st.t);
      if (!st.readied && st.t >= cureEnd) { st.readied = true; host.ready(); }
      if (st.t >= (st.first ? settleEnd : cureEnd)) { st.running = false; st.t = Math.max(st.t, settleEnd); st.dirty = true; }
    }
    draw();
  }

  function finish() {
    st.t = settleEnd; st.running = false; st.dirty = true;
    if (!st.readied) { st.readied = true; host.ready(); }
    draw();
  }

  resize();
  theme();
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
    replay() { Object.assign(st, { t: 0, speed: 1, first: false, running: true, dirty: true }); },
    pointer: () => false,
    destroy() {
      geo.dispose(); strands.material.dispose(); plate.geometry.dispose(); plate.material.dispose();
      shadow.geometry.dispose(); shadowMat.map.dispose(); shadowMat.dispose();
      renderer.dispose(); el.remove();
    },
  };
}
