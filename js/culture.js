// Method 3: cell culture. A Gray-Scott reaction-diffusion system in its
// "mitosis" regime (spots that grow and split) runs on the GPU, confined to
// the letters. Seeded cells divide until the name is confluent, then a stain
// washes across: H&E in light mode, immunofluorescence in dark mode.

import * as THREE from 'three';
import { rasterize, fit, clamp, lerp, ease, easeOut, rng } from './text.js';

export const caption = 'Name, cultured. Cells seeded on the letters and grown to confluence, then stained: H&E by day, fluorescence by night.';

const GROW = 3.1, STAIN = .8, SETTLE = 1.0;
const F = .0367, K = .0649;           // mitosis regime
const ITER_GROW = 22, ITER_IDLE = 3;

THREE.ColorManagement.enabled = false;

const quadVert = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0., 1.); }`;

const simFrag = /* glsl */`
  uniform sampler2D uState, uMask;
  uniform vec2 uRes;
  uniform vec3 uSeed;      // x, y (sim px), radius; radius 0 = off
  uniform vec3 uHover;     // x, y (sim px), strength
  varying vec2 vUv;
  vec2 S(vec2 o) { return texture2D(uState, vUv + o / uRes).rg; }
  void main() {
    vec2 s = S(vec2(0.));
    vec2 lap = -s
      + .2 * (S(vec2(1., 0.)) + S(vec2(-1., 0.)) + S(vec2(0., 1.)) + S(vec2(0., -1.)))
      + .05 * (S(vec2(1., 1.)) + S(vec2(-1., 1.)) + S(vec2(1., -1.)) + S(vec2(-1., -1.)));
    float m = smoothstep(.35, .75, texture2D(uMask, vUv).a);
    vec2 px = vUv * uRes;
    float f = ${F.toFixed(4)} + uHover.z * .012 * exp(-dot(px - uHover.xy, px - uHover.xy) / 260.);
    float k = mix(.08, ${K.toFixed(4)}, m);
    float u = s.r, v = s.g, uvv = u * v * v;
    u += .62 * lap.r - uvv + f * (1. - u);
    v += .31 * lap.g + uvv - (f + k) * v;
    v *= mix(.6, 1., m);
    if (uSeed.z > 0. && distance(px, uSeed.xy) < uSeed.z) { u = .3; v = .45; }
    gl_FragColor = vec4(clamp(u, 0., 1.), clamp(v, 0., 1.), 0., 1.);
  }`;

const viewVert = /* glsl */`
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.); }`;

const viewFrag = /* glsl */`
  uniform sampler2D uState;
  uniform vec2 uRes, uBox;
  uniform float uStain, uField, uDark, uTime;
  uniform vec3 uBg, uPhase, uHalo, uCyto, uNuc, uEdge;
  varying vec2 vUv;
  float V(vec2 uv) {
    vec2 p = uv * uRes - .5, f = fract(p), i = (floor(p) + .5) / uRes, d = 1. / uRes;
    float a = texture2D(uState, i).g, b = texture2D(uState, i + vec2(d.x, 0.)).g;
    float c = texture2D(uState, i + vec2(0., d.y)).g, e = texture2D(uState, i + d).g;
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
  }
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3. - 2. * f);
    return mix(mix(hash(i), hash(i + vec2(1., 0.)), f.x), mix(hash(i + vec2(0., 1.)), hash(i + 1.), f.x), f.y);
  }
  void main() {
    vec2 d = 1.2 / uRes;
    float v = V(vUv);
    float gx = V(vUv + vec2(d.x, 0.)) - V(vUv - vec2(d.x, 0.));
    float gy = V(vUv + vec2(0., d.y)) - V(vUv - vec2(0., d.y));
    float cyto = smoothstep(.025, .09, v);
    float nuc = smoothstep(.25, .33, v);
    float rim = cyto * (1. - smoothstep(.09, .17, v));
    float halo = smoothstep(.008, .025, v) * (1. - cyto);
    vec3 n = normalize(vec3(-gx * 9., -gy * 9., 1.));
    float spec = pow(max(dot(n, normalize(vec3(-.45, .55, 1.))), 0.), 28.);

    // stain front: noisy wipe from left to right
    float front = uStain + (noise(vUv * vec2(9., 4.)) - .5) * .12;
    float stained = smoothstep(front + .03, front - .03, vUv.x);

    // unstained phase contrast: grey bodies with a bright halo
    vec3 phase = mix(uPhase, uPhase * .6, nuc);
    float aPhase = max(cyto * .8, halo * .55);
    vec3 cPhase = mix(uHalo, phase, cyto / max(aPhase, 1e-3) * .8);

    // stained
    vec3 body = mix(uCyto, uEdge, rim * .7);
    vec3 cSt = mix(body, uNuc, nuc);
    float aSt = cyto;
    if (uDark > .5) { cSt += uNuc * nuc * .35 + uCyto * cyto * .15; }

    vec3 col = mix(cPhase, cSt, stained) + spec * .22 * cyto;
    float a = mix(aPhase, aSt, stained);

    // microscope field of view
    vec2 p = (vUv - .5) * uBox;
    float r = length(p) / (.5 * uBox.x);
    a *= 1. - smoothstep(uField - .02, uField, r);
    gl_FragColor = vec4(col, a);
  }`;

export async function create(host, opts) {
  const mask = rasterize(host.lines);
  const { W, H } = mask;
  const simW = host.lines.length > 1 ? 720 : 1000, simH = Math.round(simW * H / W);
  const growEnd = GROW, stainEnd = GROW + STAIN, settleEnd = stainEnd + SETTLE;

  const el = document.createElement('div');
  el.className = 'layer';
  const gl = document.createElement('canvas'), ov = document.createElement('canvas');
  el.append(gl, ov);
  host.root.append(el);
  const octx = ov.getContext('2d');

  const renderer = new THREE.WebGLRenderer({ canvas: gl, antialias: false, alpha: true, powerPreference: 'high-performance' });
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
  renderer.setClearColor(0x000000, 0);
  if (!renderer.capabilities.isWebGL2) throw new Error('webgl2 needed');
  const type = renderer.extensions.has('EXT_color_buffer_float') ? THREE.FloatType : THREE.HalfFloatType;
  const rtOpts = { type, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: false, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping };
  let rtA = new THREE.WebGLRenderTarget(simW, simH, rtOpts), rtB = new THREE.WebGLRenderTarget(simW, simH, rtOpts);

  // letters as a texture, sized to the sim grid
  const mc = document.createElement('canvas');
  mc.width = simW; mc.height = simH;
  mc.getContext('2d').drawImage(mask.canvas, 0, 0, simW, simH);
  const maskTex = new THREE.CanvasTexture(mc);
  maskTex.colorSpace = THREE.NoColorSpace;

  // seed a sparse monolayer inside the letters
  const init = new Float32Array(simW * simH * 4);
  for (let i = 0; i < simW * simH; i++) init[i * 4] = 1;
  const rand = rng(7);
  const inside = (x, y) => mask.b[Math.floor((1 - y / simH) * H) * W + Math.floor(x / simW * W)];
  let seeds = 0;
  for (let n = 0; n < 40000 && seeds < simW * simH / 700; n++) {
    const x = rand() * simW, y = rand() * simH;
    if (!inside(x, y)) continue;
    seeds++;
    for (let j = -3; j <= 3; j++)
      for (let i = -3; i <= 3; i++) {
        if (i * i + j * j > 9) continue;
        const xx = Math.floor(x) + i, yy = Math.floor(y) + j;
        if (xx < 0 || yy < 0 || xx >= simW || yy >= simH) continue;
        const o = (yy * simW + xx) * 4;
        init[o] = .3; init[o + 1] = .45 + rand() * .05;
      }
  }
  const initTex = new THREE.DataTexture(init, simW, simH, THREE.RGBAFormat, THREE.FloatType);
  initTex.needsUpdate = true;

  const quad = new THREE.PlaneGeometry(2, 2);
  const simCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const simU = {
    uState: { value: initTex }, uMask: { value: maskTex }, uRes: { value: new THREE.Vector2(simW, simH) },
    uSeed: { value: new THREE.Vector3() }, uHover: { value: new THREE.Vector3() },
  };
  const simScene = new THREE.Scene();
  simScene.add(new THREE.Mesh(quad, new THREE.ShaderMaterial({ vertexShader: quadVert, fragmentShader: simFrag, uniforms: simU })));

  const viewU = {
    uState: { value: rtA.texture }, uRes: { value: new THREE.Vector2(simW, simH) }, uBox: { value: new THREE.Vector2(W, H) },
    uStain: { value: -.2 }, uField: { value: .2 }, uDark: { value: 0 }, uTime: { value: 0 },
    uBg: { value: new THREE.Color() }, uPhase: { value: new THREE.Color() }, uHalo: { value: new THREE.Color() },
    uCyto: { value: new THREE.Color() }, uNuc: { value: new THREE.Color() }, uEdge: { value: new THREE.Color() },
  };
  const view = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
    new THREE.ShaderMaterial({ vertexShader: viewVert, fragmentShader: viewFrag, uniforms: viewU, transparent: true, depthTest: false }));
  const viewScene = new THREE.Scene();
  viewScene.add(view);
  const viewCam = new THREE.OrthographicCamera(0, 1, 0, -1, -1, 1);

  function step(iters) {
    for (let i = 0; i < iters; i++) {
      renderer.setRenderTarget(rtB);
      renderer.render(simScene, simCam);
      [rtA, rtB] = [rtB, rtA];
      simU.uState.value = rtA.texture;
      simU.uSeed.value.z = 0;
    }
    renderer.setRenderTarget(null);
    viewU.uState.value = rtA.texture;
  }

  const st = { t: 0, speed: 1, first: !!opts.first, running: true, readied: false, frameN: 0 };

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
    if (t < stainEnd) return a;
    const b = restRect(), k = ease((t - stainEnd) / SETTLE);
    return { x: lerp(a.x, b.x, k), y: lerp(a.y, b.y, k), s: lerp(a.s, b.s, k) };
  }
  let cur = restRect();

  function resize() {
    renderer.setPixelRatio(host.dpr);
    renderer.setSize(host.vw, host.vh, false);
    ov.width = Math.round(host.vw * host.dpr); ov.height = Math.round(host.vh * host.dpr);
    viewCam.right = host.vw; viewCam.bottom = -host.vh; viewCam.updateProjectionMatrix();
  }

  function theme() {
    const d = host.theme.dark, c = viewU;
    c.uDark.value = d ? 1 : 0;
    if (d) {
      c.uPhase.value.set('#5d564e'); c.uHalo.value.set('#b3a99b');
      c.uCyto.value.set('#e2406f'); c.uEdge.value.set('#ff7aa0'); c.uNuc.value.set('#5f86ff');
    } else {
      c.uPhase.value.set('#7a7369'); c.uHalo.value.set('#ffffff');
      c.uCyto.value.set('#e892ac'); c.uEdge.value.set('#c9517a'); c.uNuc.value.set('#4b2f86');
    }
    render();
  }

  function overlay(r, t) {
    const dpr = host.dpr;
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.clearRect(0, 0, ov.width, ov.height);
    if (!st.running || t >= stainEnd) return;
    const cx = r.x + W * r.s / 2, cy = r.y + H * r.s / 2, R = viewU.uField.value * W * r.s / 2;
    const fade = 1 - clamp((t - growEnd) / .4);
    if (fade <= 0) return;
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    octx.globalAlpha = fade;
    octx.strokeStyle = host.theme.dim;
    octx.lineWidth = 1;
    octx.beginPath(); octx.arc(cx, cy, R, 0, Math.PI * 2); octx.stroke();
    // reticle ticks + scale bar, like an eyepiece graticule
    octx.beginPath();
    for (let i = 0; i < 4; i++) {
      const a = i * Math.PI / 2;
      octx.moveTo(cx + Math.cos(a) * (R - 8), cy + Math.sin(a) * (R - 8));
      octx.lineTo(cx + Math.cos(a) * (R + 8), cy + Math.sin(a) * (R + 8));
    }
    octx.stroke();
    const bar = 60 * r.s;
    octx.fillStyle = host.theme.ink;
    octx.fillRect(r.x + 20 * r.s, r.y + H * r.s + 14, bar, 2);
    octx.font = '500 10px "IBM Plex Mono", monospace';
    octx.fillText('100 µm', r.x + 20 * r.s, r.y + H * r.s + 30);
    octx.globalAlpha = 1;
  }

  function render() {
    const r = cur;
    view.scale.set(W * r.s, H * r.s, 1);
    view.position.set(r.x + W * r.s / 2, -(r.y + H * r.s / 2), 0);
    renderer.render(viewScene, viewCam);
    overlay(r, st.t);
  }

  function hud(t) {
    if (t < growEnd) {
      const day = 1 + Math.floor(t / growEnd * 7);
      host.hud(`Day ${day} · 37 °C · 5% CO₂ · confluence ${Math.round(easeOut(clamp(t / growEnd)) * 100)}%`);
    } else host.hud(host.theme.dark ? 'Fixed · DAPI + phalloidin' : 'Fixed · H&E stain');
  }

  function frame(dt) {
    const m = host.mouse;
    if (st.running) {
      st.t += dt * st.speed;
      hud(st.t);
      if (!st.readied && st.t >= stainEnd) { st.readied = true; host.ready(); }
      if (st.t >= (st.first ? settleEnd : stainEnd)) st.running = false;
    }
    const t = st.t;
    viewU.uField.value = st.running && t < growEnd ? lerp(.22, 1.35, easeOut(clamp(t / growEnd))) : 9;
    viewU.uStain.value = t < growEnd ? -.2 : t < stainEnd ? lerp(-.15, 1.15, ease((t - growEnd) / STAIN)) : 1.3;
    cur = rect();

    // cursor feeds nearby cells
    const over = m.x >= cur.x && m.x <= cur.x + W * cur.s && m.y >= cur.y && m.y <= cur.y + H * cur.s;
    simU.uHover.value.set((m.x - cur.x) / (W * cur.s) * simW, (1 - (m.y - cur.y) / (H * cur.s)) * simH, over ? 1 : 0);

    // grow fast during the intro; afterwards keep the culture quietly alive at ~30 fps
    st.frameN++;
    const iters = st.running && t < growEnd ? Math.round(ITER_GROW * Math.min(st.speed, 3)) : (st.frameN % 2 ? 0 : ITER_IDLE * 2);
    if (host.reduce && !st.running) return;
    if (iters) { step(iters); render(); }
    else if (st.running) render();
  }

  function finish() {
    const need = Math.max(0, Math.round((GROW - st.t) * 60 * ITER_GROW));
    step(need);
    st.t = settleEnd; st.running = false;
    viewU.uField.value = 9; viewU.uStain.value = 1.3; cur = rect();
    if (!st.readied) { st.readied = true; host.ready(); }
    render();
  }

  resize();
  simU.uState.value = initTex;
  step(1);
  theme();
  if (opts.instant) finish();
  void el.offsetWidth;
  el.classList.add('in');

  return {
    el,
    get running() { return st.running; },
    frame,
    resize() { resize(); cur = rect(); render(); },
    theme,
    skip() { st.speed = 5; },
    finish,
    replay() {
      simU.uState.value = initTex;
      step(1);
      Object.assign(st, { t: 0, speed: 1, first: false, running: true });
    },
    pointer(x, y) {
      const r = cur, u = (x - r.x) / (W * r.s), v = (y - r.y) / (H * r.s);
      if (st.running || u < 0 || u > 1 || v < 0 || v > 1) return false;
      if (!mask.b[Math.floor(v * H) * W + Math.floor(u * W)]) return false;
      simU.uSeed.value.set(u * simW, (1 - v) * simH, 7);
      return true;
    },
    destroy() {
      rtA.dispose(); rtB.dispose(); maskTex.dispose(); initTex.dispose(); quad.dispose();
      view.geometry.dispose(); view.material.dispose(); renderer.dispose(); el.remove();
    },
  };
}
