// Shared helpers: easing, fitting, and rasterising the name into a mask.

export const DEG = Math.PI / 180;
export const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
export const easeOut = t => 1 - Math.pow(1 - t, 3);

export const FONT_PX = 180;
export const FONT = `800 ${FONT_PX}px Archivo`;

// Draws the name (one line per entry) into a tight canvas and returns its coverage.
export function rasterize(lines, pad = 24) {
  const canvas = document.createElement('canvas');
  const m = canvas.getContext('2d', { willReadFrequently: true });
  m.font = FONT;
  const met = lines.map(t => m.measureText(t));
  const gap = FONT_PX * .12;
  const asc = Math.max(...met.map(x => x.actualBoundingBoxAscent));
  const desc = Math.max(...met.map(x => x.actualBoundingBoxDescent));
  const lineH = asc + desc + gap;
  const wOf = x => x.actualBoundingBoxLeft + x.actualBoundingBoxRight;
  const W = Math.ceil(Math.max(...met.map(wOf)) + pad * 2);
  const H = Math.ceil(lineH * lines.length - gap + pad * 2);
  canvas.width = W; canvas.height = H;
  m.font = FONT; m.fillStyle = '#000';
  lines.forEach((t, i) => m.fillText(t, (W - wOf(met[i])) / 2 + met[i].actualBoundingBoxLeft, pad + asc + i * lineH));

  const px = m.getImageData(0, 0, W, H).data;
  const a = new Float32Array(W * H), b = new Uint8Array(W * H);
  for (let i = 0; i < a.length; i++) { a[i] = px[i * 4 + 3] / 255; b[i] = a[i] > .5; }
  return { W, H, a, b, lineH, canvas };
}

// Scale that fits a W×H×Z slab, seen at (yaw, tilt), inside w×h pixels.
export function fit(W, H, Z, yaw, tilt, w, h) {
  const cy = Math.abs(Math.cos(yaw)), sy = Math.abs(Math.sin(yaw));
  const pw = W * cy + H * sy;
  const ph = (W * sy + H * cy) * Math.cos(tilt) + Z * Math.sin(tilt);
  return Math.min(w / pw, h / ph);
}

// Small deterministic PRNG so every visit grows the same way.
export function rng(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
