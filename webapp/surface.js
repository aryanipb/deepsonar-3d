// Point cloud -> one closed 3D surface, in the browser. Geometry only: no template and no object label.
// Stray points are dropped, each horizontal slice is wrapped with its convex outline, and the outlines are joined
// into one closed skin. Needs an upright object with convex slices; a concave cross-section gets filled in.
const CELL = 0.006, MIN_NEAR = 12, SLICE = 0.005, ANGLES = 72, SMOOTH = 3;

function keepDense(p) {
  const n = p.length / 3, key = (x, y, z) => `${x},${y},${z}`, cells = new Map(), c = i => Math.floor(p[i] / CELL);
  for (let i = 0; i < n; i++) { const k = key(c(3 * i), c(3 * i + 1), c(3 * i + 2)); cells.set(k, (cells.get(k) || 0) + 1); }
  const out = [];
  for (let i = 0; i < n; i++) {
    const x = c(3 * i), y = c(3 * i + 1), z = c(3 * i + 2);
    let near = 0;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let d = -1; d <= 1; d++) near += cells.get(key(x + a, y + b, z + d)) || 0;
    if (near >= MIN_NEAR) out.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
  }
  return out;
}

function hull(q) {                                    // q: [[x, y], ...] -> convex outline, counter-clockwise
  q.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = pts => { const h = []; for (const r of pts) { while (h.length > 1 && cross(h[h.length - 2], h[h.length - 1], r) <= 0) h.pop(); h.push(r); } h.pop(); return h; };
  return [...half(q), ...half([...q].reverse())];
}

function radii(h, cx, cy) {                           // distance from the centre to the outline, at each angle
  const r = new Float64Array(ANGLES);
  for (let k = 0; k < ANGLES; k++) {
    const t = 2 * Math.PI * k / ANGLES, dx = Math.cos(t), dy = Math.sin(t);
    for (let i = 0; i < h.length; i++) {
      const a = h[i], b = h[(i + 1) % h.length], ex = b[0] - a[0], ey = b[1] - a[1], ax = a[0] - cx, ay = a[1] - cy, d = dx * ey - dy * ex;
      if (Math.abs(d) < 1e-12) continue;
      const s = (ax * ey - ay * ex) / d, u = (ax * dy - ay * dx) / d;
      if (s >= 0 && u >= -1e-9 && u <= 1 + 1e-9) { r[k] = Math.max(r[k], s); }
    }
  }
  return r;
}

const blur = (v, w) => v.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - w); j <= Math.min(v.length - 1, i + w); j++) { s += v[j]; n++; } return s / n; });

// flat [x0, y0, z0, x1, ...] metres, z up -> {x, y, z, i, j, k} closed triangle mesh
export function toSurface(flat) {
  const p = keepDense(flat), n = p.length / 3;
  if (n < 100) throw new Error('too few points for a surface');
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { lo = Math.min(lo, p[3 * i + 2]); hi = Math.max(hi, p[3 * i + 2]); }
  const L = Math.max(2, Math.round((hi - lo) / SLICE)), bins = Array.from({length: L}, () => []);
  for (let i = 0; i < n; i++) bins[Math.min(L - 1, Math.floor((p[3 * i + 2] - lo) / (hi - lo) * L))].push([p[3 * i], p[3 * i + 1]]);
  const rows = [];
  for (let l = 0; l < L; l++) {
    const q = [...(bins[l - 1] || []), ...bins[l], ...(bins[l + 1] || [])];
    if (q.length < 6) continue;
    const h = hull(q);
    if (h.length < 3) continue;
    const cx = h.reduce((s, a) => s + a[0], 0) / h.length, cy = h.reduce((s, a) => s + a[1], 0) / h.length;
    rows.push({z: lo + (l + 0.5) * (hi - lo) / L, cx, cy, r: radii(h, cx, cy)});
  }
  if (rows.length < 2) throw new Error('too few slices for a surface');
  const cx = blur(rows.map(r => r.cx), SMOOTH), cy = blur(rows.map(r => r.cy), SMOOTH);
  const R = rows.map(() => new Float64Array(ANGLES));
  for (let k = 0; k < ANGLES; k++) { const s = blur(rows.map(r => r.r[k]), SMOOTH); s.forEach((v, l) => { R[l][k] = v; }); }
  const m = {x: [], y: [], z: [], i: [], j: [], k: []}, M = rows.length, tri = (a, b, c) => { m.i.push(a); m.j.push(b); m.k.push(c); };
  for (let l = 0; l < M; l++) for (let k = 0; k < ANGLES; k++) {
    const r = (R[l][(k + ANGLES - 1) % ANGLES] + 2 * R[l][k] + R[l][(k + 1) % ANGLES]) / 4, t = 2 * Math.PI * k / ANGLES;
    m.x.push(cx[l] + r * Math.cos(t)); m.y.push(cy[l] + r * Math.sin(t)); m.z.push(rows[l].z);
  }
  const at = (l, k) => l * ANGLES + (k + ANGLES) % ANGLES;
  for (let l = 0; l + 1 < M; l++) for (let k = 0; k < ANGLES; k++) { tri(at(l, k), at(l, k + 1), at(l + 1, k + 1)); tri(at(l, k), at(l + 1, k + 1), at(l + 1, k)); }
  const cap = (l, top) => {
    const c = m.x.length;
    m.x.push(cx[l]); m.y.push(cy[l]); m.z.push(rows[l].z);
    for (let k = 0; k < ANGLES; k++) top ? tri(c, at(l, k), at(l, k + 1)) : tri(c, at(l, k + 1), at(l, k));
  };
  cap(0, false); cap(M - 1, true);
  return m;
}
