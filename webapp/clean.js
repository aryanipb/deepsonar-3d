// Browser port of majproj/clean.py: find the scanned object in ONE scan (z = up), or reject the file.
//
// Function-by-function port (same names, same order of operations, same thresholds): the decision (status, reason,
// the exact set of object points, confidence) must equal what Python returns on the same file. Numerics that decide
// discrete outcomes mirror numpy / scipy (see clean_numerics.js): pairwise sums, percentiles, np.arange / histogram bin
// edges, squared-distance comparisons, and the sign convention of LAPACK's SVD (the layer histogram and fill grid are
// anchored on an axis sign). eigh / lstsq are replaced by Jacobi eigen-solvers and scaled normal equations (agreement
// ~1e-13, not bit-identical). `node web/test/parity.mjs` checks every real file against samples/expected.json, and
// fails if majproj/clean.py changed after this port.
import {
  arange, ballQuery, components, convolveSame, eighSym, histogram, knn, lstsq, mean1, median, percentileSorted,
  pwSum, sortedCopy, std1, svd3,
} from './clean_numerics.js';

// sha256 of the majproj/clean.py this file was ported from; web/test/parity.mjs compares it with the file on disk
export const PORTED_FROM_CLEANER_SHA256 = '4515de08432d16cb4cfeec335614c6b9a70df5d781401330f3834503ce23b851';

const DEFAULTS = {
  eps_factor: 3.0, min_object_pts: 64, core_frac: 0.25, grow_frac: 0.08, dense_frac: 0.3, min_contrast: 3.0,
  max_linearity: 0.85, band: 1.5, normal_cos: Math.cos(45 * Math.PI / 180), layer_k: 2.5, max_layer_sd: 2.0, drift_aspect: 0.85,
  min_aspect: 0.2, min_aspect_rod: 0.05, rod_cos: Math.cos(25 * Math.PI / 180), min_fill: 0.35, horiz_cos: Math.cos(30 * Math.PI / 180),
  max_span: 0.5, origin: null, min_range: 0.12, behind_ratio: 1.5, ambiguity: 0.5, sor_k: 10, sor_sd: 2.5, max_candidates: 8,
};

// ---------- small helpers ----------
const round = (x, nd) => Number(x.toFixed(nd));
function pyFixed(x, nd) {                 // f'{x:.{nd}f}' (round half even on an exact tie, as Python does)
  const sc = x * 10 ** nd, fl = Math.floor(sc);
  if (sc - fl === 0.5 && (fl + 0.5) / 10 ** nd === x) return ((fl % 2 === 0 ? fl : fl + 1) / 10 ** nd).toFixed(nd);
  return x.toFixed(nd);
}
const gather = (P, idx) => {
  const out = new Float64Array(idx.length * 3);
  for (let r = 0; r < idx.length; r++) { const i = idx[r]; out[3 * r] = P[3 * i]; out[3 * r + 1] = P[3 * i + 1]; out[3 * r + 2] = P[3 * i + 2]; }
  return out;
};
const meanRows = q => {                   // q.mean(0): sequential column sums
  const n = q.length / 3, m = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) m[a] += q[3 * i + a];
  return m.map(v => v / n);
};
const dot3 = (a, b) => (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
const norm3 = a => Math.sqrt((a[0] * a[0] + a[1] * a[1]) + a[2] * a[2]);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const ptp = a => { let lo = Infinity, hi = -Infinity; for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; } return hi - lo; };
const argmax = a => { let j = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[j]) j = i; return j; };   // first maximum, like np.argmax
const arraysEqual = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

// ---------- geometry helpers ----------
// {c, ax (rows = principal axes, last = normal), sd (per-axis sd)} of flat points q, like _pca: SVD of the centred
// points. svd3 reproduces LAPACK's singular-vector signs, which the layer histogram and fill grid depend on.
function pca(q) {
  const n = q.length / 3, c = meanRows(q), X = new Float64Array(q.length);
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) X[3 * i + a] = q[3 * i + a] - c[a];
  const {vt, s} = svd3(X, n);
  return {c, ax: vt.map(r => Array.from(r)), sd: s.map(v => v / Math.sqrt(n))};
}

// per point: linearity (l1 - l2) / l1, local normal (smallest axis) and surface variation, from the k nearest neighbours
function localPca(P, nn, k) {
  const n = P.length / 3, kk = nn.k, lin = new Float64Array(n), norm = new Float64Array(n * 3), vr = new Float64Array(n);
  k = Math.min(k, kk);
  for (let i = 0; i < n; i++) {
    const m = [0, 0, 0];
    for (let t = 0; t < k; t++) for (let a = 0; a < 3; a++) m[a] += P[3 * nn.idx[i * kk + t] + a];
    for (let a = 0; a < 3; a++) m[a] /= k;
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let t = 0; t < k; t++) {
      const j = nn.idx[i * kk + t], d = [P[3 * j] - m[0], P[3 * j + 1] - m[1], P[3 * j + 2] - m[2]];
      for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) cov[a][b] += d[a] * d[b];
    }
    for (let a = 0; a < 3; a++) for (let b = a; b < 3; b++) { cov[a][b] /= k; cov[b][a] = cov[a][b]; }
    const {w, v} = eighSym(cov, 3);
    lin[i] = (w[2] - w[1]) / Math.max(w[2], 1e-18);
    vr[i] = w[0] / Math.max((w[0] + w[1]) + w[2], 1e-18);
    for (let a = 0; a < 3; a++) norm[3 * i + a] = v[a][0];
  }
  return {lin, norm, vr};
}

// Plane or quadric height field w = f(u, v) in a fixed PCA frame; residuals along the local normal.
class Surface {
  constructor(q, quadric) {
    const {c, ax} = pca(q);
    this.c = c; this.ax = ax;
    this.quadric = !!quadric && q.length / 3 >= 30;
    const {u, v, w} = this.uvw(q);
    this.coef = lstsq(this.design(u, v), w);
  }
  uvw(x) {
    const n = x.length / 3, u = new Float64Array(n), v = new Float64Array(n), w = new Float64Array(n), c = this.c, ax = this.ax;
    for (let i = 0; i < n; i++) {
      const d0 = x[3 * i] - c[0], d1 = x[3 * i + 1] - c[1], d2 = x[3 * i + 2] - c[2];
      u[i] = (d0 * ax[0][0] + d1 * ax[0][1]) + d2 * ax[0][2];
      v[i] = (d0 * ax[1][0] + d1 * ax[1][1]) + d2 * ax[1][2];
      w[i] = (d0 * ax[2][0] + d1 * ax[2][1]) + d2 * ax[2][2];
    }
    return {u, v, w};
  }
  design(u, v) {
    const one = u.map(() => 1);
    return this.quadric ? [one, u, v, u.map(x => x * x), u.map((x, i) => x * v[i]), v.map(x => x * x)] : [one, u, v];
  }
  slopes(u, v) {
    const a = this.coef, n = u.length, fu = new Float64Array(n), fv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (this.quadric) { fu[i] = (a[1] + 2 * a[3] * u[i]) + a[4] * v[i]; fv[i] = (a[2] + a[4] * u[i]) + 2 * a[5] * v[i]; }
      else { fu[i] = a[1]; fv[i] = a[2]; }
    }
    return {fu, fv};
  }
  residual(x) {
    const {u, v, w} = this.uvw(x), n = u.length, a = this.coef, {fu, fv} = this.slopes(u, v), r = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const f = this.quadric
        ? ((((a[0] + a[1] * u[i]) + a[2] * v[i]) + a[3] * (u[i] * u[i])) + a[4] * (u[i] * v[i])) + a[5] * (v[i] * v[i])
        : (a[0] + a[1] * u[i]) + a[2] * v[i];
      r[i] = (w[i] - f) / Math.sqrt((1 + fu[i] * fu[i]) + fv[i] * fv[i]);
    }
    return r;
  }
  normals(x) {
    const {u, v} = this.uvw(x), n = u.length, {fu, fv} = this.slopes(u, v), out = new Float64Array(n * 3), ax = this.ax;
    for (let i = 0; i < n; i++) {
      const g = [0, 1, 2].map(k => (-fu[i] * ax[0][k] - fv[i] * ax[1][k]) + ax[2][k]), l = norm3(g);
      for (let k = 0; k < 3; k++) out[3 * i + k] = g[k] / l;
    }
    return out;
  }
  sag(x) {                                   // peak-to-peak height of the quadratic part: how curved the patch is
    if (!this.quadric) return 0.0;
    const {u, v} = this.uvw(x), a = this.coef, n = u.length, qd = new Float64Array(n);
    for (let i = 0; i < n; i++) qd[i] = (a[3] * u[i] * u[i] + a[4] * u[i] * v[i]) + a[5] * v[i] * v[i];
    const lin = [u.map(() => 1), u, v], co = lstsq(lin, qd);
    for (let i = 0; i < n; i++) qd[i] -= (co[0] + co[1] * u[i]) + co[2] * v[i];
    return ptp(qd);
  }
}

function bfs(nb, ok, seed) {                 // sorted indices reachable from seed through nb, visiting only ok points
  const seen = new Uint8Array(nb.length);
  seen[seed] = 1;
  const out = [seed], st = [seed];
  while (st.length) {
    const i = st.pop();
    for (const j of nb[i]) if (ok[j] && !seen[j]) { seen[j] = 1; out.push(j); st.push(j); }
  }
  return Int32Array.from(out).sort();
}

// Dominant layer of the residuals r: peak of their density; spread = FWHM / 2.355 with the smoothing removed in quadrature.
function modeLayer(r, s, k) {
  const n = r.length;
  if (n < 5) return {keep: new Uint8Array(n).fill(1), spread: std1(r)};
  const h = s / 8.0, sg = s / 4.0;
  let mn = Infinity, mx = -Infinity;
  for (const v of r) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const edges = arange(mn - 4 * sg, mx + 4 * sg + h, h), hist = histogram(r, edges);
  const half_w = Math.trunc(3 * sg / h), ker = new Float64Array(2 * half_w + 1);
  for (let i = 0; i < ker.length; i++) { const x = (i - half_w) * h, t = x / sg; ker[i] = Math.exp(-0.5 * (t * t)); }
  const ks = pwSum(ker);
  const sm = convolveSame(hist, ker.map(v => v / ks));
  const mid = new Float64Array(sm.length);
  for (let i = 0; i < sm.length; i++) mid[i] = 0.5 * (edges[i] + edges[i + 1]);
  const j = argmax(sm), half = sm[j] / 2.0;
  let lo = j, hi = j;
  while (lo > 0 && sm[lo - 1] >= half) lo--;
  while (hi < sm.length - 1 && sm[hi + 1] >= half) hi++;
  const xl = lo > 0 ? mid[lo] - h * (sm[lo] - half) / Math.max(sm[lo] - sm[lo - 1], 1e-12) : mid[lo];
  const xr = hi < sm.length - 1 ? mid[hi] + h * (sm[hi] - half) / Math.max(sm[hi] - sm[hi + 1], 1e-12) : mid[hi];
  const peak = mid[j], sdObs = (xr - xl) / 2.355;
  const spread = Math.max(Math.sqrt(Math.max(sdObs * sdObs - sg * sg, 0.0)), s / 2.0);
  const keep = new Uint8Array(n);
  for (let i = 0; i < n; i++) keep[i] = Math.abs(r[i] - peak) <= k * spread ? 1 : 0;
  return {keep, spread};
}

// Mask of the residuals inside the window of half-width `half` that holds the most of them (ties: lowest).
function densestSlab(r, half) {
  const rs = sortedCopy(r), n = rs.length;
  let best = 0, bestCount = -1, hi = 0;
  for (let i = 0; i < n; i++) {              // searchsorted(rs, rs + 2 * half, side='right') - i
    const top = rs[i] + 2 * half;
    while (hi < n && rs[hi] <= top) hi++;
    if (hi - i > bestCount) { bestCount = hi - i; best = i; }
  }
  const centre = rs[best] + half, keep = new Uint8Array(r.length);
  for (let i = 0; i < r.length; i++) keep[i] = Math.abs(r[i] - centre) <= half ? 1 : 0;
  return keep;
}

// True when a too-thick layer is a surface dragged along the line of sight, not a volume (see _is_drift_plateau).
function isDriftPlateau(q, r, surf, prm) {
  if (prm.origin === null) return false;
  const m = meanRows(q), sight = m.map((v, a) => v - prm.origin[a]), ls = Math.max(norm3(sight), 1e-12), sg = sight.map(v => v / ls);
  const nm = meanRows(surf.normals(q)), ln = norm3(nm), nrm = nm.map(v => v / ln);
  const uv = toUV(q, surf.c, surf.ax), hi = pctCols(uv, 95), lo = pctCols(uv, 5);
  const extent = Math.min(hi[0] - lo[0], hi[1] - lo[1]);
  const rs = sortedCopy(r), depth = percentileSorted(rs, 95) - percentileSorted(rs, 5);
  return Math.abs(dot3(nrm, sg)) >= prm.normal_cos && depth <= prm.drift_aspect * extent;
}

const pctCols = (uv, q) => {                 // np.percentile(uv, q, 0) for an (n, 2) flat array
  const n = uv.length / 2;
  return [0, 1].map(k => percentileSorted(sortedCopy(Float64Array.from({length: n}, (_, i) => uv[2 * i + k])), q));
};
const toUV = (q, c, ax) => {                 // (q - c) @ ax[:2].T
  const n = q.length / 3, uv = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    const d = [q[3 * i] - c[0], q[3 * i + 1] - c[1], q[3 * i + 2] - c[2]];
    uv[2 * i] = dot3(d, ax[0]); uv[2 * i + 1] = dot3(d, ax[1]);
  }
  return uv;
};

// Occupied fraction of the patch's own footprint (its 2-98 percentile box, cells of size `cell`).
function fillFraction(uv, cell) {
  const n = uv.length / 2, lo = pctCols(uv, 2), hi = pctCols(uv, 98), keys = [];
  const g = [];
  for (let i = 0; i < n; i++) {
    const a = uv[2 * i], b = uv[2 * i + 1];
    if (a >= lo[0] && a <= hi[0] && b >= lo[1] && b <= hi[1]) g.push([Math.floor((a - lo[0]) / cell), Math.floor((b - lo[1]) / cell)]);
  }
  if (g.length < 3) return 0.0;
  const nx = Math.max(...g.map(v => v[0])) + 1, ny = Math.max(...g.map(v => v[1])) + 1;
  for (const [a, b] of g) keys.push(a * ny + b);
  return new Set(keys).size / (nx * ny);
}

// A weaker candidate parallel to a stronger one and lying over its footprint is the same surface seen at another depth.
function absorbDepthCopies(cands, normalCos) {
  const out = [];
  for (const c of cands) {
    let host = null;
    for (const b of out) {
      if (!b.passed) continue;                 // a rejected candidate (e.g. the table under the object) hosts nothing
      const nbm = meanRows(b.surf.normals(b.q)), ncm = meanRows(c.surf.normals(c.q));
      const parallel = Math.abs(dot3(nbm, ncm)) / (norm3(nbm) * norm3(ncm)) >= normalCos;
      const uvb = toUV(b.q, b.surf.c, b.surf.ax), uvc = toUV(c.q, b.surf.c, b.surf.ax);
      const lo = pctCols(uvb, 2), hi = pctCols(uvb, 98);
      let inside = 0;
      const nc = uvc.length / 2;
      for (let i = 0; i < nc; i++) if (uvc[2 * i] >= lo[0] && uvc[2 * i] <= hi[0] && uvc[2 * i + 1] >= lo[1] && uvc[2 * i + 1] <= hi[1]) inside++;
      const over = (inside / nc);
      const depth = median(Float64Array.from({length: c.q.length / 3}, (_, i) =>
        Math.abs(dot3([c.q[3 * i] - b.surf.c[0], c.q[3 * i + 1] - b.surf.c[1], c.q[3 * i + 2] - b.surf.c[2]], b.surf.ax[2]))));
      const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
      // a curved surface's drift copies need not be parallel to it: a candidate whose centre lies inside the stronger
      // one's footprint, close in depth, is the same object too
      const near = norm3([c.centre[0] - b.centre[0], c.centre[1] - b.centre[1], c.centre[2] - b.centre[2]]) <= 0.25 * span;
      if (over >= 0.5 && depth <= span && (parallel || near)) { host = b; break; }
    }
    if (!host) out.push(c); else { host.n_copies += c.n; (host.absorbed ||= []).push(c.centre); }
  }
  return out;
}

// Unit normal of the patch with a FIXED sign convention (see majproj/clean.py `_orient`).
function orient(surf, q, horizCos) {
  const m = meanRows(surf.normals(q)), l = norm3(m), nrm = m.map(v => v / l);
  if (Math.abs(nrm[2]) >= horizCos) return nrm[2] >= 0 ? nrm : nrm.map(v => -v);
  return nrm[1] >= 0 ? nrm : nrm.map(v => -v);
}

function growAndMeasure(P, nb, dens, growable, lnorm, seed, s, eps, dTop, sceneExt, prm) {
  const n = P.length / 3, tol = prm.band * eps;
  const hoodSet = new Set();
  for (const i of nb[seed]) for (const j of nb[i]) hoodSet.add(j);
  const hood = Int32Array.from(hoodSet).sort();
  if (hood.length < 10) return null;
  let surf = new Surface(gather(P, hood), false), sel = hood;
  for (let rnd = 0; rnd < 5; rnd++) {
    const ok = Uint8Array.from(growable);
    ok[seed] = 1;
    const cand = [];
    for (let i = 0; i < n; i++) if (ok[i]) cand.push(i);
    const cp = gather(P, cand), r = surf.residual(cp), nm = surf.normals(cp);
    for (let t = 0; t < cand.length; t++) {
      const i = cand[t];
      const agree = Math.abs((lnorm[3 * i] * nm[3 * t] + lnorm[3 * i + 1] * nm[3 * t + 1]) + lnorm[3 * i + 2] * nm[3 * t + 2]) >= prm.normal_cos;
      if (Math.abs(r[t]) > tol || !agree) ok[i] = 0;
    }
    ok[seed] = 1;
    const nw = bfs(nb, ok, seed);
    if (nw.length < 10) return null;
    surf = new Surface(gather(P, nw), rnd >= 1);
    if (arraysEqual(nw, sel)) break;
    sel = nw;
  }
  const grown = sel, nGrown = grown.length, gp = gather(P, grown);
  let idx = null, spread = 0, driftSlab = false;
  for (let it = 0; it < 3; it++) {              // always over the whole grown patch (no ratchet)
    const r = surf.residual(gp), ml = modeLayer(r, s, prm.layer_k);
    spread = ml.spread;
    let keep = ml.keep;
    // sticky: a re-fit to the slab can make the next round's half-maximum look narrow, which must not undo the slab
    if (driftSlab || (spread > prm.max_layer_sd * s && isDriftPlateau(gp, r, surf, prm))) {
      keep = densestSlab(r, prm.layer_k * s / 2);
      const inSlab = r.filter((_, t) => keep[t]);
      spread = Math.max(std1(inSlab), s / 2);
      driftSlab = true;
    }
    const kept = [];
    for (let t = 0; t < grown.length; t++) if (keep[t]) kept.push(grown[t]);
    if (kept.length < 10) return null;
    idx = Int32Array.from(kept);
    surf = new Surface(gather(P, idx), true);
  }
  const inset = new Uint8Array(n);
  for (const i of idx) inset[i] = 1;
  let best = idx[0];
  for (const i of idx) if (dens[i] > dens[best]) best = i;                 // np.argmax over idx order (ascending)
  idx = bfs(nb, inset, best);
  const nCopies = nGrown - idx.length;
  let q = gather(P, idx);
  if (idx.length > prm.sor_k + 1) {
    const kn = knn(q, prm.sor_k + 1), m = idx.length, dk = new Float64Array(m);
    for (let i = 0; i < m; i++) dk[i] = pwSum(kn.dist, i * kn.k + 1, prm.sor_k) / prm.sor_k;
    const thr = mean1(dk) + prm.sor_sd * std1(dk), kept = [];
    for (let i = 0; i < m; i++) if (dk[i] <= thr) kept.push(idx[i]);
    idx = Int32Array.from(kept);
    q = gather(P, idx);
  }
  if (idx.length < 10) return null;
  const nq = idx.length;
  surf = new Surface(q, true);
  const plane = new Surface(q, false);
  const rms = rmsOf(surf.residual(q)), rmsPlane = rmsOf(plane.residual(q)), sag = surf.sag(q);
  const shape = (sag >= s && rmsPlane >= 1.3 * rms) ? 'curved' : 'flat';
  const {c, ax, sd} = pca(q), uv = toUV(q, c, ax), fill = fillFraction(uv, 1.5 * s);
  const nm = meanRows(surf.normals(q)), nz = Math.abs(nm[2]);
  const uu = Float64Array.from({length: nq}, (_, i) => uv[2 * i]);
  const span = ptp(uu) / sceneExt;
  const dg = Float64Array.from(grown, i => dens[i]), medG = median(dg), medD = medG / dTop;
  const layerFrac = nq / nGrown;
  const horiz = nz >= prm.horiz_cos;
  const rod = prm.origin !== null && Math.abs(ax[0][2]) >= prm.rod_cos;
  const raw = {
    size: [nq, nq >= prm.min_object_pts],
    dense: [medD, medD >= prm.dense_frac],
    smooth: [spread / s, spread / s <= prm.max_layer_sd],
    not_line: [sd[1] / sd[0], sd[1] / sd[0] >= (rod ? prm.min_aspect_rod : prm.min_aspect)],
    filled: [fill, fill >= prm.min_fill],
    upright: [nz, nz < prm.horiz_cos],
    not_scene_plane: [horiz ? span : 0.0, !(horiz && span >= prm.max_span)],
  };
  const tests = {}, failed = [];
  for (const [k, [v, ok]] of Object.entries(raw)) { tests[k] = {value: round(v, 4), pass: !!ok}; if (!ok) failed.push(k); }
  return {idx, q, grown, surf, n: nq, score: round(nGrown * medD * dTop, 1), passed: failed.length === 0, failed, tests, rms, sag, shape,
    fill, layer_frac: layerFrac, n_copies: nCopies, drift_slab: driftSlab, centre: c.map(v => round(v, 3)), normal_z: round(nz, 3), rms_over_s: round(rms / s, 3),
    layer_sd_over_s: round(spread / s, 3), layer_sd: spread, median_density_frac: round(medD, 3), med_dens: medG, plane_over_quadric_rms: round(rmsPlane / rms, 3)};
}
const rmsOf = r => { const sq = new Float64Array(r.length); for (let i = 0; i < r.length; i++) sq[i] = r[i] * r[i]; return Math.sqrt(mean1(sq)); };

// ---------- main entry ----------
// points: flat Float64Array [x0, y0, z0, ...] (finite rows; the file frame, z up). origin: the sensor position when known
// (raw metric maps keep it at the map origin; exports do not). Returns {object, removed, meta}.
export function clean(points, origin = null) {
  const prm = {...DEFAULTS, origin}, nInput = points.length / 3;
  const p = uniqueRows(points), n = p.length / 3;
  const meta = {status: 'rejected', reason: null, confidence: 0.0, n_input: nInput, n_unique: n, n_object: 0, n_removed: n, eps: null,
    tests: {}, normal: null, centre: null, width: null, height: null, shape_hint: null, candidates: [], frame: 'z up',
    n_depth_copies: 0, aspect_w_over_h: null, plane_sd: null};
  const reject = reason => { meta.reason = reason; return {object: new Float32Array(0), removed: Float32Array.from(p), meta}; };
  if (n < Math.max(prm.min_object_pts, 16)) return reject(`too few points (${n} unique)`);

  const nn16 = knn(p, 16);
  const d8 = Float64Array.from({length: n}, (_, i) => nn16.dist[i * nn16.k + 8]);
  const d20 = percentileSorted(sortedCopy(d8), 20);
  const s = median(d8.filter(v => v <= d20));
  if (!(s > 0)) return reject('degenerate point spacing');
  const eps = prm.eps_factor * s;
  meta.eps = round(eps, 5);
  meta.spacing = round(s, 5);
  const nb = ballQuery(p, eps);
  const dens = Float64Array.from(nb, l => l.length - 1);
  const {lin} = localPca(p, nn16, 16);
  const dTop = mean1(sortedCopy(dens).slice(-prm.min_object_pts));
  meta.d_top = round(dTop, 1);
  const kLocal = Math.trunc(Math.min(Math.max(16, dTop / 2), n));
  const {norm: lnorm, vr: lvar} = localPca(p, knn(p, kLocal), kLocal);
  const sceneExt = Math.max(...[0, 1, 2].map(a => ptp(Float64Array.from({length: n}, (_, i) => p[3 * i + a]))));

  const core = new Uint8Array(n), growable = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const notLine = lin[i] <= prm.max_linearity;
    core[i] = dens[i] >= prm.core_frac * dTop && notLine ? 1 : 0;
    growable[i] = dens[i] >= prm.grow_frac * dTop && notLine ? 1 : 0;
  }
  const seedScore = Float64Array.from({length: n}, (_, i) => dens[i] * Math.min(Math.max(1 - 3 * lvar[i], 0), 1));
  const avail = Uint8Array.from(core), free = Uint8Array.from(growable);
  let cands = [];
  for (let iter = 0; iter < 2 * prm.max_candidates; iter++) {
    if (cands.length >= prm.max_candidates) break;
    const {lab, count} = components(nb, avail);
    const members = Array.from({length: count}, () => []);
    for (let i = 0; i < n; i++) if (lab[i] >= 0) members[lab[i]].push(i);
    let bestC = null;
    for (const idx of members) {
      if (idx.length < Math.floor(prm.min_object_pts / 2)) continue;
      const st = idx.length * median(Float64Array.from(idx, i => dens[i]));
      if (!bestC || st > bestC.st) bestC = {st, idx};
    }
    if (!bestC) break;
    const comp = bestC.idx;
    let seed = comp[0];
    for (const i of comp) if (seedScore[i] > seedScore[seed]) seed = i;
    const cand = growAndMeasure(p, nb, dens, free, lnorm, seed, s, eps, dTop, sceneExt, prm);
    const claimed = cand ? cand.grown : comp;
    for (const i of claimed) { avail[i] = 0; free[i] = 0; }
    avail[seed] = 0;
    if (cand) cands.push(cand);
  }
  // contrast: a dwelt-on surface must stand out from the rest of the file. "The rest" excludes the candidate itself and
  // any wide gravity-aligned scene plane (a saturated table / floor is explained structure, not background)
  const scene = new Uint8Array(n);
  for (const c of cands) if (!c.tests.not_scene_plane.pass) for (const i of c.grown) scene[i] = 1;
  for (const c of cands) {
    const rest = Uint8Array.from(scene, v => 1 - v);          // rest = ~scene, minus the candidate's grown patch
    for (const i of c.grown) rest[i] = 0;
    const restDens = [];
    for (let i = 0; i < n; i++) if (rest[i]) restDens.push(dens[i]);
    const bg = restDens.length ? median(Float64Array.from(restDens)) : 1.0;
    const con = c.med_dens / Math.max(bg, 1.0);
    c.contrast = round(con, 2);
    c.tests.contrast = {value: round(con, 4), pass: con >= prm.min_contrast};
    c.failed = Object.entries(c.tests).filter(([, t]) => !t.pass).map(([k]) => k);
    c.passed = c.failed.length === 0;
  }
  cands.sort((a, b) => b.score - a.score);
  cands = absorbDepthCopies(cands, prm.normal_cos);
  meta.candidates = cands.map(c => Object.fromEntries(['n', 'score', 'passed', 'failed', 'centre', 'normal_z', 'rms_over_s', 'layer_sd_over_s',
    'median_density_frac', 'contrast', 'layer_frac', 'fill'].map(k => [k, c[k]])));
  const passing = cands.filter(c => c.passed);
  if (!cands.length) return reject('no dense surface candidate (nothing reaches the saturated density level)');
  if (!passing.length) {
    const b = cands[0];
    meta.tests = b.tests;
    return reject(`no candidate passes all signal tests; best candidate fails: ${b.failed.join(', ')}`);
  }
  let passing2 = passing;
  if (prm.origin !== null) {
    // The user aims at the object, so with a known sensor position the target is the nearest dwelled-on surface;
    // surfaces well behind it (furniture, walls) are background, not rivals, however dense they are.
    const o = prm.origin, rng = c => { const dx = c.centre[0] - o[0], dy = c.centre[1] - o[1]; return Math.sqrt(dx * dx + dy * dy); };
    const inrange = passing.filter(c => rng(c) >= prm.min_range);
    if (inrange.length) {
      const near = Math.min(...inrange.map(rng));
      passing2 = inrange.filter(c => rng(c) < prm.behind_ratio * near);
      meta.range_m = round(near, 4);
    }
  }
  const best = passing2[0], ratio = passing2.length > 1 ? passing2[1].score / best.score : 0.0;
  meta.tests = {...best.tests, isolated: {value: round(1 - ratio, 4), pass: ratio < prm.ambiguity}};
  if (ratio >= prm.ambiguity) return reject(`ambiguous: two comparable object candidates (runner-up score ${pyFixed(ratio, 2)} x the best)`);

  const idx = best.idx, q = best.q, surf = best.surf, normal = orient(surf, q, prm.horiz_cos), centre = meanRows(q);
  const UP = [0, 0, 1];
  let tH = cross(UP, normal), tV;
  if (norm3(tH) < 1e-6) { tH = surf.ax[0]; tV = surf.ax[1]; }
  else { const l = norm3(tH); tH = tH.map(v => v / l); tV = cross(normal, tH); const l2 = norm3(tV); tV = tV.map(v => v / l2); }
  const nq = idx.length, hh = new Float64Array(nq), vv = new Float64Array(nq);
  for (let i = 0; i < nq; i++) {
    const d = [q[3 * i] - centre[0], q[3 * i + 1] - centre[1], q[3 * i + 2] - centre[2]];
    hh[i] = dot3(d, tH); vv[i] = dot3(d, tV);
  }
  const width = ptp(hh), height = ptp(vv), sdPlane = pca(q).sd;
  const terms = {count: 1 - Math.exp(-nq / 256.0), smooth: Math.exp(-((best.layer_sd / s) ** 2)), isolation: 1 - ratio / prm.ambiguity,
    coverage: Math.min(1.0, best.fill / 0.8), purity: best.layer_frac};
  const lg = Float64Array.from(Object.values(terms), v => Math.log(Math.min(Math.max(v, 1e-6), 1)));
  const conf = Math.exp(pwSum(lg) / lg.length);
  const mask = new Uint8Array(n);
  for (const i of idx) mask[i] = 1;
  Object.assign(meta, {status: 'ok', reason: null, confidence: round(conf, 4),
    confidence_terms: Object.fromEntries(Object.entries(terms).map(([k, v]) => [k, round(v, 4)])),
    n_object: nq, n_removed: n - nq, n_depth_copies: best.n_copies, normal: normal.map(v => round(v, 4)),
    normal_sign: 'convention (+y, or +z if facing up/down)', centre: centre.map(v => round(v, 4)), width: round(width, 4), height: round(height, 4),
    aspect_w_over_h: height > 0 ? round(width / height, 3) : null, plane_sd: round(sdPlane[2], 5), surface_rms: round(best.rms, 5),
    sag: round(best.sag, 5), shape_hint: best.shape});
  if (best.drift_slab) meta.layer_mode = 'drift_slab';
  const rest = [];
  for (let i = 0; i < n; i++) if (!mask[i]) rest.push(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
  return {object: Float32Array.from(q), removed: Float32Array.from(rest), meta};
}

// np.unique(rows, axis=0): lexicographic order, exact duplicates dropped
function uniqueRows(flat) {
  const n = flat.length / 3, rows = Array.from({length: n}, (_, i) => i);
  rows.sort((a, b) => {
    for (let k = 0; k < 3; k++) { const u = flat[3 * a + k], w = flat[3 * b + k]; if (u < w) return -1; if (u > w) return 1; }
    return 0;
  });
  const keep = [];
  for (let r = 0; r < rows.length; r++) {
    if (r > 0) {
      const a = rows[r - 1], b = rows[r];
      if (flat[3 * a] === flat[3 * b] && flat[3 * a + 1] === flat[3 * b + 1] && flat[3 * a + 2] === flat[3 * b + 2]) continue;
    }
    keep.push(rows[r]);
  }
  const p = new Float64Array(keep.length * 3);
  keep.forEach((i, r) => { p[3 * r] = flat[3 * i]; p[3 * r + 1] = flat[3 * i + 1]; p[3 * r + 2] = flat[3 * i + 2]; });
  return p;
}
