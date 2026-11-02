// Numerics for the browser port of majproj/clean.py. Where a numpy / scipy routine decides a discrete outcome, its
// floating-point behaviour is mirrored exactly (pairwise summation, percentile interpolation, np.arange, histogram bin
// edges, squared-distance comparisons). LAPACK (eigh / lstsq) is replaced by Jacobi / scaled normal equations: the
// same numbers to ~1e-13, not bit-identical.

// np.add.reduce on a contiguous 1-D float64 array (pairwise summation, identity-initialised).
export function pwSum(a, lo = 0, n = a.length - lo) {
  if (n < 8) {
    let r = 0;
    for (let i = 0; i < n; i++) r += a[lo + i];
    return r;
  }
  if (n <= 128) {
    const r = [a[lo], a[lo + 1], a[lo + 2], a[lo + 3], a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[lo + i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pwSum(a, lo, n2) + pwSum(a, lo + n2, n - n2);
}
export const mean1 = a => pwSum(a) / a.length;                      // a.mean() for 1-D
export function std1(a) {                                           // a.std() for 1-D (population)
  const m = mean1(a), d = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) { const x = a[i] - m; d[i] = x * x; }
  return Math.sqrt(pwSum(d) / a.length);
}
export const sortedCopy = a => Float64Array.from(a).sort();

// np.percentile(a, q) (method 'linear') on an ascending-sorted array
export function percentileSorted(s, q) {
  const n = s.length;
  const virt = (n - 1) * (q / 100);
  if (virt >= n - 1) return s[n - 1];
  const prev = Math.floor(virt), gamma = virt - prev;
  const a = s[prev], b = s[prev + 1], diff = b - a;
  return gamma >= 0.5 ? b - diff * (1 - gamma) : a + diff * gamma;
}
export const percentile = (values, q) => percentileSorted(sortedCopy(values), q);

export function medianSorted(s) {
  const n = s.length, h = n >> 1;
  return n % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
}
export const median = a => medianSorted(sortedCopy(a));

// np.arange(start, stop, step) for floats: n = ceil((stop - start) / step), a[i] = start + i * ((start + step) - start)
export function arange(start, stop, step) {
  const n = Math.max(0, Math.ceil((stop - start) / step));
  const out = new Float64Array(n);
  if (n > 0) out[0] = start;
  if (n > 1) out[1] = start + step;
  const delta = (start + step) - start;
  for (let i = 2; i < n; i++) out[i] = start + i * delta;
  return out;
}

// np.histogram(a, bins=edges): bin j holds edges[j] <= x < edges[j + 1]; the last bin also holds x == edges[last].
export function histogram(vals, edges) {
  const nb = edges.length - 1, counts = new Float64Array(nb);
  for (let n = 0; n < vals.length; n++) {
    const v = vals[n];
    if (!(v >= edges[0] && v <= edges[nb])) continue;
    let lo = 0, hi = nb + 1;                  // first index with edges[idx] > v
    while (lo < hi) { const mid = (lo + hi) >> 1; if (edges[mid] > v) hi = mid; else lo = mid + 1; }
    counts[Math.min(lo - 1, nb - 1)]++;
  }
  return counts;
}

// np.convolve(h, w, 'same') for a short odd kernel w
export function convolveSame(h, w) {
  const n = h.length, m = w.length, half = m >> 1, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < m; k++) {
      const j = i + half - k;
      if (j >= 0 && j < n) s += h[j] * w[k];
    }
    out[i] = s;
  }
  return out;
}

// ---------- symmetric eigen-solver (cyclic Jacobi): ascending eigenvalues, eigenvectors as columns v[row][col] ----------
export function eighSym(A, m) {
  const a = A.map(r => Float64Array.from(r)), v = Array.from({length: m}, (_, i) => Float64Array.from({length: m}, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, diag = 0;
    for (let i = 0; i < m; i++) { diag += a[i][i] * a[i][i]; for (let j = i + 1; j < m; j++) off += a[i][j] * a[i][j]; }
    if (off <= 1e-34 * diag || off === 0) break;
    for (let p = 0; p < m - 1; p++) {
      for (let q = p + 1; q < m; q++) {
        if (a[p][q] === 0) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < m; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
        for (let k = 0; k < m; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
        for (let k = 0; k < m; k++) { const vkp = v[k][p], vkq = v[k][q]; v[k][p] = c * vkp - s * vkq; v[k][q] = s * vkp + c * vkq; }
      }
    }
  }
  const order = Array.from({length: m}, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {w: order.map(i => a[i][i]), v: Array.from({length: m}, (_, r) => order.map(i => v[r][i]))};
}

// Linear least squares min ||A x - b||, A given as columns (array of Float64Array, each length n), like
// np.linalg.lstsq(A, b, rcond=None)[0]: columns are scaled to unit RMS, the Gram matrix is diagonalised and singular
// directions below eps * max(n, m) of the largest are dropped (minimum-norm solution).
export function lstsq(cols, b) {
  const m = cols.length, n = b.length;
  const scale = cols.map(c => { let s = 0; for (let i = 0; i < n; i++) s += c[i] * c[i]; return Math.sqrt(s / n) || 1; });
  const G = Array.from({length: m}, () => new Float64Array(m)), g = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    for (let j = i; j < m; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += cols[i][k] * cols[j][k];
      G[i][j] = G[j][i] = s / (scale[i] * scale[j]);
    }
    let s = 0;
    for (let k = 0; k < n; k++) s += cols[i][k] * b[k];
    g[i] = s / scale[i];
  }
  const {w, v} = eighSym(G, m);
  const cut = (2.220446049250313e-16 * Math.max(n, m)) ** 2 * Math.max(...w);
  const x = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    if (w[k] <= cut) continue;
    let d = 0;
    for (let i = 0; i < m; i++) d += v[i][k] * g[i];
    d /= w[k];
    for (let i = 0; i < m; i++) x[i] += v[i][k] * d;
  }
  return Array.from(x, (c, i) => c / scale[i]);
}

// ---------- neighbours ----------
// k nearest neighbours of every point of P (flat, n points) among P, self included, ascending by distance (ties: lower
// index first). Distance = sqrt((dx^2 + dy^2) + dz^2), as cKDTree computes it. Returns {idx: Int32Array(n*k), dist: Float64Array(n*k)}.
// Searched with a k-d tree: a raw map has about 9000 points, and the scan over all pairs took 15 s on a desktop.
export function knn(P, k) {
  const n = P.length / 3;
  k = Math.min(k, n);
  const LEAF = 12, perm = Int32Array.from({length: n}, (_, i) => i);
  const nodes = [];                         // {axis, split, left, right} or {lo, hi}
  const swap = (a, b) => { const t = perm[a]; perm[a] = perm[b]; perm[b] = t; };
  const select = (lo, hi, m, ax) => {       // quickselect: perm[m] takes its sorted place in [lo, hi)
    hi--;
    while (lo < hi) {
      const pivot = P[3 * perm[(lo + hi) >> 1] + ax];
      let i = lo, j = hi;
      while (i <= j) {
        while (P[3 * perm[i] + ax] < pivot) i++;
        while (P[3 * perm[j] + ax] > pivot) j--;
        if (i <= j) swap(i++, j--);
      }
      if (m <= j) hi = j; else if (m >= i) lo = i; else return;
    }
  };
  const build = (lo, hi) => {
    const id = nodes.length;
    nodes.push(null);
    if (hi - lo <= LEAF) { nodes[id] = {lo, hi}; return id; }
    let axis = 0, wide = -1;
    for (let a = 0; a < 3; a++) {
      let mn = Infinity, mx = -Infinity;
      for (let t = lo; t < hi; t++) { const v = P[3 * perm[t] + a]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mx - mn > wide) { wide = mx - mn; axis = a; }
    }
    const mid = (lo + hi) >> 1;
    select(lo, hi, mid, axis);
    const split = P[3 * perm[mid] + axis], left = build(lo, mid), right = build(mid, hi);
    nodes[id] = {axis, split, left, right};
    return id;
  };
  build(0, n);
  const idx = new Int32Array(n * k), dist = new Float64Array(n * k), bd = new Float64Array(k), bi = new Int32Array(k);
  let x = 0, y = 0, z = 0, cnt = 0;
  const visit = id => {
    const nd = nodes[id];
    if (nd.left === undefined) {
      for (let t = nd.lo; t < nd.hi; t++) {
        const j = perm[t], dx = P[3 * j] - x, dy = P[3 * j + 1] - y, dz = P[3 * j + 2] - z;
        const d = (dx * dx + dy * dy) + dz * dz;
        if (cnt === k && (d > bd[k - 1] || (d === bd[k - 1] && j > bi[k - 1]))) continue;
        let pos = cnt < k ? cnt++ : k - 1;
        while (pos > 0 && (bd[pos - 1] > d || (bd[pos - 1] === d && bi[pos - 1] > j))) { bd[pos] = bd[pos - 1]; bi[pos] = bi[pos - 1]; pos--; }
        bd[pos] = d; bi[pos] = j;
      }
      return;
    }
    const gap = (nd.axis === 0 ? x : nd.axis === 1 ? y : z) - nd.split, near = gap < 0 ? nd.left : nd.right, far = gap < 0 ? nd.right : nd.left;
    visit(near);
    if (cnt < k || gap * gap <= bd[k - 1]) visit(far);
  };
  for (let i = 0; i < n; i++) {
    x = P[3 * i]; y = P[3 * i + 1]; z = P[3 * i + 2]; cnt = 0;
    visit(0);
    for (let t = 0; t < k; t++) { idx[i * k + t] = bi[t]; dist[i * k + t] = Math.sqrt(bd[t]); }
  }
  return {idx, dist, k};
}

// Neighbour lists within radius r (squared-distance comparison d2 <= r * r, self included), ascending indices, like
// cKDTree.query_ball_point(P, r). Uniform grid only to skip far points; the test itself is exact.
export function ballQuery(P, r) {
  const n = P.length / 3, cell = r * 1.000001, lo = [Infinity, Infinity, Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) lo[a] = Math.min(lo[a], P[3 * i + a]);
  const cx = new Int32Array(n * 3), dims = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) { const c = Math.floor((P[3 * i + a] - lo[a]) / cell); cx[3 * i + a] = c; dims[a] = Math.max(dims[a], c + 1); }
  const key = (x, y, z) => (x * (dims[1] + 2) + y) * (dims[2] + 2) + z;
  const grid = new Map();
  for (let i = 0; i < n; i++) {
    const kk = key(cx[3 * i], cx[3 * i + 1], cx[3 * i + 2]);
    const l = grid.get(kk);
    if (l) l.push(i); else grid.set(kk, [i]);
  }
  const r2 = r * r, out = new Array(n);
  for (let i = 0; i < n; i++) {
    const res = [], x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const X = cx[3 * i] + a, Y = cx[3 * i + 1] + b, Z = cx[3 * i + 2] + c;
      if (X < 0 || Y < 0 || Z < 0) continue;
      const l = grid.get(key(X, Y, Z));
      if (!l) continue;
      for (const j of l) {
        const dx = P[3 * j] - x, dy = P[3 * j + 1] - y, dz = P[3 * j + 2] - z;
        if ((dx * dx + dy * dy) + dz * dz <= r2) res.push(j);
      }
    }
    res.sort((u, v) => u - v);
    out[i] = Int32Array.from(res);
  }
  return out;
}

// Connected components of the graph whose edges are the neighbour lists `nb`, restricted to points where mask is true.
// Returns {lab: Int32Array (-1 where mask is false), count}. Labels in order of the lowest member index.
export function components(nb, mask) {
  const n = nb.length, lab = new Int32Array(n).fill(-1);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (!mask[s] || lab[s] >= 0) continue;
    lab[s] = count;
    const st = [s];
    while (st.length) {
      const i = st.pop();
      for (const j of nb[i]) if (mask[j] && lab[j] < 0) { lab[j] = count; st.push(j); }
    }
    count++;
  }
  return {lab, count};
}

// ---------- SVD of a tall n x 3 matrix, with LAPACK's sign convention ----------
// The cleaner's surface frames come from np.linalg.svd (LAPACK dgesdd). A singular vector is only defined up to its
// sign, but the sign is NOT irrelevant downstream: the layer histogram is anchored at min(residual) and the fill grid
// at the 2nd percentile, so flipping an axis changes both. To reproduce Python's decisions this follows the dgesdd
// path for M >= 5, N = 3, JOBZ = 'S' step by step (the Householder QR dgeqr2, the bidiagonalisation dgebd2, the QR
// iteration dbdsqr with dlasv2 / dlas2 / dlartg, and VT = VT_B * P^T) and returns the right singular vectors (rows of
// vt, decreasing singular values). Only vt and s are computed; U is not needed.
const EPS_LAPACK = 1.1102230246251565e-16, SAFMIN = 2.2250738585072014e-308;
const sign = (a, b) => (b >= 0 && !Object.is(b, -0) ? Math.abs(a) : -Math.abs(a));       // Fortran SIGN(a, b)

// dlarfg: returns {beta, tau, scale} for the vector (alpha, x...) with ||x|| = xnorm
function dlarfg(alpha, xnorm) {
  if (xnorm === 0) return {beta: alpha, tau: 0, scale: 1};
  const beta = -sign(Math.hypot(alpha, xnorm), alpha);
  return {beta, tau: (beta - alpha) / beta, scale: 1 / (alpha - beta)};
}

// dlartg (LAPACK >= 3.10): c >= 0, r has the sign of f
function dlartg(f, g) {
  if (g === 0) return {c: 1, s: 0, r: f};
  if (f === 0) return {c: 0, s: sign(1, g), r: Math.abs(g)};
  const d = Math.sqrt(f * f + g * g), r = sign(d, f);
  return {c: Math.abs(f) / d, s: g / r, r};
}

function dlas2(f, g, h) {                     // singular values (ssmin, ssmax) of [[f, g], [0, h]]
  const fa = Math.abs(f), ga = Math.abs(g), ha = Math.abs(h), fhmn = Math.min(fa, ha), fhmx = Math.max(fa, ha);
  if (fhmn === 0) return {ssmin: 0, ssmax: fhmx === 0 ? ga : Math.max(fhmx, ga) * Math.sqrt(1 + (Math.min(fhmx, ga) / Math.max(fhmx, ga)) ** 2)};
  if (ga < fhmx) {
    const as = 1 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, au = (ga / fhmx) ** 2, c = 2 / (Math.sqrt(as * as + au) + Math.sqrt(at * at + au));
    return {ssmin: fhmn * c, ssmax: fhmx / c};
  }
  const au = fhmx / ga;
  if (au === 0) return {ssmin: (fhmn * fhmx) / ga, ssmax: ga};
  const as = 1 + fhmn / fhmx, at = (fhmx - fhmn) / fhmx, c = 1 / (Math.sqrt(1 + (as * au) ** 2) + Math.sqrt(1 + (at * au) ** 2));
  const m = (fhmn * c) * au;
  return {ssmin: m + m, ssmax: ga / (c + c)};
}

function dlasv2(f, g, h) {                    // SVD of [[f, g], [0, h]]: {ssmin, ssmax, snr, csr, snl, csl}
  let ft = f, fa = Math.abs(ft), ht = h, ha = Math.abs(h), pmax = 1;
  const swap = ha > fa;
  if (swap) { pmax = 3; [ft, ht] = [ht, ft]; [fa, ha] = [ha, fa]; }
  const gt = g, ga = Math.abs(gt);
  let ssmin, ssmax, clt, crt, slt, srt;
  if (ga === 0) { ssmin = ha; ssmax = fa; clt = 1; crt = 1; slt = 0; srt = 0; }
  else {
    let gasmal = true;
    if (ga > fa) {
      pmax = 2;
      if (fa / ga < EPS_LAPACK) {
        gasmal = false; ssmax = ga;
        ssmin = ha > 1 ? fa / (ga / ha) : (fa / ga) * ha;
        clt = 1; slt = ht / gt; srt = 1; crt = ft / gt;
      }
    }
    if (gasmal) {
      const d = fa - ha;
      let l = d === fa ? 1 : d / fa;
      const m = gt / ft;
      let t = 2 - l;
      const mm = m * m, tt = t * t, s = Math.sqrt(tt + mm), r = l === 0 ? Math.abs(m) : Math.sqrt(l * l + mm), a = 0.5 * (s + r);
      ssmin = ha / a; ssmax = fa * a;
      if (mm === 0) t = l === 0 ? sign(2, ft) * sign(1, gt) : gt / sign(d, ft) + m / t;
      else t = (m / (s + t) + m / (r + l)) * (1 + a);
      l = Math.sqrt(t * t + 4);
      crt = 2 / l; srt = t / l; clt = (crt + srt * m) / a; slt = (ht / ft) * srt / a;
    }
  }
  let csl, snl, csr, snr;
  if (swap) { csl = srt; snl = crt; csr = slt; snr = clt; } else { csl = clt; snl = slt; csr = crt; snr = srt; }
  let tsign;
  if (pmax === 1) tsign = sign(1, csr) * sign(1, csl) * sign(1, f);
  else if (pmax === 2) tsign = sign(1, snr) * sign(1, csl) * sign(1, g);
  else tsign = sign(1, snr) * sign(1, snl) * sign(1, h);
  ssmax = sign(ssmax, tsign);
  ssmin = sign(ssmin, tsign * sign(1, f) * sign(1, h));
  return {ssmin, ssmax, snr, csr, snl, csl};
}

// dbdsqr for an upper bidiagonal matrix (d, e), updating vt (rows) only; vt starts as the identity.
function dbdsqr(d, e, vt) {
  const n = d.length, eps = EPS_LAPACK, unfl = SAFMIN, maxitr = 6;
  const tolmul = Math.max(10, Math.min(100, eps ** -0.125)), tol = tolmul * eps;
  let smax = 0;
  for (let i = 0; i < n; i++) smax = Math.max(smax, Math.abs(d[i]));
  for (let i = 0; i < n - 1; i++) smax = Math.max(smax, Math.abs(e[i]));
  let sminoa = Math.abs(d[0]);
  if (sminoa !== 0) {
    let mu = sminoa;
    for (let i = 1; i < n; i++) { mu = Math.abs(d[i]) * (mu / (mu + Math.abs(e[i - 1]))); sminoa = Math.min(sminoa, mu); if (sminoa === 0) break; }
  }
  sminoa /= Math.sqrt(n);
  const thresh = Math.max(tol * sminoa, maxitr * (n * (n * unfl)));
  const rotRows = (r1, r2, c, s) => { for (let k = 0; k < vt[0].length; k++) { const t = c * vt[r1][k] + s * vt[r2][k]; vt[r2][k] = c * vt[r2][k] - s * vt[r1][k]; vt[r1][k] = t; } };
  // dlasr('L', 'V', direction): rotation j acts on rows (ll + j, ll + j + 1)
  const lasr = (fwd, ll, cs, sn) => {
    const cnt = cs.length;
    for (let jj = 0; jj < cnt; jj++) {
      const j = fwd ? jj : cnt - 1 - jj, c = cs[j], s = sn[j];
      if (c === 1 && s === 0) continue;
      for (let k = 0; k < vt[0].length; k++) {
        const t = vt[ll + j + 1][k];
        vt[ll + j + 1][k] = c * t - s * vt[ll + j][k];
        vt[ll + j][k] = s * t + c * vt[ll + j][k];
      }
    }
  };
  let m = n - 1, iter = 0, oldll = -1, oldm = -1, idir = 0, guard = 0;           // m, ll are 0-based indices
  const maxit = maxitr * n * n;
  main: while (m > 0) {
    if (iter++ > maxit * 4 || guard++ > 10000) break;
    let smaxl = Math.abs(d[m]), ll = -1, split = false;
    for (let lll = 0; lll < m; lll++) {
      const l = m - 1 - lll, abss = Math.abs(d[l]), abse = Math.abs(e[l]);
      if (abse <= thresh) { e[l] = 0; ll = l; split = true; break; }
      smaxl = Math.max(smaxl, abss, abse);
    }
    if (split) {
      if (ll === m - 1) { m -= 1; continue; }
      ll += 1;
    } else ll = 0;
    if (ll === m - 1) {                           // 2 x 2 block
      const r = dlasv2(d[m - 1], e[m - 1], d[m]);
      d[m - 1] = r.ssmax; e[m - 1] = 0; d[m] = r.ssmin;
      rotRows(m - 1, m, r.csr, r.snr);
      m -= 2;
      continue;
    }
    if (ll > oldm || m < oldll) idir = Math.abs(d[ll]) >= Math.abs(d[m]) ? 1 : 2;
    let smin = 0;
    if (idir === 1) {
      if (Math.abs(e[m - 1]) <= Math.abs(tol) * Math.abs(d[m])) { e[m - 1] = 0; continue; }
      let mu = Math.abs(d[ll]);
      smin = mu;
      for (let lll = ll; lll < m; lll++) {
        if (Math.abs(e[lll]) <= tol * mu) { e[lll] = 0; continue main; }
        mu = Math.abs(d[lll + 1]) * (mu / (mu + Math.abs(e[lll])));
        smin = Math.min(smin, mu);
      }
    } else {
      if (Math.abs(e[ll]) <= Math.abs(tol) * Math.abs(d[ll])) { e[ll] = 0; continue; }
      let mu = Math.abs(d[m]);
      smin = mu;
      for (let lll = m - 1; lll >= ll; lll--) {
        if (Math.abs(e[lll]) <= tol * mu) { e[lll] = 0; continue main; }
        mu = Math.abs(d[lll]) * (mu / (mu + Math.abs(e[lll])));
        smin = Math.min(smin, mu);
      }
    }
    oldll = ll; oldm = m;
    let shift;
    if (n * tol * (smin / smaxl) <= Math.max(eps, 0.01 * tol)) shift = 0;
    else {
      let sll;
      if (idir === 1) { sll = Math.abs(d[ll]); shift = dlas2(d[m - 1], e[m - 1], d[m]).ssmin; }
      else { sll = Math.abs(d[m]); shift = dlas2(d[ll], e[ll], d[ll + 1]).ssmin; }
      if (sll > 0 && (shift / sll) ** 2 < eps) shift = 0;
    }
    const len = m - ll;
    if (shift === 0) {
      if (idir === 1) {
        let cs = 1, oldcs = 1, oldsn = 0;
        const w1 = new Array(len), w2 = new Array(len), w3 = new Array(len), w4 = new Array(len);
        for (let i = ll; i < m; i++) {
          let g1 = dlartg(d[i] * cs, e[i]); cs = g1.c; const sn = g1.s, r = g1.r;
          if (i > ll) e[i - 1] = oldsn * r;
          const g2 = dlartg(oldcs * r, d[i + 1] * sn); oldcs = g2.c; oldsn = g2.s; d[i] = g2.r;
          w1[i - ll] = cs; w2[i - ll] = sn; w3[i - ll] = oldcs; w4[i - ll] = oldsn;
        }
        const h = d[m] * cs; d[m] = h * oldcs; e[m - 1] = h * oldsn;
        lasr(true, ll, w1, w2);
        if (Math.abs(e[m - 1]) <= thresh) e[m - 1] = 0;
      } else {
        let cs = 1, oldcs = 1, oldsn = 0;
        const w1 = new Array(len), w2 = new Array(len), w3 = new Array(len), w4 = new Array(len);
        for (let i = m; i > ll; i--) {
          const g1 = dlartg(d[i] * cs, e[i - 1]); cs = g1.c; const sn = g1.s, r = g1.r;
          if (i < m) e[i] = oldsn * r;
          const g2 = dlartg(oldcs * r, d[i - 1] * sn); oldcs = g2.c; oldsn = g2.s; d[i] = g2.r;
          w1[i - ll - 1] = cs; w2[i - ll - 1] = -sn; w3[i - ll - 1] = oldcs; w4[i - ll - 1] = -oldsn;
        }
        const h = d[ll] * cs; d[ll] = h * oldcs; e[ll] = h * oldsn;
        lasr(false, ll, w3, w4);
        if (Math.abs(e[ll]) <= thresh) e[ll] = 0;
      }
    } else if (idir === 1) {
      let f = (Math.abs(d[ll]) - shift) * (sign(1, d[ll]) + shift / d[ll]), g = e[ll];
      const w1 = new Array(len), w2 = new Array(len);
      for (let i = ll; i < m; i++) {
        const g1 = dlartg(f, g), cosr = g1.c, sinr = g1.s;
        if (i > ll) e[i - 1] = g1.r;
        f = cosr * d[i] + sinr * e[i];
        e[i] = cosr * e[i] - sinr * d[i];
        g = sinr * d[i + 1];
        d[i + 1] = cosr * d[i + 1];
        const g2 = dlartg(f, g), cosl = g2.c, sinl = g2.s;
        d[i] = g2.r;
        f = cosl * e[i] + sinl * d[i + 1];
        d[i + 1] = cosl * d[i + 1] - sinl * e[i];
        if (i < m - 1) { g = sinl * e[i + 1]; e[i + 1] = cosl * e[i + 1]; }
        w1[i - ll] = cosr; w2[i - ll] = sinr;
      }
      e[m - 1] = f;
      lasr(true, ll, w1, w2);
      if (Math.abs(e[m - 1]) <= thresh) e[m - 1] = 0;
    } else {
      let f = (Math.abs(d[m]) - shift) * (sign(1, d[m]) + shift / d[m]), g = e[m - 1];
      const w1 = new Array(len), w2 = new Array(len), w3 = new Array(len), w4 = new Array(len);
      for (let i = m; i > ll; i--) {
        const g1 = dlartg(f, g), cosr = g1.c, sinr = g1.s;
        if (i < m) e[i] = g1.r;
        f = cosr * d[i] + sinr * e[i - 1];
        e[i - 1] = cosr * e[i - 1] - sinr * d[i];
        g = sinr * d[i - 1];
        d[i - 1] = cosr * d[i - 1];
        const g2 = dlartg(f, g), cosl = g2.c, sinl = g2.s;
        d[i] = g2.r;
        f = cosl * e[i - 1] + sinl * d[i - 1];
        d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
        if (i > ll + 1) { g = sinl * e[i - 2]; e[i - 2] = cosl * e[i - 2]; }
        w1[i - ll - 1] = cosr; w2[i - ll - 1] = -sinr; w3[i - ll - 1] = cosl; w4[i - ll - 1] = -sinl;
      }
      e[ll] = f;
      if (Math.abs(e[ll]) <= thresh) e[ll] = 0;
      lasr(false, ll, w3, w4);
    }
  }
  for (let i = 0; i < n; i++) if (d[i] < 0) { d[i] = -d[i]; for (let k = 0; k < vt[0].length; k++) vt[i][k] = -vt[i][k]; }
  for (let i = 1; i < n; i++) {                   // sort into decreasing order, one transposition per vector
    let isub = 0, smin = d[0];
    for (let j = 1; j <= n - i; j++) if (d[j] <= smin) { isub = j; smin = d[j]; }
    if (isub !== n - i) { d[isub] = d[n - i]; d[n - i] = smin; [vt[isub], vt[n - i]] = [vt[n - i], vt[isub]]; }
  }
}

// X: flat n x 3 (row-major), n >= 5. Returns {vt: 3 rows (right singular vectors), s: singular values}.
export function svd3(X, n) {
  const A = [0, 1, 2].map(j => Float64Array.from({length: n}, (_, i) => X[3 * i + j]));   // column-major
  for (let j = 0; j < 3; j++) {                                                           // dgeqr2
    let xn = 0;
    for (let i = j + 1; i < n; i++) xn += A[j][i] * A[j][i];
    const {beta, tau, scale} = dlarfg(A[j][j], Math.sqrt(xn));
    for (let i = j + 1; i < n; i++) A[j][i] *= scale;
    A[j][j] = 1;
    for (let c = j + 1; c < 3; c++) {
      let w = 0;
      for (let i = j; i < n; i++) w += A[j][i] * A[c][i];
      for (let i = j; i < n; i++) A[c][i] -= tau * A[j][i] * w;
    }
    A[j][j] = beta;
  }
  const R = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];                                            // R[row][col], zero below the diagonal
  for (let c = 0; c < 3; c++) for (let r = 0; r <= c; r++) R[r][c] = A[c][r];
  // dgebd2 on R (3 x 3, upper bidiagonal)
  const d = [0, 0, 0], e = [0, 0];
  let taup0 = 0, x0 = 0;
  for (let i = 0; i < 3; i++) {
    let xn = 0;
    for (let r = i + 1; r < 3; r++) xn += R[r][i] * R[r][i];
    const q = dlarfg(R[i][i], Math.sqrt(xn));
    const v = [1, ...[...Array(2 - i).keys()].map(t => R[i + 1 + t][i] * q.scale)];       // v = (1, scaled x)
    d[i] = q.beta;
    for (let c = i + 1; c < 3; c++) {                                                     // H(i) from the left on R[i.., c]
      let w = 0;
      for (let t = 0; t < v.length; t++) w += v[t] * R[i + t][c];
      for (let t = 0; t < v.length; t++) R[i + t][c] -= q.tau * v[t] * w;
    }
    R[i][i] = d[i];
    if (i < 2) {
      let xp = 0;
      for (let c = i + 2; c < 3; c++) xp += R[i][c] * R[i][c];
      const p = dlarfg(R[i][i + 1], Math.sqrt(xp));
      e[i] = p.beta;
      const u = [1, ...[...Array(1 - i).keys()].map(t => R[i][i + 2 + t] * p.scale)];       // u = (1, scaled x)
      for (let r = i + 1; r < 3; r++) {                                                   // G(i) from the right on R[r, i+1..]
        let w = 0;
        for (let t = 0; t < u.length; t++) w += R[r][i + 1 + t] * u[t];
        for (let t = 0; t < u.length; t++) R[r][i + 1 + t] -= p.tau * w * u[t];
      }
      if (i === 0) { taup0 = p.tau; x0 = u[1]; }
    }
  }
  const vt = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  dbdsqr(d, e, vt);
  // VT = VT_B * P^T, P = G(0) (G(1) is the identity for n = 3), G(0) = I - tau v v^T with v = (0, 1, x0)
  const v = [0, 1, x0], out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let r = 0; r < 3; r++) {
    let w = 0;
    for (let k = 0; k < 3; k++) w += vt[r][k] * v[k];
    for (let k = 0; k < 3; k++) out[r][k] = vt[r][k] - taup0 * w * v[k];
  }
  return {vt: out, s: d};
}
