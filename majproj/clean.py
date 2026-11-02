"""Find the scanned object in one DeepSonar export (`scan_*_shapenet.npy`) by defining the SIGNAL, not the noise.

Where the file comes from. A VL53L5CX 8x8 time-of-flight sensor (~45 deg field of view, ~5.6 deg per zone, 15 Hz) and
an IMU, held in the hand. The viewer registers frames by IMU ROTATION ONLY (translation is never tracked), keeps a
10 mm voxel-thinned map, and exports 4096 points sampled from that map (with replacement when the map is smaller, so
duplicates exist), bounding-box centred and divided by the largest coordinate. Point order is random; there are no
timestamps. Frame: z = UP (gravity), x / y horizontal ("pointing UP (+Z), lying flat" in the viewer docstring).

Mapping runs from before the scan starts until after it ends, so one file is the time-union of: the ceiling seen
while the device lay pointing up (sparse zone-grid strips high in z), streaks while it was picked up / put down
(line-like, a few points per zone), near-sensor junk (hand, table edge, cover glass), shifted depth COPIES of the
object from hand drift toward / away from it (translation is untracked), arbitrary other noise, and THE OBJECT.

The signal. The object is the surface the user held the sensor on for seconds. Because the map is voxel-thinned,
seconds of dwell SATURATE the voxel grid on that surface, so the object is
  dense    - in the file's highest density mode (other things are seen briefly and stay sparse),
  surface  - locally a smooth 2-manifold: not a line (streak) and not a volume (scatter, a pile of drift copies),
  compact  - one connected blob that fills its own footprint, and is not a wide gravity-aligned plane spanning the
             scene (ceiling / floor / table),
  unique   - clearly stronger than any other candidate; two comparable candidates mean we cannot tell which one the
             user meant, so the file is REJECTED (never silently completed).
A box shows up as one vertical rectangular sheet (its front face); a bottle as a curved front patch.

Every threshold comes from the file's own point spacing or from sensor physics (no labels, no learned parts):
  s        = median distance to the 8th neighbour among the densest fifth of the points: the spacing of a fully
             saturated surface in THIS file's units (the per-file normalisation makes absolute sizes meaningless).
  eps      = 3 s, the neighbourhood radius. On a saturated surface an eps-disc holds about 8 * 3^2 = 72 points; on a
             line at the same spacing about 8 * 3 = 24; a briefly seen zone strip far fewer.
  d_top    = mean neighbour count of the `min_object_pts` densest points: the saturated density level actually reached
             in this file. Core points have >= `core_frac` * d_top neighbours (growth may pass >= `grow_frac` * d_top).
  min_object_pts = 64, one full 8x8 frame of distinct samples: a dwell of seconds at 15 Hz gives far more, so a
             patch smaller than one frame cannot be a scanned object.
  band     = 1.5 eps: how far from the current surface model a point may lie while the patch grows.
  45 deg   = a grown point's own local normal must be closer to the surface normal there than to a perpendicular one,
             so growth follows a curved surface but does not turn a 90 deg corner onto the table.
  layer    = the residual density is smoothed at s / 4 and the layer sd is floored at s / 2 (voxel quantisation and
             ranging noise make a thinner layer unobservable); a layer thicker than 2 s is not one surface.
  contrast = 3: seconds of dwell against brief glimpses; spatially uniform noise has contrast ~1.

Steps (deterministic; the output does not depend on point order):
  1. drop non-finite rows and duplicates; compute s, eps, per-point density, local PCA linearity (16 neighbours) and
     local normals / surface variation (d_top / 2 neighbours, wide enough to span several sensor zones);
  2. core points = dense AND not line-like (linearity <= `max_linearity`: streaks and zone strips are lines);
  3. candidates, strongest first: the connected component (radius eps) of not-yet-claimed core points with the largest
     size x median density, seeded at its densest point weighted by how planar the seed's own neighbourhood is;
  4. grow from the seed through unclaimed non-line points within `band` of a smooth surface model (a plane in the
     first round, then a quadric height field w = f(u, v) in the patch's own PCA frame, re-fitted each round) whose
     local normal agrees with the model's; the grown patch is claimed so the next candidate is a different structure;
  5. mode layer: hand drift stores copies of the surface at other depths. The residual of every grown point along the
     LOCAL normal of the fitted surface (a curved surface has a different normal at every point) gives a density
     whose peak is the dominant layer; points within `layer_k` spreads of it are kept and the model is re-fitted
     (3 rounds, always over the whole grown patch); then the part connected to the densest point, then statistical
     outlier removal. A weaker candidate parallel to a stronger one and lying over its footprint is the same surface
     at another depth: it is folded into the stronger one's depth-copy count instead of competing with it.
     Drifting hand (only with a known sensor `origin`): a hand that keeps creeping along the line of sight leaves a
     PLATEAU of depths instead of a peak (real box session 121018/121022/121055: 6 cm closer during one dwell, so the
     half-maximum width of the layer exceeds `max_layer_sd`). A surface can be dragged along its line of sight but
     cannot be as thick as it is wide, so when the patch faces the sensor (`normal_cos`) and its 5-95 depth range is at
     most `drift_aspect` x its smaller in-plane 5-95 extent, the layer becomes the densest one-surface-thick slab
     (half-width `layer_k` * s / 2, the floored spread: the depth the hand held longest) and the rest counts as depth
     copies. meta['layer_mode'] = 'drift_slab' marks such a file (the key is absent otherwise). That slab is a
     measured depth of the session, not the true depth of the object, which a drifting hand does not determine. The
     `smooth` test passes there by construction (the slab's width sets its spread), so it is no evidence of thinness;
     read `confidence_terms['purity']` instead;
  6. tests on each candidate: size, dense, contrast (vs the rest of the file excluding itself and any wide scene
     plane), smooth (layer sd), not_line (in-plane sd2 / sd1), filled (occupied fraction of its own footprint),
     upright (the patch is not horizontal: normal more than 30 deg from gravity, since a table / floor / shelf /
     ceiling is never the object); not_scene_plane (a patch whose normal is within 30 deg of gravity spanning >= half the scene);
     score = grown size x median density (drift copies are dwell evidence too). The best passing candidate wins; the
     file is REJECTED if none passes, or if the runner-up passing candidate scores >= `ambiguity` x the best.

meta['tests'] holds the winner's tests plus 'isolated' (value = 1 - runner-up score / best score, pass when the
ratio is < `ambiguity`). shape_hint = 'curved' iff the fitted quadric bends at least one spacing s across the patch
(`sag` >= s) AND it explains the patch clearly better than a plane (plane rms >= 1.3 x quadric rms), else 'flat'.
width / height are the patch's extents along the horizontal tangent and along the in-plane direction closest to
gravity (file units); centre is the patch mean; plane_sd the sd of the plane-fit residual; `normal` the mean normal.

Confidence in [0, 1] is the geometric mean of measurable terms: count 1 - exp(-n / 256) (four frames' worth),
smooth exp(-(layer sd / s)^2), isolation 1 - (runner-up score / best score) / ambiguity (0 at the reject boundary),
coverage min(1, fill / 0.8) and purity (fraction of the grown patch in the dominant layer).

Known limits (checked on the 15 real scans of 2026-10-15 / 2026-10-27):
  * the sign of `normal` is a convention, not observed (see `_orient`);
  * at this resolution a narrow bottle's front arc sags by only a few s, so `shape_hint` curved vs flat is marginal
    (some bottles read flat), and a doubly curved blob is not separable from a bottle by geometry alone;
  * when two drift layers have nearly equal weight, which one is "dominant" is deterministic but fragile;
  * a file whose densest structure is not what the user meant (e.g. a tilted patch denser than the real face) can be
    accepted; it then shows as LOW confidence through the isolation and smoothness terms, not as a rejection;
  * an object seen only for a moment is indistinguishable from noise and is rejected;
  * with a drifting hand and no known origin the plateau stays a `smooth` failure (exports have lost the sensor position).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

UP = np.array([0.0, 0.0, 1.0])

DEFAULTS = dict(
    eps_factor=3.0,        # eps = eps_factor * s
    min_object_pts=64,     # one full 8x8 frame of distinct samples
    core_frac=0.25,        # core point: density >= core_frac * d_top
    grow_frac=0.08,        # growth may pass through points with density >= grow_frac * d_top (patch edges)
    dense_frac=0.3,        # dense test (a floor against zone strips / streaks, not the selector): median density of
                           # the grown patch >= dense_frac * d_top
    min_contrast=3.0,      # contrast test: median density of the grown patch >= 3 x that of the rest of the file
                           # (excluding wide scene planes)
                           # (seconds of dwell vs brief glimpses; uniform noise has contrast ~1)
    max_linearity=0.85,    # local PCA linearity above this = streak point
    band=1.5,              # growth band, in eps
    normal_cos=np.cos(np.radians(45)),  # a grown point's local normal must be closer to the surface normal there
                                        # than to a perpendicular one (stops growth round a corner onto the table)
    layer_k=2.5,           # dominant layer = peak +- layer_k * spread
    max_layer_sd=2.0,      # surface test: sd of the dominant layer along local normals <= 2 s (voxel quantisation +
                           # ranging noise give ~1 s; one physical surface is not twice that thick)
    drift_aspect=0.85,     # with origin: a depth plateau below this x the patch's smaller in-plane extent is hand drift
                           # along the line of sight; a volume measures about 1.0 on these 5-95 percentile ranges, the one
                           # real drifting session 0.70, so the margin rests on that single session
    min_aspect=0.2,        # surface test: in-plane sd2 / sd1 >= this (a line is not a surface)
    min_aspect_rod=0.05,   # with origin: the same test for a patch standing within rod_cos of vertical. A rod or pipe seen
                           # at 8x8 is a line, and failing it would hand the scan to whatever passes behind it. Swing
                           # streaks are not held on a vertical line for seconds, so `contrast` still separates them
    rod_cos=np.cos(np.radians(25)),
    min_fill=0.35,         # compact test: occupied fraction of the patch's own footprint
    horiz_cos=np.cos(np.radians(30)),  # a plane whose normal is within 30 deg of gravity is horizontal
    max_span=0.5,          # a horizontal patch spanning >= this fraction of the scene extent = ceiling/floor/table
    origin=None,           # sensor position (x, y, z) if known (raw metric maps keep it at the map origin; exports do not)
    min_range=0.12,        # with origin: closer than this is hand / cover glass, never the target (metres)
    behind_ratio=1.5,      # with origin: a candidate this many times farther away than the nearest one is background
    ambiguity=0.5,         # reject if runner-up score >= ambiguity * best score
    sor_k=10, sor_sd=2.5,  # statistical outlier removal inside the final patch
    max_candidates=8,      # candidates examined per file (the object is among the strongest few)
)


@dataclass
class Cleaned:
    object: np.ndarray            # K x 3 object points (file coordinates), float32
    removed: np.ndarray           # M x 3 everything else (unique points), float32
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- geometry helpers
def _pca(q):
    c = q.mean(0)
    _, s, vt = np.linalg.svd(q - c, full_matrices=False)
    return c, vt, s / np.sqrt(len(q))              # centre, principal axes (rows, last = normal), per-axis sd


def _local_pca(p, tree, k):
    """Per point, PCA of its k nearest neighbours: linearity (l1 - l2) / l1 and the local normal (smallest axis)."""
    k = int(min(k, len(p)))
    idx = tree.query(p, k)[1]
    x = p[idx] - p[idx].mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', x, x) / k
    ev, vec = np.linalg.eigh(cov)                  # ascending
    lin = (ev[:, 2] - ev[:, 1]) / np.maximum(ev[:, 2], 1e-18)
    var = ev[:, 0] / np.maximum(ev.sum(1), 1e-18)   # surface variation: 0 on a plane, 1/3 for isotropic scatter
    return lin, vec[:, :, 0], var


class _Surface:
    """Plane or quadric height field w = f(u, v) in a fixed PCA frame; residuals along the local normal."""

    def __init__(self, q, quadric):
        self.c, self.ax, _ = _pca(q)
        u, v, w = self._uvw(q)
        self.quadric = bool(quadric) and len(q) >= 30
        A = self._design(u, v)
        self.coef = np.linalg.lstsq(A, w, rcond=None)[0]

    def _uvw(self, x):
        d = (x - self.c) @ self.ax.T
        return d[:, 0], d[:, 1], d[:, 2]

    def _design(self, u, v):
        one = np.ones_like(u)
        if self.quadric:
            return np.stack([one, u, v, u * u, u * v, v * v], 1)
        return np.stack([one, u, v], 1)

    def residual(self, x):
        u, v, w = self._uvw(x)
        f = self._design(u, v) @ self.coef
        a = self.coef
        if self.quadric:
            fu, fv = a[1] + 2 * a[3] * u + a[4] * v, a[2] + a[4] * u + 2 * a[5] * v
        else:
            fu, fv = np.full_like(u, a[1]), np.full_like(u, a[2])
        return (w - f) / np.sqrt(1 + fu * fu + fv * fv)

    def normals(self, x):
        u, v, _ = self._uvw(x)
        a = self.coef
        if self.quadric:
            fu, fv = a[1] + 2 * a[3] * u + a[4] * v, a[2] + a[4] * u + 2 * a[5] * v
        else:
            fu, fv = np.full_like(u, a[1]), np.full_like(u, a[2])
        n = -fu[:, None] * self.ax[0] - fv[:, None] * self.ax[1] + self.ax[2]
        return n / np.linalg.norm(n, axis=1, keepdims=True)

    def sag(self, x):
        """Peak-to-peak height of the quadratic part over the points: how curved the patch is."""
        if not self.quadric:
            return 0.0
        u, v, _ = self._uvw(x)
        a = self.coef
        qd = a[3] * u * u + a[4] * u * v + a[5] * v * v
        lin = np.stack([np.ones_like(u), u, v], 1)
        qd = qd - lin @ np.linalg.lstsq(lin, qd, rcond=None)[0]   # remove the part a plane already explains
        return float(np.ptp(qd))


def _bfs(nb, ok, seed):
    """Indices reachable from seed through neighbour lists, visiting only points where ok is True."""
    seen = {int(seed)}
    fr = [int(seed)]
    while fr:
        i = fr.pop()
        for j in nb[i]:
            if ok[j] and j not in seen:
                seen.add(j)
                fr.append(j)
    return np.array(sorted(seen))


def _mode_layer(r, s, k):
    """Dominant layer of the residuals r: the peak of their density (histogram in bins of s / 8, smoothed by a
    Gaussian of sd s / 4). The layer's spread is its full width
    at half maximum / 2.355 (the sd of a Gaussian layer), with the smoothing removed in quadrature and floored at s / 2.
    It is not truncated by a fixed window, so a volumetric blob shows up as a wide layer instead of being clipped to
    look thin. Returns (keep mask = within k spreads of the peak, spread)."""
    if len(r) < 5:
        return np.ones(len(r), bool), float(np.std(r))
    h, sg = s / 8.0, s / 4.0
    bins = np.arange(r.min() - 4 * sg, r.max() + 4 * sg + h, h)
    hist, edges = np.histogram(r, bins)
    x = np.arange(-int(3 * sg / h), int(3 * sg / h) + 1) * h
    ker = np.exp(-0.5 * (x / sg) ** 2)
    sm = np.convolve(hist, ker / ker.sum(), 'same')
    mid = 0.5 * (edges[:-1] + edges[1:])
    j = int(np.argmax(sm))
    half = sm[j] / 2.0
    lo = j
    while lo > 0 and sm[lo - 1] >= half:
        lo -= 1
    hi = j
    while hi < len(sm) - 1 and sm[hi + 1] >= half:
        hi += 1
    # linear interpolation of the two half-maximum crossings
    xl = mid[lo] - h * (sm[lo] - half) / max(sm[lo] - sm[lo - 1], 1e-12) if lo > 0 else mid[lo]
    xr = mid[hi] + h * (sm[hi] - half) / max(sm[hi] - sm[hi + 1], 1e-12) if hi < len(sm) - 1 else mid[hi]
    peak = float(mid[j])
    sd_obs = (xr - xl) / 2.355
    spread = max(float(np.sqrt(max(sd_obs ** 2 - sg ** 2, 0.0))), s / 2.0)
    return np.abs(r - peak) <= k * spread, spread


def _densest_slab(r, half):
    """Mask of the residuals inside the window of half-width `half` that holds the most of them (ties: lowest)."""
    rs = np.sort(r)
    count = np.searchsorted(rs, rs + 2 * half, side='right') - np.arange(len(rs))
    centre = rs[int(np.argmax(count))] + half
    return np.abs(r - centre) <= half


def _is_drift_plateau(q, r, surf, origin, P):
    """True when a too-thick layer is a surface dragged along the line of sight, not a volume: the patch faces the
    known sensor position and its depth range is small against its in-plane extents."""
    if origin is None:
        return False
    sight = q.mean(0) - np.asarray(origin, float)
    sight /= max(np.linalg.norm(sight), 1e-12)
    nrm = surf.normals(q).mean(0)
    nrm /= np.linalg.norm(nrm)
    uv = (q - surf.c) @ surf.ax[:2].T
    extent = float(np.min(np.percentile(uv, 95, 0) - np.percentile(uv, 5, 0)))
    depth = float(np.percentile(r, 95) - np.percentile(r, 5))
    return abs(float(nrm @ sight)) >= P['normal_cos'] and depth <= P['drift_aspect'] * extent


def _absorb_depth_copies(cands, normal_cos):
    """A weaker candidate parallel to a stronger one and lying over its footprint is the same surface seen at another
    depth (hand drift along the view direction; translation is untracked). It is not a competitor: fold it into the
    stronger candidate's depth-copy count. Candidates must be sorted strongest first."""
    out = []
    for c in cands:
        host = None
        for b in out:
            if not b['passed']:            # a rejected candidate (e.g. the table under the object) hosts nothing
                continue
            nb_, nc = b['surf'].normals(b['q']).mean(0), c['surf'].normals(c['q']).mean(0)
            parallel = abs(nb_ @ nc) / (np.linalg.norm(nb_) * np.linalg.norm(nc)) >= normal_cos
            uvb = (b['q'] - b['surf'].c) @ b['surf'].ax[:2].T
            uvc = (c['q'] - b['surf'].c) @ b['surf'].ax[:2].T
            lo, hi = np.percentile(uvb, 2, 0), np.percentile(uvb, 98, 0)
            over = float(np.mean(np.all((uvc >= lo) & (uvc <= hi), 1)))
            depth = float(np.median(np.abs((c['q'] - b['surf'].c) @ b['surf'].ax[2])))
            # a curved surface's drift copies need not be parallel to it: a candidate whose centre lies inside the
            # stronger one's footprint, close in depth, is the same object too
            inside = np.linalg.norm(np.asarray(c['centre']) - np.asarray(b['centre'])) <= 0.25 * float(np.max(hi - lo))
            if over >= 0.5 and depth <= float(np.max(hi - lo)) and (parallel or inside):
                host = b
                break
        if host is None:
            out.append(c)
        else:
            host['n_copies'] += c['n']
            host.setdefault('absorbed', []).append(c['centre'])
    return out


def _fill(uv, cell):
    """Occupied fraction of the patch's own footprint (its 2-98 percentile box, cells of size `cell`)."""
    lo, hi = np.percentile(uv, 2, 0), np.percentile(uv, 98, 0)
    inside = np.all((uv >= lo) & (uv <= hi), 1)
    if inside.sum() < 3:
        return 0.0
    g = np.floor((uv[inside] - lo) / cell).astype(int)
    nx, ny = g.max(0) + 1
    return float(len(np.unique(g[:, 0] * ny + g[:, 1])) / (nx * ny))


# ----------------------------------------------------------------------------- main entry
def clean(points, **params):
    P = dict(DEFAULTS)
    unknown = set(params) - set(P)
    if unknown:
        raise TypeError(f'unknown clean() parameters: {sorted(unknown)}')
    P.update(params)

    raw = np.asarray(points, np.float64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise ValueError(f'expected an N x 3 point cloud, got shape {raw.shape}')
    p = raw[np.isfinite(raw).all(1)]
    p = np.unique(p, axis=0)                       # sorted -> independent of input order
    n = len(p)
    meta = dict(status='rejected', reason=None, confidence=0.0, n_input=int(len(raw)), n_unique=int(n),
                n_object=0, n_removed=int(n), eps=None, tests={}, normal=None, centre=None, width=None, height=None,
                shape_hint=None, candidates=[], frame='z up', n_depth_copies=0, aspect_w_over_h=None, plane_sd=None)

    def reject(reason):
        meta['reason'] = reason
        return Cleaned(np.empty((0, 3), np.float32), p.astype(np.float32), meta)

    if n < max(P['min_object_pts'], 16):
        return reject(f'too few points ({n} unique)')

    tree = cKDTree(p)
    d8 = tree.query(p, 9)[0][:, -1]
    s = float(np.median(d8[d8 <= np.percentile(d8, 20)]))
    if not s > 0:
        return reject('degenerate point spacing')
    eps = P['eps_factor'] * s
    meta['eps'] = round(eps, 5)
    meta['spacing'] = round(s, 5)
    nb = tree.query_ball_point(p, eps)
    dens = np.array([len(x) - 1 for x in nb])
    lin, _, _ = _local_pca(p, tree, 16)
    d_top = float(np.sort(dens)[-P['min_object_pts']:].mean())
    meta['d_top'] = round(d_top, 1)
    # local normals over about half a saturated eps-disc: wide enough to span several sensor zones (k = 16 would sit
    # inside one zone strip on a face and give an arbitrary normal)
    _, lnorm, lvar = _local_pca(p, tree, max(16, d_top / 2))
    scene_ext = float(np.max(np.ptp(p, 0)))

    not_line = lin <= P['max_linearity']
    core = (dens >= P['core_frac'] * d_top) & not_line
    growable = (dens >= P['grow_frac'] * d_top) & not_line

    # Candidates: connected components (radius graph) of the core points not yet explained by an earlier candidate,
    # strongest first (size x median density). Each is grown from its best seed: the densest point weighted by how
    # surface-like its own neighbourhood is (a seed on a corner would start the surface model at 45 degrees).
    rows = np.repeat(np.arange(n), [len(x) for x in nb])
    cols = np.concatenate([np.asarray(x, int) for x in nb])
    seed_score = dens * np.clip(1 - 3 * lvar, 0, 1)
    avail = core.copy()
    free = growable.copy()                         # not yet claimed by an earlier candidate
    cands = []
    for _ in range(2 * P['max_candidates']):
        if len(cands) >= P['max_candidates']:
            break
        m = avail[rows] & avail[cols]
        G = coo_matrix((np.ones(m.sum()), (rows[m], cols[m])), shape=(n, n))
        _, lab = connected_components(G, directed=False)
        best = None
        for c in np.unique(lab[avail]):
            idx = np.nonzero((lab == c) & avail)[0]
            if len(idx) < P['min_object_pts'] // 2:
                continue
            st = len(idx) * float(np.median(dens[idx]))
            if best is None or st > best[0]:
                best = (st, idx)
        if best is None:
            break
        comp = best[1]
        seed = int(comp[np.lexsort((comp, -seed_score[comp]))[0]])
        cand = _grow_and_measure(p, nb, dens, free, lnorm, seed, s, eps, d_top, scene_ext, P)
        claimed = comp if cand is None else cand['grown']
        avail[claimed] = False
        free[claimed] = False
        avail[seed] = False
        if cand is not None:
            cands.append(cand)

    # contrast: a dwelt-on surface must stand out from the rest of the file. "The rest" excludes the candidate itself
    # and any wide gravity-aligned scene plane (a saturated table / floor is explained structure, not background)
    scene = np.zeros(n, bool)
    for c in cands:
        if not c['tests']['not_scene_plane']['pass']:
            scene[c['grown']] = True
    for c in cands:
        rest = ~scene
        rest[c['grown']] = False
        bg = float(np.median(dens[rest])) if rest.any() else 1.0
        con = c['med_dens'] / max(bg, 1.0)
        c['contrast'] = round(con, 2)
        c['tests']['contrast'] = {'value': round(con, 4), 'pass': bool(con >= P['min_contrast'])}
        c['failed'] = [k for k, t_ in c['tests'].items() if not t_['pass']]
        c['passed'] = not c['failed']
    cands.sort(key=lambda c: -c['score'])
    cands = _absorb_depth_copies(cands, P['normal_cos'])
    meta['candidates'] = [{k: c[k] for k in ('n', 'score', 'passed', 'failed', 'centre', 'normal_z', 'rms_over_s', 'layer_sd_over_s',
                                            'median_density_frac', 'contrast', 'layer_frac', 'fill')} for c in cands]
    passing = [c for c in cands if c['passed']]
    if not cands:
        return reject('no dense surface candidate (nothing reaches the saturated density level)')
    if not passing:
        b = cands[0]
        meta['tests'] = b['tests']
        return reject(f"no candidate passes all signal tests; best candidate fails: {', '.join(b['failed'])}")
    if P['origin'] is not None:
        # The user aims at the object, so with a known sensor position the target is the nearest dwelled-on surface;
        # surfaces well behind it (furniture, walls) are background, not rivals, however dense they are.
        o = np.asarray(P['origin'], float)
        rng_ = lambda c: float(np.linalg.norm(np.asarray(c['centre'])[:2] - o[:2]))
        inrange = [c for c in passing if rng_(c) >= P['min_range']]
        if inrange:
            near = min(rng_(c) for c in inrange)
            passing = [c for c in inrange if rng_(c) < P['behind_ratio'] * near]
            meta['range_m'] = round(near, 4)
    best = passing[0]
    ratio = passing[1]['score'] / best['score'] if len(passing) > 1 else 0.0
    meta['tests'] = dict(best['tests'])
    meta['tests']['isolated'] = {'value': round(1 - ratio, 4), 'pass': bool(ratio < P['ambiguity'])}
    if ratio >= P['ambiguity']:
        return reject(f'ambiguous: two comparable object candidates (runner-up score {ratio:.2f} x the best)')

    idx = best['idx']
    q = p[idx]
    surf = best['surf']
    normal = _orient(surf, q, P['horiz_cos'])
    centre = q.mean(0)
    # in-plane extents: width along the horizontal tangent, height along the in-plane direction closest to gravity
    t_h = np.cross(UP, normal)
    if np.linalg.norm(t_h) < 1e-6:                 # facing straight up/down: use the patch's own principal axes
        t_h, t_v = surf.ax[0], surf.ax[1]
    else:
        t_h /= np.linalg.norm(t_h)
        t_v = np.cross(normal, t_h)
        t_v /= np.linalg.norm(t_v)
    width = float(np.ptp((q - centre) @ t_h))
    height = float(np.ptp((q - centre) @ t_v))
    _, _, sd_plane = _pca(q)
    rms = best['rms']
    terms = dict(count=1 - np.exp(-len(idx) / 256.0), smooth=np.exp(-(best['layer_sd'] / s) ** 2), isolation=1 - ratio / P['ambiguity'],
                 coverage=min(1.0, best['fill'] / 0.8), purity=best['layer_frac'])
    conf = float(np.exp(np.mean(np.log(np.clip(list(terms.values()), 1e-6, 1)))))
    mask = np.zeros(n, bool)
    mask[idx] = True
    meta.update(status='ok', reason=None, confidence=round(conf, 4), confidence_terms={k: round(float(v), 4) for k, v in terms.items()},
                n_object=int(mask.sum()), n_removed=int((~mask).sum()), n_depth_copies=int(best['n_copies']),
                normal=[round(float(v), 4) for v in normal], normal_sign='convention (+y, or +z if facing up/down)', centre=[round(float(v), 4) for v in centre],
                width=round(width, 4), height=round(height, 4),
                aspect_w_over_h=round(width / height, 3) if height > 0 else None,
                plane_sd=round(float(sd_plane[2]), 5), surface_rms=round(float(rms), 5),
                sag=round(best['sag'], 5), shape_hint=best['shape'])
    if best['drift_slab']:
        meta['layer_mode'] = 'drift_slab'
    return Cleaned(q.astype(np.float32), p[~mask].astype(np.float32), meta)


def _orient(surf, q, horiz_cos):
    """Unit normal of the patch (mean of the fitted surface's local normals) with a FIXED sign convention. The file
    keeps no sensor pose and no timestamps, and on the real scans neither occlusion (which side the nearby table / junk
    lies on) nor convexity gave a consistent answer, so the sign is not claimed to be observed: a mostly horizontal
    normal (a face seen from the side) points to +y in the file frame (the v1 convention, so displays do not flip);
    a mostly vertical one (|n_z| >= horiz_cos, a face seen from above or below) points up (+z)."""
    nrm = surf.normals(q).mean(0)
    nrm /= np.linalg.norm(nrm)
    if abs(nrm[2]) >= horiz_cos:
        return nrm if nrm[2] >= 0 else -nrm
    return nrm if nrm[1] >= 0 else -nrm


def _grow_and_measure(p, nb, dens, growable, lnorm, seed, s, eps, d_top, scene_ext, P):
    tol = P['band'] * eps
    hood = np.array(sorted({j for i in nb[seed] for j in nb[i]}))
    if len(hood) < 10:
        return None
    surf = _Surface(p[hood], quadric=False)
    sel = hood
    for rnd in range(5):
        ok = growable.copy()
        ok[seed] = True
        # restrict to the band around the current model (residual along local normals)
        cand = np.nonzero(ok)[0]
        r = surf.residual(p[cand])
        agree = np.abs(np.sum(lnorm[cand] * surf.normals(p[cand]), 1)) >= P['normal_cos']
        ok[cand[(np.abs(r) > tol) | ~agree]] = False
        ok[seed] = True
        new = _bfs(nb, ok, seed)
        if len(new) < 10:
            return None
        surf = _Surface(p[new], quadric=rnd >= 1)
        if np.array_equal(new, sel):
            break
        sel = new
    grown = sel
    n_grown = len(grown)
    # mode layer along local normals, re-fit, repeat
    # (always over the whole grown patch, so the layer cannot ratchet thinner from one round to the next)
    drift_slab = False
    for _ in range(3):
        r = surf.residual(p[grown])
        keep, spread = _mode_layer(r, s, P['layer_k'])
        # sticky: a re-fit to the slab can make the next round's half-maximum look narrow, which must not undo the slab
        if drift_slab or (spread > P['max_layer_sd'] * s and _is_drift_plateau(p[grown], r, surf, P['origin'], P)):
            keep = _densest_slab(r, P['layer_k'] * s / 2)
            spread = max(float(np.std(r[keep])), s / 2)
            drift_slab = True
        if keep.sum() < 10:
            return None
        idx = grown[keep]
        surf = _Surface(p[idx], quadric=True)
    # the layer step can leave detached bits: keep the part connected to the densest point
    inset = np.zeros(len(p), bool)
    inset[idx] = True
    seed2 = idx[np.argmax(dens[idx])]
    idx = _bfs(nb, inset, seed2)
    n_copies = n_grown - len(idx)
    # statistical outlier removal
    q = p[idx]
    if len(q) > P['sor_k'] + 1:
        dk = cKDTree(q).query(q, P['sor_k'] + 1)[0][:, 1:].mean(1)
        keep = dk <= dk.mean() + P['sor_sd'] * dk.std()
        idx, q = idx[keep], q[keep]
    if len(idx) < 10:
        return None
    surf = _Surface(q, quadric=True)
    plane = _Surface(q, quadric=False)
    r = surf.residual(q)
    rms = float(np.sqrt(np.mean(r ** 2)))
    rms_plane = float(np.sqrt(np.mean(plane.residual(q) ** 2)))
    sag = surf.sag(q)
    layer_sd = spread
    # curved: the bend is at least one sample spacing deep AND the quadric removes >= 40 % of the plane's residual
    # variance (rms_plane >= 1.3 x rms); otherwise the curvature is not observable at this resolution
    shape = 'curved' if (sag >= s and rms_plane >= 1.3 * rms) else 'flat'
    c, ax, sd = _pca(q)
    uv = (q - c) @ ax[:2].T
    fill = _fill(uv, 1.5 * s)
    nz = abs(float(surf.normals(q).mean(0) @ UP))
    span = float(np.ptp(uv[:, 0])) / scene_ext
    med_d = float(np.median(dens[grown])) / d_top   # the drift copies are dwell evidence too: measure the grown patch
    layer_frac = len(idx) / n_grown
    rod = P['origin'] is not None and abs(float(ax[0] @ UP)) >= P['rod_cos']
    tests = {
        'size': (len(idx), len(idx) >= P['min_object_pts']),
        'dense': (med_d, med_d >= P['dense_frac']),
        'smooth': (layer_sd / s, layer_sd / s <= P['max_layer_sd']),
        'not_line': (sd[1] / sd[0], sd[1] / sd[0] >= (P['min_aspect_rod'] if rod else P['min_aspect'])),
        'filled': (fill, fill >= P['min_fill']),
        # the device is aimed at an object standing on something: what it dwells on is the object's side, never a
        # horizontal support (table, floor, shelf) or the ceiling
        'upright': (nz, nz < P['horiz_cos']),
        'not_scene_plane': (span if nz >= P['horiz_cos'] else 0.0,
                            not (nz >= P['horiz_cos'] and span >= P['max_span'])),
    }
    tests = {k: {'value': round(float(v), 4), 'pass': bool(ok)} for k, (v, ok) in tests.items()}
    failed = [k for k, t in tests.items() if not t['pass']]
    return dict(idx=idx, q=q, grown=grown, surf=surf, n=int(len(idx)), score=round(float(n_grown * med_d * d_top), 1),
                passed=not failed, failed=failed, tests=tests, rms=rms, sag=sag, shape=shape, fill=fill,
                layer_frac=layer_frac, n_copies=n_copies, drift_slab=drift_slab, centre=[round(float(v), 3) for v in c],
                normal_z=round(nz, 3), rms_over_s=round(rms / s, 3), layer_sd_over_s=round(layer_sd / s, 3), layer_sd=layer_sd, median_density_frac=round(med_d, 3), med_dens=float(np.median(dens[grown])), plane_over_quadric_rms=round(rms_plane / rms, 3))
