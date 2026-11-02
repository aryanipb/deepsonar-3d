"""Preprocessing v2 for ONE exported scan file (4096 x 3, bbox-centred, divided by max-abs).

Pipeline
  1. sanitize: drop non-finite rows, remove exact duplicate rows (they come from the viewer's un-thinned
     pending buffer, not from the object).
  2. locate the sensor inside the normalized cloud (mode 'stripes' > 'whole_cloud' > 'scale_free'):
       stripes     analysis.forensics.estimate_from_stripes: the background wall shows the 8 zone columns as
                   stripes at x_k = o_x + D*tan_k, giving sensor position and wall distance D.
       whole_cloud analysis.forensics.estimate_sensor (only if its score/margin are good).
       scale_free  no sensor available: nearest dense cluster, scale = its own half-extent.
  3. segment the target object = the dense cluster nearest the sensor (DBSCAN on the foreground = everything
     clearly in front of the wall), then statistical outlier removal.
  4. canonical frame (DISTANCE-CONSISTENT, not metric):
         center = centroid of the object's visible (front) surface points (one viewpoint: all of them)
         scale  = median RADIAL distance |p - o'| from the sensor to the object points, o' = (o_x, c_y, o_z),
                  i.e. the sensor height o_y is NOT observable and is ASSUMED equal to the object-centroid
                  height c_y.
         canonical = (p - center) / scale
     Deviation from the plain "z-depth of the centroid" (SCALE_MODE = 'z', also recorded as meta['depth_z']):
     hand pitch/yaw sweeps move object points along a shell around the sensor, so z-depth of a very close
     object is polluted by the sweep and can be ~0 or negative; the radial distance is rotation invariant.
     Both depend on the stripe-derived o_z, whose error (~1 % of the wall distance) is comparable to the
     object depth when the object is held within a few cm, so scale is only good to a factor ~1.3 (measured
     on simulated exports, see tests) and far worse below ~10 cm; scale is clamped at MIN_SCALE.
     If the scale were right, the sensor sits ~1 canonical unit from the object and a patch subtending an angle
     theta has size ~ tan(theta) wherever it was held (one VL53L5CX zone, ~5.9 deg, ~0.10 canonical units).
     The export destroyed metric scale: canonical units are "object-distance units", never centimetres.
     In scale_free mode scale = max-abs half-extent of the object cluster (object-size units) instead.
  5. deterministic farthest point sampling to CANON_POINTS (start = point nearest the centroid); if fewer
     points exist, deterministic cyclic tiling of the FPS order (no random duplication).
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

CANON_POINTS = 2048
MIN_SCALE = 0.03            # normalized export units; below this the depth estimate is noise
WALL_FRACTION = 0.80        # points deeper than this fraction of the wall distance are wall / behind-wall
STRIPE_REL_RMS_MAX = 0.02
STRIPE_MIN_COUNT = 4
MIN_OBJECT_POINTS = 40
RECEDING_FRACTION = 0.35     # z-range of a cluster (fraction of wall distance) above which it is floor/table-like
SCALE_MODE = 'radial'       # 'radial' | 'z'


# ----------------------------------------------------------------------------- canonical frame
def to_canonical(points, transform):
    """canonical = (p - center) / scale"""
    c = np.asarray(transform['center'], np.float64)
    return ((np.asarray(points, np.float64) - c) / float(transform['scale'])).astype(np.float32)


def from_canonical(points, transform):
    """Exact inverse of to_canonical: p = canonical * scale + center (float64 internally)."""
    c = np.asarray(transform['center'], np.float64)
    return (np.asarray(points, np.float64) * float(transform['scale']) + c).astype(np.float32)


# ----------------------------------------------------------------------------- helpers
def sanitize(points):
    p = np.asarray(points, np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    return np.unique(p, axis=0)                      # exact duplicates removed, deterministic (sorted) order


def farthest_point_sample(p, n, start=None):
    """Deterministic FPS. Returns n indices (cyclic tiling of the FPS order if len(p) < n)."""
    m = len(p)
    if m == 0:
        raise ValueError('no points to sample')
    k = min(n, m)
    if start is None:
        start = int(np.argmin(np.linalg.norm(p - p.mean(0), axis=1)))
    idx = np.empty(k, np.int64)
    idx[0] = start
    d = np.linalg.norm(p - p[start], axis=1)
    for i in range(1, k):
        j = int(np.argmax(d))                        # argmax ties -> lowest index: deterministic
        idx[i] = j
        d = np.minimum(d, np.linalg.norm(p - p[j], axis=1))
    return idx[np.arange(n) % k]


def _knn_eps(p, k=5, mult=3.0, lo=0.012, hi=0.06):
    if len(p) <= k + 1:
        return hi
    dd, _ = cKDTree(p).query(p, k=k + 1)
    return float(np.clip(mult * np.median(dd[:, -1]), lo, hi))


def _remove_outliers(p, k=8, nsigma=2.5):
    if len(p) <= k + 2:
        return np.ones(len(p), bool)
    dd, _ = cKDTree(p).query(p, k=k + 1)
    m = dd[:, 1:].mean(1)
    return m <= m.mean() + nsigma * m.std()


# ----------------------------------------------------------------------------- sensor
def locate_sensor(p):
    """Sensor position in normalized export units. Returns dict(source, oz, ox, oy, wall_depth, wall_z,
    quality, raw) or None. source in {'stripes', 'whole_cloud'}."""
    try:
        from analysis import forensics as F
    except Exception:                                # analysis package unavailable (e.g. browser port)
        return None
    est = None
    if hasattr(F, 'estimate_from_stripes'):
        try:
            est = F.estimate_from_stripes(p)
        except Exception:
            est = None
    if est is not None:
        good = (est.get('n_stripes', 0) >= STRIPE_MIN_COUNT and est.get('rel_rms', 1.0) <= STRIPE_REL_RMS_MAX
                and est['wall_depth'] > 0.2 and est['oz'] < np.percentile(p[:, 2], 5) + 0.05)
        if good:
            return dict(source='stripes', ox=float(est['ox']), oy=float(est.get('oy', 0.0)), oz=float(est['oz']),
                        wall_depth=float(est['wall_depth']), wall_z=float(est['wall_z']),
                        quality=float(1.0 - est['rel_rms'] / STRIPE_REL_RMS_MAX), raw=est)
    try:
        w = F.estimate_sensor(p)
    except Exception:
        return None
    o = np.asarray(w['o'], float)
    if w['score'] >= 0.55 and w['margin'] >= 0.05 and o[2] < np.percentile(p[:, 2], 5):
        wall_z = float(np.median(p[p[:, 2] >= np.percentile(p[:, 2], 70)][:, 2]))
        return dict(source='whole_cloud', ox=float(o[0]), oy=float(o[1]), oz=float(o[2]),
                    wall_depth=wall_z - float(o[2]), wall_z=wall_z, quality=float(w['score']),
                    raw={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in w.items()})
    return None


# ----------------------------------------------------------------------------- object segmentation
def _is_receding_sheet(q, wall_depth):
    """Floor / table / ceiling: a cluster that runs away from the sensor in depth (z-range a sizeable
    fraction of the wall distance) or is a flat horizontal sheet (normal ~ y). Objects held up to the
    sensor are compact in depth; the single-view object faces the sensor."""
    zr = np.percentile(q[:, 2], 98) - np.percentile(q[:, 2], 2)
    if wall_depth is not None and zr > RECEDING_FRACTION * wall_depth:
        return True
    c = q - q.mean(0)
    w, v = np.linalg.eigh(c.T @ c / len(q))                  # ascending eigenvalues
    return bool(w[0] < 0.02 * w[2] and abs(v[1, 0]) > 0.9)


def _pick_cluster(p, labels, depth, wall_depth=None):
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(ids) == 0:
        return None
    keep = counts >= max(MIN_OBJECT_POINTS, 0.2 * counts.max())
    cands = [(float(np.median(depth[labels == k])), k) for k, ok in zip(ids, keep) if ok]
    if not cands:
        return None
    good = [c for c in cands if not _is_receding_sheet(p[labels == c[1]], wall_depth)]
    pool = good if good else cands
    return min(pool)[1]


def segment_object(p, sensor=None):
    """Indices (into p) of the target object = dense cluster nearest the sensor, outliers removed.
    Without a sensor the 'nearest' direction is -z (the viewer frame looks along +z)."""
    if sensor is not None:
        depth = p[:, 2] - sensor['oz']
        fg = np.where(depth < WALL_FRACTION * sensor['wall_depth'])[0]
    else:
        depth = p[:, 2] - p[:, 2].min()
        fg = np.arange(len(p))
    if len(fg) < MIN_OBJECT_POINTS:
        return np.array([], np.int64)
    q = p[fg]
    labels = DBSCAN(eps=_knn_eps(q), min_samples=8).fit_predict(q)
    k = _pick_cluster(q, labels, depth[fg], sensor['wall_depth'] if sensor is not None else None)
    if k is None:
        return np.array([], np.int64)
    idx = fg[labels == k]
    return idx[_remove_outliers(p[idx])]


# ----------------------------------------------------------------------------- main entry
def _angular_size_deg(obj, o):
    v = obj - o
    z = np.maximum(v[:, 2], 1e-6)
    az = np.degrees(np.arctan2(v[:, 0], z))
    el = np.degrees(np.arctan2(v[:, 1], z))
    lo, hi = 1, 99
    return float(max(np.percentile(az, hi) - np.percentile(az, lo), np.percentile(el, hi) - np.percentile(el, lo)))


def preprocess(points, n_out=CANON_POINTS):
    """One exported file -> dict(partial (n_out,3) float32 canonical, transform=dict(center, scale), meta=dict)."""
    p = sanitize(points)
    if len(p) < 3:
        raise ValueError('scan has fewer than 3 finite points')
    sensor = locate_sensor(p)
    idx = segment_object(p, sensor)
    mode = sensor['source'] if sensor is not None else 'scale_free'
    if len(idx) < MIN_OBJECT_POINTS and sensor is not None:      # sensor fit but no clean foreground object
        sensor, mode = None, 'scale_free'
        idx = segment_object(p, None)
    if len(idx) < MIN_OBJECT_POINTS:                             # nothing compact found: whole cloud
        idx = np.arange(len(p))
        mode = 'scale_free'
        sensor = None
    obj = p[idx]
    centre = obj.mean(0)
    clamped = False
    if sensor is not None:
        o = np.array([sensor['ox'], sensor['oy'], sensor['oz']])
        depth_z = float(centre[2] - sensor['oz'])
        # rotation-invariant radial distance from the sensor (hand pitch/yaw sweeps move the object points
        # along a shell around the sensor, so the plain z-depth of the centroid is polluted by the sweep);
        # the sensor height is not observable, so assume it sits at the object-centroid height
        radial = float(np.median(np.linalg.norm(obj - np.array([o[0], centre[1], o[2]]), axis=1)))
        scale = radial if SCALE_MODE == 'radial' else depth_z
        if scale < MIN_SCALE:
            scale, clamped = MIN_SCALE, True
        depth_ratio = float(scale / sensor['wall_depth'])
        ang = _angular_size_deg(obj, o)
        sens_meta = dict(source=sensor['source'], o=o.tolist(), wall_depth=sensor['wall_depth'],
                         wall_z=sensor['wall_z'], quality=sensor['quality'],
                         canonical_o=((o - centre) / scale).tolist())
    else:
        scale = float(max(np.abs(obj - centre).max(), 1e-3))
        depth_ratio, ang, o, sens_meta = float('nan'), float('nan'), None, None
    transform = dict(center=centre.astype(np.float64), scale=float(scale))
    canon = to_canonical(obj, transform)
    sel = farthest_point_sample(canon.astype(np.float64), n_out)
    partial = canon[sel].astype(np.float32)
    meta = dict(sensor_estimate=sens_meta, object_depth_ratio=depth_ratio, angular_size_deg=ang,
                n_object_points=int(len(idx)), n_input_unique=int(len(p)),
                confident=bool(sensor is not None and not clamped and len(idx) >= 100
                               and sensor['quality'] > 0.0 and 0.0 < depth_ratio < WALL_FRACTION),
                mode=mode, scale_clamped=clamped, depth_z=(depth_z if sensor is not None else None),
                depth_radial=(radial if sensor is not None else None),
                debug=dict(object_indices_unique=idx, unique_points=p))
    return dict(partial=partial, transform=transform, meta=meta)
