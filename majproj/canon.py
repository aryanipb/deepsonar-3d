"""Canonical object frame shared by data generation, training and inference.

Input points are metres in the viewer frame (z = up, sensor at `origin`). The canonical frame is centred on the object
and aligned with the way the sensor saw it:

    +z  horizontal direction from the sensor to the object centroid (away from the sensor)
    +y  up (the viewer's +z)
    +x  y x z, so the frame is right-handed

The rotation is about the vertical axis only. The centre is the bounding-box centre of the seen face measured in the
rotated frame, and the scale is the longest side of the seen face (the x and y extents; z is depth) divided by 0.9, so
the seen face spans 0.9 units on its longest side.

`to_canonical` derives T from the seen points. Anything else in the same scene (for example the complete target surface)
must go through `apply_canonical` with that same T, not through a fresh `to_canonical`.
"""
import numpy as np

SEEN_FACE_SPAN = 0.9
N_INPUT = 2048


def _rotation(obj, origin):
    """3x3 matrix whose rows are the canonical axes in viewer coordinates."""
    h = (obj.mean(0) - origin)[:2]
    n = float(np.hypot(*h))
    # object exactly above or below the sensor: no horizontal view direction exists, so use viewer +y
    view = np.array([h[0] / n, h[1] / n, 0.0]) if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(up, view), up, view])


def to_canonical(obj_pts_m, origin=(0.0, 0.0, 0.0)):
    """Seen object points (metres, z up) -> (canonical points float64, JSON-safe T)."""
    p = np.asarray(obj_pts_m, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError(f'expected a non-empty N x 3 array, got shape {p.shape}')
    if not np.isfinite(p).all():
        raise ValueError('non-finite coordinates')
    o = np.asarray(origin, np.float64)
    R = _rotation(p, o)
    q = (p - o) @ R.T
    lo, hi = q.min(0), q.max(0)
    side = float((hi - lo)[:2].max())
    T = dict(R=R.tolist(), centre=((lo + hi) / 2).tolist(), scale=side / SEEN_FACE_SPAN if side > 0 else 1.0,
             origin=o.tolist())
    return apply_canonical(p, T), T


def apply_canonical(pts_m, T):
    """Metres -> canonical with an existing T."""
    p = np.asarray(pts_m, np.float64)
    return ((p - np.asarray(T['origin'])) @ np.asarray(T['R']).T - np.asarray(T['centre'])) / T['scale']


def from_canonical(canon_pts, T):
    """Exact inverse of `apply_canonical`: canonical -> metres, float64."""
    q = np.asarray(canon_pts, np.float64)
    return (q * T['scale'] + np.asarray(T['centre'])) @ np.asarray(T['R']) + np.asarray(T['origin'])


def sample_2048(canon_pts):
    """Exactly 2048 points, float32. Deterministic farthest point sampling from index 0 (the convention of
    majproj.model.shims.furthest_point_sample); clouds with fewer points are tiled."""
    p = np.asarray(canon_pts, np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) == 0:
        raise ValueError(f'expected a non-empty N x 3 array, got shape {p.shape}')
    if len(p) <= N_INPUT:
        return np.resize(p, (N_INPUT, 3)).astype(np.float32)
    idx = np.empty(N_INPUT, np.int64)
    dist = np.full(len(p), np.inf)
    far = 0
    for i in range(N_INPUT):
        idx[i] = far
        dist = np.minimum(dist, ((p - p[far]) ** 2).sum(1))
        far = int(dist.argmax())
    return p[idx].astype(np.float32)
