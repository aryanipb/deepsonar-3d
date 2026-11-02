"""Full targets for real scans of a KNOWN object, built by registering the object's parametric surface to the scan's front.

The DeepSonar sensor only ever sees one side, so a real scan has no measured back. For an object whose shape is known (the
red flip-cap bottle is a solid of revolution; a box is a cuboid once its three sides are measured) the complete surface
can be constructed and REGISTERED to the cleaned front face of each scan: scale is fixed from the object's height (the
measured height when given, else the face's seen height), and only translation and yaw are fitted. The registered surface
is the scan's target. It is disclosed as "parametric object registered to the scan's front face": the shape is the real
object's, the placement is fitted to the real measurement, and nothing in it is a measurement of the back. The
registration residual (median distance from the face to the target) is the floor on how well any output can agree with it.

Parts files (configs/real_*_parts.json) hold a list of parts in the 3DView format: kind revolve/box/ellipsoid, centre, size,
rotation and, for a revolve, the outline as (height fraction, half-width fraction) pairs. Units are arbitrary (photo pixels):
only proportions are used. A revolve's depth is forced equal to its width (round objects).
"""
import json
import os

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .canon import _rotation, apply_canonical, from_canonical

SEGMENTS = 96
EPS = 1e-3
TARGET_SOURCE = 'parametric object registered to the scan\'s front face'


def rot_deg(deg):
    ax, ay, az = np.deg2rad(deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


class Part:
    def __init__(self, o):
        self.kind = o['kind']
        self.center = np.asarray(o['center'], np.float64)
        self.size = np.asarray(o['size'], np.float64)
        if self.kind == 'revolve':
            self.size[2] = self.size[0]                           # round: depth = width
        self.r = rot_deg(np.asarray(o.get('rotation_deg', [0, 0, 0]), np.float64))
        prof = sorted([tuple(map(float, p)) for p in o.get('profile', [])])
        self.profile = np.asarray(prof, np.float64) if prof else np.zeros((0, 2))
        if self.kind not in ('revolve', 'box', 'ellipsoid'):
            raise ValueError(f'unknown part kind {self.kind}')
        if not (np.all(self.size > 0) and np.isfinite(self.center).all()):
            raise ValueError('part sizes must be positive and finite')
        if self.kind == 'revolve' and len(self.profile) < 2:
            raise ValueError('a revolve part needs at least 2 profile pairs')

    def width_at(self, t):
        return np.interp(t, self.profile[:, 0], self.profile[:, 1])

    def triangles(self):
        sx, sy, sz = self.size
        if self.kind == 'box':
            v = np.array([[(sx / 2 if i & 4 else -sx / 2), (sy / 2 if i & 2 else -sy / 2), (sz / 2 if i & 1 else -sz / 2)] for i in range(8)])
            faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
            tris = np.asarray([v[list(f)] for f in faces])
        else:
            if self.kind == 'ellipsoid':
                rings = [(i / 32, np.sin(np.pi * i / 32)) for i in range(33)]
            else:
                rings = [(0.0, 0.0)] + [(min(max(t, 0), 1), max(s, 0)) for t, s in self.profile] + [(1.0, 0.0)]
            th = 2 * np.pi * np.arange(SEGMENTS) / SEGMENTS
            ring = [np.stack([s * sx / 2 * np.cos(th), np.full(SEGMENTS, (t - 0.5) * sy), s * sz / 2 * np.sin(th)], 1) for t, s in rings]
            blocks = []
            for a, b in zip(ring[:-1], ring[1:]):
                a2, b2 = np.roll(a, -1, 0), np.roll(b, -1, 0)
                blocks += [np.stack([a, a2, b], 1), np.stack([a2, b2, b], 1)]
            tris = np.concatenate(blocks)
        return tris @ self.r.T + self.center

    def contains(self, p):
        q = (p - self.center) @ self.r
        sx, sy, sz = self.size
        if self.kind == 'box':
            return (np.abs(q[:, 0]) < sx / 2 - EPS) & (np.abs(q[:, 1]) < sy / 2 - EPS) & (np.abs(q[:, 2]) < sz / 2 - EPS)
        if self.kind == 'ellipsoid':
            return (q[:, 0] / (sx / 2)) ** 2 + (q[:, 1] / (sy / 2)) ** 2 + (q[:, 2] / (sz / 2)) ** 2 < 1 - 1e-2
        t = q[:, 1] / sy + 0.5
        rad = np.sqrt((q[:, 0] / (sx / 2)) ** 2 + (q[:, 2] / (sz / 2)) ** 2)
        return (t > 1e-3) & (t < 1 - 1e-3) & (rad < self.width_at(np.clip(t, 0, 1)) - 1e-2)


def sample_surface(parts, n, rng):
    """n area-weighted surface points of the united parts (points hidden inside another part are dropped)."""
    tris, owner = [], []
    for i, p in enumerate(parts):
        t = p.triangles()
        tris.append(t)
        owner.append(np.full(len(t), i))
    tris, owner = np.concatenate(tris), np.concatenate(owner)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    out = []
    while sum(len(o) for o in out) < n:
        k = rng.choice(len(tris), n * 2, p=area / area.sum())
        u, v = rng.random(n * 2), rng.random(n * 2)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        pts = tris[k, 0] + u[:, None] * (tris[k, 1] - tris[k, 0]) + v[:, None] * (tris[k, 2] - tris[k, 0])
        hidden = np.zeros(len(pts), bool)
        for i, p in enumerate(parts):
            m = owner[k] != i
            hidden[m] |= p.contains(pts[m])
        out.append(pts[~hidden])
    return np.concatenate(out)[:n]


def load_parts(path):
    if not os.path.isabs(path):                                   # relative to the repository / container root
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    d = json.load(open(path))
    return [Part(o) for o in d['parts']], d


def _ry(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def register(shape_pts, face_m, yaw_starts, height_m=None):
    """Place the parametric surface (photo frame: x right, y up, z toward the camera, any units) on the cleaned face
    (view frame metres: +z sensor -> object, +y up). Scale = height_m / shape height (or the face's 1-99 % height when
    height_m is None); translation and yaw are fitted by a clipped one-sided face -> surface distance.
    Returns (placed points in the view frame, info)."""
    sh = shape_pts * np.array([-1.0, 1.0, -1.0])              # 180 deg about y: photo frame -> view frame
    face_h = np.percentile(face_m[:, 1], 99) - np.percentile(face_m[:, 1], 1)
    scale = (height_m if height_m else face_h) / (sh[:, 1].max() - sh[:, 1].min())
    sh = sh * scale
    tree = cKDTree(sh)

    def loss(x):
        q = (face_m - x[:3]) @ _ry(x[3])
        d = tree.query(q, distance_upper_bound=0.05)[0]
        return float(np.minimum(np.where(np.isfinite(d), d, 0.05), 0.02).mean())

    best = None
    for psi0 in yaw_starts:
        sh_r = sh @ _ry(psi0).T
        x0 = [face_m[:, 0].mean() - sh_r[:, 0].mean(), face_m[:, 1].mean() - sh_r[:, 1].mean(),
              np.percentile(face_m[:, 2], 5) - sh_r[:, 2].min(), psi0]
        res = minimize(loss, x0, method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    placed = sh @ _ry(best.x[3]).T + best.x[:3]
    d = cKDTree(placed).query(face_m)[0]
    front_ok = bool(np.percentile(face_m[:, 2], 1) - 0.01 <= placed[:, 2].min() <= np.percentile(face_m[:, 2], 99) + 0.01)
    info = dict(resid_median_mm=round(1000 * float(np.median(d)), 2), resid_p90_mm=round(1000 * float(np.percentile(d, 90)), 2),
                front_within_face_depth=front_ok, scale_source='measured height' if height_m else 'seen face height',
                face_height_m=round(float(face_h), 4), yaw_deg=round(float(np.rad2deg(best.x[3])) % 360, 1))
    return placed, info


def build_target(parts, partial_canon, T, n_target, height_m=None, max_median_mm=12.0, seed=0):
    """One real scan -> (gt in canonical units, n_target x 3, or None when rejected; info). partial_canon is the model
    input (canonical 2048 points), T its canonical transform (from to_canonical)."""
    rng = np.random.default_rng(seed)
    face_m = from_canonical(partial_canon, T)                     # the seen face in viewer metres
    origin = np.asarray(T['origin'], np.float64)
    R = _rotation(face_m, origin)                                 # view frame of this scan
    face_v = (face_m - origin) @ R.T
    shape = sample_surface(parts, 60000, rng)
    kinds = {p.kind for p in parts}
    yaw_starts = [0.0] if kinds == {'revolve'} else [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    placed_v, info = register(shape, face_v, yaw_starts, height_m)
    accepted = info['resid_median_mm'] <= max_median_mm and info['front_within_face_depth']
    info.update(accepted=bool(accepted), target_source=TARGET_SOURCE, n_face=len(face_m))
    if not accepted:
        return None, info
    placed_m = placed_v @ R + origin
    idx = rng.choice(len(placed_m), n_target, replace=len(placed_m) < n_target)
    gt = apply_canonical(placed_m[idx], T)
    e = np.percentile(placed_m, 99, 0) - np.percentile(placed_m, 1, 0)
    info.update(target_size_cm=[round(float(v) * 100, 1) for v in e])
    return gt, info


def targets_for_pool(pool, targets_cfg, n_target, heights=None):
    """pool: real_pool() sessions. targets_cfg: {object name: parts json path}. heights: {object name: metres} measured.
    Returns ({session: gt canonical float64 or None}, {session: info})."""
    parts = {obj: load_parts(path)[0] for obj, path in (targets_cfg or {}).items()}
    gts, infos = {}, {}
    for s in pool:
        if s['object'] not in parts:
            continue
        h = (heights or {}).get(s['object'])
        gts[s['session']], infos[s['session']] = build_target(parts[s['object']], s['partial'].astype(np.float64), s['T'], n_target, h)
    return gts, infos
