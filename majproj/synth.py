"""Synthetic training data with REAL ground truth: procedural objects + simulated one-sided ToF scans.

Each sample is (partial, complete) where `partial` is what a single time-of-flight view of the object
would capture (visible surface only, with depth noise and along-ray streaking like the real scans) and
`complete` is the true full surface of the same object. Both are normalized with the transform computed
from the partial alone (geometry.normalize_params), exactly as at inference time.

Frame: +y up, sensor at the origin looking along +z.
"""
import numpy as np

from .geometry import normalize, normalize_params

CLASSES = ['bottle', 'box']


# ---------- complete surfaces (object-centred, unit-ish size) ----------
def _revolution(profile_y, profile_r, n, rng):
    """Uniform samples on a surface of revolution around +y, plus the two end caps."""
    y, r = np.asarray(profile_y, float), np.asarray(profile_r, float)
    seg_len = np.hypot(np.diff(y), np.diff(r))
    seg_area = np.pi * (r[:-1] + r[1:]) * seg_len
    cap_area = np.pi * np.array([r[0] ** 2, r[-1] ** 2])
    areas = np.concatenate([seg_area, cap_area])
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for i, k in enumerate(counts[:-2]):
        if k == 0:
            continue
        # area-weighted position along a frustum segment
        u = rng.uniform(0, 1, k)
        r0, r1 = r[i], r[i + 1]
        t = u if abs(r1 - r0) < 1e-9 else (np.sqrt(r0 ** 2 + u * (r1 ** 2 - r0 ** 2)) - r0) / (r1 - r0)
        rr, yy = r0 + (r1 - r0) * t, y[i] + (y[i + 1] - y[i]) * t
        a = rng.uniform(0, 2 * np.pi, k)
        pts.append(np.stack([rr * np.cos(a), yy, rr * np.sin(a)], 1))
    for (yc, rc), k in zip([(y[0], r[0]), (y[-1], r[-1])], counts[-2:]):
        if k == 0:
            continue
        a, rr = rng.uniform(0, 2 * np.pi, k), rc * np.sqrt(rng.uniform(0, 1, k))
        pts.append(np.stack([rr * np.cos(a), np.full(k, yc), rr * np.sin(a)], 1))
    return np.concatenate(pts)


def bottle_surface(n, rng):
    """Random bottle / jar / can: body, optional shoulder and neck, with varied proportions."""
    R = 0.5
    H = R * 2 * rng.uniform(1.2, 4.0)                  # height / diameter from 1.2 to 4
    kind = rng.choice(['bottle', 'jar', 'can'], p=[0.6, 0.2, 0.2])
    y0 = -H / 2
    if kind == 'can':
        ys, rs = [y0, H / 2], [R, R]
    else:
        body = rng.uniform(0.45, 0.75) if kind == 'bottle' else rng.uniform(0.8, 0.92)
        rn = R * (rng.uniform(0.25, 0.5) if kind == 'bottle' else rng.uniform(0.6, 0.85))
        shoulder = rng.uniform(0.08, 0.2) if kind == 'bottle' else 0.04
        yb = y0 + body * H
        ysh = min(yb + shoulder * H, H / 2 - 0.02 * H)
        ys, rs = [y0, yb, ysh, H / 2], [R, R, rn, rn]
    return _revolution(ys, rs, n, rng)


def box_surface(n, rng):
    """Random box with independent width, height and depth (aspect ratios 1:0.15 to 1:1)."""
    dims = np.array([1.0, rng.uniform(0.15, 1.0), rng.uniform(0.15, 1.0)])
    dims = rng.permutation(dims) * 0.5
    areas = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]]).repeat(2)
    counts = rng.multinomial(n, areas / areas.sum())
    pts = []
    for f, k in enumerate(counts):
        axis, sign = f // 2, 1 if f % 2 else -1
        p = rng.uniform(-1, 1, (k, 3)) * dims
        p[:, axis] = sign * dims[axis]
        pts.append(p)
    return np.concatenate(pts)


def random_rotation(rng):
    """Upright with random yaw, lying on its side, or arbitrarily tilted."""
    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        m = np.eye(3)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        return m
    yaw = rot(1, rng.uniform(0, 2 * np.pi))
    mode = rng.choice(['upright', 'lying', 'tilted'], p=[0.45, 0.35, 0.2])
    if mode == 'upright':
        return yaw
    if mode == 'lying':
        return rot(1, rng.uniform(0, 2 * np.pi)) @ rot(2, np.pi / 2) @ yaw
    return rot(0, rng.uniform(-0.6, 0.6)) @ rot(2, rng.uniform(-0.6, 0.6)) @ yaw


# ---------- simulated single-view ToF scan ----------
def tof_scan(surface, rng, res=None):
    """Keep the points the sensor would see (z-buffer on a pixel grid), then add ToF artefacts."""
    res = res or int(rng.integers(40, 110))            # how many pixels the object covers
    x, y, z = surface.T
    u, v = x / z, y / z                                 # perspective projection
    span = max(np.ptp(u), np.ptp(v)) * 1.05
    iu = ((u - u.min()) / span * res).astype(int)
    iv = ((v - v.min()) / span * res).astype(int)
    pix = iu * (res + 2) + iv
    order = np.lexsort((z, pix))                        # nearest point per pixel
    first = np.ones(len(order), bool)
    first[1:] = pix[order][1:] != pix[order][:-1]
    vis = order[first]
    # a pixel is valid only if its nearest point is on the front surface (reject grazing back points)
    p = surface[vis]
    size = np.ptp(surface, 0).max()
    rays = p / np.linalg.norm(p, axis=1, keepdims=True)
    depth = np.linalg.norm(p, axis=1)
    depth = depth + rng.normal(0, rng.uniform(0.003, 0.02) * size, len(p))        # range noise
    streak = rng.uniform(0, 1, len(p)) < rng.uniform(0.15, 0.6)                    # flying pixels
    depth[streak] += rng.uniform(0, rng.uniform(0.5, 2.5) * size, streak.sum())
    keep = rng.uniform(0, 1, len(p)) > rng.uniform(0.0, 0.3)                        # dropout
    return (rays * depth[:, None])[keep]


def sample(rng, n_partial=2048, n_complete=16384, n_dense=60000):
    """One training pair (partial, complete, class index), normalized from the partial."""
    cls = int(rng.integers(0, len(CLASSES)))
    surf = (bottle_surface if cls == 0 else box_surface)(n_dense, rng)
    R = random_rotation(rng)
    scale = rng.uniform(0.05, 0.4)
    offset = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(1.0, 3.0)])
    world = (surf @ R.T) * scale + offset
    partial = tof_scan(world, rng)
    if len(partial) < 30:
        return sample(rng, n_partial, n_complete, n_dense)
    complete = world[rng.choice(len(world), n_complete, replace=False)]
    center, s = normalize_params(partial)
    p = normalize(partial, center, s)
    p = p[rng.choice(len(p), n_partial, replace=len(p) < n_partial)]
    return p.astype(np.float32), normalize(complete, center, s), cls
