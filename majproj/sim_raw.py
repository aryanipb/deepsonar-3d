"""Raw-file simulator: one complete DeepSonar RAW metric map (FINAL-INPUTS/*/*.npy) with per-row truth.

simulate_raw(rng, kind=None, regime=None, **kw) -> dict(raw, complete, object_mask, truth); see `simulate_raw`.
Most of this module is the 'robust' regime (`simulate_robust`); the primary 'real' regime is majproj/sim_real.py.

What a raw file is (measured, not assumed; see majproj/raw_frames.py and configs/sim_raw_calib.json, 20 real sessions):
a time-ordered list of frames with nothing dropped and no thinning. Each frame is its valid zones in ascending zone
index; the whole block is written 1 / 2 / 3 / 4 times (p = .122 / .831 / .044 / .003), so duplicates are spread evenly over the
file, not only at rest. The device starts lying on the table facing up (ceiling, about 48 of 64 zones, 15-60 frames), is
tilted over in about 25 frames, then SWEEPS slowly: aim elevation 0-45 deg, median 20 deg above the object, 9 deg/s, azimuth
sd 4 deg, for 300-700 frames at 15 Hz, sometimes swinging back up to the ceiling and returning, and ends lying down again
for 30-800 frames (which is what makes the large files large). Hold readings have a near sd of about 2 mm and a far sd of
about 22 mm at 2.8 m (so noise grows with range squared, not linearly). Every zone off the object is usually invalid
because the walls lie past the signal limit, so a hold frame has only 13-30 valid zones. Whether a far return is valid
depends strongly on the zone (table P_FAR, stable across sessions). A fixture hangs about 0.75 m below the ceiling at the
edge of the view at rest (the "mid layer" at 2 m).

What differs from majproj.sim_scan: no thinning and no 4096 export; the timeline, sweep, noise and validity above; a
much wider object family; distance prior 15-120 cm weighted to 20-50 cm; hand junk is rare. Scene and device helpers are reused
from sim_scan, device.py and scene.py.

Limits, stated so nobody over-reads the numbers:
  * `complete` is the true surface shifted by the mean sensor position over the frames that saw the object. With hand
    drift the raw object is a smear around that, so `complete` is a mean-pose target, never a measured surface.
  * The IMU heading and the room are random. Only the real room's statistics are matched, not its layout.
  * Validity (P_FAR) and multiplicity are taken from one device; another unit's zone pattern would differ.
"""
from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from .sim import device
from .sim.acquisition import IMU_TO_TOF_OFFSET, Q14, SENSOR_YAW
from .sim.scene import Plane, Revolution, cast
from .sim_scan import (CEILING, CLUTTER, FLOOR, HAND, MIXED, NOISE, OBJECT, PERSON, SOURCES, TABLE, WALL, W2S, _box_prim, _logu, _ou,
                       _revolution, _to_scene, _u)

KINDS = ('cuboid', 'cube', 'slab', 'tower', 'bottle', 'wavy_bottle', 'can', 'jar', 'rod', 'pipe')
KIND_P = (0.24, 0.08, 0.10, 0.10, 0.20, 0.08, 0.07, 0.05, 0.05, 0.03)
ROUND = ('bottle', 'wavy_bottle', 'can', 'jar', 'rod', 'pipe')
PHASES = ('rest_start', 'pickup', 'hold', 'putdown', 'rest_end')
COMPLETE_POINTS = 16384
FPS = 15.0
COPIES_P = (0.1217, 0.8312, 0.0440, 0.0031)      # block repeats 1 / 2 / 3 / 4 times (configs/sim_raw_calib.json)
# P(valid | ceiling at 2-3.5 m, device at rest), per zone, mean of the 20 real sessions (configs/sim_raw_calib.json)
P_FAR = np.array([
    0.55, 0.88, 0.52, 0.86, 0.55, 0.71, 0.34, 0.22,
    0.89, 0.68, 0.92, 0.83, 0.55, 0.80, 0.14, 0.75,
    0.74, 0.95, 0.97, 0.97, 0.98, 0.94, 0.62, 0.69,
    0.92, 0.67, 0.99, 1.00, 0.99, 0.95, 0.48, 0.59,
    0.72, 0.65, 0.95, 0.93, 0.94, 0.98, 0.83, 0.32,
    0.32, 0.57, 0.96, 0.96, 0.75, 0.97, 0.81, 0.30,
    0.66, 0.50, 0.97, 0.85, 0.91, 0.92, 0.90, 0.54,
    0.20, 0.89, 0.96, 0.91, 0.74, 0.99, 0.78, 0.09])
_SYAW = SENSOR_YAW.as_matrix()


@dataclass
class RawConfig:
    # timeline, in frames at 15 Hz
    rest_start: tuple = (15, 60)
    p_long_start: float = 0.05           # one real session lay 143 frames first
    pickup: tuple = (15, 35)
    hold: tuple = (300, 600)
    p_short_hold: float = 0.15           # (60, 300) frames: a quick look
    p_long_hold: float = 0.08            # (600, 850)
    putdown: tuple = (5, 35)
    rest_end: tuple = (30, 240)
    p_long_end: float = 0.2              # (240, 820)
    p_excursion: float = 0.45            # swings up to the ceiling and back during the hold
    # scene
    distance_m: float = None             # sensor -> nearest object surface; None = prior (20-50 cm mostly)
    size_m: tuple = None                 # force the dimensions: cuboid-like (a, b, h), round (diameter, h)
    ceiling_m: tuple = (2.3, 3.5)        # above the table top, i.e. the ceiling as seen by the device at rest
    table_height_m: tuple = (0.6, 0.95)
    p_fixture: float = 0.7               # the 2 m "mid layer"
    p_back_wall: float = 0.2
    p_background: float = 0.6            # furniture 0.7-1.8 m away, in the field of view during the hold
    p_person: float = 0.5
    n_clutter_max: int = 3
    p_lying: float = 0.15
    p_partial: float = 0.15              # aim off so only part of the object is ever seen
    p_operator: float = 0.25             # a different hand: sweep centred on the object instead of 20 deg above it
    # hand
    drift_cm: tuple = (0.0, 6.0)         # peak translation drift, not registered
    lever_m: tuple = (0.0, 0.12)         # the sensor swings on this lever around the wrist during the sweep
    wobble_deg: float = 0.25
    imu_err_deg: tuple = (0.05, 0.4)
    imu_lag_s: tuple = (0.02, 0.08)
    # junk
    p_hand: float = 0.3
    p_burst: float = 0.25
    noise_near_mm: tuple = (1.2, 2.2)    # sd = near + far_coef * r^2
    noise_far_coef: tuple = (0.0022, 0.0032)  # m per m^2: 22 mm at 2.8 m
    signal_threshold: tuple = (0.03, 0.055)
    dropout: tuple = (0.005, 0.03)
    blur_deg_s: float = 90.0
    n_side_hold: int = 3
    n_side_other: int = 2
    attempts: int = 3
    min_object_rows: int = 200


_NAMES = {f.name for f in fields(RawConfig)}


def make_config(**kw):
    bad = set(kw) - _NAMES
    if bad:
        raise TypeError(f'unknown simulate_raw option(s): {sorted(bad)}; valid: {sorted(_NAMES)}')
    return replace(RawConfig(), **kw)


# ---------------------------------------------------------------------------------------------------------- primitives

class Posed:
    """Any primitive under a rigid motion: world = R local + c. `R` maps local to scene coordinates."""

    def __init__(self, inner, R, c):
        self.inner, self.R, self.c = inner, np.asarray(R, float), np.asarray(c, float)
        self.reflectivity = inner.reflectivity

    def bounds(self):
        lo, hi = self.inner.bounds()
        ext = np.abs(self.R) @ ((np.asarray(hi) - lo) / 2)
        mid = self.R @ ((np.asarray(hi) + lo) / 2) + self.c
        return mid - ext, mid + ext

    def intersect(self, o, d):
        t, n = self.inner.intersect((np.asarray(o, float) - self.c) @ self.R, d @ self.R)
        return t, n @ self.R.T

    def surface(self, n, rng):
        return self.inner.surface(n, rng) @ self.R.T + self.c


class Pipe:
    """Open thin-walled tube along local y, from y = 0 to h: outer wall, inner wall and the two end annuli."""

    def __init__(self, r, r_in, h, reflectivity):
        self.r, self.r_in, self.h, self.reflectivity = float(r), float(r_in), float(h), float(reflectivity)
        self._outer = Revolution(0.0, 0.0, np.array([0.0, h]), np.array([r, r]), reflectivity)

    def bounds(self):
        return self._outer.bounds()

    def intersect(self, o, d):
        o = np.broadcast_to(np.asarray(o, float), d.shape)
        t, n = self._outer.intersect(o, d)
        p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
        hole = np.isfinite(t) & (np.abs(n[:, 1]) > 0.5) & (p[:, 0] ** 2 + p[:, 2] ** 2 < self.r_in ** 2)
        if hole.any():
            pi, di = p[hole], d[hole]
            a = di[:, 0] ** 2 + di[:, 2] ** 2
            b = pi[:, 0] * di[:, 0] + pi[:, 2] * di[:, 2]
            c = pi[:, 0] ** 2 + pi[:, 2] ** 2 - self.r_in ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (-b + np.sqrt(np.maximum(b * b - a * c, 0.0))) / a       # the exit from inside the bore
            y = pi[:, 1] + s * di[:, 1]
            inside = np.isfinite(s) & (y >= 0.0) & (y <= self.h)
            tt = np.where(inside, t[hole] + s, np.inf)                       # straight through otherwise
            q = pi + np.where(inside, s, 0.0)[:, None] * di
            nn = np.stack([-q[:, 0], np.zeros(len(q)), -q[:, 2]], 1) / self.r_in
            t, n = t.copy(), n.copy()
            t[hole], n[hole] = tt, nn
        return t, n

    def surface(self, n, rng):
        r, ri, h = self.r, self.r_in, self.h
        areas = np.array([2 * np.pi * r * h, 2 * np.pi * ri * h, np.pi * (r * r - ri * ri), np.pi * (r * r - ri * ri)])
        c = rng.multinomial(n, areas / areas.sum())
        pts = []
        for rad, k in ((r, c[0]), (ri, c[1])):
            a = rng.uniform(0, 2 * np.pi, k)
            pts.append(np.stack([rad * np.cos(a), rng.uniform(0, h, k), rad * np.sin(a)], 1))
        for y, k in ((0.0, c[2]), (h, c[3])):
            a, rad = rng.uniform(0, 2 * np.pi, k), np.sqrt(rng.uniform(ri * ri, r * r, k))
            pts.append(np.stack([rad * np.cos(a), np.full(k, y), rad * np.sin(a)], 1))
        return np.concatenate(pts)


# ---------------------------------------------------------------------------------------------------------- objects

def _profile(rng, kind):
    """(height, max radius, ys, rs) of a solid of revolution; at most 12 segments, the per-ray cost."""
    if kind == 'can':
        r, h = _logu(rng, 0.025, 0.06), rng.uniform(0.08, 0.22)
        return h, r, [0.0, 0.004, h - 0.004, h], [r * 0.93, r, r, r * 0.93]
    if kind == 'jar':
        r, h = rng.uniform(0.035, 0.08), rng.uniform(0.08, 0.22)
        rn = r * rng.uniform(0.7, 0.92)
        yb = h * rng.uniform(0.72, 0.88)
        return h, r, [0.0, 0.006, yb, yb + 0.4 * (h - yb), h], [r * 0.94, r, r, rn, rn]
    if kind == 'rod':
        r, h = _logu(rng, 0.01, 0.022), rng.uniform(0.15, 0.60)
        return h, r, [0.0, h], [r, r]
    # bottles: base chamfer, body, curved shoulder, neck, lip
    r = float(np.clip(rng.lognormal(np.log(0.038), 0.25), 0.02, 0.08))
    h = float(np.clip(r * rng.uniform(4.0, 9.0), 0.12, 0.60))
    rn = r * rng.uniform(0.28, 0.5)
    yb = h * rng.uniform(0.5, 0.72)
    ys_ = min(yb + h * rng.uniform(0.1, 0.22), 0.93 * h)
    bez = np.linspace(0.0, 1.0, 4)[1:]
    sh_y = yb + (ys_ - yb) * bez
    sh_r = rn + (r - rn) * (1.0 - bez ** 2) ** 2 * 1.0
    if kind == 'bottle':
        ys = [0.0, 0.008, yb, *sh_y, h - 0.012, h - 0.012 + 1e-4, h]
        rs = [r * 0.94, r, r, *sh_r, rn, rn * 1.15, rn * 1.15]
    else:                                                           # wavy grip around the middle of the body
        a, m = rng.uniform(0.05, 0.16), int(rng.integers(1, 3))
        gy = np.linspace(0.008, yb, 6)
        gr = r * (1.0 - a * 0.5 * (1.0 - np.cos(2 * np.pi * m * (gy - gy[0]) / (gy[-1] - gy[0]))))
        ys = [0.0, *gy, *sh_y, h]
        rs = [r * 0.94, *gr, *sh_r, rn]
    ys = np.maximum.accumulate(np.asarray(ys, float) + np.arange(len(ys)) * 1e-6)
    return h, r, ys, np.asarray(rs, float)


def _draw_object(rng, cfg, kind):
    """Dimensions and pose choice of the target (nothing placed yet). Sides span 3-60 cm for every proportion."""
    o = dict(kind=kind)
    if kind not in ROUND:
        if cfg.size_m is not None:
            a, b, h = cfg.size_m
            o.update(a=a, b=b, h=h, yaw=np.deg2rad(rng.normal(0.0, 8.0)), lying=False, size=(a, b, h))
            return o
        if kind == 'cuboid':
            if rng.random() < 0.3:                                  # household cartons, like the real box
                a, b, h = rng.uniform(0.12, 0.28), rng.uniform(0.08, 0.26), rng.uniform(0.15, 0.32)
            else:
                a, b, h = _logu(rng, 0.03, 0.6), _logu(rng, 0.03, 0.6), _logu(rng, 0.03, 0.6)
        elif kind == 'cube':
            a = _logu(rng, 0.03, 0.6)
            a, b, h = a, a, a
        elif kind == 'tower':
            a, b, h = _logu(rng, 0.03, 0.15), _logu(rng, 0.03, 0.15), rng.uniform(0.25, 0.6)
        else:                                                       # slab: a book on edge, a board, or a mat lying flat
            t = _logu(rng, 0.015, 0.05)
            w, l = _logu(rng, 0.10, 0.6), _logu(rng, 0.10, 0.6)
            a, b, h = (w, t, l) if rng.random() < 0.75 else (w, l, t)
        yaw = rng.normal(0.0, 15.0) if rng.random() < 0.5 else rng.uniform(0.0, 90.0)
        o.update(a=a, b=b, h=h, yaw=np.deg2rad(yaw), lying=False, size=(a, b, h))
        return o
    if kind == 'pipe':
        r, h = _logu(rng, 0.02, 0.08), rng.uniform(0.10, 0.6)
        o.update(r=r, r_in=r * rng.uniform(0.75, 0.95), h=h, ys=None, rs=None)
    else:
        h, r, ys, rs = _profile(rng, kind)
        if cfg.size_m is not None:
            ys, rs = np.asarray(ys, float) * cfg.size_m[1] / h, np.asarray(rs, float) * cfg.size_m[0] / (2 * r)
            r, h = cfg.size_m[0] / 2, cfg.size_m[1]
        o.update(r=r, h=h, ys=np.asarray(ys, float), rs=np.asarray(rs, float))
    p_lie = 0.0 if cfg.size_m is not None else cfg.p_lying * (1.7 if kind in ('rod', 'pipe') else 1.0)
    o.update(lying=bool(rng.random() < p_lie), yaw=float(np.deg2rad(rng.uniform(0.0, 90.0))), size=(2 * r, 2 * r, h))
    return o


def _extent(o):
    """(half depth along the view, half width across it, height above the table, max half extent seen from the side)."""
    if o['kind'] not in ROUND:
        c, s = abs(np.cos(o['yaw'])), abs(np.sin(o['yaw']))
        return 0.5 * (o['a'] * c + o['b'] * s), 0.5 * (o['a'] * s + o['b'] * c), o['h'], 0.5 * max(np.hypot(o['a'], o['b']), o['h'])
    if o['lying']:
        c, s = abs(np.cos(o['yaw'])), abs(np.sin(o['yaw']))
        return 0.5 * o['h'] * c + o['r'] * s, 0.5 * o['h'] * s + o['r'] * c, 2 * o['r'], 0.5 * max(o['h'], 2 * o['r'])
    return o['r'], o['r'], o['h'], 0.5 * max(o['h'], 2 * o['r'])


def _target_prim(o, xy, H_t, psi, refl):
    if o['kind'] not in ROUND:
        return _box_prim([xy[0], xy[1], H_t + o['h'] / 2], (o['a'] / 2, o['h'] / 2, o['b'] / 2), psi + o['yaw'], refl)
    r, h = o['r'], o['h']
    pipe = o['kind'] == 'pipe'
    if not o['lying']:
        if pipe:
            s = _to_scene([xy[0], xy[1], H_t])
            return Posed(Pipe(r, o['r_in'], h, refl), np.eye(3), s)
        return _revolution(xy[0], xy[1], H_t, o['ys'], o['rs'], refl)
    ang = psi + o['yaw']
    axis = _to_scene([np.cos(ang), np.sin(ang), 0.0])
    up = np.array([0.0, 1.0, 0.0])
    e3 = np.cross(axis, up)
    R = np.stack([up, axis, e3], 1)                                  # local y -> along the table
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    c = _to_scene([xy[0], xy[1], H_t + r]) - axis * h / 2
    inner = Pipe(r, o['r_in'], h, refl) if pipe else Revolution(0.0, 0.0, o['ys'], o['rs'], refl)
    return Posed(inner, R, c)


# ---------------------------------------------------------------------------------------------------------- scene

@dataclass
class _Scene:
    prims: list
    codes: np.ndarray
    gloss: np.ndarray
    target: object
    meta: dict


def _draw_distance(rng, cfg, half_max):
    if cfg.distance_m is not None:
        return float(cfg.distance_m)
    u = rng.random()
    d = rng.uniform(0.20, 0.50) if u < 0.72 else rng.uniform(0.15, 0.20) if u < 0.85 else rng.uniform(0.50, 1.20)
    if half_max / np.tan(np.deg2rad(22.0)) > d and rng.random() < 0.7:   # most people step back from a large object
        d = min(1.2, half_max / np.tan(np.deg2rad(22.0)) * rng.uniform(0.9, 1.3))
    return float(d)


def _build_scene(rng, cfg, kind):
    psi = rng.uniform(0.0, 2 * np.pi)
    hv, lv = np.array([np.cos(psi), np.sin(psi), 0.0]), np.array([-np.sin(psi), np.cos(psi), 0.0])
    c_h, H_t = _u(rng, cfg.ceiling_m), _u(rng, cfg.table_height_m)
    H_room = H_t + 0.012 + c_h
    obj = _draw_object(rng, cfg, kind)
    hd, hw, hgt, hmax = _extent(obj)
    dist = _draw_distance(rng, cfg, hmax)
    u_c = dist + hd
    v_c = u_c * np.tan(np.deg2rad(rng.normal(0.0, 4.0)))
    xy = hv * u_c + lv * v_c
    refl = _u(rng, (0.15, 0.9))
    target = _target_prim(obj, xy, H_t, psi, refl)
    p_gloss = 0.6 if kind in ROUND else 0.25
    glossy = rng.random() < p_gloss
    obj_gloss = rng.uniform(2.0, 6.0) if glossy else 1.0
    gloss_drop = rng.uniform(0.05, 0.45) if glossy else rng.uniform(0.0, 0.05)
    # hand height above the table: real sessions sit 10-43 cm above it
    hh = float(np.clip(0.5 * hgt + rng.normal(0.0, 0.07), 0.06, 0.6))
    z_hold = H_t + hh
    r_out = max(hd, hw)
    d_c = float(np.hypot(u_c, v_c))

    ru, rv = rng.uniform(-0.5, 0.1), rng.uniform(-0.35, 0.35)        # the device lies here before and after
    rest = hv * ru + lv * rv
    rest2 = rest + rng.normal(0.0, 0.05, 3) * np.array([1.0, 1.0, 0.0])
    p_rest, p_rest2 = np.array([rest[0], rest[1], H_t + 0.012]), np.array([rest2[0], rest2[1], H_t + 0.012])

    prims, codes = [target], [OBJECT]
    uu, vv = [0.0, ru, u_c - r_out, u_c + r_out], [-0.3, 0.3, rv, v_c - r_out, v_c + r_out]
    u0, u1 = min(uu) - _u(rng, (0.15, 0.5)), max(uu) + _u(rng, (0.05, 0.5))
    v0, v1 = min(vv) - _u(rng, (0.1, 0.3)), max(vv) + _u(rng, (0.1, 0.3))
    t_c = hv * (u0 + u1) / 2 + lv * (v0 + v1) / 2
    prims.append(_box_prim([t_c[0], t_c[1], H_t - 0.02], ((u1 - u0) / 2, 0.02, (v1 - v0) / 2), psi, rng.uniform(0.03, 0.2)))
    codes.append(TABLE)
    prims.append(Plane(point=_to_scene([0, 0, H_room]), normal=_to_scene([0, 0, 1]), reflectivity=rng.uniform(0.55, 0.9)))
    codes.append(CEILING)
    prims.append(Plane(point=_to_scene([0, 0, 0]), normal=_to_scene([0, 0, 1]), reflectivity=rng.uniform(0.15, 0.5)))
    codes.append(FLOOR)
    rho = np.deg2rad(rng.uniform(-15.0, 15.0))
    has_back = rng.random() < cfg.p_back_wall
    for k in range(4):                                               # far walls lie past the signal limit in most rooms
        ang = psi + rho + k * np.pi / 2
        n = np.array([np.cos(ang), np.sin(ang), 0.0])
        w = (u1 + _u(rng, (0.05, 1.5))) if (k == 0 and has_back) else _u(rng, (3.3, 7.0))
        prims.append(Plane(point=_to_scene(n * w), normal=_to_scene(n), reflectivity=rng.uniform(0.3, 0.8)))
        codes.append(WALL)
    if rng.random() < cfg.p_fixture:                                 # beam / cupboard / AC high up: the 2 m mid layer
        a = rng.uniform(0, 2 * np.pi)
        dirn = np.array([np.cos(a), np.sin(a), 0.0])
        w_f, d_f, drop = _u(rng, (0.4, 1.4)), _u(rng, (1.0, 2.5)), _u(rng, (0.55, 0.95))
        c = p_rest + dirn * (_u(rng, (0.55, 0.9)) + w_f / 2)
        prims.append(_box_prim([c[0], c[1], H_room - drop / 2], (w_f / 2, drop / 2, d_f / 2), a, rng.uniform(0.4, 0.8)))
        codes.append(CLUTTER)
    for _ in range(int(rng.integers(0, cfg.n_clutter_max + 1))):     # table-top things next to the target
        for _try in range(20):
            cu, cv, cr = u_c + rng.uniform(-0.5, 0.5), v_c + rng.uniform(-0.5, 0.5), rng.uniform(0.02, 0.10)
            in_t = (u0 + cr < cu < u1 - cr) and (v0 + cr < cv < v1 - cr)
            clear = np.hypot(cu - u_c, cv - v_c) > r_out + cr + 0.03
            device_clear = np.hypot(cu - ru, cv - rv) > cr + 0.1 and np.hypot(cu, cv) > cr + 0.1
            on_path = 0.0 < (cu * u_c + cv * v_c) / d_c < d_c + cr and abs(cv * u_c - cu * v_c) / d_c < cr + 0.08
            if in_t and clear and device_clear and not on_path:
                break
        else:
            continue
        ch, cxy, rf = rng.uniform(0.03, 0.25), hv * cu + lv * cv, rng.uniform(0.2, 0.7)
        if rng.random() < 0.5:
            prims.append(_box_prim([cxy[0], cxy[1], H_t + ch / 2], (cr, ch / 2, cr * rng.uniform(0.6, 1.4)), rng.uniform(0, np.pi), rf))
        else:
            prims.append(_revolution(cxy[0], cxy[1], H_t, [0.0, ch], [cr, cr], rf))
        codes.append(CLUTTER)
    if rng.random() < cfg.p_background:                              # furniture behind the target, inside the view
        far = max(u_c + r_out + 0.1, _u(rng, (0.7, 1.8)))
        side = rng.normal(0.0, 0.5)
        w_b, d_b, h_b = _u(rng, (0.1, 0.6)), _u(rng, (0.1, 0.6)), _u(rng, (0.3, 1.2))
        c = hv * far + lv * side
        prims.append(_box_prim([c[0], c[1], H_t + h_b / 2 - 0.2], (w_b / 2, h_b / 2, d_b / 2), rng.uniform(0, np.pi), rng.uniform(0.2, 0.8)))
        codes.append(CLUTTER)
    if rng.random() < cfg.p_person:
        ang, dd, pr = rng.uniform(0, 2 * np.pi), rng.uniform(0.6, 1.4), rng.uniform(0.15, 0.22)
        pxy = np.array([np.cos(ang), np.sin(ang)]) * dd
        if np.hypot(*(pxy - (hv * u_c + lv * v_c)[:2])) > r_out + pr + 0.05:
            prims.append(_revolution(pxy[0], pxy[1], H_t - 0.3, [0.0, rng.uniform(1.0, 1.4)], [pr, pr], rng.uniform(0.2, 0.6)))
            codes.append(PERSON)

    gloss_of = {TABLE: (3.0, 8.0), FLOOR: (2.0, 5.0)}
    gloss = np.array([rng.uniform(*gloss_of[c]) if c in gloss_of else 1.0 for c in codes])
    gloss[0] = obj_gloss
    scene = _Scene(prims, np.array(codes, np.int8), gloss, target,
                   dict(kind=kind, size_m=tuple(float(x) for x in obj['size']), lying=bool(obj['lying']), yaw_deg=float(np.rad2deg(obj['yaw'])),
                        reflectivity=float(refl), glossy=bool(glossy), gloss_drop=float(gloss_drop), ceiling_m=float(c_h), table_height_m=float(H_t),
                        hold_height_m=float(hh), back_wall=bool(has_back), n_prims=len(prims)))
    el_obj = np.arctan2(H_t + hgt / 2 - z_hold, d_c)
    half_ang = np.arctan2(0.5 * max(hgt, 2 * hw), d_c)
    key = dict(psi=psi, p_rest=p_rest, p_rest2=p_rest2, z_hold=z_hold, heading=psi + np.arctan2(v_c, u_c), el_obj=el_obj,
               half_ang=half_ang, hv=hv, lv=lv)
    return scene, key


# ---------------------------------------------------------------------------------------------------------- motion

def _frames(rng, lo_hi):
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _aim_rot(heading, elev, roll):
    return Rotation.from_euler('ZYZ', np.stack([heading, np.pi / 2 - elev, roll], -1))


def _sweep(rng, cfg, key, n):
    """Hold-phase aim: slow pchip paths through random knots. Numbers from the real sessions: elevation 0-45 deg, median 20
    deg above the object, knots about 2.5 s apart give the measured 9 deg/s; azimuth sd 4 deg, roll sd 2 deg."""
    g = float(np.clip((np.rad2deg(key['half_ang']) + 12.0) / 32.0, 0.35, 1.0))     # small or far objects need a tighter sweep
    if rng.random() < cfg.p_operator:
        off = rng.uniform(-5.0, 10.0)
    else:
        off = rng.normal(21.0, 3.5)
    half = rng.uniform(14.0, 24.0)
    partial = rng.random() < cfg.p_partial
    az0 = 0.0
    if partial:                                                      # only part of the object is ever in the view
        az0 = float(rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 24.0))
        half, off = rng.uniform(3.0, 9.0), off + rng.choice([-1.0, 1.0]) * rng.uniform(0.0, 12.0)
    c, h = np.rad2deg(key['el_obj']) + off * g, half * g
    t_end = n / FPS
    ts = [0.0]
    while ts[-1] < t_end:
        ts.append(ts[-1] + rng.uniform(1.0, 4.0))
    ts = np.array(ts)
    el = rng.uniform(c - h, c + h, len(ts))
    az = rng.normal(az0, rng.uniform(2.5, 6.0), len(ts))
    rl = rng.normal(0.0, 2.0, len(ts))
    n_exc = 0
    if rng.random() < cfg.p_excursion and len(ts) > 4:
        n_exc = int(rng.integers(1, 4))
        el[rng.integers(1, len(ts) - 1, n_exc)] = rng.uniform(50.0, 85.0, n_exc)
    tt = (np.arange(n) + 0.5) / FPS
    wob = _ou(rng, n, 1.0 / FPS, 0.3, np.deg2rad(cfg.wobble_deg), 3)
    elev = np.deg2rad(PchipInterpolator(ts, el)(tt)) + wob[:, 0]
    azim = key['heading'] + np.deg2rad(PchipInterpolator(ts, az)(tt)) + wob[:, 1]
    r_base = np.pi if rng.random() > 0.12 else float(rng.choice([0.0, np.pi / 2, -np.pi / 2]))
    roll = r_base + np.deg2rad(rng.normal(0.0, 4.0)) + np.deg2rad(PchipInterpolator(ts, rl)(tt)) + wob[:, 2]
    return _aim_rot(azim, elev, roll), dict(sweep_centre_deg=float(c), sweep_half_deg=float(h), partial=bool(partial), excursions=n_exc)


def _rest_rot(rng):
    ax = rng.normal(0.0, 1.0, 2)
    ax /= np.linalg.norm(ax)
    return Rotation.from_rotvec(np.array([ax[0], ax[1], 0.0]) * np.deg2rad(rng.uniform(0.0, 5.0))) * Rotation.from_euler('Z', rng.uniform(-np.pi, np.pi))


def _timeline(rng, cfg, key):
    n0 = _frames(rng, (100, 180) if rng.random() < cfg.p_long_start else cfg.rest_start)
    n1 = _frames(rng, cfg.pickup)
    u = rng.random()
    nh = _frames(rng, (60, 300) if u < cfg.p_short_hold else (600, 850) if u < cfg.p_short_hold + cfg.p_long_hold else cfg.hold)
    n2 = _frames(rng, cfg.putdown)
    n3 = _frames(rng, (240, 820) if rng.random() < cfg.p_long_end else cfg.rest_end)
    N = n0 + n1 + nh + n2 + n3
    phase = np.repeat(np.arange(5, dtype=np.int8), [n0, n1, nh, n2, n3])
    R_h, info = _sweep(rng, cfg, key, nh)
    R_a, R_b = _rest_rot(rng), _rest_rot(rng)

    # translation: slow drift plus the lever of the wrist; both are invisible to the viewer
    p_hold = np.array([0.0, 0.0, key['z_hold']])
    peak = _u(rng, cfg.drift_cm) / 100.0
    per, phi = rng.uniform(4.0, 16.0, 3), rng.uniform(0, 2 * np.pi, 3)
    dirs = rng.normal(0.0, 1.0, 3) * np.array([1.0, 0.6, 0.6])
    dirs = dirs / max(np.abs(dirs).max(), 1e-6)
    tt = ((np.arange(nh) + 0.5) / FPS)[:, None]
    ramp = np.minimum(1.0, np.minimum(tt, tt[::-1]) / 0.6)
    d = (dirs * np.sin(2 * np.pi * tt / per + phi) * peak + _ou(rng, nh, 1.0 / FPS, 2.0, 0.002, 3)) * ramp
    d = d @ np.stack([key['hv'], key['lv'], [0.0, 0.0, 1.0]])
    axis = (R_h.as_matrix() @ _SYAW)[:, :, 2]
    lever = _u(rng, cfg.lever_m)
    pos_h = p_hold + d + lever * (axis - axis.mean(0))

    def move(p0, p1, r0, r1, n):
        u_ = ((np.arange(n) + 0.5) / n)
        s = 0.5 * (1 - np.cos(np.pi * np.clip(u_ / 0.7, 0, 1)))
        pos = p0[None] + (p1 - p0)[None] * s[:, None]
        rot = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))(0.5 * (1 - np.cos(np.pi * u_)))
        return pos, rot

    p1, r1 = move(key['p_rest'], pos_h[0], R_a, R_h[0], n1)
    p2, r2 = move(pos_h[-1], key['p_rest2'], R_h[-1], R_b, n2)
    p_true = np.vstack([np.tile(key['p_rest'], (n0, 1)), p1, pos_h, p2, np.tile(key['p_rest2'], (n3, 1))])
    rot_true = Rotation.concatenate([Rotation.from_quat(np.tile(R_a.as_quat(), (n0, 1))), r1, R_h, r2, Rotation.from_quat(np.tile(R_b.as_quat(), (n3, 1)))])
    t = (np.arange(N) + 0.5) / FPS
    lag = _u(rng, cfg.imu_lag_s)
    rot_lag = Slerp(t, rot_true)(np.clip(t - lag, t[0], t[-1]))
    sig = np.deg2rad(_u(rng, cfg.imu_err_deg))
    n_hold = int(N / FPS * 2) + 3
    err = _ou(rng, n_hold, 0.5, 3.0, sig, 3)[np.minimum((t * 2).astype(int), n_hold - 1)]
    rot_imu = Rotation.from_rotvec(err) * rot_lag
    rot_imu = Rotation.from_quat(np.round(rot_imu.as_quat() * Q14) / Q14)

    # hand / finger junk and noise bursts
    junk = np.zeros((N, 5))                                          # active, ux, uy, c, depth
    if rng.random() < cfg.p_hand:
        for _ in range(int(rng.integers(1, 3))):
            s, ln = int(rng.integers(0, N)), int(rng.integers(3, 21))
            a = rng.uniform(0, 2 * np.pi)
            d_h = rng.uniform(0.03, 0.15) if rng.random() < 0.7 else rng.uniform(0.15, 0.45)
            junk[s:s + ln] = [1.0, np.cos(a), np.sin(a), rng.uniform(-0.1, 0.45), d_h]
    burst = np.zeros(N, bool)
    if rng.random() < cfg.p_burst:
        for _ in range(int(rng.integers(1, 4))):
            s = int(rng.integers(0, N))
            burst[s:s + int(rng.integers(2, 16))] = True
    copies = rng.choice(np.arange(1, 5), N, p=COPIES_P)
    return dict(phase=phase, p_true=p_true, R_true=rot_true.as_matrix(), R_imu=rot_imu.as_matrix(), junk=junk, burst=burst, copies=copies,
                frames=dict(rest_start=n0, pickup=n1, hold=nh, putdown=n2, rest_end=n3), info=info, drift_peak_m=float(peak),
                lever_m=float(lever))


# ---------------------------------------------------------------------------------------------------------- sensor

def _sensor_params(rng, cfg):
    return dict(near=_u(rng, cfg.noise_near_mm) / 1000.0, coef=_u(rng, cfg.noise_far_coef), thr=_u(rng, cfg.signal_threshold),
                drop=_u(rng, cfg.dropout), pfar=1.0 / (1.0 + np.exp(-(np.log(np.clip(P_FAR, 0.03, 0.98) / (1 - np.clip(P_FAR, 0.03, 0.98)))
                                                                       + rng.normal(0.0, 0.35, 64)))))


def _cast(prims, o, d):
    """scene.cast with a bounding-sphere cull for the bounded primitives, about 3x cheaper than its slab test per ray."""
    plane_idx = [i for i, p in enumerate(prims) if isinstance(p, Plane)]
    t, n, refl, w = cast([prims[i] for i in plane_idx], o, d)
    which = np.where(w >= 0, np.asarray(plane_idx)[np.maximum(w, 0)], -1)
    for i, p in enumerate(prims):
        if isinstance(p, Plane):
            continue
        lo, hi = p.bounds()
        c, rad2 = (np.asarray(lo) + hi) / 2, float(((np.asarray(hi) - lo) ** 2).sum() / 4)
        oc = c - o
        b = oc[:, 0] * d[:, 0] + oc[:, 1] * d[:, 1] + oc[:, 2] * d[:, 2]
        idx = np.nonzero(oc[:, 0] ** 2 + oc[:, 1] ** 2 + oc[:, 2] ** 2 - b * b <= rad2)[0]
        if len(idx) == 0:
            continue
        t_s, n_s = p.intersect(o[idx], d[idx])
        upd = t_s < t[idx]
        sel = idx[upd]
        t[sel], n[sel], refl[sel], which[sel] = t_s[upd], n_s[upd], p.reflectivity, i
    return t, n, refl, which


def _frames_chunk(rng, cfg, scene, tl, sp, f0, f1, n, static):
    """Reported zones of frames f0..f1 (all with `n` sub-rays per zone side). `static` frames share one ray cast."""
    sub = device.subray_dirs(n).reshape(-1, 3)
    m, n_sub = len(sub), n * n
    vign = sub[:, 2] ** 4
    F = f1 - f0
    F_cast = 1 if static else F
    S_true = tl['R_true'][f0:f0 + F_cast] @ _SYAW
    origin = tl['p_true'][f0:f0 + F_cast] + tl['R_true'][f0:f0 + F_cast] @ IMU_TO_TOF_OFFSET
    dirs_s = (np.matmul(sub[None], S_true.transpose(0, 2, 1)) @ W2S.T).reshape(-1, 3)
    o_s = np.repeat(origin @ W2S.T, m, axis=0)
    t, normal, refl, which = _cast(scene.prims, o_s, dirs_s)
    cosi = np.abs(normal[:, 0] * dirs_s[:, 0] + normal[:, 1] * dirs_s[:, 1] + normal[:, 2] * dirs_s[:, 2])
    idx = np.maximum(which, 0)
    src = np.where(which >= 0, scene.codes[idx], -1).astype(np.int8).reshape(F_cast, m)
    expo = np.where(which >= 0, scene.gloss[idx], 1.0).reshape(F_cast, m)
    t, refl, cosi = t.reshape(F_cast, m), refl.reshape(F_cast, m), cosi.reshape(F_cast, m)
    if static:
        t, refl, cosi, src, expo = (np.broadcast_to(a, (F, m)) for a in (t, refl, cosi, src, expo))
    else:
        for k in np.nonzero(tl['junk'][f0:f1, 0])[0]:
            _, ux, uy, c, dh = tl['junk'][f0 + k]
            tx, ty = sub[:, 0] / sub[:, 2], sub[:, 1] / sub[:, 2]
            th = dh / sub[:, 2]
            upd = ((tx * ux + ty * uy) > c) & (th < t[k])
            t[k], refl[k], cosi[k] = np.where(upd, th, t[k]), np.where(upd, 0.6, refl[k]), np.where(upd, sub[:, 2], cosi[k])
            expo[k], src[k] = np.where(upd, 1.0, expo[k]), np.where(upd, HAND, src[k])
    fin = np.isfinite(t)
    w = np.where(fin, refl * cosi ** expo * vign[None] / np.maximum(np.where(fin, t, 1.0), 0.05) ** 2, 0.0)
    r = np.where(fin, t, 99.0)
    r3, w3, s3 = r.reshape(F, 64, n_sub), w.reshape(F, 64, n_sub), src.reshape(F, 64, n_sub)
    d_perp, status = device.measure_batch(r3, w3, rng, noise_rel=0.0, noise_abs_m=0.0, signal_threshold=sp['thr'], dropout=sp['drop'])
    r_rep = d_perp / 1000.0 / device.ZONE_PITCH_SIN[None]
    # the sensor's own noise: sd = near + coef r^2 (measured 2 mm at 30 cm, 22 mm at 2.8 m)
    d_perp = np.round(d_perp + 1000.0 * rng.normal(0.0, 1.0, d_perp.shape) * (sp['near'] + sp['coef'] * r_rep ** 2) * device.ZONE_PITCH_SIN[None])
    dif = np.where(w3 > 0, np.abs(r3 - r_rep[..., None]), np.inf)
    zone_src = np.take_along_axis(s3, dif.argmin(2)[..., None], 2)[..., 0].astype(np.int8)
    zone_src = np.where(dif.min(2) > 0.025, MIXED, zone_src)
    valid = (status == 5) & (d_perp >= device.MIN_RANGE_MM)
    # far returns are valid zone by zone (P_FAR); near ones are reliable. Blends between 0.8 and 2.2 m.
    far = np.clip((r_rep - 0.8) / 1.4, 0.0, 1.0)
    valid &= rng.random(valid.shape) < 1.0 - (1.0 - sp['pfar'][None]) * far
    valid &= ~((zone_src == OBJECT) & (rng.random(valid.shape) < scene.meta['gloss_drop']))
    if cfg.blur_deg_s > 0:
        valid &= rng.random(valid.shape) < np.exp(-(tl['omega'][f0:f1] / cfg.blur_deg_s) ** 2)[:, None]
    for k in np.nonzero(tl['burst'][f0:f1])[0]:                      # ghosts: zones that report nonsense for a moment
        hit = rng.random(64) < rng.uniform(0.02, 0.35)
        d_perp[k, hit] = rng.uniform(40.0, 3500.0, int(hit.sum()))
        valid[k, hit] = True
        zone_src[k, hit] = NOISE
    return d_perp, valid, zone_src


def _run(rng, cfg, scene, tl):
    N = len(tl['phase'])
    R_true = Rotation.from_matrix(tl['R_true'])
    omega = np.zeros(N)
    omega[1:] = np.rad2deg((R_true[:-1].inv() * R_true[1:]).magnitude()) * FPS
    omega[0] = omega[1]
    tl['omega'] = omega
    sp = _sensor_params(rng, cfg)
    S_imu = tl['R_imu'] @ _SYAW
    off_map = tl['R_imu'] @ IMU_TO_TOF_OFFSET
    d_all, v_all, s_all = np.zeros((N, 64)), np.zeros((N, 64), bool), np.zeros((N, 64), np.int8)
    fr = tl['frames']
    a, b = fr['rest_start'], N - fr['rest_end']
    d_ = _frames_chunk(rng, cfg, scene, tl, sp, 0, a, cfg.n_side_other, True)
    d_all[:a], v_all[:a], s_all[:a] = d_
    d_ = _frames_chunk(rng, cfg, scene, tl, sp, b, N, cfg.n_side_other, True)
    d_all[b:], v_all[b:], s_all[b:] = d_
    chunk = int(120_000 / (64 * cfg.n_side_hold ** 2))
    for f in range(a, b, chunk):
        g = min(f + chunk, b)
        d_all[f:g], v_all[f:g], s_all[f:g] = _frames_chunk(rng, cfg, scene, tl, sp, f, g, cfg.n_side_hold, False)
    # registration, exactly the viewer's: IMU rotation plus the fixed offset rotated by the IMU; no translation
    local = device.backproject_batch(d_all)
    world = np.matmul(local, S_imu.transpose(0, 2, 1)) + off_map[:, None, :]
    L = v_all.sum(1)
    keep = L > 0
    rows, srcs, phs, frs = [], [], [], []
    for f in np.nonzero(keep)[0]:
        v = v_all[f]
        c = int(tl['copies'][f])
        rows.extend([world[f][v]] * c)
        srcs.extend([s_all[f][v]] * c)
        phs.append(np.full(L[f] * c, tl['phase'][f], np.int8))
        frs.append(np.full(L[f] * c, f, np.int32))
    if not rows:
        return None
    return (np.concatenate(rows).astype(np.float64), np.concatenate(srcs), np.concatenate(phs), np.concatenate(frs),
            (s_all == OBJECT) & v_all, v_all)


# ---------------------------------------------------------------------------------------------------------- public API

def simulate_robust(rng, kind=None, **kw):
    """One 'robust' session (invented room, wider ranges), as the real raw file would hold it.

    Returns
      raw          (N, 3) float64, metres, z up, sensor at the map origin; N about 20k-110k; frames in time order, each
                   block written 1-4 times, exactly like FINAL-INPUTS/*.npy
      complete     (16384, 3) float32, the whole true object surface in the same frame (see the module limits)
      object_mask  (N,) bool, rows whose reported zone came from the object
      truth        dict: kind, size_m (cuboids: a, b, height; round: diameter, diameter, height), distance_m (sensor to the
                   nearest object surface at the mean pose), lying, yaw_deg, source (N int8, see SOURCES), phase_id (N int8, see
                   PHASES), frame (N int32), n_frames, frames (per phase), partial, excursions, drift_peak_m, n_object_rows,
                   object_frames, attempts, and the scene parameters
    `kw` are the fields of RawConfig. `kind` is one of KINDS, or None to draw from KIND_P."""
    cfg = make_config(**kw)
    kind = str(rng.choice(KINDS, p=KIND_P)) if kind is None else str(kind)
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    best = None
    for attempt in range(1, cfg.attempts + 1):
        scene, key = _build_scene(rng, cfg, kind)
        tl = _timeline(rng, cfg, key)
        out = _run(rng, cfg, scene, tl)
        if out is None:
            continue
        n_obj = int((out[1] == OBJECT).sum())
        if best is None or n_obj > best[0]:
            best = (n_obj, scene, tl, out, attempt)
        if n_obj >= cfg.min_object_rows:
            break
    if best is None:
        raise RuntimeError('simulate_robust: no session produced any valid point')
    n_obj, scene, tl, (raw, src, phase, frame, obj_zone, valid), attempt = best
    seen = obj_zone.any(1)
    sel = seen if seen.any() else tl['phase'] == PHASES.index('hold')
    p_nom = tl['p_true'][sel].mean(0)
    comp = scene.target.surface(COMPLETE_POINTS, rng) @ W2S - p_nom
    truth = dict(scene.meta)
    truth.update(tl['info'])
    truth['regime'] = 'robust'
    truth.update(distance_m=float(np.linalg.norm(comp, axis=1).min()), source=src.astype(np.int8), source_names=SOURCES, phase_id=phase,
                 phase_names=PHASES, frame=frame, n_frames=int(len(tl['phase'])), frames=tl['frames'], drift_peak_m=tl['drift_peak_m'],
                 lever_m=tl['lever_m'], n_object_rows=n_obj, object_frames=int(seen.sum()), attempts=attempt,
                 nominal_pose_m=[float(v) for v in p_nom], valid_per_frame_hold=float(np.median(valid[tl['phase'] == 2].sum(1))))
    return dict(raw=raw, complete=comp.astype(np.float32), object_mask=src == OBJECT, truth=truth)


P_ROBUST_MAX = 0.20


def simulate_raw(rng, kind=None, regime=None, p_robust=0.15, **kw):
    """One raw session; `truth['regime']` says which of the two simulators made it.

    'real'    majproj.sim_real: the measured sessions with a new object rendered in. This is the primary distribution.
    'robust'  simulate_robust above: invented rooms and the wider ranges, at most 20 % of the unforced draws.
    regime=None draws with `p_robust`; `kw` go to the chosen simulator. Same returned keys in both regimes."""
    from .sim_real import simulate_real                  # the bank is loaded only when the regime is used
    if not 0.0 <= p_robust <= P_ROBUST_MAX:
        raise ValueError(f'p_robust must be in [0, {P_ROBUST_MAX}]')
    kind = str(rng.choice(KINDS, p=KIND_P)) if kind is None else str(kind)
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    regime = ('robust' if rng.random() < p_robust else 'real') if regime is None else regime
    if regime == 'real':
        return simulate_real(rng, kind, **kw)
    if regime == 'robust':
        return simulate_robust(rng, kind, **kw)
    raise ValueError(f"regime must be 'real' or 'robust', got {regime!r}")


def simulate_batch(n, seed=0, **kw):
    for i in range(n):
        yield simulate_raw(np.random.default_rng([seed, i]), **kw)
