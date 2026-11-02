"""Random scenes and acquisition sessions for the device simulator (v2, matched to the real regime).

A scene is one object (bottle / jar / can / yawed box) in front of a background wall, optionally on a
support surface or above a room floor, optionally next to a side wall, seen from a single hand-held
viewpoint. Forensics of the real exports show the object is held VERY close to the sensor (centimetres)
and fills much of the 45 deg field of view, so 75 % of sessions draw the object distance log-uniform in
3-40 cm and 25 % in 0.4-1.5 m. Every parameter that the exported `_shapenet.npy` hides (object distance,
wall distance, sizes, hand motion) is drawn from wide ranges and returned as truth.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from .acquisition import IMU_TO_TOF_OFFSET, HandMotion, format_for_ai, scan
from .scene import Plane, Revolution, RotatedBox

CLASSES = ['bottle', 'box']
CLOSE_RANGE_M = (0.03, 0.40)
FAR_RANGE_M = (0.40, 1.50)
WALL_RANGE_M = (0.5, 3.0)


@dataclass
class Session:
    export: np.ndarray            # 4096 x 3 float32, exactly the deployment file format
    complete: np.ndarray          # complete object surface (16384 pts), same normalized frame as `export`
    cls: int
    truth: dict = field(default_factory=dict)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_object(rng, distance, support_y, cls=None):
    """Object whose base is at height `support_y` (sensor height 0) and whose nearest point is `distance`
    metres in front of the sensor. Returns (cls, primitive, size (w, h, d))."""
    return _random_object(rng, distance, support_y, cls)[:3]


def _random_object(rng, distance, support_y, cls=None):
    """As random_object, plus an info dict (object kind, box yaw) as 4th return value."""
    cls = int(rng.integers(0, len(CLASSES))) if cls is None else int(cls)
    ang = np.deg2rad(rng.uniform(-10.0, 10.0))                   # lateral offset as an angle off the axis
    info = {}
    if cls == 0:
        r = rng.uniform(0.025, 0.06)                             # bottles/jars/cans 5-12 cm diameter
        kind = str(rng.choice(['bottle', 'jar', 'can'], p=[0.6, 0.2, 0.2]))
        h = rng.uniform(0.15, 0.35) if kind == 'bottle' else rng.uniform(0.08, 0.25)
        y0 = support_y
        if kind == 'can':
            ys, rs = [y0, y0 + h], [r, r]
        else:
            body = rng.uniform(0.5, 0.75) if kind == 'bottle' else rng.uniform(0.8, 0.92)
            rn = r * (rng.uniform(0.25, 0.5) if kind == 'bottle' else rng.uniform(0.6, 0.85))
            yb = y0 + body * h
            ys_ = min(yb + rng.uniform(0.08, 0.2) * h, y0 + 0.97 * h)
            ys, rs = [y0, yb, ys_, y0 + h], [r, r, rn, rn]
        cz = distance + r
        obj = Revolution(cz * np.tan(ang), cz, np.array(ys), np.array(rs), rng.uniform(0.3, 0.9))
        size = np.array([2 * r, h, 2 * r])
        info['kind'] = kind
    else:
        w, h, d = rng.uniform(0.05, 0.30, 3)                     # boxes: 5-30 cm sides
        yaw = np.deg2rad(rng.normal(0, 10) if rng.uniform() < 0.5 else rng.uniform(-45, 45))
        half_z = (abs(np.cos(yaw)) * d + abs(np.sin(yaw)) * w) / 2
        cz = distance + half_z
        obj = RotatedBox(np.array([cz * np.tan(ang), support_y + h / 2, cz]), np.array([w, h, d]) / 2, yaw,
                         rng.uniform(0.3, 0.9))
        size = np.array([w, h, d])
        info['yaw_deg'] = float(np.degrees(yaw))
        info['kind'] = 'box'
    return cls, obj, size, info


def random_motion(rng, duration=None):
    """Hand motion of a user sweeping the device over the object from one viewpoint: up/down translation
    1-15 cm, left/right 0-5 cm, pitch sweeps (3-22 deg; the real exports' vertical extents imply roughly
    +-15 deg), small yaw wobble (0.05-2 deg; real stripes are sharp), occasional short yaw excursions (2-30 deg) that the IMU registers and a
    short hold at the end of the scan."""
    duration = rng.uniform(5, 25) if duration is None else duration
    exc = ()
    if rng.uniform() < 0.35:
        exc = tuple((float(rng.uniform(0, max(duration - 1.0, 0.5))), float(rng.uniform(0.4, 2.0)),
                     float(rng.choice([-1, 1]) * _logu(rng, 2, 30))) for _ in range(int(rng.integers(1, 3))))
    return HandMotion(
        duration_s=duration,
        trans_amp_m=(rng.uniform(0, 0.05), _logu(rng, 0.01, 0.15), rng.uniform(0, 0.02)),
        trans_period_s=(rng.uniform(4, 10), rng.uniform(2.5, 8), rng.uniform(5, 12)),
        rot_amp_deg=(rng.uniform(3, 22), _logu(rng, 0.05, 2.0), _logu(rng, 0.1, 2.0)),
        rot_period_s=(rng.uniform(2.5, 8), rng.uniform(3, 10), rng.uniform(5, 12)),
        jitter_deg=_logu(rng, 0.05, 0.4), excursions=exc, imu_err_deg=_logu(rng, 0.05, 0.5),
        hold_end_s=float(rng.uniform(0.3, 3.0)) if rng.uniform() < 0.7 else 0.0)


def _one_session(rng, n_side, n_complete, distance, close_frac):
    if distance is None:
        close = rng.uniform() < close_frac
        distance = _logu(rng, *(CLOSE_RANGE_M if close else FAR_RANGE_M))
    else:
        close = distance <= CLOSE_RANGE_M[1]
    frac = rng.uniform(0.15, 0.95)                               # sensor height as a fraction of the object height
    # object base height is set after the size is known (support_y = -frac * h); draw a provisional object
    cls, obj, size, info = _random_object(rng, distance, 0.0)
    shift = -frac * size[1]
    if isinstance(obj, Revolution):
        obj.ys = obj.ys + shift
    else:
        obj.center = obj.center + np.array([0.0, shift, 0.0])
    base_y = shift
    wall_z = max(_logu(rng, *WALL_RANGE_M), distance + size[2] + rng.uniform(0.10, 0.5))
    prims = [Plane(np.array([0, 0, wall_z]), np.array([0, 0, -1.0]), rng.uniform(0.3, 0.9)), obj]
    r_ = rng.uniform()
    if r_ < 0.5:                                                 # table / support under the object
        prims.append(Plane(np.array([0, base_y, 0]), np.array([0, 1.0, 0]), rng.uniform(0.2, 0.8)))
        info['support'] = 'table'
    elif r_ < 0.75:                                              # object held in the air above a room floor
        prims.append(Plane(np.array([0, -rng.uniform(0.6, 1.5), 0]), np.array([0, 1.0, 0]), rng.uniform(0.2, 0.8)))
        info['support'] = 'floor'
    else:
        info['support'] = 'none'
    if rng.uniform() < 0.35:                                     # optional side wall
        sx = rng.choice([-1, 1]) * rng.uniform(0.3, 1.5)
        prims.append(Plane(np.array([sx, 0, 0]), np.array([-np.sign(sx), 0, 0]), rng.uniform(0.3, 0.9)))
        info['side_wall_x'] = float(sx)
    motion = random_motion(rng)
    base = Rotation.from_euler('xyz', [rng.uniform(-10, 10), rng.uniform(-6, 6), rng.uniform(-4, 4)], degrees=True)
    sensor_kw = dict(noise_abs_m=float(rng.uniform(0.002, 0.008)), noise_rel=float(rng.uniform(0.005, 0.015)))
    raw, _ = scan(prims, motion, rng, n_side=n_side, base_rotation=base, sensor_kw=sensor_kw)
    export, center, scale = format_for_ai(raw, rng)
    gt = obj.surface(n_complete, rng)
    sensor_world = base.apply(IMU_TO_TOF_OFFSET)                 # where the viewer places the sensor
    truth = dict(distance_m=float(distance), wall_m=float(wall_z), size_m=size, scale=float(scale),
                 center=center, sensor_norm=(sensor_world - center) / scale, raw_points=len(raw),
                 motion=motion, base_rotation=base.as_euler('xyz', degrees=True),
                 regime='close' if close else 'far', sensor_height_frac=float(frac), sensor_kw=sensor_kw, **info)
    return Session(export, ((gt - center) / scale).astype(np.float32), cls, truth)


def random_session(rng, n_side=4, n_complete=16384, distance=None, close_frac=0.75, min_raw=4096, max_tries=5):
    """One simulated hand-held scan of one object, exported exactly like the real device.

    Every real export has 98-100 % unique points, i.e. its raw map held more than 4096 points. Sessions whose
    raw map is smaller than `min_raw` are redrawn (up to `max_tries`); `truth['tries']` records the number of
    draws and `truth['raw_ok']` whether the final draw met the condition."""
    for attempt in range(max_tries):
        s = _one_session(rng, n_side, n_complete, distance, close_frac)
        if s.truth['raw_points'] >= min_raw:
            break
    s.truth['tries'] = attempt + 1
    s.truth['raw_ok'] = bool(s.truth['raw_points'] >= min_raw)
    return s
