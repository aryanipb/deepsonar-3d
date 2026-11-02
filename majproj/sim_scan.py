"""Session simulator: one DeepSonar export (`scan_*_shapenet.npy`) including its real garbage, with per-point truth.

Why. The viewer maps from the moment mapping is switched on until the export button is pressed, so a file is the union
of everything the sensor saw in that time, not just the object. The real box files (inputs_box/) show it: a thin
horizontal ceiling layer (z about +0.9) with the 8x8 zone-grid pattern, an empty gap, then the dense object patch near the
sensor's own height, plus a few streaks. This module simulates such sessions end to end so a cleaner can be scored
(precision / recall against `object_mask`) and a model can be trained, without any real scan.

What is simulated (all in the viewer's z-up frame; z = gravity up exactly as in the exported files):
  1. START: device lying on the table pointing UP -> ceiling as a flat layer (device-to-ceiling 1.5-3 m);
  2. PICK-UP: it is lifted (still facing up, so the ceiling smears vertically) and rotated ~90 deg in 0.4-1.6 s ->
     sparse streaks across walls / furniture that happen to be in range;
  3. junk: bursts where a hand / fingers cover part of the FOV at 2-10 cm (a half plane in the sensor frame);
  4. SCAN: the user aims at an upright object (box / bottle / can / jar) standing on the table and holds 3-15 s with
     rotation wobble (registered by the IMU) and translation drift (NOT registered: the viewer tracks rotation only, so
     the object face is stored again at shifted depths / heights = lumps) of centimetres;
  5. PUT-DOWN swing and, usually, the ceiling again.
Optional back wall, table-top clutter, floor-standing furniture and random spurious zones. The device is the
VL53L5CX 8x8 model of `majproj.sim.device` (ST zone tables, mixed pixels, ~1 % + few mm noise, dropout, 20 mm - 4 m).

What is replicated exactly from the viewer: IMU-rotation-only registration with the fixed IMU->ToF offset rotated by
the IMU, sensor yaw 90 deg, lazy 10 mm voxel thinning once >= 500 points are pending (first point in a voxel wins),
and the export step of `format_for_ai.py` (random sample of 4096, with replacement only if fewer, bbox centre,
divide by the largest absolute coordinate). Provenance (which surface produced a point, in which phase) is carried
through the thinning and the sampling, so the truth is per exported point.

`distance_m` (sensor to the nearest object point at the nominal pose) is a parameter. The brief says 50-100 cm, but the
real box files put the object cluster within ~0.07-0.17 of the ceiling-grid half width of the sensor's own axis (a grid fit of
the 8x8 ceiling pattern gives the sensor position and scale), i.e. about 5-30 cm, with the hand moving the device up / down
by decimetres. The default prior is therefore 70 % near (0.05-0.30 m, log-uniform) and 30 % far (0.5-1.0 m); pass
`distance_m=` or `p_near=` to force either. See scripts/sim_scan_vs_real.py for the calibration table.

Calibration choices that the real files forced (each is a `SimConfig` field): the IMU heading is zeroed with the device (rest
yaw ~ 0, so the ceiling grid is axis aligned); lens roll-off cos^4 and a signal threshold that limits oblique / far returns to
~3 m (otherwise ceiling fans and walls swamp the file extents); motion blur invalidates zones during fast swings; tables
and floors are glossy (oblique returns vanish; the real files show no table layer); the swing sometimes pauses half way.

API:
    simulate_scan(rng, kind=None, **overrides) -> dict(export, object_mask, object_complete, truth)
    simulate_batch(n, seed, **overrides)       -> generator of such dicts (item i uses default_rng([seed, i]))
`overrides` are the fields of `SimConfig` (e.g. distance_m=0.3, drift_scale=0.0, p_ceiling_start=1.0).
"""
from dataclasses import dataclass, fields, replace

import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation, Slerp

from .sim import device
from .sim.acquisition import IMU_TO_TOF_OFFSET, Q14, SENSOR_YAW
from .sim.scene import Plane, Revolution, RotatedBox, cast

# ---------------------------------------------------------------------------------------------------------------
# labels

SOURCES = ('object', 'ceiling', 'floor', 'wall', 'table', 'clutter', 'hand', 'mixed', 'noise', 'person')
OBJECT, CEILING, FLOOR, WALL, TABLE, CLUTTER, HAND, MIXED, NOISE, PERSON = range(len(SOURCES))
PHASES = ('ceiling_start', 'pickup', 'scan', 'putdown', 'ceiling_end')
KINDS = ('box', 'bottle', 'can', 'jar')
KIND_P = (0.45, 0.30, 0.15, 0.10)

EXPORT_POINTS = 4096
COMPLETE_POINTS = 16384
_SYAW = SENSOR_YAW.as_matrix()
# y-up (scene.py primitives) <- z-up (viewer / IMU world): scene (x, y, z) = (x, z, -y), a proper rotation
W2S = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@dataclass
class SimConfig:
    # session timeline (seconds)
    fps: float = 15.0
    p_ceiling_start: float = 0.9
    ceiling_start_s: tuple = (2.0, 6.0)
    pickup_s: tuple = (0.4, 1.6)
    p_dwell: float = 0.6                 # the swing pauses for a moment half way (the hand re-grips): wall / furniture patches
    hold_s: tuple = (3.0, 12.0)
    putdown_s: tuple = (0.4, 1.6)
    p_ceiling_end: float = 0.6
    ceiling_end_s: tuple = (1.0, 4.0)
    # scene
    distance_m: float = None             # sensor -> nearest object point at the nominal pose; None = draw from the prior
    p_near: float = 0.7
    near_range_m: tuple = (0.05, 0.30)   # drawn log-uniform: the real box files imply an object held 5-30 cm from the sensor
    far_range_m: tuple = (0.50, 1.00)
    room_height_m: tuple = (2.5, 3.5)
    table_height_m: tuple = (0.60, 0.95)
    p_back_wall: float = 0.15            # a wall within range behind the object
    back_wall_gap_m: tuple = (0.3, 2.0)
    wall_range_m: tuple = (1.8, 6.5)     # distance of the other room walls from the scan position
    n_clutter_max: int = 3               # table-top clutter next to the object
    n_furniture_max: int = 2             # floor-standing boxes in the room
    p_person: float = 0.6                # the user standing / sitting next to the table (seen while the device is swung)
    # hand motion during the scan (scale 0 switches the effect off)
    drift_scale: float = 1.0             # translation drift, NOT registered by the viewer
    wobble_scale: float = 1.0            # rotation wobble, registered by the IMU
    imu_err_deg: tuple = (0.05, 0.4)     # slowly varying IMU orientation error (rms)
    imu_lag_s: tuple = (0.02, 0.08)      # IMU sample used for a ToF frame is this much older
    # junk
    p_hand_pickup: float = 0.5
    p_hand_scan: float = 0.25
    p_hand_putdown: float = 0.5
    blur_deg_s: float = 70.0             # zones stay valid with prob exp(-(omega/blur)^2): fast swings return almost nothing
    noise_rate_max: float = 0.04         # spurious zones per frame (rate drawn U(0, max) per session)
    # sensor
    n_side_scan: int = 4                 # sub-rays per zone side while the object is in view
    n_side_other: int = 2
    sensor: tuple = None                 # optional measure_batch kwargs as a tuple of (name, value)
    min_object_points: int = 25          # the session is redrawn if fewer exported points come from the object
    max_attempts: int = 8


_CFG_NAMES = {f.name for f in fields(SimConfig)}


def make_config(**kw):
    bad = set(kw) - _CFG_NAMES
    if bad:
        raise TypeError(f'unknown simulate_scan option(s): {sorted(bad)}; valid: {sorted(_CFG_NAMES)}')
    return replace(SimConfig(), **kw)


# ---------------------------------------------------------------------------------------------------------------
# helpers

def _u(rng, rngs):
    return float(rng.uniform(*rngs))


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _cosine_ramp(u):
    u = np.clip(u, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _aim_rotation(heading, elevation, roll):
    """IMU rotation whose optical axis (local +z) points along azimuth `heading`, `elevation` above the horizon."""
    return Rotation.from_matrix(_rz(heading) @ _ry(np.pi / 2 - elevation) @ _rz(roll))


def _ou(rng, n, dt, tau, sigma, dim):
    """Ornstein-Uhlenbeck samples on a grid: (n, dim), stationary sd `sigma`, correlation time `tau`."""
    a = np.exp(-dt / tau)
    z = rng.normal(0.0, 1.0, (n, dim))
    return lfilter([np.sqrt(1 - a * a) * sigma], [1.0, -a], z, axis=0)


class _Track:
    """Piecewise true pose of the IMU: position (m, z-up world) and rotation, evaluated at arbitrary times."""

    def __init__(self):
        self.segs = []

    def static(self, t0, t1, p, rot):
        self.segs.append(('static', t0, t1, np.asarray(p, float), rot))

    def move(self, t0, t1, p0, p1, r0, r1, dwell=None):
        """Lift + rotate from pose 0 to pose 1; `dwell` = (fraction of the rotation reached, start u, end u) pauses the
        rotation (not the lift) on the way."""
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        self.segs.append(('move', t0, t1, np.asarray(p0, float), np.asarray(p1, float), slerp, dwell))

    def hold(self, t0, t1, p, rot, drift_fn, wobble_fn):
        self.segs.append(('hold', t0, t1, np.asarray(p, float), rot, drift_fn, wobble_fn))

    @property
    def t_end(self):
        return self.segs[-1][2]

    def at(self, t):
        t = np.clip(np.asarray(t, float), 0.0, self.t_end - 1e-9)
        pos, quat = np.empty((len(t), 3)), np.empty((len(t), 4))
        for seg in self.segs:
            kind, t0, t1 = seg[:3]
            m = (t >= t0) & (t < t1)
            if not m.any():
                continue
            tt = t[m]
            if kind == 'static':
                pos[m] = seg[3]
                quat[m] = np.tile(seg[4].as_quat(), (len(tt), 1))
            elif kind == 'move':
                u = (tt - t0) / (t1 - t0)
                z_early = _cosine_ramp(u / 0.65)                   # lift first (still facing up) ...
                xy = _cosine_ramp(u)
                rot_u = _cosine_ramp((u - 0.25) / 0.75)            # ... then rotate
                if seg[6] is not None:
                    sd, ua, ub = seg[6]
                    rot_u = np.where(u < ua, sd * _cosine_ramp(u / ua),
                                     np.where(u < ub, sd, sd + (1 - sd) * _cosine_ramp((u - ub) / (1 - ub))))
                p0, p1 = seg[3], seg[4]
                pos[m] = np.stack([p0[0] + (p1[0] - p0[0]) * xy, p0[1] + (p1[1] - p0[1]) * xy,
                                   p0[2] + (p1[2] - p0[2]) * z_early], 1)
                quat[m] = seg[5](rot_u).as_quat()
            else:
                w = np.minimum((tt - t0) / 0.6, (t1 - tt) / 0.6)   # ramp the motion in/out at the boundaries
                ramp = _cosine_ramp(w)[:, None]
                pos[m] = seg[3] + ramp * seg[5](tt - t0)
                quat[m] = (seg[4] * Rotation.from_rotvec(ramp * seg[6](tt - t0))).as_quat()
        return pos, Rotation.from_quat(quat)


# ---------------------------------------------------------------------------------------------------------------
# scene

@dataclass
class _Scene:
    prims: list
    codes: np.ndarray            # source label of each primitive
    gloss: np.ndarray            # return strength ~ cos(incidence) ** gloss (1 = matte; polished table / floor: oblique returns vanish)
    target: object               # the object primitive (scene frame)
    meta: dict


def _to_scene(v):
    return np.asarray(v, float) @ W2S.T


def _box_prim(center_w, half_xyz, yaw, refl):
    """Box centred at `center_w` (z-up world) with half extents (hx along azimuth `yaw`, half height, hz along azimuth
    yaw - 90 deg); `yaw` is the world azimuth of the local x axis (checked against the W2S mapping)."""
    return RotatedBox(center=_to_scene(center_w), half=np.array(half_xyz, float), yaw=float(yaw), reflectivity=float(refl))


def _revolution(cx_w, cy_w, base_z, ys, rs, refl):
    s = _to_scene([cx_w, cy_w, 0.0])
    return Revolution(cx=float(s[0]), cz=float(s[2]), ys=np.asarray(ys, float) + base_z, rs=np.asarray(rs, float),
                      reflectivity=float(refl))


def _bottle_profile(rng, kind):
    r = rng.uniform(0.03, 0.045) if kind != 'jar' else rng.uniform(0.035, 0.06)
    if kind == 'bottle':
        h = rng.uniform(0.15, 0.30)
    elif kind == 'can':
        h = rng.uniform(0.10, 0.20)
    else:
        h = rng.uniform(0.08, 0.18)
    if kind == 'can':
        return h, r, [0.0, h], [r, r]
    body = rng.uniform(0.5, 0.75) if kind == 'bottle' else rng.uniform(0.8, 0.92)
    rn = r * (rng.uniform(0.25, 0.5) if kind == 'bottle' else rng.uniform(0.6, 0.85))
    yb = body * h
    ys_ = min(yb + rng.uniform(0.08, 0.2) * h, 0.97 * h)
    return h, r, [0.0, yb, ys_, h], [r, r, rn, rn]


def _build_scene(rng, cfg, kind):
    """Random room + table + target, defined around the hold position (the origin of the horizontal plane) and a
    heading psi. Returns the scene, the true track keypoints and the truth that does not depend on the motion."""
    psi = rng.uniform(0.0, 2 * np.pi)                        # azimuth the user aims at (the IMU frame yaw is arbitrary)
    hv = np.array([np.cos(psi), np.sin(psi), 0.0])           # heading
    lv = np.array([-np.sin(psi), np.cos(psi), 0.0])          # lateral
    H_room, H_t = _u(rng, cfg.room_height_m), _u(rng, cfg.table_height_m)
    if cfg.distance_m is not None:
        dist = float(cfg.distance_m)
    else:
        dist = _logu(rng, *cfg.near_range_m) if rng.random() < cfg.p_near else _u(rng, cfg.far_range_m)

    # target object (base on the table top at z = H_t); u = along the heading, v = lateral
    if kind == 'box':
        a, b, hgt = (_u(rng, (0.05, 0.30)) for _ in range(3))
        theta = float(np.clip(rng.normal(0.0, np.deg2rad(20.0)), -np.deg2rad(45.0), np.deg2rad(45.0)))
        half_depth = 0.5 * (a * abs(np.cos(theta)) + b * abs(np.sin(theta)))
        r_out = 0.5 * np.hypot(a, b)
        size = (a, b, hgt)
    else:
        hgt, r, ys, rs = _bottle_profile(rng, kind)
        half_depth = r_out = r
        size = (2 * r, 2 * r, hgt)
    u_c = dist + half_depth
    v_c = u_c * np.tan(rng.normal(0.0, np.deg2rad(4.0)))
    obj_xy = hv * u_c + lv * v_c
    refl_obj = rng.uniform(0.25, 0.7)
    if kind == 'box':
        target = _box_prim([obj_xy[0], obj_xy[1], H_t + hgt / 2], (a / 2, hgt / 2, b / 2), psi + theta, refl_obj)
    else:
        target = _revolution(obj_xy[0], obj_xy[1], H_t, ys, rs, refl_obj)
    d_centre = float(np.hypot(u_c, v_c))

    # hand pose during the scan: about the object's height, aimed at it with a little error
    hh = max(0.06, 0.55 * hgt + rng.normal(0.0, 0.05))
    z_hold = H_t + hh
    elev = np.arctan2(H_t + hgt / 2 - z_hold, d_centre) + np.deg2rad(rng.normal(0.0, 3.0))
    heading = psi + np.arctan2(v_c, u_c)
    roll = np.deg2rad(rng.normal(0.0, 12.0) + (rng.choice([0.0, 90.0, -90.0], p=[0.8, 0.1, 0.1])))

    # where the device lies on the table before / after
    for _ in range(40):
        ru, rv = rng.uniform(-0.5, 0.2), rng.uniform(-0.35, 0.35)
        if np.hypot(ru - u_c, rv - v_c) > r_out + 0.12:
            break
    rest = hv * ru + lv * rv
    rest2 = rest + (rng.normal(0.0, 0.04, 3) * [1, 1, 0] if rng.random() < 0.6 else 0.0)
    p_rest = np.array([rest[0], rest[1], H_t + 0.012])
    p_rest2 = np.array([rest2[0], rest2[1], H_t + 0.012])

    prims = [target]
    codes = [OBJECT]
    # table: a slab covering the device spots and the object (with margins; its near edge is the 'table edge')
    uu = [0.0, ru, rest2 @ hv, u_c - r_out, u_c + r_out]
    vv = [-0.3, 0.3, rv, rest2 @ lv, v_c - r_out, v_c + r_out]
    u0, u1 = min(uu) - _u(rng, (0.15, 0.5)), max(uu) + _u(rng, (0.1, 0.5))
    v0, v1 = min(vv) - _u(rng, (0.1, 0.3)), max(vv) + _u(rng, (0.1, 0.3))
    t_c = hv * (u0 + u1) / 2 + lv * (v0 + v1) / 2
    prims.append(_box_prim([t_c[0], t_c[1], H_t - 0.02], ((u1 - u0) / 2, 0.02, (v1 - v0) / 2), psi, rng.uniform(0.03, 0.2)))
    codes.append(TABLE)
    # ceiling and floor
    prims.append(Plane(point=_to_scene([0, 0, H_room]), normal=_to_scene([0, 0, 1]), reflectivity=rng.uniform(0.5, 0.85)))
    codes.append(CEILING)
    prims.append(Plane(point=_to_scene([0, 0, 0]), normal=_to_scene([0, 0, 1]), reflectivity=rng.uniform(0.15, 0.5)))
    codes.append(FLOOR)
    # four walls of a rectangular room, slightly rotated against the heading
    rho = np.deg2rad(rng.uniform(-15.0, 15.0))
    has_back = rng.random() < cfg.p_back_wall
    for k in range(4):
        ang = psi + rho + k * np.pi / 2
        n = np.array([np.cos(ang), np.sin(ang), 0.0])
        if k == 0:
            w = (u_c + r_out + _u(rng, cfg.back_wall_gap_m)) if has_back else _u(rng, (4.8, 7.5))
        else:
            w = _u(rng, cfg.wall_range_m)
        prims.append(Plane(point=_to_scene(n * w), normal=_to_scene(n), reflectivity=rng.uniform(0.4, 0.8)))
        codes.append(WALL)
    # table-top clutter
    n_clutter = int(rng.integers(0, cfg.n_clutter_max + 1))
    for _ in range(n_clutter):
        for _try in range(20):
            cu, cv = u_c + rng.uniform(-0.5, 0.5), v_c + rng.uniform(-0.5, 0.5)
            cr = rng.uniform(0.02, 0.10)
            in_table = (u0 + cr < cu < u1 - cr) and (v0 + cr < cv < v1 - cr)
            clear_obj = np.hypot(cu - u_c, cv - v_c) > r_out + cr + 0.02
            clear_dev = np.hypot(cu - ru, cv - rv) > cr + 0.1 and np.hypot(cu, cv) > cr + 0.1
            line = abs(cv * u_c - cu * v_c) / max(d_centre, 1e-6)          # distance to the sensor -> object line
            on_path = 0.0 < (cu * u_c + cv * v_c) / d_centre < d_centre + cr and line < cr + 0.08
            if in_table and clear_obj and clear_dev and (not on_path or rng.random() < 0.15):
                break
        else:
            continue
        ch = rng.uniform(0.03, 0.25)
        cxy = hv * cu + lv * cv
        refl = rng.uniform(0.2, 0.7)
        if rng.random() < 0.5:
            prims.append(_box_prim([cxy[0], cxy[1], H_t + ch / 2], (cr, ch / 2, cr * rng.uniform(0.6, 1.4)), rng.uniform(0, np.pi), refl))
        else:
            prims.append(_revolution(cxy[0], cxy[1], H_t, [0.0, ch], [cr, cr], refl))
        codes.append(CLUTTER)
    # the user: a torso + head column next to the table
    if rng.random() < cfg.p_person:
        ang, dd, pr = rng.uniform(0, 2 * np.pi), rng.uniform(0.45, 0.9), rng.uniform(0.15, 0.22)
        pxy = np.array([np.cos(ang), np.sin(ang)]) * dd
        prims.append(_revolution(pxy[0], pxy[1], H_t - 0.3, [0.0, rng.uniform(1.0, 1.4)], [pr, pr], rng.uniform(0.2, 0.6)))
        codes.append(PERSON)
    # floor-standing furniture
    for _ in range(int(rng.integers(0, cfg.n_furniture_max + 1))):
        ang, dd = rng.uniform(0, 2 * np.pi), rng.uniform(1.0, 3.5)
        fxy = np.array([np.cos(ang), np.sin(ang)]) * dd
        fw, fd, fh = rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.8), rng.uniform(0.4, 2.0)
        prims.append(_box_prim([fxy[0], fxy[1], fh / 2], (fw / 2, fh / 2, fd / 2), rng.uniform(0, np.pi), rng.uniform(0.2, 0.7)))
        codes.append(CLUTTER)

    gloss_of = {TABLE: (3.0, 8.0), FLOOR: (2.0, 5.0)}
    gloss = np.array([rng.uniform(*gloss_of[c]) if c in gloss_of else 1.0 for c in codes])
    scene = _Scene(prims, np.array(codes, np.int8), gloss, target,
                   dict(kind=kind, size_m=tuple(float(x) for x in size), distance_m=float(dist), room_height_m=float(H_room),
                        table_height_m=float(H_t), hold_height_m=float(hh), back_wall=bool(has_back),
                        heading_rad=float(psi), n_clutter=int(sum(c == CLUTTER for c in codes))))
    key = dict(p_rest=p_rest, p_rest2=p_rest2, z_hold=z_hold, heading=heading, elev=elev, roll=roll, psi=psi, hv=hv, lv=lv,
               u_c=u_c, v_c=v_c)
    return scene, key


# ---------------------------------------------------------------------------------------------------------------
# timeline and poses

def _build_timeline(rng, cfg, key):
    """True IMU track + frame times, per-frame phase ids and hand-junk parameters."""
    t = 0.0
    segs = []                                   # (phase, t0, t1)
    ceiling_start = rng.random() < cfg.p_ceiling_start
    ceiling_end = rng.random() < cfg.p_ceiling_end
    if ceiling_start:
        segs.append(('ceiling_start', t, t + _u(rng, cfg.ceiling_start_s)))
        t = segs[-1][2]
    segs.append(('pickup', t, t + _u(rng, cfg.pickup_s)))
    segs.append(('scan', segs[-1][2], segs[-1][2] + _u(rng, cfg.hold_s)))
    segs.append(('putdown', segs[-1][2], segs[-1][2] + _u(rng, cfg.putdown_s)))
    segs.append(('ceiling_end', segs[-1][2], segs[-1][2] + (_u(rng, cfg.ceiling_end_s) if ceiling_end else _u(rng, (0.1, 0.5)))))
    T = segs[-1][2]
    ph = {s[0]: s for s in segs}

    yaw0 = rng.normal(0.0, np.deg2rad(2.5))     # the IMU heading is zeroed with the device: the real ceiling grids are axis aligned
    def tilt():                                  # a device lying on a table is level to within about a degree
        ax = rng.normal(0, 1, 2)
        ax = ax / np.linalg.norm(ax)
        return Rotation.from_rotvec(np.array([ax[0], ax[1], 0.0]) * np.deg2rad(rng.uniform(0.0, 1.0)))
    R_rest = Rotation.from_matrix(_rz(yaw0)) * tilt()
    R_rest2 = Rotation.from_matrix(_rz(yaw0 + rng.normal(0.0, 0.15))) * tilt()
    R_aim = _aim_rotation(key['heading'], key['elev'], key['roll'])
    p_hold = np.array([0.0, 0.0, key['z_hold']])

    hold_n = 1.0 / 60.0
    t_hold0, t_hold1 = ph['scan'][1], ph['scan'][2]
    n_grid = int((t_hold1 - t_hold0) / hold_n) + 3
    grid_t = np.arange(n_grid) * hold_n
    hv, lv = key['hv'], key['lv']
    s_amp = cfg.drift_scale
    amp = np.array([_u(rng, (0.01, 0.05)), _u(rng, (0.01, 0.06)), _u(rng, (0.05, 0.22))]) * s_amp   # toward / side / up
    per = rng.uniform(2.0, 8.0, 3)
    phi = rng.uniform(0, 2 * np.pi, 3)
    rw = _ou(rng, n_grid, hold_n, 2.0, 0.006, 3) * s_amp
    basis = np.stack([hv, lv, [0.0, 0.0, 1.0]])               # (3, 3) rows: heading, lateral, up

    def drift_fn(tt):
        d = amp * np.sin(2 * np.pi * tt[:, None] / per + phi)
        d = d + np.stack([np.interp(tt, grid_t, rw[:, i]) for i in range(3)], 1)
        return d @ basis

    w_amp = np.deg2rad(rng.uniform(0.5, 3.0, 3)) * cfg.wobble_scale
    w_per = rng.uniform(1.5, 6.0, 3)
    w_phi = rng.uniform(0, 2 * np.pi, 3)
    w_n = _ou(rng, n_grid, hold_n, 1.0, np.deg2rad(0.4) * cfg.wobble_scale, 3)

    def wobble_fn(tt):
        w = w_amp * np.sin(2 * np.pi * tt[:, None] / w_per + w_phi)
        return w + np.stack([np.interp(tt, grid_t, w_n[:, i]) for i in range(3)], 1)

    tr = _Track()
    if ceiling_start:
        tr.static(*ph['ceiling_start'][1:], key['p_rest'], R_rest)
    def dwell():
        if rng.random() >= cfg.p_dwell:
            return None
        ua = rng.uniform(0.25, 0.5)
        return (rng.uniform(0.6, 0.9), ua, ua + rng.uniform(0.15, 0.35))

    tr.move(ph['pickup'][1], ph['pickup'][2], key['p_rest'], p_hold, R_rest, R_aim, dwell())
    tr.hold(t_hold0, t_hold1, p_hold, R_aim, drift_fn, wobble_fn)
    tr.move(ph['putdown'][1], ph['putdown'][2], p_hold, key['p_rest2'], R_aim, R_rest2, dwell())
    tr.static(ph['ceiling_end'][1], T, key['p_rest2'], R_rest2)

    n_frames = max(int(T * cfg.fps), 8)
    t_frame = (np.arange(n_frames) + 0.5) / cfg.fps
    phase_id = np.zeros(n_frames, np.int8)
    for i, name in enumerate(PHASES):
        if name in ph:
            phase_id[(t_frame >= ph[name][1]) & (t_frame < ph[name][2])] = i
    phase_id[t_frame >= T] = PHASES.index('ceiling_end')

    # hand / finger junk bursts: a half plane in the sensor frame at 2-10 cm
    junk = np.zeros(n_frames, bool)
    jp = np.zeros((n_frames, 5))                              # ux, uy, c, depth_m, refl
    for name, prob in (('pickup', cfg.p_hand_pickup), ('scan', cfg.p_hand_scan), ('putdown', cfg.p_hand_putdown)):
        if rng.random() >= prob:
            continue
        idx = np.nonzero(phase_id == PHASES.index(name))[0]
        if len(idx) == 0:
            continue
        for _ in range(int(rng.integers(1, 3))):
            ln = int(rng.integers(3, 21))
            s0 = int(rng.choice(idx))
            sl = slice(s0, min(s0 + ln, n_frames))
            a = rng.uniform(0, 2 * np.pi)
            junk[sl] = True
            jp[sl] = [np.cos(a), np.sin(a), rng.uniform(-0.45, 0.35), rng.uniform(0.02, 0.10), rng.uniform(0.4, 0.8)]

    # IMU-reported rotation: lagged, with a slowly varying error, quantized like the BNO08X Q14 quaternion
    lag = _u(rng, cfg.imu_lag_s)
    T_total = T
    p_true, rot_true = tr.at(t_frame)
    _, rot_lag = tr.at(t_frame - lag)
    sig = np.deg2rad(_u(rng, cfg.imu_err_deg))
    # the BNO08X orientation error is slow and, for a device at rest, constant for a while (hence exact duplicate
    # points in the un-thinned pending buffer): an OU process sampled at 2 Hz and held between samples
    n_hold = int(T_total * 2) + 3
    err_grid = _ou(rng, n_hold, 0.5, 3.0, sig, 3)
    err = err_grid[np.minimum((t_frame * 2).astype(int), n_hold - 1)]
    rot_imu = Rotation.from_rotvec(err) * rot_lag
    q = np.round(rot_imu.as_quat() * Q14) / Q14
    rot_imu = Rotation.from_quat(q)
    tl = dict(phase_id=phase_id, junk=junk, junk_params=jp, p_true=p_true, R_true=rot_true.as_matrix(),
              R_imu=rot_imu.as_matrix(), t_frame=t_frame, duration_s=float(T), phases=[s[0] for s in segs],
              hold_s=float(t_hold1 - t_hold0), drift_amp_m=amp.tolist(), imu_err_deg=float(np.rad2deg(sig)), imu_lag_s=float(lag))
    return tl


# ---------------------------------------------------------------------------------------------------------------
# measurement + viewer map

class _LabelledMap:
    """`ViewerMap` (majproj.sim.acquisition) / viewer.MappingState with a parallel label array (source, phase).

    Thinning is the viewer's: only when >= 500 points are pending, one point per 10 mm voxel, the first one in
    arrival order (the existing master points come first) wins."""

    def __init__(self, voxel_m=0.010):
        self.voxel = voxel_m
        self.points = np.empty((0, 3))
        self.labels = np.empty((0, 2), np.int8)
        self.pending, self.pending_labels = [], []
        self._n_pending = 0

    def add(self, pts, labels):
        self.pending.append(pts)
        self.pending_labels.append(labels)
        self._n_pending += len(pts)
        if self._n_pending < 500:
            return
        allp = np.vstack([self.points] + self.pending)
        keys = np.floor(allp / self.voxel).astype(np.int64)
        keys -= keys.min(0)
        span = keys.max(0) + 1
        flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        _, idx = np.unique(flat, return_index=True)
        self.points = allp[idx]
        self.labels = np.vstack([self.labels] + self.pending_labels)[idx]
        self.pending, self.pending_labels, self._n_pending = [], [], 0

    def export(self):
        if not self.pending:
            return self.points, self.labels
        return np.vstack([self.points] + self.pending), np.vstack([self.labels] + self.pending_labels)


def _export(points, rng, target=EXPORT_POINTS):
    """The viewer's format_for_ai step; also returns the sample indices, the bbox centre and the scale."""
    n = len(points)
    idx = rng.choice(n, target, replace=n < target)
    p = points[idx].astype(np.float64)
    centre = (p.max(0) + p.min(0)) / 2.0
    p = p - centre
    m = float(np.max(np.abs(p)))
    if m > 0:
        p = p / m
    return p.astype(np.float32), idx, centre, m


def _run_session(rng, cfg, scene, tl):
    """Ray cast + measure every frame, register like the viewer, return the raw map and per-frame info."""
    sensor_kw = dict(cfg.sensor or ())
    sensor_kw.setdefault('noise_abs_m', _u(rng, (0.002, 0.004)))
    sensor_kw.setdefault('noise_rel', _u(rng, (0.002, 0.006)))
    sensor_kw.setdefault('dropout', _u(rng, (0.005, 0.03)))
    sensor_kw.setdefault('signal_threshold', _u(rng, (0.03, 0.06)))   # indoor ambient light: ~3 m for a 60 % target, oblique far returns vanish
    noise_abs, noise_rel = sensor_kw['noise_abs_m'], sensor_kw['noise_rel']
    noise_rate = _u(rng, (0.0, cfg.noise_rate_max))

    R_true = Rotation.from_matrix(tl['R_true'])
    omega = np.zeros(len(tl['phase_id']))                         # device angular rate (deg/s) from the true rotation
    if len(omega) > 1:
        omega[1:] = np.rad2deg((R_true[:-1].inv() * R_true[1:]).magnitude()) * cfg.fps
        omega[0] = omega[1]
    S_true = tl['R_true'] @ _SYAW
    S_imu = tl['R_imu'] @ _SYAW
    origin = tl['p_true'] + tl['R_true'] @ IMU_TO_TOF_OFFSET     # true ToF position (moves with the hand)
    off_map = tl['R_imu'] @ IMU_TO_TOF_OFFSET                    # viewer: fixed offset rotated by the IMU
    phase_id, junk, jp = tl['phase_id'], tl['junk'], tl['junk_params']
    n_frames = len(phase_id)
    ns = np.where(phase_id == PHASES.index('scan'), cfg.n_side_scan, cfg.n_side_other)
    prims, codes = scene.prims, scene.codes
    vmap = _LabelledMap()
    n_raw = 0
    obj_seen = np.zeros(n_frames, bool)
    chunk_max_rays = 120_000
    f = 0
    while f < n_frames:
        n = int(ns[f])
        sub = device.subray_dirs(n).reshape(-1, 3)                # (64 n^2, 3) local, zone-major
        m = len(sub)
        n_sub = n * n                                             # sub-rays per zone
        vign = sub[:, 2] ** 4                                     # lens roll-off (cos^4): corner zones return less
        f2 = f
        while f2 < n_frames and ns[f2] == n and (f2 - f + 1) * m <= chunk_max_rays:
            f2 += 1
        sl = slice(f, f2)
        F = f2 - f
        dirs_w = np.matmul(sub[None], S_true[sl].transpose(0, 2, 1))             # (F, m, 3) true world dirs
        dirs_s = (dirs_w @ W2S.T).reshape(-1, 3)
        o_s = np.repeat(origin[sl] @ W2S.T, m, axis=0)
        t, normal, refl, which = cast(prims, o_s, dirs_s)
        cosi = np.abs((normal * dirs_s).sum(1))
        src = np.where(which >= 0, codes[np.maximum(which, 0)], -1).astype(np.int8)
        t, refl, cosi = t.reshape(F, m), refl.reshape(F, m), cosi.reshape(F, m)
        src = src.reshape(F, m)
        expo = np.where(which >= 0, scene.gloss[np.maximum(which, 0)], 1.0).reshape(F, m)
        for k in np.nonzero(junk[sl])[0]:                                         # hand / fingers: local half plane
            ux, uy, c, dh, rh = jp[f + k]
            tx, ty = sub[:, 0] / sub[:, 2], sub[:, 1] / sub[:, 2]
            occ = (tx * ux + ty * uy) > c
            th = dh / sub[:, 2]
            upd = occ & (th < t[k])
            t[k] = np.where(upd, th, t[k])
            refl[k] = np.where(upd, rh, refl[k])
            cosi[k] = np.where(upd, sub[:, 2], cosi[k])
            expo[k] = np.where(upd, 1.0, expo[k])
            src[k] = np.where(upd, HAND, src[k])
        fin = np.isfinite(t)
        w = np.where(fin, refl * cosi ** expo * vign[None, :] / np.maximum(np.where(fin, t, 1.0), 0.05) ** 2, 0.0)
        r = np.where(fin, t, 99.0)
        d_perp, status = device.measure_batch(r.reshape(F, device.NUM_ZONES, n_sub), w.reshape(F, device.NUM_ZONES, n_sub),
                                              rng, **sensor_kw)
        # provenance of every reported zone: the surface of the sub-ray whose range is closest to the reported one
        r_rep = d_perp / 1000.0 / device.ZONE_PITCH_SIN[None, :]
        r3, w3, s3 = (a.reshape(F, device.NUM_ZONES, n_sub) for a in (r, w, src))
        dif = np.where(w3 > 0, np.abs(r3 - r_rep[..., None]), np.inf)
        j = dif.argmin(2)
        zone_src = np.take_along_axis(s3, j[..., None], 2)[..., 0].astype(np.int8)
        tol = 0.025 + 4.0 * (noise_abs + noise_rel * r_rep)
        zone_src = np.where(dif.min(2) > tol, MIXED, zone_src)
        valid = (status == 5) & (d_perp >= device.MIN_RANGE_MM)
        if cfg.blur_deg_s > 0:                                                     # motion blur invalidates zones
            valid &= rng.random(valid.shape) < np.exp(-(omega[sl] / cfg.blur_deg_s) ** 2)[:, None]
        # spurious zones: a random range in a random zone, reported as valid
        n_noise = rng.poisson(noise_rate, F)
        for k in np.nonzero(n_noise)[0]:
            ok = np.nonzero(valid[k])[0]
            if len(ok) == 0:
                continue
            for z in rng.choice(ok, min(n_noise[k], len(ok)), replace=False):
                d_perp[k, z] *= rng.uniform(0.3, 1.3)        # a plausible but wrong range (multipath / ghost)
                d_perp[k, z] = max(d_perp[k, z], device.MIN_RANGE_MM)
                zone_src[k, z] = NOISE
        local = device.backproject_batch(d_perp)                                   # (F, 64, 3)
        world = np.matmul(local, S_imu[sl].transpose(0, 2, 1)) + off_map[sl][:, None, :]
        for k in range(F):
            v = valid[k]
            if v.any():
                lab = np.stack([zone_src[k][v], np.full(v.sum(), phase_id[f + k], np.int8)], 1)
                vmap.add(world[k][v], lab)
                n_raw += int(v.sum())
                obj_seen[f + k] = bool((zone_src[k][v] == OBJECT).any())
        f = f2
    return vmap, n_raw, obj_seen


# ---------------------------------------------------------------------------------------------------------------
# public API

def _scene_to_world(pts_scene):
    return np.asarray(pts_scene, float) @ W2S


def simulate_scan(rng, kind=None, **kw):
    """Simulate one session and its export. See the module docstring.

    Returns dict(export (4096, 3) float32, object_mask (4096,) bool, object_complete (16384, 3) float32, truth dict).
    `truth` has kind, size_m (extent x, extent y, height; diameter, diameter, height for round objects), distance_m,
    phases (names from PHASES present in the session) and hand_junk (bool: hand / finger bursts occurred), n_raw (valid points before thinning), n_map (points in the map at export), n_unique, scale_m (metres per
    export unit), centre_m, source (4096 int labels, see SOURCES), source_names, phase_id (4096 ints, see PHASES),
    attempts and the scene / hand-motion parameters."""
    cfg = make_config(**kw)
    kind = str(rng.choice(KINDS, p=KIND_P)) if kind is None else str(kind)
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}')
    best = None
    for attempt in range(1, cfg.max_attempts + 1):
        scene, key = _build_scene(rng, cfg, kind)
        tl = _build_timeline(rng, cfg, key)
        vmap, n_raw, obj_seen = _run_session(rng, cfg, scene, tl)
        pts, lab = vmap.export()
        if len(pts) < 50:
            continue
        exp, idx, centre, m = _export(pts, rng)
        lab_e = lab[idx]
        n_obj = int((lab_e[:, 0] == OBJECT).sum())
        res = (scene, key, tl, pts, exp, lab_e, centre, m, n_raw, obj_seen, attempt)
        if best is None or n_obj > int((best[5][:, 0] == OBJECT).sum()):
            best = res
        if n_obj >= cfg.min_object_points:
            break
    if best is None:
        raise RuntimeError('simulate_scan: no session produced enough valid points')
    scene, key, tl, pts, exp, lab_e, centre, m, n_raw, obj_seen, attempt = best
    # complete object surface in the export frame: the viewer places the object relative to the (untracked)
    # sensor, so use the mean true IMU position over the scan frames in which the object was seen as the nominal pose
    in_scan = tl['phase_id'] == PHASES.index('scan')
    sel = obj_seen & in_scan if (obj_seen & in_scan).any() else in_scan
    p_nom = tl['p_true'][sel].mean(0)
    comp = _scene_to_world(scene.target.surface(COMPLETE_POINTS, rng)) - p_nom
    comp = ((comp - centre) / m).astype(np.float32)
    mask = lab_e[:, 0] == OBJECT
    uniq = int(len(np.unique(exp, axis=0)))
    present = [PHASES[i] for i in range(len(PHASES)) if (tl['phase_id'] == i).any()]
    truth = dict(scene.meta)
    truth.update(
        phases=present, hand_junk=bool(tl['junk'].any()),
        n_raw=int(n_raw), n_map=int(len(pts)), n_unique=uniq, n_object=int(mask.sum()),
        scale_m=float(m), centre_m=[float(c) for c in centre], source=lab_e[:, 0].astype(np.int8),
        source_names=SOURCES, phase_id=lab_e[:, 1].astype(np.int8), phase_names=PHASES, attempts=attempt,
        duration_s=tl['duration_s'], hold_s=tl['hold_s'], drift_amp_m=tl['drift_amp_m'], imu_err_deg=tl['imu_err_deg'],
        imu_lag_s=tl['imu_lag_s'], n_frames=int(len(tl['phase_id'])), nominal_pose_m=[float(v) for v in p_nom],
        object_visible_frames=int(obj_seen.sum()))
    return dict(export=exp, object_mask=mask, object_complete=comp, truth=truth)


def simulate_batch(n, seed=0, **kw):
    """Generator of `n` sessions; item i is `simulate_scan(default_rng([seed, i]), **kw)` (so batches can be split)."""
    for i in range(n):
        yield simulate_scan(np.random.default_rng([seed, i]), **kw)
