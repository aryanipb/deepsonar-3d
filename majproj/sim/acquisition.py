"""Simulated acquisition: hand-held motion -> VL53L5CX frames -> DeepSonar viewer mapping -> export.

The viewer replica mirrors VL53L5CX-BNO08X-viewer/viewer/viewer.py (MIT, (c) 2026 Henrique Ferrolho):
points are registered with the IMU rotation only and a fixed sensor offset (translation is never
tracked), accumulated with lazy 10 mm voxel thinning (only when >= 500 points are pending), and the
export is the viewer's own `format_for_ai` step (random sample / duplicate to 4096, bbox centre,
divide by max-abs).

v2: the whole session (all frames) is ray cast and measured in vectorized batches; only the map update
(which is order dependent) runs frame by frame. Hand motion supports up/down and left/right translation,
pitch/yaw/roll wobble and occasional short large yaw excursions that the IMU registers. The IMU
orientation the viewer sees is quantized like the BNO08X rotation vector (Q14) with a small slowly
varying error; the true device pose is used for the ray geometry.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from . import device
from .scene import cast

IMU_TO_TOF_OFFSET = np.array([0.0, -0.0254, 0.0])   # config.TOF_BOARD.world_position - IMU world_position
SENSOR_YAW = Rotation.from_euler('z', 90.0, degrees=True)
_SYAW_M = SENSOR_YAW.as_matrix()
Q14 = 16384.0                                        # BNO08X rotation-vector quaternion resolution


@dataclass
class HandMotion:
    duration_s: float = 30.0
    fps: float = 15.0
    trans_amp_m: tuple = (0.03, 0.10, 0.01)      # left-right, up-down, forward-back amplitude
    trans_period_s: tuple = (7.0, 4.0, 9.0)
    rot_amp_deg: tuple = (8.0, 2.0, 2.0)          # pitch (about x), yaw (about y), roll (about z)
    rot_period_s: tuple = (4.0, 7.0, 11.0)
    jitter_m: float = 0.002
    jitter_deg: float = 0.3
    excursions: tuple = ()                        # short yaw turns: (t_start_s, duration_s, peak_deg)
    imu_err_deg: float = 0.0                      # slowly varying IMU orientation error (rms, deg)
    imu_quantize: bool = True                     # BNO08X Q14 quaternion quantization
    hold_end_s: float = 0.0                       # device held still at the end (pose frozen, IMU included)

    def pose_arrays(self, rng, base_rotation=None):
        """Per ToF frame: true IMU rotation matrices (F,3,3), IMU positions (F,3) and the rotation
        matrices the IMU reports to the viewer (F,3,3)."""
        n = max(int(self.duration_s * self.fps), 1)
        t = np.arange(n) / self.fps
        ph = rng.uniform(0, 2 * np.pi, 6)
        trans = np.stack([a * np.sin(2 * np.pi * t / p + q) for a, p, q in
                          zip(self.trans_amp_m, self.trans_period_s, ph[:3])], 1)
        trans += np.cumsum(rng.normal(0, self.jitter_m, (n, 3)), 0) * 0.2
        ang = np.stack([a * np.sin(2 * np.pi * t / p + q) for a, p, q in
                        zip(self.rot_amp_deg, self.rot_period_s, ph[3:])], 1)
        ang += rng.normal(0, self.jitter_deg, (n, 3))
        for t0, dur, peak in self.excursions:                       # raised-cosine yaw bump about y
            u = (t - t0) / max(dur, 1e-3)
            inside = (u >= 0) & (u <= 1)
            ang[:, 1] += np.where(inside, peak * 0.5 * (1 - np.cos(2 * np.pi * np.clip(u, 0, 1))), 0.0)
        base = (base_rotation or Rotation.identity()).as_matrix()
        R = base @ Rotation.from_euler('xyz', ang, degrees=True).as_matrix()
        R_imu = R
        if self.imu_err_deg > 0:
            e = np.zeros((n, 3))
            a = np.exp(-1.0 / (3.0 * self.fps))                      # ~3 s correlation time
            z = rng.normal(0, 1, (n, 3))
            for i in range(1, n):
                e[i] = a * e[i - 1] + np.sqrt(1 - a * a) * z[i]
            R_imu = R @ Rotation.from_rotvec(np.deg2rad(self.imu_err_deg) * e).as_matrix()
        if self.imu_quantize:
            q = np.round(Rotation.from_matrix(R_imu).as_quat() * Q14) / Q14
            R_imu = Rotation.from_quat(q).as_matrix()               # from_quat renormalizes
        n_hold = int(round(self.hold_end_s * self.fps))
        if 0 < n_hold < n - 1:                                       # frozen pose -> identical IMU readings
            k = n - n_hold - 1
            R[k + 1:], trans[k + 1:], R_imu[k + 1:] = R[k], trans[k], R_imu[k]
        return R, trans, R_imu

    def poses(self, rng, base_rotation=None):
        """True device poses per ToF frame: list of (Rotation of IMU, IMU position)."""
        R, trans, _ = self.pose_arrays(rng, base_rotation)
        return list(zip(Rotation.from_matrix(R), trans))


class ViewerMap:
    """MappingState replica (viewer.py): pending buffer + lazy voxel downsample + max-points cap."""

    def __init__(self, voxel_m=0.010, max_points=100_000):
        self.voxel, self.max_points = voxel_m, max_points
        self.points = np.empty((0, 3))
        self.pending = []
        self._n_pending = 0

    def add(self, pts):
        self.pending.append(pts)
        self._n_pending += len(pts)
        if self._n_pending < 500:
            return
        allp = np.vstack([self.points] + self.pending)
        keys = np.floor(allp / self.voxel).astype(np.int64)
        keys -= keys.min(0)
        span = keys.max(0) + 1
        flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]   # same (x, y, z) order as viewer
        _, idx = np.unique(flat, return_index=True)
        new = allp[idx]
        self.points = new[-self.max_points:] if len(new) > self.max_points else new
        self.pending = []
        self._n_pending = 0

    def export(self):
        return np.vstack([self.points] + self.pending) if self.pending else self.points


def scan(prims, motion, rng, imu_connected=True, n_side=6, sensor_kw=None, base_rotation=None, chunk=64):
    """Run a hand-held mapping session over a scene. Returns (raw metric map, true ToF positions)."""
    sub_local = device.subray_dirs(n_side).reshape(-1, 3)             # (64 n, 3), zone-major
    n = sub_local.shape[0] // device.NUM_ZONES
    R, pos, R_imu = motion.pose_arrays(rng, base_rotation)
    S_true, S_imu = R @ _SYAW_M, R_imu @ _SYAW_M
    origin = pos + R @ IMU_TO_TOF_OFFSET                              # true ToF position (moves with the hand)
    off_map = R_imu @ IMU_TO_TOF_OFFSET                               # viewer: fixed offset rotated by the IMU
    vmap = ViewerMap()
    kept = []
    for a in range(0, len(R), chunk):
        sl = slice(a, a + chunk)
        F = len(origin[sl])
        dirs = np.matmul(sub_local[None], S_true[sl].transpose(0, 2, 1)).reshape(-1, 3)
        o = np.repeat(origin[sl], len(sub_local), axis=0)
        t, normal, refl, _ = cast(prims, o, dirs)
        cosi = np.abs((normal * dirs).sum(1))
        fin = np.isfinite(t)
        w = np.where(fin, refl * cosi / np.maximum(np.where(fin, t, 1.0), 0.05) ** 2, 0.0)
        r = np.where(fin, t, 99.0)
        d_perp, status = device.measure_batch(r.reshape(F, device.NUM_ZONES, n),
                                              w.reshape(F, device.NUM_ZONES, n), rng, **(sensor_kw or {}))
        valid = (status == 5) & (d_perp >= device.MIN_RANGE_MM)
        local = device.backproject_batch(d_perp)                       # (F, 64, 3)
        # viewer registration: IMU rotation only; the ToF position is the fixed offset rotated by the IMU
        if imu_connected:
            world = np.matmul(local, S_imu[sl].transpose(0, 2, 1)) + off_map[sl][:, None, :]
        else:
            world = local @ _SYAW_M.T + IMU_TO_TOF_OFFSET
        for f in range(F):
            if valid[f].any():
                vmap.add(world[f][valid[f]])
                kept.append(origin[sl][f])
    return vmap.export(), np.array(kept)


def format_for_ai(points, rng, target=4096):
    """Exact copy of the viewer export / format_for_ai.py preprocessing."""
    n = len(points)
    idx = rng.choice(n, target, replace=n < target)
    p = points[idx].astype(np.float64)
    center = (p.max(0) + p.min(0)) / 2.0
    p = p - center
    m = np.max(np.abs(p))
    if m > 0:
        p = p / m
    return p.astype(np.float32), center, m
