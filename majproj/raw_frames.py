"""Invert a raw DeepSonar map file back into the sensor frames that produced it.

Found on the 27 real files of 2026-10-30: a raw file is NOT a thinned map. It is the time-ordered list of frames, nothing
dropped: each frame's valid zones in ascending zone index, the whole block written 1, 2 (usual) or 3 times in a row.
A row is ZONE_DIRS[z] * d / sin(pitch[z]) rotated by the IMU, plus the fixed IMU->ToF offset rotated too (`device.py`,
`acquisition.IMU_TO_TOF_OFFSET`). So every frame can be solved exactly: the rotation and the zone of each row fit with
residuals of about 1e-6 degrees, and the integer millimetre reading of every zone comes back.

    frames(a)   -> list of (first row, valid zones L, copies)
    recover(a)  -> dict(R (F, 3, 3) sensor->world, NaN where unsolved; mm (F, 64) int, 0 = invalid; start, length, copies)

Unsolved frames are the very short ones (fewer than 3 zones) and frames after a jump too large to track with few zones.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .sim import device
from .sim.acquisition import IMU_TO_TOF_OFFSET, SENSOR_YAW

Z = device.ZONE_DIRS
_O2 = SENSOR_YAW.as_matrix().T @ np.asarray(IMU_TO_TOF_OFFSET)
_RZ90 = Rotation.from_euler('z', 90, degrees=True).as_matrix()
_GRID = Rotation.from_euler('zyx', [(y, p, r) for y in range(0, 360, 20) for p in range(-180, 180, 30) for r in range(-90, 91, 30)],
                            degrees=True).as_matrix()
_LOCAL = Rotation.from_rotvec(np.radians([(a, b, c) for a in (-8, -4, 0, 4, 8) for b in (-8, -4, 0, 4, 8) for c in (-8, -4, 0, 4, 8)])).as_matrix()
MAX_ERR_DEG = 0.02


def frames(a):
    """Split the rows into frames by exact block repetition. Rows that repeat nothing form single-copy frames (up to 64 rows)."""
    n, out, i, pend = len(a), [], 0, None

    def flush(p, e):
        while p < e:
            out.append((p, min(64, e - p), 1))
            p += min(64, e - p)

    while i < n:
        found = 0
        for L in range(1, 65):
            if i + 2 * L <= n and np.array_equal(a[i:i + L], a[i + L:i + 2 * L]):
                found = L
                break
        if found:
            if pend is not None:
                flush(pend, i)
                pend = None
            c, k = 2, i + 2 * found
            while k + found <= n and np.array_equal(a[i:i + found], a[k:k + found]):
                c, k = c + 1, k + found
            out.append((i, found, c))
            i = k
        else:
            pend = i if pend is None else pend
            i += 1
    if pend is not None:
        flush(pend, n)
    return out


def _fit(p, R, iters):
    for _ in range(iters):
        q = p - R @ _O2
        u = q / np.linalg.norm(q, axis=1, keepdims=True)
        j = (u @ (Z @ R.T).T).argmax(1)
        U, _, Vt = np.linalg.svd(Z[j].T @ u)
        V = Vt.T
        R = V @ np.diag([1.0, 1.0, np.linalg.det(V @ U.T)]) @ U.T
    q = p - R @ _O2
    u = q / np.linalg.norm(q, axis=1, keepdims=True)
    c = u @ (Z @ R.T).T
    return R, c.argmax(1), np.degrees(np.arccos(np.clip(c.max(1), -1, 1)))


def _good(j, e):
    return e.max() < MAX_ERR_DEG and np.all(np.diff(j) > 0)


def _solve(p, Rp):
    L = len(p)
    if Rp is not None and L >= 3:
        for R0 in [Rp] + list(Rp @ _LOCAL):
            R, j, e = _fit(p, R0, 8 if R0 is Rp else 5)
            if _good(j, e):
                return R, j
    if L >= 8:
        u = p / np.linalg.norm(p, axis=1, keepdims=True)
        score = np.median(np.einsum('ij,gkj->gik', u, Z @ _GRID.transpose(0, 2, 1)).max(2), axis=1)
        for k in np.argsort(score)[-24:]:
            for q in range(4):
                R, j, e = _fit(p, _GRID[k] @ np.linalg.matrix_power(_RZ90, q), 12)
                if _good(j, e):
                    return R, j
    return None


def recover(a):
    fs = frames(a)
    n = len(fs)
    R, mm, Rp = np.full((n, 3, 3), np.nan), np.zeros((n, 64), int), None
    for i, (s, L, _) in enumerate(fs):
        p = a[s:s + L]
        sol = _solve(p, Rp)
        if sol is None:
            continue
        Rp, j = sol
        R[i] = Rp
        mm[i, j] = np.round(((p - Rp @ _O2) @ Rp)[:, 2] * 1000.0).astype(int)
    t = np.array(fs).reshape(-1, 3)
    return dict(R=R, mm=mm, start=t[:, 0], length=t[:, 1], copies=t[:, 2])
