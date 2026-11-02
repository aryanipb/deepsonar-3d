"""Real-environment bank for majproj.sim_raw: per-session measured frames with the real object REMOVED.

    PYTHONPATH=. uv run python scripts/build_real_bank.py    -> configs/real_bank.npz, configs/real_bank.json

A raw file is a sequence of sensor frames (valid zones in ascending order), each written usually twice. Every point
satisfies p = R (local + o) for one frame rotation R, so fitting the ST zone tables recovers per frame the exact IMU
rotation and the 8x8 grid of integer millimetre ranges (angular residual about 1e-6 deg). Those measured frames are the
environment of the 'real' simulation regime: nothing about the room is invented.

The object is cut out geometrically (a sector around the cleaned object, above the table) and never stored, so the bank
can not leak the shape of the real objects. Only one 1-D series derived from it is kept: the hand drift along the view.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as Rot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from majproj.clean import clean                                              # noqa: E402
from majproj.sim import device as D                                          # noqa: E402
from majproj.sim.acquisition import IMU_TO_TOF_OFFSET, SENSOR_YAW            # noqa: E402
from scripts.run_pipeline import thin                                        # noqa: E402

# the longest export of each session (the others are subsets of these)
SESSIONS = ('114126 115102 115330 115840 120051 120408 120745 121055 121223 121457 121600 '
            '122622 122735 122907 123028 123146 123259 123523 123658 123815').split()
Z = D.ZONE_DIRS
O2 = SENSOR_YAW.as_matrix().T @ np.asarray(IMU_TO_TOF_OFFSET)
_RZ90 = Rot.from_euler('z', 90, degrees=True).as_matrix()
_GRID = [Rot.from_euler('zyx', [y, p, r], degrees=True).as_matrix()
         for y in range(0, 360, 20) for p in range(-180, 180, 30) for r in range(-90, 91, 30)]
_LOCAL = [Rot.from_rotvec(np.radians([a, b, c])).as_matrix() for a in (-8, -4, 0, 4, 8) for b in (-8, -4, 0, 4, 8) for c in (-8, -4, 0, 4, 8)]
AZ_HALF_DEG, HR_MAX, HR_MIN, Z_MAX, TABLE_MARGIN = 40.0, 0.54, 0.15, 0.50, 0.012


def split_frames(a):
    """(start, length, copies): a block repeated exactly is a frame; unrepeated rows become blocks of at most 64."""
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
        if not found:
            pend = i if pend is None else pend
            i += 1
            continue
        if pend is not None:
            flush(pend, i)
            pend = None
        c, k = 2, i + 2 * found
        while k + found <= n and np.array_equal(a[i:i + found], a[k:k + found]):
            c, k = c + 1, k + found
        out.append((i, found, c))
        i = k
    if pend is not None:
        flush(pend, n)
    return out


def _fit(p, R, iters):
    for _ in range(iters):
        q = p - R @ O2
        u = q / np.linalg.norm(q, axis=1, keepdims=True)
        j = (u @ (Z @ R.T).T).argmax(1)
        U, _, Vt = np.linalg.svd(Z[j].T @ u)
        V = Vt.T
        R = V @ np.diag([1, 1, np.linalg.det(V @ U.T)]) @ U.T
    q = p - R @ O2
    u = q / np.linalg.norm(q, axis=1, keepdims=True)
    c = u @ (Z @ R.T).T
    return R, c.argmax(1), np.degrees(np.arccos(np.clip(c.max(1), -1, 1)))


def _good(j, e):
    return e.max() < 0.02 and np.all(np.diff(j) > 0)


def recover(a):
    out, Rp = [], None
    for s, l, c in split_frames(a):
        p, best = a[s:s + l], None
        if Rp is not None and l >= 3:
            for g in [np.eye(3)] + _LOCAL:
                r = _fit(p, Rp @ g, 8 if g is _LOCAL[0] else 5)
                if _good(r[1], r[2]):
                    best = r
                    break
        if best is None and l >= 8:
            u = p / np.linalg.norm(p, axis=1, keepdims=True)
            sc = [np.median((u @ (Z @ g.T).T).max(1)) for g in _GRID]
            for k in np.argsort(sc)[-24:]:
                for q in range(4):
                    r = _fit(p, _GRID[k] @ np.linalg.matrix_power(_RZ90, q), 12)
                    if _good(r[1], r[2]) and (best is None or r[2].mean() < best[2].mean()):
                        best = r
        if best is None:
            out.append((s, l, c, None, None))
        else:
            Rp = best[0]
            out.append((s, l, c, best[0], best[1]))
    return out


def table_height(a, lo_z):
    """Mode of z just under the object's lower edge (the table is the only horizontal surface there)."""
    hr = np.hypot(a[:, 0], a[:, 1])
    z = a[(hr > 0.15) & (hr < 1.0) & (a[:, 2] < lo_z + 0.03) & (a[:, 2] > lo_z - 0.15), 2]
    if len(z) < 50:
        return lo_z - 0.01
    h, e = np.histogram(z, np.arange(lo_z - 0.15, lo_z + 0.03, 0.005))
    return float(e[np.argmax(np.convolve(h, np.ones(3), 'same'))] + 0.0025)


def session(u):
    f = glob.glob(os.path.join(ROOT, 'FINAL-INPUTS', '*', f'scan_20261030_{u}.npy'))[0]
    a = np.load(f)
    kind = 'box' if 'BOX' in f else 'bottle'
    r = clean(thin(a)[0], origin=(0.0, 0.0, 0.0))
    fr = recover(a)
    F = len(fr)
    Rs = np.full((F, 3, 3), np.nan, np.float32)
    mm = np.zeros((F, 64), np.uint16)
    n_pts = np.array([x[1] for x in fr], np.int32)
    copies = np.array([x[2] for x in fr], np.uint16)
    for i, (s, l, c, R, j) in enumerate(fr):
        if R is not None:
            Rs[i] = R
            mm[i, j] = np.round(((a[s:s + l] - R @ O2) @ R)[:, 2] * 1000).astype(np.uint16)
    meta = dict(session=u, kind=kind, clean=r.meta['status'], width=r.meta.get('width'), height=r.meta.get('height'))
    if r.meta['status'] == 'ok':
        o = r.object.astype(np.float64)
        az0 = float(np.arctan2(o[:, 1].mean(), o[:, 0].mean()))
        hr_med = float(np.median(np.hypot(o[:, 0], o[:, 1])))
        lo_z, hi_z = float(np.percentile(o[:, 2], 1)), float(np.percentile(o[:, 2], 99))
        n = np.array(r.meta['normal'], float)
        yaw = float(np.arctan2(*(np.array([-n[0], n[1]]) * np.sign(n[0] * np.cos(az0) + n[1] * np.sin(az0)))[::-1]))
    else:                                # the cleaner failed on this session: same statistics straight from the near points
        hr = np.hypot(a[:, 0], a[:, 1])
        s = (hr > 0.18) & (hr < 0.5) & (a[:, 2] > -0.05) & (a[:, 2] < 0.3)
        az0 = float(np.median(np.arctan2(a[s, 1], a[s, 0])))
        hr_med = float(np.median(hr[s]))
        lo_z, hi_z = float(np.percentile(a[s, 2], 1)) - 0.03, float(np.percentile(a[s, 2], 99))
        yaw = None
    tz = table_height(a, lo_z)
    meta.update(az_rad=az0, range_m=hr_med, table_z=tz, top_z=hi_z, low_z=lo_z)

    # environment rows = everything except the object sector above the table
    v = np.array([np.cos(az0), np.sin(az0)])
    def is_obj(p):
        hr = np.hypot(p[..., 0], p[..., 1])
        da = np.abs(np.angle(np.exp(1j * (np.arctan2(p[..., 1], p[..., 0]) - az0))))
        return (hr < HR_MAX) & (hr > HR_MIN) & (da < np.radians(AZ_HALF_DEG)) & (p[..., 2] > tz + TABLE_MARGIN) & (p[..., 2] < Z_MAX)

    good = np.isfinite(Rs[:, 0, 0])
    local = D.backproject_batch(mm.astype(np.float64))
    world = np.einsum('fij,fzj->fzi', Rs.astype(np.float64)[good], local[good] + O2)
    out = np.zeros((F, 64), bool)
    out[good] = is_obj(world) & (mm[good] > 0)
    drift = np.full(F, np.nan)
    for i, g in zip(np.nonzero(good)[0], world):
        m = out[i]
        if m.sum() >= 4:
            drift[i] = np.median(g[m, :2] @ v)
    mm[out] = 0
    # frames the fit could not explain are kept as plain rows (object sector removed)
    xrows, xidx = [], np.zeros(F + 1, np.int32)
    for i, (s, l, c, R, j) in enumerate(fr):
        if R is None:
            p = a[s:s + l]
            xrows.append(p[~is_obj(p)])
        xidx[i + 1] = xidx[i] + (len(xrows[-1]) if R is None else 0)
    xrows = np.vstack(xrows).astype(np.float32) if xrows else np.empty((0, 3), np.float32)

    ax = Rs[:, :, 2].astype(np.float64)
    elev = np.degrees(np.arcsin(np.clip(ax[:, 2], -1, 1)))
    low = np.nonzero(np.nan_to_num(elev, nan=90.0) < 70.0)[0]
    t0, t1 = int(low[0]), int(low[-1]) + 1
    phase = np.zeros(F, np.int8)
    phase[t0:t1] = 1
    phase[t1:] = 2
    # depth drift: frame median of the forward coordinate, gaps held, centred on the session
    d = drift[t0:t1]
    ok = np.isfinite(d)
    d = np.interp(np.arange(len(d)), np.nonzero(ok)[0], d[ok])
    d = median_filter(d, 5, mode='nearest')
    d = np.clip(d - np.median(d), -0.08, 0.08)
    meta.update(n_frames=F, hold=[t0, t1], n_rows=int(len(a)), rows_object=int((out * copies[:, None]).sum()),
                frames_recovered=float(good.mean()), yaw_rad=yaw, drift_sd_m=float(d.std()), drift_p2p_m=float(np.percentile(d, 98) - np.percentile(d, 2)),
                elev_deg_hold=[float(x) for x in np.nanpercentile(elev[t0:t1], [5, 50, 95])])

    # direction cells of the room (5 deg): how often a zone looking there returned, and at what range
    hits, trials, rng_ = _cells(Rs[good], mm[good], out[good])
    return dict(R=Rs, mm=mm, n_pts=n_pts, copies=copies, phase=phase, drift=d.astype(np.float32), xrows=xrows, xidx=xidx,
                was_obj=out, hits=hits, trials=trials, rng=rng_), meta


def _cell(d):
    az = np.floor((np.arctan2(d[..., 1], d[..., 0]) + np.pi) / np.radians(5)).astype(int) % 72
    el = np.clip(np.floor((np.arcsin(np.clip(d[..., 2], -1, 1)) + np.pi / 2) / np.radians(5)).astype(int), 0, 35)
    return az * 36 + el


def _cells(R, mm, was_obj):
    d = np.einsum('fij,zj->fzi', R.astype(np.float64), Z)
    c = _cell(d)
    use = ~was_obj                                        # a zone that saw the object says nothing about the room there
    trials = np.bincount(c[use], minlength=2592).astype(np.float32)
    val = use & (mm > 0)
    r = mm[val] / 1000.0 / np.broadcast_to(D.ZONE_PITCH_SIN, mm.shape)[val]
    far = r > 0.45                                        # only what lies beyond the object's sector can fill behind it
    hits = np.bincount(c[val][far], minlength=2592).astype(np.float32)
    s = np.bincount(c[val][far], r[far], minlength=2592)
    return hits, trials, (s / np.maximum(hits, 1)).astype(np.float32)


def main():
    with Pool(11) as p:
        res = p.map(session, SESSIONS)
    out, metas = {}, []
    for u, (d, m) in zip(SESSIONS, res):
        metas.append(m)
        for k, v in d.items():
            out[f'{u}_{k}'] = v
    np.savez_compressed(os.path.join(ROOT, 'configs', 'real_bank.npz'), **out)
    json.dump(dict(sessions=metas), open(os.path.join(ROOT, 'configs', 'real_bank.json'), 'w'), indent=1)
    for m in metas:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})


if __name__ == '__main__':
    main()
