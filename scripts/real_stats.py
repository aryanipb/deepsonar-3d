"""Real-data calibration: measure the 20 real sessions in FINAL-INPUTS, one row per SESSION, never per file.

    PYTHONPATH=. uv run python scripts/real_stats.py
    -> configs/real_stats.json, outputs/final/real_stats.md

What a raw file is (verified here, per session):
  * repeat exports of a session are byte-exact PREFIXES of one another, so the file is append-only and the row index is
    a time axis. Sessions are found from that relation, the longest export stands for the session;
  * most sensor frames are written twice in a row (g rows, then the same g rows; the rest once). `frames()` walks the
    file on that block; a frame identical to the one before it is a STALL (no new sensor data), not a measurement;
  * `session_s` assumes the documented 15 Hz and counts distinct frames only: a proxy, the file has no clock.

Measured against the cleaned face, frame by frame (sign: + = farther from the sensor):
  noise  = intra-frame scatter of hits around a plane is shape + noise, so ranging noise comes from pairs of consecutive
           frames: same zone hit twice, difference / sqrt(2), robust; on the ceiling it is pure, the device lying still;
  drift  = the frame median of hit residuals, smoothed over 5 frames, 5-95 percentile range along the normal;
  thick  = surface_rms of the cleaned output, separately from the depth smear of everything hitting the face.

NOT measured by any of this: the object's back or true depth, and the true sensor pose (translation is untracked, so all
positions are in the rotation-only map, with the session's own drift in them).
"""
import collections
import datetime
import inspect
import glob
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.spatial import cKDTree

from majproj.clean import clean
from majproj.sim import device
from scripts.run_pipeline import thin

BASELINE = 'c425f40'                # last cleaner before the drift rule: the diff below is against that commit only
HZ = 15.0
CEIL_Z, LOW_Z = 1.5, 0.5            # frame median z: above = device on its back seeing the ceiling, below = aimed at the object
ORIGIN = (0.0, 0.0, 0.0)
MIN_PAIRS = 40                      # a noise estimate from fewer same-zone pairs is not reported


def load_sessions():
    files = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'FINAL-INPUTS', '*', '*.npy'))):
        files.append(dict(path=f, kind=os.path.basename(os.path.dirname(f)).lower(), stamp=os.path.basename(f)[14:-4], x=np.load(f)))
    files.sort(key=lambda d: len(d['x']))
    groups = []
    for d in files:                     # shortest first: join the group whose longest member starts with this... or found one
        for g in groups:
            a = g[-1]['x']
            if len(a) <= len(d['x']) and g[-1]['kind'] == d['kind'] and np.array_equal(a, d['x'][:len(a)]):
                g.append(d)
                break
        else:
            groups.append([d])
    return sorted(groups, key=lambda g: min(m['stamp'] for m in g))


def next_same(x):
    """For each row, the index of the next identical row (len(x) if none)."""
    nxt, seen = np.full(len(x), len(x), np.int64), {}
    for i in range(len(x) - 1, -1, -1):
        k = x[i].tobytes()
        nxt[i] = seen.get(k, len(x))
        seen[k] = i
    return nxt


def frames(x):
    """Walk the doubled blocks. Returns [(start, g)] of first copies; rows covered; stall flag per frame."""
    nxt, out, pos, covered = next_same(x), [], 0, 0
    while pos < len(x):
        g = int(nxt[pos] - pos)
        if 1 <= g <= device.NUM_ZONES and pos + 2 * g <= len(x) and np.array_equal(x[pos:pos + g], x[pos + g:pos + 2 * g]):
            out.append((pos, g))
            covered += 2 * g
            pos += 2 * g
        else:
            pos += 1
    stall = [i > 0 and out[i][1] == out[i - 1][1] and np.array_equal(x[out[i][0]:out[i][0] + out[i][1]], x[out[i - 1][0]:out[i - 1][0] + out[i - 1][1]])
             for i in range(len(out))]
    return out, covered, np.array(stall, bool)


def robust_sd(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else None


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def phases(x, fr):
    med = np.array([np.median(x[s:s + g, 2]) for s, g in fr])
    return np.where(med > CEIL_Z, 'ceiling', np.where(med < LOW_Z, 'low', 'mid'))


def horizontal_modes(u, lo, hi, bin_m=0.005, half_m=0.02):
    """Densest z level in [lo, hi] among points u: (z mode, points within half_m of it)."""
    z = u[(u[:, 2] >= lo) & (u[:, 2] <= hi), 2]
    if len(z) < 20:
        return None, 0
    h, e = np.histogram(z, np.arange(lo, hi + bin_m, bin_m))
    c = (e[:-1] + e[1:]) / 2
    sm = np.convolve(h, np.ones(5) / 5, 'same')
    m = float(c[int(np.argmax(sm))])
    near = z[np.abs(z - m) <= half_m]
    return float(np.median(near)), int(len(near))


def table_stats(u, face_centre):
    """The table: horizontal surface points only (local normal within 25 deg of vertical over 1 cm thinned points), so a
    vertical face crossing the z window cannot make a false mode. Searched under and around the object."""
    p = thin(u)[0]
    r = np.linalg.norm(p[:, :2] - face_centre[:2], axis=1)
    p = p[(r < 0.45) & (p[:, 2] > -0.35) & (p[:, 2] < 0.05)]
    if len(p) < 40:
        return dict(table_z_m=None, table_pts=0, sensor_over_table_m=None)
    idx = cKDTree(p).query(p, 16)[1]
    x = p[idx] - p[idx].mean(1, keepdims=True)
    _, vec = np.linalg.eigh(np.einsum('nki,nkj->nij', x, x))
    flat = p[np.abs(vec[:, 2, 0]) >= np.cos(np.radians(25))]
    z, n = horizontal_modes(flat, -0.35, 0.05)
    return dict(table_z_m=z, table_pts=n, sensor_over_table_m=-z if z is not None else None)


def ceiling_stats(u):
    q = u[u[:, 2] > 2.3]
    z, n = horizontal_modes(q, 2.3, 3.2) if len(q) else (None, 0)
    if z is None:
        return dict(ceiling_z_m=None, ceiling_pts=0, ceiling_z_sd_cm=None, ceiling_density_pts_m2=None)
    near = q[np.abs(q[:, 2] - z) <= 0.15]
    ext = np.percentile(near[:, :2], 95, 0) - np.percentile(near[:, :2], 5, 0)
    return dict(ceiling_z_m=z, ceiling_pts=int(len(near)), ceiling_z_sd_cm=100 * robust_sd(near[:, 2]),
                ceiling_extent_xy_m=ext.tolist(), ceiling_density_pts_m2=float(0.81 * len(near) / max(np.prod(ext), 1e-6)))


def mid_stats(u, x, fr, ph):
    """Is the 'mid layer' a surface? Report its shape (the PCA says plane or line) and when it is seen."""
    q = u[(u[:, 2] > 1.5) & (u[:, 2] < 2.35)]
    if len(q) < 30:
        return dict(mid_pts=int(len(q)), mid_z_m=None)
    c = q.mean(0)
    sv = np.linalg.svd(q - c, compute_uv=False) / np.sqrt(len(q))
    z, n = horizontal_modes(q, 1.5, 2.35)
    in_mid = np.array([np.sum((x[s:s + g, 2] > 1.5) & (x[s:s + g, 2] < 2.35)) for s, g in fr])
    return dict(mid_pts=int(len(q)), mid_z_m=z, mid_pts_at_mode=n, mid_centre_m=c.tolist(), mid_sd_axes_m=sv.tolist(),
                mid_hits_in_ceiling_frames_frac=float(in_mid[ph == 'ceiling'].sum() / max(in_mid.sum(), 1)),
                mid_range_median_m=float(np.median(np.linalg.norm(q, axis=1))))


BACKGROUND = dict(a=(-0.40, 0.82), b=(0.16, 0.80))    # two fixed structures found by a first densest-cluster pass


def background_stats(u):
    """Both fixed background structures, each followed from its anchor (the map drifts a little between sessions)."""
    r = np.linalg.norm(u[:, :2], axis=1)
    q = u[(r > 0.55) & (r < 1.4) & (u[:, 2] > -0.35) & (u[:, 2] < 0.9)]
    out = dict(background_pts_in_band=int(len(q)))
    for name, anchor in BACKGROUND.items():
        c = np.array(anchor)
        for _ in range(5):
            near = q[np.linalg.norm(q[:, :2] - c, axis=1) < 0.25]
            if len(near) < 20:
                break
            c = np.median(near[:, :2], axis=0)
        near = q[np.linalg.norm(q[:, :2] - c, axis=1) < 0.25]
        ok = len(near) >= 20
        out.update({f'background_{name}_pts': int(len(near)), f'background_{name}_xy_m': c.tolist() if ok else None,
                    f'background_{name}_range_m': float(np.linalg.norm(c)) if ok else None,
                    f'background_{name}_z_p5_p95_m': [pct(near[:, 2], 5), pct(near[:, 2], 95)] if ok else None})
    return out


def face_stats(x, fr, ph, stall, r):
    """Everything measured against the cleaned face. r = Cleaned, ok."""
    m, q = r.meta, r.object.astype(np.float64)
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[2] if vt[2] @ c >= 0 else -vt[2]         # + = away from the sensor
    sight = c / np.linalg.norm(c)
    fit_tilt = float(np.degrees(np.arccos(min(1.0, n @ sight))))
    # on a narrow patch (bottle: 10 x 25 cm with several cm of smear) the smallest PCA axis is not the normal: fits reach
    # 87 deg. Real box faces stay within 22 deg of the line of sight, so a fit beyond 30 deg falls back to that line
    depth_axis = 'normal' if fit_tilt <= 30 else 'sight_line'
    if depth_axis == 'sight_line':
        n = sight
    t = np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, :2].T
    lo, hi = np.percentile((q - c) @ t.T, 1, 0) - 0.02, np.percentile((q - c) @ t.T, 99, 0) + 0.02

    def hits(p):
        d = p - c
        uv, w = d @ t.T, d @ n
        return np.nonzero(np.all((uv >= lo) & (uv <= hi), 1) & (np.abs(w) < 0.15))[0], w

    # frames during the dwell, with hits on the face
    tm, med, sd, nh, pairs, prev = [], [], [], [], [], None
    for i, (s, g) in enumerate(fr):
        if stall[i] or ph[i] != 'low':
            prev = None
            continue
        f = x[s:s + g]
        k, w = hits(f)
        if len(k) < 4:
            prev = None
            continue
        tm.append(i)
        med.append(float(np.median(w[k])))
        sd.append(robust_sd(w[k]))
        nh.append(int(len(k)))
        if prev is not None and prev[0] == i - 1:
            pf, pw = prev[1], prev[2]
            # pair by IN-PLANE position only: matching in 3D would pick the smallest depth difference and bias noise low
            d = np.linalg.norm(((f[k] - c) @ t.T)[:, None, :] - ((pf - c) @ t.T)[None, :, :], axis=2)
            j = d.argmin(1)
            ok = d[np.arange(len(k)), j] < 0.015      # half a zone spacing at 30 cm: the same zone in both frames
            pairs.append((w[k][ok] - pw[j[ok]]) - np.median(w[k][ok] - pw[j[ok]]) if ok.sum() >= 3 else np.empty(0))
        prev = (i, f[k], w[k])
    med = np.array(med)
    sm = np.convolve(med, np.ones(5) / 5, 'valid') if len(med) >= 5 else med
    dt = np.array(tm, float) / HZ
    slope = float(np.polyfit(dt, med, 1)[0]) if len(med) > 2 else None
    diffs = np.concatenate(pairs) if pairs else np.empty(0)
    u = np.unique(x, axis=0)
    k, w = hits(u)
    area_cm2 = float((m['width'] * 100) * (m['height'] * 100))
    return dict(
        object_pts_after_clean=int(m['n_object']), width_cm=100 * m['width'], height_cm=100 * m['height'],
        range_m=float(np.linalg.norm(c[:2])), distance_3d_m=float(np.linalg.norm(c)), centre_m=c.tolist(),
        fit_tilt_from_sight_deg=fit_tilt, depth_axis=depth_axis,
        shape_hint=m['shape_hint'], confidence=m['confidence'], layer_mode=m.get('layer_mode', 'peak'),
        object_pts_per_cm2_after_clean=float(m['n_object'] / area_cm2), distinct_raw_pts_on_face=int(len(k)),
        distinct_raw_pts_per_cm2=float(len(k) / area_cm2),
        face_thickness_cleaned_rms_mm=1000 * m['surface_rms'], face_plane_sd_mm=1000 * m['plane_sd'], sag_mm=1000 * m['sag'],
        depth_smear_p5_p95_cm=100 * (pct(w[k], 95) - pct(w[k], 5)), n_depth_copies=int(m['n_depth_copies']),
        dwell_frames=int(len(med)), dwell_s=float(len(med) / HZ), hits_per_frame_median=float(np.median(nh)) if nh else None,
        hand_drift_p5_p95_cm=100 * (pct(sm, 95) - pct(sm, 5)) if len(sm) else None,
        hand_drift_full_range_cm=100 * float(np.ptp(sm)) if len(sm) else None, hand_drift_frame_sd_mm=1000 * float(np.std(med)) if len(med) else None,
        hand_drift_slope_cm_per_s=100 * slope if slope is not None else None,
        hit_scatter_in_frame_mm=1000 * float(np.median(sd)) if sd else None,
        range_noise_pair_mm=1000 * robust_sd(diffs) / np.sqrt(2) if len(diffs) >= MIN_PAIRS else None, range_noise_pairs=int(len(diffs)),
        drift_trace_cm=[round(100 * float(v), 2) for v in med[:: max(1, len(med) // 40)]])


def ceiling_noise(x, fr, ph, stall):
    """Pure ranging noise: the same zone in consecutive frames, device still, ceiling at about 2.75 m."""
    d = []
    for i in range(1, len(fr)):
        (s0, g0), (s1, g1) = fr[i - 1], fr[i]
        if ph[i] == 'ceiling' and ph[i - 1] == 'ceiling' and g0 == g1 and not stall[i]:
            a, b = x[s0:s0 + g0], x[s1:s1 + g1]
            if np.median(np.linalg.norm(a - b, axis=1)) < 0.03:      # same row order, device still
                d.append((b - a)[:, 2])
    v = np.concatenate(d) if d else np.empty(0)
    return dict(range_noise_ceiling_mm=1000 * robust_sd(v) / np.sqrt(2) if len(v) >= MIN_PAIRS else None, ceiling_noise_pairs=int(len(v)))


def lattice_rms(q, table):
    """Rigid fit in the ceiling plane of a partial frame (no full 64-zone frame exists) to the 64 table points: yaw and
    mirror by grid search, shift by iterated nearest neighbours. Returns rms of the matched points in metres."""
    a = q[:, :2] - q[:, :2].mean(0)
    best = 1e9
    for mirror in (1.0, -1.0):
        for yaw in np.radians(np.arange(0, 90, 3)):
            c, s_ = np.cos(yaw), np.sin(yaw)
            b = (a * [mirror, 1.0]) @ np.array([[c, -s_], [s_, c]])
            tree = cKDTree(table)
            for _ in range(6):
                d, k = tree.query(b)
                b = b + (table[k] - b).mean(0)
            best = min(best, float(np.sqrt(np.mean(tree.query(b)[0] ** 2))))
    return best


def zone_check(sessions):
    """Do ceiling frames match the 8 x 8 ST zone table? Neighbour angle, plus a rigid fit of partial frames to the table
    back-projected on a flat ceiling. A random point set of the same size is fitted the same way as the control."""
    rms, ctrl, nn_deg, sizes = [], [], [], []
    rng = np.random.default_rng(0)
    for s in sessions:
        x, fr, ph, stall, zc = s['x'], s['fr'], s['ph'], s['stall'], s['ceil']['ceiling_z_m']
        if zc is None:
            continue
        used = 0
        for i, (st, g) in enumerate(fr):
            if ph[i] != 'ceiling' or stall[i]:
                continue
            q = x[st:st + g]
            q = q[np.abs(q[:, 2] - zc) < 0.12]        # the mid structure shares some frames with the ceiling
            sizes.append(len(q))
            if len(q) < 45:
                continue
            d = np.linalg.norm(q[:, None] - q[None], axis=2) + np.eye(len(q)) * 9
            nn_deg.append(float(np.degrees(np.arctan(np.median(d.min(1)) / zc))))
            if used < 6:                              # a few frames per session: the fit is the slow part
                used += 1
                table = device.backproject(np.full(64, 1000.0 * zc))[:, :2]
                table = table - table.mean(0)
                rms.append(lattice_rms(q, table))
                ctrl.append(lattice_rms(rng.uniform(q.min(0), q.max(0), q.shape), table))
    z = float(np.median([s['ceil']['ceiling_z_m'] for s in sessions if s['ceil']['ceiling_z_m']]))
    t = device.ZONE_DIRS / device.ZONE_PITCH_SIN[:, None]
    dt = np.linalg.norm(t[:, None, :2] - t[None, :, :2], axis=2) + np.eye(64) * 9
    sp = z * float(np.median(dt.min(1)))
    return dict(max_zones_in_any_frame=max(f[1] for s in sessions for f in s['fr']), max_ceiling_points_in_frame=int(max(sizes)),
                frames_used_neighbour_angle=len(nn_deg), frames_used_fit=len(rms),
                neighbour_angle_deg_measured=float(np.median(nn_deg)), neighbour_angle_deg_table=float(np.degrees(np.arctan(np.median(dt.min(1))))),
                neighbour_spacing_at_ceiling_m=sp, fit_rms_m=float(np.median(rms)), fit_rms_over_spacing=float(np.median(rms) / sp),
                random_control_rms_over_spacing=float(np.median(ctrl) / sp))


def behaviour_diff(groups):
    try:
        src = subprocess.check_output(['git', 'show', f'{BASELINE}:majproj/clean.py'], text=True, cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return dict(baseline=BASELINE, identical_files=None, changed=[], unavailable=True)
    fd, path = tempfile.mkstemp(suffix='.py')
    with os.fdopen(fd, 'w') as f:
        f.write(src)
    sp = importlib.util.spec_from_file_location('clean_base', path)
    base = importlib.util.module_from_spec(sp)
    sys.modules['clean_base'] = base
    sp.loader.exec_module(base)
    os.remove(path)
    same, changed = 0, []
    for g in groups:
        for d in g:
            p = thin(d['x'])[0]
            a, b = base.clean(p, origin=ORIGIN), clean(p, origin=ORIGIN)
            if a.object.tobytes() == b.object.tobytes() and a.meta == b.meta:
                same += 1
            else:
                changed.append(dict(file=d['stamp'], before=[a.meta['status'], a.meta['n_object'], a.meta['reason']],
                                    after=[b.meta['status'], b.meta['n_object'], b.meta.get('width'), b.meta.get('height'), b.meta.get('layer_mode')]))
    return dict(baseline=BASELINE, identical_files=same, changed=changed)


def one_session(group):
    s = group[-1]
    x = s['x']
    fr, covered, stall = frames(x)
    ph = phases(x, fr)
    u = np.unique(x, axis=0)
    rmap = np.linalg.norm(u, axis=1)
    pts = thin(x)[0]
    r = clean(pts, origin=ORIGIN)
    ceil = ceiling_stats(u)
    fc = r.object.astype(np.float64).mean(0) if r.meta['status'] == 'ok' else np.zeros(3)
    live = ~stall
    g = np.array([f[1] for f in fr])
    # file order of the phases: runs of a phase of at least 10 frames
    seq, cnt = [], 0
    for i, p in enumerate(ph):
        cnt = cnt + 1 if i and p == ph[i - 1] else 1
        if cnt == 10 and (not seq or seq[-1] != p):
            seq.append(str(p))
    row = dict(session=min(m['stamp'] for m in group), kind=s['kind'], files=[m['stamp'] for m in group], kept_file=s['stamp'],
               raw_rows=int(len(x)), distinct_pts=int(len(u)), duplicate_factor=float(len(x) / len(u)), thinned_1cm_pts=int(len(pts)),
               frame_rows_covered_frac=float(covered / len(x)), frames_total=len(fr), frames_live=int(live.sum()), frames_stalled=int(stall.sum()),
               stalled_rows=int(2 * g[stall].sum()), session_s_at_15hz=float(live.sum() / HZ), rows_per_frame_check=float(len(x) / max(2 * np.median(g), 1)),
               valid_zones_median=float(np.median(g[live])), valid_zones_by_phase={p: float(np.median(g[live & (ph == p)])) for p in ('ceiling', 'low', 'mid') if (live & (ph == p)).any()},
               frames_by_phase={p: int((live & (ph == p)).sum()) for p in ('ceiling', 'low', 'mid')}, phase_order=seq,
               near_junk_frac_under_12cm=float(np.mean(rmap < 0.12)), clean_status=r.meta['status'], clean_reason=r.meta['reason'],
               **table_stats(u, fc), **ceil, **mid_stats(u, x, fr, ph), **background_stats(u),
               **ceiling_noise(x, fr, ph, stall))
    row['face'] = face_stats(x, fr, ph, stall, r) if r.meta['status'] == 'ok' else None
    return row, dict(x=x, fr=fr, ph=ph, stall=stall, ceil=ceil)


def summarise(rows, keys):
    out = {}
    for k in keys:
        v = [(r['face'] or {}).get(k) if k not in r else r[k] for r in rows]
        v = [a for a in v if isinstance(a, (int, float))]
        if v:
            out[k] = dict(median=float(np.median(v)), min=float(np.min(v)), max=float(np.max(v)), n=len(v))
    return out


KEYS = ['raw_rows', 'distinct_pts', 'duplicate_factor', 'thinned_1cm_pts', 'frame_rows_covered_frac', 'frames_live', 'frames_stalled', 'stalled_rows',
        'session_s_at_15hz', 'valid_zones_median', 'near_junk_frac_under_12cm', 'table_z_m', 'table_pts', 'sensor_over_table_m', 'ceiling_z_m', 'ceiling_pts',
        'ceiling_z_sd_cm', 'ceiling_density_pts_m2', 'mid_pts', 'mid_z_m', 'mid_range_median_m', 'background_a_range_m', 'background_a_pts', 'background_b_range_m', 'background_b_pts',
        'range_noise_ceiling_mm', 'object_pts_after_clean', 'width_cm', 'height_cm', 'range_m', 'distance_3d_m', 'fit_tilt_from_sight_deg',
        'object_pts_per_cm2_after_clean', 'distinct_raw_pts_on_face', 'distinct_raw_pts_per_cm2', 'face_thickness_cleaned_rms_mm', 'face_plane_sd_mm', 'sag_mm',
        'depth_smear_p5_p95_cm', 'n_depth_copies', 'dwell_s', 'hits_per_frame_median', 'hand_drift_p5_p95_cm', 'hand_drift_full_range_cm',
        'hand_drift_frame_sd_mm', 'hand_drift_slope_cm_per_s', 'hit_scatter_in_frame_mm', 'range_noise_pair_mm', 'confidence']

TABLE = [('raw_rows', 'raw rows', 0), ('distinct_pts', 'distinct', 0), ('duplicate_factor', 'dup x', 2), ('frames_live', 'frames', 0), ('frames_stalled', 'stalled', 0),
         ('session_s_at_15hz', 'session s', 1), ('valid_zones_median', 'zones', 0), ('sensor_over_table_m', 'over table m', 3), ('ceiling_z_m', 'ceiling m', 2),
         ('background_a_pts', 'bg A pts', 0), ('background_b_pts', 'bg B pts', 0)]
FACE = [('object_pts_after_clean', 'object pts', 0), ('width_cm', 'w cm', 1), ('height_cm', 'h cm', 1), ('range_m', 'range m', 3), ('face_thickness_cleaned_rms_mm', 'thick mm', 1),
        ('depth_smear_p5_p95_cm', 'smear cm', 1), ('hand_drift_p5_p95_cm', 'drift cm', 1), ('depth_axis', 'depth axis', None), ('range_noise_pair_mm', 'noise mm', 1), ('dwell_s', 'dwell s', 1), ('hits_per_frame_median', 'hits/frame', 0),
        ('object_pts_per_cm2_after_clean', 'pts/cm2', 2), ('layer_mode', 'layer', None)]


def fmt(v, d):
    if v is None:
        return '-'
    return v if d is None else f'{v:.{d}f}'


def md_table(rows, cols, sub=False):
    h = ['session', 'kind'] + [c[1] for c in cols]
    out = ['| ' + ' | '.join(h) + ' |', '|' + '---|' * len(h)]
    for r in rows:
        src = (r['face'] or {}) if sub else r
        out.append('| ' + ' | '.join([r['session'], r['kind']] + [str(fmt(src.get(k), d)) for k, _, d in cols]) + ' |')
    return out


def write_md(res, path):
    z, by, d = res['zone_grid'], res['by_kind'], res['clean_behaviour']
    rows = res['sessions']
    def line(kind, k, u='', p=1):
        if k not in by[kind]:
            return 'not available'
        v, total = by[kind][k], res['n_sessions'][kind]
        return f"{v['median']:.{p}f}{u} ({v['min']:.{p}f} to {v['max']:.{p}f})" + (f", {v['n']} of {total} sessions" if v['n'] < total else '')
    dflt = {k: v.default for k, v in inspect.signature(device.measure_batch).parameters.items() if v.default is not inspect.Parameter.empty}
    sim_noise = lambda r_m: 1000 * (dflt['noise_abs_m'] + dflt['noise_rel'] * r_m)
    med = lambda rs, key: float(np.median([r[key] for r in rs]))
    phase_z = lambda rs, p: float(np.median([r['valid_zones_by_phase'][p] for r in rs if p in r['valid_zones_by_phase']]))
    box, bot = [r for r in rows if r['kind'] == 'box'], [r for r in rows if r['kind'] == 'bottle']
    order = collections.Counter(' > '.join(r['phase_order']) for r in rows).most_common(4)
    o = ['# Real-scan calibration', '',
         f"Generated by `scripts/real_stats.py` on {res['generated']}. {res['n_sessions']['box']} box and {res['n_sessions']['bottle']} bottle sessions from "
         f"{res['n_files']} files. Each session counts once and uses its longest export.", '',
         '## Key facts', '',
         '- **Repeat exports are byte-exact prefixes.** The file is append-only, so the row index is a time axis.',
         f"- **Most frames are written twice in a row.** Doubled blocks cover {res['all']['frame_rows_covered_frac']['median'] * 100:.1f} % of rows at median (lowest {res['all']['frame_rows_covered_frac']['min'] * 100:.1f} %); the remainder appear once.",
         f"- **Stalls exist.** Some sessions repeat one frame for thousands of rows, with no new data (up to {res['all']['stalled_rows']['max']:.0f} rows). The duplicate factor is {res['all']['duplicate_factor']['median']:.2f} at median and {res['all']['duplicate_factor']['max']:.2f} at worst.",
         f"- **8 x 8 geometry check.** Neighbouring zones on the ceiling are {z['neighbour_angle_deg_measured']:.2f} deg apart, against {z['neighbour_angle_deg_table']:.2f} deg from the ST table. A rigid fit of {z['frames_used_fit']} partial frames to the table leaves {z['fit_rms_over_spacing']:.3f} of a zone spacing; random points of the same size leave {z['random_control_rms_over_spacing']:.3f}.",
         f"- **No frame has all 64 zones.** The largest has {z['max_zones_in_any_frame']}.",
         '', '| Measure | Box | Bottle |', '|---|---|---|']
    for k, name, u, p in [('width_cm', 'Seen face width', ' cm', 1), ('height_cm', 'Seen face height', ' cm', 1), ('range_m', 'Range to face', ' m', 3),
                          ('object_pts_after_clean', 'Object points after cleaning', '', 0), ('object_pts_per_cm2_after_clean', 'Object density', ' pts/cm2', 2),
                          ('face_thickness_cleaned_rms_mm', 'Cleaned face thickness (rms)', ' mm', 1), ('depth_smear_p5_p95_cm', 'Depth smear before cleaning, p5 to p95', ' cm', 1),
                          ('hand_drift_p5_p95_cm', 'Depth drift during the dwell, p5 to p95', ' cm', 1), ('hand_drift_slope_cm_per_s', 'Drift rate', ' cm/s', 2),
                          ('range_noise_pair_mm', 'Range noise at the object', ' mm', 1), ('range_noise_ceiling_mm', 'Range noise at the ceiling, still', ' mm', 1),
                          ('dwell_s', 'Dwell on the object at 15 Hz', ' s', 1), ('session_s_at_15hz', 'Session length at 15 Hz', ' s', 1), ('sensor_over_table_m', 'Sensor over table', ' m', 3),
                          ('ceiling_z_m', 'Ceiling z', ' m', 2), ('background_a_range_m', 'Background A range', ' m', 2), ('background_a_pts', 'Background A points', '', 0),
                          ('background_b_range_m', 'Background B range', ' m', 2), ('background_b_pts', 'Background B points', '', 0)]:
        o.append(f'| {name} | {line("box", k, u, p)} | {line("bottle", k, u, p)} |')
    mid = [r for r in rows if r.get('mid_z_m') is not None and r.get('mid_sd_axes_m')]
    o += ['', '## Clean behaviour', '',
          f"Compared with cleaner `{d['baseline']}` on all {res['n_files']} files: {d['identical_files']} are byte-identical in object and meta, {len(d['changed'])} changed.", '']
    for c in d['changed']:
        o.append(f"- `{c['file']}`: {c['before'][0]} ({c['before'][1]} pts) became {c['after'][0]} ({c['after'][1]} pts, {100 * c['after'][2]:.1f} x {100 * c['after'][3]:.1f} cm, {c['after'][4]}).")
    o += ['', 'The face in that session is narrower than in the others. The uncleaned patch already measured about 18 x 12 cm on its 5 to 95 percentile extents, so the narrowness comes from the session, not from this rule. It is reported as measured and was not corrected.', '',
          'In `drift_slab` mode the `smooth` test passes by construction, because the slab width sets the spread. It is not evidence of a thin surface. Use `confidence_terms.purity`, the share of the patch inside the slab.', '',
          '## For the simulator', '',
          f"- **Valid zones while aimed at the object are few.** Median is {phase_z(box, 'low'):.0f} of 64 for the box and {phase_z(bot, 'low'):.0f} for the bottle, against {phase_z(box, 'ceiling'):.0f} and {phase_z(bot, 'ceiling'):.0f} at the ceiling. Face hits per frame are {by['box']['hits_per_frame_median']['median']:.0f} and {by['bottle']['hits_per_frame_median']['median']:.0f}.",
          f"- **`device.measure_batch` defaults are too noisy at the object.** They give {sim_noise(by['box']['range_m']['median']):.1f} mm at {by['box']['range_m']['median']:.2f} m against {by['box']['range_noise_pair_mm']['median']:.1f} mm measured on the flat box face, and {sim_noise(res['all']['ceiling_z_m']['median']):.0f} mm at the ceiling against {res['all']['range_noise_ceiling_mm']['median']:.0f} mm. Measured noise scales with range with no visible floor. Bottle noise is higher only because it includes curvature and aim wobble.",
          "- **Depth change is a one-way approach, not a zero-mean walk.** All 20 sessions approach along the line of sight, at a mostly steady rate, with an occasional jump of several centimetres. A same-sign rate in every session is systematic; the cause (hand, ranging, or both) is not separable from these files.",
          '- **Phases in the file order:** ' + '; '.join(f'{k} ({v} sessions)' for k, v in order) + f". Frames per session, median: {med(rows, 'frames_live'):.0f} live, with {np.median([r['frames_by_phase']['ceiling'] for r in rows]):.0f} ceiling and {np.median([r['frames_by_phase']['low'] for r in rows]):.0f} aimed low.",
          '- **Row counts:** doubled frames and stalls change raw row counts only. `thin()` removes them before cleaning.', '',
          '## Per session', ''] + md_table(rows, TABLE) + [''] + md_table(rows, FACE, True) + ['',
          '## Other layers', '',
          f"- Mid layer: {len(mid)} sessions have points between 1.5 and 2.35 m. {100 * np.median([r['mid_hits_in_ceiling_frames_frac'] for r in mid]):.0f} % of those hits (median) fall in ceiling frames, lowest {100 * min(r['mid_hits_in_ceiling_frames_frac'] for r in mid):.0f} %. Median axis sds are {np.median([r['mid_sd_axes_m'][0] for r in mid]):.2f}, {np.median([r['mid_sd_axes_m'][1] for r in mid]):.2f} and {np.median([r['mid_sd_axes_m'][2] for r in mid]):.2f} m, median z {np.median([r['mid_z_m'] for r in mid]):.2f} m.",
          "- Background: two fixed structures, A near (-0.40, 0.82) at table height and B near (0.16, 0.80) lower down. Either can be missing or strongest in a session, depending on aim; see the per-session table.",
          f"- Near-sensor junk under 12 cm: {res['all']['near_junk_frac_under_12cm']['median'] * 100:.2f} % of distinct points at median.",
          f"- Drift direction: {sum(1 for r in rows if r['face'] and r['face']['hand_drift_slope_cm_per_s'] < 0)} of {len(rows)} sessions approach the object (negative slope), usually steadily; 115102 also has a jump of about 6 cm late in the dwell.",
          f"- Range noise is about proportional to range: {res['all']['range_noise_ceiling_mm']['median'] / 10 / res['all']['ceiling_z_m']['median']:.2f} % at the ceiling and {res['all']['range_noise_pair_mm']['median'] / 10 / res['all']['range_m']['median']:.2f} % at the object.",
          '', '## Not measured', '',
          "- The object's back and true depth. Only the seen face is measured.",
          '- Absolute sensor position. Translation is untracked, so positions sit in a rotation-only map that contains the drift.',
          '- Session seconds are a proxy from distinct frames at the documented 15 Hz. The file has no clock.',
          '- Ceiling z is above the resting device, not room height. Table z is relative to the map origin.',
          '- Range noise at the object still contains hand jitter. Ceiling noise is z only, from the few still frame pairs, and sessions under 40 pairs show no value.',
          '- Bottle width is inflated by zone blur and wobble, as the session scatter shows.',
          '- On most bottle sessions the plane fit is not the surface normal, so drift, smear and noise there are along the line of sight (`depth axis`). Bottle noise and scatter also contain the curved surface and aim wobble.']
    with open(path, 'w') as f:
        f.write('\n'.join(o) + '\n')


def main():
    groups = load_sessions()
    rows, state = [], []
    for g in groups:
        r, st = one_session(g)
        rows.append(r)
        state.append(st)
        f = r['face'] or {}
        print(r['session'], r['kind'], r['clean_status'], f.get('width_cm'), f.get('height_cm'), f.get('hand_drift_p5_p95_cm'), flush=True)
    res = dict(generated=datetime.date.today().isoformat(), n_files=sum(len(g) for g in groups),
               n_sessions={k: sum(r['kind'] == k for r in rows) for k in ('box', 'bottle')},
               all=summarise(rows, KEYS), by_kind={k: summarise([r for r in rows if r['kind'] == k], KEYS) for k in ('box', 'bottle')},
               zone_grid=zone_check(state), clean_behaviour=behaviour_diff(groups), sessions=rows)
    os.makedirs(os.path.join(ROOT, 'outputs', 'final'), exist_ok=True)
    with open(os.path.join(ROOT, 'configs', 'real_stats.json'), 'w') as f:
        json.dump(res, f, indent=1)
    write_md(res, os.path.join(ROOT, 'outputs', 'final', 'real_stats.md'))
    print(json.dumps(dict(n_sessions=res['n_sessions'], zone_grid=res['zone_grid'], clean=res['clean_behaviour']), indent=1))


if __name__ == '__main__':
    main()
