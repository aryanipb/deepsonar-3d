"""The 'real' regime of majproj.sim_raw: the measured sessions with a new object rendered into them.

This does not simulate a room. configs/real_bank.npz (scripts/build_real_bank.py) holds the measured frames of the real
sessions with the real object cut out: every rotation, every ceiling / mid layer / table / background / swing / junk return,
the frame order, the repeat counts. A new object is ray cast into those frames at the session's measured place (table height,
range, azimuth, face direction), through the VL53L5CX model, along the session's measured depth drift. So view, distance,
elevation, yaw, dwell, phases, geometry, density and duplicates are the measured ones. Variation comes from the object, a small
placement jitter, the start / end rest periods spliced from another session, a crop of the hold, and mm jitter.

Holdout. Sessions in configs/final.yaml real.holdout are excluded by default, so a training set never holds their room returns,
trajectories or envelope. Pass exclude_sessions=(...) for leave-session-out folds, () for none.

Assumed, not measured: lateral / vertical drift (only depth drift is observable; OU, sd 2-10 mm); object reflectivity and gloss;
what lies behind the old object comes from pooled 5 deg direction cells of the other frames, so it is coarse.
Limits: objects are upright (nothing lying was measured); rest frames do not render the object, so the top of a very tall
object is not seen from the rest pose; everything is one room, one table, one device.
"""
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from . import sim_scan as ss
from .sim import device as D
from .sim.acquisition import IMU_TO_TOF_OFFSET, SENSOR_YAW
from .sim.scene import Revolution, cast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLETE_POINTS = 16384
O2 = SENSOR_YAW.as_matrix().T @ np.asarray(IMU_TO_TOF_OFFSET)
UP = np.array([0.0, 0.0, 1.0])
MARGIN_RANGE_M, MARGIN_AZ_RAD = 0.015, 0.05       # the 'small margin' around the measured envelope
FACE_RANGE_OFFSET_M = 0.012                       # the bank's range is the median over the face; the front is nearer (fitted, proxy tables)
OBJECT_DROPOUT = (0.02, 0.25)                     # per-zone loss on the object (fitted to n_object of the cleaned real objects)
MIN_HOLD_FRAMES = 340                             # about the shortest real hold
_SIN = D.ZONE_PITCH_SIN
_bank, _POOL = None, {}


def bank():
    global _bank
    if _bank is None:
        z = np.load(os.path.join(ROOT, 'configs', 'real_bank.npz'))
        meta = json.load(open(os.path.join(ROOT, 'configs', 'real_bank.json')))['sessions']
        data = {m['session']: {k: z[f"{m['session']}_{k}"] for k in
                               ('R', 'mm', 'copies', 'phase', 'drift', 'xrows', 'xidx', 'was_obj', 'hits', 'trials', 'rng')} for m in meta}
        for m in meta:
            data[m['session']]['R'] = data[m['session']]['R'].astype(np.float64)
        _bank = dict(data=data, meta={m['session']: m for m in meta})
    return _bank


def holdout_sessions():
    """real.holdout of configs/final.yaml, read at call time (that file belongs to the training config)."""
    try:
        txt = open(os.path.join(ROOT, 'configs', 'final.yaml')).read()
    except OSError:
        return ()
    m = re.search(r'^\s*holdout:\s*\[(.*?)\]', txt, re.M)
    return tuple(re.findall(r'(\d{6})\b', m.group(1))) if m else ()


# ------------------------------------------------------------------------------------------------ objects
@dataclass
class Pipe(Revolution):
    """Open tube: seen from the side like a cylinder, but its surface has an inner wall and annular ends."""
    wall: float = 0.003

    def surface(self, n, rng):
        ro, ri, y0, y1 = float(self.rs[0]), float(self.rs[0]) - self.wall, float(self.ys[0]), float(self.ys[-1])
        h = y1 - y0
        ar = np.array([2 * np.pi * ro * h, 2 * np.pi * ri * h, 2 * np.pi * (ro ** 2 - ri ** 2)])
        k = rng.multinomial(n, ar / ar.sum())
        a = rng.uniform(0, 2 * np.pi, n)
        r = np.concatenate([np.full(k[0], ro), np.full(k[1], ri), np.sqrt(rng.uniform(ri ** 2, ro ** 2, k[2]))])
        y = np.concatenate([rng.uniform(y0, y1, k[0] + k[1]), rng.choice([y0, y1], k[2])])
        return np.stack([self.cx + r * np.cos(a), y, self.cz + r * np.sin(a)], 1)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _shoulder(r, rn, n=6):
    u = np.linspace(0, 1, n)
    return u, rn + (r - rn) * 0.5 * (1 + np.cos(np.pi * u))


def sample_shape(rng, kind):
    """dict(type='box', a=width, b=depth, h) or dict(type='rev', ys, rs, h, r[, wall]), metres, base at 0."""
    if kind == 'cuboid':
        return dict(type='box', a=_logu(rng, .03, .6), b=_logu(rng, .03, .6), h=_logu(rng, .03, .6))
    if kind == 'cube':
        s = _logu(rng, .03, .6)
        return dict(type='box', a=s, b=s, h=s)
    if kind == 'slab':
        t, x, y = _logu(rng, .015, .05), _logu(rng, .12, .6), _logu(rng, .12, .6)
        w = int(rng.choice(3, p=[.25, .60, .15]))                     # thin along width (edge on), depth (facing), height (flat)
        return dict(type='box', a=[t, x, x][w], b=[x, t, y][w], h=[y, y, t][w])
    if kind == 'tower':
        return dict(type='box', a=_logu(rng, .03, .15), b=_logu(rng, .03, .15), h=float(rng.uniform(.3, .6)))
    if kind in ('can', 'rod', 'pipe'):
        r, h = {'can': (rng.uniform(.025, .06), rng.uniform(.08, .2)), 'rod': (rng.uniform(.01, .025), rng.uniform(.2, .6)),
                'pipe': (rng.uniform(.015, .08), rng.uniform(.15, .6))}[kind]
        r, h = float(r), float(h)
        d = dict(type='rev', ys=[0.0, h], rs=[r, r], h=h, r=r)
        if kind == 'pipe':
            d['wall'] = float(rng.uniform(.002, min(.01, r / 2)))
        return d
    if kind == 'jar':
        r, h = float(rng.uniform(.035, .08)), float(rng.uniform(.07, .2))
        yb, rn = rng.uniform(.75, .9) * h, r * rng.uniform(.75, .95)
        ys_ = yb + rng.uniform(.3, .7) * (h - yb)
        u, s = _shoulder(r, rn, 4)
        return dict(type='rev', ys=[0.0, *(yb + u * (ys_ - yb)), h], rs=[r, *s, rn], h=h, r=r)
    if kind not in ('bottle', 'wavy_bottle'):
        raise ValueError(f'unknown kind {kind!r}')
    r, h = float(rng.uniform(.02, .06)), float(rng.uniform(.12, .40))
    yb = float(rng.uniform(.45, .72)) * h
    ysh = min(yb + float(rng.uniform(.1, .2)) * h, .93 * h)
    rn = float(r * rng.uniform(.28, .5))
    u, s = _shoulder(r, rn, 6)
    if kind == 'wavy_bottle':
        y = np.linspace(0, yb, 15)
        amp, k, ph = rng.uniform(.04, .14), rng.integers(2, 7), rng.uniform(0, 2 * np.pi)
        env = np.clip(np.minimum(y, yb - y) / (0.08 * h), 0, 1)
        body = r * (1 - amp * env * (0.5 + 0.5 * np.sin(2 * np.pi * k * y / yb + ph)))
    else:
        y, body = np.array([0.0, yb]), np.array([r, r])
    return dict(type='rev', ys=np.concatenate([y, yb + u[1:] * (ysh - yb), [h]]).tolist(), rs=np.concatenate([body, s[1:], [rn]]).tolist(), h=h, r=r)


def _depth_half(sh, theta):
    if sh['type'] == 'box':
        return 0.5 * (sh['b'] * abs(np.cos(theta)) + sh['a'] * abs(np.sin(theta)))
    return sh['r']


def build_prim(sh, centre_xy, base_z, normal_az, refl):
    """Scene primitive; a box has its width along normal_az + 90 deg and its depth along normal_az."""
    if sh['type'] == 'box':
        return ss._box_prim([centre_xy[0], centre_xy[1], base_z + sh['h'] / 2], (sh['a'] / 2, sh['h'] / 2, sh['b'] / 2), normal_az + np.pi / 2, refl)
    p = ss._revolution(centre_xy[0], centre_xy[1], base_z, sh['ys'], sh['rs'], refl)
    return Pipe(p.cx, p.cz, p.ys, p.rs, p.reflectivity, wall=sh['wall']) if 'wall' in sh else p


def size_of(sh):
    return (sh['a'], sh['b'], sh['h']) if sh['type'] == 'box' else (2 * sh['r'], 2 * sh['r'], sh['h'])


# ------------------------------------------------------------------------------------------------ frames -> rows
def _rows(R, mm, copies, xrows=None, xidx=None, labels=None):
    """Frames -> rows in file order. xidx[f]:xidx[f + 1] are the plain rows of a frame without a rotation."""
    p = np.einsum('fij,fzj->fzi', np.nan_to_num(R), D.backproject_batch(mm.astype(np.float64)) + O2)
    pts, lab = [], []
    good = np.isfinite(R[:, 0, 0])
    for f in range(len(R)):
        if good[f]:
            v = mm[f] > 0
            q, l = p[f][v], (labels[f][v] if labels is not None else np.zeros(int(v.sum()), bool))
        elif xrows is not None:
            q = xrows[xidx[f]:xidx[f + 1]].astype(np.float64)
            l = np.zeros(len(q), bool)
        else:
            continue
        if len(q):
            c = int(copies[f])
            pts.append(np.tile(q, (c, 1)))
            lab.append(np.tile(l, c))
    if not pts:
        return np.empty((0, 3)), np.empty(0, bool)
    return np.vstack(pts), np.concatenate(lab)


def _jitter(mm, rng):
    out = mm.astype(np.int32)
    return np.where(out > 0, np.maximum(out + rng.integers(-3, 4, out.shape), D.MIN_RANGE_MM), 0)


def _env_pool(allowed):
    key = tuple(allowed)
    if key not in _POOL:
        B = bank()['data']
        h = sum(B[s]['hits'] for s in allowed)
        t = sum(B[s]['trials'] for s in allowed)
        r = sum(B[s]['rng'] * B[s]['hits'] for s in allowed) / np.maximum(h, 1)
        _POOL[key] = (h / np.maximum(t, 1), r)
    return _POOL[key]


def _cell(d):
    az = np.floor((np.arctan2(d[..., 1], d[..., 0]) + np.pi) / np.radians(5)).astype(int) % 72
    el = np.clip(np.floor((np.arcsin(np.clip(d[..., 2], -1, 1)) + np.pi / 2) / np.radians(5)).astype(int), 0, 35)
    return az * 36 + el


def envelope(allowed):
    m = [bank()['meta'][s] for s in allowed]
    rr, az = [x['range_m'] for x in m], [x['az_rad'] for x in m]
    return (min(rr) - MARGIN_RANGE_M, max(rr) + MARGIN_RANGE_M), (min(az) - MARGIN_AZ_RAD, max(az) + MARGIN_AZ_RAD)


def _face_azimuths(allowed):
    """Measured outward direction of the box faces (world azimuth, away from the sensor)."""
    m = [bank()['meta'][s] for s in allowed]
    return [np.pi - x['yaw_rad'] for x in m if x['kind'] == 'box' and x['yaw_rad'] is not None]


# ------------------------------------------------------------------------------------------------ public
def simulate_real(rng, kind, exclude_sessions=None, session=None, n_side=3, shape=None, distance_m=None, az_rad=None,
                  face_az=None, refl=None, gloss=None, render_object=True):
    B = bank()
    excl = holdout_sessions() if exclude_sessions is None else tuple(exclude_sessions)
    allowed = sorted(s for s in B['data'] if s not in excl)
    if not allowed:
        raise ValueError('every session is excluded')
    s = str(rng.choice(allowed)) if session is None else str(session)
    d, m = B['data'][s], B['meta'][s]
    t0, t1 = m['hold']

    # hold: a contiguous crop of the measured one
    n_hold = t1 - t0
    keep = int(min(n_hold, max(MIN_HOLD_FRAMES, rng.uniform(0.75, 1.0) * n_hold)))
    h0 = t0 + int(rng.integers(0, n_hold - keep + 1))
    idx = np.arange(h0, h0 + keep)
    R = d['R'][idx]
    Rn = np.nan_to_num(R)
    mm = _jitter(d['mm'][idx], rng)
    was = d['was_obj'][idx]
    good = np.isfinite(R[:, 0, 0])
    Z_w = np.einsum('fij,zj->fzi', Rn, D.ZONE_DIRS)

    # zones that showed the old object: what lies behind it, from the pooled room cells of the allowed sessions
    pv, pr = _env_pool(allowed)
    c = _cell(Z_w)
    fill = was & (rng.random(was.shape) < pv[c])
    mm = np.where(fill, np.clip(pr[c] * _SIN[None] * 1000 * (1 + rng.normal(0, .01, was.shape)), D.MIN_RANGE_MM, D.MAX_RANGE_MM).astype(np.int32), mm)

    # placement inside the measured envelope
    (r_lo, r_hi), (a_lo, a_hi) = envelope(allowed)
    D_m = float(np.clip(m['range_m'] - FACE_RANGE_OFFSET_M + rng.normal(0, .012), r_lo, r_hi)) if distance_m is None else float(distance_m)
    az = float(np.clip(m['az_rad'] + rng.normal(0, .03), a_lo, a_hi)) if az_rad is None else float(az_rad)
    n_az = float(rng.choice(_face_azimuths(allowed)) + rng.normal(0, .03)) if face_az is None else float(face_az)
    v = np.array([np.cos(az), np.sin(az), 0.0])
    l = np.array([-np.sin(az), np.cos(az), 0.0])
    sh = sample_shape(rng, kind) if shape is None else shape
    theta = n_az - az
    centre = v[:2] * (D_m + _depth_half(sh, theta))
    refl_ = float(rng.uniform(.25, .75)) if refl is None else float(refl)
    gloss_ = float(rng.uniform(1.0, 2.0) if sh['type'] == 'box' else rng.uniform(1.0, 4.0)) if gloss is None else float(gloss)
    prim = build_prim(sh, centre, m['table_z'], n_az, refl_)

    # true sensor position: measured depth drift + assumed lateral / vertical drift
    dep = d['drift'][idx - t0].astype(np.float64)
    lat_sd, vert_sd = float(rng.uniform(.002, .01)), float(rng.uniform(.002, .01))
    drift = -dep[:, None] * v + ss._ou(rng, keep, 1 / 15., 2.0, lat_sd, 1) * l + ss._ou(rng, keep, 1 / 15., 2.0, vert_sd, 1) * UP

    labels = np.zeros((keep, 64), bool)
    if render_object:
        sub = D.subray_dirs(n_side)
        n2 = n_side * n_side
        org = drift + np.einsum('fij,j->fi', Rn, O2)
        to = np.array([centre[0], centre[1], m['table_z'] + sh['h'] / 2]) - org
        dist = np.linalg.norm(to, axis=1)
        rho = 0.5 * float(np.linalg.norm(size_of(sh)))
        cone = np.arcsin(np.minimum(rho / np.maximum(dist, rho + 1e-6), 1.0)) + np.radians(6.0)
        sel = (np.einsum('fzi,fi->fz', Z_w, to / dist[:, None]) > np.cos(np.minimum(cone, np.pi / 2))[:, None]) & good[:, None]
        f, z = np.nonzero(sel)
        if len(f):
            dw = np.einsum('mij,mkj->mki', R[f], sub[z])
            ds = (dw @ ss.W2S.T).reshape(-1, 3)
            t, nrm, _, which = cast([prim], np.repeat(org[f] @ ss.W2S.T, n2, axis=0), ds)
            cosi = np.abs((nrm * ds).sum(1)).reshape(-1, n2)
            hit, t = (which >= 0).reshape(-1, n2), t.reshape(-1, n2)
            a = hit.any(1)
            f, z, t, cosi, hit = f[a], z[a], t[a], cosi[a], hit[a]
            vign = sub[z][..., 2] ** 4
            bg = mm[f, z] > 0
            rb = np.where(bg, mm[f, z] / 1000.0 / _SIN[z], 99.0)[:, None]
            wb = np.where(bg, 0.4, 0.0)[:, None] * vign / np.maximum(rb, .05) ** 2
            r_ = np.where(hit, t, rb)
            w_ = np.where(hit, refl_ * cosi ** gloss_ * vign / np.maximum(t, .05) ** 2, wb)
            rr, ww = np.full((keep, 64, n2), 99.0), np.zeros((keep, 64, n2))
            rr[f, z], ww[f, z] = r_, w_
            dp, st = D.measure_batch(rr, ww, rng, noise_abs_m=float(rng.uniform(.002, .004)), noise_rel=float(rng.uniform(.002, .006)),
                                     dropout=float(rng.uniform(*OBJECT_DROPOUT)), signal_threshold=float(rng.uniform(.03, .06)))
            rep = dp[f, z] / 1000.0 / _SIN[z]
            dif = np.where(ww[f, z] > 0, np.abs(r_ - rep[:, None]), np.inf)
            j = dif.argmin(1)
            labels[f, z] = hit[np.arange(len(j)), j] & (dif.min(1) <= .03) & (st[f, z] == 5)
            mm[f, z] = dp[f, z].astype(np.int32)

    def rest(sess, a, b):
        e = B['data'][sess]
        if b <= a:
            return np.empty((0, 3))
        return _rows(e['R'][a:b], _jitter(e['mm'][a:b], rng), e['copies'][a:b], e['xrows'], e['xidx'][a:b + 1])[0]

    p_hold, l_hold = _rows(R, mm, d['copies'][idx], d['xrows'], d['xidx'][h0:h0 + keep + 1], labels)
    # rest periods: this session's, or spliced from another one; the part of the hold outside the crop stays in the file
    s0 = str(rng.choice(allowed)) if rng.random() < .5 else s
    s1 = str(rng.choice(allowed)) if rng.random() < .5 else s
    start, pre = rest(s0, 0, B['meta'][s0]['hold'][0]), rest(s, t0, h0)
    post, end = rest(s, h0 + keep, t1), rest(s1, B['meta'][s1]['hold'][1], len(B['data'][s1]['R']))
    raw = np.vstack([start, pre, p_hold, post, end])
    mask = np.concatenate([np.zeros(len(start) + len(pre), bool), l_hold, np.zeros(len(post) + len(end), bool)])

    seen = labels.any(1)
    shift = drift[seen].mean(0) if seen.any() else np.zeros(3)
    complete = (prim.surface(COMPLETE_POINTS, rng) @ ss.W2S - shift).astype(np.float32)
    truth = dict(regime='real', kind=kind, size_m=tuple(float(x) for x in size_of(sh)), distance_m=D_m, session=s, session_start=s0, session_end=s1,
                 az_rad=az, face_az_rad=n_az, yaw_deg=float(np.degrees(theta)), table_z_m=float(m['table_z']), hold_frames=keep, hold_s=keep / 15.0,
                 reflectivity=refl_, gloss=gloss_, drift_sd_m=[float(dep.std()), lat_sd, vert_sd], lying=False, n_raw=int(len(raw)),
                 n_object_rows=int(mask.sum()), object_frames=int(seen.sum()), excluded=excl, nominal_pose_m=shift.tolist())
    return dict(raw=raw, complete=complete, object_mask=mask, truth=truth)
