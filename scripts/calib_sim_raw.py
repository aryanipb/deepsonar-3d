"""Calibration numbers of majproj.sim_raw from the real raw files: recovers every frame (majproj.raw_frames) of the 20
distinct sessions and writes configs/sim_raw_calib.json.

    uv run python -m scripts.calib_sim_raw
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.raw_frames import recover

# longest export of each session (the shorter exports are subsets)
SESSIONS = dict(box='114126 115102 115330 115840 120051 120408 120745 121055 121223 121457 121600'.split(),
                bottle='122622 122735 122907 123028 123146 123259 123523 123658 123815'.split())
OUT = os.path.join(ROOT, 'configs', 'sim_raw_calib.json')


def path(stem, kind):
    return os.path.join(ROOT, 'FINAL-INPUTS', kind.upper(), f'scan_20261030_{stem}.npy')


def one(job):
    kind, stem = job
    a = np.load(path(stem, kind))
    r = recover(a)
    R, mm, L, C = r['R'], r['mm'], r['length'], r['copies']
    ok = np.isfinite(R[:, 0, 0])
    el = np.full(len(R), np.nan)
    el[ok] = np.degrees(np.arcsin(np.clip(R[ok, 2, 2], -1, 1)))
    az = np.degrees(np.arctan2(R[:, 1, 2], R[:, 0, 2]))
    rest, hold = el > 75, el < 50
    d = dict(kind=kind, stem=stem, n_rows=len(a), n_unique=len(np.unique(a, axis=0)), frames=len(R), solved=float(ok.mean()),
             copies=np.bincount(np.minimum(C, 4), minlength=5).tolist(), rest_frames=int(rest.sum()), hold_frames=int(hold.sum()),
             first_rest=int(np.argmax(~rest & ok)), last_rest=int(np.argmax((~rest & ok)[::-1])),
             valid_rest=float(np.median((mm[rest] > 0).sum(1))) if rest.any() else None, valid_hold=float(np.median((mm[hold] > 0).sum(1))),
             elev_pct=np.percentile(el[hold], [5, 50, 95]).tolist(), az_sd=float(np.std(np.unwrap(np.radians(az[hold]))) * 57.3))
    h = np.nonzero(hold)[0]
    near = np.array([np.median(m[m > 0]) if (m > 0).any() else np.nan for m in mm[h]])
    d['near_mm_pct'] = np.nanpercentile(near, [5, 50, 95]).tolist()
    res = {}
    for name, idx, lo, hi in (('far', np.nonzero(rest)[0], 1500, 4000), ('near', h, 100, 700)):
        ii = idx[np.nonzero(np.diff(idx) == 1)[0]]
        x, y = mm[ii].astype(float), mm[ii + 1].astype(float)
        q = (x >= lo) & (y >= lo) & (x <= hi) & (y <= hi)
        res[name] = np.percentile(np.abs(x - y)[q], [50, 90]).tolist() if q.any() else None
    d['abs_diff_mm'] = res
    d['pv_rest'] = (mm[rest] > 0).mean(0).tolist() if rest.any() else None
    i = np.nonzero(ok[:-1] & ok[1:] & hold[:-1] & hold[1:])[0]
    rr = np.einsum('nij,nik->njk', R[i], R[i + 1])
    d['hold_deg_s_pct'] = np.percentile(np.degrees(np.arccos(np.clip((np.trace(rr, axis1=1, axis2=2) - 1) / 2, -1, 1))) * 15, [50, 90]).tolist()
    return d


if __name__ == '__main__':
    jobs = [(k, s) for k, v in SESSIONS.items() for s in v]
    with Pool(11) as p:
        rows = p.map(one, jobs)
    agg = dict(sessions=rows)
    agg['copies_p'] = (np.sum([r['copies'] for r in rows], 0) / np.sum([r['copies'] for r in rows])).round(4).tolist()
    agg['pv_rest_mean'] = np.mean([r['pv_rest'] for r in rows if r['pv_rest']], 0).round(3).tolist()
    with open(OUT, 'w') as f:
        json.dump(agg, f, indent=0)
    for r in rows:
        print(r['stem'], r['kind'], 'rows', r['n_rows'], 'fr', r['frames'], 'first', r['first_rest'], 'last', r['last_rest'], 'hold', r['hold_frames'], 'vh', r['valid_hold'], 'vr', r['valid_rest'],
              np.round(r['elev_pct'], 1), 'azsd %.1f' % r['az_sd'], 'near', np.round(r['near_mm_pct']), r['abs_diff_mm'], np.round(r['hold_deg_s_pct'], 1))
    print(agg['copies_p'])
    print(np.round(np.array(agg['pv_rest_mean']).reshape(8, 8), 2).tolist())
