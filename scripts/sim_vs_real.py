"""Calibration check of simulator v2 against the 10 real exports (statistics only; nothing is trained or tuned).

For the 10 real `inputs/*.npy` and N simulated sessions from `random_session` (default prior: 75 % close
3-40 cm, 25 % far 0.4-1.5 m) compute the same cloud statistics and compare them:
  unique fraction, duplicate groups, dominant axis share, half-extent ratios (x/y, z/y), dense-point
  fraction (8th-NN distance < 0.03), stripe count / wall depth / near-cluster fraction from the stripe
  estimator (analysis.forensics.estimate_from_stripes), plus per-session simulation time.
Writes outputs/forensics/sim_vs_real.{png,md,json}.

usage: python scripts/sim_vs_real.py [n_sims=50] [seed=7]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # honest 1-core timings in the worker pool
    os.environ.setdefault(_v, "1")

import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

from analysis.forensics import estimate_from_stripes
from majproj.sim.scenes import random_session

OUT = 'outputs/forensics/sim_vs_real'


def cloud_stats(p):
    p = np.asarray(p, np.float64)
    u, cnt = np.unique(p, axis=0, return_counts=True)
    ext = (p.max(0) - p.min(0)) / 2
    d8 = cKDTree(p).query(p, k=9)[0][:, -1]
    rec = dict(unique_frac=len(u) / len(p), dup_groups=int((cnt > 1).sum()),
               dominant_axis='xyz'[int(np.abs(p).max(0).argmax())],
               ext_x=ext[0], ext_y=ext[1], ext_z=ext[2],
               x_over_y=ext[0] / max(ext[1], 1e-9), z_over_y=ext[2] / max(ext[1], 1e-9),
               dense_frac=float(np.mean(d8 < 0.03)))
    try:
        est = estimate_from_stripes(p)
    except Exception:
        est = None
    if est is None:
        rec.update(n_stripes=0, wall_depth=np.nan, near_frac=np.nan, near_depth_ratio=np.nan)
    else:
        depth = (p[:, 2] - est['oz']) / est['wall_depth']                # depth as a fraction of the wall distance
        rec.update(n_stripes=int(est['n_stripes']), wall_depth=float(est['wall_depth']),
                   near_frac=float(np.mean(depth < 0.25)), near_depth_ratio=float(np.percentile(depth, 5)))
    return rec


def one_sim(args):
    i, seed = args
    rng = np.random.default_rng(seed * 100003 + i)
    t = time.perf_counter()
    s = random_session(rng)
    dt = time.perf_counter() - t
    rec = cloud_stats(s.export)
    tr = s.truth
    rec.update(sim_seconds=dt, cls=int(s.cls), regime=tr['regime'], distance_m=tr['distance_m'], wall_m=tr['wall_m'],
               true_ratio=tr['distance_m'] / tr['wall_m'], raw_points=int(tr['raw_points']), tries=int(tr['tries']), raw_ok=bool(tr['raw_ok']),
               n_excursions=len(tr['motion'].excursions), duration_s=tr['motion'].duration_s)
    return rec, s.export


ROWS = [('unique_frac', 'unique fraction of 4096', '{:.3f}'), ('dup_groups', 'duplicate groups', '{:.0f}'),
        ('x_over_y', 'half-extent x / y', '{:.2f}'), ('z_over_y', 'half-extent z / y', '{:.2f}'),
        ('dense_frac', 'dense-point fraction (8-NN < 0.03)', '{:.2f}'), ('n_stripes', 'wall stripes found', '{:.0f}'),
        ('wall_depth', 'wall depth D (normalized)', '{:.2f}'), ('near_frac', 'points within 0.25 D of sensor', '{:.2f}'),
        ('near_depth_ratio', '5th pct depth / D', '{:.2f}')]


def q(v, qs):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return np.percentile(v, qs) if len(v) else [np.nan] * len(qs)


def main(n_sims=50, seed=7):
    n_sims, seed = int(n_sims), int(seed)
    files = sorted(glob.glob('inputs/*_shapenet.npy'))
    real = [cloud_stats(np.load(f)) for f in files]
    real_pts = [np.load(f) for f in files]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(one_sim, [(i, seed) for i in range(n_sims)]))
    sims = [o[0] for o in out]
    sim_pts = [o[1] for o in out]
    secs = np.array([s['sim_seconds'] for s in sims])

    lines = ['| statistic | real min / median / max (n=%d) | sim p10 / median / p90 (n=%d) |' % (len(real), len(sims)),
             '|---|---|---|']
    table = {}
    for key, label, fmt in ROWS:
        rv = q([r[key] for r in real], [0, 50, 100])
        sv = q([s[key] for s in sims], [10, 50, 90])
        lines.append('| %s | %s | %s |' % (label, ' / '.join(fmt.format(x) for x in rv),
                                           ' / '.join(fmt.format(x) for x in sv)))
        table[key] = dict(real=[float(x) for x in rv], sim=[float(x) for x in sv])
    dom_r = {a: sum(r['dominant_axis'] == a for r in real) for a in 'xyz'}
    dom_s = {a: sum(s['dominant_axis'] == a for s in sims) for a in 'xyz'}
    lines.append('| dominant axis count x / y / z | %d / %d / %d | %d / %d / %d |' % (*dom_r.values(), *dom_s.values()))
    lines.append('| stripe fit found (>= 4 stripes) | %d / %d | %d / %d |' % (
        sum(r['n_stripes'] >= 4 for r in real), len(real), sum(s['n_stripes'] >= 4 for s in sims), len(sims)))
    lines.append('| sim seconds per session (1 core, mean / p90 / max) | - | %.2f / %.2f / %.2f |' % (
        secs.mean(), np.percentile(secs, 90), secs.max()))
    lines.append('| sim raw map points (median / p10 / p90), redraws used | not observable (>4096) | %d / %d / %d, %.2f draws avg |' % (
        np.median([s['raw_points'] for s in sims]), np.percentile([s['raw_points'] for s in sims], 10),
        np.percentile([s['raw_points'] for s in sims], 90), np.mean([s['tries'] for s in sims])))
    lines.append('| sims whose raw map stayed < 4096 after redraws (exported with replacement) | 0 / %d | %d / %d |' % (
        len(real), sum(not s['raw_ok'] for s in sims), len(sims)))
    md = '\n'.join(lines)
    print(md)
    open(OUT + '.md', 'w').write(md + '\n')
    json.dump(dict(table=table, dominant_real=dom_r, dominant_sim=dom_s, sim_seconds=secs.tolist(),
                   real=real, sims=sims), open(OUT + '.json', 'w'), indent=1, default=float)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 6, figsize=(21, 7.5))
    hist_keys = [('unique_frac', 'unique fraction'), ('dup_groups', 'duplicate groups'), ('x_over_y', 'half-extent x / y'),
                 ('z_over_y', 'half-extent z / y'), ('dense_frac', 'dense-point fraction'),
                 ('near_frac', 'points within 0.25 D of sensor')]
    for a, (k, lab) in zip(ax[0], hist_keys):
        sv = np.array([s[k] for s in sims], float)
        sv = sv[np.isfinite(sv)]
        a.hist(sv, bins=18, color='#6b8fb5', alpha=0.85, label='sim (n=%d)' % len(sims))
        for j, r in enumerate(real):
            if np.isfinite(r[k]):
                a.axvline(r[k], color='#d1550f', lw=1.4, label='real (n=10)' if j == 0 else None)
        a.set_title(lab, fontsize=10)
        a.set_yticks([])
    ax[0, 0].legend(fontsize=8)
    pick = [('real', 0), ('real', 4), ('real', 7), ('sim', 0), ('sim', 1), ('sim', 2)]
    for a, (src, i) in zip(ax[1], pick):
        p = (real_pts if src == 'real' else sim_pts)[i]
        a.scatter(p[:, 2], p[:, 1], s=1.5, c='#d1550f' if src == 'real' else '#6b8fb5', alpha=0.6)
        a.set_aspect('equal'); a.set_xlim(-1.05, 1.05); a.set_ylim(-1.05, 1.05)
        a.set_xlabel('z (depth, normalized)'); a.set_ylabel('y (up)')
        ttl = files[i].split('scan_')[-1][:15] if src == 'real' else 'sim %d: %s, obj %.0f cm, wall %.1f m' % (
            i, sims[i]['regime'], sims[i]['distance_m'] * 100, sims[i]['wall_m'])
        a.set_title('%s  %s' % (src, ttl), fontsize=9)
    fig.suptitle('Simulator v2 vs 10 real exports: cloud statistics (top) and side views z-y (bottom)', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT + '.png', dpi=110)
    print('wrote', OUT + '.png')


if __name__ == '__main__':
    main(*sys.argv[1:])
