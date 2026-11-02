"""Validate the stripe-ruler estimator (estimate_from_stripes) and object_geometry on simulated exports
with known truth, in two regimes: default random_session (object 0.15-1.5 m) and a close-object regime
(0.05-0.2 m, like the real scans). Writes outputs/forensics/stripes_validation.{jsonl,json}.

usage: python scripts/validate_stripes.py [n_per_regime=48] [seed=100]
       python scripts/validate_stripes.py real       # run on the 10 real files -> outputs/forensics/forensics.json + stripes.png
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from analysis.forensics import estimate_from_stripes, object_geometry
from majproj.sim.scenes import random_session

OUT = 'outputs/forensics/stripes_validation'
FOV_CLIP_DEG = 23.5


def _angles(pts, o):
    d = pts[:, 2] - o[2]
    ok = d > 0.01
    az = np.degrees(np.arctan2(pts[ok, 0] - o[0], d[ok]))
    el = np.degrees(np.arctan2(pts[ok, 1] - o[1], d[ok]))
    return az, el


def one(args):
    regime, i, seed = args
    rng = np.random.default_rng(seed * 1000 + i)
    dist = float(rng.uniform(0.05, 0.2)) if regime == 'close' else None
    s = random_session(rng, distance=dist)
    tr = s.truth
    o_true = np.asarray(tr['sensor_norm'], float)
    D_true = float(tr['wall_m'] / tr['scale'])          # sensor is at metric z ~ 0, so wall depth = wall_m
    rec = dict(regime=regime, case=i, cls=int(s.cls), distance_m=float(tr['distance_m']), wall_m=float(tr['wall_m']),
               raw_points=int(tr['raw_points']), true_ox=float(o_true[0]), true_oy=float(o_true[1]),
               true_oz=float(o_true[2]), true_D=D_true, true_rel=float(tr['distance_m'] / tr['wall_m']))
    # truth silhouette of the whole object as seen from the true sensor position, clipped to the field of view
    az, el = _angles(s.complete.astype(np.float64), o_true)
    azc, elc = az[np.abs(az) <= FOV_CLIP_DEG], el[np.abs(el) <= FOV_CLIP_DEG]
    rec['true_width_deg'] = float(azc.max() - azc.min()) if len(azc) else None
    rec['true_height_deg'] = float(elc.max() - elc.min()) if len(elc) else None
    for mode in ('mid', 'top'):
        e = estimate_from_stripes(s.export, y_mode=mode)
        if e is None:
            rec[f'ok_{mode}'] = False
            continue
        rec[f'ok_{mode}'] = True
        rec[f'oy_{mode}'] = e['oy']
        if mode == 'mid':
            rec.update(est_ox=e['ox'], est_oy=e['oy'], est_oz=e['oz'], est_D=e['wall_depth'], n_stripes=e['n_stripes'],
                       rel_rms=e['rel_rms'], roll_deg=e['roll_deg'], n_alt=e['n_alternatives'], confident=e['confident'],
                       n_found_peaks=e['n_found_peaks'], columns=e['columns'])
            # (a) isolates the ruler: object front taken from ground truth, depth measured from the estimated sensor
            obj_front = float(s.complete[:, 2].min())
            rec['est_rel_truthfront'] = float((obj_front - e['oz']) / e['wall_depth'])
            g = object_geometry(s.export, e)
            rec['geom_found'] = bool(g and g.get('found'))
            if rec['geom_found']:
                rec.update(est_rel_cluster=g['front_rel'], est_width_deg=g['width_deg_fov_clipped'], est_height_deg=g['height_deg_fov_clipped'],
                           angles_reliable=g['angles_reliable'],
                           cluster_n=g['n_cluster'], cluster_depth_extent_rel=g['depth_extent_rel'])
    return rec


def summarize(rows):
    def stats(v):
        v = np.abs(np.asarray(v, float))
        return None if len(v) == 0 else dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                                              max=float(v.max()))
    out = {}
    for regime in ('default', 'close'):
        R = [r for r in rows if r['regime'] == regime]
        F = [r for r in R if r.get('ok_mid')]
        C = [r for r in F if r['confident']]
        d = {'n_cases': len(R), 'n_fit': len(F), 'n_confident': len(C)}
        for name, S in (('all_fits', F), ('confident_only', C)):
            if not S:
                continue
            d[name] = dict(
                oz_abs_err=stats([r['est_oz'] - r['true_oz'] for r in S]),
                D_rel_err=stats([(r['est_D'] - r['true_D']) / r['true_D'] for r in S]),
                ox_abs_err=stats([r['est_ox'] - r['true_ox'] for r in S]),
                oy_mid_abs_err=stats([r['oy_mid'] - r['true_oy'] for r in S]),
                oy_top_abs_err=stats([r['oy_top'] - r['true_oy'] for r in S if 'oy_top' in r]),
                rel_truthfront_abs_err=stats([r['est_rel_truthfront'] - r['true_rel'] for r in S]),
                rel_cluster_abs_err=stats([r['est_rel_cluster'] - r['true_rel'] for r in S if 'est_rel_cluster' in r]),
                width_deg_abs_err=stats([r['est_width_deg'] - r['true_width_deg'] for r in S
                                         if 'est_width_deg' in r and r['true_width_deg'] is not None]),
                height_deg_abs_err=stats([r['est_height_deg'] - r['true_height_deg'] for r in S
                                          if 'est_height_deg' in r and r['true_height_deg'] is not None]))
        d['fail_cases_oz_err_gt_0.15'] = [dict(case=r['case'], oz_err=r['est_oz'] - r['true_oz'], rel_rms=r['rel_rms'],
                                                n_stripes=r['n_stripes'], confident=r['confident'], raw_points=r['raw_points'],
                                                distance_m=r['distance_m'], wall_m=r['wall_m'])
                                           for r in F if abs(r['est_oz'] - r['true_oz']) > 0.15]
        out[regime] = d
    return out


def main(n=48, seed=100):
    n, seed = int(n), int(seed)
    jobs = [(rg, i, seed) for rg in ('default', 'close') for i in range(n)]
    t = time.time()
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(one, jobs))
    print(f'{len(rows)} sims in {time.time() - t:.0f}s', flush=True)
    with open(OUT + '.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    summ = summarize(rows)
    json.dump(summ, open(OUT + '.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


def main_real():
    """Run the estimators on the 10 real scans (TEST data: nothing is tuned or trained on them)."""
    import glob
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from analysis.forensics import COLUMN_U, TAN_STRIPE, ZONE_PITCH_DEG, export_stats

    files = sorted(glob.glob('inputs/scan_*_shapenet.npy'))
    res, data = {}, {}
    for f in files:
        name = os.path.basename(f)[:-4]
        p = np.load(f)
        e = estimate_from_stripes(p)
        g = object_geometry(p, e) if e else None
        res[name] = dict(export_stats=export_stats(p), stripe_fit=e, object_geometry=g,
                         confident=bool(e and e['confident']))
        data[name] = (p, e, g)
    try:
        val = json.load(open(OUT + '.json'))
    except OSError:
        val = None
    json.dump(dict(note='sensor-relative geometry from the stripe ruler; absolute metres are NOT recoverable from one file. '
                        'Real files are TEST data: no threshold or model here was tuned on them (confidence thresholds were chosen on 64 '
                        'simulated exports, seed 200; held-out check on 96 sims, seed 100, is in `simulation_validation`).',
                   caveats=['o_y is not observable from the stripes; the wall-extent midpoint estimate has ~0.25 normalised-unit error in simulation',
                            'object front depth (<~2% of wall distance in most real files) is below the oz error (median ~0.02, p90 ~0.06 normalised, i.e. 2-5% of D): '
                            'the object is effectively touching the sensor, its depth cannot be resolved further',
                            'angular width/height are ill-conditioned at that depth; median error ~9 deg on confident sims, so treat as coarse',
                            'confident = rel_rms <= 0.008, >= 5 stripes, no equally good alternative column assignment'],
                   zone_pitch_deg=ZONE_PITCH_DEG, simulation_validation=val, files=res),
              open('outputs/forensics/forensics.json', 'w'), indent=1)

    fig, axs = plt.subplots(5, 4, figsize=(16, 20))
    for k, (name, (p, e, g)) in enumerate(data.items()):
        a_xy, a_xz = axs[k // 2, (k % 2) * 2], axs[k // 2, (k % 2) * 2 + 1]
        a_xy.scatter(p[:, 0], p[:, 1], s=1.2, c=p[:, 2], cmap='viridis')
        a_xz.scatter(p[:, 0], p[:, 2], s=1.2, c='0.55')
        a_xy.set_title(name[5:20] + ' front view (x,y)', fontsize=9)
        a_xz.set_title('top view (x,z)', fontsize=9)
        for a in (a_xy, a_xz):
            a.set_aspect('equal'); a.set_xlim(-1.05, 1.05)
        if e:
            c, s_ = np.cos(np.radians(e['roll_deg'])), np.sin(np.radians(e['roll_deg']))
            for j in range(8):
                xr = (e['ox'] * c + e['oy'] * s_) + e['wall_depth'] * TAN_STRIPE[j]      # stripe position in the de-rolled frame
                ys = np.array([-1, 1.0])
                # de-rolled stripe: x' = c x + s y = xr  ->  x = (xr - s y)/c
                col = 'tab:red' if j in e['columns'] else '0.6'
                a_xy.plot((xr - s_ * ys) / c, ys, color=col, lw=0.6, alpha=0.8)
            a_xz.scatter([e['ox']], [e['oz']], marker='^', s=90, c='tab:red', zorder=5)
            a_xz.axhline(e['wall_z'], color='tab:red', lw=0.6, ls='--')
            for sg in (-1, 1):
                a_xz.plot([e['ox'], e['ox'] + sg * 0.43 * e['wall_depth']], [e['oz'], e['wall_z']], color='tab:red', lw=0.5)
            a_xz.set_title(f"top view (x,z)  D={e['wall_depth']:.2f} oz={e['oz']:.2f} stripes={e['n_stripes']} "
                           f"rel_rms={e['rel_rms']:.3f} conf={e['confident']}", fontsize=7)
            if g and g.get('found'):
                a_xz.text(0.02, 0.02, f"obj front {g['front_rel']*100:.0f}% of wall dist; {g['width_deg']:.0f} deg = "
                          f"{g['width_zones']:.1f} zones wide, {g['height_zones']:.1f} high", transform=a_xz.transAxes, fontsize=7)
    fig.suptitle('Stripe ruler on the 10 real scans: predicted zone-column stripes (red = columns used in the fit, grey = unused) '
                 'and sensor position (red triangle)', fontsize=11)
    fig.tight_layout()
    fig.savefig('outputs/forensics/stripes.png', dpi=80)
    for name, r in res.items():
        e, g = r['stripe_fit'], r['object_geometry']
        print(name[5:20], e and dict(oz=round(e['oz'], 3), D=round(e['wall_depth'], 3), n=e['n_stripes'], rel_rms=round(e['rel_rms'], 4),
              roll=e['roll_deg'], conf=e['confident'], alt=e['n_alternatives']),
              g and {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items() if k in
                     ('found', 'n_cluster', 'front_rel', 'median_rel', 'back_rel', 'width_deg', 'height_deg', 'width_zones', 'height_zones')})


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'real':
        main_real()
    else:
        main(*sys.argv[1:])
