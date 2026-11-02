"""Held-out simulated evaluation of the FULL single-file pipeline (preprocess v2 -> AdaPoinTr -> from_canonical).

Sessions are generated with the device simulator (majproj.sim) at seeds >= 900000, i.e. disjoint from any
training stream, with the object distance drawn log-uniformly inside each distance bin. For every session the
exported 4096 x 3 file goes through exactly the deployment pipeline and the completed object (in the file's
coordinates) is compared with the true complete object (same frame). Truth is used for scoring only.

Metrics (all computed in the export's normalized frame; sizes relative to the true object):
  object size   longest side of the true object's bounding box
  CD-L1         (mean NN distance pred->gt + mean NN distance gt->pred) / 2, reported as a % of object size
                (and in cm using the simulator's true scale, which a real file does not have)
  F@1%, F@2%    F-score with threshold tau = 1 % / 2 % of the object size
  observed-only the same metrics for the segmented observed points WITHOUT completion (what the model adds)
  seg ok        fraction of segmented points within 5 % of object size of the true surface (segmentation quality)

Usage:  PYTHONPATH=. uv run python scripts/evaluate_sim.py [--n-per-bin 24] [--ckpt PATH] [--workers 4]
Appends/replaces the section 'sim' of outputs/evaluation_v3.md and writes outputs/eval_sim_v3.json.
"""
import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORT = os.path.join(ROOT, 'outputs', 'evaluation_v3.md')
BINS_CM = [(3, 10), (10, 20), (20, 40), (40, 150)]
CLASSES = ['bottle', 'box']
SEED0 = 900000
SECTION_ORDER = ['header', 'summary', 'real', 'probes', 'sim', 'baseline']


# ------------------------------------------------------------------------------------ report helpers (shared)
def update_report(name, text, path=REPORT):
    """Replace (or add) one named section of the markdown report; other sections are kept as they are."""
    secs = {}
    if os.path.exists(path):
        cur = open(path).read()
        for n in SECTION_ORDER:
            a, b = f'<!-- section:{n} -->', f'<!-- /section:{n} -->'
            if a in cur and b in cur:
                secs[n] = cur.split(a, 1)[1].split(b, 1)[0].strip('\n')
    secs[name] = text.strip('\n')
    body = '\n\n'.join(f'<!-- section:{n} -->\n{secs[n]}\n<!-- /section:{n} -->' for n in SECTION_ORDER if n in secs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').write(body + '\n')


def header_text(comp):
    lines = ['# Evaluation of the single-file pipeline (preprocess v2 + AdaPoinTr)', '']
    if not comp.frame_ok:
        lines += [f'> **SMOKE TEST, NOT A RESULT.** The loaded checkpoint `{comp.ckpt_name}` was trained on the OLD (v1) '
                  'per-object normalization, not on preprocess-v2 canonical partials. Every number below only shows '
                  'that the scripts run; the completed shapes are not meaningful. Re-run both evaluation scripts '
                  'after `checkpoints/adapointr_v3.pth` exists.', '']
    lines += [f'Checkpoint: `{comp.ckpt_name}`' + (f" (steps {comp.info['steps']})" if 'steps' in comp.info else ''),
              '', 'The 10 real scans are a **test set only**: nothing here was trained, fused or tuned on them. '
              'Their labels are used only to group results. The simulated sessions use seeds >= 900000.', '',
              'Canonical units are object-distance units, never centimetres: the export divides by the largest '
              'coordinate, so no metric size exists in a file.']
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------ metrics (shared)
def nn_dist(a, b):
    return cKDTree(b).query(a)[0]


def cd_fscore(pred, gt, size, pcts=(1, 2)):
    """CD-L1 and F-score@tau (tau = pct % of `size`). pred, gt: (N, 3)."""
    d1, d2 = nn_dist(pred, gt), nn_dist(gt, pred)
    out = {'cd': float((d1.mean() + d2.mean()) / 2)}
    for pc in pcts:
        tau = pc / 100 * size
        p, r = float((d1 <= tau).mean()), float((d2 <= tau).mean())
        out[f'f{pc}'] = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return out


def longest_side(p, lo=0.0, hi=100.0):
    a, b = np.percentile(p, [lo, hi], axis=0)
    return float((b - a).max())


def robust_longest_side(p):
    """Longest side of the 1-99 percentile box. The model's 16384 output points include a thin diffuse cloud of stray
    points whose full bounding box is 2-3x the extent of the bulk, so the full box is NOT a size estimate."""
    return longest_side(p, 1.0, 99.0)


def one_sided_fit(observed, completed):
    """Mean NN distance observed -> completed, as a fraction of the completed object's longest side (robust 1-99
    percentile box, see robust_longest_side). 0 = the prediction passes exactly through what the sensor saw.
    A necessary check, not a sufficient one."""
    return float(nn_dist(observed, completed).mean() / robust_longest_side(completed))


def angular_extent_deg(pts, o):
    """Same definition as majproj.preprocess angular size: max of the 1-99 percentile azimuth / elevation ranges."""
    v = pts - o
    z = np.maximum(v[:, 2], 1e-6)
    az, el = np.degrees(np.arctan2(v[:, 0], z)), np.degrees(np.arctan2(v[:, 1], z))
    return float(max(np.ptp(np.percentile(az, [1, 99])), np.ptp(np.percentile(el, [1, 99]))))


# ------------------------------------------------------------------------------------ session generation
def _gen(job):
    b, i, lo, hi, n_side = job
    from majproj.sim.scenes import random_session
    seed = SEED0 + b * 10000 + i
    d = float(np.exp(np.random.default_rng([seed, 7]).uniform(np.log(lo / 100), np.log(hi / 100))))
    s = random_session(np.random.default_rng(seed), n_side=n_side, distance=d)
    t = s.truth
    keep = {k: t[k] for k in ('distance_m', 'wall_m', 'size_m', 'scale', 'sensor_norm', 'raw_points', 'raw_ok',
                              'regime', 'kind', 'support') if k in t}
    keep['trans_amp_max_m'] = float(max(t['motion'].trans_amp_m))
    return dict(bin=b, seed=seed, export=s.export, complete=s.complete, cls=int(s.cls), truth=keep)


def make_sessions(n_per_bin, n_side, workers, cache):
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    jobs = [(b, i, lo, hi, n_side) for b, (lo, hi) in enumerate(BINS_CM) for i in range(n_per_bin)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        sessions = list(ex.map(_gen, jobs))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pickle.dump(sessions, open(cache, 'wb'))
    return sessions


# ------------------------------------------------------------------------------------ one session
def eval_session(comp, s):
    res = comp(s.pop('export'))
    gt, t = s['complete'], s['truth']
    size = longest_side(gt)
    full = cd_fscore(res.completed, gt, size)
    obs = cd_fscore(res.observed, gt, size)
    dseg = nn_dist(res.observed, gt)
    d_pg = nn_dist(res.completed, gt)                      # completed point -> true surface
    unit_m = t['scale']                                   # metres per export unit (simulator truth)
    row = dict(bin=s['bin'], seed=s['seed'], cls=s['cls'], kind=t.get('kind'), dist_cm=100 * t['distance_m'],
               size_cm=100 * size * unit_m, mode=res.meta['mode'], confident=res.meta['confident'],
               n_obj=res.meta['n_object_points'], cd_pct=100 * full['cd'] / size, cd_cm=100 * full['cd'] * unit_m,
               f1=full['f1'], f2=full['f2'], obs_cd_pct=100 * obs['cd'] / size, obs_f1=obs['f1'], obs_f2=obs['f2'],
               seg5=float((dseg <= 0.05 * size).mean()),
               seg_hand=float((dseg <= (0.02 + t['trans_amp_max_m']) / unit_m).mean()),
               fit1=one_sided_fit(res.observed, res.completed),
               size_ratio=robust_longest_side(res.completed) / size,       # 1-99 percentile box of the prediction
               size_ratio_full=longest_side(res.completed) / size,         # full box: inflated by stray points
               far_frac=float((d_pg > 0.25 * size).mean()),                # share of predicted points > 25 % size off
               scale_ratio=res.scale / (t['distance_m'] / unit_m),
               depth_ratio_true=t['distance_m'] / t['wall_m'], depth_ratio_pred=res.meta['object_depth_ratio'],
               ang_true=angular_extent_deg(gt, np.asarray(t['sensor_norm'])), ang_pred=res.meta['angular_size_deg'],
               ang_canon_pred=res.meta['angular_size_canonical_deg'])
    return row


# ------------------------------------------------------------------------------------ tables
def _stat(x, f=np.mean):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return f(x) if len(x) else float('nan')


def _med_iqr(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return 'n/a'
    q = np.percentile(x, [25, 50, 75])
    return f'{q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]'


def _table(rows, groups, label):
    head = (f'| {label} | n | CD-L1 % size mean / median | CD-L1 cm (median) | F@1% | F@2% | '
            'observed-only CD-L1 % | observed-only F@2% | seg ok (5 % size) | pred/true longest side, 1-99 pct box (median) |')
    out = [head, '|' + '---|' * 10]
    for name, sel in groups:
        r = [x for x in rows if sel(x)]
        if not r:
            out.append(f'| {name} | 0 |' + ' |' * 8)
            continue
        g = lambda k: [x[k] for x in r]
        out.append(f'| {name} | {len(r)} | {_stat(g("cd_pct")):.1f} / {_stat(g("cd_pct"), np.median):.1f} | '
                   f'{_stat(g("cd_cm"), np.median):.1f} | {_stat(g("f1")):.2f} | {_stat(g("f2")):.2f} | '
                   f'{_stat(g("obs_cd_pct")):.1f} | {_stat(g("obs_f2")):.2f} | {_stat(g("seg5"), np.median):.2f} | '
                   f'{_stat(g("size_ratio"), np.median):.2f} |')
    return '\n'.join(out)


def sim_section(rows, comp, args, seconds):
    bin_name = lambda b: f'{BINS_CM[b][0]}-{BINS_CM[b][1]} cm'
    by_bin = [(bin_name(b), (lambda x, b=b: x['bin'] == b)) for b in range(len(BINS_CM))]
    by_cls = [(c, (lambda x, k=k: x['cls'] == k)) for k, c in enumerate(CLASSES)]
    both = [(f'{bin_name(b)} / {c}', (lambda x, b=b, k=k: x['bin'] == b and x['cls'] == k))
            for b in range(len(BINS_CM)) for k, c in enumerate(CLASSES)]
    L = ['## Held-out simulated sessions (end to end, known truth)', '']
    if not comp.frame_ok:
        L += [f'> **SMOKE TEST, NOT A RESULT**: `{comp.ckpt_name}` was trained on the old v1 normalization. '
              'Re-run after `adapointr_v3.pth` is trained.', '']
    L += [f'Checkpoint `{comp.ckpt_name}`; {len(rows)} sessions ({args.n_per_bin} per distance bin), seeds '
          f'{SEED0}..; simulator `n_side={args.n_side}`; evaluated in {seconds:.0f} s.', '',
          'Distance = distance from the sensor to the nearest point of the object (simulator truth). Each session is '
          'one hand-held scan exported like the real device and processed with the deployment pipeline: '
          '`preprocess` -> AdaPoinTr -> `from_canonical`. Sizes and distances are in the export frame, divided by '
          'the true object size (longest side of the true bounding box). `CD-L1 cm` uses the simulator\'s true '
          'metric scale, which a real file does not have.', '',
          '### Per distance bin', '', _table(rows, by_bin, 'Distance'), '',
          '### Per class', '', _table(rows, by_cls, 'Class'), '',
          '### Per distance bin and class', '', _table(rows, both, 'Distance / class'), '']
    # preprocess quality per bin: segmentation, scale, distance ratio, angular size
    L += ['### Preprocess quality per distance bin (explains the numbers above)', '',
          '| Distance | n | seg ok 5 % size (median) | seg ok 2 cm + hand amplitude (median) | segmented pts (median) | '
          'canonical scale / true front-face distance (median [IQR]) | depth ratio pred / true (median [IQR]) | '
          'angular size error, deg (median abs) | canonical-patch angle error, deg (median abs) | '
          'one-sided fit % (median) | mode stripes | flagged confident |', '|' + '---|' * 12]
    for b in range(len(BINS_CM)):
        r = [x for x in rows if x['bin'] == b]
        g = lambda k: [x[k] for x in r]
        dr = [x['depth_ratio_pred'] / x['depth_ratio_true'] for x in r if x['depth_ratio_pred']]
        ae = [abs(x['ang_pred'] - x['ang_true']) for x in r if x['ang_pred'] is not None]
        ac = [abs(x['ang_canon_pred'] - x['ang_true']) for x in r if x['ang_canon_pred'] is not None]
        L.append(f'| {bin_name(b)} | {len(r)} | {_stat(g("seg5"), np.median):.2f} | {_stat(g("seg_hand"), np.median):.2f} | '
                 f'{_stat(g("n_obj"), np.median):.0f} | {_med_iqr(g("scale_ratio"))} | {_med_iqr(dr)} | '
                 f'{_stat(ae, np.median):.0f} | {_stat(ac, np.median):.0f} | {100 * _stat(g("fit1"), np.median):.1f} | '
                 f'{sum(x["mode"] == "stripes" for x in r)}/{len(r)} | {sum(x["confident"] for x in r)}/{len(r)} |')
    n_conf = sum(x['confident'] for x in rows)
    bad5 = sum(x['confident'] and x['seg5'] < 0.5 for x in rows)
    bad_hand = sum(x['confident'] and x['seg_hand'] < 0.5 for x in rows)
    L += ['', f'Flagged `confident`: {n_conf}/{len(rows)} sessions. Of those, {bad5} have fewer than half of their '
              f'segmented points within 5 % of object size of the true surface (strict: hand motion smears close '
              f'objects by more than that), and {bad_hand} have fewer than half within 2 cm plus the hand-translation '
              'amplitude (lenient). A True flag therefore does not guarantee a good segmentation: it is not a '
              'reliability score.', '',
          '### How to read this', '',
          '- These are **simulated** sessions: they share the simulator\'s assumptions (optics, noise, hand motion, '
          'object families bottle / jar / can / box). Good numbers here do not prove the model works on the real '
          'device; the real scans have no ground truth for the hidden side.',
          '- CD-L1 and F-score are measured against the TRUE object in the file\'s frame, so they include '
          'preprocess errors (segmentation, scale), not only model errors. "observed-only" is the same metric for '
          'the segmented points without any completion: the completion only helps if it beats that column.',
          '- The F-score thresholds (1 % / 2 % of the object\'s longest side) are strict; a prediction with the '
          'right shape but a wrong scale scores near zero. The canonical scale (object distance) is only known roughly: '
          'the preprocess validation measured a median error factor of about 1.3-1.6 and several-fold errors in the '
          'tails at 3-10 cm (see the scale ratio column), which caps what end-to-end scores can reach at close range.',
          '- Per-class cells with a small n are noisy; compare bins, not single cells.',
          '- The model output contains a thin diffuse cloud of stray points besides the dense sheet on the object. '
          'Share of predicted points farther than 25 % of the object size from the true surface (median per bin, '
          + ', '.join(f'{bin_name(b)}: {100 * _stat([x["far_frac"] for x in rows if x["bin"] == b], np.median):.1f} %'
                      for b in range(len(BINS_CM))) +
          '); the full bounding box of the prediction relative to the true object is '
          + ', '.join(f'{_stat([x["size_ratio_full"] for x in rows if x["bin"] == b], np.median):.1f}x'
                      for b in range(len(BINS_CM))) +
          ' (median per bin) against the 1-99 percentile box in the table, which is why sizes are reported on the '
          'percentile box. CD-L1 is a mean and is barely moved by the strays; F-score precision is.']
    return '\n'.join(L)


# ------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-bin', type=int, default=24)
    ap.add_argument('--n-side', type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--cache', default=None, help='pickle with the generated sessions (reused when it exists)')
    ap.add_argument('--no-report', action='store_true', help='do not touch outputs/evaluation_v3.md')
    ap.add_argument('--from-json', action='store_true',
                    help='rebuild the report section from --out-json without re-running anything')
    ap.add_argument('--out-json', default=os.path.join(ROOT, 'outputs', 'eval_sim_v3.json'))
    args = ap.parse_args(argv)

    if args.from_json:
        from types import SimpleNamespace
        d = json.load(open(args.out_json))
        comp = SimpleNamespace(frame_ok=d['frame_ok'], ckpt_name=d['checkpoint'], info=d.get('info', {}))
        args.n_per_bin, args.n_side = d['n_per_bin'], d['n_side']
        text = sim_section(d['rows'], comp, args, d['seconds'])
        if not args.no_report:
            update_report('header', header_text(comp))
            update_report('sim', text)
        return d['rows']

    t0 = time.time()
    sessions = make_sessions(args.n_per_bin, args.n_side, args.workers, args.cache)   # before torch (fork safety)
    print(f'{len(sessions)} sessions generated in {time.time() - t0:.0f} s', flush=True)

    from majproj.infer import Completer
    comp = Completer(args.ckpt)
    print('checkpoint:', comp.ckpt, '' if comp.frame_ok else '(SMOKE TEST: v1 frame)', flush=True)
    t1, rows = time.time(), []
    for k, s in enumerate(sessions):
        rows.append(eval_session(comp, dict(s)))
        if (k + 1) % 12 == 0:
            print(f'  {k + 1}/{len(sessions)}', flush=True)
    seconds = time.time() - t1
    text = sim_section(rows, comp, args, seconds)
    json.dump(dict(checkpoint=comp.ckpt_name, frame_ok=comp.frame_ok, info={k: v for k, v in comp.info.items()
                                                                           if k == 'steps'},
                   n_per_bin=args.n_per_bin, n_side=args.n_side, seconds=seconds, rows=rows), open(args.out_json, 'w'),
              default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if not args.no_report:
        update_report('header', header_text(comp))
        update_report('sim', text)
    print(text)
    return rows


if __name__ == '__main__':
    main()
