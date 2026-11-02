"""Contrast table: fine-tuned v3 vs the official PCN-pretrained AdaPoinTr on the SAME held-out simulated sessions.

Both rows files come from scripts/evaluate_sim.py run on the same session cache (so preprocess, segmentation, scale and
truth are identical and any difference is the model alone):

  PYTHONPATH=. uv run python -m scripts.evaluate_sim --n-per-bin 40 --cache CACHE.pkl
  PYTHONPATH=. uv run python -m scripts.evaluate_sim --n-per-bin 40 --cache CACHE.pkl \
        --ckpt checkpoints/AdaPoinTr_PCN.pth --no-report --out-json outputs/eval_sim_v3_pretrained.json
  PYTHONPATH=. uv run python -m scripts.evaluate_baseline

The pretrained checkpoint is NOT a smoke test here: it is the untuned baseline, fed the same preprocess-v2 canonical
partials as the fine-tuned model. A small extra table runs both models on the 10 real scans (test set, never trained on)
and reports only the one-sided fit, which is a necessary check, not a measure of accuracy.

Replaces the section 'baseline' of outputs/evaluation_v3.md.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from scripts.evaluate_sim import BINS_CM, CLASSES, one_sided_fit, robust_longest_side, update_report


def _m(x, f=np.mean):
    return float(f(np.asarray(x, float)))


def paired_table(a, b, label, groups):
    """a = fine-tuned rows, b = pretrained rows (same order). One line per group."""
    out = [f'| {label} | n | CD-L1 % size, fine-tuned mean / median | CD-L1 % size, pretrained mean / median | '
           'CD-L1 % size, observed points only (no completion) mean / median | F@2% fine-tuned / pretrained / observed-only | '
           'F@1% fine-tuned / pretrained | fine-tuned better than pretrained | fine-tuned better than observed-only |',
           '|' + '---|' * 9]
    for name, sel in groups:
        idx = [i for i in range(len(a)) if sel(a[i])]
        if not idx:
            continue
        A, B = [a[i] for i in idx], [b[i] for i in idx]
        g = lambda R, k: [r[k] for r in R]
        win_p = np.mean([x['cd_pct'] < y['cd_pct'] for x, y in zip(A, B)])
        win_o = np.mean([x['cd_pct'] < x['obs_cd_pct'] for x in A])
        out.append(f'| {name} | {len(idx)} | {_m(g(A, "cd_pct")):.1f} / {_m(g(A, "cd_pct"), np.median):.1f} | '
                   f'{_m(g(B, "cd_pct")):.1f} / {_m(g(B, "cd_pct"), np.median):.1f} | '
                   f'{_m(g(A, "obs_cd_pct")):.1f} / {_m(g(A, "obs_cd_pct"), np.median):.1f} | '
                   f'{_m(g(A, "f2")):.2f} / {_m(g(B, "f2")):.2f} / {_m(g(A, "obs_f2")):.2f} | '
                   f'{_m(g(A, "f1")):.2f} / {_m(g(B, "f1")):.2f} | {100 * win_p:.0f} % | {100 * win_o:.0f} % |')
    return '\n'.join(out)


def real_contrast():
    """One-sided fit of both models on the 10 real scans (necessary check only; no ground truth exists)."""
    from majproj.infer import CKPT_PRETRAINED, CKPT_V3, Completer
    files = sorted(glob.glob(os.path.join(ROOT, 'inputs', 'scan_*_shapenet.npy')))
    c3, cp = Completer(str(CKPT_V3)), Completer(str(CKPT_PRETRAINED))
    rows = []
    for f in files:
        scan = np.load(f)
        r3, rp = c3(scan), cp(scan)
        wall = r3.meta['sensor']['wall_depth'] if r3.meta['sensor'] else None
        rows.append((os.path.basename(f)[14:20], one_sided_fit(r3.observed, r3.completed),
                     one_sided_fit(rp.observed, rp.completed),
                     robust_longest_side(r3.completed_canonical), robust_longest_side(rp.completed_canonical),
                     None if not wall else robust_longest_side(r3.completed_canonical) * r3.scale / wall,
                     None if not wall else robust_longest_side(rp.completed_canonical) * rp.scale / wall))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ft', default=os.path.join(ROOT, 'outputs', 'eval_sim_v3.json'))
    ap.add_argument('--pre', default=os.path.join(ROOT, 'outputs', 'eval_sim_v3_pretrained.json'))
    ap.add_argument('--no-real', action='store_true')
    args = ap.parse_args(argv)
    dft, dpre = json.load(open(args.ft)), json.load(open(args.pre))
    a, b = dft['rows'], dpre['rows']
    assert [r['seed'] for r in a] == [r['seed'] for r in b], 'the two runs must use the same sessions'
    bn = lambda i: f'{BINS_CM[i][0]}-{BINS_CM[i][1]} cm'
    by_bin = [(bn(i), (lambda x, i=i: x['bin'] == i)) for i in range(len(BINS_CM))]
    by_cls = [(c, (lambda x, k=k: x['cls'] == k)) for k, c in enumerate(CLASSES)]
    L = ['## Contrast: fine-tuned v3 vs the official pretrained weights (same sessions, same preprocessing)', '',
         f'Fine-tuned: `{dft["checkpoint"]}`. Baseline: `{dpre["checkpoint"]}` (the official AdaPoinTr PCN weights, '
         'no fine-tuning). Both models get exactly the same preprocess-v2 canonical partial for each of the '
         f'{len(a)} held-out simulated sessions ({dft["n_per_bin"]} per distance bin), so segmentation, distance '
         'estimate and truth are identical and the columns differ only by the model. The pretrained model is '
         'expected to do badly: it was trained on a different input scale and frame (PCN complete-the-shape '
         'benchmark, partial = a fraction of a centred, unit-sized shape). Lower CD-L1 is better; "observed points '
         'only" is the segmented partial returned unchanged, i.e. the do-nothing reference a completion model has '
         'to beat.', '',
         '### Per distance bin', '', paired_table(a, b, 'Distance', by_bin), '',
         '### Per class', '', paired_table(a, b, 'Class', by_cls), '',
         '### All sessions', '', paired_table(a, b, 'All', [('all', lambda x: True)]), '']
    if not args.no_real:
        rr = real_contrast()
        L += ['### Same two models on the 10 real scans (one-sided fit only)', '',
              'Mean distance from the observed object points to the completed surface, as a % of the completed '
              'object\'s longest side (0 = passes through what the sensor saw). A necessary check only: a blob that '
              'engulfs the patch also scores low, and real scans have no ground truth for the hidden side.', '',
              '| Scan | one-sided fit, fine-tuned (%) | one-sided fit, pretrained (%) | completed longest side, fine-tuned / pretrained (canonical units) | '
              'same, x wall distance |', '|---|---|---|---|---|']
        for s, f3, fp, l3, lp, w3, wp in rr:
            w = 'n/a' if w3 is None else f'{w3:.2f} / {wp:.2f}'
            L.append(f'| {s} | {100 * f3:.1f} | {100 * fp:.1f} | {l3:.2f} / {lp:.2f} | {w} |')
        L += ['', f'Mean one-sided fit: fine-tuned {100 * np.mean([r[1] for r in rr]):.1f} %, pretrained '
                  f'{100 * np.mean([r[2] for r in rr]):.1f} %.']
    update_report('baseline', '\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
