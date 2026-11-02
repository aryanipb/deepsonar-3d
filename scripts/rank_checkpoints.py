"""Score checkpoints on the real scans: distance to the registered targets, and how steady the size is across sessions.

    uv run python -m scripts.rank_checkpoints outputs/final/checkpoint_ranking.json name=path ...
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from majproj.infer import load_checkpoint  # noqa: E402
from majproj.real_targets import targets_for_pool  # noqa: E402
from majproj.train_final import cd_per_scan  # noqa: E402
from scripts.gen_final import N_TARGET, real_pool  # noqa: E402

HELD = ['scan_20261030_121600', 'scan_20261030_123815']


def main(out, *pairs):
    pool, _ = real_pool(os.path.join(ROOT, 'FINAL-INPUTS'), HELD)
    gts, infos = targets_for_pool(pool, {'BOTTLE': 'configs/real_bottle_parts.json', 'BOX': 'configs/real_box_parts.json'}, N_TARGET, {'BOX': 0.234})
    res = {}
    for pair in pairs:
        name, path = pair.split('=', 1)
        model, _ = load_checkpoint(os.path.expanduser(path))
        rows = []
        with torch.no_grad():
            for s in pool:
                pred = model(torch.from_numpy(s['partial'][None]))[1].float()
                e = (np.percentile(pred[0].numpy(), 99, 0) - np.percentile(pred[0].numpy(), 1, 0)) * s['T']['scale'] * 100
                g = gts.get(s['session'])
                cd = None if g is None else cd_per_scan(pred, torch.from_numpy(np.asarray(g, np.float32)[None]))[0].item() * s['T']['scale'] * 1000
                rows.append(dict(session=s['session'], object=s['object'], held=s['held_out'], cd_mm=cd, w=float(e[0]), h=float(e[1]), d=float(e[2])))
        r = dict(rows=rows)
        for ob in ('BOTTLE', 'BOX'):
            a = [x for x in rows if x['object'] == ob]
            cds = [x['cd_mm'] for x in a if x['cd_mm'] is not None and not x['held']]
            held = [x['cd_mm'] for x in a if x['held']][0]
            r[ob] = dict(held_cd_mm=round(held, 2), train_cd_mm=round(float(np.median(cds)), 2), width_cv=round(float(np.std([x['w'] for x in a]) / np.mean([x['w'] for x in a])), 3),
                         w=round(float(np.median([x['w'] for x in a])), 1), d=round(float(np.median([x['d'] for x in a])), 1), h=round(float(np.median([x['h'] for x in a])), 1))
        res[name] = r
        print(name, 'BOTTLE', r['BOTTLE'], 'BOX', r['BOX'], flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
