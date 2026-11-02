"""Look at a final-model checkpoint: simulated val scans (input, target, output) and the real sessions (input, output).

  uv run python -m scripts.final_visuals ckpt.pt val_shard.npz out_prefix [--per-kind 2] [--holdout-only]

Writes <out_prefix>_val.png, <out_prefix>_real.png and <out_prefix>.json. Red = input, grey = target, blue = output.
The JSON has, per kind, the Chamfer of the scans drawn, and the back coverage: for the target points behind the mean depth
of the target, the median distance to the nearest output point, in mm. Real rows are only what can be read off the output:
the back of a real object was never measured, and `leave_session_out` is true only for the sessions in real.holdout,
which neither the simulator's room bank nor any real stage has seen.
"""
import argparse
import json

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from majproj.model import load_adapointr
from majproj.train_final import cd_per_scan
from scripts.gen_final import GT_STEP, KINDS, load_config, real_pool


def load(path):
    model = load_adapointr('checkpoints/AdaPoinTr_PCN.pth', 'cpu')
    st = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(st.get('ema', st))
    return model.eval(), {k: st.get(k) for k in ('step', 'hashes')}


@torch.no_grad()
def predict(model, x):
    return model(torch.from_numpy(np.asarray(x, np.float32))[None])[1][0].numpy()


def panel(ax, x, pred, y=None, title=''):
    for a, (i, j, name) in zip(ax, ((0, 1, 'front x,y'), (2, 0, 'top z,x'), (2, 1, 'side z,y'))):
        if y is not None:
            a.scatter(y[::8, i], y[::8, j], s=.3, c='0.75')
        a.scatter(pred[::8, i], pred[::8, j], s=.3, c='tab:blue')
        a.scatter(x[::2, i], x[::2, j], s=.5, c='tab:red')
        a.set_aspect('equal')
        a.set_title(f'{title} {name}' if a is ax[0] else name, fontsize=8)
        a.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('shard')
    ap.add_argument('out')
    ap.add_argument('--per-kind', type=int, default=2)
    ap.add_argument('--input-root', default='FINAL-INPUTS')
    a = ap.parse_args()
    model, info = load(a.ckpt)
    cfg, _ = load_config()
    z = np.load(a.shard)
    rng = np.random.default_rng(0)
    good = np.where((z['regime'] == 0) & (z['seg_frac'] >= .5))[0]
    idx = np.concatenate([rng.choice(good[z['kind'][good] == k], a.per_kind, replace=False) for k in range(len(KINDS))])
    fig, ax = plt.subplots(len(idx) // 2 + len(idx) % 2, 6, figsize=(18, 3 * (len(idx) // 2 + len(idx) % 2)), squeeze=False)
    rows = []
    for n, i in enumerate(idx):
        x, y = z['partial'][i], z['gt'][i] * GT_STEP
        pred = predict(model, x)
        cd = float(cd_per_scan(torch.from_numpy(pred)[None], torch.from_numpy(y.astype(np.float32))[None])[0]) * 1000
        back = y[y[:, 2] > y[:, 2].mean()][::8]
        d_back = torch.cdist(torch.from_numpy(back[None].astype(np.float32)), torch.from_numpy(pred[None])).amin(-1)[0].numpy()
        rows.append(dict(kind=KINDS[z['kind'][i]], cd=round(cd, 2), back_cov_mm=round(float(np.median(d_back) * z['scale'][i] * 1000), 2),
                         scale_m=round(float(z['scale'][i]), 3)))
        panel(ax[n // 2, 3 * (n % 2):3 * (n % 2) + 3], x, pred, y, f"{rows[-1]['kind']} cd={cd:.0f}")
    plt.tight_layout()
    plt.savefig(a.out + '_val.png', dpi=55)
    plt.close()

    pool, rejected = real_pool(a.input_root, cfg['real']['holdout'])
    fig, ax = plt.subplots(len(pool) // 2 + len(pool) % 2, 6, figsize=(18, 3 * (len(pool) // 2 + len(pool) % 2)), squeeze=False)
    real = []
    for n, s in enumerate(pool):
        pred = predict(model, s['partial'])
        e = (np.percentile(pred, 99, 0) - np.percentile(pred, 1, 0)) * s['T']['scale']
        d = torch.cdist(torch.from_numpy(s['partial'][None]), torch.from_numpy(pred[None])).amin(-1).mean().item() * s['T']['scale'] * 1000
        real.append(dict(session=s['session'][-6:], object=s['object'], leave_session_out=s['held_out'], width_cm=round(float(e[0]) * 100, 1),
                         height_cm=round(float(e[1]) * 100, 1), depth_cm_unmeasured=round(float(e[2]) * 100, 1), observed_to_output_mm=round(d, 1)))
        panel(ax[n // 2, 3 * (n % 2):3 * (n % 2) + 3], s['partial'], pred, None, f"{s['object']} {s['session'][-6:]}{' HELD OUT' if s['held_out'] else ''}")
    plt.tight_layout()
    plt.savefig(a.out + '_real.png', dpi=55)
    res = dict(ckpt=info, val=rows, real=real, real_rejected=rejected)
    with open(a.out + '.json', 'w') as f:
        json.dump(res, f, indent=1, default=str)
    for r in real:
        print(r)
    for k in KINDS:
        q = [r for r in rows if r['kind'] == k]
        print(k, 'cd', round(np.mean([r['cd'] for r in q]), 1), 'back_cov_mm', round(np.mean([r['back_cov_mm'] for r in q]), 1))


if __name__ == '__main__':
    main()
