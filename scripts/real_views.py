"""Every real file through the deployment path: raw input, cleaned input, predicted output. Real scans have no target.

    uv run python -m scripts.real_views checkpoints/adapointr_final.pth outputs/final/real_views.html
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.infer import load_checkpoint
from scripts.four_views import PLOTLY, TEMPLATE, card
from scripts.gen_final import load_config, metric_thin


def main(ckpt, out):
    model, info = load_checkpoint(ckpt)
    holdout = set(load_config()[0]['real']['holdout'])
    rng = np.random.default_rng(0)
    cards = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'FINAL-INPUTS', '*', '*.npy'))):
        ob, name = os.path.basename(os.path.dirname(f)), os.path.basename(f)[:-4]
        raw = np.load(f)
        pts, _ = metric_thin(raw)
        note = 'held out of training' if name in holdout else 'session fed the simulator room bank'
        cards.append(card(f'Real {ob.lower()} {name[5:]}', note, raw, pts, None, model, rng))
        print(cards[-1]['title'], cards[-1]['status'], cards[-1].get('seen_size'), cards[-1].get('pred_size'), flush=True)
    head = f"Best checkpoint so far (lowest validation score), step {info.get('step')}. Every file below is a real scan from the device."
    page = (TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards)).replace('/*HEAD*/', head)
            .replace('Raw input, cleaned input, predicted output, target output', 'Real scans: raw input, cleaned input, predicted output')
            .replace('repeat(4,1fr)', 'repeat(3,1fr)')
            .replace('<div><div class="t">4. Target output</div><div class="p"></div></div>', '')
            .replace(" A target exists only for simulated scans:\nnobody measured the back of the real box or bottle, so real scans have no target and the predicted depth cannot be checked.",
                     " Nobody measured the back of the real box or bottle, so the predicted depth is a learned guess and cannot be checked.")
            .replace("else P[3].innerHTML = '<div class=\"none\">No target exists for a real scan: the back was never measured.</div>';", '')
            .replace("for (const i of [1, 2, 3]) P[i]", "for (const i of [1, 2]) P[i]")
            .replace(" · no target exists", ""))
    with open(out, 'w') as f:
        f.write(page)
    print('->', out)


if __name__ == '__main__':
    main(*sys.argv[1:])
