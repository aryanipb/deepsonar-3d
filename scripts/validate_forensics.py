"""Validate the single-file sensor-location estimator on simulated exports with known truth.

Writes one JSON line per case to logs/forensics_validation.jsonl (truth vs estimate, score, margin)."""
import json
import sys
import time

import numpy as np

from analysis.forensics import estimate_sensor
from majproj.sim.scenes import random_session


def depth_ratio(export, complete_or_none, oz):
    far = np.percentile(export[:, 2], 99)
    near = complete_or_none[:, 2].min()
    return (near - oz) / (far - oz)


def main(n=12, seed=0, out='logs/forensics_validation.jsonl'):
    rng = np.random.default_rng(int(seed))
    with open(out, 'a') as fh:
        for i in range(int(n)):
            s = random_session(rng)
            t = time.time()
            est = estimate_sensor(s.export)
            tr = s.truth
            rec = dict(seed=int(seed), case=i, cls=s.cls, distance_m=tr['distance_m'], wall_m=tr['wall_m'],
                       true_oz=float(tr['sensor_norm'][2]), est_oz=float(est['o'][2]), alt_oz=float(est['alt_o'][2]),
                       true_ratio=tr['distance_m'] / tr['wall_m'],
                       est_ratio=float(depth_ratio(s.export, s.complete, est['o'][2])),
                       alt_ratio=float(depth_ratio(s.export, s.complete, est['alt_o'][2])),
                       score=est['score'], margin=est['margin'], seconds=time.time() - t)
            fh.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
