"""Training data for the final model: simulated raw scan -> the deployment pipeline -> (partial, complete target).

Each sample runs exactly what inference runs on a real file:
    simulate_raw -> run_pipeline.thin -> clean(origin) -> to_canonical -> sample_2048
and the target is the complete surface in the transform of the SEEN points. Scans the cleaner rejects are skipped and
counted, per kind and per reason.

Every scan carries truth['regime'] from the simulator: 'real' (fitted to the measured sessions, at least 80 % of what the
simulator draws) or 'robust' (wider ranges). The regime is stored in the shards, so validation can be reported per regime
and the best checkpoint chosen on 'real' alone. A scan without a valid tag is an error: nothing defaults to 'real'.

Local run (small):   uv run python -m scripts.gen_final out_dir --split val --shards 1 --per-shard 8 --workers 4
On Modal:            modal_final.py gen_shard (see its docstring)

A shard is two files in <root>/<split>/: shard_00000.npz (uncompressed, fast to read back) and shard_00000.json.
`write_manifest` then joins all shard reports into <root>/manifest.json with the dataset hash that training records.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.spatial import cKDTree

from majproj.canon import apply_canonical, sample_2048, to_canonical
from majproj.clean import clean
from scripts.run_pipeline import thin

KINDS = ('cuboid', 'cube', 'slab', 'tower', 'bottle', 'wavy_bottle', 'can', 'jar', 'rod', 'pipe')
REGIMES = ('real', 'robust')
N_TARGET = 16384

# Targets are stored as int16 fixed point. A thin face seen end-on makes the back several units deep, so the range is
# +-16 units at 0.0005 units per step: about 0.1 mm on a 20 cm object.
GT_STEP = 1.0 / 2048
GT_LIMIT = 32767 * GT_STEP

CONFIG = os.path.join(ROOT, 'configs', 'final.yaml')


def load_config(path=CONFIG):
    import yaml
    with open(path, 'rb') as f:
        raw = f.read()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def code_hash():
    """sha256 over every source file that decides what the data or the model is."""
    h = hashlib.sha256()
    files = glob.glob(os.path.join(ROOT, 'majproj', '**', '*.py'), recursive=True)
    files += [os.path.join(ROOT, 'scripts', n) for n in ('gen_final.py', 'run_pipeline.py')]
    files += [os.path.join(ROOT, 'configs', n) for n in ('real_bank.npz', 'real_bank.json')]
    for f in sorted(files):
        h.update(os.path.relpath(f, ROOT).encode())
        with open(f, 'rb') as fh:
            h.update(fh.read())
    return h.hexdigest()


def metric_thin(raw):
    """1 cm voxel thinning for a raw metric map. thin() treats a file with every coordinate within +-1 as an already
    normalised export; a raw map is metric whatever its range, so such a scan is thinned too. Returns (points, forced)."""
    pts, metric = thin(raw)
    if metric:
        return pts, False
    pts, _ = thin(np.vstack([raw, [[2.0, 2.0, 2.0]]]))       # one point out of range selects the metric branch
    return pts[np.abs(pts - 2.0).max(1) > 1e-9], True


def make_sample(key, kind, simulate=None, seg_tolerance_m=0.03):
    """One scan -> dict(status='ok', arrays) or dict(status='rejected'|'error', reason). Everything is seeded by `key`."""
    if simulate is None:
        from majproj.sim_raw import simulate_raw as simulate
    rng = np.random.default_rng(key)
    try:
        s = simulate(rng, kind=kind)
        regime = s['truth'].get('regime')
        if regime not in REGIMES:
            raise ValueError(f'truth regime is {regime!r}')
        pts, forced = metric_thin(s['raw'])
        r = clean(pts, origin=(0.0, 0.0, 0.0))
        if r.meta['status'] != 'ok' or len(r.object) < 20:
            return dict(status='rejected', kind=kind, regime=regime, reason=str(r.meta.get('reason') or 'too few object points'))
        complete = np.asarray(s['complete'], np.float64)
        if len(complete) != N_TARGET:
            raise ValueError(f'complete has {len(complete)} points')
        canon, T = to_canonical(r.object)
        gt = apply_canonical(complete, T)
        if np.abs(gt).max() >= GT_LIMIT:
            return dict(status='rejected', kind=kind, regime=regime, reason='target outside the stored range')
        d, _ = cKDTree(complete).query(np.asarray(r.object, np.float64), k=1, workers=1)
        truth = s['truth']
        return dict(status='ok', kind=kind, regime=regime, partial=sample_2048(canon), gt=np.rint(gt / GT_STEP).astype(np.int16),
                    scale=T['scale'], distance_m=float(truth.get('distance_m', np.nan)), n_raw=len(s['raw']),
                    n_object=len(r.object), seg_frac=float((d <= seg_tolerance_m).mean()), forced_thin=forced,
                    truth=truth, T=T)
    except Exception as e:                  # one bad scan must not lose a shard; the error rate is checked per shard
        return dict(status='error', kind=kind, reason=f'{type(e).__name__}: {e}')


def _job(a):
    return make_sample(*a)


def _plain(o):
    """numpy scalars and arrays in a simulator's truth -> JSON."""
    return o.tolist() if hasattr(o, 'tolist') else str(o)


def choose_kinds(seed, n, weights):
    names = list(weights)
    p = np.array([weights[k] for k in names], np.float64)
    return [names[i] for i in np.random.default_rng(seed).choice(len(names), n, p=p / p.sum())]


def gen_shard_file(root, split, shard, cfg, workers=1, per_shard=None, simulate=None):
    """Writes one shard and returns its report. Deterministic for a given (seed, split, shard)."""
    g = cfg['gen']
    n = per_shard or g['per_shard']
    split_id = dict(train=0, val=1, test=2)[split]
    base = [cfg['seed'], split_id, shard]
    kinds = choose_kinds(base, n, g['kinds'])
    jobs = [(base + [i], k, simulate, g['seg_tolerance_m']) for i, k in enumerate(kinds)]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_job, jobs, chunksize=2))
    else:
        res = [_job(j) for j in jobs]
    ok = [(j, r) for j, r in zip(jobs, res) if r['status'] == 'ok']
    report = dict(split=split, shard=shard, attempted=n, kept=len(ok), seconds=round(time.time() - t0, 1), workers=workers,
                  attempted_by_kind=dict(Counter(kinds)), kept_by_kind=dict(Counter(r['kind'] for _, r in ok)),
                  attempted_by_regime=dict(Counter(r['regime'] for r in res if 'regime' in r)),
                  kept_by_regime=dict(Counter(r['regime'] for _, r in ok)),
                  rejected=dict(Counter(f"{r['kind']}: {r['reason']}" for r in res if r['status'] == 'rejected')),
                  errors=dict(Counter(r['reason'] for r in res if r['status'] == 'error')),
                  forced_thin=sum(r['forced_thin'] for _, r in ok))
    n_err = sum(report['errors'].values())
    if n_err > g['max_error_rate'] * n:
        raise RuntimeError(f'{split} shard {shard}: {n_err} of {n} scans raised exceptions: {report["errors"]}')
    if not ok:
        raise RuntimeError(f'{split} shard {shard}: no scan survived: {report["rejected"]}')
    d = os.path.join(root, split)
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f'shard_{shard:05d}')
    col = lambda k, t: np.array([r[k] for _, r in ok], t)
    np.savez(stem + '.tmp.npz', partial=np.stack([r['partial'] for _, r in ok]), gt=np.stack([r['gt'] for _, r in ok]),
             kind=np.array([KINDS.index(r['kind']) for _, r in ok], np.int16),
             regime=np.array([REGIMES.index(r['regime']) for _, r in ok], np.int16), scale=col('scale', np.float32),
             distance_m=col('distance_m', np.float32), seg_frac=col('seg_frac', np.float32),
             n_object=col('n_object', np.int32), key_index=np.array([j[0][-1] for j, _ in ok], np.int32))
    os.replace(stem + '.tmp.npz', stem + '.npz')
    with open(stem + '.npz', 'rb') as f:
        report['sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    report['truth'] = [r['truth'] for _, r in ok]
    with open(stem + '.json', 'w') as f:
        json.dump(report, f, default=_plain)
    return {k: v for k, v in report.items() if k != 'truth'}


def write_manifest(root, cfg_hash):
    """Joins the shard reports. Two runs with equal shards and code get equal hashes."""
    shards = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'shard_*.json'))):
        with open(f) as fh:
            r = json.load(fh)
        r.pop('truth', None)
        shards.append(r)
    if not shards:
        raise RuntimeError(f'no shards under {root}')
    h = hashlib.sha256()
    splits = {}
    for r in shards:
        h.update(f"{r['split']}/{r['shard']}:{r['sha256']}\n".encode())
        s = splits.setdefault(r['split'], dict(shards=0, attempted=0, kept=0, seconds=0.0, attempted_by_kind=Counter(),
                                               kept_by_kind=Counter(), attempted_by_regime=Counter(),
                                               kept_by_regime=Counter(), rejected=Counter(), errors=Counter(), forced_thin=0))
        s['shards'] += 1
        for k in ('attempted', 'kept', 'seconds', 'forced_thin'):
            s[k] += r[k]
        for k in ('attempted_by_kind', 'kept_by_kind', 'attempted_by_regime', 'kept_by_regime', 'rejected', 'errors'):
            s[k].update(r[k])
    for s in splits.values():
        s['reject_rate'] = round(1 - s['kept'] / s['attempted'], 4)
        s['reject_rate_by_kind'] = {k: round(1 - s['kept_by_kind'][k] / v, 4) for k, v in s['attempted_by_kind'].items()}
        s['reject_rate_by_regime'] = {k: round(1 - s['kept_by_regime'][k] / v, 4) for k, v in s['attempted_by_regime'].items()}
        # share of the kept scans: the cleaner rejects per regime, so this can differ from what the simulator drew
        s['real_frac'] = round(s['kept_by_regime']['real'] / s['kept'], 4)
    m = dict(dataset_hash=h.hexdigest(), code_hash=code_hash(), config_hash=cfg_hash, splits=splits)
    with open(os.path.join(root, 'manifest.json'), 'w') as f:
        json.dump(m, f, indent=1, default=dict)
    return m


# ------------------------------------------------------------------------------------------------- real sessions

def real_sessions(input_root):
    """One entry per real session: repeat exports of a session collapse to the largest file, so a session counts once."""
    groups = {'BOX': [('121018', '121022', '121055'), ('121422', '121457')],
              'BOTTLE': [('122617', '122622'), ('123027', '123028'), ('123257', '123259'), ('123814', '123815')]}
    out = []
    for obj, gs in groups.items():
        files = sorted(glob.glob(os.path.join(input_root, obj, '*.npy')))
        stamp = lambda f: os.path.basename(f)[:-4].rsplit('_', 1)[1]
        grouped = {t for g in gs for t in g}
        sessions = [[f for f in files if stamp(f) in g] for g in gs] + [[f] for f in files if stamp(f) not in grouped]
        for fs in sessions:
            if fs:
                best = max(fs, key=lambda f: len(np.load(f, mmap_mode='r')))
                out.append(dict(object=obj, session=os.path.basename(best)[:-4], path=best, exports=len(fs)))
    return sorted(out, key=lambda s: (s['object'], s['session']))


def real_pool(input_root, holdout):
    """Real sessions through the deployment pipeline. A held-out name may be any export of its session.
    Returns one dict per session that the cleaner accepts, plus the list of those it rejects."""
    pool, rejected = [], []
    for s in real_sessions(input_root):
        pts, _ = metric_thin(np.load(s['path']))
        r = clean(pts, origin=(0.0, 0.0, 0.0))
        s = dict(s, held_out=s['session'] in holdout)
        if r.meta['status'] != 'ok' or len(r.object) < 20:
            rejected.append(dict(session=s['session'], object=s['object'], held_out=s['held_out'],
                                 reason=str(r.meta.get('reason'))))
            continue
        canon, T = to_canonical(r.object)
        pool.append(dict(s, partial=sample_2048(canon), T=T, n_object=len(r.object)))
    return pool, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--per-shard', type=int)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    cfg, cfg_hash = load_config()
    for i in range(a.start, a.start + a.shards):
        print(json.dumps(gen_shard_file(a.root, a.split, i, cfg, a.workers, a.per_shard)), flush=True)
    print(json.dumps(write_manifest(a.root, cfg_hash), default=dict)[:2000])


if __name__ == '__main__':
    main()
