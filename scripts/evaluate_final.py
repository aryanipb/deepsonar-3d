"""Evaluation of a completion checkpoint: held-out simulation, every real session, probes. No Modal, runs on the local CPU.

  uv run python -m scripts.evaluate_final --final --run-log logs/final_final1.log --per-kind 0     # the shipped model, whole test split
  uv run python -m scripts.evaluate_final --ckpt path/to/best.pt --per-kind 40                     # any other checkpoint, 40 test scans per kind

--per-kind 0 uses the whole test split. Writes outputs/final/evaluation.md, evaluation.json and index.html. Unless --final
is given every output is labelled PILOT. Per-scan results are cached by checkpoint hash in ~/.cache/majproj_eval, so a stopped
run resumes, a wider --per-kind only adds scans, and the official PCN weights are not run twice.

Sim scans come from the TEST split only: val picked the checkpoint. Real sessions go through the deployment pipeline
(metric_thin -> clean(origin) -> to_canonical -> sample_2048 -> model -> from_canonical), one row per session.

GATES below were fixed before any result was seen and are not tuned to a checkpoint. A failed gate is reported, nothing else.

Honest limits, repeated in the report:
- The back and depth of a real object were never measured, so real depth is a network claim, checked only for self-consistency.
- leave_session_out is true only for configs/final.yaml real.holdout. The other sessions fed the simulator's room bank and,
  when real.enabled, the real stage. The cleaner's thresholds were tuned with all sessions in view.
"""
import argparse
import base64
import hashlib
import html
import json
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy.spatial import cKDTree

from majproj.canon import from_canonical, sample_2048, to_canonical
from majproj.clean import clean
from majproj.model import load_adapointr
from scripts.gen_final import GT_STEP, KINDS, REGIMES, load_config, metric_thin, real_sessions

CKPT = os.path.join(ROOT, 'checkpoints', 'adapointr_final.pth')         # the one line to change for a new model
PCN = os.path.join(ROOT, 'checkpoints', 'AdaPoinTr_PCN.pth')
TEST_DIR = os.path.expanduser('~/.cache/majproj_ds/ds_final_test')
CACHE = os.path.expanduser('~/.cache/majproj_eval')
OUT = os.path.join(ROOT, 'outputs', 'final')
DIST_BINS = ((0.0, 0.28), (0.28, 0.33), (0.33, 0.40), (0.40, 9.0))
CUBOIDS, ROUNDS = ('cuboid', 'cube', 'slab', 'tower'), ('bottle', 'wavy_bottle', 'can', 'jar', 'rod', 'pipe')
PLOT_POINTS = dict(raw=4000, clean=2000, out=5000, sim=2500)

GATES = dict(sim_cd_mm=5.0,            # real-regime scans with the cleaner on the object: median CD-L1 of every kind
             sim_f10=0.80,             # same scans: mean F-score at 10 mm of every kind
             one_sided_mm=10.0,        # median observed -> output, every real session
             round_rms_over_r=0.10,    # bottle: residual of the section circle / radius
             round_coverage=0.90,      # bottle: share of 24 angular bins holding output points
             round_aspect=0.80,        # bottle: minor / major extent of the section
             box_flat_mm=5.0,          # box: median face plane RMS
             box_angle_deg=5.0,        # box: worst adjacent-face angle error
             box_on_surface=0.80,      # box: share of output points within 8 mm of their fitted face plane
             session_cv=0.10,          # std / mean of width and height across the sessions of one object
             probe_margin=3.0,         # smallest between-probe distance / largest within-probe distance
             stretch_ratio=0.75)       # output width/height after x * 0.5, relative to before (a template would give 1)


# ------------------------------------------------------------------------------------------------- geometry on point clouds

def nn(a, b):
    return cKDTree(b).query(a, workers=1)[0]


def extents(q):
    """1st-99th percentile extent per axis, in the units of q. A few stray output points do not set the size."""
    return np.percentile(q, 99, 0) - np.percentile(q, 1, 0)


def roundness(q, slices=8):
    """q: metric, y up. Circle fit on 8 horizontal slices over the middle 60 % of the height, medians over the slices.
    Coverage guards the fit: a half shell also fits a circle, but fills only half of the angular bins."""
    lo, hi = np.percentile(q[:, 1], [1, 99])
    edges = np.linspace(lo + .2 * (hi - lo), hi - .2 * (hi - lo), slices + 1)
    rows = []
    for a, b in zip(edges, edges[1:]):
        s = q[(q[:, 1] >= a) & (q[:, 1] < b)][:, [0, 2]]
        if len(s) < 30:
            continue
        A = np.c_[2 * s, np.ones(len(s))]
        c, *_ = np.linalg.lstsq(A, (s ** 2).sum(1), rcond=None)
        r = np.sqrt(max(c[2] + c[0] ** 2 + c[1] ** 2, 1e-12))
        d = s - c[:2]
        e = np.hypot(*d.T) - r
        bins = np.floor((np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi) * 24).astype(int) % 24
        w = np.linalg.eigvalsh(np.cov(s.T))
        rows.append((r, np.sqrt((e ** 2).mean()), (np.bincount(bins, minlength=24) >= 3).mean(), np.sqrt(w[0] / max(w[1], 1e-12))))
    if not rows:
        return dict(radius_mm=np.nan, rms_mm=np.nan, rms_over_r=np.nan, coverage=0.0, aspect=0.0)
    r, rms, cov, asp = np.median(rows, 0)
    return dict(radius_mm=r * 1000, rms_mm=rms * 1000, rms_over_r=rms / r, coverage=cov, aspect=asp)


def box_check(q):
    """q: metric, y up. Fits the oriented box with the smallest horizontal p1-p99 footprint, gives every point to its nearest
    face, fits a plane per face. flat_mm is the median face RMS, angle_deg the worst adjacent-face error from 90 degrees,
    on_surface the share of points within 8 mm of their face plane. An 8 mm band tells shapes apart at the size of the real
    box; a 5 cm radius cylinder fits it, so a bottle-sized object is told apart by width and roundness, not by this."""
    yaws = np.deg2rad(np.arange(0, 90, 1.0))
    c, s = np.cos(yaws), np.sin(yaws)
    u, v = np.outer(c, q[:, 0]) + np.outer(s, q[:, 2]), -np.outer(s, q[:, 0]) + np.outer(c, q[:, 2])
    span = lambda a: np.percentile(a, 99, 1) - np.percentile(a, 1, 1)
    best = int(np.argmin(span(u) * span(v)))
    b = np.c_[u[best], q[:, 1], v[best]]
    lo, hi = np.percentile(b, 1, 0), np.percentile(b, 99, 0)
    d = np.c_[np.abs(b - lo), np.abs(b - hi)]                     # distance to each of the six planes
    face = d.argmin(1)
    normals, flat, fill = {}, [], []
    on = 0
    for f in range(6):
        p = b[face == f]
        if len(p) < 30:
            continue
        p = p - p.mean(0)
        _, sv, vt = np.linalg.svd(p, full_matrices=False)
        normals[f] = vt[2] * np.sign(vt[2, f % 3] or 1)
        on += int((np.abs(p @ vt[2]) <= .008).sum())
        flat.append(sv[2] / np.sqrt(len(p)) * 1000)
        ax = [i for i in range(3) if i != f % 3]
        g = np.floor((b[face == f][:, ax] - lo[ax]) / np.maximum(hi[ax] - lo[ax], 1e-6) * 8).clip(0, 7).astype(int)
        fill.append(len(np.unique(g[:, 0] * 8 + g[:, 1])) / 64)
    ang = [abs(90 - np.degrees(np.arccos(min(abs(normals[a] @ normals[b_]), 1.0))))
           for a in normals for b_ in normals if a < b_ and a % 3 != b_ % 3]
    return dict(faces=len(normals), flat_mm=float(np.median(flat)) if flat else np.nan, flat_max_mm=max(flat, default=np.nan),
                angle_deg=max(ang, default=np.nan), face_fill=float(np.min(fill)) if fill else 0.0,
                on_surface=on / len(b), yaw_deg=float(best))


def scan_metrics(pred, gt, x, scale):
    """pred, gt, x canonical. Distances in metres through the scan's own scale. The CD equals train_final.cd_per_scan x scale."""
    p, g, i = (np.asarray(a, np.float64) * scale for a in (pred, gt, x))
    pg, gp = nn(p, g), nn(g, p)
    m = dict(cd_mm=(pg.mean() + gp.mean()) / 2 * 1000)
    for t in (5, 10):
        P, R = (pg <= t / 1000).mean(), (gp <= t / 1000).mean()
        m[f'f{t}'] = 2 * P * R / (P + R) if P + R else 0.0
    unseen = nn(g, i) > .01                                       # target surface more than 1 cm from any input point
    m['unseen_frac'] = unseen.mean()
    m['unseen_recall10'] = (gp[unseen] <= .01).mean() if unseen.any() else np.nan
    e, eg = extents(p), extents(g)
    for n, k in enumerate('whd'):
        m[f'{k}_err_mm'] = abs(e[n] - eg[n]) * 1000
    return m


def shape_metrics(pred, scale):
    q = np.asarray(pred, np.float64) * scale
    return dict(**{f'r_{k}': v for k, v in roundness(q).items()}, **{f'b_{k}': v for k, v in box_check(q).items()})


def aligned_cd(a, b, yaws=36):
    """Symmetric mean distance between two clouds (metric, y up) after centring, over the best rotation about y. Mm."""
    a, b = a - (a.min(0) + a.max(0)) / 2, b - (b.min(0) + b.max(0)) / 2
    t, best = cKDTree(a), np.inf
    for y in np.linspace(0, 2 * np.pi, yaws, endpoint=False):
        c, s = np.cos(y), np.sin(y)
        r = b @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
        best = min(best, (t.query(r)[0].mean() + cKDTree(r).query(a)[0].mean()) / 2)
    return best * 1000


# ------------------------------------------------------------------------------------------------- model, cache, inputs

def file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_model(path):
    """The official weights, or a training checkpoint on top of them. Returns (model, step, hashes)."""
    model = load_adapointr(PCN, 'cpu')
    step, hashes = 0, None
    if path != PCN:
        st = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(st.get('ema', st))
        step, hashes = st.get('step') or st.get('record', {}).get('selected_step'), st.get('hashes')
    return model.eval(), step, hashes


@torch.no_grad()
def predict(model, x):
    return model(torch.from_numpy(np.asarray(x, np.float32)))[1].numpy()


def memo(path, make):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    v = make()
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(v, f)
    os.replace(path + '.tmp', path)
    return v


def load_test(d):
    with open(os.path.join(d, 'manifest.json')) as f:
        manifest = json.load(f)
    parts = [np.load(os.path.join(d, n)) for n in sorted(os.listdir(d)) if n.endswith('.npz')]
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    return data, manifest


def stratified(kinds, per_kind, seed=0):
    """Indices of the first `per_kind` scans of a seeded order per kind, so a wider run contains a narrower one. 0 = all."""
    rng = np.random.default_rng(seed)
    idx = []
    for k in range(len(KINDS)):
        q = rng.permutation(np.where(kinds == k)[0])
        idx.append(q[:per_kind] if per_kind else q)
    return np.sort(np.concatenate(idx))


def sim_rows(name, model, data, idx, batch):
    """Per-scan metrics for one model, resumable: one JSON line per scan. model=None is the input returned unchanged."""
    path = os.path.join(CACHE, f'{name}.sim.jsonl')
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            done = {r['i']: r for r in map(json.loads, f)}
    todo = [int(i) for i in idx if int(i) not in done]
    t0 = time.time()
    with open(path, 'a') as f:
        for n in range(0, len(todo), batch):
            b = todo[n:n + batch]
            pred = predict(model, data['partial'][b]) if model is not None else data['partial'][b]
            for j, i in enumerate(b):
                gt = data['gt'][i].astype(np.float64) * GT_STEP
                s = float(data['scale'][i])
                r = dict(i=i, **scan_metrics(pred[j], gt, data['partial'][i], s), **shape_metrics(pred[j], s))
                done[i] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in r.items()}
                f.write(json.dumps(done[i]) + '\n')
            f.flush()
            print(f'[sim {name[:12]}] {n + len(b)}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    return [done[int(i)] for i in idx]


def real_inputs(input_root, holdout):
    """Every real session once, through the deployment pipeline. Rejected sessions stay in the list."""
    rng = np.random.default_rng(0)
    out = []
    for s in real_sessions(input_root):
        raw = np.load(s['path'])
        pts, _ = metric_thin(raw)
        r = clean(pts, origin=(0.0, 0.0, 0.0))
        row = dict(s, held_out=s['session'] in holdout, n_raw=len(raw), n_thin=len(pts), status=r.meta['status'],
                   reason=str(r.meta.get('reason')), raw_view=pts[rng.permutation(len(pts))[:PLOT_POINTS['raw']]])
        if r.meta['status'] == 'ok' and len(r.object) >= 20:
            canon, T = to_canonical(r.object)
            row.update(obj=np.asarray(r.object, np.float64), partial=sample_2048(canon), T=T, canon=canon)
        else:
            row['status'] = 'rejected'
        out.append(row)
    return out


def probe_clouds(seed):
    """Seen surfaces in metres, sensor at the origin, object centred about 35 cm along +y, 3 mm range noise."""
    rng = np.random.default_rng(seed)
    n = 3000

    def box_faces(w, h, d, yaw):
        """Front faces of a box seen by the sensor. Faces are sampled by visible area."""
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = []
        for ax, sign, a, b in ((0, -1, d, h), (0, 1, d, h), (1, -1, w, h), (1, 1, w, h)):
            nrm = Rz @ np.eye(3)[ax] * sign
            centre = Rz @ (np.eye(3)[ax] * sign * [w, d, h][ax] / 2) + [0, .35, 0]
            if nrm @ centre >= 0:
                continue
            uv = rng.uniform(-.5, .5, (int(n * a * b / (w * h + d * h)), 2)) * [a, b]
            f = np.zeros((len(uv), 3))
            f[:, 1 - ax] = uv[:, 0]
            f[:, 2] = uv[:, 1]
            f[:, ax] = sign * [w, d][ax] / 2
            p.append(f @ Rz.T + [0, .35, 0])
        return np.vstack(p)

    a = rng.uniform(-np.pi / 2, np.pi / 2, n)
    rod = np.c_[.01 * np.sin(a), .35 - .01 * np.cos(a), rng.uniform(-.15, .15, n)]
    noise = rng.uniform(-.5, .5, (n, 3)) * .2 + [0, .35, 0]
    clouds = dict(plate=box_faces(.30, .20, .004, 0.0), rod=rod, cube=box_faces(.12, .12, .12, np.deg2rad(35)), noise=noise)
    return {k: v + rng.normal(0, .003, v.shape) for k, v in clouds.items()}


def run_probes(model):
    """Between-probe vs within-probe distance of the outputs. The canonical frame hides absolute size, so distances are in
    canonical units x 1000; a frontal plate and a frontal cube face are the same input by construction and are not compared."""
    res = {}
    for seed in range(3):
        for k, c in probe_clouds(seed).items():
            canon, T = to_canonical(c)
            res.setdefault(k, []).append(dict(x=sample_2048(canon), T=T, seen=c))
    names = list(res)
    x = np.stack([r['x'] for k in names for r in res[k]])
    pred = predict(model, x)
    for j, (k, n) in enumerate((k, n) for k in names for n in range(3)):
        res[k][n]['pred'] = pred[j]
    dist = lambda a, b: (nn(a, b).mean() + nn(b, a).mean()) / 2 * 1000
    within = {k: max(dist(res[k][a]['pred'], res[k][b]['pred']) for a, b in ((0, 1), (0, 2), (1, 2))) for k in names}
    between = {f'{a}/{b}': dist(res[a][0]['pred'], res[b][0]['pred']) for n, a in enumerate(names) for b in names[n + 1:]}
    dims = {k: (extents(res[k][0]['pred']) * res[k][0]['T']['scale']).tolist() for k in names}
    return dict(within=within, between=between, dims=dims, views={k: dict(seen=res[k][0]['seen'], out=from_canonical(res[k][0]['pred'], res[k][0]['T'])) for k in names})


def stretch_probe(model, partial):
    """The same real scan with its width halved. A network that answers with a stored shape would not follow."""
    x = np.stack([partial, partial * np.array([.5, 1, 1], np.float32)])
    p = predict(model, x)
    e, s = extents(p[0]), extents(p[1])
    return dict(before=e.tolist(), after=s.tolist(), ratio=float((s[0] / s[1]) / (e[0] / e[1])))


def real_outputs(model, sessions):
    rows = []
    for s in sessions:
        if s['status'] != 'ok':
            rows.append(None)
            continue
        p = predict(model, s['partial'][None])[0]
        scale = s['T']['scale']
        out = from_canonical(p, s['T'])
        d = nn(s['obj'], out) * 1000
        e = extents(p) * scale
        keep = np.sort(d)[:max(1, int(round(len(d) * .9)))]
        rows.append(dict(pred=p, out=out, width_m=float(e[0]), height_m=float(e[1]), depth_m=float(e[2]),
                         one_sided_mean_mm=float(d.mean()), one_sided_median_mm=float(np.median(d)),
                         one_sided_p95_mm=float(np.percentile(d, 95)), one_sided_trim10_mm=float(keep.mean()),
                         **shape_metrics(p, scale)))
    return rows


# ------------------------------------------------------------------------------------------------- tables

def fmt(v, d=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    return f'{v:.{d}f}' if isinstance(v, (float, np.floating)) else str(v)


def table(title, head, rows, note=''):
    return dict(title=title, head=head, rows=[[fmt(c) if not isinstance(c, str) else c for c in r] for r in rows], note=note)


def col(rows, key):
    return np.array([r[key] for r in rows], np.float64)


def sim_table(title, rows, group, names, note=''):
    """rows: dict model -> per-scan list, same order. group: per-scan group index, names: its labels."""
    m, o, i = rows['model'], rows['pcn'], rows['input']
    out = []
    for g, name in enumerate(names):
        q = np.where(group == g)[0]
        if not len(q):
            continue
        pick = lambda r: [r[j] for j in q]
        M, O, I = pick(m), pick(o), pick(i)
        out.append([name, str(len(q)), np.mean(col(M, 'cd_mm')), np.median(col(M, 'cd_mm')), np.median(col(O, 'cd_mm')),
                    np.median(col(I, 'cd_mm')), np.mean(col(M, 'f5')) * 1, np.mean(col(M, 'f10')) * 1, np.mean(col(O, 'f10')),
                    np.mean(col(I, 'f10')), np.nanmean(col(M, 'unseen_recall10')), np.nanmean(col(I, 'unseen_recall10')),
                    np.median(col(M, 'w_err_mm')), np.median(col(M, 'h_err_mm')), np.median(col(M, 'd_err_mm'))])
    for r in out:
        for j in (6, 7, 8, 9, 10, 11):
            r[j] = f'{r[j]:.3f}'
    return table(title, ['group', 'n', 'CD mean mm', 'CD median mm', 'official PCN median', 'input median', 'F@5 mm', 'F@10 mm',
                         'official F@10', 'input F@10', 'unseen recall@10 mm', 'input unseen recall', 'W err mm', 'H err mm', 'D err mm'],
                 out, note)


def check(name, passed, detail):
    return dict(gate=name, result='PASS' if passed else 'FAIL', detail=detail)


def sim_gates(rows, data, idx):
    g = GATES
    kind, reg, seg = data['kind'][idx], data['regime'][idx], data['seg_frac'][idx]
    good = (reg == REGIMES.index('real')) & (seg >= .5)
    res = {}
    for k, name in enumerate(KINDS):
        q = np.where((kind == k) & good)[0]
        res[name] = {w: (np.median(col([rows[w][j] for j in q], 'cd_mm')), np.mean(col([rows[w][j] for j in q], 'f10')))
                     for w in rows} if len(q) else None
    live = {k: v for k, v in res.items() if v}
    bad = lambda f: [k for k, v in live.items() if not f(v)] or ['none']
    a, b = bad(lambda v: v['model'][0] < v['input'][0]), bad(lambda v: v['model'][0] < v['pcn'][0])
    c, d = bad(lambda v: v['model'][0] <= g['sim_cd_mm']), bad(lambda v: v['model'][1] >= g['sim_f10'])
    return [check('sim: median CD below returning the input, every kind', a == ['none'], f'failing kinds: {", ".join(a)}'),
            check('sim: median CD below official PCN weights, every kind', b == ['none'], f'failing kinds: {", ".join(b)}'),
            check(f'sim: median CD at most {g["sim_cd_mm"]} mm, every kind', c == ['none'], f'failing kinds: {", ".join(c)}'),
            check(f'sim: mean F@10 mm at least {g["sim_f10"]}, every kind', d == ['none'], f'failing kinds: {", ".join(d)}')]


def real_gates(sessions, out, consistency, probes, stretch):
    g = GATES
    pairs = list(zip(sessions, out))
    ok = [(s, r) for s, r in pairs if r]
    tag = lambda s: s['session'][-6:] + ('*' if s['held_out'] else '')
    fails = lambda f, ob=None: [tag(s) for s, r in ok if (ob is None or s['object'] == ob) and not f(r)]
    res = []

    def gate(name, f, ob=None):
        bad = fails(f, ob)
        res.append(check(name, not bad, f'{len(bad)} sessions fail: {" ".join(bad)}' if bad else 'all sessions pass'))

    rejected = [s['session'][-6:] for s, r in pairs if not r]
    res.append(check(f'real: every one of the {len(pairs)} sessions gives an output', not rejected, f'rejected: {" ".join(rejected) or "none"}'))
    gate(f'real: one-sided median at most {g["one_sided_mm"]} mm', lambda r: r['one_sided_median_mm'] <= g['one_sided_mm'])
    gate(f'bottle: section residual / radius at most {g["round_rms_over_r"]}', lambda r: r['r_rms_over_r'] <= g['round_rms_over_r'], 'BOTTLE')
    gate(f'bottle: angular coverage at least {g["round_coverage"]}', lambda r: r['r_coverage'] >= g['round_coverage'], 'BOTTLE')
    gate(f'bottle: section aspect at least {g["round_aspect"]}', lambda r: r['r_aspect'] >= g['round_aspect'], 'BOTTLE')
    gate(f'box: median face flatness at most {g["box_flat_mm"]} mm', lambda r: r['b_flat_mm'] <= g['box_flat_mm'], 'BOX')
    gate(f'box: worst right-angle error at most {g["box_angle_deg"]} deg', lambda r: r['b_angle_deg'] <= g['box_angle_deg'], 'BOX')
    gate(f'box: at least {g["box_on_surface"]} of the output within 8 mm of a fitted face', lambda r: r['b_on_surface'] >= g['box_on_surface'], 'BOX')
    for ob in ('BOX', 'BOTTLE'):
        c = consistency[ob]
        for k in ('width', 'height'):
            res.append(check(f'{ob.lower()}: {k} std/mean across sessions at most {g["session_cv"]}', c[k]['cv'] <= g['session_cv'],
                             f'{c[k]["cv"]:.3f} over {c["n"]} sessions'))
    margin = min(probes['between'].values()) / max(max(probes['within'].values()), 1e-9)
    res.append(check(f'probes: plate, rod, cube, noise differ, margin at least {g["probe_margin"]}', margin >= g['probe_margin'], f'margin {margin:.2f}'))
    res.append(check(f'probe: width/height of the output follows a halved width, ratio at most {g["stretch_ratio"]}', stretch['ratio'] <= g['stretch_ratio'],
                     f'ratio {stretch["ratio"]:.3f}; 0.5 is full tracking, 1 is a fixed shape'))
    return res


def consistency_of(sessions, out):
    res = {}
    for ob in ('BOX', 'BOTTLE'):
        rs = [r for s, r in zip(sessions, out) if r and s['object'] == ob]
        c = dict(n=len(rs))
        for k in ('width', 'height', 'depth'):
            v = col(rs, f'{k}_m') * 100
            c[k] = dict(mean=float(v.mean()), std=float(v.std()), min=float(v.min()), max=float(v.max()), cv=float(v.std() / v.mean()))
        clouds = [(r['pred'] * s['T']['scale'])[:: 8] for s, r in zip(sessions, out) if r and s['object'] == ob]
        d = [aligned_cd(a, b) for n, a in enumerate(clouds) for b in clouds[n + 1:]]
        c['pairwise_cd_mm'] = dict(median=float(np.median(d)), max=float(np.max(d)))
        res[ob] = c
    box = [(r['pred'] * s['T']['scale'])[:: 8] for s, r in zip(sessions, out) if r and s['object'] == 'BOX']
    bot = [(r['pred'] * s['T']['scale'])[:: 8] for s, r in zip(sessions, out) if r and s['object'] == 'BOTTLE']
    res['box_vs_bottle_cd_mm'] = float(np.median([aligned_cd(a, b) for a in box[:4] for b in bot[:4]]))
    return res


# ------------------------------------------------------------------------------------------------- report

def real_tables(sessions, out, pcn_out, used, label):
    a, b = [], []
    for s, r, o in zip(sessions, out, pcn_out):
        sid = s['session'][-6:]
        lso = 'yes' if s['held_out'] else 'no'
        stage = 'unknown' if used is None else ('yes' if s['session'] in used else 'no')
        if not r:
            a.append([s['object'], sid, lso, stage, s['status'], s['reason']] + ['-'] * 7)
            continue
        a.append([s['object'], sid, lso, stage, 'ok', str(len(s['obj'])), s['T']['scale'] * 90, r['width_m'] * 100, r['height_m'] * 100, r['depth_m'] * 100,
                  r['one_sided_median_mm'], r['one_sided_p95_mm'], o['one_sided_median_mm']])
        b.append([s['object'], sid, lso, r['r_radius_mm'] * 2 / 10, r['r_rms_mm'], f"{r['r_rms_over_r']:.3f}", f"{r['r_coverage']:.2f}", f"{r['r_aspect']:.2f}",
                  str(r['b_faces']), r['b_flat_mm'], r['b_flat_max_mm'], r['b_angle_deg'], f"{r['b_on_surface']:.2f}", f"{r['b_face_fill']:.2f}"])
    return [table(f'Real sessions, {label}: size and one-sided fit',
                  ['object', 'session', 'leave-session-out', 'in real stage', 'cleaner', 'cleaned points', 'seen face long side cm', 'output width cm',
                   'output height cm', 'output depth cm (never measured)', 'one-sided median mm', 'one-sided p95 mm', 'official PCN one-sided median mm'], a,
                  'One-sided fit: every cleaned point to its nearest output point. It says the output covers what was measured, nothing about the back.'),
            table(f'Real sessions, {label}: shape of the output',
                  ['object', 'session', 'leave-session-out', 'section diameter cm', 'section residual mm', 'residual / radius', 'angular coverage', 'section aspect',
                   'faces', 'face flatness mm', 'worst face mm', 'worst right-angle error deg', 'share within 8 mm of face', 'worst face fill'], b,
                  'Both checks run on every session with no label, so a box row under the round columns and a bottle row under the box columns show how far apart the two shapes are.')]


def reference_table(rows, kinds):
    """The same shape measurements on simulated outputs, for reading the real rows."""
    out = []
    for k, name in enumerate(KINDS):
        q = [r for r, j in zip(rows, kinds) if j == k]
        if q:
            med = lambda key: float(np.nanmedian(col(q, key)))
            out.append([name, str(len(q)), med('r_radius_mm') * 2 / 10, med('r_rms_mm'), f"{med('r_rms_over_r'):.3f}", f"{med('r_coverage'):.2f}", f"{med('r_aspect'):.2f}",
                        med('b_flat_mm'), med('b_angle_deg'), f"{med('b_on_surface'):.2f}"])
    return table('Same shape checks on held-out simulated outputs (median per kind)',
                 ['kind', 'n', 'section diameter cm', 'section residual mm', 'residual / radius', 'angular coverage', 'section aspect', 'face flatness mm',
                  'worst right-angle error deg', 'share within 8 mm of face'], out)


def write_md(path, title, intro, gates, tables):
    L = [f'# {title}', '', *intro, '', '## Gates', '', '| gate | result | detail |', '|---|---|---|']
    L += [f"| {g['gate']} | {g['result']} | {g['detail']} |" for g in gates]
    for t in tables:
        L += ['', f"## {t['title']}", '']
        if t['note']:
            L += [t['note'], '']
        L += ['| ' + ' | '.join(t['head']) + ' |', '|' + '---|' * len(t['head'])]
        L += ['| ' + ' | '.join(r) + ' |' for r in t['rows']]
    with open(path, 'w') as f:
        f.write('\n'.join(L) + '\n')


def layer(pts, color, n, size=1.3):
    pts = np.asarray(pts, np.float64)
    if len(pts) > n:
        pts = pts[np.random.default_rng(0).permutation(len(pts))[:n]]
    mm = np.clip(np.rint(pts * 1000), -32767, 32767).astype('<i2')
    return dict(b=base64.b64encode(mm.tobytes()).decode(), n=len(mm), c=color, s=size)


def html_table(t):
    head = ''.join(f'<th>{html.escape(h)}</th>' for h in t['head'])
    body = ''.join('<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in r) + '</tr>' for r in t['rows'])
    note = f"<p class=note>{html.escape(t['note'])}</p>" if t['note'] else ''
    return f"<h2>{html.escape(t['title'])}</h2>{note}<div class=scroll><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_html(path, title, intro, gates, tables, groups):
    gate_rows = ''.join(f"<tr class={g['result'].lower()}><td>{html.escape(g['gate'])}</td><td><b>{g['result']}</b></td><td>{html.escape(g['detail'])}</td></tr>" for g in gates)
    gate_html = f'<h2>Gates</h2><div class=scroll><table><thead><tr><th>gate</th><th>result</th><th>detail</th></tr></thead><tbody>{gate_rows}</tbody></table></div>'
    doc = TEMPLATE.replace('__TITLE__', html.escape(title)).replace('__INTRO__', ''.join(f'<p>{html.escape(i)}</p>' for i in intro))
    doc = doc.replace('__TABLES__', gate_html + ''.join(html_table(t) for t in tables)).replace('__DATA__', json.dumps(groups).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(doc)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completion Evaluation</title><style>
:root{--bg:#fbfbf9;--fg:#1d1f21;--mute:#62676d;--line:#d9dbd6;--card:#fff;--pass:#176b45;--fail:#b3261e;--axis:#8a8f94}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15171a;--fg:#e8e9e6;--mute:#a0a5a9;--line:#33373b;--card:#1d2024;--pass:#6fd0a0;--fail:#ff8a80;--axis:#7c8186}}
:root[data-theme="dark"]{--bg:#15171a;--fg:#e8e9e6;--mute:#a0a5a9;--line:#33373b;--card:#1d2024;--pass:#6fd0a0;--fail:#ff8a80;--axis:#7c8186}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1240px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 12px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:15px;margin:20px 0 4px}
p{margin:6px 0;max-width:90ch}.note{color:var(--mute);font-size:13px}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--card)}
table{border-collapse:collapse;font-size:13px;width:100%}th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;font-variant-numeric:tabular-nums}
th{font-weight:600;color:var(--mute);white-space:normal;vertical-align:bottom;min-width:64px}tr.pass b{color:var(--pass)}tr.fail b{color:var(--fail)}
.row{margin:14px 0;padding:10px;border:1px solid var(--line);border-radius:6px;background:var(--card)}.row h3{margin:0 0 6px}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}.view{min-width:0}.view div{font-size:12px;color:var(--mute);margin-bottom:2px}
canvas{display:block;width:100%;height:280px;touch-action:none;cursor:grab;border:1px solid var(--line);border-radius:4px}
.key span{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 3px 0 8px}
</style></head><body><main><h1>__TITLE__</h1>__INTRO__
<p class=note>Drag a view to rotate its whole row, wheel to zoom. Views are metres, drawn when scrolled to. Axis ticks are labelled in metres.</p>
__TABLES__<div id=rows></div></main>
<script type="application/json" id=data>__DATA__</script>
<script>
const G=JSON.parse(document.getElementById('data').textContent);
const cssv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function decode(l){if(l.p)return l.p;const b=atob(l.b),a=new Int16Array(b.length/2);for(let i=0;i<a.length;i++){const v=b.charCodeAt(2*i)|(b.charCodeAt(2*i+1)<<8);a[i]=v>32767?v-65536:v}return l.p=a}
function nice(x){const p=10**Math.floor(Math.log10(x)),f=x/p;return (f<1.5?1:f<3.5?2:f<7.5?5:10)*p}
function draw(v,st){const c=v.cv,dpr=devicePixelRatio||1,W=c.clientWidth,H=c.clientHeight;if(!W)return;c.width=W*dpr;c.height=H*dpr;
 const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,W,H);const P=v.layers.map(decode);
 if(!v.box){let lo=[1e9,1e9,1e9],hi=[-1e9,-1e9,-1e9];for(const a of P)for(let i=0;i<a.length;i+=3)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],a[i+k]);hi[k]=Math.max(hi[k],a[i+k])}v.box={lo,hi}}
 const {lo,hi}=v.box,m=[0,1,2].map(k=>(lo[k]+hi[k])/2),R=Math.max(1,Math.hypot(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])/2),
 cy=Math.cos(st.yaw),sy=Math.sin(st.yaw),cp=Math.cos(st.pitch),sp=Math.sin(st.pitch),f=.46*Math.min(W,H)/R*st.zoom;
 const pr=(a,b,d)=>{const X=a-m[0],Y=b-m[1],Z=d-m[2],u=cy*X+sy*Y,w=-sy*X+cy*Y,t=cp*Z-sp*w,dep=sp*Z+cp*w;return [W/2+f*u,H/2-f*t,dep]};
 const ax=cssv('--axis');x.strokeStyle=ax;x.fillStyle=ax;x.lineWidth=1;x.font='10px system-ui';const step=nice(Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])/4);
 ['x','y','z'].forEach((nm,k)=>{const s=[lo[0],lo[1],lo[2]],e=s.slice();e[k]=hi[k];const A=pr(...s),B=pr(...e);x.beginPath();x.moveTo(A[0],A[1]);x.lineTo(B[0],B[1]);x.stroke();
  for(let t=Math.ceil(lo[k]/step)*step;t<=hi[k];t+=step){const q=s.slice();q[k]=t;const T=pr(...q);x.fillRect(T[0]-1,T[1]-1,2,2);x.fillText((t/1000).toFixed(step<100?2:1),T[0]+3,T[1]-3)}
  x.fillText(nm+' m',B[0]+4,B[1]+4)});
 P.forEach((a,n)=>{const l=v.layers[n];x.fillStyle=l.c;const s=l.s;for(let i=0;i<a.length;i+=3){const q=pr(a[i],a[i+1],a[i+2]);x.globalAlpha=Math.max(.3,Math.min(1,.65-q[2]/(2.4*R)));x.fillRect(q[0]-s/2,q[1]-s/2,s,s)}});x.globalAlpha=1}
function mount(g,root){const r=document.createElement('section');r.className='row';r.innerHTML='<h3></h3>';r.firstChild.textContent=g.title;const vs=document.createElement('div');vs.className='views';r.append(vs);
 const st={yaw:.5,pitch:.35,zoom:1},views=g.views.map(o=>{const d=document.createElement('div');d.className='view';const t=document.createElement('div');t.textContent=o.title;const cv=document.createElement('canvas');d.append(t,cv);vs.append(d);return {cv,layers:o.layers}});
 const redraw=()=>views.forEach(v=>draw(v,st));let drag=null,raf=0;const later=()=>{if(!raf)raf=requestAnimationFrame(()=>{raf=0;redraw()})};
 views.forEach(v=>{v.cv.onpointerdown=e=>{drag=[e.clientX,e.clientY];v.cv.setPointerCapture(e.pointerId)};v.cv.onpointermove=e=>{if(!drag)return;st.yaw+=(e.clientX-drag[0])*.01;st.pitch=Math.max(-1.5,Math.min(1.5,st.pitch+(e.clientY-drag[1])*.01));drag=[e.clientX,e.clientY];later()};v.cv.onpointerup=()=>drag=null;
  v.cv.onwheel=e=>{e.preventDefault();st.zoom=Math.max(.3,Math.min(20,st.zoom*Math.exp(-e.deltaY*.001)));later()}});
 new IntersectionObserver((es,ob)=>{if(es.some(e=>e.isIntersecting)){ob.disconnect();redraw();addEventListener('resize',later)}},{rootMargin:'300px'}).observe(r);root.append(r)}
const root=document.getElementById('rows');for(const g of G){const h=document.createElement('h2');h.textContent=g.title;root.append(h);if(g.note){const p=document.createElement('p');p.className='note';p.textContent=g.note;root.append(p)}g.rows.forEach(r=>mount(r,root))}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT)
    ap.add_argument('--final', action='store_true', help='the checkpoint is the final model; without this every output says PILOT')
    ap.add_argument('--run-log', help='training log of this checkpoint: its start event lists the sessions in the real stage')
    ap.add_argument('--per-kind', type=int, default=40, help='test scans per kind, 0 for the whole split')
    ap.add_argument('--test-dir', default=TEST_DIR)
    ap.add_argument('--input-root', default=os.path.join(ROOT, 'FINAL-INPUTS'))
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--batch', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    label = 'FINAL' if a.final else 'PILOT, NOT FINAL'
    t0 = time.time()

    cfg, _ = load_config()
    holdout = set(cfg['real']['holdout'])
    data, manifest = load_test(a.test_dir)
    sha, pcn_sha = file_sha(a.ckpt)[:16], file_sha(PCN)[:16]
    model, step, hashes = load_model(a.ckpt)
    pcn, _, _ = load_model(PCN)
    if hashes and hashes.get('dataset') != manifest['dataset_hash']:
        raise SystemExit(f"checkpoint dataset {hashes.get('dataset')} is not the test manifest {manifest['dataset_hash']}: wrong test split")
    if hashes and hashes.get('code') != manifest['code_hash']:
        print('note: checkpoint code hash differs from the dataset code hash', flush=True)
    used = None
    if a.run_log:
        with open(a.run_log) as f:
            start = [json.loads(s) for s in f if s.startswith('{"event": "start"')]
        used = set(start[-1]['real_train'])

    idx = stratified(data['kind'], a.per_kind)
    ds = manifest['dataset_hash'][:12]
    rows = dict(input=sim_rows(f'input_{ds}', None, data, idx, 64), model=sim_rows(f'{sha}_{ds}', model, data, idx, a.batch),
                pcn=sim_rows(f'{pcn_sha}_{ds}', pcn, data, idx, a.batch))
    code = hashlib.sha256(b''.join(open(os.path.join(ROOT, p), 'rb').read() for p in ('majproj/clean.py', 'majproj/canon.py', 'scripts/gen_final.py'))).hexdigest()[:16]
    sessions = memo(os.path.join(CACHE, f'real_{code}.pkl'), lambda: real_inputs(a.input_root, holdout))
    names = {s['session'] for s in sessions}
    if holdout - names:
        raise SystemExit(f'held-out names are not the listed export of a session: {holdout - names}')
    out = memo(os.path.join(CACHE, f'real_{code}_{sha}.pkl'), lambda: real_outputs(model, sessions))
    pcn_out = memo(os.path.join(CACHE, f'real_{code}_{pcn_sha}.pkl'), lambda: real_outputs(pcn, sessions))
    probes = memo(os.path.join(CACHE, f'probes_{sha}.pkl'), lambda: run_probes(model))
    box = next(s for s in sessions if s['object'] == 'BOX' and s['held_out'] and s['status'] == 'ok')
    stretch = memo(os.path.join(CACHE, f'stretch_{sha}_{code}.pkl'), lambda: stretch_probe(model, box['partial']))
    consistency = consistency_of(sessions, out)

    kind, regime, dist, seg = (data[k][idx] for k in ('kind', 'regime', 'distance_m', 'seg_frac'))
    good = (regime == REGIMES.index('real')) & (seg >= .5)
    sub = lambda m: {k: [v[j] for j in np.where(m)[0]] for k, v in rows.items()}
    d_bin = np.digitize(dist, [b[1] for b in DIST_BINS[:-1]])
    d_names = [f'{lo:.2f}-{hi:.2f} m' if hi < 9 else f'{lo:.2f} m and over' for lo, hi in DIST_BINS]
    n_all, n_split = len(idx), len(data['kind'])
    sim_note = ('CD-L1 is the mean of both directions, in millimetres on the scan\'s own scale. Official PCN weights were trained in PCN\'s normalisation, '
                'not this canonical frame; they get the same inputs as asked. Unseen recall: target points over 1 cm from every input point that an output point reaches within 10 mm.')
    tables = [sim_table(f'Simulation by kind, cleaner on the object, regime real ({int(good.sum())} scans)', sub(good), kind[good], KINDS, sim_note),
              sim_table(f'Simulation by kind, every accepted scan ({n_all} scans)', rows, kind, KINDS,
                        'Includes scans where the cleaner kept the wrong object: no network can fix those, and mean far above median shows them.'),
              sim_table('Simulation by distance, every accepted scan', rows, d_bin, d_names),
              sim_table('Simulation by regime, every accepted scan', rows, regime, REGIMES),
              reference_table(rows['model'], kind)]
    tables += real_tables(sessions, out, pcn_out, used, label)
    c_rows = [[ob, str(c['n']), *(f"{c[k]['mean']:.1f} / {c[k]['std']:.1f} / {c[k]['min']:.1f}-{c[k]['max']:.1f}" for k in ('width', 'height', 'depth')),
               f"{c['width']['cv']:.3f}", f"{c['height']['cv']:.3f}", f"{c['depth']['cv']:.3f}", c['pairwise_cd_mm']['median'], c['pairwise_cd_mm']['max']]
              for ob, c in consistency.items() if ob in ('BOX', 'BOTTLE')]
    tables.append(table('Consistency across sessions of the one box and the one bottle',
                        ['object', 'sessions', 'width cm mean / std / range', 'height cm', 'depth cm (never measured)', 'width std/mean', 'height std/mean', 'depth std/mean',
                         'pairwise distance median mm', 'pairwise distance max mm'], c_rows,
                        f"Pairwise distance is between two sessions' outputs, centred, at the best rotation about the vertical. Box against bottle for scale: {consistency['box_vs_bottle_cd_mm']:.1f} mm."))
    d_rows = [[k, *(f'{v * 100:.1f}' for v in probes['dims'][k]), probes['within'][k]] for k in probes['dims']]
    tables.append(table('Probes: output size and repeatability', ['probe', 'width cm', 'height cm', 'depth cm', 'worst same-probe distance (canonical x 1000)'], d_rows,
                        'Clouds are built in metres, not by the simulator: a 30 x 20 cm plate, a 2 cm rod 30 cm long, a 12 cm cube turned 35 degrees, uniform noise in a 20 cm cube, each with three noise seeds.'))
    tables.append(table('Probes: distance between outputs (canonical x 1000)', ['pair', 'distance'], [[k, v] for k, v in probes['between'].items()],
                        f"Halved width probe on held-out box {box['session'][-6:]}: output width/height changes by a factor {stretch['ratio']:.3f} (0.5 tracks fully, 1 means a fixed shape)."))

    gates = sim_gates(rows, data, idx) + real_gates(sessions, out, consistency, probes, stretch)
    n_fail = sum(g['result'] == 'FAIL' for g in gates)
    held = [s for s, r in zip(sessions, out) if s['held_out'] and r]
    used_txt = 'unknown (no --run-log given)' if used is None else f'{len(used)} sessions'
    title = f'Completion evaluation: {label}'
    sub_txt = f'{n_all} of {n_split} scans (seeded per-kind sample)' if a.per_kind else f'all {n_split} scans'
    intro = [f'Checkpoint {os.path.basename(a.ckpt)}, step {step}, sha256 {sha}. Label: {label}. {sum(g["result"] == "PASS" for g in gates)} of {len(gates)} gates pass, {n_fail} fail.',
             f'Simulation: test split, {sub_txt}, {manifest["splits"]["test"]["reject_rate"] * 100:.1f} % of drawn test scans were rejected by the cleaner and are not in these numbers. Val picked the checkpoint and is not used here.',
             f'Real: {len(sessions)} sessions, each once, {len(held)} are leave-session-out at every stage ({" ".join(s["session"][-6:] for s in held)}, starred in the gates). Sessions in the real stage of this checkpoint: {used_txt}. The other 18 fed the simulator\'s room bank, so they are not held out and are never labelled so. The cleaner\'s thresholds were tuned with all sessions in view, so no session is held out from the cleaner.',
             'The back and depth of a real object were never measured. Real depth is what the network claims; the only check on it is agreement across sessions and with the simulation error on depth.',
             'The network input is the 2048 canonical points only. No label, kind or size reaches it at inference. All gate thresholds were written down before the first result.']

    groups = [dict(title=f'Real sessions: raw, cleaned, output ({label})', note='Raw is the thinned file with the cleaned object in red. Depth of the output was never measured.', rows=[])]
    for s, r in zip(sessions, out):
        sid = s['session'][-6:] + (' (leave-session-out)' if s['held_out'] else '')
        raw = dict(title=f"raw, {s['n_raw']} rows", layers=[layer(s['raw_view'], '#7b8794', PLOT_POINTS['raw'], 1.2)])
        if r:
            raw['layers'].append(layer(s['obj'], '#d6452b', PLOT_POINTS['clean'], 1.8))
            views = [raw, dict(title=f"cleaned, {len(s['obj'])} points", layers=[layer(s['obj'], '#d6452b', PLOT_POINTS['clean'], 2)]),
                     dict(title=f"output, {r['width_m'] * 100:.1f} x {r['height_m'] * 100:.1f} x {r['depth_m'] * 100:.1f} cm, red is cleaned",
                          layers=[layer(r['out'], '#2b7bd6', PLOT_POINTS['out'], 1.5), layer(s['obj'], '#d6452b', 800, 2)])]
        else:
            views = [raw]
            sid += f": cleaner rejected ({s['reason']})"
        groups[0]['rows'].append(dict(title=f"{s['object']} {sid}", views=views))

    def median_scan(k):
        q = np.where((kind == k) & good)[0]
        if not len(q):
            return None
        cd = col([rows['model'][j] for j in q], 'cd_mm')
        return int(idx[q[np.argsort(cd)[len(q) // 2]]])

    picks = [(k, median_scan(k)) for k in range(len(KINDS))]
    picks = [(k, i) for k, i in picks if i is not None]
    x = data['partial'][[i for _, i in picks]]
    pm, po = memo(os.path.join(CACHE, f'gallery_{sha}_{a.per_kind}.pkl'), lambda: (predict(model, x), predict(pcn, x)))
    zup = lambda q, s: np.asarray(q, np.float64)[:, [0, 2, 1]] * s
    sim_rows_html = []
    for n, (k, i) in enumerate(picks):
        s = float(data['scale'][i])
        gt = data['gt'][i].astype(np.float64) * GT_STEP
        v = [('input', x[n], '#d6452b'), ('target', gt, '#7b8794'), ('output', pm[n], '#2b7bd6'), ('official PCN', po[n], '#9a6fb0')]
        sim_rows_html.append(dict(title=f"{KINDS[k]}, median scan of its kind, {data['distance_m'][i]:.2f} m",
                                  views=[dict(title=t, layers=[layer(zup(q, s), c, PLOT_POINTS['sim'], 1.5)]) for t, q, c in v]))
    groups.append(dict(title=f'Held-out simulation ({label})', note='The scan with the median CD of its kind, not the best one.', rows=sim_rows_html))
    groups.append(dict(title=f'Probes ({label})', rows=[dict(title=k, views=[dict(title='input', layers=[layer(v['seen'], '#d6452b', 2500, 1.5)]),
                                                                             dict(title='output', layers=[layer(v['out'], '#2b7bd6', 4000, 1.5)])]) for k, v in probes['views'].items()]))

    write_md(os.path.join(a.out, 'evaluation.md'), title, intro, gates, tables)
    write_html(os.path.join(a.out, 'index.html'), title, intro, gates, tables, groups)
    summary = dict(label=label, ckpt=os.path.basename(a.ckpt), step=step, sha=sha, gates=gates, gate_thresholds=GATES, consistency=consistency, probes={k: v for k, v in probes.items() if k != 'views'},
                   stretch=stretch, leave_session_out=sorted(s['session'] for s in held), real_stage_sessions=None if used is None else sorted(used), n_sim=n_all, n_test_split=n_split,
                   real=[dict(session=s['session'], object=s['object'], leave_session_out=s['held_out'], status=s['status'],
                              **({k: v for k, v in r.items() if k not in ('pred', 'out')} if r else {})) for s, r in zip(sessions, out)],
                   tables=[dict(t) for t in tables])
    with open(os.path.join(a.out, 'evaluation.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)
    print(json.dumps(dict(label=label, passed=len(gates) - n_fail, failed=n_fail, seconds=round(time.time() - t0)), default=float))
    for g in gates:
        print(g['result'], '|', g['gate'], '|', g['detail'])


if __name__ == '__main__':
    main()
