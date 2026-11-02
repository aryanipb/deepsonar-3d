"""Honest evaluation of the single-file pipeline on the 10 real scans (a TEST SET: never trained on, fused or tuned on)
plus input-sensitivity probes. Companion of scripts/evaluate_sim.py (held-out simulated sessions with known truth).

Each real file is processed ALONE: preprocess v2 -> AdaPoinTr -> from_canonical. No ground truth exists for the hidden
side of a real object (and no metric size), so the only real-data numbers are one-sided: how well the prediction
passes through the observed points, how big the completed object is relative to the wall distance, and how the
predictions of repeated scans of the same object agree. Labels (red/blue bottle, box) are used only to group results.

Writes the sections 'real' and 'probes' of outputs/evaluation_v3.md plus outputs/renders/real_v3.png and
outputs/renders/probes_v3.png.

Usage:  PYTHONPATH=. uv run python scripts/evaluate_real.py [--ckpt PATH]
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from majproj.infer import Completer
from majproj.labels import USER_NAMES
from scripts.evaluate_sim import (header_text, nn_dist, one_sided_fit, robust_longest_side,  # noqa: E402
                                  update_report)

RENDERS = os.path.join(ROOT, 'outputs', 'renders')
COLORS = {'Red bottle': '#d62728', 'Blue bottle': '#1f77b4', 'Box': '#d98e04'}


# ------------------------------------------------------------------------------------ helpers
def extents(p, lo=1, hi=99):
    a, b = np.percentile(p, [lo, hi], axis=0)
    return b - a


def old_classifier(observed, device='cpu'):
    """DIAGNOSTIC ONLY. The v1 PointNet (bottle vs box) was trained on simulated v1-normalized scans, said 'box' for all
    10 real scans and has not been re-validated, so it appears in no user-facing output. Fed in its own v1 frame."""
    from majproj.classifier import N_POINTS, PointNetCls
    from majproj.geometry import normalize, normalize_params, resample_fixed
    path = os.path.join(ROOT, 'checkpoints', 'classifier.pth')
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location='cpu')
    net = PointNetCls(len(state['classes']))
    net.load_state_dict(state['model'])
    net.eval()
    p = normalize(observed, *normalize_params(observed))
    with torch.no_grad():
        prob = torch.softmax(net(torch.from_numpy(resample_fixed(p, N_POINTS))[None]), 1)[0].numpy()
    return state['classes'][int(prob.argmax())], float(prob[state['classes'].index('bottle')])


def panel(fig, rows, r, title, observed, completed, color, sensor=None):
    """One row of three views of the observed (black) and completed (coloured) points; sensor = red triangle.
    View limits are the 1-99 percentile box of completed + observed (plus 15 % margin): the model output has a thin
    diffuse cloud of stray points that would otherwise shrink the object to a speck. Points outside the view are not
    drawn; their share is printed in the title. The sensor is drawn only if it falls inside the view."""
    observed = np.unique(observed, axis=0)                       # the model input tiles few points; show each once
    observed = observed[::max(1, len(observed) // 700)]
    both = np.concatenate([completed, observed])
    lo, hi = np.percentile(both, [1, 99], axis=0)
    m, h = (hi + lo) / 2, float((hi - lo).max()) / 2 * 1.15
    inside = np.all(np.abs(completed - m) <= h, axis=1)
    c = completed[inside]
    out_pct = 100 * (1 - inside.mean())
    sens = None if sensor is None else np.asarray(sensor, float)
    sens_in = sens is not None and bool(np.all(np.abs(sens - m) <= h))
    for j, (el, az, v) in enumerate([(15, -70, 'front (sensor side)'), (15, 110, 'back (never seen)'), (85, -90, 'top')]):
        ax = fig.add_subplot(rows, 3, r * 3 + j + 1, projection='3d')
        ax.scatter(c[::2, 0], c[::2, 2], c[::2, 1], s=0.8, c=color, alpha=0.45)
        ax.scatter(observed[:, 0], observed[:, 2], observed[:, 1], s=1.2, c='k')
        if sens_in and j == 2:
            ax.scatter(*sens[[0, 2, 1]], s=30, c='red', marker='^')
        ax.set_xlim(m[0] - h, m[0] + h); ax.set_ylim(m[2] - h, m[2] + h); ax.set_zlim(m[1] - h, m[1] + h)
        ax.view_init(el, az)
        ax.tick_params(labelsize=5)
        extra = ''
        if j == 0:
            extra = f' [{out_pct:.0f}% of completed pts outside view]'
        elif j == 2 and sens is not None and not sens_in:
            extra = f' [sensor off view, {np.linalg.norm(sens - observed.mean(0)):.1f} units from patch]'
        ax.set_title(f'{title}: {v}{extra}', fontsize=6)


def canonical_probes(rng, n=2048):
    """Bare partial patches in the canonical frame (centroid at the origin, sensor ~1 unit away at -z, one VL53L5CX
    zone ~ 0.10 units). They bypass preprocess (no wall, so no stripes) and test the model's dependence on its input."""
    sig = 0.003
    u = rng.uniform
    a = u(-1.2, 1.2, n)
    R = 0.12
    nz = rng.normal(0, sig, n)
    x = u(-.3, .3, n)
    probes = {
        'flat square patch (3 x 3 zones)': np.c_[u(-.15, .15, n), u(-.15, .15, n), nz],
        'wide flat strip': np.c_[u(-.3, .3, n), u(-.075, .075, n), nz],
        'tall cylinder patch (vertical axis)': np.c_[R * np.sin(a), u(-.2, .2, n), R * (1 - np.cos(a)) + nz],
        'lying cylinder patch (horizontal axis)': np.c_[u(-.2, .2, n), R * np.sin(a), R * (1 - np.cos(a)) + nz],
        'box edge pointing at the sensor': np.c_[x / 2, u(-.2, .2, n), np.abs(x / 2) + nz],
        'random noise ball': rng.normal(0, .07, (n, 3)),
    }
    return {k: (v - v.mean(0)).astype(np.float32) for k, v in probes.items()}


def pair_cd(a, b):
    """CD-L1 between two completed shapes as a fraction of their mean longest side."""
    d1, d2 = nn_dist(a, b), nn_dist(b, a)
    return float((d1.mean() + d2.mean()) / 2 / ((robust_longest_side(a) + robust_longest_side(b)) / 2))


def fmt3(e):
    return f'{e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f}'


# ------------------------------------------------------------------------------------ real scans
def evaluate_real(comp):
    files = sorted(glob.glob(os.path.join(ROOT, 'inputs', 'scan_*_shapenet.npy')))
    rows, results = [], []
    for f in files:
        stem = os.path.basename(f)[:-4]
        res = comp(np.load(f))
        results.append(res)
        m = res.meta
        comp_ext = extents(res.completed_canonical)
        wall = m['sensor']['wall_depth'] if m['sensor'] else None
        cls = old_classifier(res.observed)
        rows.append(dict(
            scan=stem[14:20], stem=stem, said=USER_NAMES[stem], mode=m['mode'], confident=m['confident'],
            ratio=m['object_depth_ratio'], n_obj=m['n_object_points'], seen=extents(res.partial),
            comp=comp_ext, comp_longest_canon=float(comp_ext.max()),
            comp_longest_wall=(float(comp_ext.max() * res.scale / wall) if wall else None),
            fit=one_sided_fit(res.observed, res.completed), ang=m['angular_size_deg'],
            ang_canon=m['angular_size_canonical_deg'], scale=res.scale, clamped=m['scale_clamped'], cls=cls))
    return files, rows, results


def real_stability(comp, results, rng):
    """(a) own prediction vs another scan's prediction (centroid aligned): is the visible surface fit input-specific?
    (b) repeat the same file with a small canonical jitter: how much does the output move?"""
    own, foreign, jitter = [], [], []
    for i, r in enumerate(results):
        c_obs = r.partial.mean(0)
        own.append(one_sided_fit(r.partial, r.completed_canonical))
        f = []
        for j, o in enumerate(results):
            if j != i:
                f.append(one_sided_fit(r.partial, o.completed_canonical - o.completed_canonical.mean(0) + c_obs))
        foreign.append(float(np.mean(f)))
        noisy = (r.partial + rng.normal(0, 0.01, r.partial.shape)).astype(np.float32)   # ~10 % of one zone
        jitter.append(pair_cd(r.completed_canonical, comp.complete_canonical(noisy)))
    return own, foreign, jitter


def real_section(comp, rows, stab, group_rows, sim_ref):
    own, foreign, jitter = stab
    L = ['## Real scans (test set, one file at a time)', '']
    if not comp.frame_ok:
        L += [f'> **SMOKE TEST, NOT A RESULT**: `{comp.ckpt_name}` was trained on the old v1 normalization, so the '
              'completed shapes below are not meaningful. Re-run after `adapointr_v3.pth` is trained.', '']
    L += [f'Checkpoint `{comp.ckpt_name}`. "You said" is the label the user gave each scan (a test label, never a '
          'prediction). Sizes are in canonical units (object-distance units, NOT cm) unless stated otherwise.', '',
          '| Scan | You said | Frame | Object distance (% of wall distance) | Segmented pts | Seen patch w x h x d | '
          'Completed w x h x d | Completed longest side (x wall distance) | One-sided fit (% of completed size) | '
          'Angle: sensor-direction / canonical-patch (deg) | `confident` flag (unvalidated) | v1 classifier (diagnostic) |',
          '|' + '---|' * 12]
    for r in rows:
        ratio = 'n/a' if r['ratio'] is None else f"{100 * r['ratio']:.0f}%" + (' (floor)' if r['clamped'] else '')
        ang = 'n/a' if r['ang'] is None else f"{r['ang']:.0f} / {r['ang_canon']:.0f}"
        cw = 'n/a' if r['comp_longest_wall'] is None else f"{r['comp_longest_wall']:.2f}"
        cls = 'n/a' if r['cls'] is None else f"{r['cls'][0]} (P(bottle) {r['cls'][1]:.2f})"
        L.append(f"| {r['scan']} | {r['said']} | {r['mode']} | {ratio} | {r['n_obj']} | {fmt3(r['seen'])} | "
                 f"{fmt3(r['comp'])} | {cw} | {100 * r['fit']:.1f} | {ang} | {r['confident']} | {cls} |")
    fits = [r['fit'] for r in rows]
    L += ['', f"One-sided fit = mean distance from the observed object points to the completed surface / longest side "
              f"of the completed object (0 = passes exactly through what the sensor saw). Mean over the 10 scans: "
              f"{100 * np.mean(fits):.1f} %." + (f" For comparison, the same statistic on {sim_ref}" if sim_ref else '')
          + ' It is a **necessary** check, not a sufficient one: a blob that engulfs the patch also scores well, and '
            'there is no ground truth for the hidden side of a real object, so no CD-L1 / F-score can be computed '
            'on real files.', '',
          'Angle columns: the first is the angular extent of the segmented cluster as seen from the estimated sensor '
          'position; hand sweeps inflate it (values above the 45 deg field of view are not an object property). The '
          'second is the angle the seen patch would subtend one canonical unit away, which is only right if the '
          'estimated distance (scale) is right.', '',
          f"v1 classifier column: {sum(r['cls'] is not None and r['cls'][0] == ('bottle' if r['said'].endswith('bottle') else 'box') for r in rows)}/10 "
          'correct. It was trained on simulated scans in the old frame, is not used anywhere in the apps and has '
          'not been re-validated (red vs blue cannot be told apart anyway: the scans have no colour).', '',
          '### Is the prediction input-specific?', '',
          '| Check | Value |', '|---|---|',
          f'| One-sided fit of each scan against ITS OWN completion (mean %) | {100 * np.mean(own):.1f} |',
          f'| ... against another scan\'s completion, centroid aligned (mean over the 9 others, then over scans, %) | '
          f'{100 * np.mean(foreign):.1f} |',
          f'| Output change when the canonical input is jittered by 0.01 units (CD-L1 / size, mean %) | '
          f'{100 * np.mean(jitter):.1f} |', '',
          'If the model ignored its input, "own" and "another scan\'s" would be equal. Jitter should be small '
          'compared with the difference between scans.', '',
          '### Agreement between repeated scans of the same object (labels used only for grouping)', '',
          'The completed object\'s longest side in units of the wall distance is dimensionless, so it is comparable '
          'between scans only if the wall was at the same distance, which is not known. Large spreads therefore mean '
          '"the distance-consistent frame did not make the scans agree", not necessarily a model error.', '',
          '| Object | Scans | Completed longest side (x wall distance), min / median / max | max / min |', '|---|---|---|---|']
    L += group_rows
    return '\n'.join(L)


def group_table(rows):
    out = []
    for name in ['Red bottle', 'Blue bottle', 'Box']:
        v = [r['comp_longest_wall'] for r in rows if r['said'] == name and r['comp_longest_wall'] is not None]
        if v:
            out.append(f'| {name} | {len(v)} | {min(v):.2f} / {np.median(v):.2f} / {max(v):.2f} | {max(v) / min(v):.1f}x |')
    return out


def sim_reference(comp):
    p = os.path.join(ROOT, 'outputs', 'eval_sim_v3.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d.get('checkpoint') != comp.ckpt_name:
        return None
    v = [100 * r['fit1'] for r in d['rows'] if r['bin'] <= 2]
    return (f'simulated sessions at 3-40 cm (same checkpoint): median {np.median(v):.1f} %, '
            f'IQR {np.percentile(v, 25):.1f}-{np.percentile(v, 75):.1f} %.')


# ------------------------------------------------------------------------------------ probes
def probes_section(comp, outs, outs2, names):
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = pair_cd(outs[names[i]], outs[names[j]])
    off = M[~np.eye(n, dtype=bool)]
    rep = [pair_cd(outs[k], outs2[k]) for k in names]
    L = ['## Input-sensitivity probes (synthetic canonical partials the model never saw)', '']
    if not comp.frame_ok:
        L += [f'> **SMOKE TEST, NOT A RESULT**: `{comp.ckpt_name}` was trained on the old v1 normalization.', '']
    L += ['Each probe is a bare patch in the canonical frame (centroid at the origin, one zone ~ 0.10 units), fed '
          'straight to the model (no wall, so the preprocessing step is bypassed). If the model ignored its input, '
          'all outputs would be the same shape. Completed shapes are compared by CD-L1 divided by their mean longest side.',
          '', '| Probe | Completed w x h x d (canonical units) |', '|---|---|']
    for k in names:
        L.append(f'| {k} | {fmt3(extents(outs[k]))} |')
    L += ['', '**Pairwise difference between the completed shapes (CD-L1 / size, %):**', '',
          '| | ' + ' | '.join(str(i + 1) for i in range(n)) + ' |', '|---|' + '---|' * n]
    for i in range(n):
        L.append(f'| {i + 1}. {names[i]} | ' + ' | '.join('-' if i == j else f'{100 * M[i, j]:.1f}' for j in range(n)) + ' |')
    ok = off.min() > 0.02
    L += ['', f'Smallest pairwise difference: {100 * off.min():.1f} %, median {100 * np.median(off):.1f} %. '
              f'Same probe re-drawn with different noise: median {100 * np.median(rep):.1f} % (max {100 * max(rep):.1f} %). '
              f"Criterion (every pair differs by more than 2 %, the strict F-score threshold): **{'PASS' if ok else 'FAIL'}**"
              + ('' if comp.frame_ok else ' (smoke-test checkpoint: not informative)') + '.',
          '', 'Different outputs show that the model reacts to its input, not that the outputs are correct: probes '
              'have no ground truth.']
    return '\n'.join(L), M, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args(argv)
    os.makedirs(RENDERS, exist_ok=True)
    comp = Completer(args.ckpt)
    print('checkpoint:', comp.ckpt, '' if comp.frame_ok else '(SMOKE TEST: v1 frame)', flush=True)

    # ---- real scans
    files, rows, results = evaluate_real(comp)
    stab = real_stability(comp, results, np.random.default_rng(5))
    text_real = real_section(comp, rows, stab, group_table(rows), sim_reference(comp))
    fig = plt.figure(figsize=(14, 3.6 * len(rows)))
    for r, (row, res) in enumerate(zip(rows, results)):
        s = res.meta['sensor']
        ratio = 'n/a' if row['ratio'] is None else f"{100 * row['ratio']:.0f}% of wall dist."
        panel(fig, len(rows), r, f"{row['scan']} ({row['said']}, {row['mode']}, {ratio})", res.partial,
              res.completed_canonical, COLORS[row['said']], None if s is None else s['canonical_o'])
    fig.suptitle(f'Real scans, canonical frame (object-distance units), checkpoint {comp.ckpt_name}'
                 + ('' if comp.frame_ok else ' [SMOKE TEST]') + '\nBlack = seen, colour = completed (axes zoomed to the 1-99 pct box, strays outside are not drawn), '
                 'red triangle = estimated sensor when inside the view (its lateral error is comparable to the object distance at close range)',
                 fontsize=9, y=0.997)
    plt.tight_layout(); plt.savefig(os.path.join(RENDERS, 'real_v3.png'), dpi=70); plt.close(fig)

    # ---- probes
    probes = canonical_probes(np.random.default_rng(1))
    probes2 = canonical_probes(np.random.default_rng(2))
    names = list(probes)
    outs = {k: comp.complete_canonical(v) for k, v in probes.items()}
    outs2 = {k: comp.complete_canonical(v) for k, v in probes2.items()}
    text_probes, M, ok = probes_section(comp, outs, outs2, names)
    fig = plt.figure(figsize=(14, 3.6 * len(names)))
    for r, k in enumerate(names):
        panel(fig, len(names), r, k, probes[k], outs[k], '#2ca02c')
    fig.suptitle(f'Input-sensitivity probes, canonical frame, checkpoint {comp.ckpt_name}'
                 + ('' if comp.frame_ok else ' [SMOKE TEST]'), fontsize=9, y=0.995)
    plt.tight_layout(); plt.savefig(os.path.join(RENDERS, 'probes_v3.png'), dpi=70); plt.close(fig)

    update_report('header', header_text(comp))
    update_report('real', text_real)
    update_report('probes', text_probes)
    print(text_real, '\n', text_probes)
    return rows, M, ok


if __name__ == '__main__':
    main()
