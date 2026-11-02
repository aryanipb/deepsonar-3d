"""Validate `majproj.clean.clean` on (1) simulated scans where the true object points are known and (2) the 15 real files.

    PYTHONPATH=. uv run python -m scripts.validate_clean                       # 200 sims, seed 20261028, all cores
    PYTHONPATH=. uv run python -m scripts.validate_clean --n 50 --workers 4    # quick run
    -> outputs/clean_validation/report.md, index.html (self-contained 3D cards), sim_results.json, real_results.json

SIM. `majproj.sim_scan.simulate_scan(rng)` returns the 4096 x 3 export plus the truth (`object_mask`, kind, distance...).
The cleaner only ever receives a bare copy of the export (default arguments, no mask, no kind), so there is no leak.
Scoring is at the level of unique export coordinates: a point returned in `Cleaned.object` is a true positive when its
export row was masked as object, a false positive otherwise; masked rows the cleaner did not return are false negatives
(duplicate rows of one coordinate are merged; a second, multiplicity-weighted F1 over the 4096 rows is reported too).
A rejected scan delivers nothing, so its gated F1 is 0; the raw F1 (what `Cleaned.object` held anyway) is
reported separately. Because a rejected scan normally returns nothing, whether a rejection was justified is judged from the
true object size (a scan whose object has a few dozen points may be unrecoverable) and the largest rejected objects are listed.
Per-scan seeds are `default_rng([seed, i])`, so the run is reproducible whatever the number of workers.

REAL. The 10 files in inputs/ and the 5 in inputs_box/ have no ground truth, so only what the cleaner reports
(status, confidence, size, tests) and self-consistency are checked: the 5 box scans show one box, so the accepted ones
should agree on width / height (file units differ per file, so only the ratio is comparable). Names from
majproj/labels.py are shown for reading only.

CHECKS ON THE CLEANER ITSELF (on the first few sims and on every real file): does it modify its input, give the same
answer twice, give the same answer when the point order is shuffled (the file keeps no order), return only points
that were in the input, return the same count it reports.
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):      # one thread per worker process
    os.environ.setdefault(_v, '1')

import argparse
import glob
import hashlib
import html as htmllib
import json
import multiprocessing as mp
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

from majproj.labels import USER_NAMES

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
OUT = os.path.join(ROOT, 'outputs', 'clean_validation')
SEED = 20261028
GOOD_F1 = 0.7                      # a result with F1 >= this is called "good" in the status x quality table
BAD_F1 = 0.5                       # an accepted result below this is a "bad accept"
DIST_EDGES = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, np.inf]
SIZE_EDGES = [0, 50, 100, 200, 400, np.inf]
N_INVARIANCE = 20                  # sims that also get the shuffle / repeat checks
REAL_DIRS = [('inputs', None), ('inputs_box', 'Box')]


# ----------------------------------------------------------------------------------------------- scoring
def prf(tp, fp, fn):
    """precision, recall, F1, IoU; nothing to find and nothing found counts as perfect, otherwise empty = nan / 0."""
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    den = 2 * tp + fp + fn
    f1 = 2 * tp / den if den else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return p, r, f1, iou


def rows_in(u_tree, pts):
    """Index of each row of `pts` in the unique input coordinates, and whether the match is exact."""
    if len(pts) == 0:
        return np.empty(0, int), np.empty(0, bool)
    d, j = u_tree.query(np.asarray(pts, np.float32))
    return j, d == 0


def score(export, object_mask, obj):
    """Point-level score of `obj` (the cleaner's object points) against the true `object_mask` of `export`.
    Returns (record, pred_mask_over_unique, truth_mask_over_unique, unique_coords)."""
    export = np.asarray(export, np.float32)
    object_mask = np.asarray(object_mask, bool)
    u, inv = np.unique(export, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    truth = np.zeros(len(u), bool)
    np.logical_or.at(truth, inv, object_mask)
    every = np.ones(len(u), bool)
    np.logical_and.at(every, inv, object_mask)
    pred = np.zeros(len(u), bool)
    j, ok = rows_in(cKDTree(u), obj)
    pred[j[ok]] = True
    tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
    p, r, f1, iou = prf(tp, fp, fn)
    pm = pred[inv]                                                        # weighted by how often each coord occurs
    _, _, wf1, _ = prf(int((pm & object_mask).sum()), int((pm & ~object_mask).sum()), int((~pm & object_mask).sum()))
    rec = dict(tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, iou=iou, wf1=wf1, n_true=int(truth.sum()), n_pred=int(pred.sum()),
               foreign=int((~ok).sum()), dup_pred=int(len(j[ok]) - len(np.unique(j[ok]))),
               label_conflicts=int((truth & ~every).sum()))
    return rec, pred, truth, u


def selftest():
    """The scorer must give F1 = 1 for the true object, 0 for nothing, and exact counts for a known subset (with duplicate rows)."""
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(300, 3)).astype(np.float32)
    pts = np.concatenate([pts, pts[:50]])                       # 50 duplicated rows
    mask = np.zeros(len(pts), bool)
    mask[:100] = True
    mask[300:320] = True                                        # duplicates of masked rows are masked too
    full, part, none = score(pts, mask, pts[mask])[0], score(pts, mask, pts[:50])[0], score(pts, mask, np.empty((0, 3), np.float32))[0]
    shifted = score(pts, mask, pts[:50] + np.float32(1e-3))[0]
    ok = (full['f1'] == 1.0 and full['n_true'] == 100 and none['f1'] == 0.0 and none['fn'] == 100
          and (part['tp'], part['fp'], part['fn']) == (50, 0, 50) and shifted['foreign'] == 50 and shifted['tp'] == 0)
    if not ok:
        raise AssertionError(f'scorer self-test failed: {full} {part} {none} {shifted}')
    return True


def jsonable(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist()) if x.size <= 64 else f'array{list(x.shape)}'
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in list(x)[:32]]
    return repr(x)[:80]


def true_normal_z(q):
    """|z component| of the thinnest axis of the true object points (a frame sanity check: ~0 for an upright object
    seen from the side in a z-up frame)."""
    if len(q) < 10:
        return None
    _, _, vt = np.linalg.svd(q - q.mean(0), full_matrices=False)
    return float(abs(vt[2][2]))


# ----------------------------------------------------------------------------------------------- calling the cleaner
def apply_clean(points):
    """Run the cleaner on a private copy; returns (Cleaned | None, error text | None, seconds, input_was_modified)."""
    from majproj.clean import clean
    inp = np.array(points, np.float32, copy=True)
    ref = inp.copy()
    t0 = time.perf_counter()
    try:
        r, err = clean(inp), None
    except Exception as e:                                                # a crash is a finding, not a harness failure
        tb = e.__traceback__
        while tb.tb_next:
            tb = tb.tb_next
        r, err = None, f'{type(e).__name__}: {e} ({os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno})'
    return r, err, time.perf_counter() - t0, not np.array_equal(inp, ref)


def meta_view(r, err):
    """Normalised view of the cleaner's metadata, tolerant of missing optional keys."""
    if r is None:
        return dict(status='error', status_inferred=False, confidence=None, reason=err, shape_hint=None, tests_failed=[],
                    n_unique=None, n_object=0, width=None, height=None, normal=None, n_object_reported=None)
    m = r.meta
    n_obj = int(len(r.object))
    inferred = 'status' not in m
    status = m['status'] if not inferred else ('ok' if n_obj > 0 and not m.get('reason') else 'rejected')
    conf = m.get('confidence')
    tests = m.get('tests') or {}
    failed = [f"{k}={v.get('value')}" if isinstance(v, dict) else str(k) for k, v in tests.items()
              if isinstance(v, dict) and not v.get('pass', True)]
    return dict(status=status, status_inferred=inferred, confidence=None if conf is None else float(conf),
                reason=m.get('reason') or '', shape_hint=m.get('shape_hint'), tests_failed=failed,
                n_unique=m.get('n_unique'), n_object=n_obj, width=m.get('width'), height=m.get('height'),
                normal=m.get('normal'), n_object_reported=m.get('n_object'))


def invariance(points, r, rng, mask_fn):
    """Repeat and shuffle the cleaner's input; compare object sets. `mask_fn(obj)` -> bool mask over unique input rows."""
    base = mask_fn(r.object)
    r2, e2, _, _ = apply_clean(points)
    perm = rng.permutation(len(points))
    r3, e3, _, _ = apply_clean(np.asarray(points)[perm])
    out = dict(repeat_identical=None, shuffle_jaccard=None, shuffle_same_status=None, shuffle_error=e3, repeat_error=e2)
    if r2 is not None:
        m2 = mask_fn(r2.object)
        out['repeat_identical'] = bool(np.array_equal(base, m2) and meta_view(r2, None)['status'] == meta_view(r, None)['status'])
    if r3 is not None:
        m3 = mask_fn(r3.object)
        union = int((base | m3).sum())
        out['shuffle_jaccard'] = float((base & m3).sum() / union) if union else 1.0
        out['shuffle_same_status'] = bool(meta_view(r3, None)['status'] == meta_view(r, None)['status'])
    return out


def unique_mask_fn(points):
    u = np.unique(np.asarray(points, np.float32), axis=0)
    tree = cKDTree(u)

    def fn(obj):
        m = np.zeros(len(u), bool)
        j, ok = rows_in(tree, obj)
        m[j[ok]] = True
        return m
    return fn


# ----------------------------------------------------------------------------------------------- one sim scan / one real file
def run_sim(i, seed, keep=False):
    from majproj.sim_scan import simulate_scan
    rng = np.random.default_rng([seed, i])
    t0 = time.perf_counter()
    s = simulate_scan(rng)
    t_sim = time.perf_counter() - t0
    export = np.ascontiguousarray(s['export'], np.float32)
    mask = np.asarray(s['object_mask'], bool)
    truth = s.get('truth', {})
    r, err, t_clean, mutated = apply_clean(export)                        # the cleaner sees the bare export only
    mv = meta_view(r, err)
    if r is not None:
        sc, pred, tru, u = score(export, mask, r.object)
    else:
        sc, pred, tru, u = score(export, mask, np.empty((0, 3), np.float32))
    ok = mv['status'] == 'ok'
    dist = truth.get('distance_m')
    rec = dict(i=i, kind=str(truth.get('kind')), distance_m=None if dist is None else float(dist), n_raw=truth.get('n_raw'),
               phases=str(truth.get('phases'))[:80], t_sim=t_sim, t_clean=t_clean, mutated=bool(mutated), error=err,
               f1g=sc['f1'] if ok else 0.0, true_nz=true_normal_z(export[mask]), n_export_unique=int(len(u)),
               truth=jsonable({k: v for k, v in truth.items() if k not in ('motion',)}), **mv, **sc)
    rec['normal_z'] = None if not mv['normal'] else abs(float(mv['normal'][2]))
    src, names = truth.get('source'), truth.get('source_names')
    if src is not None and names is not None and len(src) == len(export):          # which simulated surface each point came from
        _, first = np.unique(export, axis=0, return_index=True)
        su = np.asarray(src)[first]
        fp = pred & ~tru
        rec['src_counts'] = {str(names[k]): int((su == k).sum()) for k in range(len(names))}
        rec['fp_src'] = {str(names[k]): int((fp & (su == k)).sum()) for k in range(len(names))}
    if i < N_INVARIANCE and r is not None:
        rec['inv'] = invariance(export, r, np.random.default_rng([seed, i, 1]), unique_mask_fn(export))
    if keep:
        rec['_arr'] = dict(u=u, pred=pred, truth=tru, complete=np.asarray(s.get('object_complete', np.empty((0, 3))), np.float32),
                           obj=None if r is None else r.object, front_pts=export[mask])
    return rec


def run_real(path, label):
    pts = np.load(path)
    r, err, t_clean, mutated = apply_clean(pts)
    mv = meta_view(r, err)
    u = np.unique(pts.astype(np.float32), axis=0)
    w, h = mv['width'], mv['height']
    nz = None if not mv['normal'] else abs(float(mv['normal'][2]))
    stem = os.path.basename(path)[:-len('.npy')]
    rec = dict(file=os.path.basename(path), folder=os.path.basename(os.path.dirname(path)), label=label or USER_NAMES.get(stem, '?'),
               raw_shape=list(pts.shape), n_distinct=int(len(u)), t_clean=t_clean, mutated=bool(mutated), error=err,
               wh=(w / h) if (w and h) else None, normal_z=nz, **mv)
    if r is not None:
        sel_tree = cKDTree(u)
        _, ok = rows_in(sel_tree, r.object)
        rec['foreign'] = int((~ok).sum())
        rec['removed_overlap_object'] = int(len(set(map(bytes, np.asarray(r.object, np.float32))) &
                                                set(map(bytes, np.asarray(r.removed, np.float32)))))
        rec['inv'] = invariance(pts, r, np.random.default_rng([SEED, 7]), unique_mask_fn(pts))
    return rec, r


# ----------------------------------------------------------------------------------------------- aggregation
def nm(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return float(np.mean(a)) if a else float('nan')


def nmed(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return float(np.median(a)) if a else float('nan')


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    if isinstance(v, (float, np.floating)):
        return f'{v:.{nd}f}'
    return str(v)


def pct(v):
    return '-' if v is None or not np.isfinite(v) else f'{100 * v:.1f}%'


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float('nan'), int(ok.sum())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def auroc(score_, pos):
    s, pos = np.asarray(score_, float), np.asarray(pos, bool)
    ok = np.isfinite(s)
    s, pos = s[ok], pos[ok]
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_row(name, rs):
    acc = [r for r in rs if r['status'] == 'ok']
    return [name, len(rs), pct(len(acc) / len(rs)) if rs else '-',
            fmt(nm([r['f1'] for r in acc])), fmt(nmed([r['f1'] for r in acc])), fmt(nm([r['p'] for r in acc])),
            fmt(nm([r['r'] for r in acc])), fmt(nm([r['f1g'] for r in rs])),
            pct(np.mean([r['f1'] >= GOOD_F1 for r in acc]) if acc else float('nan')),
            pct(np.mean([r['f1'] < BAD_F1 for r in acc]) if acc else float('nan')),
            fmt(nmed([r['confidence'] for r in rs]))]


GROUP_COLS = ['group', 'n', 'accept rate', 'F1 (accepted) mean', 'F1 (accepted) median', 'precision (accepted)',
              'recall (accepted)', 'F1 gated (rejected = 0)', f'accepted with F1 >= {GOOD_F1}', f'bad accepts (F1 < {BAD_F1})',
              'median confidence']


def analyse_sim(S):
    A = {}
    acc = [r for r in S if r['status'] == 'ok']
    rej = [r for r in S if r['status'] == 'rejected']
    err = [r for r in S if r['status'] == 'error']
    A['acc'], A['rej'], A['err'] = acc, rej, err
    A['accept_rate'] = len(acc) / len(S)
    good = [r['f1'] >= GOOD_F1 for r in S]
    A['conf_all'] = spear([r['confidence'] if r['confidence'] is not None else np.nan for r in S], [r['f1'] for r in S])
    A['conf_acc'] = spear([r['confidence'] if r['confidence'] is not None else np.nan for r in acc], [r['f1'] for r in acc])
    A['auroc'] = auroc([r['confidence'] if r['confidence'] is not None else np.nan for r in S], good)
    A['bad_accept_rate'] = float(np.mean([r['f1'] < BAD_F1 for r in acc])) if acc else float('nan')
    A['f1_acc'] = nm([r['f1'] for r in acc])
    A['f1_gated'] = nm([r['f1g'] for r in S])
    return A


def tables_sim(S, A, N):
    T = []
    acc, rej = A['acc'], A['rej']

    def row(name, rs):
        return [name, len(rs), fmt(nm([r['p'] for r in rs])), fmt(nm([r['r'] for r in rs])), fmt(nm([r['f1'] for r in rs])),
                fmt(nmed([r['f1'] for r in rs])), fmt(float(np.percentile([r['f1'] for r in rs], 10)) if rs else float('nan')),
                fmt(nm([r['iou'] for r in rs])), fmt(nm([r['wf1'] for r in rs]))]
    T.append(dict(title='SIM accuracy (unique export coordinates; mean over scans)',
                  cols=['set', 'n', 'precision', 'recall', 'F1', 'F1 median', 'F1 10th pct', 'IoU', 'F1 weighted by duplicates'],
                  rows=[row('accepted', acc), row('all scans, raw output (rejected scored on what it returned)', S),
                        ['all scans, gated (rejected = nothing delivered, F1 = 0)', len(S), '-', '-', fmt(A['f1_gated']),
                         fmt(nmed([r['f1g'] for r in S])), fmt(float(np.percentile([r['f1g'] for r in S], 10))), '-', '-']],
                  note=f'accept rate {pct(A["accept_rate"])} ({len(acc)} accepted, {len(rej)} rejected, {len(A["err"])} errors of {len(S)}). '
                       f'Precision / recall are means over scans where they are defined.'))
    qual = lambda rs, good_: sum(1 for r in rs if (r['f1'] >= GOOD_F1) == good_)
    T.append(dict(title=f'Status x quality (good = F1 of the returned object >= {GOOD_F1})',
                  cols=['', 'good result', 'poor result', 'total'],
                  rows=[['accepted', qual(acc, True), qual(acc, False), len(acc)],
                        ['rejected (a rejected scan normally returns nothing, so it cannot score as good)', qual(rej, True), qual(rej, False), len(rej)],
                        ['error', 0, len(A['err']), len(A['err'])]],
                  note='Accepted + poor = junk delivered to the model (the dangerous case). Whether the rejections were justified is not visible here: '
                       'see the tables by true object size and the largest rejected objects.'))
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]
    crow = []
    for lo, hi in bins:
        rs = [r for r in S if r['confidence'] is not None and lo <= r['confidence'] < hi]
        crow.append([f'{lo:.1f} - {min(hi, 1.0):.1f}', len(rs), pct(np.mean([r['status'] == 'ok' for r in rs])) if rs else '-',
                     fmt(nm([r['f1'] for r in rs])), pct(np.mean([r['f1'] >= GOOD_F1 for r in rs])) if rs else '-'])
    nconf = sum(1 for r in S if r['confidence'] is not None)
    T.append(dict(title='Confidence calibration', cols=['confidence', 'n', 'accepted', 'mean F1 (raw output)', f'share with F1 >= {GOOD_F1}'], rows=crow,
                  note=f'Spearman(confidence, F1): accepted only {fmt(A["conf_acc"][0])} (n={A["conf_acc"][1]}); all scans, raw F1 '
                       f'{fmt(A["conf_all"][0])} (n={A["conf_all"][1]}). AUROC of confidence for finding a good result '
                       f'(F1 >= {GOOD_F1}) across all scans: {fmt(A["auroc"])}. {nconf} of {len(S)} scans carried a confidence. '
                       'Spearman near 1 = higher confidence means better F1; near 0 = confidence is uninformative; negative = misleading.'))
    kinds = sorted({r['kind'] for r in S})
    T.append(dict(title='By object kind', cols=GROUP_COLS, rows=[group_row(k, [r for r in S if r['kind'] == k]) for k in kinds]))
    drows = []
    for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:]):
        rs = [r for r in S if r['distance_m'] is not None and lo <= r['distance_m'] < hi]
        drows.append(group_row(f'{lo * 100:.0f} - {hi * 100:.0f} cm' if np.isfinite(hi) else f'>= {lo * 100:.0f} cm', rs))
    T.append(dict(title='By object distance (truth, metres from the sensor)', cols=GROUP_COLS, rows=drows,
                  note='Cells with n < 10 are anecdotes, not rates.'))
    srows = []
    for lo, hi in zip(SIZE_EDGES[:-1], SIZE_EDGES[1:]):
        rs = [r for r in S if lo <= r['n_true'] < hi]
        srows.append(group_row(f'{lo} - {hi - 1} points' if np.isfinite(hi) else f'>= {lo} points', rs))
    T.append(dict(title='By true object size (distinct export points that belong to the object)', cols=GROUP_COLS, rows=srows,
                  note='A scan whose object has only a few dozen points in the export may be unrecoverable by any method; '
                       'a rejection there is not a failure. Rejections of scans with hundreds of object points are.'))
    return T


def tables_sources(S, A):
    with_src = [r for r in S if r.get('src_counts')]
    if not with_src:
        return []
    names = list(with_src[0]['src_counts'])
    acc = [r for r in with_src if r['status'] == 'ok']
    bad = [r for r in acc if r['f1'] < BAD_F1]
    tot_fp = sum(sum(r['fp_src'].values()) for r in acc) or 1
    rows = []
    for k in names:
        share = np.mean([r['src_counts'][k] / max(sum(r['src_counts'].values()), 1) for r in with_src])
        fp_k = sum(r['fp_src'][k] for r in acc)
        main = sum(1 for r in bad if sum(r['fp_src'].values()) and max(r['fp_src'], key=r['fp_src'].get) == k)
        rows.append([k, pct(share), fp_k, pct(fp_k / tot_fp), main])
    nzh = [r for r in A['acc'] if r.get('normal_z') is not None and r['normal_z'] > 0.5]
    return [dict(title='Where the false positives come from (simulator truth: which surface produced each point)',
                 cols=['source', 'mean share of an export', 'false-positive points in accepted scans', 'share of all false positives',
                       f'bad accepts (F1 < {BAD_F1}) where it supplies most false positives'], rows=rows,
                 note=f'{len(nzh)} of {len(A["acc"])} accepted sim scans returned a face whose normal is more than 60 degrees from horizontal '
                      f'(|normal z| > 0.5), i.e. a ceiling / table-like surface rather than an upright object. '
                      f'"hand" and "mixed" points are the simulator\'s own junk categories. The object is a small part of every export (first row).')]


def fail_rows(rs):
    return [[r['i'], r['kind'], fmt(r['distance_m'], 3), r['status'], fmt(r['confidence'], 2), fmt(r['p'], 2), fmt(r['r'], 2),
             fmt(r['f1'], 2), r['n_pred'], r['n_true'], (r['reason'] or '')[:70], r['phases'][:50]] for r in rs]


FAIL_COLS = ['scan #', 'kind', 'dist m', 'status', 'conf', 'precision', 'recall', 'F1', 'n returned', 'n true', 'reason', 'phases']


def tables_fail(S, A):
    acc, rej = A['acc'], A['rej']
    worst = sorted(acc, key=lambda r: r['f1'])[:8]
    big_rej = sorted(rej, key=lambda r: -r['n_true'])[:8]
    T = [dict(title='Lowest-F1 accepted scans', cols=FAIL_COLS, rows=fail_rows(worst)),
         dict(title='Rejected scans with the largest true object (possible false rejects)', cols=FAIL_COLS, rows=fail_rows(big_rej),
              note='Rejected scans normally return nothing, so n returned = 0 and F1 = 0 by construction; the question is whether the object was clearly visible (n true).')]
    errs = A['err']
    if errs:
        T.append(dict(title='Cleaner crashes', cols=['scan #', 'kind', 'error'], rows=[[r['i'], r['kind'], r['error']] for r in errs[:10]]))
    return T


def sanity_flags_sim(S, A, R=()):
    """(sanity table, flags). Flags are things a reader should look at before trusting the cleaner."""
    flags = []
    with_inv = [r for r in S if r.get('inv')]
    sj = [r['inv']['shuffle_jaccard'] for r in with_inv if r['inv']['shuffle_jaccard'] is not None]
    rep_bad = [r['i'] for r in with_inv if r['inv']['repeat_identical'] is False]
    shuf_bad = [r['i'] for r in with_inv if (r['inv']['shuffle_jaccard'] is not None and r['inv']['shuffle_jaccard'] < 0.99)
                or r['inv']['shuffle_same_status'] is False]
    foreign, dup_pred = sum(r['foreign'] for r in S), sum(r['dup_pred'] for r in S)
    conflicts = sum(r['label_conflicts'] for r in S)
    mutated = [r['i'] for r in S if r['mutated']]
    inc = [r['i'] for r in S if r['status'] != 'error' and r['n_object_reported'] is not None and r['n_object_reported'] != r['n_object']]
    nz = {k: nmed([r['true_nz'] for r in S if r['kind'] == k]) for k in sorted({r['kind'] for r in S})}
    inferred = any(r['status_inferred'] for r in S)
    rows = [['scorer self-test (true object -> F1 1, nothing -> 0, known subset -> exact counts, moved points -> foreign)', 'passed', ''],
            ['inputs modified by the cleaner', len(mutated), 'scans'], ['crashes', len(A['err']), 'scans'],
            ['returned points not present in the input', foreign, 'points'], ['duplicate points returned', dup_pred, 'points'],
            ['meta n_object != returned count', len(inc), 'scans'],
            [f'repeat run differs (first {N_INVARIANCE} sims)', len(rep_bad), f'of {len(with_inv)}'],
            [f'shuffled input changes the object set (Jaccard < 0.99 or status flips)', len(shuf_bad), f'of {len(with_inv)}'],
            ['shuffle Jaccard, mean / min', f'{fmt(nm(sj))} / {fmt(min(sj) if sj else float("nan"))}', ''],
            ['sim label conflicts (same coordinate masked both ways)', conflicts, 'coordinates'],
            ['distinct points per export, sim median (5th-95th pct) vs real median (min-max)',
             f'{np.median([r["n_export_unique"] for r in S]):.0f} ({np.percentile([r["n_export_unique"] for r in S], 5):.0f}-{np.percentile([r["n_export_unique"] for r in S], 95):.0f})'
             + (f' vs {np.median([r["n_distinct"] for r in R]):.0f} ({min(r["n_distinct"] for r in R)}-{max(r["n_distinct"] for r in R)})' if len(R) else ''),
             'of 4096 rows; a gap is a sim-vs-real domain gap'],
            ['median |normal z| of the TRUE object points, ' + ', '.join(f'{k}: {fmt(v, 2)}' for k, v in nz.items()), '', 'frame check: small = z is up']]
    T = dict(title='Sanity checks on the cleaner and on the simulator', cols=['check', 'value', 'unit'], rows=rows,
             note='Zero is the expected value for every count in this table.')
    if A['err']:
        flags.append(f'{len(A["err"])} of {len(S)} sim scans crashed the cleaner (first: {A["err"][0]["error"]}).')
    if mutated:
        flags.append(f'The cleaner modified its input array in {len(mutated)} scans (scans {mutated[:5]}).')
    if foreign:
        flags.append(f'{foreign} returned points were not in the input (the cleaner moved or invented points; matching is exact).')
    if dup_pred:
        flags.append(f'{dup_pred} duplicate points in the returned object sets.')
    if inc:
        flags.append(f'meta["n_object"] disagrees with the returned object size in {len(inc)} scans.')
    if rep_bad:
        flags.append(f'Not deterministic: repeating the call changed the result in scans {rep_bad[:5]}.')
    if shuf_bad:
        flags.append(f'Depends on the point order (the file keeps no order): shuffling changed the object set in {len(shuf_bad)} of {len(with_inv)} scans {shuf_bad[:5]}.')
    horiz = [r['i'] for r in A['acc'] if r.get('normal_z') is not None and r['normal_z'] > 0.5]
    if A['acc'] and len(horiz) > 0.05 * len(A['acc']):
        flags.append(f'{len(horiz)} of {len(A["acc"])} accepted sim scans return a face whose normal is more than 60 degrees from horizontal '
                     f'(|nz| > 0.5): a ceiling / table-like surface accepted as the object.')
    if conflicts:
        flags.append(f'Simulator: {conflicts} coordinates are labelled both object and not-object (scoring takes "object" for these).')
    if any(v is not None and np.isfinite(v) and v > 0.5 for v in nz.values()):
        flags.append('Frame suspicion: the true object points have a large vertical normal component, so the simulator frame may not be z-up.')
    if inferred:
        flags.append('The cleaner does not return meta["status"]: status was inferred from the returned point count (old cleaner?).')
    if not np.isfinite(A['conf_acc'][0]):
        flags.append('Calibration could not be computed: confidence is missing or constant over the accepted scans.')
    elif A['conf_acc'][0] < 0.2:
        flags.append(f'Confidence is a weak predictor of F1 among accepted scans (Spearman {A["conf_acc"][0]:.2f}).')
    confs = [r['confidence'] for r in S if r['confidence'] is not None]
    if confs and (min(confs) < 0 or max(confs) > 1):
        flags.append(f'Confidence outside 0..1 (min {min(confs):.3f}, max {max(confs):.3f}).')
    if confs and len(set(np.round(confs, 3))) <= 3:
        flags.append('Confidence takes at most 3 distinct values over all sim scans: it is not graded.')
    if len(A['acc']) and A['bad_accept_rate'] > 0.10:
        flags.append(f'{pct(A["bad_accept_rate"])} of accepted sim scans are bad (F1 < {BAD_F1}): the accept gate lets junk through.')
    if A['accept_rate'] < 0.5:
        flags.append(f'Only {pct(A["accept_rate"])} of sim scans are accepted.')
    kr = {k: np.mean([r['status'] == 'ok' for r in S if r['kind'] == k]) for k in {r['kind'] for r in S}}
    if len(kr) > 1 and max(kr.values()) - min(kr.values()) > 0.3:
        flags.append('Accept rate differs by more than 30 points between kinds: ' + ', '.join(f'{k} {pct(v)}' for k, v in sorted(kr.items())) + '.')
    return T, flags


def tables_real(R):
    rows = []
    for r in R:
        n_obj = r['n_object']
        rows.append([r['folder'] + '/' + r['file'][5:20], r['label'], r['status'], r['n_distinct'], n_obj, fmt(r['confidence'], 2),
                     r['shape_hint'] or '-', fmt(r['width'], 3), fmt(r['height'], 3), fmt(r['wh'], 2), fmt(r['normal_z'], 2),
                     (r['reason'] or '')[:60], ', '.join(r['tests_failed'])[:60]])
    T = [dict(title='REAL files: what the cleaner reports (labels are the user\'s, for reading only)',
              cols=['file', 'label', 'status', 'distinct points', 'object points', 'confidence', 'shape hint', 'width', 'height', 'w / h',
                    '|normal z|', 'reason', 'failed tests'], rows=rows,
              note='Width and height are in file units (each file is normalised on its own), so only w / h is comparable between files. '
                   'There is no ground truth for the real files: this table says what the cleaner claims, not that it is right.')]
    box = [r for r in R if r['folder'] == 'inputs_box']
    bacc = [r for r in box if r['status'] == 'ok' and r['wh']]
    brows = [[r['file'][5:20], r['status'], r['n_object'], fmt(r['confidence'], 2), fmt(r['width'], 3), fmt(r['height'], 3), fmt(r['wh'], 3),
              r['shape_hint'] or '-'] for r in box]
    cv = lambda a: float(np.std(a, ddof=1) / np.mean(a)) if len(a) > 1 and np.mean(a) else float('nan')
    whs = [r['wh'] for r in bacc]
    note = f'{len(bacc)} of {len(box)} box scans accepted. '
    if len(whs) > 1:
        note += (f'w / h over the accepted: mean {np.mean(whs):.3f}, sd {np.std(whs, ddof=1):.3f}, CV {100 * cv(whs):.1f}%, '
                 f'range {min(whs):.3f} - {max(whs):.3f}. Width CV {100 * cv([r["width"] for r in bacc]):.1f}%, height CV '
                 f'{100 * cv([r["height"] for r in bacc]):.1f}% (file units, which differ per file, so these two are only indicative). '
                 'The same box was scanned each time, but the scans may differ in distance, angle and which face was seen, so a spread in w / h '
                 'is not by itself proof that the cleaner is wrong; a very small spread is evidence that it is stable.')
    T.append(dict(title='Box consistency (inputs_box: one box, scanned 5 times)', cols=['scan', 'status', 'object points', 'confidence', 'width', 'height', 'w / h', 'shape hint'],
                  rows=brows, note=note))
    return T, cv(whs), len(bacc), len(box)


def flags_real(R, cv_box, n_bacc, n_box):
    flags = []
    for r in R:
        if r['status'] == 'error':
            flags.append(f'REAL {r["file"]}: the cleaner crashed ({r["error"]}).')
        if r['status'] == 'ok' and r['normal_z'] is not None and r['normal_z'] > 0.5:
            flags.append(f'REAL {r["file"]}: accepted although the face normal is far from horizontal (|nz| {r["normal_z"]:.2f}).')
        if r['status'] == 'ok' and r['n_object'] == 0:
            flags.append(f'REAL {r["file"]}: status ok but zero object points.')
        if r['status'] == 'rejected' and r['n_object'] > 0:
            flags.append(f'REAL {r["file"]}: rejected but {r["n_object"]} object points were still returned (check the consumer does not use them).')
        if r.get('foreign'):
            flags.append(f'REAL {r["file"]}: {r["foreign"]} returned points are not in the file.')
        if r.get('removed_overlap_object'):
            flags.append(f'REAL {r["file"]}: {r["removed_overlap_object"]} points appear in both object and removed.')
        if r['mutated']:
            flags.append(f'REAL {r["file"]}: the cleaner modified its input.')
        iv = r.get('inv') or {}
        if iv.get('repeat_identical') is False:
            flags.append(f'REAL {r["file"]}: not deterministic (repeat run differs).')
        if (iv.get('shuffle_jaccard') is not None and iv['shuffle_jaccard'] < 0.99) or iv.get('shuffle_same_status') is False:
            flags.append(f'REAL {r["file"]}: result depends on point order (shuffle Jaccard {fmt(iv.get("shuffle_jaccard"))}, same status {iv.get("shuffle_same_status")}).')
        if r['status_inferred']:
            flags.append(f'REAL {r["file"]}: no meta["status"] (inferred).')
    nrej = sum(1 for r in R if r['status'] != 'ok')
    if nrej:
        flags.append(f'{nrej} of {len(R)} real files were not accepted: ' + ', '.join(f'{r["folder"]}/{r["file"][5:20]} ({r["label"]})' for r in R if r['status'] != 'ok') + '.')
    if np.isfinite(cv_box) and cv_box > 0.10:
        flags.append(f'Box w / h varies by CV {100 * cv_box:.1f}% over {n_bacc} accepted scans of the same box.')
    if n_bacc < n_box:
        flags.append(f'Only {n_bacc} of {n_box} box scans accepted.')
    return flags


# ----------------------------------------------------------------------------------------------- tables -> markdown / html
def md_table(t):
    esc = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')
    out = [f'### {t["title"]}', '', '| ' + ' | '.join(esc(c) for c in t['cols']) + ' |', '|' + '|'.join(['---'] * len(t['cols'])) + '|']
    out += ['| ' + ' | '.join(esc(c) for c in r) + ' |' for r in t['rows']]
    if not t['rows']:
        out.append('| ' + ' | '.join(['(none)'] + [''] * (len(t['cols']) - 1)) + ' |')
    if t.get('note'):
        out += ['', t['note']]
    return '\n'.join(out) + '\n'


def html_table(t):
    e = lambda x: htmllib.escape(str(x))
    head = ''.join(f'<th>{e(c)}</th>' for c in t['cols'])
    body = ''.join('<tr>' + ''.join(f'<td>{e(c)}</td>' for c in r) + '</tr>' for r in t['rows']) or f'<tr><td colspan="{len(t["cols"])}">(none)</td></tr>'
    note = f'<p class="mut">{e(t["note"])}</p>' if t.get('note') else ''
    return f'<h3>{e(t["title"])}</h3><div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}'


# ----------------------------------------------------------------------------------------------- 3D cards
def _round(a):
    return np.round(np.asarray(a, np.float64), 4).T.tolist() if len(a) else [[], [], []]


def _cube(pts, pad=0.6):
    pts = np.asarray(pts, np.float64)
    lo, hi = pts.min(0), pts.max(0)
    c, half = (lo + hi) / 2, max(pad * float(np.max(hi - lo)), 1e-3)
    return [[float(c[k] - half), float(c[k] + half)] for k in range(3)]


def _front(pts):
    """Unit horizontal direction along which the points are thinnest (the face normal of a flat, upright object)."""
    xy = np.asarray(pts, np.float64)[:, :2]
    if len(xy) < 3:
        return [0.0, -1.0]
    w, v = np.linalg.eigh(np.cov(xy.T))
    f = v[:, 0]
    if f[1] > 0:
        f = -f
    return [float(f[0]), float(f[1])]


def _pct_box(pts, lo=1, hi=99):
    pts = np.asarray(pts, np.float64)
    return np.array([np.percentile(pts, lo, axis=0), np.percentile(pts, hi, axis=0)])


def _badge(status):
    return dict(ok=('ACCEPTED', 'ok'), rejected=('REJECTED', 'rej')).get(status, ('ERROR', 'err'))


def real_card(rec, r):
    obj = np.asarray(r.object, np.float32) if r is not None else np.empty((0, 3), np.float32)
    rem = np.asarray(r.removed, np.float32) if r is not None else np.empty((0, 3), np.float32)
    if len(obj):
        focus, front = obj, _front(obj)
    else:
        focus = _pct_box(rem) if len(rem) else np.zeros((2, 3))
        front = _front(rem) if len(rem) else [0.0, -1.0]
    txt, cls = _badge(rec['status'])
    bits = [f'{rec["n_object"]} of {rec["n_distinct"]} distinct points kept']
    if rec['shape_hint']:
        bits.append(f'shape: {rec["shape_hint"]}')
    if rec['wh']:
        bits.append(f'w {rec["width"]:.3f} / h {rec["height"]:.3f} = {rec["wh"]:.2f}')
    if rec['tests_failed']:
        bits.append('failed tests: ' + ', '.join(rec['tests_failed']))
    return dict(id=f'real-{rec["folder"]}-{rec["file"][5:20]}', title=f'{rec["label"]} · {rec["folder"]}/{rec["file"]}',
                badge=txt, cls=cls, conf=rec['confidence'], reason=rec['reason'] or '', sub=' · '.join(bits), front=front, rng=_cube(focus),
                traces=[dict(name=f'object ({len(obj)})', color='#ea580c', size=2.6, vis=True, xyz=_round(obj)),
                        dict(name=f'removed ({len(rem)})', color='#9aa0a6', size=1.5, vis=True if not len(obj) else 'legendonly', xyz=_round(rem))])


def sim_card(rec, why):
    a = rec['_arr']
    u, pred, truth = a['u'], a['pred'], a['truth']
    tp, fp, fn, other = u[pred & truth], u[pred & ~truth], u[~pred & truth], u[~pred & ~truth]
    focus_pts = u[pred | truth]
    comp = a['complete']
    if len(comp) > 4000:
        comp = comp[np.random.default_rng(0).choice(len(comp), 4000, replace=False)]
    txt, cls = _badge(rec['status'])
    d = rec['distance_m']
    sub = (f'{why} · true object {rec["n_true"]} pts, returned {rec["n_pred"]} · precision {fmt(rec["p"], 2)}, recall {fmt(rec["r"], 2)}, '
           f'F1 {rec["f1"]:.2f}' + (f' · distance {d * 100:.0f} cm' if d is not None else ''))
    return dict(id=f'sim-{rec["i"]}', title=f'sim #{rec["i"]} · {rec["kind"]}', badge=txt, cls=cls, conf=rec['confidence'],
                reason=rec['reason'] or '', sub=sub, front=_front(a['front_pts']) if len(a['front_pts']) else [0, -1], rng=_cube(focus_pts),
                traces=[dict(name=f'true positive ({len(tp)})', color='#1b9e77', size=2.8, vis=True, xyz=_round(tp)),
                        dict(name=f'false positive ({len(fp)})', color='#d62728', size=3.4, vis=True, xyz=_round(fp)),
                        dict(name=f'false negative ({len(fn)})', color='#1f77b4', size=3.4, vis=True, xyz=_round(fn)),
                        dict(name=f'other points ({len(other)})', color='#9aa0a6', size=1.5, vis='legendonly', xyz=_round(other)),
                        dict(name='true surface, complete (sampled)', color='#c28f00', size=1.2, vis='legendonly', xyz=_round(comp), opacity=0.35)])


def pick_cards(S, k=8):
    acc = sorted([r for r in S if r['status'] == 'ok'], key=lambda r: r['f1'])
    rej = [r for r in S if r['status'] != 'ok']
    picks, used = [], set()

    def add(r, why):
        if r is not None and r['i'] not in used and len(picks) < k:
            used.add(r['i'])
            picks.append((r, why))
    if acc:
        med = float(np.median([r['f1'] for r in acc]))
        seen_kinds = set()
        for r in sorted(acc, key=lambda r: abs(r['f1'] - med)):                      # typical, different kinds first
            if r['kind'] not in seen_kinds and len([1 for _, w in picks if w == 'typical accepted']) < 3:
                seen_kinds.add(r['kind'])
                add(r, 'typical accepted')
        for r in sorted(acc, key=lambda r: abs(r['f1'] - med)):
            if len([1 for _, w in picks if w == 'typical accepted']) < 3:
                add(r, 'typical accepted')
        for r in acc[:3]:
            add(r, 'worst accepted')
    big = sorted(rej, key=lambda r: -r['n_true'])
    add(big[0] if big else None, 'rejected, large true object')
    add(big[-1] if big else None, 'rejected, small true object')
    for r in acc[3:]:                                                                # fill up with the next-worst accepted
        add(r, 'next-worst accepted')
    for r in big[1:]:
        add(r, 'rejected')
    return picks


# ----------------------------------------------------------------------------------------------- driver
def git_rev():
    try:
        return subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return '?'


def file_sig(rel):
    p = os.path.join(ROOT, rel)
    return f'{rel} sha256 {hashlib.sha256(open(p, "rb").read()).hexdigest()[:10]}' if os.path.exists(p) else f'{rel} (missing)'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--n', type=int, default=200, help='number of simulated scans')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--workers', type=int, default=min(12, os.cpu_count() or 1))
    ap.add_argument('--cards', type=int, default=8, help='number of simulated scans drawn as 3D cards')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    T0 = time.perf_counter()
    selftest()
    sig0 = (file_sig('majproj/clean.py'), file_sig('majproj/sim_scan.py'))

    # ---- SIM
    print(f'SIM: {a.n} scans, seed {a.seed}, {a.workers} workers')
    ts = time.perf_counter()
    if a.workers > 1 and a.n > 1:
        with ProcessPoolExecutor(max_workers=min(a.workers, a.n), mp_context=mp.get_context('fork')) as ex:
            S = list(ex.map(run_sim, range(a.n), [a.seed] * a.n, chunksize=2))
    else:
        S = [run_sim(i, a.seed) for i in range(a.n)]
    t_sim_wall = time.perf_counter() - ts
    A = analyse_sim(S)
    print(f'  done in {t_sim_wall:.0f}s: accept {pct(A["accept_rate"])}, F1 accepted {fmt(A["f1_acc"])}, Spearman {fmt(A["conf_acc"][0])}')

    # ---- REAL
    R, real_r = [], []
    for folder, label in REAL_DIRS:
        for f in sorted(glob.glob(os.path.join(ROOT, folder, 'scan_*_shapenet.npy'))):
            rec, r = run_real(f, label)
            R.append(rec)
            real_r.append(r)
            print(f'  real {folder}/{rec["file"][5:20]} {rec["label"]:<12} {rec["status"]:<8} obj {rec["n_object"]:>4} conf {fmt(rec["confidence"], 2)} '
                  f'w/h {fmt(rec["wh"], 2)} {rec["reason"][:50] if rec["reason"] else ""}')

    # ---- cards (re-run the chosen sims: deterministic, so the stored metrics must reproduce exactly)
    picks = pick_cards(S, a.cards)
    cards_sim, repro_bad = [], []
    for rec, why in picks:
        again = run_sim(rec['i'], a.seed, keep=True)
        if abs(again['f1'] - rec['f1']) > 1e-12 or again['status'] != rec['status']:
            repro_bad.append(rec['i'])
        cards_sim.append(sim_card(again, why))
    cards_real = [real_card(rec, r) for rec, r in zip(R, real_r)]

    # ---- tables and flags
    Tsim = tables_sim(S, A, a.n)
    Tfail = tables_fail(S, A)
    Tsrc = tables_sources(S, A)
    Tsan, flags = sanity_flags_sim(S, A, R)
    Treal, cv_box, n_bacc, n_box = tables_real(R)
    flags += flags_real(R, cv_box, n_bacc, n_box)
    if repro_bad:
        flags.append(f'Not reproducible: re-running sim scans {repro_bad} with the same seed gave a different result.')
    real_acc = sum(1 for r in R if r['status'] == 'ok')
    if np.isfinite(A['accept_rate']) and R and abs(A['accept_rate'] - real_acc / len(R)) > 0.3:
        flags.append(f'Sim accept rate {pct(A["accept_rate"])} vs real accept rate {pct(real_acc / len(R))}: the simulator and the real files disagree by more than 30 points.')
    if sig0 != (file_sig('majproj/clean.py'), file_sig('majproj/sim_scan.py')):
        flags.insert(0, 'majproj/clean.py or majproj/sim_scan.py changed on disk while this validation ran: numbers may mix two versions, re-run.')
    tcl = np.array([r['t_clean'] for r in S])
    tsm = np.array([r['t_sim'] for r in S])
    run_info = dict(title='Run', cols=['item', 'value'], rows=[
        ['date', time.strftime('%Y-%m-%d %H:%M:%S')], ['repo commit', git_rev()], ['cleaner', sig0[0]],
        ['simulator', sig0[1]], ['sim scans', f'{a.n}, seed {a.seed}, per-scan rng default_rng([seed, i])'],
        ['clean() time per scan (s)', f'mean {tcl.mean():.2f}, p95 {np.percentile(tcl, 95):.2f}, max {tcl.max():.2f}'],
        ['simulate_scan() time per scan (s)', f'mean {tsm.mean():.2f}, max {tsm.max():.2f}'],
        ['wall clock', f'sim stage {t_sim_wall:.0f}s with {a.workers} workers; total {time.perf_counter() - T0:.0f}s']])
    sim_tables = Tsim + Tsrc + Tfail
    ordered = sim_tables + [Tsan] + Treal

    # ---- headlines
    box_line = (f'{n_bacc} of {n_box} box scans accepted; w / h CV {100 * cv_box:.1f}% over the accepted ones' if n_bacc > 1
                else f'{n_bacc} of {n_box} box scans accepted')
    head = [f'SIM ({a.n} scans): accept rate {pct(A["accept_rate"])}; mean F1 of accepted {fmt(A["f1_acc"])} '
            f'(precision {fmt(nm([r["p"] for r in A["acc"]]))}, recall {fmt(nm([r["r"] for r in A["acc"]]))}); '
            f'bad accepts (F1 < {BAD_F1}) {pct(A["bad_accept_rate"])} of accepted; gated F1 over all scans {fmt(A["f1_gated"])}.',
            f'Calibration: Spearman(confidence, F1) {fmt(A["conf_acc"][0])} on accepted scans, {fmt(A["conf_all"][0])} on all; AUROC {fmt(A["auroc"])}.',
            f'REAL: {real_acc} of {len(R)} files accepted; {box_line}.']
    md = ['# Cleaner validation (`majproj.clean`)', '', 'Generated by `scripts/validate_clean.py`. Simulated scans have known object points; '
          'real scans do not, so the REAL half only checks what the cleaner reports and whether it is consistent. '
          'The simulator is a model of the device, not the device: good sim numbers show the cleaner does what it was designed to do '
          'on scans like the simulator produces, not that it works on every real scan.', '', '## Headlines', '']
    md += [f'- {h}' for h in head]
    md += ['', '## Things to look at', ''] + ([f'- {f}' for f in flags] or ['- nothing flagged by the automatic checks']) + ['']
    md += ['## Run', '', md_table(run_info).split('\n', 2)[2]]
    md += ['## Simulated scans', ''] + [md_table(t) for t in sim_tables] + ['## Sanity', '', md_table(Tsan)]
    md += ['## Real scans', ''] + [md_table(t) for t in Treal]
    md += ['## Method notes', '',
           f'- Scoring unit: unique export coordinate. TP = returned and masked object; FP = returned, not masked; FN = masked, not returned. '
           f'A scan counts as "good" at F1 >= {GOOD_F1}, as a bad accept at F1 < {BAD_F1} (arbitrary, stated thresholds).',
           '- Rejected scans: "raw" F1 uses whatever `Cleaned.object` held (empty = 0); "gated" F1 is 0 for every rejected scan.',
           '- Leak-proof: only a copy of `export` reaches `clean()`, with default arguments. Masks, kinds and distances are used only to score.',
           '- Real files: no ground truth. Labels are shown for reading and never used in any check except choosing the box group (folder name).',
           '- The 3D page (`index.html`) shows each real file (object orange, removed grey, hidden by default unless nothing was kept) and '
           f'{len(cards_sim)} sim scans (green true positive, red false positive, blue false negative).', '']
    with open(os.path.join(a.out, 'report.md'), 'w') as fh:
        fh.write('\n'.join(md))

    # ---- json dumps
    def clean_rec(r):
        return {k: jsonable(v) for k, v in r.items() if not k.startswith('_')}
    json.dump([clean_rec(r) for r in S], open(os.path.join(a.out, 'sim_results.json'), 'w'), indent=1)
    json.dump([clean_rec(r) for r in R], open(os.path.join(a.out, 'real_results.json'), 'w'), indent=1)

    # ---- html
    tables_html = ''.join(html_table(t) for t in [run_info] + ordered)
    flags_html = '<ul>' + ''.join(f'<li>{htmllib.escape(f)}</li>' for f in flags) + '</ul>' if flags else '<p>nothing flagged by the automatic checks</p>'
    head_html = '<ul>' + ''.join(f'<li>{htmllib.escape(h)}</li>' for h in head) + '</ul>'
    data = json.dumps(dict(real=cards_real, sim=cards_sim)).replace('</', '<\\/')
    page = (TEMPLATE.replace('/*HEAD*/', head_html).replace('/*FLAGS*/', flags_html).replace('/*TABLES*/', tables_html)
            .replace('/*DATA*/', data).replace('/*PLOTLY*/', open(PLOTLY).read()))
    with open(os.path.join(a.out, 'index.html'), 'w') as fh:
        fh.write(page)
    print(f'-> {os.path.join(a.out, "report.md")}\n-> {os.path.join(a.out, "index.html")} ({os.path.getsize(os.path.join(a.out, "index.html")) / 1e6:.1f} MB)'
          f'\ntotal {time.perf_counter() - T0:.0f}s')
    return S, R


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cleaner validation</title><style>
:root{--bg:#fbfbfa;--fg:#1b1d21;--mut:#5b6068;--card:#fff;--line:#d9dbe0;--ok:#146c43;--rej:#8a5a00;--err:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#a0a6af;--card:#1c1f24;--line:#33383f;--ok:#5fd39a;--rej:#f0b44c;--err:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}h1{font-size:1.5rem;margin:.2em 0}h2{font-size:1.2rem;margin:1.6em 0 .4em}h3{font-size:1rem;margin:1.2em 0 .3em}
p{max-width:110ch}.mut{color:var(--mut)}code{font-size:.85em}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:.82rem}th,td{padding:3px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}.card h2{font-size:1rem;margin:0 0 2px}
.hd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.85rem}.badge{font-weight:700;font-size:.75rem;padding:1px 8px;border-radius:99px;border:1.5px solid currentColor}
.ok{color:var(--ok)}.rej{color:var(--rej)}.err{color:var(--err)}
.bar{display:inline-block;width:70px;height:8px;border:1px solid var(--line);border-radius:4px;vertical-align:middle;overflow:hidden}.bar i{display:block;height:100%;background:var(--fg);opacity:.55}
.sub{font-size:.78rem;color:var(--mut);margin-top:2px}.plot{height:460px;touch-action:none}
.views{margin:6px 0 0;display:flex;gap:6px;flex-wrap:wrap}
.views button{font:inherit;font-size:.8rem;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Cleaner validation</h1>
<p class="mut">Checks of <code>majproj.clean</code>: simulated scans with known object points, and the 15 real files. Same content as <code>report.md</code>.
The simulator is a model of the device, not the device, and the real files have no ground truth: read the real cards as "what the cleaner claims", then look.</p>
<h2>Headlines</h2>/*HEAD*/
<h2>Things to look at</h2>/*FLAGS*/
<h2>Tables</h2>/*TABLES*/
<h2>Real scans in 3D</h2>
<p class="mut">Orange = the object the cleaner kept. Grey = everything it removed (hidden by default; click "removed" in the legend; shown when nothing was kept).
z is up. Each plot is zoomed on the object with true proportions (file units; each file is normalised on its own). Drag to turn around the vertical axis, scroll to zoom.</p>
<div class="grid" id="real"></div>
<h2>Simulated scans in 3D</h2>
<p class="mut">Green = true positive, red = false positive (returned but not object), blue = false negative (object, not returned). Grey (other points) and the complete true
surface are hidden by default; click them in the legend. The zoom covers the true object and what was returned, so junk far from the object widens the view.</p>
<div class="grid" id="sim"></div></main>
<script>
const D = /*DATA*/;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e6e8eb' : '#1b1d21', gridc = dark ? '#33383f' : '#d9dbe0';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const UP = {x: 0, y: 0, z: 1};
function views(f) {
  const R = 2.3, p = [-f[1], f[0]];
  return {'3D': {eye: {x: 1.5, y: -1.7, z: 0.9}, up: UP},
    'Front': {eye: {x: R * f[0], y: R * f[1], z: 0.05}, up: UP},
    'Top': {eye: {x: 0.01 * f[0], y: 0.01 * f[1], z: 2.4}, up: {x: -f[0], y: -f[1], z: 0}},
    'Side': {eye: {x: R * p[0], y: R * p[1], z: 0.05}, up: UP}};
}
const ax = (t, r) => ({title: t, range: r, color: fg, gridcolor: gridc, zerolinecolor: gridc, backgroundcolor: 'rgba(0,0,0,0)'});
function layout(c, cam) {
  return {margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: 'rgba(0,0,0,0)', legend: {x: 0, y: 1, font: {color: fg, size: 11}},
    scene: {aspectmode: 'data', dragmode: 'turntable', xaxis: ax('x', c.rng[0]), yaxis: ax('y', c.rng[1]), zaxis: ax('z (up)', c.rng[2]), camera: cam}};
}
// Chrome allows ~16 live WebGL contexts and there are more cards than that: draw a card only while it is near the viewport.
function mount(c, host) {
  const V = views(c.front);
  const el0 = document.createElement('div'); el0.className = 'card'; el0.id = c.id;
  const conf = c.conf === null || c.conf === undefined ? '' : `<span>confidence ${c.conf.toFixed(2)} <span class="bar"><i style="width:${Math.round(100 * Math.max(0, Math.min(1, c.conf)))}%"></i></span></span>`;
  el0.innerHTML = `<h2>${esc(c.title)}</h2><div class="hd"><span class="badge ${c.cls}">${c.badge}</span>${conf}<span class="mut">${esc(c.reason)}</span></div>
    <div class="sub">${esc(c.sub)}</div><div class="views">${Object.keys(V).map(v => `<button data-v="${v}">${v}</button>`).join('')}</div><div class="plot"></div>`;
  host.appendChild(el0);
  const el = el0.querySelector('.plot');
  let saved = null, live = false;
  const draw = () => {
    if (live) return; live = true;
    const T = c.traces.map((t, k) => ({type: 'scatter3d', mode: 'markers', name: t.name, x: t.xyz[0], y: t.xyz[1], z: t.xyz[2], hoverinfo: 'skip',
      marker: {size: t.size, color: t.color, opacity: t.opacity || 1}, visible: saved ? saved.vis[k] : t.vis}));
    Plotly.newPlot(el, T, layout(c, saved ? saved.cam : V['3D']), {responsive: true, displaylogo: false});
  };
  const drop = () => {
    if (!live) return; live = false;
    saved = {cam: el._fullLayout.scene.camera, vis: el.data.map(t => t.visible)};
    Plotly.purge(el);
  };
  el0.querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => { if (live) Plotly.relayout(el, {'scene.camera': V[b.dataset.v]}); else saved = {cam: V[b.dataset.v], vis: c.traces.map(t => t.vis)}; }));
  new IntersectionObserver(es => es.forEach(e => e.isIntersecting ? draw() : drop()), {rootMargin: '500px'}).observe(el0);
}
D.real.forEach(c => mount(c, document.getElementById('real')));
D.sim.forEach(c => mount(c, document.getElementById('sim')));
</script></body></html>
"""

if __name__ == '__main__':
    main()
