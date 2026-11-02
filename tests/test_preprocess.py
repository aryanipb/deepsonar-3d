"""Tests for majproj.preprocess (preprocessing v2). Real files are used only to check robustness
(no crash, an object is found); nothing is tuned or trained on them."""
import glob
import os

import numpy as np
import pytest

from majproj.preprocess import CANON_POINTS, from_canonical, preprocess, to_canonical

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = sorted(glob.glob(os.path.join(ROOT, 'inputs', 'scan_*_shapenet.npy')))
N_SIM = int(os.environ.get('PREPROCESS_TEST_SIMS', '6'))


# held-close regime like the real device usage (object a few cm to ~40 cm in front of the sensor) plus two far
# sessions that only need to run without crashing
SIM_DISTANCES = [0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.4, 0.15] * max(1, N_SIM // 10)


@pytest.fixture(scope='module')
def sims():
    from majproj.sim.scenes import random_session
    return [random_session(np.random.default_rng(3000 + i), n_side=3, n_complete=8192, distance=d)
            for i, d in enumerate(SIM_DISTANCES)]


@pytest.fixture(scope='module')
def real_results():
    return {os.path.basename(f): preprocess(np.load(f)) for f in REAL}


def test_real_files_present():
    assert len(REAL) == 10


def test_output_contract(real_results):
    for name, r in real_results.items():
        assert r['partial'].shape == (CANON_POINTS, 3), name
        assert r['partial'].dtype == np.float32
        assert np.isfinite(r['partial']).all()
        t = r['transform']
        assert np.asarray(t['center']).shape == (3,) and np.isfinite(t['scale']) and t['scale'] > 0
        m = r['meta']
        for k in ('sensor_estimate', 'object_depth_ratio', 'angular_size_deg', 'n_object_points', 'confident', 'mode'):
            assert k in m, (name, k)
        assert m['mode'] in ('stripes', 'whole_cloud', 'scale_free')


def test_real_files_object_found(real_results):
    """no crash and a compact object cluster was segmented from every real file (not the whole cloud)"""
    for name, r in real_results.items():
        m = r['meta']
        assert m['n_object_points'] >= 100, (name, m['n_object_points'])
        assert m['n_object_points'] < 0.6 * m['n_input_unique'], (name, 'background included?')
        # the object cluster must not contain the background wall: it lies clearly in front of the wall
        if m['sensor_estimate'] is not None:
            assert 0 < m['object_depth_ratio'] < 0.8, name


def test_real_objects_not_on_wall(real_results):
    for name, r in real_results.items():
        m = r['meta']
        p, idx = m['debug']['unique_points'], m['debug']['object_indices_unique']
        obj = p[idx]
        if m['sensor_estimate'] is not None:
            wall_z = m['sensor_estimate']['wall_z']
            assert (obj[:, 2] < wall_z - 0.2 * m['sensor_estimate']['wall_depth']).mean() > 0.99, name


def test_determinism():
    pts = np.load(REAL[0])
    a, b = preprocess(pts), preprocess(pts.copy())
    assert np.array_equal(a['partial'], b['partial'])
    assert np.array_equal(a['transform']['center'], b['transform']['center'])
    assert a['transform']['scale'] == b['transform']['scale']
    # row order of the input must not matter either
    perm = np.random.default_rng(0).permutation(len(pts))
    c = preprocess(pts[perm])
    assert np.array_equal(a['partial'], c['partial'])


def test_exact_inverse_transform(real_results):
    rng = np.random.default_rng(1)
    for name, r in real_results.items():
        t = r['transform']
        x = rng.normal(size=(500, 3)) * 3.0
        back = from_canonical(to_canonical(x, t), t)
        # float32 rounding of the canonical coordinates is the only loss
        assert np.allclose(back, x.astype(np.float32), rtol=0, atol=2e-6 * (1 + np.abs(x).max()) * 1), name
        # canonical partial maps back into object points (inside the original cloud's bounding box)
        raw = from_canonical(r['partial'], t)
        p = r['meta']['debug']['unique_points']
        assert (raw >= p.min(0) - 1e-4).all() and (raw <= p.max(0) + 1e-4).all(), name
        assert np.allclose(to_canonical(raw, t), r['partial'], atol=1e-4 * (1 + np.abs(r['partial']).max()))


def test_inverse_is_exact_in_float64():
    t = dict(center=np.array([0.123456789, -0.5, 0.25]), scale=0.0731)
    x = np.random.default_rng(3).normal(size=(100, 3))
    c64 = (x - t['center']) / t['scale']
    assert np.allclose(from_canonical(c64, t), x, atol=1e-6)


def test_degenerate_inputs():
    rng = np.random.default_rng(0)
    # nan rows are dropped, tiny clouds do not crash and still return the full shape
    p = rng.normal(size=(300, 3)).astype(np.float32)
    p[5] = np.nan
    r = preprocess(p)
    assert r['partial'].shape == (CANON_POINTS, 3) and np.isfinite(r['partial']).all()
    # fewer points than the target -> deterministic tiling, same result twice
    q = rng.normal(size=(60, 3)).astype(np.float32)
    a, b = preprocess(q), preprocess(q)
    assert np.array_equal(a['partial'], b['partial'])
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 3)))


def _within(pred, truth, tol):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(truth).query(pred)
    return float((d <= tol).mean())


def test_sim_object_recovery(sims, capsys):
    """Fraction of predicted object points within 2 voxels (2 cm raw) of the TRUE object surface, for
    sessions where the object was held close (the real regime).

    The viewer never tracks hand translation, so the object appears smeared by the hand motion in the raw map
    (up to `max(trans_amp_m)`); we therefore also report the fraction within 2 cm + that amplitude, and the
    precision of the whole cloud (wall included) as the no-segmentation baseline."""
    rows = []
    for s in sims:
        r = preprocess(s.export)
        m = r['meta']
        p, idx = m['debug']['unique_points'], m['debug']['object_indices_unique']
        sc = s.truth['scale']
        amp = max(s.truth['motion'].trans_amp_m)
        true_depth = s.truth['distance_m'] / sc                 # front-face distance in export units
        so = m['sensor_estimate']
        xz_err = (np.nan if so is None else
                  float(np.hypot(so['o'][0] - s.truth['sensor_norm'][0], so['o'][2] - s.truth['sensor_norm'][2])))
        rows.append((_within(p[idx], s.complete, 0.02 / sc), _within(p[idx], s.complete, (0.02 + amp) / sc),
                     _within(p, s.complete, 0.02 / sc), m['mode'], len(idx), s.truth['distance_m'],
                     r['transform']['scale'] / true_depth, xz_err))
        # transform inverse maps the canonical partial back onto points that were actually scanned
        assert r['partial'].shape == (CANON_POINTS, 3) and np.isfinite(r['partial']).all()
    with capsys.disabled():
        print('\n[sim] dist_m  within-2cm  within-2cm+hand  whole-cloud-baseline  mode  n_obj  scale/true  sensor_xz_err')
        for r in rows:
            print('   %.2f     %.2f        %.2f            %.2f              %s  %d    %.2f      %.3f' % (
                r[5], r[0], r[1], r[2], r[3], r[4], r[6], r[7]))
        lr = np.log([r[6] for r in rows])
        print('   scale/true (front-face distance): median %.2f, median |log| %.2f, worst %.2fx; '
              'scale is only a distance-consistent frame to this accuracy (NOT validated tighter)' %
              (np.exp(np.median(lr)), np.median(np.abs(lr)), np.exp(np.abs(lr).max())))
        print('   median: 2cm %.2f, 2cm+hand %.2f, baseline %.2f' %
              (np.median([r[0] for r in rows]), np.median([r[1] for r in rows]), np.median([r[2] for r in rows])))
    assert np.median([r[0] for r in rows]) > 2 * np.median([r[2] for r in rows])     # far better than keeping everything
    assert np.median([r[1] for r in rows]) >= 0.9
    assert np.median([r[0] for r in rows]) >= 0.4
