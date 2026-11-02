"""Tests of majproj.sim_raw.    uv run --with pytest pytest -q tests/test_sim_raw.py"""
import json
import os
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from majproj import sim_raw as sr
from majproj import sim_real as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = json.load(open(os.path.join(ROOT, 'configs', 'real_bank.json')))['sessions']


def run(kind, regime, seed=0, **kw):
    return sr.simulate_raw(np.random.default_rng(seed), kind, regime=regime, **kw)


@pytest.mark.parametrize('regime', ['real', 'robust'])
@pytest.mark.parametrize('kind', sr.KINDS)
def test_format(kind, regime):
    r = run(kind, regime, 3)
    assert set(r) == {'raw', 'complete', 'object_mask', 'truth'}
    assert r['raw'].ndim == 2 and r['raw'].shape[1] == 3 and r['raw'].dtype == np.float64
    assert r['complete'].shape == (16384, 3) and r['complete'].dtype == np.float32
    assert r['object_mask'].shape == (len(r['raw']),) and r['object_mask'].dtype == bool
    assert np.isfinite(r['raw']).all() and np.isfinite(r['complete']).all()
    t = r['truth']
    assert t['regime'] == regime and t['kind'] == kind
    for k in ('size_m', 'distance_m'):
        assert k in t


@pytest.mark.parametrize('kind', sr.KINDS)
def test_object_rows_lie_on_the_complete_surface(kind):
    r = run(kind, 'real', 5)
    assert r['object_mask'].sum() > 100
    d = cKDTree(r['complete']).query(np.unique(r['raw'][r['object_mask']], axis=0))[0]
    # zone blur, ranging noise and up to several cm of hand drift separate them; the median must stay small
    assert np.median(d) < 0.05


def test_deterministic():
    a, b = run('bottle', 'real', 9), run('bottle', 'real', 9)
    assert np.array_equal(a['raw'], b['raw']) and np.array_equal(a['complete'], b['complete'])


def test_duplicates_match_real_files():
    fr = [1 - len(np.unique(r, axis=0)) / len(r) for r in (run('cuboid', 'real', s)['raw'] for s in range(6))]
    assert 0.40 < np.mean(fr) < 0.56


def test_regime_share_and_cap():
    rng = np.random.default_rng(0)
    assert np.mean([rng.random() < .15 for _ in range(4000)]) < .2
    with pytest.raises(ValueError):
        sr.simulate_raw(rng, 'can', p_robust=0.25)
    with pytest.raises(ValueError):
        sr.simulate_raw(rng, 'can', regime='other')
    with pytest.raises(ValueError):
        sr.simulate_raw(rng, 'nope')


def test_real_regime_stays_inside_the_measured_envelope():
    lo = min(m['range_m'] for m in META) - sp.MARGIN_RANGE_M
    hi = max(m['range_m'] for m in META) + sp.MARGIN_RANGE_M
    az_lo = min(m['az_rad'] for m in META) - sp.MARGIN_AZ_RAD
    az_hi = max(m['az_rad'] for m in META) + sp.MARGIN_AZ_RAD
    for s in range(30):
        t = run('can', 'real', s)['truth']
        assert lo <= t['distance_m'] <= hi and az_lo <= t['az_rad'] <= az_hi


def test_holdout_sessions_never_used_by_default():
    h = sp.holdout_sessions()
    assert len(h) == 2
    for s in range(40):
        t = run('cuboid', 'real', s)['truth']
        assert not {t['session'], t['session_start'], t['session_end']} & set(h)


def test_leave_session_out_fold():
    for s in range(15):
        t = run('can', 'real', s, exclude_sessions=('114126', '115102'))['truth']
        assert not {t['session'], t['session_start'], t['session_end']} & {'114126', '115102'}


def test_bank_holds_no_object():
    """Only stray rows may remain in the old object's sector (cut region, inner 8 cm of depth around the median range)."""
    for m in META:
        r = run('cube', 'real', 1, session=m['session'], exclude_sessions=(), render_object=False)['raw']
        hr = np.hypot(r[:, 0], r[:, 1])
        da = np.abs(np.angle(np.exp(1j * (np.arctan2(r[:, 1], r[:, 0]) - m['az_rad']))))
        s = (np.abs(hr - m['range_m']) < .08) & (da < np.radians(35)) & (r[:, 2] > m['table_z'] + .03) & (r[:, 2] < .3)
        assert s.sum() == 0, m['session']


def test_box_front_face_is_at_the_requested_range():
    sh = dict(type='box', a=.2, b=.1, h=.25)
    m = next(x for x in META if x['session'] == '115840')
    r = sp.simulate_real(np.random.default_rng(0), 'cuboid', exclude_sessions=(), session='115840', shape=sh, distance_m=.3, az_rad=m['az_rad'], face_az=m['az_rad'])
    p = np.unique(r['raw'][r['object_mask']], axis=0)
    v = np.array([np.cos(m['az_rad']), np.sin(m['az_rad'])])
    assert abs(np.median(p[:, :2] @ v) - .3) < .03
    assert abs(np.percentile(r['complete'][:, :2] @ v, 1) - .3) < .04


def test_pipe_is_hollow():
    sh = dict(type='rev', ys=[0, .3], rs=[.05, .05], h=.3, r=.05, wall=.005)
    p = sp.build_prim(sh, (0, 0), 0.0, 0.0, .5).surface(4000, np.random.default_rng(0))
    assert ((np.hypot(p[:, 0], p[:, 2]) < .044) & (p[:, 1] > .01) & (p[:, 1] < .29)).sum() == 0


def test_coverage_of_dimensions():
    rng = np.random.default_rng(0)
    b = np.array([sp.size_of(sp.sample_shape(rng, 'cuboid')) for _ in range(3000)])
    assert b.min() < .032 and b.max() > .58
    c = np.array([sp.size_of(sp.sample_shape(rng, k)) for k in ('bottle', 'wavy_bottle', 'can', 'jar', 'rod', 'pipe') for _ in range(500)])
    assert c[:, 0].min() < .024 and c[:, 0].max() > .15 and c[:, 2].min() < .09 and c[:, 2].max() > .58


def test_robust_ranges():
    t = [run('can', 'robust', s)['truth'] for s in range(12)]
    assert all(.05 <= x['distance_m'] <= 1.3 for x in t) and all(x['regime'] == 'robust' for x in t)


@pytest.mark.parametrize('regime', ['real', 'robust'])
def test_speed(regime):
    ts = []
    for s in range(8):
        t = time.time()
        run('wavy_bottle', regime, s)
        ts.append(time.time() - t)
    assert np.median(ts) < 2.0
