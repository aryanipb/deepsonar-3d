"""Tests of the session simulator `majproj.sim_scan` (device-format export + per-point truth).

    uv run --with pytest pytest -q tests/test_sim_scan.py
"""
import time
from functools import lru_cache

import numpy as np
import pytest
from scipy.spatial import cKDTree

from majproj import sim_scan as ss
from majproj.sim.acquisition import ViewerMap, format_for_ai

KINDS = ss.KINDS
FULL = dict(p_ceiling_start=1.0, p_ceiling_end=1.0)            # all five phases present


@lru_cache(maxsize=None)
def scan(kind='box', seed=0, **kw):
    return ss.simulate_scan(np.random.default_rng(seed), kind=kind, **dict(kw))


def _sc(kind='box', seed=0, **kw):
    return scan(kind, seed, **kw)


# ------------------------------------------------------------------------------------------------ format

@pytest.mark.parametrize('kind', KINDS)
def test_shapes_and_dtypes(kind):
    r = _sc(kind, 1)
    assert set(r) == {'export', 'object_mask', 'object_complete', 'truth'}
    assert r['export'].shape == (4096, 3) and r['export'].dtype == np.float32
    assert r['object_mask'].shape == (4096,) and r['object_mask'].dtype == bool
    assert r['object_complete'].shape == (16384, 3) and r['object_complete'].dtype == np.float32
    assert np.isfinite(r['export']).all() and np.isfinite(r['object_complete']).all()
    t = r['truth']
    for k in ('kind', 'size_m', 'distance_m', 'phases', 'n_raw', 'n_map', 'n_unique', 'scale_m', 'source', 'phase_id'):
        assert k in t, k
    assert t['kind'] == kind and len(t['size_m']) == 3 and t['n_raw'] >= t['n_map'] > 0
    assert t['source'].shape == (4096,) and t['phase_id'].shape == (4096,)


@pytest.mark.parametrize('kind', KINDS)
def test_bbox_centred_and_max_abs_one(kind):
    e = _sc(kind, 2)['export'].astype(np.float64)
    assert np.allclose((e.max(0) + e.min(0)) / 2, 0.0, atol=1e-6)      # bbox centre at the origin
    assert abs(np.abs(e).max() - 1.0) < 1e-6                           # divided by the largest absolute coordinate


def test_z_is_up_ceiling_points_are_the_highest():
    for seed in range(3):
        r = _sc('box', seed, **FULL)
        e, src = r['export'], r['truth']['source']
        ceil = src == ss.CEILING
        assert ceil.sum() > 300
        assert np.median(e[ceil, 2]) > 0.3
        assert np.median(e[ceil, 2]) > np.median(e[~ceil, 2]) + 0.5      # everything else lies far below the ceiling
        top = np.argsort(e[:, 2])[-60:]
        assert (src[top] == ss.CEILING).mean() > 0.9                      # the highest points are the ceiling layer
        # the layer is flat: its z spread is small against the file extent, x / y spread over the 8x8 grid footprint
        assert np.percentile(e[ceil, 2], 90) - np.percentile(e[ceil, 2], 10) < 0.35
        assert np.ptp(e[ceil, 0]) > 0.5 and np.ptp(e[ceil, 1]) > 0.5


# ------------------------------------------------------------------------------------------------ truth

@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_mask_non_empty_and_consistent_with_source(kind, seed):
    r = _sc(kind, seed)
    assert r['object_mask'].sum() >= 25
    assert np.array_equal(r['object_mask'], r['truth']['source'] == ss.OBJECT)
    assert r['truth']['n_object'] == int(r['object_mask'].sum())
    assert (~r['object_mask']).sum() > 100                            # and there is real garbage next to the object


@pytest.mark.parametrize('kind', KINDS)
def test_object_complete_aligned_with_masked_points(kind):
    """Without hand drift / wobble the masked export points lie on the complete surface (same normalized frame)."""
    errs = []
    for seed in range(3):
        r = _sc(kind, seed, drift_scale=0.0, wobble_scale=0.0)
        errs.append(cKDTree(r['object_complete']).query(r['export'][r['object_mask']])[0] * r['truth']['scale_m'])
    d = np.concatenate(errs)
    assert np.median(d) < 0.02                                        # noise + footprint of a zone + 10 mm voxels + IMU error
    assert (d < 0.04).mean() > 0.85


def test_depth_copies_from_untracked_translation():
    """Rotation is registered, translation is not: with drift the same face is stored at shifted depths, so the masked
    points move away from the complete surface; without drift they sit on it."""
    p90 = {}
    for scale in (0.0, 1.0):
        d = []
        for seed in range(3):
            r = _sc('box', seed, distance_m=0.3, drift_scale=scale)
            d.append(cKDTree(r['object_complete']).query(r['export'][r['object_mask']])[0] * r['truth']['scale_m'])
        p90[scale] = np.percentile(np.concatenate(d), 90)
    assert p90[1.0] > 1.5 * p90[0.0] and p90[1.0] > 0.015


@pytest.mark.parametrize('kind', KINDS)
def test_complete_surface_scale_and_height(kind):
    r = _sc(kind, 3)
    t = r['truth']
    height_m = np.ptp(r['object_complete'][:, 2]) * t['scale_m']
    assert abs(height_m - t['size_m'][2]) < 0.01                       # z is the object's up axis, scale metres/unit is right
    wide = max(np.ptp(r['object_complete'][:, 0]), np.ptp(r['object_complete'][:, 1])) * t['scale_m']
    assert 0.04 < wide < 0.46


@pytest.mark.parametrize('dist', [0.2, 0.5, 0.9])
def test_distance_param_is_sensor_to_nearest_object_point(dist):
    r = _sc('box', 4, distance_m=dist)
    t = r['truth']
    assert t['distance_m'] == dist
    origin = -np.array(t['centre_m']) / t['scale_m']                    # the (untracked) sensor in the export frame
    nearest = np.linalg.norm(r['object_complete'] - origin, axis=1).min() * t['scale_m']
    # nominal pose = mean true position; the nearest face point is within the hand-drift amplitude of the nominal distance
    assert abs(nearest - dist) < 0.06


# ------------------------------------------------------------------------------------------------ session structure

def test_phases_follow_options():
    t = _sc('box', 5, **FULL)['truth']
    assert t['phases'] == list(ss.PHASES) and all(p in ss.PHASES for p in t['phases'])
    t = _sc('box', 5, p_ceiling_start=0.0, p_ceiling_end=0.0)['truth']
    assert 'ceiling_start' not in t['phases'] and 'scan' in t['phases'] and 'pickup' in t['phases']
    assert not (t['phase_id'] == ss.PHASES.index('ceiling_start')).any()


def test_hand_junk_is_near_the_sensor():
    found = 0
    for seed in range(4):
        r = _sc('box', seed, p_hand_pickup=1.0, p_hand_scan=1.0, p_hand_putdown=1.0)
        e, src, t = r['export'], r['truth']['source'], r['truth']
        hand = src == ss.HAND
        if not hand.any():
            continue
        found += 1
        origin = -np.array(t['centre_m']) / t['scale_m']
        d_m = np.linalg.norm(e[hand] - origin, axis=1) * t['scale_m']
        assert np.median(d_m) < 0.2                                     # 2-10 cm from the sensor (+ the IMU offset)
        assert t['hand_junk'] is True
        assert not (r['object_mask'] & hand).any()
    assert found >= 2


def test_pool_size_decides_duplicates():
    """Raw map < 4096 points -> sampling with replacement -> duplicate points; >= 4096 -> (almost) all distinct."""
    short = dict(hold_s=(3.0, 3.5), ceiling_start_s=(1.0, 1.5), p_ceiling_end=0.0, p_back_wall=0.0, n_clutter_max=0, distance_m=0.6)
    small = [_sc('box', s, **short) for s in range(8)]
    n_small = [r for r in small if r['truth']['n_map'] < 4096]
    assert n_small, [r['truth']['n_map'] for r in small]
    for r in n_small:
        t = r['truth']
        assert t['n_unique'] <= t['n_map'] and t['n_unique'] < 4096 - 200       # duplicates from the replacement
    big = [_sc('box', s, hold_s=(11.0, 12.0), distance_m=0.25) for s in range(4)]
    n_big = [r for r in big if r['truth']['n_map'] >= 4096]
    assert n_big
    for r in n_big:
        assert r['truth']['n_unique'] >= 4096 - 150                             # only exact repeats in the pending buffer


def test_default_prior_covers_both_pool_regimes():
    """Like the real files: some sessions end with < 4096 map points (duplicates), others with more."""
    n_map = [ss.simulate_scan(np.random.default_rng([77, s]))['truth']['n_map'] for s in range(14)]
    assert min(n_map) < 4096 < max(n_map), n_map


# ------------------------------------------------------------------------------------------------ exactness of the viewer pieces

def test_labelled_map_replicates_viewer_map():
    rng = np.random.default_rng(0)
    a, b = ViewerMap(), ss._LabelledMap()
    for _ in range(60):
        n = int(rng.integers(1, 60))
        pts = rng.normal(0.0, 0.05, (n, 3)) + rng.integers(0, 3, (n, 1)) * 0.2
        pts = np.round(pts, 3)                                                  # create exact duplicates / shared voxels
        a.add(pts)
        b.add(pts, np.zeros((n, 2), np.int8))
    pa, (pb, lb) = a.export(), b.export()
    assert pa.shape == pb.shape and np.array_equal(pa, pb) and len(lb) == len(pb)


@pytest.mark.parametrize('n', [900, 4096, 9000])
def test_export_step_matches_format_for_ai(n):
    pts = np.random.default_rng(1).normal(0, 0.4, (n, 3)) + [0.3, -0.2, 0.1]
    mine, idx, centre, m = ss._export(pts, np.random.default_rng(7))
    ref, centre_ref, m_ref = format_for_ai(pts, np.random.default_rng(7))
    assert np.array_equal(mine, ref) and np.allclose(centre, centre_ref) and m == m_ref
    assert (len(np.unique(idx)) < 4096) == (n < 4096)                           # repeats iff sampled with replacement


def test_provenance_survives_thinning_and_sampling():
    """The label of an export point is the label of the map point it was sampled from."""
    rng = np.random.default_rng(3)
    m = ss._LabelledMap()
    for k in range(40):
        n = 40
        pts = rng.uniform(-1, 1, (n, 3))
        m.add(pts, np.stack([np.full(n, k % 7, np.int8), np.full(n, k % 5, np.int8)], 1))
    pts, lab = m.export()
    e, idx, centre, scale = ss._export(pts, np.random.default_rng(0))
    assert np.allclose(e, ((pts[idx] - centre) / scale).astype(np.float32), atol=1e-6)
    assert lab[idx].shape == (4096, 2)


# ------------------------------------------------------------------------------------------------ determinism, API

def test_determinism_by_seed():
    a = ss.simulate_scan(np.random.default_rng(11), kind='bottle')
    b = ss.simulate_scan(np.random.default_rng(11), kind='bottle')
    c = ss.simulate_scan(np.random.default_rng(12), kind='bottle')
    for k in ('export', 'object_mask', 'object_complete'):
        assert np.array_equal(a[k], b[k])
    assert np.array_equal(a['truth']['source'], b['truth']['source']) and a['truth']['distance_m'] == b['truth']['distance_m']
    assert not np.array_equal(a['export'], c['export'])


def test_batch_generator():
    out = list(ss.simulate_batch(3, seed=9))
    assert len(out) == 3
    ref = ss.simulate_scan(np.random.default_rng([9, 1]))
    assert np.array_equal(out[1]['export'], ref['export']) and np.array_equal(out[1]['object_mask'], ref['object_mask'])
    assert len({r['truth']['kind'] for r in ss.simulate_batch(12, seed=1)}) >= 2     # kind=None draws the kind
    again = list(ss.simulate_batch(3, seed=9))
    assert all(np.array_equal(x['export'], y['export']) for x, y in zip(out, again))


def test_options_and_validation():
    with pytest.raises(TypeError):
        ss.simulate_scan(np.random.default_rng(0), bogus_option=1)
    with pytest.raises(ValueError):
        ss.simulate_scan(np.random.default_rng(0), kind='sphere')


def test_time_per_scan_under_budget():
    t0 = time.perf_counter()
    for s in range(3):
        ss.simulate_scan(np.random.default_rng(900 + s), kind=KINDS[s % 4])
    assert (time.perf_counter() - t0) / 3 < 3.0
