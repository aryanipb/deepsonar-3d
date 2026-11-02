import json

import numpy as np
import pytest

from majproj.canon import apply_canonical, from_canonical, sample_2048, to_canonical


def _face(rng, n=500, centre=(0.1, 0.35, -0.02), w=0.19, h=0.23):
    """A vertical rectangle facing the origin, with a little depth noise."""
    c = np.asarray(centre)
    side = np.array([-c[1], c[0], 0.0]) / np.hypot(c[0], c[1])
    uv = rng.uniform(-0.5, 0.5, (n, 2))
    return c + uv[:, :1] * w * side + uv[:, 1:] * h * np.array([0.0, 0.0, 1.0]) + rng.normal(0, 1e-3, (n, 1)) * c / np.linalg.norm(c)


def test_roundtrip_and_json():
    rng = np.random.default_rng(0)
    p = _face(rng)
    q, T = to_canonical(p)
    T2 = json.loads(json.dumps(T))
    assert np.abs(from_canonical(q, T2) - p).max() < 1e-12
    q2 = apply_canonical(p, T2)
    assert np.abs(q2 - q).max() < 1e-12


def test_axes():
    """Up stays up, +z points away from the sensor, the frame is right-handed and the rotation is about vertical."""
    p = _face(np.random.default_rng(1))
    q, T = to_canonical(p)
    R = np.asarray(T['R'])
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R[1], [0, 0, 1]) and np.isclose(R[0, 2], 0) and np.isclose(R[2, 2], 0)
    view = p.mean(0) * [1, 1, 0]
    assert np.allclose(R[2], view / np.linalg.norm(view))
    assert np.allclose(np.ptp(q, 0)[1], np.ptp(p, 0)[2] / T['scale'])     # height preserved up to scale
    a = from_canonical(np.array([[0, 0, 0], [0, 0, 1.0]]), T)
    assert np.linalg.norm((a[1] - a[0])[:2]) > 0 and np.dot((a[1] - a[0])[:2], view[:2]) > 0


def test_centre_and_scale():
    rng = np.random.default_rng(2)
    q, T = to_canonical(_face(rng, w=0.19, h=0.23))
    lo, hi = q.min(0), q.max(0)
    assert np.allclose((lo + hi) / 2, 0)
    assert np.isclose(np.ptp(q, 0)[:2].max(), 0.9)
    assert np.isclose(T['scale'], 0.23 / 0.9, rtol=0.02)


def test_scale_ignores_depth():
    """A long depth streak must not change the scale: it is the side of the seen FACE."""
    rng = np.random.default_rng(3)
    p = _face(rng)
    deep = p[:5] * 1.8
    _, T0 = to_canonical(p)
    _, T1 = to_canonical(np.vstack([p, deep]))
    assert T1['scale'] < T0['scale'] * 1.8


def test_target_uses_the_seen_transform():
    rng = np.random.default_rng(4)
    p = _face(rng)
    back = p + p.mean(0) / np.linalg.norm(p.mean(0)) * 0.1
    q, T = to_canonical(p)
    qb = apply_canonical(back, T)
    assert np.isclose(np.median(qb[:, 2] - q[:, 2]), 0.1 / T['scale'], rtol=0.02)     # 10 cm farther = +z


def test_origin_is_respected():
    p = _face(np.random.default_rng(5))
    o = np.array([0.3, -0.2, 0.5])
    q0, T0 = to_canonical(p)
    q1, T1 = to_canonical(p + o, origin=o)
    assert np.abs(q0 - q1).max() < 1e-9
    assert np.abs(from_canonical(q1, T1) - (p + o)).max() < 1e-9


def test_degenerate_inputs():
    q, T = to_canonical(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.5]]))          # straight above the sensor
    assert np.isfinite(q).all() and np.isclose(np.linalg.det(np.asarray(T['R'])), 1.0)
    q, T = to_canonical(np.array([[0.1, 0.2, 0.3]]))                           # single point: no extent
    assert T['scale'] == 1.0 and np.allclose(q, 0)
    for bad in (np.zeros((0, 3)), np.zeros((4, 2)), np.array([[np.nan, 0, 0]])):
        with pytest.raises(ValueError):
            to_canonical(bad)
    with pytest.raises(ValueError):
        sample_2048(np.zeros((0, 3)))


def test_sample_2048():
    rng = np.random.default_rng(6)
    p = rng.normal(size=(5000, 3))
    a, b = sample_2048(p), sample_2048(p)
    assert a.shape == (2048, 3) and a.dtype == np.float32 and np.array_equal(a, b)
    assert len(np.unique(a, axis=0)) == 2048
    assert np.allclose(a[0], p[0])

    # FPS reaches the extremes that a random subset would usually miss
    assert np.abs(a).max() == pytest.approx(np.abs(p).max(), rel=1e-6)

    small = rng.normal(size=(300, 3))
    t = sample_2048(small)
    assert t.shape == (2048, 3) and len(np.unique(t, axis=0)) == 300
    assert sample_2048(p[:2048]).shape == (2048, 3)
