"""Registered parametric targets for real scans of a known object."""
import json

import numpy as np
import pytest

from majproj.canon import from_canonical, sample_2048, to_canonical
from majproj.real_targets import Part, build_target, register, sample_surface
from scripts.gen_final import GT_LIMIT, N_TARGET


def cylinder_parts(w=200.0, h=700.0):
    return [Part(dict(kind='revolve', center=[0, h / 2, 0], size=[w, h, w], rotation_deg=[0, 0, 0], profile=[[0, 1], [1, 1]]))]


def front_face(radius_m=0.035, height_m=0.25, centre=(0.0, 0.33, 0.12), n=1500, seed=0):
    """Points on the sensor-facing half of a vertical cylinder in viewer metres (z up, sensor at the origin)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(-np.pi / 2, np.pi / 2, n)             # the half that faces -y (towards the sensor at the origin)
    z = rng.uniform(0, height_m, n)
    x = centre[0] + radius_m * np.sin(th)
    y = centre[1] - radius_m * np.cos(th)
    return np.stack([x, y, centre[2] + z], 1)


def test_revolve_depth_equals_width_and_surface_is_closed():
    p = cylinder_parts()[0]
    assert p.size[2] == p.size[0]
    pts = sample_surface([p], 5000, np.random.default_rng(0))
    assert pts.shape == (5000, 3)
    r = np.hypot(pts[:, 0], pts[:, 2])
    inside = (pts[:, 1] > 1) & (pts[:, 1] < 699)
    assert np.allclose(r[inside], 100, atol=0.1)             # side wall at the radius (96-gon chord error is 0.054)
    assert (~inside).sum() > 100                             # flat ends exist


def test_register_recovers_the_cylinder_within_the_noise():
    face = front_face()
    origin = np.zeros(3)
    from majproj.canon import _rotation
    R = _rotation(face, origin)
    face_v = (face - origin) @ R.T
    shape = sample_surface(cylinder_parts(), 40000, np.random.default_rng(1))
    placed, info = register(shape, face_v, [0.0], height_m=0.25)
    assert info['resid_median_mm'] < 2.0 and info['front_within_face_depth']
    e = np.percentile(placed, 99, 0) - np.percentile(placed, 1, 0)
    assert abs(e[0] - 0.07) < 0.006 and abs(e[1] - 0.25) < 0.006      # width from the parts, height from the measurement


def test_build_target_has_the_shard_form_and_rejects_a_bad_fit():
    face = front_face()
    canon, T = to_canonical(face)
    gt, info = build_target(cylinder_parts(), sample_2048(canon).astype(np.float64), T, N_TARGET, height_m=0.25)
    assert info['accepted'] and gt.shape == (N_TARGET, 3) and np.abs(gt).max() < GT_LIMIT
    back = from_canonical(gt, T)
    assert back[:, 2].min() > 0.11 and back[:, 2].max() < 0.38          # sits on the face's z range
    assert (back[:, 1] > face[:, 1].max() + 0.01).sum() > N_TARGET // 10   # has a back the face does not have
    # a face twice as wide as the object's outline cannot be registered within the limit
    wide = front_face(radius_m=0.09)
    canon, T = to_canonical(wide)
    gt2, info2 = build_target(cylinder_parts(), sample_2048(canon).astype(np.float64), T, N_TARGET, height_m=0.25, max_median_mm=5.0)
    assert gt2 is None and not info2['accepted']


def test_parts_file_loads():
    d = json.load(open('configs/real_bottle_parts.json'))
    parts = [Part(o) for o in d['parts']]
    assert parts and parts[0].kind == 'revolve' and len(parts[0].profile) >= 10 and '_source' in d
