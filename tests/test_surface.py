"""The surface step: one closed surface that stays on the predicted points."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from majproj.surface import is_closed, to_mesh, write_obj


def cylinder(n=16384, r=0.035, h=0.25, seed=0):
    g = np.random.default_rng(seed)
    t, z = g.uniform(0, 2 * np.pi, n), g.uniform(0, h, n)
    return np.stack([r * np.cos(t), r * np.sin(t), z], 1)


def test_shell_becomes_one_closed_surface_on_the_points():
    p = cylinder()
    v, f = to_mesh(p)
    assert is_closed(f)
    assert np.median(cKDTree(p).query(v)[0]) < 0.005
    r = np.hypot(v[:, 0], v[:, 1])
    assert 0.03 < np.percentile(r, 95) < 0.042            # an outer wall, not a thin rod
    assert (r < 0.015).mean() < 0.15                      # no second sheet inside, only the end caps


def test_rejects_bad_input_and_writes_obj(tmp_path):
    with pytest.raises(ValueError):
        to_mesh(np.zeros((5, 3)))
    v, f = to_mesh(cylinder())
    write_obj(tmp_path / 'm.obj', v, f)
    lines = open(tmp_path / 'm.obj').read().splitlines()
    assert len(lines) == len(v) + len(f) and lines[0].startswith('v ') and lines[-1].startswith('f ')
