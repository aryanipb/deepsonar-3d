"""The measurements in scripts/evaluate_final.py on clouds with a known answer."""
import numpy as np
import torch

from majproj.train_final import cd_per_scan
from scripts.evaluate_final import GATES, aligned_cd, box_check, extents, probe_clouds, roundness, scan_metrics, stratified


def box_surface(rng, w, h, d, n=16384, yaw=0.0):
    """Metric, y up."""
    q = rng.uniform(-.5, .5, (n, 3)) * [w, h, d]
    ax = rng.integers(0, 3, n)
    q[np.arange(n), ax] = np.sign(q[np.arange(n), ax]) * np.array([w, h, d])[ax] / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return q @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def cylinder_surface(rng, r, h, n=16384):
    a = rng.uniform(0, 2 * np.pi, n)
    return np.c_[r * np.cos(a), rng.uniform(-h / 2, h / 2, n), r * np.sin(a)]


def test_roundness_on_a_cylinder():
    m = roundness(cylinder_surface(np.random.default_rng(0), .035, .25))
    assert abs(m['radius_mm'] - 35) < .5 and m['rms_over_r'] < .01 and m['coverage'] == 1 and m['aspect'] > .95


def test_roundness_sees_a_half_shell_and_a_flat_section():
    q = cylinder_surface(np.random.default_rng(0), .035, .25)
    assert roundness(q[q[:, 2] > 0])['coverage'] < .6
    assert roundness(q * [1, 1, .4])['aspect'] < .5


def test_box_check_on_a_turned_box():
    m = box_check(box_surface(np.random.default_rng(0), .19, .23, .12, yaw=.5))
    assert m['faces'] == 6 and m['flat_mm'] < 1 and m['angle_deg'] < 1.5 and m['on_surface'] > .9 and m['face_fill'] > .9


def test_box_check_rejects_a_box_sized_cylinder():
    m = box_check(cylinder_surface(np.random.default_rng(0), .095, .23))
    assert m['flat_mm'] > GATES['box_flat_mm'] and m['on_surface'] < GATES['box_on_surface']


def test_box_check_keeps_a_noisy_box():
    rng = np.random.default_rng(0)
    m = box_check(box_surface(rng, .19, .23, .15) + rng.normal(0, .004, (16384, 3)))
    assert m['flat_mm'] < GATES['box_flat_mm'] and m['on_surface'] > GATES['box_on_surface'] and m['angle_deg'] < 2


def test_scan_metrics_cd_equals_the_trainer():
    rng = np.random.default_rng(0)
    p, g = rng.normal(size=(512, 3)).astype(np.float32), rng.normal(size=(512, 3)).astype(np.float32)
    want = float(cd_per_scan(torch.from_numpy(p)[None], torch.from_numpy(g)[None])[0])
    m = scan_metrics(p, g, p[:64], .3)
    assert abs(m['cd_mm'] - want * .3 * 1000) < 1e-2 * want * 300


def test_perfect_output_scores_perfectly():
    g = box_surface(np.random.default_rng(0), .4, .9, .3)
    m = scan_metrics(g, g, g[:2048], .3)
    assert m['cd_mm'] == 0 and m['f5'] == 1 and m['w_err_mm'] == 0


def test_extents_ignore_strays():
    q = np.random.default_rng(0).uniform(-1, 1, (5000, 3))
    q[0] = 100
    assert (extents(q) < 2.1).all()


def test_aligned_cd_undoes_rotation_about_y():
    rng = np.random.default_rng(0)
    a = box_surface(rng, .2, .3, .1, 2048)
    b = box_surface(rng, .2, .3, .1, 2048, yaw=np.pi / 3) + [.5, .1, 0]
    assert aligned_cd(a, b) < 8


def test_stratified_is_nested_and_covers_kinds():
    kinds = np.random.default_rng(0).integers(0, 10, 2000)
    a, b = stratified(kinds, 5), stratified(kinds, 20)
    assert set(a) <= set(b) and len(a) == 50 and len(stratified(kinds, 0)) == 2000


def test_probes_are_the_shapes_they_claim():
    c = probe_clouds(0)
    assert np.ptp(c['plate'][:, 1]) < .03 and np.ptp(c['rod'][:, 0]) < .04 and np.ptp(c['rod'][:, 2]) > .28
    assert np.ptp(c['cube'][:, 1]) > .05 and (c['cube'][:, 1] < .35 + .015).mean() > .9
