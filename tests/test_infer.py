"""majproj.infer: file type detection, the training pipeline, sizes, and "rejected scan -> no completion"."""
import json

import numpy as np

from majproj.canon import apply_canonical, sample_2048, to_canonical
from majproj.clean import clean
from majproj.infer import N_OUT, Completer, export_origin, extents, is_export, json_safe, prepare, stage_of
from scripts.gen_final import metric_thin


class Echo:
    """Stands in for the network: outputs its input, tiled to N_OUT points, and counts calls."""
    calls = 0

    def __call__(self, x):
        Echo.calls += 1
        return None, x.repeat(1, 8, 1)


def completer(model=None, stage='pilot'):
    c = Completer.__new__(Completer)
    c.device, c.ckpt, c.stage, c.info, c.model = 'cpu', 'x.pt', stage, {}, model or Echo()
    return c


def scene(seed=0, n_face=4000):
    """Raw-like map: a dense vertical face 30 cm away with sparse room around it."""
    r = np.random.default_rng(seed)
    face = np.c_[r.uniform(-0.1, 0.1, n_face), np.full(n_face, 0.3) + r.normal(0, 0.001, n_face), r.uniform(-0.1, 0.15, n_face)]
    room = r.uniform(-2.5, 2.5, (1500, 3))
    return np.vstack([face, room])


def test_file_type():
    raw = scene()
    assert not is_export(raw)
    ex = raw[:4096] / np.abs(raw[:4096]).max()
    assert is_export(ex)
    assert prepare(ex)[1] is False and prepare(raw)[1] is True
    small = raw[np.abs(raw).max(1) < 0.9]              # a raw map inside +-1 m is still metric
    assert not is_export(small) and prepare(small)[1] is True


def test_pipeline_is_the_training_pipeline():
    raw = scene()
    res = completer()(raw)
    assert res.ok and res.metric
    pts = metric_thin(raw)[0]
    r = clean(pts, origin=(0.0, 0.0, 0.0))
    canon, T = to_canonical(r.object)
    assert np.array_equal(res.partial, sample_2048(canon)) and res.T == T
    assert res.completed.shape == (N_OUT, 3) and res.completed.dtype == np.float32
    back = apply_canonical(res.completed, T)           # inverse of the transform, back at the canonical output
    assert np.allclose(back, res.completed_canon, atol=2e-5)


def test_sizes_separate_measured_and_predicted():
    res = completer()(scene())
    s = res.sizes
    assert s['unit'] == 'cm'
    assert np.isclose(s['measured']['face_width'], 100 * res.clean_meta['width'])
    assert np.isclose(s['measured']['distance'], 100 * res.clean_meta['range_m'])
    assert np.allclose([s['predicted'][k] for k in ('width', 'height', 'depth')], 100 * extents(res.completed_canon, res.T['scale']))
    assert any('predicted' in w for w in res.warnings)
    json.dumps(res.meta('x.pt', 'pilot'))


def test_export_result_says_it_is_not_metric():
    raw = scene()
    r = np.random.default_rng(1)
    ex = raw[r.choice(len(raw), 4096, replace=False)]
    ex = ex / np.abs(ex).max()
    res = completer()(ex)
    assert res.ok and not res.metric and res.sizes['unit'] == 'file units' and res.sizes['measured']['distance'] is None
    assert any('export' in w for w in res.warnings)
    assert len(export_origin(res.clean_meta)) == 3


def test_stage_is_pilot_unless_final():
    assert stage_of('/x/pilot1_best.pt') == 'pilot' and stage_of('/x/best.pt') == 'pilot' and stage_of('/x/final.pt') == 'final'
    assert stage_of('checkpoints/adapointr_final.pth') == 'final'
    res = completer()(scene())
    assert any('pilot' in w for w in res.warnings)
    assert not any('stage' in w for w in completer(stage='final')(scene()).warnings)


def test_rejected_scan_is_not_completed():
    Echo.calls = 0
    noise = np.random.default_rng(1).uniform(-2, 2, (4096, 3))
    res = completer()(noise)
    assert Echo.calls == 0
    assert res.status == 'rejected' and not res.ok and res.reason
    assert res.completed is None and res.partial is None and res.sizes == {} and len(res.observed) == 0 and len(res.background)
    assert res.meta()['n_completed'] is None


def test_json_safe():
    assert json_safe({'a': np.float32(1), 'b': np.nan, 'c': np.array([1, 2])}) == {'a': 1.0, 'b': None, 'c': [1, 2]}


def test_held_out_sessions_are_found_with_their_repeat_exports():
    from majproj.sim_real import holdout_sessions
    from scripts.build_web import session_of
    held = set(holdout_sessions())
    assert held and session_of('FINAL-INPUTS/BOX/scan_20261030_121600.npy') in held
    assert session_of('FINAL-INPUTS/BOTTLE/scan_20261030_123814.npy') in held
    assert session_of('FINAL-INPUTS/BOX/scan_20261030_121018.npy') not in held
