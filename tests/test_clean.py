"""Tests for majproj.clean: the signal-defined object finder.

Synthetic scenes are built in metres the way the device builds a file: a 10 mm voxel-saturated object surface plus
ceiling zone strips, pick-up streaks, hand-drift depth copies and random junk; the map is then sampled to 4096 points
WITH replacement (duplicates), bounding-box centred, divided by the max-abs coordinate and shuffled. z = up.

    uv run --with pytest pytest -q tests/test_clean.py
"""
import glob
import os
import time

import numpy as np
import pytest

from majproj.clean import Cleaned, _Surface, _densest_slab, _is_drift_plateau, clean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOX = 0.01                                       # the viewer's voxel thinning, metres

META_KEYS = {'status', 'reason', 'confidence', 'n_input', 'n_unique', 'n_object', 'n_removed', 'eps', 'tests', 'normal',
             'centre', 'width', 'height', 'shape_hint', 'candidates', 'frame', 'n_depth_copies', 'aspect_w_over_h',
             'plane_sd'}


# ----------------------------------------------------------------------------- synthetic scene parts (metres)
def sheet(rng, centre=(0.0, 0.3, 0.0), w=0.15, h=0.18, depth_sd=0.003, keep=1.0, normal_y=True):
    """Vertical rectangle facing -y, saturated on the voxel grid (one point per 10 mm voxel, jittered)."""
    u, v = np.meshgrid(np.arange(-w / 2, w / 2, VOX), np.arange(-h / 2, h / 2, VOX))
    u, v = u.ravel(), v.ravel()
    m = rng.uniform(size=len(u)) < keep
    u, v = u[m] + rng.uniform(-VOX / 3, VOX / 3, m.sum()), v[m] + rng.uniform(-VOX / 3, VOX / 3, m.sum())
    d = rng.normal(0, depth_sd, len(u))
    pts = np.c_[u, d, v] if normal_y else np.c_[d, u, v]
    return pts + np.asarray(centre)


def cylinder_front(rng, centre=(0.0, 0.3, 0.0), radius=0.04, h=0.2, arc_deg=120):
    """Front patch of an upright cylinder (bottle) facing -y, saturated on the voxel grid."""
    a = np.radians(arc_deg) / 2
    arc = np.arange(-a * radius, a * radius, VOX)
    s, z = np.meshgrid(arc, np.arange(-h / 2, h / 2, VOX))
    s, z = s.ravel() + rng.uniform(-VOX / 3, VOX / 3, s.size), z.ravel() + rng.uniform(-VOX / 3, VOX / 3, s.size)
    th = s / radius
    r = radius + rng.normal(0, 0.003, len(s))
    return np.c_[r * np.sin(th), -r * np.cos(th) + radius, z] + np.asarray(centre)


def ceiling(rng, height=1.6, n_strips=8, per_strip=40):
    """Sparse zone-grid strips on a horizontal ceiling (seen while the device lay pointing up)."""
    out = []
    for k in range(n_strips):
        x = -0.6 + 1.2 * k / (n_strips - 1)
        y = rng.uniform(-0.6, 0.6, per_strip)
        out.append(np.c_[x + rng.normal(0, 0.01, per_strip), y, height + rng.normal(0, 0.01, per_strip)])
    return np.concatenate(out)


def streaks(rng, n=12, per=12):
    """Line-like pick-up / put-down streaks: a few points per zone along random segments."""
    out = []
    for _ in range(n):
        a, b = rng.uniform(-0.6, 0.6, 3) * [1, 1, 1.2] + [0, 0, 0.4], rng.uniform(-0.6, 0.6, 3) * [1, 1, 1.2] + [0, 0, 0.4]
        t = rng.uniform(0, 1, per)[:, None]
        out.append(a + t * (b - a) + rng.normal(0, 0.004, (per, 3)))
    return np.concatenate(out)


def junk(rng, n=250):
    return rng.uniform([-0.7, -0.7, -0.4], [0.7, 0.7, 1.7], (n, 3))


def export(parts, rng, n=4096):
    """parts: list of (points, label). Mimics the viewer export; returns (file points, labels)."""
    p = np.concatenate([q for q, _ in parts])
    lab = np.concatenate([np.full(len(q), l) for q, l in parts])
    pick = rng.choice(len(p), n, replace=len(p) < n)
    p, lab = p[pick], lab[pick]
    p = p - (p.max(0) + p.min(0)) / 2
    p = p / np.abs(p).max()
    order = rng.permutation(n)
    return p[order].astype(np.float32), lab[order]


def labels_of(points, file_pts, file_lab):
    """Labels of returned points, by exact coordinate match against the exported file."""
    lut = {tuple(r): l for r, l in zip(file_pts.tolist(), file_lab.tolist())}
    return np.array([lut[tuple(r)] for r in np.asarray(points, np.float32).tolist()])


def full_scene(seed, obj=None):
    rng = np.random.default_rng(seed)
    obj = sheet(rng) if obj is None else obj
    copies = []                                   # hand drift: thinned copies of the object shifted along the view (y)
    for dy in (-0.035, 0.05):
        c = obj[rng.uniform(size=len(obj)) < 0.3] + [0.0, dy, 0.0]
        copies.append(c + rng.normal(0, 0.002, c.shape))
    parts = [(obj, 'object'), (ceiling(rng), 'ceiling'), (streaks(rng), 'streak'), (junk(rng), 'junk')]
    parts += [(c, 'copy') for c in copies]
    return export(parts, rng)


def precision_recall(r, pts, lab):
    got = labels_of(r.object, pts, lab)
    uniq_obj = len(np.unique(pts[lab == 'object'], axis=0))
    tp = int(np.sum(got == 'object'))
    return tp / max(len(got), 1), tp / uniq_obj


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_recovers_sheet_from_cluttered_scene(seed):
    pts, lab = full_scene(seed)
    r = clean(pts)
    assert isinstance(r, Cleaned)
    m = r.meta
    assert m['status'] == 'ok', m['reason']
    prec, rec = precision_recall(r, pts, lab)
    assert prec >= 0.95, (prec, rec)
    assert rec >= 0.6, (prec, rec)
    assert m['shape_hint'] == 'flat'
    assert abs(m['normal'][1]) > 0.95                       # faces along y
    assert 0.6 < m['aspect_w_over_h'] < 1.0                  # 0.15 x 0.18 m
    assert m['n_depth_copies'] > 0
    assert 0.5 < m['confidence'] <= 1.0


@pytest.mark.parametrize('seed', [7, 8, 9])
def test_recovers_curved_front_patch(seed):
    """A bottle-like curved patch among ceiling strips, streaks and junk (no drift copies)."""
    rng = np.random.default_rng(seed)
    parts = [(cylinder_front(rng), 'object'), (ceiling(rng), 'ceiling'), (streaks(rng), 'streak'), (junk(rng), 'junk')]
    pts, lab = export(parts, rng)
    r = clean(pts)
    assert r.meta['status'] == 'ok', r.meta['reason']
    prec, rec = precision_recall(r, pts, lab)
    assert prec >= 0.95 and rec >= 0.8, (prec, rec)
    assert r.meta['shape_hint'] == 'curved'
    assert abs(r.meta['normal'][1]) > 0.9


def test_curved_patch_with_heavy_drift_is_a_known_limit():
    """KNOWN LIMIT, pinned so a change in behaviour is noticed: a 7 cm wide curved patch with two 30 % drift copies
    3.5 / 5 cm away. A copy shifted along the view lands within noise of a curved surface near its rim, so no layer
    test can separate it there; we only require that the object is found and is the majority of what is returned."""
    rng = np.random.default_rng(7)
    pts, lab = full_scene(7, obj=cylinder_front(rng))
    r = clean(pts)
    if r.meta['status'] == 'ok':
        prec, _ = precision_recall(r, pts, lab)
        assert prec >= 0.5, prec
    else:
        assert 'ambiguous' in r.meta['reason']


@pytest.mark.parametrize('seed', range(5))
def test_pure_noise_is_rejected(seed):
    rng = np.random.default_rng(seed)
    for p in (rng.uniform(-1, 1, (4096, 3)), rng.normal(0, 0.3, (4096, 3))):
        r = clean(p)
        assert r.meta['status'] == 'rejected'
        assert r.meta['reason']
        assert r.object.shape == (0, 3) and r.object.dtype == np.float32
        assert len(r.removed) == r.meta['n_unique']


def test_noise_with_clutter_but_no_object_is_rejected():
    rng = np.random.default_rng(11)
    pts, _ = export([(ceiling(rng), 'ceiling'), (streaks(rng), 'streak'), (junk(rng, 600), 'junk')], rng)
    assert clean(pts).meta['status'] == 'rejected'


def test_two_equal_objects_side_by_side_are_rejected():
    rng = np.random.default_rng(5)
    a = sheet(rng, centre=(-0.3, 0.3, 0.0))
    b = sheet(rng, centre=(0.3, 0.3, 0.0))
    pts, _ = export([(a, 'a'), (b, 'b'), (ceiling(rng), 'ceiling'), (junk(rng), 'junk')], rng)
    r = clean(pts)
    assert r.meta['status'] == 'rejected'
    assert 'ambiguous' in r.meta['reason']


@pytest.mark.parametrize('offset', [0.06, 0.12])
def test_parallel_depth_copy_is_absorbed_not_a_competitor(offset):
    """A whole 70 % copy of the face behind it is hand drift, not a second object. At 6 cm it lies inside the growth
    band (removed by the mode layer); at 12 cm it is beyond the band and grows as its own candidate, which must be
    folded into the face's depth copies (otherwise its score ratio ~0.7 would make the file ambiguous)."""
    rng = np.random.default_rng(9)
    face = sheet(rng)
    copy = sheet(rng, centre=(0.0, 0.3 + offset, 0.0), keep=0.7)
    pts, lab = export([(face, 'object'), (copy, 'copy'), (ceiling(rng), 'ceiling'), (junk(rng), 'junk')], rng)
    r = clean(pts)
    assert r.meta['status'] == 'ok', r.meta['reason']
    prec, _ = precision_recall(r, pts, lab)
    assert prec >= 0.95
    assert r.meta['n_depth_copies'] >= 100                  # the copy was folded in, not left as a rival


def test_wide_horizontal_plane_is_not_the_object():
    rng = np.random.default_rng(3)
    xs, ys = np.meshgrid(np.arange(-0.6, 0.6, VOX), np.arange(-0.6, 0.6, VOX))
    table = np.c_[xs.ravel(), ys.ravel(), np.full(xs.size, -0.2) + rng.normal(0, 0.002, xs.size)]
    pts, _ = export([(table, 'table'), (ceiling(rng), 'ceiling'), (junk(rng), 'junk')], rng)
    r = clean(pts)
    assert r.meta['status'] == 'rejected'
    assert any(not c['passed'] and 'not_scene_plane' in c['failed'] for c in r.meta['candidates'])


def test_sheet_standing_on_dense_table():
    """The spec's selection case: a saturated table spanning the scene outscores the sheet on size, but is a
    gravity-aligned scene plane; the sheet standing on it must be chosen and must not glue to the table at its foot."""
    rng = np.random.default_rng(13)
    xs, ys = np.meshgrid(np.arange(-0.6, 0.6, VOX), np.arange(-0.6, 0.6, VOX))
    keep = rng.uniform(size=xs.size) < 0.5
    table = np.c_[xs.ravel()[keep], ys.ravel()[keep], np.full(keep.sum(), -0.09) + rng.normal(0, 0.002, keep.sum())]
    face = sheet(rng)                                # spans z -0.09 .. 0.09: stands on the table
    pts, lab = export([(face, 'object'), (table, 'table'), (ceiling(rng), 'ceiling'), (junk(rng), 'junk')], rng)
    r = clean(pts)
    assert r.meta['status'] == 'ok', r.meta['reason']
    prec, rec = precision_recall(r, pts, lab)
    assert prec >= 0.95 and rec >= 0.5, (prec, rec)
    assert any('not_scene_plane' in c['failed'] for c in r.meta['candidates'])


def test_deterministic_and_order_independent():
    pts, _ = full_scene(4)
    a, b = clean(pts), clean(pts)
    c = clean(pts[np.random.default_rng(0).permutation(len(pts))])
    for r in (b, c):
        assert r.meta == a.meta
        np.testing.assert_array_equal(np.sort(r.object, 0), np.sort(a.object, 0))
        np.testing.assert_array_equal(np.sort(r.removed, 0), np.sort(a.removed, 0))


def test_api_contract_ok_and_rejected():
    ok = clean(full_scene(1)[0])
    bad = clean(np.random.default_rng(0).uniform(-1, 1, (4096, 3)))
    for r in (ok, bad):
        m = r.meta
        assert META_KEYS <= set(m)
        assert m['frame'] == 'z up'
        assert m['status'] in ('ok', 'rejected')
        assert 0.0 <= m['confidence'] <= 1.0
        assert r.object.dtype == np.float32 and r.removed.dtype == np.float32
        assert r.object.ndim == 2 and r.object.shape[1] == 3
        assert m['n_object'] == len(r.object) and m['n_removed'] == len(r.removed)
        assert m['n_object'] + m['n_removed'] == m['n_unique']
        for t in m['tests'].values():
            assert set(t) == {'value', 'pass'} and isinstance(t['pass'], bool)
        assert isinstance(m['candidates'], list)
    assert ok.meta['reason'] is None and ok.meta['shape_hint'] in ('flat', 'curved') and len(ok.meta['normal']) == 3
    assert bad.meta['reason'] and bad.meta['normal'] is None and bad.meta['shape_hint'] is None


def test_bad_input():
    with pytest.raises(ValueError):
        clean(np.zeros((10, 2)))
    with pytest.raises(TypeError):
        clean(np.zeros((100, 3)), no_such_param=1)
    r = clean(np.full((4096, 3), np.nan))
    assert r.meta['status'] == 'rejected' and r.meta['n_unique'] == 0


def smeared_scene(rng, depth, n=1500, w=0.18, h=0.22):
    """A face 30 cm from the sensor at the origin, seen from a hand creeping `depth` metres along the line of sight:
    each hit lands at a continuous random depth (a plateau, no dominant layer), then the 1 cm thinning of a raw map."""
    f = np.c_[rng.uniform(-w / 2, w / 2, n), 0.30 + rng.uniform(-depth / 2, depth / 2, n), rng.uniform(-h / 2, h / 2, n)]
    _, i = np.unique(np.floor(f / VOX).astype(np.int64), axis=0, return_index=True)
    return f[i]


def blob_scene(rng, side=0.2):
    """A volume of the same size in every direction: not a surface, however it is viewed."""
    g = np.arange(-side / 2, side / 2, VOX)
    v = np.stack(np.meshgrid(g, g, g), -1).reshape(-1, 3)
    return v[rng.uniform(size=len(v)) < 0.35] + [0.0, 0.30, 0.0]


def test_hand_drift_plateau_with_known_origin_keeps_one_thin_face():
    """Real failure of box session 121018/121022/121055: a 6-10 cm depth plateau over the face fails `smooth` and a
    sliver wins (or the file is rejected). With the sensor position known it is one face, one surface thick."""
    rng = np.random.default_rng(23)
    pts = np.concatenate([smeared_scene(rng, 0.10), ceiling(rng, 2.7, per_strip=20), junk(rng, 3000)])
    r = clean(pts, origin=(0.0, 0.0, 0.0))
    assert r.meta['status'] == 'ok', r.meta['reason']
    assert r.meta['layer_mode'] == 'drift_slab'
    assert r.meta['surface_rms'] < 0.015                      # one surface thick (the plateau's own sd is 2.9 cm)
    assert r.meta['width'] > 0.13 and r.meta['height'] > 0.18


def rod_scene(rng, tilt_deg=0.0):
    """A rod 30 cm from the sensor at the origin, tilted in the view plane, and a real background surface 85 cm away."""
    rod = sheet(rng, centre=(0.0, 0.0, 0.0), w=0.03, h=0.5, depth_sd=0.002)
    t = np.radians(tilt_deg)
    rod = rod @ np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]).T + [0.0, 0.30, 0.1]
    back = sheet(rng, centre=(-0.35, 0.80, 0.1), w=0.25, h=0.25, keep=0.8)
    return np.concatenate([rod, back, rod + rng.normal(0, 0.003, rod.shape), junk(rng, 800)])


def test_upright_rod_with_known_origin_beats_the_background():
    """Real-regime rods failed `not_line` and the cleaner returned the background surface behind them."""
    rng = np.random.default_rng(31)
    r = clean(rod_scene(rng), origin=(0.0, 0.0, 0.0))
    assert r.meta['status'] == 'ok', r.meta['reason']
    assert np.linalg.norm(r.object.mean(0)[:2] - [0.0, 0.30]) < 0.03


@pytest.mark.parametrize('origin, tilt', [(None, 0.0), ((0.0, 0.0, 0.0), 45.0)])
def test_rod_rule_is_narrow(origin, tilt):
    """Without a known origin, or for a line far from vertical, a line is still not a surface."""
    rng = np.random.default_rng(31)
    r = clean(rod_scene(rng, tilt), origin=origin)
    assert r.object.size == 0 or np.linalg.norm(r.object.mean(0)[:2] - [0.0, 0.30]) > 0.2


def test_drift_rule_needs_a_known_origin():
    rng = np.random.default_rng(23)
    pts = np.concatenate([smeared_scene(rng, 0.10), ceiling(rng, 2.7, per_strip=20), junk(rng, 3000)])
    assert 'layer_mode' not in clean(pts).meta


def test_volume_as_deep_as_wide_is_not_a_drift_plateau():
    rng = np.random.default_rng(22)
    pts = np.concatenate([blob_scene(rng), ceiling(rng, 2.7, per_strip=20), junk(rng, 3000)])
    r = clean(pts, origin=(0.0, 0.0, 0.0))
    assert 'layer_mode' not in r.meta
    assert r.meta['status'] == 'rejected', r.meta


def test_drift_plateau_gate_geometry():
    """The physical gate itself: a slab facing the sensor passes; the same slab seen edge-on, or a cube, does not."""
    rng = np.random.default_rng(1)
    P = dict(normal_cos=np.cos(np.radians(45)), drift_aspect=0.85)
    slab = np.c_[rng.uniform(-0.09, 0.09, 2000), 0.3 + rng.uniform(-0.04, 0.04, 2000), rng.uniform(-0.11, 0.11, 2000)]
    surf = _Surface(slab, quadric=False)
    origin = (0.0, 0.0, 0.0)
    assert _is_drift_plateau(slab, surf.residual(slab), surf, origin, P)
    assert not _is_drift_plateau(slab, surf.residual(slab), surf, None, P)
    side = (0.3, 0.3, 0.0)                                    # sensor off to the side: depth is not along the sight line
    assert not _is_drift_plateau(slab, surf.residual(slab), surf, side, P)
    cube = rng.uniform(-0.1, 0.1, (2000, 3)) + [0.0, 0.3, 0.0]
    surf = _Surface(cube, quadric=False)
    assert not _is_drift_plateau(cube, surf.residual(cube), surf, origin, P)


def test_densest_slab():
    r = np.r_[np.linspace(-0.05, 0.05, 101), np.full(50, 0.02)]
    keep = _densest_slab(r, 0.01)
    assert keep[-50:].all() and np.ptp(r[keep]) <= 0.02 + 1e-12


# ----------------------------------------------------------------------------- real scans (skipped if absent)
REAL = sorted(glob.glob(os.path.join(ROOT, 'inputs', 'scan_*_shapenet.npy'))
              + glob.glob(os.path.join(ROOT, 'inputs_box', 'scan_*_shapenet.npy')))
GOOD_BOXES = ['140721', '140942', '141318', '141329']


@pytest.mark.skipif(not REAL, reason='real scans not present')
def test_real_scans_fast_and_boxes_found():
    res = {}
    for f in REAL:
        x = np.load(f)
        t = time.perf_counter()
        r = clean(x)
        assert time.perf_counter() - t < 2.0, f
        res[os.path.basename(f).split('_')[2]] = r.meta
    for k in GOOD_BOXES:
        if k not in res:
            continue
        m = res[k]
        assert m['status'] == 'ok', (k, m['reason'])
        assert m['shape_hint'] == 'flat'
        assert abs(m['normal'][2]) < 0.4 and abs(m['normal'][1]) > 0.9          # a vertical face
        assert 0.8 < m['aspect_w_over_h'] < 1.15, (k, m['aspect_w_over_h'])
    if '140248' in res:                                                         # no clean sheet in this file
        m = res['140248']
        assert m['status'] == 'rejected' or m['confidence'] < 0.5
