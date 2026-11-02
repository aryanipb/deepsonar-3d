"""Smoke and invariants of the final training stack, on a small fake simulator with the simulate_raw contract."""
import json
import os

import numpy as np
import pytest
import torch

import modal_final
from majproj.canon import apply_canonical, to_canonical
from majproj.train_final import _keep_lowest, augment, load_split, one_sided, renorm, report, run
from scripts.gen_final import GT_STEP, KINDS, REGIMES, code_hash, gen_shard_file, load_config, real_sessions, write_manifest

CKPT = 'checkpoints/AdaPoinTr_PCN.pth'
HAS_CKPT = os.path.exists(CKPT)
CODE = code_hash()      # once: another edit in the tree during the long run must not look like a code change to resume
OFF = dict(mirror_p=0, yaw_deg=0, drop_p=0, drop_keep=[1, 1], crop_p=0, crop_keep=[1, 1], hole_p=0, hole_keep=[1, 1],
           depth_copy_p=0, depth_copy_frac=[.1, .1], depth_copy_m=[.02, .02], jitter_sigma=[0, 0], jitter_clip=.03, scale=[1, 1])


def fake_simulate(rng, kind=None, **kw):
    """A box face dwelled on from about 35 cm plus a ceiling, in the raw file's statistics."""
    w, h, d = rng.uniform(.12, .25), rng.uniform(.15, .3), rng.uniform(.05, .2)
    dist = rng.uniform(.28, .4)
    grid = np.stack(np.meshgrid(np.arange(-w / 2, w / 2, .004), np.arange(-h / 2, h / 2, .004)), -1).reshape(-1, 2)
    face = np.c_[grid[:, 0], np.full(len(grid), dist), grid[:, 1]]
    face = np.repeat(face, 3, axis=0) + rng.normal(0, .002, (len(face) * 3, 3))
    ceil = np.c_[rng.uniform(-1.5, 1.5, (4000, 2)), np.full(4000, 2.8) + rng.normal(0, .01, 4000)]
    raw = np.vstack([face, ceil])
    u = rng.uniform(-.5, .5, (16384, 3))
    a = rng.integers(0, 3, 16384)
    u[np.arange(16384), a] = rng.choice([-.5, .5], 16384)
    complete = (u * [w, d, h] + [0, dist + d / 2, 0]).astype(np.float32)
    return dict(raw=raw, complete=complete, object_mask=np.arange(len(raw)) < len(face),
                truth=dict(kind=kind, size_m=[w, d, h], distance_m=dist, regime='robust' if rng.random() < .2 else 'real',
                           centre=np.array([0, dist, 0.]), n=np.int64(3)))


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root = str(tmp_path_factory.mktemp('ds'))
    cfg, cfg_hash = load_config()
    reports = [gen_shard_file(root, s, 0, cfg, per_shard=n, simulate=fake_simulate) for s, n in (('train', 6), ('val', 3), ('test', 3))]
    return root, cfg, cfg_hash, reports


def test_gen_is_the_deployment_pipeline(data):
    root, cfg, cfg_hash, reports = data
    assert reports[0]['kept'] > 0, reports[0]
    tr = load_split(root, 'train')
    assert tr['partial'].shape[1:] == (2048, 3) and tr['gt'].shape[1:] == (16384, 3) and tr['gt'].dtype == np.int16
    assert (tr['seg_frac'] > 0.5).all()
    assert tr['regime'].shape == tr['kind'].shape and set(tr['regime']) <= {0, 1}
    assert np.isclose(np.ptp(tr['partial'], axis=1)[:, :2].max(1), 0.9, atol=1e-3).all()
    gt = tr['gt'] * GT_STEP
    # the seen face is the near side of the target
    assert (np.median(gt[..., 2], 1) > np.median(tr['partial'][..., 2], 1)).all()
    m = write_manifest(root, cfg_hash)
    assert m['splits']['train']['attempted'] == 6 and len(m['dataset_hash']) == 64
    t = m['splits']['train']
    assert sum(t['kept_by_regime'].values()) == t['kept'] and t['real_frac'] == pytest.approx(t['kept_by_regime'].get('real', 0) / t['kept'], abs=1e-4)
    assert m == write_manifest(root, cfg_hash)


def test_gen_is_deterministic(tmp_path, data):
    _, cfg, _, reports = data
    r = gen_shard_file(str(tmp_path), 'train', 0, cfg, per_shard=6, simulate=fake_simulate)
    assert r['sha256'] == reports[0]['sha256']


def test_rejects_are_counted(tmp_path, data):
    _, cfg, _, _ = data

    def junk(rng, kind=None):
        s = fake_simulate(rng, kind)
        return dict(s, raw=rng.uniform(-3, 3, (500, 3))) if rng.random() < .5 else s

    r = gen_shard_file(str(tmp_path), 'val', 1, cfg, per_shard=8, simulate=junk)
    assert sum(r['rejected'].values()) > 0 and r['kept'] + sum(r['rejected'].values()) == 8


def test_simulator_errors_fail_loudly(tmp_path, data):
    _, cfg, _, _ = data

    def broken(rng, kind=None):
        raise KeyError('x')

    with pytest.raises(RuntimeError):
        gen_shard_file(str(tmp_path), 'val', 2, cfg, per_shard=4, simulate=broken)


def test_regime_is_mandatory(tmp_path, data):
    _, cfg, _, _ = data

    def untagged(rng, kind=None):
        s = fake_simulate(rng, kind)
        s['truth'].pop('regime')
        return s

    with pytest.raises(RuntimeError, match='regime'):
        gen_shard_file(str(tmp_path), 'val', 3, cfg, per_shard=4, simulate=untagged)


def test_only_the_real_regime_selects():
    n = 12
    data = dict(scale=np.full(n, .2), seg_frac=np.ones(n), kind=np.arange(n) % 2,
                regime=np.array([REGIMES.index('real')] * 10 + [REGIMES.index('robust')] * 2))
    good_real = report(np.r_[np.full(10, .01), np.full(2, .9)], data, KINDS)
    good_robust = report(np.r_[np.full(10, .02), np.full(2, .001)], data, KINDS)
    assert good_real['select_cd'] == 10 and good_robust['select_cd'] == 20
    assert good_real['mixed']['mean_cd'] > good_robust['mixed']['mean_cd']       # the mixed number would pick the wrong one
    assert good_real['select_cd'] < good_robust['select_cd']
    wrong = dict(data, seg_frac=np.r_[np.zeros(2), np.ones(n - 2)])          # a wrong accept in 'real' never picks the checkpoint
    assert report(np.r_[np.full(2, 5.), np.full(10, .01)], wrong, KINDS)['select_cd'] == 10
    assert report(np.r_[np.full(2, 5.), np.full(10, .01)], wrong, KINDS)['by_regime']['real']['macro_cd'] > 10    # still logged
    with pytest.raises(RuntimeError):
        report(np.full(n, .01), dict(data, seg_frac=np.zeros(n)), KINDS)
    assert good_real['by_regime']['robust']['n'] == 2 and good_real['by_regime']['real']['n'] == 10


def _rng():
    return torch.Generator().manual_seed(0)


def _batch(B=4):
    x = torch.rand(B, 2048, 3) * torch.tensor([.9, .5, .2]) - .3
    return renorm(x, torch.rand(B, 16384, 3))[:2]


def test_augment_off_is_identity():
    x, y = _batch()
    x2, y2, s = augment(x, y, torch.full((4,), .25), OFF, _rng())
    assert torch.allclose(x2.sort(1).values, x.sort(1).values, atol=1e-5) and torch.allclose(y2, y, atol=1e-5)
    assert torch.allclose(s, torch.full((4,), .25), atol=1e-5)


def test_partial_only_augments_keep_the_target_shape():
    """Drop, crop, hole, depth copies and jitter change the scan, never the object: the target only moves with the frame."""
    x, y = _batch()
    A = dict(OFF, drop_p=1, drop_keep=[.5, .9], crop_p=1, crop_keep=[.6, .9], hole_p=1, hole_keep=[.7, .9], depth_copy_p=1,
             jitter_sigma=[.005, .01])
    x2, y2, s = augment(x, y, torch.full((4,), .25), A, _rng())
    assert x2.shape == x.shape and not torch.isclose(x2, x).all()
    for a, b, sc in zip(y, y2, s):
        r = .25 / sc                                                                          # scale_m stays true
        assert (b - (r * a + (b - r * a).mean(0))).abs().max() < 1e-5                         # uniform scale + shift


def test_result_has_deployment_form():
    x, y = _batch(8)
    A = load_config()[0]['augment']
    x2, y2, _ = augment(x, y, torch.full((8,), .25), A, _rng())
    span = (x2.amax(1) - x2.amin(1))[:, :2].amax(-1)
    assert ((span > .9 * A['scale'][0] - 1e-4) & (span < .9 * A['scale'][1] + 1e-4)).all()
    assert torch.isfinite(x2).all() and torch.isfinite(y2).all()


def test_keep_lowest_tiles():
    x = torch.arange(2048.)[None, :, None].expand(2, -1, 3).clone()
    out = _keep_lowest(x, x[..., 0], torch.tensor([.5, .5]), torch.tensor([True, False]))
    assert out[0, :, 0].max() == 1023 and out[0, :, 0].unique().numel() == 1024
    assert torch.equal(out[1], x[1])


def test_one_sided_is_one_sided():
    obs = torch.rand(1, 100, 3)
    extra = torch.rand(1, 5000, 3) + 10
    assert one_sided(obs, torch.cat([obs, extra], 1)).item() == pytest.approx(0, abs=1e-5)     # extra output is free
    far = obs.clone()
    far[0, :10] += 5
    assert one_sided(far, obs, trim=.1).item() < 1e-4                                          # worst 10 % are ignored
    assert one_sided(far, obs).item() > .4


def test_sessions_counted_once():
    if not os.path.isdir('FINAL-INPUTS'):
        pytest.skip('no real scans')
    s = real_sessions('FINAL-INPUTS')
    assert sum(x['object'] == 'BOX' for x in s) == 11 and sum(x['object'] == 'BOTTLE' for x in s) == 9
    assert 'scan_20261030_121055' in [x['session'] for x in s] and 'scan_20261030_121022' not in [x['session'] for x in s]


def _smoke_cfg():
    cfg, h = load_config()
    cfg['train'].update(batch_size=2, warmup_steps=2, max_steps=3, eval_every=2, ckpt_every=2, log_every=1, eval_batch=2,
                        amp_dtype='bfloat16', final_reserve_s=0)
    cfg['real'].update(batch_size=1)
    return cfg, dict(code=CODE, config=h)


@pytest.mark.skipif(not HAS_CKPT, reason='official weights absent')
def test_train_real_resume_and_limits(data, tmp_path):
    root = data[0]
    cfg, hashes = _smoke_cfg()
    out = str(tmp_path)

    rec = run(cfg, hashes, root, out, 'sim', 3600, CKPT, 'cpu')
    assert rec['steps'] == 3 and 'test' in rec and 'test_official_weights' in rec
    assert run(cfg, hashes, root, out, 'sim', 3600, CKPT, 'cpu')['steps'] == 3                 # a finished run stays finished
    lines = [json.loads(l) for l in open(f'{out}/sim/train_final.jsonl')]
    assert lines[0]['hashes']['dataset'] is not None and [l for l in lines if l['event'] == 'val']
    assert lines[0]['regimes']['val']['real'] > 0
    vals = [l for l in lines if l['event'] == 'val' and l['what'] != 'partial_as_output']
    real = [l for l in vals if l['regime'] == 'real']
    assert real and all(l['selects'] for l in real) and not any(l['selects'] for l in vals if l['regime'] != 'real')
    best = torch.load(f'{out}/sim/best.pt', map_location='cpu', weights_only=False)
    assert best['val']['select_cd'] == min(l['select_cd'] for l in real)

    # stops by itself at the wall limit and leaves a checkpoint
    cfg2, h2 = _smoke_cfg()
    cfg2['train'].update(max_steps=10**6)
    rec = run(cfg2, h2, root, out, 'wall', 30, CKPT, 'cpu')
    assert 0 < rec['steps'] < 10**6 and os.path.exists(f'{out}/wall/final.pt')

    # continues another run with its optimizer state; a different dataset or code hash is refused
    cfg3, h3 = _smoke_cfg()
    cfg3['train'].update(max_steps=1)
    rec = run(cfg3, h3, root, out, 'more', 3600, CKPT, 'cpu', resume_from=f'{out}/sim/latest.pt')
    assert rec['steps'] == 4
    with pytest.raises(RuntimeError, match='refusing'):
        run(cfg3, dict(h3, code='other'), root, out, 'other', 3600, CKPT, 'cpu', resume_from=f'{out}/sim/latest.pt')

    # same run id after a restart: continues and the clock keeps counting
    st = torch.load(f'{out}/sim/latest.pt', map_location='cpu', weights_only=False)
    os.makedirs(f'{out}/restart')
    st['wall_used_s'] = 10**6
    torch.save(st, f'{out}/restart/latest.pt')
    rec = run(cfg, hashes, root, out, 'restart', 3600, CKPT, 'cpu')
    assert rec['steps'] == 3

    # real stage: held-out sessions are scored at the end and never trained on
    cfg4, h4 = _smoke_cfg()
    cfg4['real']['enabled'] = True
    rec = run(cfg4, h4, root, out, 'real', 3600, CKPT, 'cpu', input_root='FINAL-INPUTS', init_from=f'{out}/sim/final.pt')
    rows = {r['session']: r for r in rec['real']}
    assert not rows['scan_20261030_121600']['used_in_training'] and not rows['scan_20261030_123815']['used_in_training']
    assert sum(r['used_in_training'] for r in rec['real']) == len(rows) - 2
    lines = [json.loads(l) for l in open(f'{out}/real/train_final.jsonl')]
    assert all(s not in lines[0]['real_train'] for s in cfg4['real']['holdout'])
    assert any(l.get('real') is not None for l in lines if l['event'] == 'train')


def test_guard(tmp_path):
    l = str(tmp_path / 'b.log')
    modal_final.guard('a', 10, run_s=3600, billed=8.66, ledger=l, now=1000)
    with pytest.raises(SystemExit):
        modal_final.guard('b', 6, run_s=60, billed=8.66, ledger=l, now=1000 + 4000)     # a: 3600 s run + 1200 s billing lag
    modal_final.guard('c', 6, run_s=60, billed=8.66, ledger=l, now=1000 + 5000)     # a is billed by now
    with pytest.raises(SystemExit):
        modal_final.guard('d', 16, billed=8.66, ledger=l, now=10**7)
    rows = [json.loads(x) for x in open(l)]
    assert [r['launched'] for r in rows] == [True, False, True, False]


def test_modal_functions_obey_the_rules():
    src = open('modal_final.py').read()
    assert 'schedule' not in src.split('app = modal.App')[1] and 'min_containers' not in src and "'H100'" in src
    assert modal_final.LIMITS['retries'] == 0 and modal_final.LIMITS['max_containers'] == 1
    assert modal_final.train.with_options(timeout=60) is not None
    assert modal_final.GEN_CPU <= 16


def test_sample_weights_set_the_batch_mix():
    from majproj.train_final import draw, sample_weights
    names = ('a', 'b', 'c')
    labels = np.array([0] * 90 + [1] * 10)                      # c is absent, b is rare
    assert sample_weights(labels, None, names) is None
    w = sample_weights(labels, {'b': 3.0}, names)               # a:1, b:3, c:0 -> b drawn 75 % of the time
    assert w.shape == (100,) and abs(float(w.sum()) - 1) < 1e-6
    g = torch.Generator().manual_seed(0)
    idx = draw(100, w, 20000, torch.device('cpu'), g)
    assert abs(float((labels[idx.numpy()] == 1).mean()) - 0.75) < 0.02
    assert draw(100, None, 8, torch.device('cpu'), g).shape == (8,)
    with pytest.raises(ValueError):
        sample_weights(labels, {'a': 0, 'b': 0}, names)


@pytest.mark.skipif(not HAS_CKPT, reason='official weights absent')
def test_real_targets_train_two_sided(data, tmp_path):
    root = data[0]
    cfg, h = _smoke_cfg()
    cfg['real'].update(enabled=True, batch_size=4, targets={'BOTTLE': 'configs/real_bottle_parts.json'},
                       object_weights={'BOTTLE': 1, 'BOX': 0})
    rec = run(cfg, h, root, str(tmp_path), 'tg', 3600, CKPT, 'cpu', input_root='FINAL-INPUTS')
    lines = [json.loads(l) for l in open(f'{tmp_path}/tg/train_final.jsonl')]
    start = lines[0]
    acc = [k for k, v in start['real_targets'].items() if v['accepted']]
    assert len(acc) >= 5 and 'scan_20261030_123815' in start['real_targets']           # the held-out bottle gets a target too
    assert all(k not in start['real_train'] for k in cfg['real']['holdout'])
    assert any(l.get('real_targets_in_batch', 0) > 0 for l in lines if l['event'] == 'train')
    rows = {r['session']: r for r in rec['real']}
    held = rows['scan_20261030_123815']
    assert not held['used_in_training'] and held.get('cd_vs_constructed_target_mm', 0) > 0 and 'registered' in held['target_source']
    assert all('cd_vs_constructed_target_mm' not in r for r in rec['real'] if r['object'] == 'BOX')
