"""Fine-tune AdaPoinTr from the official PCN weights on ds_final (scripts/gen_final.py). Runs anywhere; modal_final.py
only wraps `run` with the GPU and the Volume.

Inputs are the deployment format (canonical frame, 2048 points). The network sees geometry only: no kind, no size, no
label. Kinds are used for sampling the data and for reporting per-kind validation numbers, nothing else.

Validation is reported per regime ('real': fitted to the measured sessions, 'robust': wider ranges). The best checkpoint
is chosen on the 'real' regime alone, on the scans where the cleaner kept the object; 'robust', the mixed numbers and
the all-scans 'real' number are logged and never select anything.

Loss: model.get_loss (sparse + dense Chamfer L1 plus AdaPoinTr's own denoising term). Chamfer runs in float32 outside
autocast, because shims.ChamferDistanceL1 clamps squared distances at 1e-12, which vanishes in float16.

Augmentation (GPU, per scan). Everything that changes the seen points is followed by `renorm`, which recomputes the
canonical centre and scale from the augmented partial and moves the target with it, so a sample always has exactly the
form that inference produces.
    partial and target together: mirror, yaw (error of the view direction), scale.
    partial only: drop, half-plane crop, local hole, depth copies (hand drift), jitter. Deliberately NOT applied to the
    target: the object did not change, only the scan of it did.

Real stage (real.enabled): each batch adds scans of the training sessions with a one-sided Chamfer, from each observed
point to its nearest output point. That only asks the output to cover what was measured; no target is invented, and
the unseen sides stay supervised by the simulation alone. real.holdout sessions never enter training or selection; they
are scored once at the end and every number is labelled with `used_in_training`.

Hard limits. `wall_limit_s` covers data loading too and survives a container restart: restarts reuse the run directory
and continue the clock from the checkpoint. SIGINT and SIGTERM stop the loop and save.
"""
import copy
import glob
import json
import math
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from scripts.gen_final import GT_STEP, KINDS, N_TARGET, REGIMES, real_pool

from .model import load_adapointr
from .model.shims import chamfer_sqdist

SEEN_SPAN = 0.9
SELECT_SEG_FRAC = 0.5


# ------------------------------------------------------------------------------------------------- augmentation

def _u(rng, lo, hi, *shape):
    return lo + (hi - lo) * torch.rand(*shape, device=rng.device, generator=rng)


def _hit(rng, p, B):
    return torch.rand(B, device=rng.device, generator=rng) < p


def renorm(x, y):
    """Canonical centre and scale of the partial x (B N 3) -> (x, y, factor), where factor is the old units per new unit."""
    lo, hi = x.amin(1, keepdim=True), x.amax(1, keepdim=True)
    c = (lo + hi) / 2
    s = (hi - lo)[..., :2].amax(-1, keepdim=True).clamp_min(1e-6) / SEEN_SPAN
    return (x - c) / s, (y - c) / s, s


def _keep_lowest(x, score, frac, hit):
    """Keep the fraction `frac` of points with the lowest score, in the scans where `hit`, tiling back to N points."""
    B, N, _ = x.shape
    k = (frac * N).long().clamp_min(16)
    order = score.argsort(1)
    idx = order.gather(1, torch.arange(N, device=x.device)[None].expand(B, -1) % k[:, None])
    idx = torch.where(hit[:, None], idx, torch.arange(N, device=x.device)[None])
    return x.gather(1, idx[..., None].expand(-1, -1, 3))


def _rot_y(a):
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(len(a), 3, 3, device=a.device)
    R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1, -s, c
    return R


def augment(x, y, scale_m, A, rng):
    """x B 2048 3 partial, y B M 3 target, scale_m B metres per canonical unit, A the `augment` config."""
    B, N, _ = x.shape
    dev = x.device

    flip = torch.where(_hit(rng, A['mirror_p'], B), -1.0, 1.0)[:, None, None]
    m = torch.tensor([1.0, 0, 0], device=dev)[None, None]
    x, y = x * (1 + (flip - 1) * m), y * (1 + (flip - 1) * m)

    R = _rot_y(_u(rng, -1, 1, B) * math.radians(A['yaw_deg']))
    x, y = x @ R.transpose(1, 2), y @ R.transpose(1, 2)

    x = _keep_lowest(x, torch.rand(B, N, device=dev, generator=rng), _u(rng, *A['drop_keep'], B), _hit(rng, A['drop_p'], B))

    a = _u(rng, 0, 2 * math.pi, B)
    plane = x[..., 0] * torch.cos(a)[:, None] + x[..., 1] * torch.sin(a)[:, None]
    x = _keep_lowest(x, plane, _u(rng, *A['crop_keep'], B), _hit(rng, A['crop_p'], B))

    seed = x[torch.arange(B), torch.randint(0, N, (B,), device=dev, generator=rng)][:, None]
    x = _keep_lowest(x, -(x - seed).norm(dim=-1), _u(rng, *A['hole_keep'], B), _hit(rng, A['hole_p'], B))

    # depth copies: a random part of the surface is stored again at a shifted depth, in place of other points
    frac = _u(rng, *A['depth_copy_frac'], B)[:, None]
    mask = (torch.rand(B, N, device=dev, generator=rng) < frac) & _hit(rng, A['depth_copy_p'], B)[:, None]
    src = torch.randint(0, N, (B, N), device=dev, generator=rng)
    shift = (_u(rng, *A['depth_copy_m'], B) * torch.where(_hit(rng, 0.5, B), -1.0, 1.0) / scale_m)[:, None]
    copied = x.gather(1, src[..., None].expand(-1, -1, 3)).clone()
    copied[..., 2] += shift
    x = torch.where(mask[..., None], copied, x)

    sigma = _u(rng, *A['jitter_sigma'], B)[:, None, None]
    noise = torch.randn(x.shape, device=dev, generator=rng) * sigma
    x = x + noise.clamp(-A['jitter_clip'], A['jitter_clip'])

    x, y, s = renorm(x, y)
    z = _u(rng, *A['scale'], B)[:, None, None]
    x, y = x * z, y * z

    order = torch.rand(B, N, device=dev, generator=rng).argsort(1)
    x = x.gather(1, order[..., None].expand(-1, -1, 3))
    return x, y, scale_m * s[:, 0, 0] / z[:, 0, 0]


# ------------------------------------------------------------------------------------------------- losses and metrics

def one_sided(obs, pred, trim=0.0):
    """Mean distance from each observed point to its nearest output point, per scan, ignoring the worst `trim` fraction."""
    d = (obs - pred.gather(1, torch.cdist(obs, pred).argmin(-1)[..., None].expand(-1, -1, 3))).norm(dim=-1)
    keep = max(1, int(round(d.shape[1] * (1 - trim))))
    return d.sort(1).values[:, :keep].mean(1)


def cd_per_scan(pred, gt):
    d1, d2 = chamfer_sqdist(pred, gt)
    return (d1.clamp_min(1e-12).sqrt().mean(1) + d2.clamp_min(1e-12).sqrt().mean(1)) / 2


# ------------------------------------------------------------------------------------------------- data

def load_split(root, split, workers=8):
    """All shards of a split -> dict of numpy arrays. Targets stay int16 until a batch is drawn."""
    files = sorted(glob.glob(os.path.join(root, split, 'shard_*.npz')))
    files = [f for f in files if not f.endswith('.tmp.npz')]
    if not files:
        return None

    def read(f):
        with np.load(f) as z:
            return {k: z[k] for k in z.files}

    with ThreadPoolExecutor(workers) as ex:
        parts = list(ex.map(read, files))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def select(data, mask):
    return {k: v[mask] for k, v in data.items()}


def sample_weights(labels, weights, names):
    """Per-sample draw weights so that a batch's label mix follows `weights` (label name -> weight, missing = 1).
    None when no weights are given: every sample equally likely. Labels are integer indices into `names`."""
    if not weights:
        return None
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=len(names)).astype(np.float64)
    per_label = np.array([float(weights.get(n, 1.0)) if counts[i] else 0.0 for i, n in enumerate(names)])
    if per_label.sum() <= 0:
        raise ValueError('sample weights leave nothing to draw')
    w = (per_label / np.maximum(counts, 1))[labels]
    return torch.tensor(w / w.sum(), dtype=torch.float32)


def draw(n, weights, batch, dev, rng):
    """Batch indices: uniform, or multinomial with replacement when `weights` is set."""
    if weights is None:
        return torch.randint(0, n, (batch,), device=dev, generator=rng).cpu()
    return torch.multinomial(weights.to(dev), batch, replacement=True, generator=rng).cpu()


def macro(per, kinds):
    return float(np.mean([per[kinds == k].mean() for k in np.unique(kinds)]))


@torch.no_grad()
def evaluate(model, data, dev, amp_ctx, batch, identity=False):
    """Per-scan Chamfer L1 in canonical units, on the unaugmented deployment input."""
    model.eval()
    per = []
    for i in range(0, len(data['partial']), batch):
        x = torch.from_numpy(data['partial'][i:i + batch]).to(dev)
        y = torch.from_numpy(data['gt'][i:i + batch].astype(np.float32) * GT_STEP).to(dev)
        if identity:
            pred = x
        else:
            with amp_ctx():
                pred = model(x)[1]
        per.append(cd_per_scan(pred.float(), y).cpu().numpy())
    return np.concatenate(per).astype(np.float64)


def summary(per, data, kind_names):
    """Mean and median by kind, in canonical units x 1000 and in millimetres, plus the scans the cleaner got right."""
    mm = per * data['scale'] * 1000
    k = data['kind']
    table = {kind_names[i]: dict(n=int((k == i).sum()), cd=round(float(per[k == i].mean() * 1000), 3),
                                 cd_median=round(float(np.median(per[k == i]) * 1000), 3),
                                 mm=round(float(mm[k == i].mean()), 3)) for i in np.unique(k)}
    good = data['seg_frac'] >= 0.5
    return dict(n=len(per), macro_cd=round(macro(per, k) * 1000, 3), mean_cd=round(float(per.mean() * 1000), 3),
                median_cd=round(float(np.median(per) * 1000), 3), mean_mm=round(float(mm.mean()), 3),
                seg_good_frac=round(float(good.mean()), 4),
                seg_good_cd=round(float(per[good].mean() * 1000), 3) if good.any() else None, by_kind=table)


@torch.no_grad()
def real_report(model, pool, dev, amp_ctx, used):
    """Measured on real scans only: how far the observed points are from the output, and the output's extents. The back
    of a real object was never measured, so no number here says anything about the accuracy of the back."""
    from .canon import from_canonical
    model.eval()
    rows = []
    for s in pool:
        x = torch.from_numpy(s['partial'][None]).to(dev)
        with amp_ctx():
            pred = model(x)[1].float()
        d = one_sided(x, pred)[0].item() * s['T']['scale'] * 1000
        out = from_canonical(pred[0].cpu().numpy(), s['T'])
        e = (np.percentile(pred[0].cpu().numpy(), 99, 0) - np.percentile(pred[0].cpu().numpy(), 1, 0)) * s['T']['scale']
        row = dict(session=s['session'], object=s['object'], used_in_training=s['session'] in used,
                   observed_to_output_mm=round(d, 2), output_width_m=round(float(e[0]), 4),
                   output_height_m=round(float(e[1]), 4), output_depth_m_unmeasured=round(float(e[2]), 4),
                   output_centre_m=out.mean(0).round(4).tolist())
        if s.get('gt') is not None:                     # agreement with the registered parametric object, not with a measured back
            g = torch.from_numpy(np.asarray(s['gt'], np.float32)[None]).to(dev)
            row.update(cd_vs_constructed_target_mm=round(cd_per_scan(pred, g)[0].item() * s['T']['scale'] * 1000, 2),
                       target_source=s['target_info']['target_source'], target_registration_median_mm=s['target_info']['resid_median_mm'])
        rows.append(row)
    return rows


def report(per, data, kind_names):
    """summary() over all scans ('mixed') and for each regime present. select_cd is the only number that picks a checkpoint:
    macro Chamfer over the 'real' regime scans whose cleaned points are on the object. A wrong accept pairs background
    points with a target far away, which no network can fix, so it would only add noise to the choice. The all-scans
    numbers stay in by_regime: they are what deployment sees."""
    r = data['regime']
    by_regime = {name: summary(per[r == i], select(data, r == i), kind_names) for i, name in enumerate(REGIMES) if (r == i).any()}
    good = (r == REGIMES.index('real')) & (data['seg_frac'] >= SELECT_SEG_FRAC)
    if not good.any():
        raise RuntimeError('no real regime scan with the cleaner on the object: the best checkpoint can not be selected')
    return dict(select_cd=round(macro(per[good], data['kind'][good]) * 1000, 3), mixed=summary(per, data, kind_names), by_regime=by_regime)


def log_val(log, step, what, rep):
    """One line per regime and one for the mix, so no reader can mistake the mixed number for the selection number."""
    for regime, row in rep['by_regime'].items():
        log(event='val', step=step, what=what, regime=regime, selects=regime == 'real',
            select_cd=rep['select_cd'] if regime == 'real' else None, **row)
    log(event='val', step=step, what=what, regime='mixed', selects=False, **rep['mixed'])


# ------------------------------------------------------------------------------------------------- run

def _rng_state(rng):
    return dict(gen=rng.get_state().cpu(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(rng, st):
    rng.set_state(st['gen'].to(torch.uint8).cpu())
    torch.set_rng_state(st['torch'].cpu())
    if st['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in st['cuda']])


def _save(obj, path):
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Log:
    def __init__(self, path):
        self.path = path
        self.t0 = time.time()

    def __call__(self, **rec):
        rec['t'] = round(time.time() - self.t0, 1)
        line = json.dumps(rec, default=float)
        print(line, flush=True)
        with open(self.path, 'a') as f:
            f.write(line + '\n')


def run(cfg, hashes, data_root, out_root, run_id, wall_limit_s, pcn_ckpt, device, input_root=None, init_from=None,
        resume_from=None, force=False, commit=lambda: None):
    """Returns the final record. Layout: <out_root>/<run_id>/{latest.pt, best.pt, final.pt, train_final.jsonl}.
    latest.pt in the run directory means a restart of this same run: it continues, and its clock keeps counting.
    resume_from continues another run's latest.pt with a fresh clock. init_from loads weights only (a later stage)."""
    t_entry = time.time()
    out = os.path.join(out_root, run_id)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, 'final.pt')):
        return torch.load(os.path.join(out, 'final.pt'), map_location='cpu', weights_only=False)['record']
    log = Log(os.path.join(out, 'train_final.jsonl'))
    Tc, A, Rc = cfg['train'], cfg['augment'], cfg['real']
    dev = torch.device(device)
    torch.manual_seed(cfg['seed'])
    rng = torch.Generator(device=dev).manual_seed(cfg['seed'])

    amp_dtype = getattr(torch, Tc['amp_dtype'])
    amp_ctx = lambda: torch.autocast(dev.type, dtype=amp_dtype, enabled=Tc['amp'])
    scaler = torch.amp.GradScaler('cuda', enabled=Tc['amp'] and amp_dtype == torch.float16 and dev.type == 'cuda')

    stop = dict(signal=None)
    def on_signal(sig, _):
        stop['signal'] = sig
    try:
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
    except ValueError:                  # not the main thread: the wall limit still applies
        pass

    manifest_path = os.path.join(data_root, 'manifest.json')
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    hashes = dict(hashes, dataset=manifest.get('dataset_hash'))

    tr, va, te = (load_split(data_root, s) for s in ('train', 'val', 'test'))
    if tr is None or va is None:
        raise RuntimeError(f'train or val split missing under {data_root}')
    regimes = {}
    for name, d in (('train', tr), ('val', va), ('test', te)):
        if d is None:
            continue
        if 'regime' not in d:
            raise RuntimeError(f'{name} split under {data_root} has no regime column: regenerate it')
        regimes[name] = {r: int((d['regime'] == i).sum()) for i, r in enumerate(REGIMES)}
    if not regimes['val']['real']:
        raise RuntimeError('no real regime scans in val: the best checkpoint can not be selected')
    tr = select(tr, tr['seg_frac'] >= Tc['min_seg_frac'])
    Xtr, Ytr, Str = (torch.from_numpy(tr[k]) for k in ('partial', 'gt', 'scale'))
    N = len(Xtr)
    wsim = sample_weights(tr['kind'], Tc.get('kind_weights'), KINDS)     # bottle emphasis etc.: batch mix by kind

    pool, real_rejected, real_train = [], [], None
    if Rc['enabled'] or input_root:
        pool, real_rejected = real_pool(input_root, Rc['holdout'])
    train_sessions = [s for s in pool if not s['held_out']] if Rc['enabled'] else []
    used = {s['session'] for s in train_sessions}
    if Rc['enabled']:
        if not train_sessions:
            raise RuntimeError('real stage without real training sessions')
        real_train = (torch.from_numpy(np.stack([s['partial'] for s in train_sessions])).to(dev),
                      torch.tensor([s['T']['scale'] for s in train_sessions], dtype=torch.float32, device=dev))
    objects = sorted({s['object'] for s in train_sessions})
    wreal = sample_weights([objects.index(s['object']) for s in train_sessions], Rc.get('object_weights'), objects) if train_sessions else None
    # Full targets for real scans of a known object (majproj.real_targets): sessions with an accepted target train with
    # the two-sided loss, the others keep the one-sided loss on their observed points.
    real_gt, real_has, target_info = None, None, {}
    if Rc['enabled'] and Rc.get('targets'):
        from .real_targets import targets_for_pool
        gts, target_info = targets_for_pool(pool, Rc['targets'], N_TARGET, Rc.get('heights'))
        for s in pool:
            s['gt'] = gts.get(s['session'])
            s['target_info'] = target_info.get(s['session'])
        ys = [s['gt'] if s.get('gt') is not None else np.resize(s['partial'], (N_TARGET, 3)) for s in train_sessions]
        real_gt = torch.from_numpy(np.stack(ys).astype(np.float32)).to(dev)
        real_has = torch.tensor([s.get('gt') is not None for s in train_sessions], device=dev)
    log(event='start', run_id=run_id, hashes=hashes, n_train=N, n_val=len(va['partial']), regimes=regimes,
        n_train_real=int((tr['regime'] == REGIMES.index('real')).sum()),
        n_test=None if te is None else len(te['partial']), real_train=sorted(used),
        real_rejected=real_rejected, load_s=round(time.time() - t_entry, 1), wall_limit_s=wall_limit_s, cfg=cfg,
        kind_weights=Tc.get('kind_weights'), object_weights=Rc.get('object_weights') if train_sessions else None,
        real_targets={k: {kk: v[kk] for kk in ('accepted', 'resid_median_mm', 'front_within_face_depth', 'scale_source', 'target_source')}
                      for k, v in target_info.items()}, targets_cfg=Rc.get('targets'), heights_cfg=Rc.get('heights'),
        torch=torch.__version__, device=str(dev))

    model = load_adapointr(pcn_ckpt, dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=Tc['lr'], weight_decay=Tc['weight_decay'])

    step, first_step, ema_updates, used_s, best = 0, 0, 0, 0.0, math.inf
    latest = os.path.join(out, 'latest.pt')
    restart = os.path.exists(latest)
    source = latest if restart else resume_from
    if source:
        st = torch.load(source, map_location='cpu', weights_only=False)
        bad = [k for k in ('code', 'config', 'dataset') if st['hashes'].get(k) != hashes.get(k)]
        if bad and not force:
            raise RuntimeError(f'refusing to resume {source}: {bad} hash differs (force=True overrides)')
        model.load_state_dict(st['model'])
        ema.load_state_dict(st['ema'])
        opt.load_state_dict(st['opt'])
        scaler.load_state_dict(st['scaler'])
        _set_rng_state(rng, st['rng'])
        step, ema_updates, best = st['step'], st['ema_updates'], st['best']
        used_s = st['wall_used_s'] if restart else 0.0
        first_step = st['first_step'] if restart else step      # the LR schedule and max_steps belong to this run
        log(event='resumed', source=source, step=step, wall_used_s=round(used_s), mismatched_hashes=bad)
    elif init_from:
        st = torch.load(init_from, map_location='cpu', weights_only=False)
        model.load_state_dict(st['ema'])
        ema.load_state_dict(st['ema'])
        log(event='init', source=init_from, hashes=st.get('hashes'))

    wall = lambda: used_s + time.time() - t_entry
    budget_s = max(1.0, wall_limit_s - Tc['final_reserve_s'])
    last_step = first_step + Tc['max_steps']
    if not restart:
        per = evaluate(model, va, dev, amp_ctx, Tc['eval_batch'], identity=True)
        log_val(log, step, 'partial_as_output', report(per, va, KINDS))
        per = evaluate(ema, va, dev, amp_ctx, Tc['eval_batch'])
        rep = report(per, va, KINDS)
        log_val(log, step, 'starting_weights', rep)
        best = rep['select_cd']
        _save(dict(ema=ema.state_dict(), step=step, val=rep, hashes=hashes), os.path.join(out, 'best.pt'))

    def checkpoint():
        _save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                   rng=_rng_state(rng), step=step, first_step=first_step, ema_updates=ema_updates, best=best, wall_used_s=wall(),
                   hashes=hashes, run_id=run_id), latest)
        commit()

    model.train()
    avg, bad_steps, skipped, t_mark, s_mark = None, 0, 0, time.time(), step
    while step < last_step and wall() < budget_s and stop['signal'] is None:
        progress = max((step - first_step) / Tc['max_steps'], wall() / budget_s)
        warm = min(1.0, (step - first_step + 1) / Tc['warmup_steps'])
        floor = Tc['lr_floor']
        lr = Tc['lr'] * warm * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))
        for g in opt.param_groups:
            g['lr'] = lr

        idx = draw(N, wsim, Tc['batch_size'], dev, rng)
        x = Xtr[idx].to(dev, non_blocking=True)
        y = Ytr[idx].to(dev, non_blocking=True).float() * GT_STEP
        x, y, _ = augment(x, y, Str[idx].to(dev), A, rng)
        n_sim = len(x)
        if real_train is not None:
            j = draw(len(real_train[0]), wreal, Rc['batch_size'], dev, rng).to(dev)
            yj = real_gt[j] if real_gt is not None else real_train[0][j]
            hj = real_has[j] if real_has is not None else None
            xr, obs, _ = augment(real_train[0][j], yj, real_train[1][j], A, rng)
            x = torch.cat([x, xr])

        with amp_ctx():
            ret = model(x)
        pc, dc, df, pf = (r.float() for r in ret)
        loss_dn, loss_rec = model.get_loss((pc[:n_sim], dc[:n_sim], df[:n_sim], pf[:n_sim]), y)
        loss = loss_dn + loss_rec
        loss_real = None
        if real_train is not None:
            if hj is not None and bool(hj.any()):
                two = model.get_loss(tuple(t[n_sim:][hj] for t in (pc, dc, df, pf)), obs[hj])
                loss_real = two[0] + two[1]
                if bool((~hj).any()):
                    loss_real = loss_real + one_sided(obs[~hj], pf[n_sim:][~hj], Rc['trim']).mean()
            else:
                loss_real = one_sided(obs, pf[n_sim:], Rc['trim']).mean()
            loss = loss + Rc['weight'] * loss_real

        if not torch.isfinite(loss):
            bad_steps += 1
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if bad_steps >= Tc['max_bad_steps']:
                log(event='abort', step=step, reason='non-finite loss', bad_steps=bad_steps)
                checkpoint()
                raise RuntimeError(f'{bad_steps} consecutive non-finite losses at step {step}')
            continue
        bad_steps = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Tc['grad_clip'])
        scaler.step(opt)
        scaler.update()

        step += 1
        ema_updates += 1
        decay = min(Tc['ema_decay'], (1 + ema_updates) / (10 + ema_updates))
        with torch.no_grad():
            for e, m in zip(ema.state_dict().values(), model.state_dict().values()):
                e.lerp_(m, 1 - decay) if e.is_floating_point() else e.copy_(m)

        avg = loss_rec.item() if avg is None else 0.98 * avg + 0.02 * loss_rec.item()
        if step % Tc['log_every'] == 0:
            now = time.time()
            log(event='train', step=step, lr=lr, recon_ema=round(avg, 5), denoise=round(loss_dn.item(), 5),
                real=None if loss_real is None else round(loss_real.item(), 5), skipped=skipped,
                real_targets_in_batch=None if real_train is None else (int(hj.sum()) if hj is not None else 0),
                steps_per_s=round((step - s_mark) / (now - t_mark), 2), wall_s=round(wall()))
            t_mark, s_mark = now, step
        if step % Tc['eval_every'] == 0:
            rep = report(evaluate(ema, va, dev, amp_ctx, Tc['eval_batch']), va, KINDS)
            log_val(log, step, 'ema', rep)
            if rep['select_cd'] < best:
                best = rep['select_cd']
                _save(dict(ema=ema.state_dict(), step=step, val=rep, hashes=hashes), os.path.join(out, 'best.pt'))
            if real_train is not None:              # training sessions only; held-out sessions wait for the end
                rows = real_report(ema, train_sessions, dev, amp_ctx, used)
                log(event='real_train', step=step, observed_to_output_mm=round(float(np.mean([r['observed_to_output_mm'] for r in rows])), 2))
            model.train()
        if step % Tc['ckpt_every'] == 0:
            checkpoint()

    checkpoint()
    if stop['signal'] is not None:              # preempted: Modal restarts the input and it resumes from latest.pt
        log(event='stopped', signal=stop['signal'], step=step, wall_s=round(wall()))
        return dict(run_id=run_id, stopped_by_signal=stop['signal'], steps=step)

    rep = report(evaluate(ema, va, dev, amp_ctx, Tc['eval_batch']), va, KINDS)
    log_val(log, step, 'ema_final', rep)
    if rep['select_cd'] < best:
        best = rep['select_cd']
        _save(dict(ema=ema.state_dict(), step=step, val=rep, hashes=hashes), os.path.join(out, 'best.pt'))

    chosen = torch.load(os.path.join(out, 'best.pt'), map_location='cpu', weights_only=False)
    ema.load_state_dict(chosen['ema'])
    record = dict(run_id=run_id, selected_step=chosen['step'], steps=step, val=chosen['val'], hashes=hashes,
                  real_rejected=real_rejected, skipped_steps=skipped)
    if te is not None:                          # evaluated once, on the checkpoint selected on val
        record['test'] = report(evaluate(ema, te, dev, amp_ctx, Tc['eval_batch']), te, KINDS)
        base = load_adapointr(pcn_ckpt, dev)
        record['test_official_weights'] = report(evaluate(base, te, dev, amp_ctx, Tc['eval_batch']), te, KINDS)
    if pool:
        record['real'] = real_report(ema, pool, dev, amp_ctx, used)
    log(event='final', **record)
    _save(dict(ema=ema.state_dict(), base_model=ema.state_dict(), record=record, hashes=hashes), os.path.join(out, 'final.pt'))
    commit()
    return record
