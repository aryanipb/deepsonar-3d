"""Production inference: ONE scan file -> the complete 3D object, or a rejection with its reason.

    file (N x 3, z = up)
      -> metric_thin                 raw metric map: 1 cm voxels, the density the cleaner works at
      -> majproj.clean.clean         the dwelled-on surface; else the file is REJECTED and the model is never run
      -> majproj.canon.to_canonical  sensor-aligned, longest seen side = 0.9
      -> majproj.canon.sample_2048   the model input, the same sampler training used
      -> the checkpoint              (1, 2048, 3) -> (1, 16384, 3), model(x)[1]
      -> majproj.canon.from_canonical  metres, sensor at the origin, z up

This is the path scripts/gen_final.py ran on every training scan. The network gets points only: no label, no template.

Two file types:
  raw map     metres with the sensor at the origin. Sizes are real centimetres.
  export      scan_*_shapenet.npy: at most 4096 rows, bounding-box centred and divided by the largest coordinate. Metric
              scale and the sensor position are gone, so results stay in file units (never cm) and the cleaner runs
              without an origin. Decision: an export is accepted, but the view direction is taken from the patch normal,
              whose sign is a convention, so the predicted back can land on the wrong side of the face. Every export
              result carries that warning. Training never saw this path; raw files are the supported input.

What is measured and what is predicted. The seen face (width, height) and the distance come from the scan. Width is
blurred by the 5.6 degree zones, so a narrow object reads wider than it is. Everything behind the face, including the
depth, is the network's prediction: a real object's back is not in the scan.

To swap the model change CHECKPOINT below, or pass --ckpt, or set MAJPROJ_CKPT.

CLI:  PYTHONPATH=. uv run python -m majproj.infer FINAL-INPUTS/BOX/scan_20261030_121600.npy --out outputs/infer
"""
import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .canon import N_INPUT, from_canonical, sample_2048, to_canonical
from .clean import clean
from .model import PCN_CONFIG
from .model.AdaPoinTr import AdaPoinTr

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / 'checkpoints' / 'adapointr_final.pth'               # run final1, step 37446; the one line to change for a new model
N_OUT = 16384
EXPORT_MAX_ROWS = 4096
MIN_OBJECT_PTS = 20                                                     # gen_final's rule
CM = 100.0


def default_checkpoint():
    return os.environ.get('MAJPROJ_CKPT') or str(CHECKPOINT)


def stage_of(ckpt):
    """'final' only for final.pt or adapointr_final.pth, the output of a finished run; anything else is reported as a pilot."""
    return os.environ.get('MAJPROJ_STAGE') or ('final' if os.path.basename(str(ckpt)) in ('final.pt', 'adapointr_final.pth') else 'pilot')


def load_checkpoint(path, device='cpu'):
    """-> (model in eval mode, the checkpoint's own record). final.pt holds the weights as base_model, best.pt as ema."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    key = next((k for k in ('base_model', 'ema', 'model') if k in state), None)
    sd = state[key] if key else state
    model = AdaPoinTr(PCN_CONFIG)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in sd.items()}, strict=True)
    info = {k: state[k] for k in ('step', 'hashes', 'record') if k in state} if key else {}
    if 'record' in info:
        info.setdefault('step', info['record'].get('selected_step'))
    return model.to(device).eval(), info


def json_safe(v):
    """numpy -> python, nan / inf -> None."""
    if isinstance(v, dict):
        return {str(k): json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return json_safe(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def is_export(points):
    """Exports are divided by their largest coordinate, so it is exactly 1. A raw map may be small enough to fit inside
    +-1 m, so the range alone does not decide."""
    return len(points) <= EXPORT_MAX_ROWS and abs(float(np.abs(points).max()) - 1.0) < 1e-5


def prepare(points):
    """File rows -> (points for the cleaner, metric?). Same thinning as gen_final; exports pass unchanged."""
    from scripts.gen_final import metric_thin          # scripts import this package's neighbours; keep import time light
    p = np.asarray(points, np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f'expected an N x 3 point cloud, got shape {p.shape}')
    p = p[np.isfinite(p).all(1)]
    if not len(p):
        raise ValueError('no finite points in the file')
    if is_export(p):
        return p, False
    return metric_thin(p)[0], True


def export_origin(meta):
    """Stand-in sensor position for an export: one unit along the patch normal, whose sign is a convention."""
    c, n = np.asarray(meta['centre'], float), np.asarray(meta['normal'], float)
    return (c[0] + n[0], c[1] + n[1], c[2])


def extents(canon_pts, scale):
    """1-99 percentile extents along the canonical axes (width, height, depth), in the units of T. Same as train_final's
    real_report: a few stray output points must not set a size."""
    return (np.percentile(canon_pts, 99, 0) - np.percentile(canon_pts, 1, 0)) * scale


@dataclass
class Completion:
    status: str                       # 'ok' or 'rejected'
    reason: str                       # why it was rejected; '' when ok
    confidence: float | None          # the cleaner's heuristic confidence
    metric: bool                      # True: metres. False: an export, in file units
    scan: np.ndarray                  # the input file as given
    observed: np.ndarray              # K x 3 cleaned object points (file frame); empty when rejected
    background: np.ndarray            # M x 3 unique points after thinning that are not the object
    completed: np.ndarray | None      # 16384 x 3 complete object (file frame); None when rejected
    partial: np.ndarray | None        # 2048 x 3 model input (canonical)
    completed_canon: np.ndarray | None  # 16384 x 3 model output (canonical)
    T: dict | None                    # canonical transform
    sizes: dict = field(default_factory=dict)      # what is measured and what is predicted
    clean_meta: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def ok(self):
        return self.status == 'ok'

    def meta(self, checkpoint='', stage=''):
        return json_safe(dict(
            status=self.status, reason=self.reason, confidence=self.confidence, checkpoint=checkpoint, stage=stage,
            metric=self.metric, n_input=len(self.scan), n_object=len(self.observed), n_background=len(self.background),
            n_completed=None if self.completed is None else len(self.completed), sizes=self.sizes, transform=self.T,
            cleaner=self.clean_meta, warnings=self.warnings))


def measure(cm, canon_out, T, metric):
    """Measured numbers come from the cleaner, predicted ones from the output. cm for a raw map, file units otherwise."""
    k = CM if metric else 1.0
    e = extents(canon_out, T['scale'])
    return dict(unit='cm' if metric else 'file units',
                measured=dict(face_width=k * cm['width'], face_height=k * cm['height'],
                              distance=k * cm['range_m'] if metric and cm.get('range_m') is not None else None),
                predicted=dict(width=k * float(e[0]), height=k * float(e[1]), depth=k * float(e[2])))


def caveats(status, cm, metric, stage):
    out = []
    if status != 'ok':
        return ['The cleaner found no object in this file, so the model was not run.']
    if stage != 'final':
        out.append(f'Model stage: {stage}. These are not final results.')
    out.append('Depth and the back of the object are predicted by the network, not measured: one scan never sees them.')
    if metric:
        out.append('Measured width is blurred by the 5.6 degree sensor zones, so narrow objects read wider than they are.')
    else:
        out.append('This is an export: metric scale and the sensor position are gone. Sizes are in file units, the '
                   'predicted back may lie on the wrong side of the face, and training never saw this format.')
    if cm.get('layer_mode') == 'drift_slab':
        out.append('The hand drifted along the line of sight in this scan. The face position is the depth held longest, '
                   'not the true surface depth.')
    return out


class Completer:
    def __init__(self, ckpt=None, device=None, stage=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt = str(ckpt or default_checkpoint())
        if not os.path.exists(self.ckpt):
            raise FileNotFoundError(f'checkpoint not found: {self.ckpt}')
        self.stage = stage or stage_of(self.ckpt)
        self.model, self.info = load_checkpoint(self.ckpt, self.device)

    @property
    def ckpt_name(self):
        return os.path.basename(self.ckpt)

    @torch.no_grad()
    def complete_canon(self, partial):
        """Model only: canonical (2048, 3) -> (16384, 3)."""
        x = np.asarray(partial, np.float32)
        if x.shape != (N_INPUT, 3):
            raise ValueError(f'expected shape ({N_INPUT}, 3), got {x.shape}')
        return self.model(torch.from_numpy(x[None]).to(self.device))[1][0].cpu().numpy()

    def __call__(self, scan):
        scan = np.asarray(scan)
        pts, metric = prepare(scan)
        c = clean(pts, origin=(0.0, 0.0, 0.0)) if metric else clean(pts)
        cm = json_safe(dict(c.meta))
        obj = np.asarray(c.object, np.float32).reshape(-1, 3)
        bg = np.asarray(c.removed, np.float32).reshape(-1, 3)
        status, reason = cm['status'], str(cm.get('reason') or '')
        if status == 'ok' and len(obj) < MIN_OBJECT_PTS:
            status, reason = 'rejected', f'too few object points ({len(obj)})'
        if status != 'ok':
            all_bg = np.concatenate([bg, obj]) if len(obj) else bg
            return Completion('rejected', reason or 'the cleaner found no object', cm.get('confidence'), metric, scan,
                              np.empty((0, 3), np.float32), all_bg, None, None, None, None, {}, cm,
                              caveats('rejected', cm, metric, self.stage))
        origin = (0.0, 0.0, 0.0) if metric else export_origin(cm)
        canon, T = to_canonical(obj, origin)
        partial = sample_2048(canon)
        pred = self.complete_canon(partial)
        done = from_canonical(pred, T).astype(np.float32)
        return Completion('ok', '', cm.get('confidence'), metric, scan, obj, bg, done, partial, pred, T,
                          measure(cm, pred, T, metric), cm, caveats('ok', cm, metric, self.stage))


# ----------------------------------------------------------------------------- output files
def write_ply(path, parts):
    """parts: list of (points N x 3, (r, g, b)). One ascii ply."""
    n = sum(len(p) for p, _ in parts)
    with open(path, 'w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n'
                'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for pts, (r, g, b) in parts:
            for x, y, z in pts:
                f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')


COMPLETED_RGB, SEEN_RGB, BACKGROUND_RGB = (11, 143, 214), (17, 17, 17), (170, 170, 170)


def scene_parts(res, include_background=True):
    parts = [(res.background, BACKGROUND_RGB)] if include_background else []
    if res.ok:
        parts += [(res.completed, COMPLETED_RGB), (res.observed, SEEN_RGB)]
    return parts


def write_outputs(res, out_dir, stem, checkpoint='', stage=''):
    """Rejected scans get only the meta json; accepted ones also the completed object (.npy, .ply of the scene)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = [f'{out_dir}/{stem}_meta.json']
    with open(paths[0], 'w') as f:
        json.dump(res.meta(checkpoint, stage), f, indent=1)
    if res.ok:
        np.save(f'{out_dir}/{stem}_completed.npy', res.completed)
        write_ply(f'{out_dir}/{stem}_completed.ply', scene_parts(res))
        paths += [f'{out_dir}/{stem}_completed.npy', f'{out_dir}/{stem}_completed.ply']
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scans', nargs='+')
    ap.add_argument('--out', default='outputs/infer')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--stage', default=None, help="'pilot' or 'final'; default: 'final' only for final.pt or adapointr_final.pth")
    args = ap.parse_args()
    completer = Completer(args.ckpt, stage=args.stage)
    print(f'checkpoint {completer.ckpt} ({completer.stage.upper()}) on {completer.device}')
    for path in args.scans:
        res = completer(np.load(path, allow_pickle=False))
        stem = os.path.splitext(os.path.basename(path))[0]
        write_outputs(res, args.out, stem, completer.ckpt_name, completer.stage)
        if res.ok:
            m, p, u = res.sizes['measured'], res.sizes['predicted'], res.sizes['unit']
            d = 1 if res.metric else 3
            print(f"{stem}: ok conf {res.confidence:.2f}, face measured {m['face_width']:.{d}f} x {m['face_height']:.{d}f} {u}, "
                  f"output predicted {p['width']:.{d}f} x {p['height']:.{d}f} x {p['depth']:.{d}f} {u} (depth not measured)")
        else:
            print(f'{stem}: REJECTED, no completion: {res.reason}')


if __name__ == '__main__':
    main()
