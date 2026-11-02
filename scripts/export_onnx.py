"""Export the production checkpoint to ONNX (opset 17; input 1x2048x3 canonical partial -> 1x16384x3) and check the ONNX
model against PyTorch.

The checkpoint is majproj.infer.default_checkpoint() unless one is named. Agreement is measured with an order-invariant
symmetric Chamfer distance (ONNX output vs torch output), required < 1e-5; the max-abs difference is printed too. The model
is numerically chaotic on many real inputs (discrete kNN / top-k choices flip under float32 noise: perturbing the INPUT by
1e-5 moves the torch output itself by up to ~5e-3), so two float32 runtimes agree to ~5e-7 on most inputs but can land 1e-4
apart on others. The gate is therefore:
  * strict: Chamfer(onnx, torch) < 1e-5 on at least 75 % of the probes (a wrong graph does not match to 5e-7 on that many
    diverse inputs);
  * every probe: Chamfer(onnx, torch) <= the largest Chamfer between torch(x) and torch(x + 1e-5 noise) over 3
    draws, i.e. ONNX never differs from torch by more than torch differs from itself under float-scale noise.
Probes are deployment-format inputs:
  * real: the model input of one export of every raw session in FINAL-INPUTS the cleaner accepts (the same path the app uses);
  * synthetic: a box face, a cylinder side and a sphere cap in the canonical frame (so the check never depends on how
    many real scans the cleaner accepts).

Usage: PYTHONPATH=. uv run python scripts/export_onnx.py [checkpoint] [data/adapointr.onnx]
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial import cKDTree

from majproj.canon import N_INPUT as N_IN
from majproj.infer import Completer, default_checkpoint



class Fine(torch.nn.Module):
    """The model's fine output only (the coarse output is not used by the app)."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x)[1]


def chamfer(a, b):
    return 0.5 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def synthetic_partials(n=N_IN):
    """Box face, cylinder side, sphere cap: partial views in the canonical frame (y up, longest seen side 0.9)."""
    r = np.random.default_rng(7)
    face = np.c_[r.uniform(-0.45, 0.45, n), r.uniform(-0.3, 0.3, n), r.normal(0, 0.004, n)]
    th = r.uniform(-1.2, 1.2, n)
    cyl = np.c_[0.2 * np.sin(th), r.uniform(-0.45, 0.45, n), 0.2 * np.cos(th)]
    u, v = r.uniform(-0.9, 0.9, n), r.uniform(0.2, 1.3, n)
    cap = 0.4 * np.c_[np.sin(v) * np.cos(u), np.cos(v), np.sin(v) * np.sin(u)]
    return {'synthetic box face': face, 'synthetic cylinder side': cyl, 'synthetic sphere cap': cap}


def real_partials(completer):
    from scripts.build_web import session_files
    out = {}
    for f in session_files():
        res = completer(np.load(f))
        if res.ok:
            out[os.path.basename(f)] = res.partial
    return out


def probes(completer):
    items = dict(real_partials(completer))
    items.update(synthetic_partials())
    return {k: np.asarray(v, np.float32)[None] for k, v in items.items()}


def main(ckpt=None, out='data/adapointr.onnx'):
    completer = Completer(ckpt or default_checkpoint(), device='cpu')
    model = Fine(completer.model).eval()
    cases = probes(completer)
    first = next(iter(cases.values()))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (torch.from_numpy(first),), out, input_names=['partial'], output_names=['complete'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    sess = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    rng = torch.Generator().manual_seed(0)
    fails, strict = 0, 0
    for name, x in cases.items():
        xt = torch.from_numpy(x)
        with torch.no_grad():
            ref = model(xt).numpy()
            sens = max(chamfer(model(xt + 1e-5 * torch.randn(xt.shape, generator=rng)).numpy()[0].astype(np.float64),
                               ref[0].astype(np.float64)) for _ in range(3))
        got = sess.run(None, {'partial': x})[0]
        cdv = chamfer(got[0].astype(np.float64), ref[0].astype(np.float64))
        ext = float(np.ptp(ref[0], 0).max())
        is_strict = cdv < 1e-5
        strict += is_strict
        ok = is_strict or cdv <= sens
        fails += not ok
        print(f'{name}: shape {got.shape}  order-invariant Chamfer(onnx, torch) = {cdv:.3e} '
              f'({cdv / ext:.1e} of output extent {ext:.2f})  max abs diff = {float(np.abs(got - ref).max()):.3e}  '
              f'torch vs torch under 1e-5 input noise: {sens:.1e}  '
              f'{"PASS (strict)" if is_strict else "PASS (within torch noise response)" if ok else "FAIL"}')
    if fails or strict < 0.75 * len(cases):
        raise SystemExit(f'{fails}/{len(cases)} probes failed; {strict} strict probes (need at least 75 %)')
    print(f'PASS on all {len(cases)} probes ({strict} agree to < 1e-5)')


if __name__ == '__main__':
    main(*sys.argv[1:])
