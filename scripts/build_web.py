"""Package the web app: the production checkpoint as ONNX, the real scans, and the Python reference for the parity test.

  * model: <checkpoint> as an ONNX file split into <50 MB chunks in web/public/model, with a manifest naming the checkpoint,
    its stage ('pilot' unless the file is final.pt or adapointr_final.pth) and a content-addressed version (the browser cache key). Re-exported
    only when the packaged chunks do not reproduce the checkpoint;
  * samples: one export per raw session (the longest; the shorter ones are subsets of it) and the 15 older normalized
    exports. Labels are the folder the user put a scan in, shown as such, never given to the model;
  * samples/expected.json: for ALL 27 raw files and the 15 exports, the output of majproj.infer.Completer: the cleaner's
    decision, a fingerprint of the object points, the canonical transform, a fingerprint of the float32 model input and the
    PyTorch completion. `node web/test/parity.mjs` and `index.html?selftest` compare the JS port with it. It also records
    the sha256 of majproj/clean.py; the parity test fails when clean.js was ported from another version.

Every scan also records whether its session is held out: configs/final.yaml real.holdout never enters training, selection or
the simulator's real bank. All other sessions fed the simulator's room data, so results on them are not test results.

Usage: PYTHONPATH=. uv run python -m scripts.build_web [checkpoint] [--stage pilot|final] [--reexport]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import onnxruntime as ort
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from majproj.canon import N_INPUT
from majproj.infer import Completer, default_checkpoint, json_safe, stage_of
from majproj.sim_real import holdout_sessions
from scripts.export_onnx import chamfer, synthetic_partials
from scripts.export_onnx import main as export_onnx

WEB = 'web/public'
CHUNK = 45 * 1024 * 1024
RAW_DIRS = {'FINAL-INPUTS/BOX': 'Box', 'FINAL-INPUTS/BOTTLE': 'Bottle'}
EXPORT_DIRS = {'inputs': 'Older export', 'inputs_box': 'Older export'}
CLEANER = 'majproj/clean.py'
# Repeat exports of one session: every file on the left is a subset of the file on the right, the session's longest.
REPEATS = {'121018': '121055', '121022': '121055', '121422': '121457', '122617': '122622', '123027': '123028',
           '123257': '123259', '123814': '123815'}


def stamp(path):
    return os.path.basename(path)[:-4].split('_')[-1]


def session_of(path):
    """The six-digit time stamp of the session's longest export, the form majproj.sim_real.holdout_sessions() returns."""
    s = stamp(path)
    return REPEATS.get(s, s)


def session_files():
    """The longest export of each raw session, one per session."""
    return [f for d in RAW_DIRS for f in sorted(glob.glob(f'{ROOT}/{d}/*.npy')) if stamp(f) not in REPEATS]


def fnv1a(b):
    h = 0x811c9dc5
    for x in b:
        h = ((h ^ x) * 0x01000193) & 0xffffffff
    return h


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def packaged_model_matches(ckpt_digest, manifest, completer):
    """True when the packaged chunks exist, are this checkpoint, and reproduce the torch model on a probe."""
    names = manifest.get('chunks', [])
    if not names or manifest.get('ckpt_sha256') != ckpt_digest or not all(os.path.exists(f'{WEB}/model/{n}') for n in names):
        return False
    data = b''.join(open(f'{WEB}/model/{n}', 'rb').read() for n in names)
    if len(data) != manifest.get('size') or hashlib.sha256(data).hexdigest()[:12] != manifest.get('version', '').split('-')[-1]:
        return False
    probe = np.asarray(synthetic_partials()['synthetic box face'], np.float32)
    got = ort.InferenceSession(data, providers=['CPUExecutionProvider']).run(None, {'partial': probe[None]})[0]
    return float(chamfer(got[0].astype(np.float64), completer.complete_canon(probe).astype(np.float64))) < 1e-5


def package_model(ckpt, completer, reexport):
    os.makedirs(f'{WEB}/model', exist_ok=True)
    digest = sha256(ckpt)
    mpath = f'{WEB}/model/manifest.json'
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    if not reexport and packaged_model_matches(digest, old, completer):
        manifest = dict(old)
        print('model: packaged chunks already reproduce', ckpt, '(no re-export)')
    else:
        onnx_path = 'data/adapointr.onnx'
        export_onnx(ckpt, onnx_path)
        for f in glob.glob(f'{WEB}/model/*'):
            os.remove(f)
        data = open(onnx_path, 'rb').read()
        chunks = []
        for i in range(0, len(data), CHUNK):
            name = f'adapointr.onnx.part{len(chunks)}'
            open(f'{WEB}/model/{name}', 'wb').write(data[i:i + CHUNK])
            chunks.append(name)
        manifest = {'size': len(data), 'chunks': chunks, 'version': 'ck-' + hashlib.sha256(data).hexdigest()[:12]}
        print('model:', len(data) / 1e6, 'MB in', len(chunks), 'chunks')
    manifest.update(checkpoint=os.path.basename(ckpt), stage=completer.stage, frame='canonical', ckpt_sha256=digest,
                    step=completer.info.get('step'), real_sessions_scored='record' in completer.info)
    json.dump(manifest, open(mpath, 'w'))
    return manifest


def torch_sensitivity(completer, partial, noise=1e-6):
    """How far the torch completion moves (Chamfer) when its input is perturbed by `noise`: some inputs are numerically
    chaotic (discrete kNN / top-k flips), so runtimes cannot agree to float precision on those."""
    g = torch.Generator().manual_seed(0)
    x = torch.from_numpy(np.asarray(partial, np.float32))
    a = completer.complete_canon(partial)
    b = completer.complete_canon((x + noise * torch.randn(x.shape, generator=g)).numpy())
    return float(chamfer(a.astype(np.float64), b.astype(np.float64)))


def reference(completer, path):
    res = completer(np.load(path))
    e = {'status': res.status, 'reason': res.reason or None, 'confidence': res.confidence, 'metric': res.metric,
         'n_input': res.clean_meta.get('n_input'), 'n_unique': res.clean_meta.get('n_unique'), 'n_object': int(len(res.observed)),
         'layer_mode': res.clean_meta.get('layer_mode')}
    if res.ok:
        e.update(object_fnv1a=fnv1a(np.ascontiguousarray(res.observed, '<f4').tobytes()), T=res.T, sizes=res.sizes,
                 partial_fnv1a=fnv1a(np.ascontiguousarray(res.partial, '<f4').tobytes()),
                 partial_head=np.round(res.partial[::16], 6).tolist(),                # 128 of 2048 points
                 completed_canon=np.round(res.completed_canon[::32], 5).tolist(),     # 512 of 16384 points
                 completed_file_head=np.round(res.completed[::512], 6).tolist(),      # 32 points, file frame
                 torch_sensitivity=torch_sensitivity(completer, res.partial))
    return json_safe(e), res


def main(ckpt=None, stage=None, reexport=False):
    ckpt = str(ckpt or default_checkpoint())
    stage = stage or stage_of(ckpt)
    print('checkpoint:', ckpt, 'stage:', stage)
    os.makedirs(f'{WEB}/samples', exist_ok=True)
    completer = Completer(ckpt, device='cpu', stage=stage)
    manifest = package_model(ckpt, completer, reexport)

    for f in glob.glob(f'{WEB}/samples/*.npy'):
        os.remove(f)
    held = set(holdout_sessions())
    keep = {os.path.realpath(f) for f in session_files()}
    samples, expected = [], {}
    for dirs, raw in ((RAW_DIRS, True), (EXPORT_DIRS, False)):
        for d, label in dirs.items():
            for f in sorted(glob.glob(f'{d}/*.npy')):
                name = os.path.basename(f)
                in_app = not raw or os.path.realpath(f) in keep
                if in_app:
                    shutil.copy(f, f'{WEB}/samples/{name}')
                session = session_of(f) if raw else None
                samples.append({'file': name, 'path': f, 'label': label, 'raw': raw, 'in_app': in_app, 'session': session,
                                'held_out': session in held if raw else False})
                expected[name], res = reference(completer, f)
                print(f'{d}/{name}: {res.status} {res.reason or ""} confidence {res.confidence} object points {len(res.observed)}')
    json.dump(samples, open(f'{WEB}/samples/samples.json', 'w'), indent=1)
    json.dump({'checkpoint': manifest['checkpoint'], 'stage': stage, 'frame': manifest['frame'], 'version': manifest['version'],
               'n_in': N_INPUT, 'cleaner_sha256': sha256(CLEANER), 'files': expected}, open(f'{WEB}/samples/expected.json', 'w'))
    n_ok = sum(v['status'] == 'ok' for v in expected.values())
    print(f'{len(expected)} files ({n_ok} accepted, {len(expected) - n_ok} rejected); cleaner sha256 {sha256(CLEANER)[:12]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?', default=None)
    ap.add_argument('--stage', choices=('pilot', 'final'), default=None)
    ap.add_argument('--reexport', action='store_true', help='force a fresh ONNX export')
    a = ap.parse_args()
    main(a.ckpt, a.stage, a.reexport)
