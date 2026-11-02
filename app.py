"""Local inference UI: ONE scan file -> the cleaned object and the complete object (Python/Gradio version of the web app).

Run:  uv run python app.py [--ckpt path]   ->  http://127.0.0.1:7860

Flow (majproj.infer). If the cleaner rejects the scan the model is not run and the reason is shown instead. Depth and the
back of the object are predicted, never measured. Non-final checkpoints are labelled in every output.
"""
import argparse
import glob
import os
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from majproj.infer import Completer, write_outputs
from scripts.build_web import RAW_DIRS, session_of
from majproj.sim_real import holdout_sessions

ROOT = os.path.dirname(os.path.abspath(__file__))
PRED = '#0b8fd6'  # colour of the complete object
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default=None)
ap.add_argument('--stage', default=None, choices=('pilot', 'final'))
ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 7860)))
args = ap.parse_args()
HELD = set(holdout_sessions())
CHOICES = []
for folder, label in RAW_DIRS.items():  # the label is the user's folder, not a model output
    for f in sorted(glob.glob(os.path.join(ROOT, folder, '*.npy'))):
        held = ', held out' if session_of(f) in HELD else ''
        CHOICES.append((f'{label}: {os.path.basename(f)[:-4]}{held}', f))
completer = Completer(args.ckpt, stage=args.stage)
BANNER = '' if completer.stage == 'final' else f'**{completer.stage.upper()} MODEL: not final results.** '


def _trace(pts, color, name, size=1.6, opacity=0.9, max_pts=12000):
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='markers', name=name,
                        marker=dict(size=size, color=color, opacity=opacity), hoverinfo='skip')


def _figure(traces, title, height=520):
    fig = go.Figure(traces)
    fig.update_layout(title=title, height=height, margin=dict(l=0, r=0, t=40, b=0), legend=dict(itemsizing='constant'),
                      scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y', zaxis_title='z (up)',
                                 camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.8), up=dict(x=0, y=0, z=1))))
    return fig


def describe(res):
    """Markdown summary of one result: status, reason, measured and predicted sizes, and the caveats that apply."""
    cm = res.clean_meta
    rows = ['| | |', '|---|---|']
    rows.append(f"| Status | **{'OK: object found' if res.ok else 'REJECTED: no completion'}** |")
    if not res.ok:
        rows.append(f'| Reason | {res.reason} |')
    if res.confidence is not None:
        rows.append(f'| Cleaner confidence | {res.confidence:.2f} |')
    rows.append(f"| Points | {cm.get('n_unique', '?')} unique after thinning; {len(res.observed)} belong to the object"
                + (f' -> {len(res.completed)} complete' if res.ok else '') + ' |')
    if res.ok:
        m, p, u = res.sizes['measured'], res.sizes['predicted'], res.sizes['unit']
        d = 1 if res.metric else 3
        rows.append(f"| Face, measured | {m['face_width']:.{d}f} wide x {m['face_height']:.{d}f} high {u} |")
        if m['distance'] is not None:
            rows.append(f"| Distance, measured | {m['distance']:.{d}f} {u} |")
        rows.append(f"| Output, predicted | {p['width']:.{d}f} wide x {p['height']:.{d}f} high {u} |")
        rows.append(f"| Depth, predicted (not measured) | {p['depth']:.{d}f} {u} |")
    rows.append(f'| Model | `{completer.ckpt_name}`, stage {completer.stage}, {completer.device} |')
    return '\n'.join(rows) + '\n\n' + '\n'.join(f'- {w}' for w in res.warnings)


def run(choice, upload, show_background):
    path = upload if upload else choice
    if not path:
        raise gr.Error('Pick a scan or upload a .npy file (N x 3).')
    try:
        scan = np.load(path, allow_pickle=False)
    except Exception as e:
        raise gr.Error(f'Could not read {os.path.basename(path)}: {e}')
    if scan.ndim != 2 or scan.shape[1] != 3 or len(scan) < 3:
        raise gr.Error(f'Expected an N x 3 array, got shape {scan.shape}.')
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        res = completer(scan)
    except ValueError as e:
        raise gr.Error(str(e))
    if res.ok:
        close = _figure([_trace(res.completed, PRED, 'completed object', 1.6, 0.45),
                         _trace(res.observed, '#111111', 'cleaned object (what the sensor saw)', 2.4, 1.0)],
                        BANNER.replace('*', '') + 'Black = cleaned object, blue = complete object')
    else:
        close = _figure([_trace(res.background, '#9aa0a6', 'everything in the file (rejected)', 1.4, 0.6)],
                        'Rejected: no object found, nothing completed')
    traces = []
    if show_background or not res.ok:
        traces.append(_trace(res.background, '#9aa0a6', 'background (wall, ceiling, strays)', 1.2, 0.35))
    if res.ok:
        traces += [_trace(res.completed, PRED, 'completed object', 1.2, 0.6),
                   _trace(res.observed, '#111111', 'cleaned object', 1.8, 1.0)]
    scene = _figure(traces, 'Whole scene (sensor at the origin, z up)')
    tmp = tempfile.mkdtemp(prefix='majproj_')
    files = write_outputs(res, tmp, stem, completer.ckpt_name, completer.stage)
    return close, scene, f'### {stem}\n\n{BANNER}\n\n' + describe(res), files


with gr.Blocks(title='One-File 3D Completion') as demo:
    gr.Markdown('# One-file 3D completion\n' + BANNER +
                'One hand-held 8x8 time-of-flight scan goes in. A geometry-only cleaner finds the object or **rejects the '
                'file with a reason**. For an accepted scan the network outputs the complete object in centimetres.\n\n'
                '**Depth and the back of the object are predicted, not measured.** Black = the cleaned object the sensor '
                'saw, blue = the network output. The network gets points only: no label, no template.')
    with gr.Row():
        choice = gr.Dropdown(CHOICES, value=CHOICES[0][1] if CHOICES else None,
                             label='Real scan (the name is your folder, not a prediction)')
        upload = gr.File(label='...or upload one scan (.npy, N x 3)', file_types=['.npy'], type='filepath')
        show_bg = gr.Checkbox(True, label='Show background in scene view')
    btn = gr.Button('Complete', variant='primary')
    summary = gr.Markdown()
    with gr.Row():
        fig_obj = gr.Plot(label='Object close-up')
        fig_scene = gr.Plot(label='Scene')
    files = gr.File(label='Download: complete object (.npy / .ply, accepted scans only) and metadata (.json)',
                    file_count='multiple')
    inputs, outputs = [choice, upload, show_bg], [fig_obj, fig_scene, summary, files]
    btn.click(run, inputs, outputs)
    choice.change(run, inputs, outputs)
    demo.load(run, inputs, outputs)

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=args.port)
