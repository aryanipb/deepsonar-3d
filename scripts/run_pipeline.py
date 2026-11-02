"""Full pipeline on every scan in a folder: raw file -> majproj.clean -> official AdaPoinTr (PCN weights).

    uv run python -m scripts.run_pipeline inputs_new outputs/pipeline_new
    -> <out>/index.html (3 views per scan: raw, cleaned object, model output), <out>/<scan>_object.npy,
       <out>/<scan>_completed.npy (file coordinates), <out>/summary.json
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from majproj.clean import clean
from majproj.model import load_adapointr

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
CKPT = os.path.join(ROOT, 'checkpoints', 'AdaPoinTr_PCN.pth')


def thin(points, voxel=0.01):
    """Raw metric maps (metres, far denser than the 4096-point exports) -> one point per `voxel` metres, the device's own
    10 mm map resolution, which is the density the cleaner's thresholds work at. Exports (max-abs = 1) pass unchanged."""
    if np.abs(points).max() <= 1.0 + 1e-6:
        return points, False
    p = np.unique(points.astype(np.float64), axis=0)
    _, i = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
    return p[i], True


def complete(model, obj, rng):
    """Cleaned object (file coords, z up) -> 16384 completed points in file coords.
    PCN frame is y up with the object centred and about unit size, so swap y/z, centre, scale longest side to 0.9."""
    q = obj.astype(np.float64)[:, [0, 2, 1]]
    c = (q.max(0) + q.min(0)) / 2
    s = np.ptp(q, 0).max() / 0.9
    part = ((q - c) / s)[rng.choice(len(q), 2048, replace=len(q) < 2048)].astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(part[None]))[1][0].numpy().astype(np.float64)
    return (pred * s + c)[:, [0, 2, 1]].astype(np.float32)


def main(folder, out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    model = load_adapointr(CKPT, 'cpu').eval()
    rows, cards = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, folder, '*.npy'))):
        name = os.path.basename(f)
        raw = np.load(f)
        pts, metric = thin(raw)
        r = clean(pts, origin=(0.0, 0.0, 0.0)) if metric else clean(pts)   # raw maps keep the sensor at the origin
        m = r.meta
        row = dict(name=name, status=m['status'], reason=m.get('reason'), confidence=m.get('confidence'),
                   n_unique=m['n_unique'], n_object=m.get('n_object', 0), width=m.get('width'), height=m.get('height'),
                   aspect_w_over_h=m.get('aspect_w_over_h'), shape_hint=m.get('shape_hint'), metric=metric,
                   n_raw=len(raw), centre=m.get('centre'))
        shown = raw[rng.choice(len(raw), min(len(raw), 12000), replace=False)]       # keep the page light on large raw files
        card = dict(name=name, row=row, raw=shown.round(4).T.tolist())
        if m['status'] == 'ok' and len(r.object) >= 20:
            comp = complete(model, r.object, rng)
            stem = name.replace('_shapenet.npy', '')
            np.save(os.path.join(out, f'{stem}_object.npy'), r.object)
            np.save(os.path.join(out, f'{stem}_completed.npy'), comp)
            ext = lambda p: (np.percentile(p, 99, 0) - np.percentile(p, 1, 0)).round(3).tolist()
            row.update(object_extent=ext(r.object), completed_extent=ext(comp))
            card.update(obj=r.object.round(4).T.tolist(), comp=comp[rng.choice(len(comp), 4096, replace=False)].round(4).T.tolist())
        rows.append(row)
        cards.append(card)
        print(f"{name}: {len(raw)} pts, {row['status']} conf {row['confidence']} object {row['n_object']} pts, w/h {row['aspect_w_over_h']}, "
              f"object extent {row.get('object_extent')} -> completed {row.get('completed_extent')} {row['reason'] or ''}")
    json.dump(rows, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards))
    open(os.path.join(out, 'index.html'), 'w').write(html)
    print('->', os.path.join(out, 'index.html'))


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scan pipeline</title><style>
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#fbfbfa;color:#1b1d21}main{max-width:1500px;margin:0 auto;padding:14px}
h1{font-size:1.3rem;margin:.2em 0}p{max-width:120ch;color:#555}.scan{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0;min-height:500px}
.scan h2{font-size:1rem;margin:0 0 4px}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.p{height:420px;touch-action:none}
.t{font-size:.85rem;font-weight:600;margin:4px 0}.m{font-size:.8rem;color:#555}.rej{color:#b91c1c;font-weight:600}
@media(max-width:900px){.row{grid-template-columns:1fr}}</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Scan pipeline: raw file &rarr; cleaned object &rarr; model output</h1>
<p>Left: the file exactly as saved by the device (colour = height). Middle: the object kept by the cleaner (<code>majproj.clean</code>, no labels, no learning).
Right: black = cleaned object, orange = output of the official AdaPoinTr PCN weights (not fine-tuned; trained on 8 categories without bottles or boxes).
Up is up (z). Drag to spin.</p><div id="g"></div></main><script>
const S = /*DATA*/;
const CAM = {eye: {x: 1.4, y: -1.6, z: 0.8}, up: {x: 0, y: 0, z: 1}};
function box(arrs) { const lo = [0, 1, 2].map(k => Math.min(...arrs.flatMap(a => a[k]))), hi = [0, 1, 2].map(k => Math.max(...arrs.flatMap(a => a[k])));
  const h = 0.55 * Math.max(...hi.map((v, k) => v - lo[k]), 1e-3); return [0, 1, 2].map(k => [(lo[k] + hi[k]) / 2 - h, (lo[k] + hi[k]) / 2 + h]); }
function plot(el, traces, rng) { Plotly.newPlot(el, traces, {margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
  scene: {aspectmode: 'cube', dragmode: 'turntable', camera: CAM, xaxis: {range: rng[0], title: 'x'}, yaxis: {range: rng[1], title: 'y'}, zaxis: {range: rng[2], title: 'z (up)'}}}, {displaylogo: false, responsive: true}); }
const T = (a, col, sz, cs) => ({type: 'scatter3d', mode: 'markers', x: a[0], y: a[1], z: a[2], hoverinfo: 'skip', marker: cs ? {size: sz, color: a[2], colorscale: 'Viridis'} : {size: sz, color: col}});
for (const s of S) {
  const r = s.row, d = document.createElement('div'); d.className = 'scan';
  const head = r.status === 'ok' ? `confidence ${r.confidence} · object ${r.n_object} of ${r.n_unique} points${r.metric ? ` (after 1 cm thinning of ${r.n_raw})` : ''} · ${r.metric ? `${(100 * r.width).toFixed(1)} cm wide x ${(100 * r.height).toFixed(1)} cm tall, ${(100 * Math.hypot(r.centre[0], r.centre[1])).toFixed(0)} cm from the sensor origin` : `width/height ${r.aspect_w_over_h}`} · ${r.shape_hint}`
    : `<span class="rej">rejected: ${r.reason}</span>`;
  d.innerHTML = `<h2>${s.name}</h2><div class="m">${head}</div><div class="row"><div><div class="t">1. Raw file</div><div class="p"></div></div>
    <div><div class="t">2. Cleaned object</div><div class="p"></div></div><div><div class="t">3. Model output</div><div class="p"></div></div></div>`;
  document.getElementById('g').appendChild(d);
  const P = d.querySelectorAll('.p');
  // Chrome allows about 16 WebGL contexts: draw a row only while it is near the viewport, free it when it leaves
  const draw = () => {
    plot(P[0], [T(s.raw, null, 1.6, true)], box([s.raw]));
    if (s.obj) { const rng = box([s.obj, s.comp]); plot(P[1], [T(s.obj, '#111', 2.4)], rng); plot(P[2], [T(s.comp, '#ea580c', 1.6), T(s.obj, '#111', 2.2)], rng); }
    else { P[1].innerHTML = P[2].innerHTML = '<p class="rej">no object (scan rejected)</p>'; }
  };
  const free = () => P.forEach(e => { if (e.data) Plotly.purge(e); });
  let on = false;
  new IntersectionObserver(es => { const v = es[0].isIntersecting; if (v !== on) { on = v; v ? draw() : free(); } }, {rootMargin: '150px'}).observe(d);
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
