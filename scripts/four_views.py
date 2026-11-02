"""Four 3D views per scan for a training checkpoint: raw input, cleaned input, predicted output, target output.

    uv run python -m scripts.four_views ~/.cache/majproj_ck/final1_now.pt outputs/final/live_4views.html

Sim scans use keys no split ever drew and are the first accepted scan per kind, unselected by quality. Real scans have no
target: the back of the real objects was never measured.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from scipy.spatial import cKDTree

from majproj.canon import from_canonical, sample_2048, to_canonical
from majproj.clean import clean
from majproj.infer import load_checkpoint
from majproj.sim_raw import simulate_raw
from scripts.gen_final import load_config, metric_thin

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
SIM_KINDS = ('cuboid', 'cuboid', 'cube', 'tower', 'slab', 'bottle', 'bottle', 'can', 'jar', 'rod')
UNSEEN_SHARD = 9000
REAL = (('BOX', 'scan_20261030_121600', True), ('BOTTLE', 'scan_20261030_123815', True),
        ('BOX', 'scan_20261030_114126', False), ('BOTTLE', 'scan_20261030_122622', False))


def chamfer_mm(a, b):
    return 500.0 * (cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean())


def size_cm(p):
    return [round(float(v), 1) for v in 100 * (np.percentile(p, 99, 0) - np.percentile(p, 1, 0))]


def thin_for_page(p, n, rng):
    return p[rng.choice(len(p), min(n, len(p)), replace=False)].round(4).T.tolist()


@torch.no_grad()
def complete(model, obj):
    """Deployment path: cleaned object in metres -> completed object in metres."""
    canon, T = to_canonical(obj)
    pred = model(torch.from_numpy(sample_2048(canon)[None]))[1][0].numpy()
    return from_canonical(pred, T)


def card(title, note, raw, pts, target, model, rng):
    r = clean(pts, origin=(0.0, 0.0, 0.0))
    c = dict(title=title, note=note, raw=thin_for_page(raw, 8000, rng), status=r.meta['status'], reason=r.meta.get('reason'))
    if r.meta['status'] != 'ok' or len(r.object) < 20:
        return c
    pred = complete(model, r.object)
    c.update(obj=thin_for_page(r.object, 4000, rng), pred=thin_for_page(pred, 5000, rng),
             seen_size=size_cm(r.object), pred_size=size_cm(pred))
    if target is not None:
        c.update(target=thin_for_page(target, 5000, rng), target_size=size_cm(target), cd=round(chamfer_mm(pred, target), 1))
    return c


def main(ckpt, out):
    model, info = load_checkpoint(ckpt)
    cfg = load_config()[0]
    page_rng = np.random.default_rng(0)
    cards = []
    i = 0
    for n, kind in enumerate(SIM_KINDS):
        while True:                                  # the first scan the cleaner accepts, good or bad
            i += 1
            s = simulate_raw(np.random.default_rng([cfg['seed'], 2, UNSEEN_SHARD, i]), kind=kind, regime='real')
            pts, _ = metric_thin(s['raw'])
            c = card(f'Simulated {kind}', 'never seen in training', s['raw'], pts, np.asarray(s['complete'], np.float64), model, page_rng)
            if 'pred' in c:
                break
        cards.append(c)
        print(c['title'], c['cd'], 'mm', c['pred_size'], 'target', c['target_size'], flush=True)
    for ob, name, held in REAL:
        raw = np.load(os.path.join(ROOT, 'FINAL-INPUTS', ob, name + '.npy'))
        pts, _ = metric_thin(raw)
        cards.append(card(f'Real {ob.lower()} {name[-6:]}', 'held out of training' if held else 'session fed the simulator room bank',
                          raw, pts, None, model, page_rng))
        print(cards[-1]['title'], cards[-1].get('pred_size'), cards[-1]['status'], flush=True)
    head = f"Checkpoint step {info.get('step')} of about 38,500: training is still running, these are not final results."
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards)).replace('/*HEAD*/', head)
    with open(out, 'w') as f:
        f.write(html)
    print('->', out)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Four Views</title><style>
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#fbfbfa;color:#1b1d21}main{max-width:1700px;margin:0 auto;padding:14px}
h1{font-size:1.3rem;margin:.2em 0}p{max-width:120ch;color:#555}.scan{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0;min-height:400px}
.scan h2{font-size:1rem;margin:0 0 4px}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.p{height:340px;touch-action:none}
.t{font-size:.85rem;font-weight:600;margin:4px 0}.m{font-size:.8rem;color:#555}.none{display:flex;align-items:center;justify-content:center;height:100%;color:#777;text-align:center;padding:12px;background:#f3f3f1;border-radius:6px}
@media(max-width:1000px){.row{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.row{grid-template-columns:1fr}}</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Raw input, cleaned input, predicted output, target output</h1>
<p><b>/*HEAD*/</b> Units are metres, z is up, drag to spin. Grey in the predicted view is the cleaned input. A target exists only for simulated scans:
nobody measured the back of the real box or bottle, so real scans have no target and the predicted depth cannot be checked.</p><div id="g"></div></main><script>
const S = /*DATA*/;
const CAM = {eye: {x: 1.4, y: -1.6, z: 0.8}, up: {x: 0, y: 0, z: 1}};
function box(arrs) { const lo = [0, 1, 2].map(k => Math.min(...arrs.flatMap(a => a[k]))), hi = [0, 1, 2].map(k => Math.max(...arrs.flatMap(a => a[k])));
  const h = 0.55 * Math.max(...hi.map((v, k) => v - lo[k]), 1e-3); return [0, 1, 2].map(k => [(lo[k] + hi[k]) / 2 - h, (lo[k] + hi[k]) / 2 + h]); }
function plot(el, traces, rng) { Plotly.newPlot(el, traces, {margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
  scene: {aspectmode: 'cube', dragmode: 'turntable', camera: CAM, xaxis: {range: rng[0], title: 'x'}, yaxis: {range: rng[1], title: 'y'}, zaxis: {range: rng[2], title: 'z (up)'}}}, {displaylogo: false, responsive: true}); }
const T = (a, col, sz, cs) => ({type: 'scatter3d', mode: 'markers', x: a[0], y: a[1], z: a[2], hoverinfo: 'skip', marker: cs ? {size: sz, color: a[2], colorscale: 'Viridis'} : {size: sz, color: col}});
const cm = s => s ? s.join(' x ') + ' cm' : '';
for (const s of S) {
  const d = document.createElement('div'); d.className = 'scan';
  const m = s.obj ? `${s.note} · seen ${cm(s.seen_size)} · predicted ${cm(s.pred_size)}${s.target ? ` · target ${cm(s.target_size)} · error ${s.cd} mm (Chamfer L1)` : ' · no target exists'}` : `${s.note} · rejected: ${s.reason}`;
  d.innerHTML = `<h2>${s.title}</h2><div class="m">${m}</div><div class="row"><div><div class="t">1. Raw input</div><div class="p"></div></div>
    <div><div class="t">2. Cleaned input</div><div class="p"></div></div><div><div class="t">3. Predicted output</div><div class="p"></div></div>
    <div><div class="t">4. Target output</div><div class="p"></div></div></div>`;
  document.getElementById('g').appendChild(d);
  const P = d.querySelectorAll('.p');
  const draw = () => {
    plot(P[0], [T(s.raw, null, 1.4, true)], box([s.raw]));
    if (!s.obj) { for (const i of [1, 2, 3]) P[i].innerHTML = '<div class="none">scan rejected</div>'; return; }
    const rng = box(s.target ? [s.obj, s.pred, s.target] : [s.obj, s.pred]);
    plot(P[1], [T(s.obj, '#111', 2)], rng); plot(P[2], [T(s.pred, '#ea580c', 1.5), T(s.obj, '#9ca3af', 1.2)], rng);
    if (s.target) plot(P[3], [T(s.target, '#15803d', 1.5)], rng); else P[3].innerHTML = '<div class="none">No target exists for a real scan: the back was never measured.</div>';
  };
  const free = () => P.forEach(e => { if (e.data) Plotly.purge(e); });
  let on = false;
  new IntersectionObserver(es => { const v = es[0].isIntersecting; if (v !== on) { on = v; v ? draw() : free(); } }, {rootMargin: '100px'}).observe(d);
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
