"""The real bottle files through the deployment path: raw, cleaned, predicted in 3D, plus the predicted cross-section seen
from above, which is where roundness shows.

    uv run python -m scripts.bottle_views ~/.cache/majproj_ck/f1/best.pt outputs/final/bottle_views.html
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.canon import from_canonical, sample_2048, to_canonical
from majproj.clean import clean
from majproj.infer import load_checkpoint
from scripts.four_views import PLOTLY, complete, size_cm, thin_for_page
from scripts.gen_final import load_config, metric_thin


def section(pred):
    """Middle 40 % of the predicted height, seen from above: radius statistics of the cross-section in cm."""
    z = pred[:, 2]
    lo, hi = np.percentile(z, 30), np.percentile(z, 70)
    m = pred[(z >= lo) & (z <= hi)][:, :2]
    c = np.median(m, 0)
    r = np.linalg.norm(m - c, axis=1) * 100
    return dict(diam_x=round(float(np.percentile(m[:, 0], 97) - np.percentile(m[:, 0], 3)) * 100, 1),
                diam_y=round(float(np.percentile(m[:, 1], 97) - np.percentile(m[:, 1], 3)) * 100, 1),
                r_median=round(float(np.median(r)), 1), r_spread=round(float(np.percentile(r, 90) - np.percentile(r, 10)), 1))


def main(ckpt, out):
    model, info = load_checkpoint(ckpt)
    holdout = set(load_config()[0]['real']['holdout'])
    rng = np.random.default_rng(0)
    cards = []
    for f in sorted(glob.glob(os.path.join(ROOT, 'FINAL-INPUTS', 'BOTTLE', '*.npy'))):
        name = os.path.basename(f)[:-4]
        raw = np.load(f)
        pts, _ = metric_thin(raw)
        r = clean(pts, origin=(0.0, 0.0, 0.0))
        c = dict(title=f'Real bottle {name[5:]}', note='held out of training' if name in holdout else 'session fed the simulator room bank',
                 raw=thin_for_page(raw, 8000, rng), status=r.meta['status'], reason=r.meta.get('reason'))
        if r.meta['status'] == 'ok' and len(r.object) >= 20:
            pred = complete(model, r.object)
            c.update(obj=thin_for_page(r.object, 4000, rng), pred=thin_for_page(pred, 5000, rng), seen_size=size_cm(r.object),
                     pred_size=size_cm(pred), sec=section(pred), sensor=[0.0, 0.0])
        cards.append(c)
        print(c['title'], c['status'], c.get('seen_size'), c.get('pred_size'), c.get('sec'), flush=True)
    head = f"Latest trained model, step {info.get('step')}, every real bottle file. The real bottle is about 7 cm across and 25 cm tall."
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards)).replace('/*HEAD*/', head)
    with open(out, 'w') as fh:
        fh.write(html)
    print('->', out)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bottle Views</title><style>
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#fbfbfa;color:#1b1d21}main{max-width:1700px;margin:0 auto;padding:14px}
h1{font-size:1.3rem;margin:.2em 0}p{max-width:120ch;color:#555}.scan{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0;min-height:400px}
.scan h2{font-size:1rem;margin:0 0 4px}.row{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.p{height:340px;touch-action:none}
.t{font-size:.85rem;font-weight:600;margin:4px 0}.m{font-size:.8rem;color:#555}.none{display:flex;align-items:center;justify-content:center;height:100%;color:#777;background:#f3f3f1;border-radius:6px}
@media(max-width:1000px){.row{grid-template-columns:repeat(2,1fr)}}@media(max-width:560px){.row{grid-template-columns:1fr}}</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Real bottles: raw input, cleaned input, predicted output, predicted cross-section from above</h1>
<p><b>/*HEAD*/</b> Metres, z up, drag to spin. In the predicted view grey is the cleaned input and orange the model's output.
The fourth panel looks straight down at the middle 40 % of the output's height: a good bottle is a round ring about 7 cm across.
The back of the bottle was never measured, so the output's back is the model's inference from the curve of the front.</p><div id="g"></div></main><script>
const S = /*DATA*/;
const CAM = {eye: {x: 1.4, y: -1.6, z: 0.8}, up: {x: 0, y: 0, z: 1}};
function box(arrs) { const lo = [0, 1, 2].map(k => Math.min(...arrs.flatMap(a => a[k]))), hi = [0, 1, 2].map(k => Math.max(...arrs.flatMap(a => a[k])));
  const h = 0.55 * Math.max(...hi.map((v, k) => v - lo[k]), 1e-3); return [0, 1, 2].map(k => [(lo[k] + hi[k]) / 2 - h, (lo[k] + hi[k]) / 2 + h]); }
function plot(el, traces, rng) { Plotly.newPlot(el, traces, {margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
  scene: {aspectmode: 'cube', dragmode: 'turntable', camera: CAM, xaxis: {range: rng[0], title: 'x'}, yaxis: {range: rng[1], title: 'y'}, zaxis: {range: rng[2], title: 'z (up)'}}}, {displaylogo: false, responsive: true}); }
const T = (a, col, sz, cs) => ({type: 'scatter3d', mode: 'markers', x: a[0], y: a[1], z: a[2], hoverinfo: 'skip', marker: cs ? {size: sz, color: a[2], colorscale: 'Viridis'} : {size: sz, color: col}});
function topView(el, s) {   // middle 40 % of the height, x-y only, equal axes
  const pick = (a, lo, hi) => { const o = [[], []]; for (let i = 0; i < a[2].length; i++) if (a[2][i] >= lo && a[2][i] <= hi) { o[0].push(a[0][i]); o[1].push(a[1][i]); } return o; };
  const zs = [...s.pred[2]].sort((p, q) => p - q), lo = zs[Math.floor(zs.length * .3)], hi = zs[Math.floor(zs.length * .7)];
  const pp = pick(s.pred, lo, hi), oo = pick(s.obj, lo, hi), cx = pp[0].reduce((p, q) => p + q, 0) / pp[0].length, cy = pp[1].reduce((p, q) => p + q, 0) / pp[1].length, h = 0.08;
  Plotly.newPlot(el, [{type: 'scatter', mode: 'markers', x: pp[0], y: pp[1], marker: {size: 3, color: '#ea580c'}, hoverinfo: 'skip', name: 'predicted'},
    {type: 'scatter', mode: 'markers', x: oo[0], y: oo[1], marker: {size: 3, color: '#111'}, hoverinfo: 'skip', name: 'seen'}],
    {margin: {l: 36, r: 8, t: 8, b: 30}, showlegend: false, xaxis: {range: [cx - h, cx + h], title: 'x (m)'}, yaxis: {range: [cy - h, cy + h], title: 'y (m), sensor side is -y', scaleanchor: 'x', scaleratio: 1}}, {displaylogo: false, responsive: true});
}
const cm = s => s ? s.join(' x ') + ' cm' : '';
for (const s of S) {
  const d = document.createElement('div'); d.className = 'scan';
  const m = s.obj ? `${s.note} · seen ${cm(s.seen_size)} · predicted ${cm(s.pred_size)} · cross-section ${s.sec.diam_x} x ${s.sec.diam_y} cm, median radius ${s.sec.r_median} cm, radius spread (10-90 %) ${s.sec.r_spread} cm` : `${s.note} · rejected: ${s.reason}`;
  d.innerHTML = `<h2>${s.title}</h2><div class="m">${m}</div><div class="row"><div><div class="t">1. Raw input</div><div class="p"></div></div>
    <div><div class="t">2. Cleaned input</div><div class="p"></div></div><div><div class="t">3. Predicted output</div><div class="p"></div></div>
    <div><div class="t">4. Predicted cross-section, from above</div><div class="p"></div></div></div>`;
  document.getElementById('g').appendChild(d);
  const P = d.querySelectorAll('.p');
  const draw = () => {
    plot(P[0], [T(s.raw, null, 1.4, true)], box([s.raw]));
    if (!s.obj) { for (const i of [1, 2, 3]) P[i].innerHTML = '<div class="none">scan rejected</div>'; return; }
    const rng = box([s.obj, s.pred]);
    plot(P[1], [T(s.obj, '#111', 2)], rng); plot(P[2], [T(s.pred, '#ea580c', 1.5), T(s.obj, '#9ca3af', 1.2)], rng); topView(P[3], s);
  };
  const free = () => P.forEach(e => { if (e.data) Plotly.purge(e); });
  let on = false;
  new IntersectionObserver(es => { const v = es[0].isIntersecting; if (v !== on) { on = v; v ? draw() : free(); } }, {rootMargin: '100px'}).observe(d);
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
