"""Show the registered parametric target next to each real scan's cleaned face.

    uv run python -m scripts.target_views outputs/real_targets/target_views.html
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from majproj.canon import from_canonical  # noqa: E402
from majproj.real_targets import targets_for_pool  # noqa: E402
from scripts.gen_final import N_TARGET, real_pool  # noqa: E402

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')


def main(out, cfg_path='configs/final.yaml'):
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, cfg_path)))
    pool, rejected = real_pool(os.path.join(ROOT, 'FINAL-INPUTS'), cfg['real']['holdout'])
    targets = cfg['real'].get('targets') or {'BOTTLE': 'configs/real_bottle_parts.json'}
    gts, infos = targets_for_pool(pool, {k: os.path.join(ROOT, v) for k, v in targets.items()}, N_TARGET, cfg['real'].get('heights'))
    rng = np.random.default_rng(0)
    cards = []
    for s in pool:
        if s['session'] not in infos:
            continue
        face = from_canonical(s['partial'], s['T'])
        c = dict(session=s['session'], object=s['object'], held=s['held_out'], info=infos[s['session']], face=face.round(4).T.tolist())
        g = gts[s['session']]
        if g is not None:
            t = from_canonical(g, s['T'])
            c['target'] = t[rng.choice(len(t), 5000, replace=False)].round(4).T.tolist()
        cards.append(c)
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards))
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    open(out, 'w').write(html)
    json.dump({k: v for k, v in infos.items()}, open(os.path.splitext(out)[0] + '.json', 'w'), indent=1)
    print(sum(1 for c in cards if 'target' in c), 'accepted of', len(cards), '->', out)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Registered Targets</title><style>
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#fbfbfa;color:#1b1d21}main{max-width:1500px;margin:0 auto;padding:14px}
h1{font-size:1.3rem;margin:.2em 0}p{max-width:120ch;color:#555}.scan{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:12px 0}
.scan h2{font-size:1rem;margin:0 0 4px}.row{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.p{height:380px}.m{font-size:.8rem;color:#555}
@media(max-width:700px){.row{grid-template-columns:1fr}}</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Targets for real scans: the known object registered to each scan's front face</h1>
<p>The bottle is a solid of revolution, so its complete surface is known from its outline. For each real scan the outline is scaled to the
scan's height and placed (translation and yaw only) so that its front lies on the cleaned face. The placed surface (blue) is the training
target for that scan; the face (black) is what the sensor measured. This is not a measurement of the back: it is the real object's shape
registered to the real front, and the registration residual is the floor of any agreement with it. Metres, z up, drag to spin.</p>
<div id="g"></div></main><script>
const S = /*DATA*/;
const CAM = {eye: {x: 1.4, y: -1.6, z: 0.8}, up: {x: 0, y: 0, z: 1}};
function box(arrs) { const lo = [0, 1, 2].map(k => Math.min(...arrs.flatMap(a => a[k]))), hi = [0, 1, 2].map(k => Math.max(...arrs.flatMap(a => a[k])));
  const h = 0.55 * Math.max(...hi.map((v, k) => v - lo[k]), 1e-3); return [0, 1, 2].map(k => [(lo[k] + hi[k]) / 2 - h, (lo[k] + hi[k]) / 2 + h]); }
function plot(el, traces, rng) { Plotly.newPlot(el, traces, {margin: {l: 0, r: 0, t: 0, b: 0}, showlegend: false,
  scene: {aspectmode: 'cube', dragmode: 'turntable', camera: CAM, xaxis: {range: rng[0]}, yaxis: {range: rng[1]}, zaxis: {range: rng[2], title: 'z (up)'}}}, {displaylogo: false, responsive: true}); }
const T = (a, col, sz) => ({type: 'scatter3d', mode: 'markers', x: a[0], y: a[1], z: a[2], hoverinfo: 'skip', marker: {size: sz, color: col}});
for (const s of S) {
  const d = document.createElement('div'); d.className = 'scan'; const i = s.info;
  d.innerHTML = `<h2>${s.object.toLowerCase()} ${s.session.slice(-6)} ${s.held ? '(held out of training)' : ''}</h2><div class="m">${i.accepted ? 'ACCEPTED' : 'REJECTED'} · registration median ${i.resid_median_mm} mm, p90 ${i.resid_p90_mm} mm · front inside the face depth: ${i.front_within_face_depth} · scale from ${i.scale_source} (seen height ${(i.face_height_m*100).toFixed(1)} cm) · target size ${i.target_size_cm ? i.target_size_cm.join(' x ') + ' cm' : '-'}</div>
    <div class="row"><div><div class="m">cleaned face (sensor measurement)</div><div class="p"></div></div><div><div class="m">target: registered object (blue) over the face (grey)</div><div class="p"></div></div></div>`;
  document.getElementById('g').appendChild(d); const P = d.querySelectorAll('.p');
  const rng = box(s.target ? [s.face, s.target] : [s.face]);
  plot(P[0], [T(s.face, '#111', 2)], rng);
  if (s.target) plot(P[1], [T(s.target, '#1d4ed8', 1.5), T(s.face, '#9ca3af', 1.5)], rng); else P[1].innerHTML = '<div class="m">no target: registration rejected, this session keeps the one-sided loss</div>';
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
