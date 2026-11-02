"""Interactive 3D view of the raw input files, exactly as given (no cleaning, cropping or model).

    uv run python -m scripts.plot_inputs            -> outputs/inputs_3d.html (self-contained, opens offline)
    uv run python -m scripts.plot_inputs inputs_box outputs/inputs_box_3d.html "Box"   (other folder, one label for all)
    uv run python -m scripts.plot_inputs inputs_box outputs/box_object_only_3d.html "Box" object   (wall removed)
"""
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.labels import USER_NAMES

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
OUT = os.path.join(ROOT, 'outputs', 'inputs_3d.html')


def near_cluster(p, bin_w=0.05, min_gap=0.2):
    """Points in front of the first empty depth gap of at least `min_gap` (the wall is the far layer behind it)."""
    edges = np.arange(p[:, 2].min(), p[:, 2].max() + bin_w, bin_w)
    h, _ = np.histogram(p[:, 2], bins=edges)
    run = int(round(min_gap / bin_w))
    for i in range(len(h) - run + 1):
        if h[i:i + run].sum() == 0 and h[:i].sum() > 0:
            return p[p[:, 2] < edges[i]], float(edges[i])
    return p, None


def main(folder='inputs', out=OUT, label=None, mode='raw'):
    files = sorted(glob.glob(os.path.join(ROOT, folder, 'scan_*_shapenet.npy')))
    scans = []
    for f in files:
        raw = open(f, 'rb').read()
        p = np.load(f)
        name = os.path.basename(f)
        n_all = len(p)
        cut = None
        if mode == 'object':
            p, cut = near_cluster(p)
        uniq = len(np.unique(p, axis=0))
        scans.append(dict(name=name, label=label or USER_NAMES[name[:-len('.npy')]], sha=hashlib.sha256(raw).hexdigest()[:12],
                          shape=list(p.shape), dtype=str(p.dtype), unique=uniq, n_all=n_all, cut=cut,
                          lo=[round(float(v), 3) for v in p.min(0)], hi=[round(float(v), 3) for v in p.max(0)],
                          x=p[:, 0].round(5).tolist(), y=p[:, 1].round(5).tolist(), z=p[:, 2].round(5).tolist()))
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(scans))
    html = html.replace('Your 10 input files', f'Your {len(scans)} input files')
    if mode == 'object':
        html = html.replace('in 3D, exactly as given</h1>', 'in 3D: object only (wall removed)</h1>')
        html = html.replace('with no cleaning, cropping or model.', 'with the far wall removed: only the points in front of the first empty depth gap (at least 0.2 units deep) are kept. Nothing else is changed. Each plot zooms to its own points with true proportions.')
        html = html.replace(' Same axes (-1 to 1) for every file.', '')
        html = html.replace("aspectmode: 'cube'", "aspectmode: 'data'").replace("range: [-1.05, 1.05], ", "")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, 'w').write(html)
    print(f'{len(scans)} files -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Input scans 3D</title>
<style>
:root{--bg:#fbfbfa;--fg:#1b1d21;--mut:#5b6068;--card:#fff;--line:#d9dbe0}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#a0a6af;--card:#1c1f24;--line:#33383f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}
.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h2{font-size:1rem;margin:0}.plot{height:480px;touch-action:none}code{font-size:.85em}
.views{margin:6px 0 0;display:flex;gap:6px;flex-wrap:wrap}.views button{font:inherit;font-size:.8rem;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Your 10 input files in 3D, exactly as given</h1>
<p class="mut">Each plot shows all points of one <code>scan_*_shapenet.npy</code> file with no cleaning, cropping or model.
Up is up, depth goes away from the sensor. Colour = depth (dark blue = near the sensor, yellow = far). Same axes (-1 to 1) for every file.
Drag to spin around the vertical axis, scroll or pinch to zoom, or use the view buttons. Labels are the ones you gave.</p>
<div class="grid" id="grid"></div></main>
<script>
const scans = /*DATA*/;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e6e8eb' : '#1b1d21', grid = dark ? '#33383f' : '#d9dbe0';
const ax = t => ({title: t, range: [-1.05, 1.05], color: fg, gridcolor: grid, zerolinecolor: grid, backgroundcolor: 'rgba(0,0,0,0)'});
// Plotly's turntable drag always spins about its own z axis, so plot z = file y (up) and plot y = file z (depth).
const VIEWS = {
  '3D': {eye: {x: 1.5, y: -1.7, z: 0.9}, up: {x: 0, y: 0, z: 1}},
  'From the sensor': {eye: {x: 0, y: -2.3, z: 0.05}, up: {x: 0, y: 0, z: 1}},
  'Top': {eye: {x: 0, y: -0.01, z: 2.4}, up: {x: 0, y: 1, z: 0}},
  'Side': {eye: {x: 2.4, y: 0, z: 0.05}, up: {x: 0, y: 0, z: 1}},
};
for (const s of scans) {
  const c = document.createElement('div'); c.className = 'card';
  c.innerHTML = `<h2>${s.label} <span class="mut">· ${s.name}</span></h2>
  <div class="mut" style="font-size:.8rem">${s.cut === null ? '' : `kept ${s.shape[0]} of ${s.n_all} points (depth < ${s.cut.toFixed(2)}), `}${s.shape.join(' x ')} ${s.dtype}, ${s.unique} unique points, sha256 ${s.sha}…,
  x ${s.lo[0]}..${s.hi[0]}, y ${s.lo[1]}..${s.hi[1]}, z ${s.lo[2]}..${s.hi[2]}</div>
  <div class="views">${Object.keys(VIEWS).map(v => `<button data-v="${v}">${v}</button>`).join('')}</div><div class="plot"></div>`;
  document.getElementById('grid').appendChild(c);
  const el = c.querySelector('.plot');
  Plotly.newPlot(el, [{type: 'scatter3d', mode: 'markers', x: s.x, y: s.z, z: s.y,
    marker: {size: 2.5, color: s.z, colorscale: 'Viridis', showscale: false},
    hovertemplate: 'x %{x:.3f}<br>up (y) %{z:.3f}<br>depth (z) %{y:.3f}<extra></extra>'}],
    {margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: 'rgba(0,0,0,0)', showlegend: false,
     scene: {xaxis: ax('x'), yaxis: ax('depth (z)'), zaxis: ax('up (y)'), aspectmode: 'cube', dragmode: 'turntable',
             camera: VIEWS['3D']}},
    {responsive: true, displaylogo: false});
  c.querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => Plotly.relayout(el, {'scene.camera': VIEWS[b.dataset.v]})));
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
