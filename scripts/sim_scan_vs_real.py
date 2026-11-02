"""Simulated sessions (majproj.sim_scan) next to the real box exports, plus the numbers that decide whether they match.

    uv run python -m scripts.sim_scan_vs_real                 -> outputs/sim_vs_real_box.html (self-contained, offline)
    uv run python -m scripts.sim_scan_vs_real --n 60 --seed 3 --out outputs/x.html

Top row: the 5 real files of inputs_box/ exactly as exported (no truth exists for them). Bottom row: 4 simulated box
scans from the default prior; their points are coloured by the simulator's ground truth (object / not object, or by
source surface or session phase). Every plot uses the file's own axes: x, y horizontal, z = gravity up, so the ceiling
layer sits on top. Drag to spin around the vertical axis, scroll to zoom.

The table compares the same statistics on the real files and on N simulated sessions (default prior, and the near
(0.05-0.3 m) and far (0.5-1.0 m) object-distance regimes separately). All statistics use only the exported points, so
they are computed identically for real and simulated files:
  unique         unique points of the 4096 (duplicates come from sampling with replacement when the raw map is small,
                 or from the un-thinned pending buffer of a device that stood still at the end)
  ceiling frac   fraction of unique points with z > 0.6 (the ceiling layer; its median z is `ceiling z`)
  mid band       unique points with -0.3 < z <= 0.6: everything between the ceiling and the object (swing streaks, walls)
  ceiling hw     half width of the ceiling grid footprint (2nd to 98th percentile of x and y)
  low cluster    points with z <= -0.3 within 0.4 of their median: the object plus whatever sits next to it (hand,
                 table); its size (unique points, 5-95 % extent), height and horizontal offset from the ceiling-grid
                 centre in units of `ceiling hw`
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.sim_scan import PHASES, SOURCES, simulate_scan

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
OUT = os.path.join(ROOT, 'outputs', 'sim_vs_real_box.html')
REAL_GLOB = os.path.join(ROOT, 'inputs_box', 'scan_*_shapenet.npy')
CEIL_Z, MID_LO = 0.6, -0.3


def cloud_stats(p):
    """Statistics of one exported (4096, 3) cloud (see the module docstring)."""
    p = np.asarray(p, np.float64)
    u, c = np.unique(p, axis=0, return_counts=True)
    ceil = u[u[:, 2] > CEIL_Z]
    if len(ceil) < 10:
        ceil = u[u[:, 2] > np.percentile(u[:, 2], 70)]
    lo = [np.percentile(ceil[:, i], 2) for i in (0, 1)]
    hi = [np.percentile(ceil[:, i], 98) for i in (0, 1)]
    cc = (np.array(lo) + np.array(hi)) / 2
    hw = float(np.mean((np.array(hi) - np.array(lo)) / 2))
    d = dict(unique=len(u), max_multiplicity=int(c.max()), ceiling_frac=len(ceil) / len(u), ceiling_z=float(np.median(ceil[:, 2])),
             mid_band=int(((u[:, 2] > MID_LO) & (u[:, 2] <= CEIL_Z)).sum()), ceiling_hw=hw)
    low = u[u[:, 2] <= MID_LO]
    if len(low) > 20:
        near = low[np.linalg.norm(low - np.median(low, axis=0), axis=1) < 0.4]
        if len(near) > 10:
            ext = [float(np.ptp(np.percentile(near[:, i], [5, 95]))) for i in range(3)]
            d.update(low_n=len(near), low_z=float(near[:, 2].mean()), low_offset=float(np.hypot(*(near[:, :2].mean(0) - cc)) / hw),
                     low_ext_x=ext[0], low_ext_y=ext[1], low_ext_z=ext[2])
    return d


def _one(args):
    kind, seed, i, kw = args
    r = simulate_scan(np.random.default_rng([seed, i]), kind=kind, **kw)
    d = cloud_stats(r['export'])
    t = r['truth']
    d.update(pool=t['n_map'], pool_lt_4096=float(t['n_map'] < 4096), obj_unique=int(len(np.unique(r['export'][r['object_mask']], axis=0))),
             obj_points=int(r['object_mask'].sum()), distance_m=t['distance_m'], scale_m=t['scale_m'])
    return d


def sim_stats(n, seed, kw, workers):
    jobs = [('box', seed, i, kw) for i in range(n)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_one, jobs, chunksize=2))


ROWS = [('unique', 'unique points of 4096', '{:.0f}'), ('ceiling_frac', 'ceiling fraction (z > 0.6)', '{:.2f}'),
        ('ceiling_z', 'ceiling z (median)', '{:.2f}'), ('mid_band', 'mid band points (-0.3 < z <= 0.6)', '{:.0f}'),
        ('ceiling_hw', 'ceiling half width', '{:.2f}'), ('low_n', 'low cluster unique points', '{:.0f}'),
        ('low_z', 'low cluster mean z', '{:.2f}'), ('low_offset', 'low cluster offset / ceiling hw', '{:.2f}'),
        ('low_ext_x', 'low cluster extent x (5-95 %)', '{:.2f}'), ('low_ext_y', 'low cluster extent y', '{:.2f}'),
        ('low_ext_z', 'low cluster extent z', '{:.2f}'), ('max_multiplicity', 'max copies of one point', '{:.0f}')]
SIM_ONLY = [('pool', 'raw map size at export', '{:.0f}'), ('pool_lt_4096', 'fraction with raw map < 4096 (duplicates)', '{:.2f}'),
            ('obj_unique', 'object points (truth), unique', '{:.0f}'), ('distance_m', 'object distance (m)', '{:.2f}')]


def _q(rows, key, qs):
    v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return [float(np.percentile(v, q)) for q in qs] if v else [np.nan] * len(qs)


def summary_table(real, groups):
    """Markdown-ish rows: statistic | real min/median/max | each sim group p10/median/p90."""
    out = []
    for key, label, fmt in ROWS + SIM_ONLY:
        if key == 'pool_lt_4096':                       # a 0/1 indicator: show the share; real analogue = files with heavy duplicates
            real_share = np.mean([r['unique'] < 3500 for r in real])
            out.append((label, [f'{real_share:.2f} (unique < 3500)'] + [f'{np.mean([r[key] for r in rows]):.2f}' for _, rows in groups]))
            continue
        cells = []
        r = _q(real, key, (0, 50, 100)) if key in real[0] or any(key in x for x in real) else None
        cells.append(' / '.join(fmt.format(x) for x in r) if r else 'n/a (no truth)')
        for name, rows in groups:
            qs = _q(rows, key, (10, 50, 90))
            cells.append(' / '.join(fmt.format(x) for x in qs))
        out.append((label, cells))
    return out


def main(n=40, seed=7, out=OUT, workers=None):
    workers = workers or min(8, os.cpu_count() or 2)
    files = sorted(glob.glob(REAL_GLOB))
    real_pts = [np.load(f) for f in files]
    real = [cloud_stats(p) for p in real_pts]
    t0 = time.perf_counter()
    groups = [('sim default prior', sim_stats(n, seed, {}, workers)),
              ('sim near 0.05-0.3 m', sim_stats(n, seed, dict(p_near=1.0), workers)),
              ('sim far 0.5-1.0 m', sim_stats(n, seed, dict(p_near=0.0), workers))]
    print(f'{3 * n} simulated sessions in {time.perf_counter() - t0:.0f} s ({workers} workers)')
    table = summary_table(real, groups)

    cards = []
    for f, p in zip(files, real_pts):
        name = os.path.basename(f)
        cards.append(dict(kind='real', title=f'real {name[5:20]}', note=f'{len(np.unique(p, axis=0))} unique points, no truth',
                          x=p[:, 0].round(4).tolist(), y=p[:, 1].round(4).tolist(), z=p[:, 2].round(4).tolist()))
    for i in range(4):
        r = simulate_scan(np.random.default_rng([2026, i]), kind='box')
        e, t = r['export'], r['truth']
        a, b, h = t['size_m']
        cards.append(dict(kind='sim', title=f'simulated box {i + 1}',
                          note=f"{t['n_unique']} unique, raw map {t['n_map']}, object {int(r['object_mask'].sum())} pts, "
                               f"{a * 100:.0f}x{b * 100:.0f}x{h * 100:.0f} cm at {t['distance_m'] * 100:.0f} cm, {t['duration_s']:.0f} s session",
                          x=e[:, 0].round(4).tolist(), y=e[:, 1].round(4).tolist(), z=e[:, 2].round(4).tolist(),
                          mask=r['object_mask'].astype(int).tolist(), source=t['source'].astype(int).tolist(),
                          phase=t['phase_id'].astype(int).tolist()))
    head = '<tr><th>statistic</th><th>real (min / median / max, n=%d)</th>%s</tr>' % (
        len(real), ''.join(f'<th>{name} (p10 / median / p90, n={n})</th>' for name, _ in groups))
    body = ''.join('<tr><td>%s</td>%s</tr>' % (lab, ''.join(f'<td>{c}</td>' for c in cells)) for lab, cells in table)
    html = (TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards))
            .replace('/*SOURCES*/', json.dumps(list(SOURCES))).replace('/*PHASES*/', json.dumps(list(PHASES)))
            .replace('<!--TABLE-->', f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, 'w').write(html)
    print(f'-> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')
    print(f'\n{"statistic":42s} | real min/med/max | ' + ' | '.join(name for name, _ in groups))
    for lab, cells in table:
        print(f'{lab:42s} | ' + ' | '.join(cells))
    return table


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Simulated vs real box scans</title>
<style>
:root{--bg:#fbfbfa;--fg:#1b1d21;--mut:#5b6068;--card:#fff;--line:#d9dbe0}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#a0a6af;--card:#1c1f24;--line:#33383f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}h2{font-size:1.1rem;margin:1.2em 0 .3em}
.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.card h3{font-size:1rem;margin:0}.card .n{font-size:.8rem;color:var(--mut)}.plot{height:440px;touch-action:none}
.views{margin:6px 0 0;display:flex;gap:6px;flex-wrap:wrap}.views button,.views select{font:inherit;font-size:.8rem;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
table{border-collapse:collapse;font-size:.85rem;width:100%}th,td{border:1px solid var(--line);padding:4px 8px;text-align:left}th{background:var(--card)}
.tw{overflow-x:auto}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Simulated vs real box scans</h1>
<p class="mut">Top row: the 5 real exports of <code>inputs_box/</code>, exactly as given (no truth exists). Bottom row: 4 simulated box sessions
(<code>majproj.sim_scan</code>), coloured by the simulator's per-point truth. Same axes for every plot: x, y horizontal, z = gravity up (the exported
files' own frame), range -1 to 1. The ceiling is the thin layer on top (an 8 x 8 zone-grid pattern), the object is the dense patch near the
sensor height, the rest is garbage. Drag to spin about the vertical axis, scroll to zoom, or use the buttons.</p>
<h2>Real files</h2><div class="grid" id="real"></div>
<h2>Simulated sessions <span class="mut" style="font-weight:400">colour:
<select id="mode"><option value="mask">object / not object</option><option value="source">source surface</option><option value="phase">session phase</option></select></span></h2>
<div class="grid" id="sim"></div>
<h2>Do they match? Same statistics on exported points</h2><div class="tw"><!--TABLE--></div>
<p class="mut">Reading the table: a statistic is reproduced when the real range (min to max of 5 files) lies inside the simulated p10 to p90 band.
Real files carry no truth, so the lower block exists only for the simulation.</p>
<script>
const cards = /*DATA*/, SOURCES = /*SOURCES*/, PHASES = /*PHASES*/;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e6e8eb' : '#1b1d21', grid = dark ? '#33383f' : '#d9dbe0';
const ax = t => ({title: t, range: [-1.05, 1.05], color: fg, gridcolor: grid, zerolinecolor: grid, backgroundcolor: 'rgba(0,0,0,0)'});
// Plot axes are the file's own: plot z = file z (up), so Plotly's turntable drag spins about gravity.
const VIEWS = {
  '3D': {eye: {x: 1.5, y: -1.7, z: 0.9}, up: {x: 0, y: 0, z: 1}},
  'Front': {eye: {x: 0, y: -2.3, z: 0.05}, up: {x: 0, y: 0, z: 1}},
  'Side': {eye: {x: 2.4, y: 0, z: 0.05}, up: {x: 0, y: 0, z: 1}},
  'Top': {eye: {x: 0, y: -0.01, z: 2.4}, up: {x: 0, y: 1, z: 0}},
};
const PAL = ['#e8590c','#1c7ed6','#2f9e44','#862e9c','#f08c00','#0b7285','#c2255c','#868e96','#5c940d','#a52a2a'];
const GREY = dark ? '#8a93a0' : '#6c7a89', HOT = '#e8590c';
function traces(c, mode) {
  const mk = (idx, name, color, size) => ({type: 'scatter3d', mode: 'markers', name, x: idx.map(i => c.x[i]), y: idx.map(i => c.y[i]), z: idx.map(i => c.z[i]),
    marker: {size, color, opacity: 0.85}, hovertemplate: name + '<br>x %{x:.3f} y %{y:.3f} z %{z:.3f}<extra></extra>'});
  const all = [...c.x.keys()];
  if (c.kind === 'real') return [mk(all, 'points', GREY, 2.2)];
  const groups = (labels, names, pal) => names.map((nm, k) => [nm, all.filter(i => labels[i] === k), pal[k % pal.length]]).filter(g => g[1].length);
  if (mode === 'mask') return [mk(all.filter(i => !c.mask[i]), 'not object', GREY, 2.2), mk(all.filter(i => c.mask[i]), 'object', HOT, 2.8)];
  const g = mode === 'source' ? groups(c.source, SOURCES, PAL) : groups(c.phase, PHASES, PAL);
  return g.map(([nm, idx, col]) => mk(idx, nm, col, 2.4));
}
const layout = {margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: 'rgba(0,0,0,0)', showlegend: true, legend: {font: {color: fg}, x: 0, y: 1},
  scene: {xaxis: ax('x'), yaxis: ax('y'), zaxis: ax('z (up)'), aspectmode: 'cube', dragmode: 'turntable', camera: VIEWS['3D']}};
const els = [];
for (const c of cards) {
  const d = document.createElement('div'); d.className = 'card';
  d.innerHTML = `<h3>${c.title}</h3><div class="n">${c.note}</div><div class="views">${Object.keys(VIEWS).map(v => `<button data-v="${v}">${v}</button>`).join('')}</div><div class="plot"></div>`;
  document.getElementById(c.kind === 'real' ? 'real' : 'sim').appendChild(d);
  const el = d.querySelector('.plot');
  Plotly.newPlot(el, traces(c, 'mask'), JSON.parse(JSON.stringify(layout)), {responsive: true, displaylogo: false});
  d.querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => Plotly.relayout(el, {'scene.camera': VIEWS[b.dataset.v]})));
  els.push([c, el]);
}
document.getElementById('mode').addEventListener('change', e => {
  for (const [c, el] of els) if (c.kind === 'sim') {
    const cam = el.layout.scene.camera;
    const l = JSON.parse(JSON.stringify(layout)); l.scene.camera = cam;
    Plotly.react(el, traces(c, e.target.value), l);
  }
});
</script></main></body></html>
"""

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40, help='simulated sessions per statistics group')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    main(a.n, a.seed, a.out, a.workers)
