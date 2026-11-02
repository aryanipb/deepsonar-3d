"""Run majproj.clean on every scan in a folder; write cleaned .npy files, a JSON summary and a 3D page.

    uv run python -m scripts.clean_scans inputs_box outputs/clean_box
    -> outputs/clean_box/<scan>_object.npy, outputs/clean_box/summary.json, outputs/clean_box/index.html

Each card shows the cleaner's verdict (status, confidence, reason, failed tests) next to the points: the object in orange,
everything removed in grey (hidden by default; shown when nothing was kept so the card is never empty). z is up.
For accuracy numbers against known truth see `scripts/validate_clean.py`.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.clean import clean

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')


def _default(o):
    return o.item() if hasattr(o, 'item') else str(o)


def failed_tests(meta):
    return [f"{k}={v.get('value')}" for k, v in (meta.get('tests') or {}).items() if isinstance(v, dict) and not v.get('pass', True)]


def front_dir(meta):
    """Horizontal unit vector along the face normal (camera direction for the Front view); -y when there is no face."""
    n = meta.get('normal')
    if not n or np.hypot(n[0], n[1]) < 1e-6:
        return [0.0, -1.0]
    f = np.array(n[:2], float) / np.hypot(n[0], n[1])
    return [float(-f[0]), float(-f[1])] if f[1] > 0 else [float(f[0]), float(f[1])]


def main(folder, out):
    os.makedirs(out, exist_ok=True)
    rows, plots = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, folder, 'scan_*_shapenet.npy'))):
        name = os.path.basename(f)
        r = clean(np.load(f))
        m = r.meta
        np.save(os.path.join(out, name.replace('_shapenet.npy', '_object.npy')), r.object)
        rows.append(dict(name=name, **{k: v for k, v in m.items() if k != 'candidates'}, candidates=(m.get('candidates') or [])[:6]))
        plots.append(dict(name=name, status=m.get('status', 'ok' if len(r.object) else 'rejected'), conf=m.get('confidence'),
                          reason=m.get('reason') or '', failed=failed_tests(m), meta={k: v for k, v in m.items() if k not in ('candidates', 'tests')},
                          front=front_dir(m), obj=r.object.round(4).T.tolist(), rem=r.removed.round(4).T.tolist()))
        conf = m.get('confidence')
        print(f"{name}: {m.get('status', '?')} conf {'-' if conf is None else format(conf, '.2f')} kept {m.get('n_object', 0)} of {m.get('n_unique')} unique points; "
              f"width {m.get('width')} height {m.get('height')} w/h {m.get('aspect_w_over_h')} shape {m.get('shape_hint')} "
              f"normal {m.get('normal')} {m.get('reason') or ''}")
    json.dump(rows, open(os.path.join(out, 'summary.json'), 'w'), indent=1, default=_default)
    data = json.dumps(plots, default=_default).replace('</', '<\\/')
    html = TEMPLATE.replace('/*DATA*/', data).replace('/*PLOTLY*/', open(PLOTLY).read())
    open(os.path.join(out, 'index.html'), 'w').write(html)
    print('->', os.path.join(out, 'index.html'))


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cleaned scans</title><style>
:root{--bg:#fbfbfa;--fg:#1b1d21;--mut:#5b6068;--card:#fff;--line:#d9dbe0;--ok:#146c43;--rej:#8a5a00;--err:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e6e8eb;--mut:#a0a6af;--card:#1c1f24;--line:#33383f;--ok:#5fd39a;--rej:#f0b44c;--err:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:16px}h1{font-size:1.4rem;margin:.2em 0}p{max-width:110ch}
.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(440px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0}.card h2{font-size:1rem;margin:0 0 2px}
.hd{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.85rem}.badge{font-weight:700;font-size:.75rem;padding:1px 8px;border-radius:99px;border:1.5px solid currentColor}
.ok{color:var(--ok)}.rej{color:var(--rej)}.err{color:var(--err)}
.bar{display:inline-block;width:70px;height:8px;border:1px solid var(--line);border-radius:4px;vertical-align:middle;overflow:hidden}.bar i{display:block;height:100%;background:var(--fg);opacity:.55}
.plot{height:480px;touch-action:none}.views{margin:6px 0 0;display:flex;gap:6px;flex-wrap:wrap}
.views button{font:inherit;font-size:.8rem;padding:3px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:.85rem;margin:8px 0 14px}th,td{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
@media (max-width:480px){.grid{grid-template-columns:1fr}}
</style><script>/*PLOTLY*/</script></head><body><main>
<h1>Cleaned scans: object only</h1>
<p class="mut">Orange = the object found by <code>majproj.clean</code>. Grey = everything removed (ceiling, streaks, table, hand, strays), hidden by default
(click "removed" in the legend) and shown when nothing was kept. Up is up (z). Each plot is zoomed on the object with true proportions.
Sizes are in file units (not cm): each file has its own scale, so compare the width / height ratio, not the sizes.</p>
<div class="tw"><table id="tbl"><thead><tr><th>Scan</th><th>Status</th><th>Confidence</th><th>Kept</th><th>Removed</th><th>Shape</th><th>Width</th><th>Height</th><th>Width / height</th><th>Face normal (x, y, z)</th><th>Reason / failed tests</th></tr></thead><tbody></tbody></table></div>
<div class="grid" id="grid"></div></main>
<script>
const S = /*DATA*/;
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const fg = dark ? '#e6e8eb' : '#1b1d21', gridc = dark ? '#33383f' : '#d9dbe0';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const f2 = v => v === null || v === undefined ? '-' : (+v).toFixed(2);
const UP = {x: 0, y: 0, z: 1};
const views = f => { const R = 2.3, p = [-f[1], f[0]]; return {
  '3D': {eye: {x: 1.5, y: -1.7, z: 0.9}, up: UP}, 'Front': {eye: {x: R * f[0], y: R * f[1], z: 0.05}, up: UP},
  'Top': {eye: {x: 0.01 * f[0], y: 0.01 * f[1], z: 2.4}, up: {x: -f[0], y: -f[1], z: 0}}, 'Side': {eye: {x: R * p[0], y: R * p[1], z: 0.05}, up: UP}}; };
const ax = (t, r) => ({title: t, range: r, color: fg, gridcolor: gridc, zerolinecolor: gridc, backgroundcolor: 'rgba(0,0,0,0)'});
const cls = s => s === 'ok' ? 'ok' : s === 'rejected' ? 'rej' : 'err', txt = s => s === 'ok' ? 'ACCEPTED' : s === 'rejected' ? 'REJECTED' : 'ERROR';
const tb = document.querySelector('#tbl tbody');
function bbox(a, trim) {   // [lo, hi] per axis; 1-99 percentile when `trim` (used when only the removed points exist)
  return [0, 1, 2].map(k => { const v = Float64Array.from(a[k]).sort(); if (!v.length) return [0, 1];
    return trim ? [v[Math.floor(0.01 * (v.length - 1))], v[Math.ceil(0.99 * (v.length - 1))]] : [v[0], v[v.length - 1]]; });
}
for (const s of S) {
  const m = s.meta, V = views(s.front), has = s.obj[0].length > 0;
  tb.insertAdjacentHTML('beforeend', `<tr><td>${esc(s.name.slice(5, 20))}</td><td class="${cls(s.status)}"><b>${txt(s.status)}</b></td><td>${f2(s.conf)}</td><td>${m.n_object ?? 0}</td><td>${m.n_removed ?? m.n_unique}</td><td>${esc(m.shape_hint ?? '-')}</td><td>${m.width ?? '-'}</td><td>${m.height ?? '-'}</td><td><b>${m.aspect_w_over_h ?? '-'}</b></td><td>${(m.normal || []).join(', ')}</td><td>${esc(s.reason)} ${esc(s.failed.join(', '))}</td></tr>`);
  const c = document.createElement('div'); c.className = 'card';
  const conf = s.conf === null ? '' : `<span>confidence ${f2(s.conf)} <span class="bar"><i style="width:${Math.round(100 * Math.max(0, Math.min(1, s.conf)))}%"></i></span></span>`;
  c.innerHTML = `<h2>${esc(s.name)}</h2><div class="hd"><span class="badge ${cls(s.status)}">${txt(s.status)}</span>${conf}<span class="mut">kept ${m.n_object ?? 0} of ${m.n_unique} unique points${m.shape_hint ? ' · shape ' + esc(m.shape_hint) : ''}${m.aspect_w_over_h ? ' · w/h ' + m.aspect_w_over_h : ''}</span></div>
  <div class="mut" style="font-size:.78rem">${esc(s.reason)} ${s.failed.length ? 'failed tests: ' + esc(s.failed.join(', ')) : ''}</div>
  <div class="views">${Object.keys(V).map(v => `<button data-v="${v}">${v}</button>`).join('')}</div><div class="plot"></div>`;
  document.getElementById('grid').appendChild(c);
  const el = c.querySelector('.plot');
  const bb = bbox(has ? s.obj : s.rem, !has), half = 0.6 * Math.max(...bb.map(b => b[1] - b[0]), 1e-3);
  const rng = k => [(bb[k][0] + bb[k][1]) / 2 - half, (bb[k][0] + bb[k][1]) / 2 + half];
  const T = (n, a, col, sz, vis) => ({type: 'scatter3d', mode: 'markers', name: n, x: a[0], y: a[1], z: a[2], marker: {size: sz, color: col}, visible: vis, hoverinfo: 'skip'});
  let saved = null, live = false;     // draw a card only near the viewport (Chrome allows ~16 live WebGL contexts)
  const draw = () => { if (live) return; live = true;
    Plotly.newPlot(el, [T(`removed (${s.rem[0].length})`, s.rem, '#b0b0b0', 1.5, saved ? saved.vis[0] : (has ? 'legendonly' : true)), T(`object (${s.obj[0].length})`, s.obj, '#ea580c', 2.5, saved ? saved.vis[1] : true)],
      {margin: {l: 0, r: 0, t: 0, b: 0}, paper_bgcolor: 'rgba(0,0,0,0)', legend: {x: 0, y: 1, font: {color: fg, size: 11}},
       scene: {aspectmode: 'data', dragmode: 'turntable', xaxis: ax('x', rng(0)), yaxis: ax('y', rng(1)), zaxis: ax('z (up)', rng(2)), camera: saved ? saved.cam : V['3D']}},
      {responsive: true, displaylogo: false}); };
  const drop = () => { if (!live) return; live = false; saved = {cam: el._fullLayout.scene.camera, vis: el.data.map(t => t.visible)}; Plotly.purge(el); };
  c.querySelectorAll('[data-v]').forEach(b => b.addEventListener('click', () => { if (live) Plotly.relayout(el, {'scene.camera': V[b.dataset.v]}); else saved = {cam: V[b.dataset.v], vis: [has ? 'legendonly' : true, true]}; }));
  new IntersectionObserver(es => es.forEach(e => e.isIntersecting ? draw() : drop()), {rootMargin: '500px'}).observe(c);
}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
