"""Simulated raw files (majproj.sim_raw) against the real sessions: table + side-by-side Plotly page.

    uv run python -m scripts.sim_raw_vs_real [--n 40] [--out outputs/final/sim_vs_real.html]

File statistics use only the rows, identically for real and simulated files, on the longest export of each of the 20
sessions. The clean() columns run scripts.run_pipeline.thin + majproj.clean(origin=0): `size` is width x height of the
cleaned patch. Simulated groups: "box*" / "bottle*" are robust-regime objects forced to the real sizes (19 x 12 x 23 cm,
7 x 25 cm); the rest is each kind in the 'real' regime (majproj.sim_real) and the 'robust' regime (simulate_robust).
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from majproj.clean import clean
from majproj.raw_frames import frames
from majproj.sim_raw import KINDS, simulate_raw, simulate_robust
from scripts.calib_sim_raw import SESSIONS, path
from scripts.run_pipeline import thin

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')
Z_BINS = [-9, -0.08, 0.2, 1.5, 2.3, 9]
R_BINS = [0, 0.15, 0.6, 1.5, 2.3, 9]


def stats(a):
    r = np.linalg.norm(a, axis=1)
    fs = frames(a)
    L = np.array([f[1] for f in fs])
    C = np.array([f[2] for f in fs])
    d = dict(rows=len(a), uniq=len(np.unique(a, axis=0)) / len(a), frames=len(fs), L_med=float(np.median(L)), one_copy=float((C == 1).mean()),
             ceil_z=float(np.median(a[a[:, 2] > 2.0, 2])) if (a[:, 2] > 2.0).any() else np.nan, ext_xy=float(np.ptp(a[:, :2], 0).mean()))
    for i, v in enumerate(np.histogram(a[:, 2], Z_BINS)[0] / len(a)):
        d[f'z{i}'] = float(v)
    for i, v in enumerate(np.histogram(r, R_BINS)[0] / len(a)):
        d[f'r{i}'] = float(v)
    return d


def cleaned(a):
    p, metric = thin(a)
    c = clean(p, origin=(0.0, 0.0, 0.0)) if metric else clean(p)
    m = c.meta
    return dict(ok=m['status'] == 'ok', n_obj=m.get('n_object', 0), w=m.get('width'), h=m.get('height'), rng=m.get('range_m'), reason=(m.get('reason') or '')[:40])


def real_job(j):
    kind, stem = j
    a = np.load(path(stem, kind))
    return dict(kind=kind, stem=stem, **stats(a), **cleaned(a))


def _sim(rng, kind, kw):
    """kw with size_m forces the real sizes and needs the robust simulator; 'regime' picks one otherwise."""
    return simulate_robust(rng, kind=kind, **kw) if 'size_m' in kw else simulate_raw(rng, kind, **kw)


def sim_job(j):
    name, kind, seed, kw = j
    r = _sim(np.random.default_rng([seed, 11]), kind, kw)
    T = r['truth']
    return dict(group=name, kind=kind, size=T['size_m'], dist=T['distance_m'], **stats(r['raw']), **cleaned(r['raw']))


def summ(rows, keys):
    out = {}
    for k in keys:
        v = np.array([x[k] for x in rows if x.get(k) is not None], float)
        v = v[np.isfinite(v)]
        out[k] = (np.percentile(v, 25), np.median(v), np.percentile(v, 75)) if len(v) else (np.nan,) * 3
    return out


KEYS = ['rows', 'uniq', 'frames', 'L_med', 'one_copy', 'ceil_z', 'ext_xy', 'z0', 'z1', 'z2', 'z3', 'z4', 'r0', 'r1', 'r2', 'r3', 'r4']


def table(groups):
    head = ['group', 'n'] + KEYS + ['accept', 'w_cm', 'h_cm', 'n_obj', 'range_m']
    lines = [head]
    for name, rows in groups.items():
        s = summ(rows, KEYS + ['n_obj', 'rng'])
        ok = [x for x in rows if x['ok']]
        w = summ(ok, ['w', 'h']) if ok else dict(w=(np.nan,) * 3, h=(np.nan,) * 3)
        fm = lambda t, f='{:.3g}': f.format(t[1])
        lines.append([name, str(len(rows))] + [fm(s[k]) for k in KEYS] +
                     [f'{len(ok)}/{len(rows)}', f'{100 * w["w"][1]:.1f}', f'{100 * w["h"][1]:.1f}', fm(s['n_obj']), fm(s['rng'])])
    return lines


def page(real_sets, sim_sets, lines, out):
    with open(PLOTLY) as f:
        js = f.read()
    cards = []
    for title, a, c in real_sets + sim_sets:
        idx = np.random.default_rng(0).choice(len(a), min(len(a), 7000), replace=False)
        cards.append(dict(title=title, x=a[idx, 0].round(3).tolist(), y=a[idx, 1].round(3).tolist(), z=a[idx, 2].round(3).tolist(), c=(None if c is None else c[idx].tolist())))
    n_real = len(real_sets)
    tb = ''.join('<tr>' + ''.join(f'<{"th" if i == 0 else "td"}>{v}</{"th" if i == 0 else "td"}>' for v in l) + '</tr>' for i, l in enumerate(lines))
    html = f'''<!doctype html><meta charset=utf-8><title>Simulated vs real raw</title>
<style>:root{{--bg:#fff;--fg:#111;--mut:#555;--line:#ddd}}@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#eee;--mut:#aaa;--line:#333}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px system-ui}}h1{{font-size:20px}}h2{{font-size:15px;margin:20px 0 6px}}
.t{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}}.c{{border:1px solid var(--line);border-radius:6px;padding:6px}}.p{{height:320px}}.c b{{font-weight:600;font-size:12px}}</style>
<h1>Simulated raw files against the real sessions</h1>
<p>Rows are medians per file. z0-z4: fraction of rows with z below -0.08, up to 0.2, up to 1.5, up to 2.3, above. r0-r4: range below 0.15, 0.6, 1.5, 2.3, above. In the plots below, orange = rows the simulator knows are object, grey = the rest; real files have no truth.</p>
<div class=t><table>{tb}</table></div><h2>Real files</h2><div class=g id=a></div><h2>Simulated files</h2><div class=g id=b></div>
<script>{js}</script><script>
const C={json.dumps(cards)},N={n_real};
function add(k,host){{const d=document.createElement('div');d.className='c';d.innerHTML='<b>'+C[k].title+'</b><div class=p></div>';host.appendChild(d);const e=d.lastChild;let on=false;
new IntersectionObserver(es=>{{const v=es[0].isIntersecting;if(v===on)return;on=v;if(!v){{Plotly.purge(e);return}}
const c=C[k];Plotly.newPlot(e,[{{type:'scatter3d',mode:'markers',x:c.x,y:c.y,z:c.z,marker:{{size:1.4,color:c.c?c.c.map(v=>v?'#e8710a':'#8a8a8a'):'#3b7dd8'}}}}],{{margin:{{l:0,r:0,t:0,b:0}},paper_bgcolor:'rgba(0,0,0,0)',scene:{{aspectmode:'data',xaxis:{{title:''}},yaxis:{{title:''}},zaxis:{{title:''}}}}}},{{displayModeBar:false}})}},{{rootMargin:'150px'}}).observe(d)}}
C.forEach((_,k)=>add(k,document.getElementById(k<N?'a':'b')));
</script>'''
    with open(out, 'w') as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--per-kind', type=int, default=20)
    ap.add_argument('--out', default=os.path.join(ROOT, 'outputs', 'final', 'sim_vs_real.html'))
    ap.add_argument('--no-page', action='store_true')
    args = ap.parse_args()
    jobs = [(k, s) for k, v in SESSIONS.items() for s in v]
    sj = []
    for i in range(args.n):
        sj.append(('box*', 'cuboid', i, dict(size_m=(0.19, 0.12, 0.23))))
        sj.append(('bottle*', 'bottle', 1000 + i, dict(size_m=(0.07, 0.25))))
    for k in KINDS:
        sj.extend((f'{k} real', k, 2000 + i, dict(regime='real')) for i in range(args.per_kind))
        sj.extend((f'{k} robust', k, 3000 + i, dict(regime='robust')) for i in range(args.per_kind))
    with ProcessPoolExecutor(11) as ex:
        real = list(ex.map(real_job, jobs))
        sim = list(ex.map(sim_job, sj, chunksize=2))
    groups = {'REAL box': [r for r in real if r['kind'] == 'box'], 'REAL bottle': [r for r in real if r['kind'] == 'bottle']}
    for name in ['box*', 'bottle*'] + [f'{k} {g}' for k in KINDS for g in ('real', 'robust')]:
        groups[name] = [r for r in sim if r['group'] == name]
    lines = table(groups)
    print('| ' + ' | '.join(lines[0]) + ' |')
    for l in lines[1:]:
        print('| ' + ' | '.join(l) + ' |')
    for name in groups:
        if name in ('REAL box', 'REAL bottle', 'box*', 'bottle*'):
            c = {}
            for x in groups[name]:
                if not x['ok']:
                    c[x['reason']] = c.get(x['reason'], 0) + 1
            print(name, 'rejects', c)
    json.dump(dict(lines=lines), open(os.path.join(ROOT, 'outputs', 'final', 'sim_vs_real.json'), 'w'))
    if not args.no_page:
        real_sets = [(f'REAL {k} {s}', np.load(path(s, k)), None) for k, s in (('box', '120408'), ('box', '121223'), ('box', '115330'), ('bottle', '123523'), ('bottle', '123815'), ('bottle', '122907'))]
        sim_sets = []
        for name, kind, kw, seeds in (('box real', 'cuboid', dict(regime='real'), (1, 2)), ('bottle real', 'bottle', dict(regime='real'), (3, 4)),
                                      ('box*', 'cuboid', dict(size_m=(0.19, 0.12, 0.23)), (5,)), ('bottle*', 'bottle', dict(size_m=(0.07, 0.25)), (6,)),
                                      ('wavy robust', 'wavy_bottle', dict(regime='robust'), (7,)), ('pipe robust', 'pipe', dict(regime='robust'), (8,)), ('slab robust', 'slab', dict(regime='robust'), (9,))):
            for s in seeds:
                r = _sim(np.random.default_rng([s, 7]), kind, kw)
                sim_sets.append((f'SIM {name or kind} seed {s}', r['raw'], r['object_mask']))
        page(real_sets, sim_sets, lines, args.out)
        print(args.out)


if __name__ == '__main__':
    main()
