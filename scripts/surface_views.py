"""Run the model on every real file, build the 3D surface, save OBJ files and one page.

    uv run python -m scripts.surface_views checkpoints/adapointr_final.pth outputs/surfaces
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from majproj.clean import clean  # noqa: E402
from majproj.infer import load_checkpoint  # noqa: E402
from majproj.surface import is_closed, to_mesh, write_obj  # noqa: E402
from scripts.four_views import complete  # noqa: E402
from scripts.gen_final import metric_thin  # noqa: E402

PLOTLY = os.path.join(ROOT, 'web', 'public', 'vendor', 'plotly.min.js')


def main(ckpt, out):
    model, info = load_checkpoint(ckpt)
    os.makedirs(out, exist_ok=True)
    cards, rows = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, 'FINAL-INPUTS', '*', '*.npy'))):
        ob, name = f.split(os.sep)[-2], os.path.basename(f)[:-4]
        r = clean(metric_thin(np.load(f))[0], origin=(0.0, 0.0, 0.0))
        pred = np.asarray(complete(model, r.object))
        v, t = to_mesh(pred)
        write_obj(os.path.join(out, f'{ob.lower()}_{name[-6:]}.obj'), v, t)
        d = cKDTree(pred).query(v)[0]
        row = dict(file=name, object=ob, closed=is_closed(t), triangles=len(t), mesh_to_points_median_mm=round(1000 * float(np.median(d)), 2),
                   size_cm=[round(float(x) * 100, 1) for x in (v.max(0) - v.min(0))])
        rows.append(row)
        print(row, flush=True)
        keep = t
        cards.append(dict(title=f'{ob.lower()} {name[-6:]}', row=row, v=v.round(4).T.tolist(), f=keep.T.tolist()))
    json.dump(rows, open(os.path.join(out, 'surfaces.json'), 'w'), indent=1)
    html = TEMPLATE.replace('/*PLOTLY*/', open(PLOTLY).read()).replace('/*DATA*/', json.dumps(cards)).replace('/*STEP*/', str(info.get('step')))
    open(os.path.join(out, 'surfaces.html'), 'w').write(html)
    print('closed', sum(r['closed'] for r in rows), 'of', len(rows), '->', out)


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3D Surfaces</title>
<style>body{margin:0;font:14px/1.45 system-ui,sans-serif;background:#fbfbfa;color:#1b1d21}main{max-width:1500px;margin:0 auto;padding:14px}h1{font-size:1.3rem}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:10px}.c{background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}.p{height:340px;touch-action:none}.m{font-size:.8rem;color:#555}</style><script>/*PLOTLY*/</script></head>
<body><main><h1>3D surfaces from the model's output, step /*STEP*/</h1><p class="m">Geometry only: the surface wraps the predicted points. Drag to spin.</p><div class="g" id="g"></div></main><script>
const S=/*DATA*/;for(const s of S){const d=document.createElement('div');d.className='c';d.innerHTML=`<b>${s.title}</b><div class="m">${s.row.size_cm.join(' x ')} cm · closed: ${s.row.closed}</div><div class="p"></div>`;g.appendChild(d);const e=d.querySelector('.p');
new IntersectionObserver((x,o)=>{if(x[0].isIntersecting){o.disconnect();Plotly.newPlot(e,[{type:'mesh3d',x:s.v[0],y:s.v[1],z:s.v[2],i:s.f[0],j:s.f[1],k:s.f[2],color:s.title.startsWith('box')?'#2563eb':'#dc2626',flatshading:false,lighting:{ambient:.5,diffuse:.8,specular:.3}}],{margin:{l:0,r:0,t:0,b:0},scene:{aspectmode:'data',xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false}}},{displaylogo:false,responsive:true})}}).observe(e)}
</script></body></html>
"""

if __name__ == '__main__':
    main(*sys.argv[1:])
