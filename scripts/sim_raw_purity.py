"""Is what clean() accepts really the object? Purity = share of the cleaned points whose raw row is an object row in the
simulator's truth. A wrong accept has purity below 0.5. Acceptance alone cannot show this.

    uv run python -m scripts.sim_raw_purity
"""
import numpy as np,collections
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
from majproj.sim_raw import simulate_raw,KINDS
from majproj.clean import clean
from scripts.run_pipeline import thin
def job(j):
    k,g,s=j
    r=simulate_raw(np.random.default_rng([s,77]),k,regime=g)
    c=clean(thin(r['raw'])[0],origin=(0,0,0))
    if c.meta['status']!='ok' or len(c.object)==0: return k,g,None,None
    d,i=cKDTree(r['raw']).query(c.object.astype(float))
    p=float(r['object_mask'][i].mean())
    dc=float(np.median(cKDTree(r['complete']).query(c.object)[0]))
    return k,g,p,dc
if __name__=='__main__':
    jobs=[(k,g,s) for k in KINDS for g in ('real','robust') for s in range(16)]
    with ProcessPoolExecutor(10) as ex: res=list(ex.map(job,jobs,chunksize=2))
    for g in ('real','robust'):
        R=[x for x in res if x[1]==g]; A=[x for x in R if x[2] is not None]
        p=np.array([x[2] for x in A]); dc=np.array([x[3] for x in A])
        print(g,'n',len(R),'accepted',len(A),'purity med %.2f'%np.median(p),'wrong(<.5)',int((p<.5).sum()),'far(>5cm from complete)',int((dc>.05).sum()),'tol3cm fail',int((dc>.03).sum()))
        print('  ',{k:(sum(1 for x in A if x[0]==k),int(sum(x[2]<.5 for x in A if x[0]==k))) for k in KINDS})
