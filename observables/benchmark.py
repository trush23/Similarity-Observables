"""Fixed prototype retrieval with known labels, held seeds and explicit invariances."""
from pathlib import Path
import numpy as np
import pandas as pd
from .patterns import stripe
from .images import orientation, radial_power
from .core import power, relative_distance

AXES = ((1,0),(0,1),(1,1),(1,-1))
SCALES = (4,8)

def features(a):
    angular, _ = orientation(a)
    return {'pixels':a, 'Q':angular[1:2], 'fourth':angular[2:],
            'power':power(a), 'radial power':radial_power(a)}

def evaluate(replicates=30, seed=701):
    """Each base source has four noisy copies sharing phase and polarity."""
    meta=[(axis,scale) for axis in range(4) for scale in SCALES]
    gallery=[features(stripe(wave=tuple(scale*v for v in AXES[axis]))) for axis,scale in meta]
    rng=np.random.default_rng(seed)
    rows=[]
    for axis,scale in meta:
        for replicate in range(replicates):
            phase=rng.uniform(0,2*np.pi)
            polarity=rng.choice([-1,1])
            clean=polarity*stripe(wave=tuple(scale*v for v in AXES[axis]),phase=phase)
            source=f'{axis}-{scale}-{replicate}'
            for sigma in (0.,.1,.3,.6):
                query=features(clean+rng.normal(0,sigma,clean.shape))
                for metric,a in query.items():
                    distances=np.array([relative_distance(a,g[metric]) for g in gallery])
                    ties=np.isclose(distances,distances.min(),atol=1e-8,rtol=0)
                    credit=ties/ties.sum()
                    axis_hit=sum(p for p,(other,_) in zip(credit,meta) if other==axis)
                    scale_hit=sum(p for p,(_,other) in zip(credit,meta) if other==scale)
                    exact=sum(p for p,m in zip(credit,meta) if m==(axis,scale))
                    rows.append((source,axis,scale,replicate,sigma,metric,exact,axis_hit,scale_hit))
    return pd.DataFrame(rows,columns=['source','axis','scale','replicate','noise_sd','metric','joint','axis_hit','scale_hit'])

def summarize(results, draws=2000):
    rows=[]
    for (metric,sigma),g in results.groupby(['metric','noise_sd']):
        values=g.sort_values(['axis','scale','replicate'])[['joint','axis_hit','scale_hit']].to_numpy()
        per_cell=len(values)//8
        # Balance the eight prespecified prototype types in every resample.
        idx=(np.random.default_rng(55).integers(per_cell,size=(draws,8,per_cell))
             + np.arange(8)[None,:,None]*per_cell).reshape(draws,-1)
        ci=np.quantile(values[idx].mean(1),[.025,.975],axis=0)
        for j,endpoint in enumerate(['joint','axis_hit','scale_hit']):
            rows.append((metric,sigma,endpoint,values[:,j].mean(),ci[0,j],ci[1,j]))
    return pd.DataFrame(rows,columns=['metric','noise_sd','endpoint','estimate','low95','high95'])

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results'
    root.mkdir(exist_ok=True)
    data=evaluate()
    data.to_csv(root/'retrieval_queries.csv',index=False)
    summarize(data).to_csv(root/'retrieval_summary.csv',index=False)
