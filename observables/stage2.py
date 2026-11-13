"""Stage 2 controlled spatial discrimination, nuisance robustness and source CIs."""
from pathlib import Path
import numpy as np
import pandas as pd
from .spatial import LAYOUTS,mosaic,representations,candidate_distances

CONDITIONS=('clean','cell shift','pixel shift','quarter turn','shift rotation noise')

def evaluate(sources_per_layout=24,seed=9021):
    gallery=[representations(mosaic(name)) for name in LAYOUTS]
    rows=[]
    rng=np.random.default_rng(seed)
    for label,name in enumerate(LAYOUTS):
        base=mosaic(name)
        for source in range(sources_per_layout):
            # Independent perturbation trajectories; clean originals are four fixed prototypes.
            dy,dx=map(int,rng.integers(1,8,2))
            turn=int(rng.integers(1,4))
            noisy=(-1 if rng.random()<.5 else 1)*np.rot90(np.roll(base,(dy,dx),(0,1)),turn)
            noise=rng.normal(size=base.shape)
            noisy+=.15*base.std()*noise
            views=[base,np.roll(base,(8*dy,8*dx),(0,1)),np.roll(base,(dy,dx),(0,1)),
                   np.rot90(base,turn),noisy]
            for condition,image in zip(CONDITIONS,views):
                for metric,distances in candidate_distances(image,gallery).items():
                    tied=np.isclose(distances,distances.min(),atol=1e-7,rtol=0)
                    credit=float(tied[label]/tied.sum())
                    margin=float(np.min(np.delete(distances,label))-distances[label])
                    rows.append((name,source,condition,metric,credit,margin))
    return pd.DataFrame(rows,columns=['layout','source','condition','metric','accuracy','margin'])

def summarize(frame,draws=2000):
    rows=[]
    n=frame.source.nunique()
    idx=(np.random.default_rng(57).integers(n,size=(draws,4,n))+np.arange(4)[None,:,None]*n).reshape(draws,-1)
    for (condition,metric),g in frame.groupby(['condition','metric']):
        values=g.sort_values(['layout','source']).accuracy.to_numpy()
        ci=np.quantile(values[idx].mean(1),[.025,.975])
        rows.append((condition,metric,values.mean(),*ci))
    return pd.DataFrame(rows,columns=['condition','metric','accuracy','low95','high95'])

def ensure():
    out=Path(__file__).resolve().parents[1]/'results'/'stage2'
    out.mkdir(exist_ok=True,parents=True)
    # Always recompute explicitly. No implicit reuse across changed package code.
    data=evaluate()
    data.to_csv(out/'retrieval_queries.csv',index=False)
    summary=summarize(data)
    summary.to_csv(out/'retrieval_summary.csv',index=False)
    return data,summary

if __name__=='__main__':
    _,summary=ensure()
    print(summary.pivot(index='metric',columns='condition',values='accuracy').round(3).to_string())
