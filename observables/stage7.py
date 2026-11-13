"""Audit rank redundancy and responses to known image interventions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .open_images import ABLATIONS, COMPONENTS, Observation, similarity_profile, translate
from .stage6 import CONDITIONS

METHODS = (*COMPONENTS,*ABLATIONS)
DRAWS = 4000
SEED = 7031


def load_panel(folder):
    folder = Path(folder)
    keys = ['gallery','source']
    pairs = pd.read_csv(folder/'pair_distances.csv.gz').query("split=='test'")
    membership = pd.read_csv(folder/'gallery_membership.csv').query("split=='test'")
    index = membership.set_index([*keys]).label
    pairs['source_label'] = index.loc[list(zip(pairs.gallery,pairs.source))].to_numpy()
    pairs['candidate_label'] = index.loc[list(zip(pairs.gallery,pairs.candidate))].to_numpy()
    data = {}
    query_keys = None
    for condition in CONDITIONS:
        rows = pairs[pairs.condition==condition].sort_values([*keys,'candidate'])
        if not rows.groupby(keys).size().eq(20).all() or rows.duplicated([*keys,'candidate']).any():
            raise ValueError('Expected one 20-candidate gallery per query')
        group_keys = rows[keys].to_numpy().reshape(-1,20,2)[:,0]
        candidates = rows.candidate.to_numpy().reshape(-1,20)
        if query_keys is None:
            query_keys,reference_candidates = group_keys,candidates
        if not np.array_equal(query_keys,group_keys) or not np.array_equal(reference_candidates,candidates):
            raise ValueError('Every intervention must retain the same query and candidate IDs')
        identity = (rows.source==rows.candidate).to_numpy().reshape(-1,20)
        same_class = (rows.source_label==rows.candidate_label).to_numpy().reshape(-1,20)
        if not identity.sum(1).tolist()==[1]*len(group_keys) or not np.all(same_class.sum(1)==2):
            raise ValueError('Expected one source and one distinct same-label image')
        values = rows[list(METHODS)].to_numpy().reshape(-1,20,len(METHODS))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite distances')
        data[condition] = values
    return data,query_keys,identity,same_class


def normalized_ranks(values):
    """Center ranks within each query and leave constant columns undefined."""
    ranks = rankdata(np.round(values,10),axis=1,method='average')
    centered = ranks-ranks.mean(1,keepdims=True)
    norms = np.linalg.norm(centered,axis=1,keepdims=True)
    return np.divide(centered,norms,out=np.full_like(centered,np.nan),where=norms>0)


def rank_redundancy(data,identity):
    rows = []
    for condition,values in data.items():
        alternatives = values[~identity].reshape(len(values),19,len(METHODS))
        ranks = normalized_ranks(alternatives)
        correlation = np.einsum('qcm,qcn->qmn',ranks,ranks)
        for i,left in enumerate(METHODS):
            for j,right in enumerate(METHODS):
                x = correlation[:,i,j]
                valid = np.isfinite(x)
                rows.append((condition,left,right,x[valid].mean() if valid.any() else np.nan,
                             int(valid.sum()),len(x)))
    return pd.DataFrame(rows,columns=['condition','left','right','mean_rho','defined_queries','queries'])


def endpoints(values,identity,same_class):
    """Distances from the designated target and from its nearest eligible impostor."""
    class_target = same_class&~identity
    target = {'identity':identity,'class':class_target}
    distractor = {'identity':~identity,'class':~same_class}
    for endpoint in target:
        positive = np.min(np.where(target[endpoint][...,None],values,np.inf),axis=1)
        negative = np.min(np.where(distractor[endpoint][...,None],values,np.inf),axis=1)
        best = np.minimum(positive,negative)
        eligible = target[endpoint]|distractor[endpoint]
        ties = eligible[...,None]&np.isclose(values,best[:,None,:],rtol=0,atol=1e-8)
        credit = (ties&target[endpoint][...,None]).sum(1)/ties.sum(1)
        yield endpoint,positive,negative-positive,credit


def gallery_bootstrap(values,groups,draws=DRAWS,seed=SEED):
    """Paired bootstrap of whole galleries, averaging within each sampled block."""
    blocks = np.stack([values[groups==group].mean(0) for group in np.unique(groups)])
    indices = np.random.default_rng(seed).integers(len(blocks),size=(draws,len(blocks)))
    draws = blocks[indices].mean(1)
    return blocks.mean(0),*np.quantile(draws,[.025,.975],axis=0)


def response_audit(data,query_keys,identity,same_class):
    rows = []
    clean = {name:(p,m) for name,p,m,_ in endpoints(data['clean'],identity,same_class)}
    for condition,values in data.items():
        if condition=='clean':
            continue
        for name,positive,margin,_ in endpoints(values,identity,same_class):
            for statistic,delta in [('target distance change',positive-clean[name][0]),
                                    ('retrieval margin change',margin-clean[name][1])]:
                avg,lo,hi = gallery_bootstrap(delta,query_keys[:,0])
                rows.extend((condition,name,statistic,method,avg[j],lo[j],hi[j])
                            for j,method in enumerate(METHODS))
    return pd.DataFrame(rows,columns=['condition','endpoint','statistic','method','change','low95','high95'])


def error_overlap(data,identity,same_class):
    rows = []
    for endpoint_index,endpoint in enumerate(('identity','class')):
        scores = np.concatenate([list(endpoints(x,identity,same_class))[endpoint_index][-1]
                                 for condition,x in data.items() if condition!='clean'])
        miss = scores==0
        tied = (scores>0)&(scores<1)
        for i,left in enumerate(METHODS):
            for j,right in enumerate(METHODS):
                common = np.sum(miss[:,i]&miss[:,j])
                union = np.sum(miss[:,i]|miss[:,j])
                rows.append((endpoint,left,right,int(miss[:,i].sum()),int(miss[:,j].sum()),
                             int(common),int(union),common/union if union else np.nan,
                             int(tied[:,i].sum()),int(tied[:,j].sum())))
    return pd.DataFrame(rows,columns=['endpoint','left','right','left_misses','right_misses','shared_misses',
                                      'miss_union','jaccard','left_partial_ties','right_partial_ties'])


def uniform_ablation(data,query_keys,identity,same_class):
    names = ['full uniform',*('omit '+method for method in COMPONENTS)]
    rows = []
    credits = {'identity':[], 'class':[]}
    for condition,x in data.items():
        if condition=='clean':
            continue
        full = x[:,:,:len(COMPONENTS)]
        total = full.sum(2)
        models = np.concatenate([(total/len(COMPONENTS))[...,None],
                                 (total[...,None]-full)/(len(COMPONENTS)-1)],axis=2)
        for endpoint,_,_,score in endpoints(models,identity,same_class):
            credits[endpoint].append(score)
    for endpoint,views in credits.items():
        values = np.mean(views,axis=0)
        diff,lo,hi = gallery_bootstrap(values-values[:,:1],query_keys[:,0])
        rows.extend((endpoint,name,values[:,i].mean(),diff[i],lo[i],hi[i])
                    for i,name in enumerate(names))
    return pd.DataFrame(rows,columns=['endpoint','variant','top1','change','low95','high95'])


def synthetic_controls():
    from .spatial import mosaic
    y,x = np.indices((64,64))
    ramp = .14+.43*x/63+.16*y/63
    compact = np.zeros_like(ramp)
    compact[20:43,23:40] = ramp[20:43,23:40]
    full = Observation.full
    cases = {
        'add brightness offset': (full(ramp),full(ramp+.1)),
        'gain 0.55': (full(ramp),full(.55*ramp)),
        'reverse contrast': (full(ramp),full(1-ramp)),
        'shift compact pattern': (full(compact),translate(full(compact),(3,-2))),
        'rotate gradient axis 90°': (full(ramp),full(np.rot90(ramp))),
        'rearrange identical motifs': (full(mosaic('domains')),full(mosaic('checker'))),
    }
    rows = []
    for name,(left,right) in cases.items():
        scores,_,_ = similarity_profile(right,left)
        rows.extend((name,method,1-agreement) for method,agreement in scores.items())
    return pd.DataFrame(rows,columns=['intervention','method','distance'])


def run(source,output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    data,keys,identity,same_class = load_panel(source)
    results = {
        'rank_redundancy':rank_redundancy(data,identity),
        'error_overlap':error_overlap(data,identity,same_class),
        'uniform_ablation':uniform_ablation(data,keys,identity,same_class),
        'intervention_responses':response_audit(data,keys,identity,same_class),
        'synthetic_controls':synthetic_controls(),
    }
    for name,table in results.items():
        table.to_csv(output/(name+'.csv'),index=False)
    hashes = {}
    for file in ['pair_distances.csv.gz','gallery_membership.csv','manifest.json']:
        with (Path(source)/file).open('rb') as stream:
            hashes[file] = hashlib.file_digest(stream,'sha256').hexdigest()
    (output/'manifest.json').write_text(json.dumps(dict(stage6_sha256=hashes,
        bootstrap_draws=DRAWS,bootstrap_seed=SEED,galleries=int(np.unique(keys[:,0]).size),
        sources=len(keys),methods=METHODS,conditions=CONDITIONS),indent=2))
    return results


if __name__=='__main__':
    root = Path(__file__).resolve().parents[1]/'results'
    results = run(root/'stage6',root/'stage7')
    for name,table in results.items():
        print(name,len(table))
