"""Matched baselines and source-disjoint MNIST replication."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from .core import relative_distance
from .open_images import (ABLATIONS, COMPONENTS, Gallery, Observation, finite_gradient,
                          overlap_slices, window_distance)
from .stage6 import CONDITIONS, SEED, SHA256, galleries, load_data, view

REPLICATION_SEED = 8319
REPLICATION_GALLERIES = 12
VIEWS = CONDITIONS[1:]
BASELINES = ('ZNCC', 'aligned ZNCC', 'standard HOG', 'HOG after ZNCC',
             'structure tensor', 'aligned structure tensor')
FROZEN = ('ResNet-50 cosine', 'ResNet-50 local mean', 'ResNet-50 local Q',
          'EfficientNet-B0 cosine', 'EfficientNet-B0 local mean', 'EfficientNet-B0 local Q')
PRIMARY = ('aligned local Q','local Q','aligned pixels','aligned Q magnitude','aligned gradient strength')
KEYS = ['panel','gallery','source','condition','candidate']


def replication_galleries(data, excluded, count=REPLICATION_GALLERIES, *, seed=REPLICATION_SEED, panel="replication"):
    rng = np.random.default_rng(seed)
    labels = data['y_test']
    images = data['x_test']
    used = set(excluded)
    used_images = {images[i].tobytes() for i in used}
    pools = []
    for digit in range(10):
        pool = []
        for idx in rng.permutation(np.flatnonzero(labels==digit)):
            token = images[idx].tobytes()
            if idx in used or token in used_images:
                continue
            pool.append(idx)
            used_images.add(token)
            if len(pool)==4*count:
                break
        if len(pool)!=4*count:
            raise ValueError('Insufficient disjoint digit or garment sources')
        pools.append(np.array(pool).reshape(count,4))
    for group in range(count):
        ids = rng.permutation(np.concatenate([p[group] for p in pools]))
        yield panel,group,ids,images[ids].astype(float)/255,labels[ids]


def observations(data, excluded):
    # Replay the development draws first to reproduce the Stage 6 test perturbations.
    rng = np.random.default_rng(SEED+1)
    for split,group,ids,images,labels in galleries(data):
        queries = [view(x,k,rng) for x in images for k in CONDITIONS]
        if split=='test':
            yield 'pilot',group,ids,images,labels,queries
    rng = np.random.default_rng(REPLICATION_SEED+1)
    for panel,group,ids,images,labels in replication_galleries(data,excluded):
        queries = [view(x,k,rng) for x in images for k in CONDITIONS]
        yield panel,group,ids,images,labels,queries


def zncc_distance(a,b,mask):
    """Half of 1 minus the correlation after centering on shared support."""
    count = np.maximum(mask.sum((-2,-1),keepdims=True),1)
    a0 = np.where(mask,a,0.)
    b0 = np.where(mask,b,0.)
    a0 = np.where(mask,a0-a0.sum((-2,-1),keepdims=True)/count,0.)
    b0 = np.where(mask,b0-b0.sum((-2,-1),keepdims=True)/count,0.)
    dot = (a0*b0).sum((-2,-1))
    norm = np.sqrt((a0*a0).sum((-2,-1))*(b0*b0).sum((-2,-1)))
    corr = np.divide(dot,norm,out=np.zeros_like(dot),where=norm>1e-12)
    corr = np.where((norm<=1e-12)&np.all(a0==b0,axis=(-2,-1)),1.,corr)
    return (1-np.clip(corr,-1,1))/2


def structure_field(obs):
    g = finite_gradient(np.where(obs.valid,obs.image,0.),obs.valid)
    mass = uniform_filter(np.abs(g)**2,5,mode='constant')
    support = uniform_filter(obs.valid.astype(float),5,mode='constant')
    square = g*g
    numerator = uniform_filter(square.real,5,mode='constant')+1j*uniform_filter(square.imag,5,mode='constant')
    denominator = mass+.05**2*support
    return np.divide(numerator,denominator,out=np.zeros_like(numerator),where=(denominator>1e-12)&obs.valid)


def standard_hog(image):
    from skimage.feature import hog
    return hog(image,orientations=9,pixels_per_cell=(7,7),cells_per_block=(2,2),
               block_norm='L2-Hys',feature_vector=True)


class Classical:
    def __init__(self,images):
        self.images = images
        self.tensors = np.stack([structure_field(Observation.full(x)) for x in images])
        self.hog = [standard_hog(x) for x in images]
        self.windows = [overlap_slices(images.shape[1:],[(0,0),(dy,dx)])
                        for dy in range(-3,4) for dx in range(-3,4)]

    def compare(self,query):
        n = len(self.images)
        mask = query.valid
        pixels = query.image
        tensor = structure_field(query)
        fixed_ncc = zncc_distance(pixels,self.images,mask)
        fixed_tensor = window_distance(tensor,self.tensors,mask)
        query_hog = standard_hog(pixels)
        fixed_hog = [relative_distance(query_hog,h) for h in self.hog]
        minimum_ncc = np.full(n,np.inf)
        minimum_tensor = np.full(n,np.inf)
        chosen = np.full(n,-1)
        for i,(a,b) in enumerate(self.windows):
            shared = mask[a]
            if shared.sum()<.65*mask.size:
                continue
            d = zncc_distance(pixels[a],self.images[:,b[0],b[1]],shared)
            better = d<minimum_ncc
            minimum_ncc[better] = d[better]
            chosen[better] = i
            d = window_distance(tensor[a],self.tensors[:,b[0],b[1]],shared)
            minimum_tensor = np.minimum(minimum_tensor,d)
        registered_hog = np.ones(n)
        # One descriptor for each chosen overlap on the query, shared by candidates at that offset.
        for i in np.unique(chosen[chosen>=0]):
            a,b = self.windows[i]
            left = np.zeros_like(pixels)
            left[a] = np.where(mask[a],pixels[a],0.)
            feature = standard_hog(left)
            for j in np.flatnonzero(chosen==i):
                right = np.zeros_like(pixels)
                right[a] = np.where(mask[a],self.images[j][b],0.)
                registered_hog[j] = relative_distance(feature,standard_hog(right))
        return np.column_stack([fixed_ncc,np.minimum(minimum_ncc,1.),fixed_hog,registered_hog,
                                fixed_tensor,np.minimum(minimum_tensor,1.)])


def run_classical(data,excluded,output, *, generate=observations):
    rows,groups = [],[]
    times = {'classical seconds':0.,'profile seconds':0.}
    for panel,group,ids,images,labels,queries in generate(data,excluded):
        print(panel,group,'classical',flush=True)
        classic = Classical(images)
        reference = Gallery([Observation.full(x) for x in images]) if panel!='pilot' else None
        for j,(source,label) in enumerate(zip(ids,labels)):
            groups.append((panel,group,source,label))
            for k,condition in enumerate(CONDITIONS):
                if condition=='clean':
                    continue
                q = queries[j*len(CONDITIONS)+k]
                start = perf_counter()
                extra = classic.compare(q)
                times['classical seconds'] += perf_counter()-start
                if reference is not None:
                    start = perf_counter()
                    values,_,_ = reference.compare(q,ablations=True)
                    values = np.column_stack([values,values[:,:len(COMPONENTS)].mean(1)])
                    times['profile seconds'] += perf_counter()-start
                    extra = np.column_stack([extra,values])
                rows.extend((panel,group,source,condition,c,*d) for c,d in zip(ids,extra))
    names = [*BASELINES,*COMPONENTS,*ABLATIONS,'uniform profile']
    pd.DataFrame(rows,columns=[*KEYS,*names]).to_csv(output/'classical_pairs.csv.gz',index=False)
    pd.DataFrame(groups,columns=['panel','gallery','source','label']).to_csv(output/'membership.csv',index=False)
    (output/'classical_timing.json').write_text(json.dumps(times,indent=2))


def frozen_extract(model,architecture,images,mean,std):
    import torch
    from torch.nn import functional as F
    global_vectors,local_vectors,local_q = [],[],[]
    with torch.inference_mode():
        for first in range(0,len(images),64):
            batch = images[first:first+64]
            batch = batch[:,None] if batch.ndim==3 else batch.transpose(0,3,1,2)
            x = torch.from_numpy(batch.astype('float32'))
            x = F.interpolate(x,size=(56,56),mode='bilinear',align_corners=False)
            x = ((x-mean)/std).contiguous(memory_format=torch.channels_last)
            if architecture=='ResNet-50':
                x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
                local = model.layer2(model.layer1(x))
                final = model.layer4(model.layer3(local))
            else:
                local = model.features[:4](x)
                final = model.features[4:](local)
            global_vectors.append(F.normalize(final.mean((2,3)),dim=1).numpy())
            local = F.normalize(local.flatten(2).transpose(1,2),dim=2)
            local_vectors.append(F.normalize(local.mean(1),dim=1).numpy())
            dim = local.shape[-1]
            count = (local.square().sum(2)>0).sum(1).clamp_min(1)[:,None,None]
            q = local.transpose(1,2)@local/count
            trace = q.diagonal(dim1=-2,dim2=-1).sum(1)
            q = q - trace[:,None,None]*torch.eye(dim)/dim
            local_q.append(F.normalize(q.flatten(1),dim=1).numpy())
    return [np.concatenate(rows) for rows in (global_vectors,local_vectors,local_q)]


def run_frozen(data,excluded,output, *, generate=observations):
    import torch
    import torchvision
    from torchvision.models import (ResNet50_Weights, EfficientNet_B0_Weights,
                                    resnet50, efficientnet_b0)
    torch.set_num_threads(4)
    configs = [('ResNet-50',resnet50,ResNet50_Weights.IMAGENET1K_V1),
               ('EfficientNet-B0',efficientnet_b0,EfficientNet_B0_Weights.IMAGENET1K_V1)]
    models,metadata = [],{}
    for name,build,weights in configs:
        model = build(weights=weights).eval().to(memory_format=torch.channels_last)
        norm = weights.transforms()
        mean,std = (torch.tensor(v)[None,:,None,None] for v in (norm.mean,norm.std))
        models.append((name,model,mean,std))
        cache = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1]
        with cache.open('rb') as f:
            metadata[name] = {'weights':weights.url,'sha256':hashlib.file_digest(f,'sha256').hexdigest(),'seconds':0.}
    rows = []
    for panel,group,ids,images,labels,queries in generate(data,excluded):
        print(panel,group,'frozen',flush=True)
        array = np.array([q.image for q in queries])
        matrices = []
        for name,model,mean,std in models:
            start = perf_counter()
            fields = frozen_extract(model,name,array,mean,std)
            for field in fields:
                matrices.append((1-np.clip(field@field[::len(CONDITIONS)].T,-1,1))/2)
            metadata[name]['seconds'] += perf_counter()-start
        matrices = np.array(matrices)
        for j,source in enumerate(ids):
            for k,condition in enumerate(CONDITIONS):
                if condition=='clean':
                    continue
                scores = matrices[:,j*len(CONDITIONS)+k,:].T
                rows.extend((panel,group,source,condition,c,*d) for c,d in zip(ids,scores))
    pd.DataFrame(rows,columns=[*KEYS,*FROZEN]).to_csv(output/'frozen_pairs.csv.gz',index=False)
    (output/'frozen_manifest.json').write_text(json.dumps(dict(models=metadata,torch=torch.__version__,
                                                             torchvision=torchvision.__version__,input_size=56),indent=2))


def rank(scores,relevant):
    best = np.isclose(scores,scores.min(0),rtol=0,atol=1e-8)
    return (best&relevant[:,None]).sum(0)/best.sum(0)


def summarize(source,output,data, *, shortlist_seed=REPLICATION_SEED):
    classic = pd.read_csv(output/'classical_pairs.csv.gz')
    frozen = pd.read_csv(output/'frozen_pairs.csv.gz')
    if len(classic)!=len(frozen):
        raise ValueError('Baseline runs have different query counts')
    merged = classic.merge(frozen,on=KEYS,validate='one_to_one')
    if len(merged)!=len(classic):
        raise ValueError('Baseline runs do not have identical pair keys')
    from_stage6 = [*COMPONENTS,*ABLATIONS]
    if source is not None:
        pilot = pd.read_csv(source/'pair_distances.csv.gz').query("split=='test' and condition!='clean'").copy()
        pilot['panel']='pilot'
        pilot['uniform profile']=pilot[list(COMPONENTS)].mean(1)
        new = merged[merged.panel=='pilot'].drop(columns=[*from_stage6,'uniform profile']).merge(
            pilot[[*KEYS,*from_stage6,'uniform profile']],on=KEYS,validate='one_to_one')
        if len(new)!=len(pilot):
            raise ValueError('Pilot does not match Stage 6 candidates')
        merged = pd.concat([new,merged[merged.panel=='replication']],ignore_index=True)
    keep = [name for name in COMPONENTS if name!='open Q triplets']
    merged['uniform minus triplets'] = merged[keep].mean(1)
    methods = [*from_stage6,'uniform profile','uniform minus triplets',*BASELINES,*FROZEN]
    membership = pd.read_csv(output/'membership.csv')
    y = data['y_test']
    # Clean 7×7 average-pool descriptors are used only to construct challenge panels.
    thumb = data['x_test'].astype('float32').reshape(-1,7,4,7,4).mean((2,4)).reshape(-1,49)/255
    rows = []
    for key,g in merged.groupby(KEYS[:-1]):
        panel,block,query,condition = key
        candidates = g.candidate.to_numpy()
        values = g[methods].to_numpy()
        identity = candidates==query
        same = (y[candidates]==y[query])&~identity
        wrong = (y[candidates]!=y[query])
        protocols = [('full',~identity)]
        if len(candidates)==40:
            wrong_ix = np.flatnonzero(wrong)
            order = np.argsort(np.linalg.norm(thumb[candidates[wrong_ix]]-thumb[query],axis=1),kind='stable')
            hard = same.copy()
            hard[wrong_ix[order[:16]]] = True
            random = same.copy()
            rng = np.random.default_rng([shortlist_seed,block,query])
            random[rng.choice(wrong_ix,16,replace=False)] = True
            protocols.extend([('hard 19',hard),('random 19',random)])
        for name,mask in protocols:
            identity_result = rank(values,identity) if name=='full' else np.full(len(methods),np.nan)
            class_result = rank(values[mask],same[mask])
            rows.extend((*key,name,method,a,b) for method,a,b in zip(methods,identity_result,class_result))
    decisions = pd.DataFrame(rows,columns=[*KEYS[:-1],'gallery_type','method','identity','class'])
    assert not set(membership.query("panel=='pilot'").source)&set(membership.query("panel=='replication'").source)
    decisions.to_csv(output/'decisions.csv.gz',index=False)
    summary = decisions.groupby(['panel','gallery_type','method'])[['identity','class']].mean().reset_index()
    summary.to_csv(output/'summary.csv',index=False)
    return decisions,summary


def contrast(frame,left,right,endpoint,seed=8401):
    mean = frame.groupby(['gallery','method'])[endpoint].mean().unstack('method')
    delta = (mean[left]-mean[right]).to_numpy()
    draws = np.random.default_rng(seed).choice(delta,(4000,len(delta)),replace=True).mean(1)
    return delta.mean(),*np.quantile(draws,[.025,.975])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['classical','frozen','summarize'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'results'
    output = root/'stage8'
    output.mkdir(exist_ok=True)
    excluded = set(pd.read_csv(root/'stage6'/'gallery_membership.csv').query("split=='test'").source)
    data = load_data()
    if args.part=='classical':
        run_classical(data,excluded,output)
    elif args.part=='frozen':
        run_frozen(data,excluded,output)
    else:
        _,summary = summarize(root/'stage6',output,data)
        print(summary.round(4).to_string(index=False))
