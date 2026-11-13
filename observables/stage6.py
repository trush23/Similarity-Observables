"""Original MNIST retrieval using the finite-image similarity profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, rotate
from .open_images import ABLATIONS, COMPONENTS, Gallery, Observation, translate

SEED = 6028
SHA256 = '731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1'
CONDITIONS = ('clean','shift','gain','noise','blur','rotation','crop','occlusion','shift + noise')


def load_data(path=None):
    path = Path(path or os.environ.get('MNIST_NPZ',Path.home()/'.keras/datasets/mnist.npz'))
    with path.open('rb') as f:
        checksum = hashlib.file_digest(f,'sha256').hexdigest()
    if checksum!=SHA256:
        raise ValueError('Expected the original MNIST Keras NPZ with its published SHA-256')
    with np.load(path,allow_pickle=False) as d:
        return {key:d[key] for key in ['x_train','y_train','x_test','y_test']}


def galleries(data,seed=SEED,dev_blocks=6,test_blocks=16):
    rng = np.random.default_rng(seed)
    for split,source,blocks in [('development','train',dev_blocks),('test','test',test_blocks)]:
        x,y = data['x_'+source],data['y_'+source]
        pools = [rng.choice(np.flatnonzero(y==label),2*blocks,replace=False).reshape(blocks,2) for label in range(10)]
        for block in range(blocks):
            ids = rng.permutation(np.concatenate([p[block] for p in pools]))
            yield split,block,ids,x[ids].astype(float)/255,y[ids]


def view(image, condition, rng):
    obs = Observation.full(image)
    if condition in ('shift','shift + noise'):
        offset = tuple(rng.choice([-3,-2,2,3],size=2))
        obs = translate(obs,offset)
    if condition in ('noise','shift + noise'):
        obs.image = np.clip(obs.image+rng.normal(0,.23,image.shape),0,1)
    elif condition=='gain':
        obs.image = image*.55
    elif condition=='blur':
        obs.image = gaussian_filter(image,1.25,mode='constant')
    elif condition=='rotation':
        angle = rng.choice([-18.,18.])
        obs.image = rotate(image,angle,reshape=False,order=1,mode='constant')
        obs.valid = rotate(obs.valid,angle,reshape=False,order=0,mode='constant')
    elif condition=='crop':
        axis = int(rng.integers(2))
        cut = slice(0,6) if rng.random()<.5 else slice(-6,None)
        if axis==0:
            obs.valid[cut,:] = False
        else:
            obs.valid[:,cut] = False
    elif condition=='occlusion':
        y,x = rng.integers(7,16,size=2)
        obs.image = image.copy()
        obs.image[y:y+7,x:x+7] = 0.
    return Observation(np.where(obs.valid,obs.image,0.),obs.valid)


def rank_credit(distances, truth):
    best = np.isclose(distances,distances.min(),rtol=0,atol=1e-8)
    return float(best[truth]/best.sum())


def run(output,path=None,seed=SEED,dev_blocks=6,test_blocks=16):
    output = Path(output)
    output.mkdir(exist_ok=True,parents=True)
    data = load_data(path)
    rng = np.random.default_rng(seed+1)
    records, pair_rows, membership = [],[],[]
    for split,block,ids,images,labels in galleries(data,seed,dev_blocks,test_blocks):
        gallery = Gallery([Observation.full(x) for x in images])
        for position,(source,image,label) in enumerate(zip(ids,images,labels)):
            membership.append((split,block,int(source),int(label)))
            other = np.arange(len(ids))!=position
            class_truth = np.flatnonzero(labels[other]==label).item()
            for condition in CONDITIONS:
                query = view(image,condition,rng)
                distances,offsets,coverage = gallery.compare(query,ablations=True)
                for i,candidate in enumerate(ids):
                    pair_rows.append((split,block,int(source),int(candidate),condition,*distances[i],
                                      *offsets[0,i],*offsets[1,i],*coverage[:2,i]))
                for method,d in zip([*COMPONENTS,*ABLATIONS,'uniform profile'],
                                    [*distances.T,distances[:,:len(COMPONENTS)].mean(1)]):
                    records.append((split,block,int(source),int(label),condition,method,
                                    rank_credit(d,position),rank_credit(d[other],class_truth)))
    pairs = pd.DataFrame(pair_rows,columns=['split','gallery','source','candidate','condition',*COMPONENTS,*ABLATIONS,
                                          'pixel_dy','pixel_dx','Q_dy','Q_dx','pixel_coverage','Q_coverage'])
    decisions = pd.DataFrame(records,columns=['split','gallery','source','label','condition','method','identity','class'])
    sources = pd.DataFrame(membership,columns=['split','gallery','source','label'])
    selected = decisions.query("split=='development' and condition!='clean'").groupby('method')[['identity','class']].mean().sort_index().idxmax().to_dict()
    metadata = dict(seed=seed,sha256=SHA256,development_galleries=dev_blocks,test_galleries=test_blocks,
                    sources_per_gallery=20,radius=3,min_coverage=.65,development_selected=selected)
    pairs.to_csv(output/'pair_distances.csv.gz',index=False)
    decisions.to_csv(output/'decisions.csv.gz',index=False)
    sources.to_csv(output/'gallery_membership.csv',index=False)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    return decisions,pairs,metadata


def paired_interval(frame,left,right,endpoint,seed=6061):
    scores = frame.groupby(['gallery','method'])[endpoint].mean().unstack('method')
    delta = (scores[left]-scores[right]).to_numpy()
    draws = np.random.default_rng(seed).choice(delta,(4000,len(delta)),replace=True).mean(1)
    return delta.mean(),*np.quantile(draws,[.025,.975])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data')
    args = parser.parse_args()
    scores,_,manifest = run(Path(__file__).resolve().parents[1]/'results'/'stage6',args.data)
    print(manifest,flush=True)
    print(scores.query("split=='test' and condition!='clean'").groupby('method')[['identity','class']].mean())
