"""Interpretable similarity profiles and family-disjoint synthetic evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from .core import power, relative_distance
from .spatial import orbit_distance, patches
from .stage3 import tile_library

SPLITS = {
    'train': ('independent', 'compact'),
    'development': ('diagonal',),
    'test': ('elongated', 'checker-biased', 'separable'),
}
CONDITIONS = {
    'wave': ('windowed wave', 0., False),
    'ridge': ('ridge', 0., False),
    'bar': ('bar', 0., False),
    'mixed noise 1': ('ridge', 1., False),
    'mixed noise 2': ('bar', 2., False),
    'occlusion': ('bar', .35, True),
}
COMPONENTS = ('orientation histogram', 'orientation pairs', 'orientation triplets',
              'aligned orientation', 'intensity power', 'aligned sign-free pixels')
LAGS = ((0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(2,2),(2,-2))
GRID = 8
POSITIONS = np.indices((GRID,GRID)).reshape(2,-1)
SHIFTS = (POSITIONS[:,None,:] + POSITIONS[:,:,None]) % GRID
SHIFT_INDEX = SHIFTS[0]*GRID + SHIFTS[1]


def arrangement(family, rng):
    y,x = np.indices((GRID,GRID))
    white = rng.normal(size=(GRID,GRID))
    if family == 'independent':
        field = white
    elif family == 'compact':
        field = gaussian_filter(white,1.2,mode='wrap')
    elif family == 'diagonal':
        field = np.cos(2*np.pi*(x+y)/GRID+rng.uniform(0,2*np.pi))+.4*white
    elif family == 'elongated':
        field = gaussian_filter(white,(.65,2.2),mode='wrap')
        if rng.random()<.5:
            field = field.T
    elif family == 'checker-biased':
        field = .85*(-1.)**(x+y)+white
    elif family == 'separable':
        row = rng.permutation(np.repeat([0,1],GRID//2))
        col = rng.permutation(np.repeat([0,1],GRID//2))
        return np.bitwise_xor(row[:,None],col[None,:])
    else:
        raise ValueError(family)
    bits = np.zeros(GRID*GRID,dtype=int)
    bits[np.argsort(field.ravel())[-len(bits)//2:]] = 1
    return bits.reshape(GRID,GRID)


def orbit_key(bits):
    return min(row.tobytes() for row in bits.ravel()[SHIFT_INDEX].astype(np.uint8))


def gallery_groups(seed=5109, groups_per_family=32):
    rng = np.random.default_rng(seed)
    used = set()
    for split,families in SPLITS.items():
        for family in families:
            for group in range(groups_per_family):
                for _ in range(1000):
                    source = arrangement(family,rng)
                    raw = source.ravel()
                    variants = [source]
                    for swaps in (2,6,12):
                        altered = raw.copy()
                        zero = rng.choice(np.flatnonzero(raw==0),swaps,replace=False)
                        one = rng.choice(np.flatnonzero(raw==1),swaps,replace=False)
                        altered[np.r_[zero,one]] ^= 1
                        variants.append(altered.reshape(source.shape))
                    keys = {orbit_key(v) for v in variants}
                    if len(keys)==4 and keys.isdisjoint(used):
                        used.update(keys)
                        break
                else:
                    raise RuntimeError('Could not construct a new disjoint gallery')
                order = rng.permutation(4)
                yield split,family,group,np.stack(variants)[order],int(np.flatnonzero(order==0)[0])


def tile_view(bits, condition, rng):
    family,noise,occlusion = CONDITIONS[condition]
    dy,dx = rng.integers(GRID,size=2)
    labels = np.roll(bits,(dy,dx),(0,1))
    clean = tile_library(family)[labels]
    signs = rng.choice([-1,1],size=labels.shape)
    observed = clean*signs[...,None,None] + noise*np.sqrt(np.mean(clean**2))*rng.normal(size=clean.shape)
    if occlusion:
        y,x = rng.integers(GRID,size=2)
        observed[np.ix_((y+np.arange(3))%GRID,(x+np.arange(3))%GRID)] = 0.
    return observed.transpose(0,2,1,3).reshape(GRID*8,GRID*8)


def signatures(image):
    tiles = patches(image).reshape(GRID,GRID,8,8)
    gx = (np.roll(tiles,-1,axis=-1)-np.roll(tiles,1,axis=-1))/2
    gy = (np.roll(tiles,-1,axis=-2)-np.roll(tiles,1,axis=-2))/2
    length = np.hypot(gx,gy)
    mass = length.sum((-2,-1))
    axial = np.divide(gx*gx-gy*gy,length,out=np.zeros_like(gx),where=length>1e-12)
    field = np.divide(axial.sum((-2,-1)),mass,out=np.zeros_like(mass),where=mass>1e-12)
    shifted = np.array([np.roll(field,lag,(0,1)) for lag in LAGS])
    triplets = np.einsum('ij,kij,lij->kl',field,shifted,shifted)/field.size
    return (np.histogram(field,bins=np.linspace(-1,1,17))[0]/field.size,
            np.fft.ifft2(np.abs(np.fft.fft2(field))**2).real/field.size,
            triplets,field,power(image))


def image_distances(image_a, image_b):
    """Symmetric descriptive distances for equal 64 × 64 grayscale arrays."""
    a,b = (np.asarray(x,dtype=float) for x in (image_a,image_b))
    if any(x.shape != (GRID*8,GRID*8) or not np.isfinite(x).all() for x in (a,b)):
        raise ValueError('Expected two finite 64 × 64 grayscale arrays')
    left,right = signatures(a),signatures(b)
    output = [relative_distance(left[j],right[j]) for j in (0,1,2)]
    output += [orbit_distance(left[3],right[3]),relative_distance(left[4],right[4])]
    x,y = patches(a),patches(b)
    norms = np.linalg.norm(x),np.linalg.norm(y)
    absolute_dot = np.abs(x@y.T)
    best = absolute_dot[np.arange(GRID**2),SHIFT_INDEX].sum(1).max()
    residual = max(0.,norms[0]**2+norms[1]**2-2*best)
    output.append(np.sqrt(residual)/sum(norms) if sum(norms)>1e-12 else 0.)
    return np.array(output)


def similarity_profile(image_a, image_b):
    """A pair description on a common scale, separate from fitted decisions."""
    return pd.Series(1-image_distances(image_a,image_b),index=COMPONENTS,name='agreement')


def weighted_similarity(profile, weights):
    """Expose every term in the scalar score for a supplied similarity profile."""
    values = profile.reindex(COMPONENTS)
    weights = np.asarray(weights,dtype=float)
    if weights.shape != (len(COMPONENTS),) or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):
        raise ValueError('Weights must be a nonnegative simplex vector')
    if not np.isfinite(values).all() or np.any(values < -1e-9) or np.any(values > 1+1e-9):
        raise ValueError('Expected all named similarity coordinates in [0,1]')
    return pd.DataFrame({'agreement':values,'weight':weights,'contribution':values*weights})


def make_images(seed=5109, groups_per_family=32):
    perturb_rng = np.random.default_rng(seed+1)
    library = tile_library('windowed wave')
    for split,family,group,gallery,truth in gallery_groups(seed,groups_per_family):
        references = library[gallery].transpose(0,1,3,2,4).reshape(4,GRID*8,GRID*8)
        for condition in CONDITIONS:
            query = tile_view(gallery[truth],condition,perturb_rng)
            yield (split,family,group,condition),query,references,truth


def generate(seed=5109, groups_per_family=32):
    rows = []
    for key,query,references,truth in make_images(seed,groups_per_family):
        for candidate,reference in enumerate(references):
            rows.append((*key,candidate,candidate==truth,*image_distances(query,reference)))
    return pd.DataFrame(rows,columns=['split','family','group','condition','candidate','match',*COMPONENTS])


def fit_weights(train, temperature=.05, penalty=.02):
    if set(train.split) != {'train'}:
        raise ValueError('Fusion fitting requires training families only')
    differences = []
    for _,group in train.groupby(['family','group','condition']):
        positive = group.loc[group.match,list(COMPONENTS)].to_numpy()
        negatives = group.loc[~group.match,list(COMPONENTS)].to_numpy()
        differences.append(negatives-positive)
    delta = np.vstack(differences)
    start = np.full(len(COMPONENTS),1/len(COMPONENTS))
    def loss(w):
        return np.logaddexp(0,-delta@w/temperature).mean() + penalty*np.sum((w-start)**2)
    result = minimize(loss,start,method='SLSQP',bounds=[(0,1)]*len(start),
                      constraints={'type':'eq','fun':lambda w:np.sum(w)-1},
                      options={'ftol':1e-12,'maxiter':200})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def training_scales(train):
    if set(train.split) != {'train'}:
        raise ValueError('Scaling requires training families only')
    spans = train.groupby(['family','group','condition'])[list(COMPONENTS)].agg(np.ptp)
    return spans.median().to_numpy()


def calibrated_distances(values, scales):
    """Map distances into comparable ranges using the frozen training scales."""
    return np.divide(values, values+scales, out=np.zeros_like(values), where=scales>1e-12)


def evaluate(frame, weights, scales, calibrated_weights):
    rows = []
    for key,group in frame.groupby(['split','family','group','condition']):
        distance = group[list(COMPONENTS)].to_numpy()
        calibrated = calibrated_distances(distance,scales)
        for method,d in zip([*COMPONENTS,'uniform profile','trained profile','scaled trained profile'],
                            [*distance.T,distance.mean(1),distance@weights,calibrated@calibrated_weights]):
            best = np.isclose(d,d.min(),atol=1e-7,rtol=0)
            accuracy = float(best[group.match.to_numpy()].sum()/best.sum())
            rows.append((*key,method,accuracy))
    return pd.DataFrame(rows,columns=['split','family','group','condition','method','top1'])


def paired_interval(frame, left, right, seed=5110, draws=4000):
    values = frame.groupby(['family','group','method']).top1.mean().unstack('method')
    delta = values[left]-values[right]
    rng = np.random.default_rng(seed)
    means = [rng.choice(v.to_numpy(),(draws,len(v)),replace=True).mean(1)
             for _,v in delta.groupby('family')]
    lo,hi = np.quantile(np.mean(means,axis=0),[.025,.975])
    return float(delta.mean()),float(lo),float(hi)


def explain(group, weights, scales=None):
    """Per-candidate contributions to the score and to the winner–runner-up margin."""
    values = group[list(COMPONENTS)].to_numpy()
    if scales is not None:
        values = calibrated_distances(values,scales)
    order = np.argsort(values@weights,kind='stable')
    first,second = order[:2]
    contributions = weights*(values[second]-values[first])
    result = pd.DataFrame({'component':COMPONENTS,'weight':weights,
                           'winner_distance':values[first],'runner_up_distance':values[second],
                           'margin_contribution':contributions})
    return int(group.iloc[first].candidate),int(group.iloc[second].candidate),result


def run(output, seed=5109, groups_per_family=32):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    pairs = generate(seed,groups_per_family)
    train = pairs[pairs.split=='train'].copy()
    weights = fit_weights(train)
    scales = training_scales(train)
    train[list(COMPONENTS)] = calibrated_distances(train[list(COMPONENTS)].to_numpy(),scales)
    calibrated_weights = fit_weights(train)
    decisions = evaluate(pairs,weights,scales,calibrated_weights)
    development = decisions[(decisions.split=='development')&decisions.method.isin(COMPONENTS)]
    selected = development.groupby('method').top1.mean().sort_index().idxmax()
    info = dict(seed=seed,groups_per_family=groups_per_family,splits=SPLITS,
                selected_single=selected,temperature=.05,penalty=.02,
                weights=dict(zip(COMPONENTS,map(float,weights))),
                training_scales=dict(zip(COMPONENTS,map(float,scales))),
                scaled_weights=dict(zip(COMPONENTS,map(float,calibrated_weights))))
    pairs.to_csv(output/'profiles.csv',index=False)
    decisions.to_csv(output/'decisions.csv',index=False)
    (output/'fit.json').write_text(json.dumps(info,indent=2))
    return pairs,decisions,info


if __name__=='__main__':
    _,decisions,info=run(Path(__file__).resolve().parents[1]/'results'/'stage5')
    print('Development-selected single:',info['selected_single'])
    print(decisions[decisions.split=='test'].groupby(['family','method']).top1.mean().unstack().round(3))
