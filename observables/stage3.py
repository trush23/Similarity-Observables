"""Controlled tests of information lost by pair statistics."""
from pathlib import Path
import numpy as np
import pandas as pd
from .core import relative_distance, periodic_correlation
from .patterns import homometric
from .images import tiled_orientation
from .spatial import tiles

FAMILIES = ('windowed wave', 'ridge', 'bar')
CONDITIONS = {'clean': 0., 'tile sign': 0., 'noise 0.35': .35, 'noise 1.0': 1., 'noise 2.0': 2.}
METHODS = ('Q pair', 'Q triple', 'aligned Q', 'aligned pixels', 'aligned sign-free patches')


def tile_library(family, size=8):
    if family == FAMILIES[0]:
        return tiles(size)
    coords = np.linspace(-1, 1, size)
    y, x = np.meshgrid(coords, coords, indexing='ij')
    window = np.outer(np.hanning(size), np.hanning(size))
    if family == 'ridge':
        profile = np.exp(-x*x/.08)
    elif family == 'bar':
        profile = (np.abs(x) < .35).astype(float)
    else:
        raise ValueError(family)
    motif = profile * window
    # Match per-tile norm across motif families to control the noise scale.
    motif *= np.linalg.norm(tiles(size)[0]) / np.linalg.norm(motif)
    return np.stack([motif, motif.T])


def render(labels, library):
    """One row of square image tiles, each drawn from a two-orientation library."""
    return np.concatenate(library[np.asarray(labels, dtype=int)], axis=1)


def tile_q(image):
    """Real nematic component in the known tile frame; exclude cross-tile edges."""
    size = image.shape[0]
    patches = image.reshape(size, -1, size).transpose(1, 0, 2)
    return np.array([tiled_orientation(patch, 1)[0, 1].real for patch in patches])


def cyclic_statistics(sequence):
    x = np.asarray(sequence)
    shifted = x[(np.arange(len(x))[:, None] + np.arange(len(x))[None, :]) % len(x)]
    return x @ shifted / len(x), (shifted.T * x) @ shifted / len(x)


def open_statistics(sequence):
    """Overlap-normalized moments of a finite consecutive window."""
    x = np.asarray(sequence)
    n = len(x)
    pair = np.array([np.mean(x[:n-r]*x[r:]) for r in range(n)])
    triple = np.array([[np.mean(x[:n-max(r,s)]*x[r:n-max(r,s)+r]*x[s:n-max(r,s)+s])
                        for s in range(n)] for r in range(n)])
    return pair, triple


def matrix_profile(field):
    return np.array([periodic_correlation(field, lag) for lag in range(len(field))])


def orbit(sequence):
    return np.array([np.roll(sequence, i, axis=0) for i in range(len(sequence))])


def crop_census(lengths=(4,6,8,10,12)):
    """Enumerate all cuts. Class ties receive half credit; lags use only observed sites."""
    rows = []
    for length in lengths:
        windows = [orbit(x)[:, :length] for x in homometric()]
        signature = [[open_statistics(w) for w in group] for group in windows]
        for label, group in enumerate(windows):
            for origin, window in enumerate(group):
                q = signature[label][origin]
                checks = {
                    'open pair': any(np.array_equal(q[0], v[0]) for v in signature[1-label]),
                    'open triple': any(np.array_equal(q[1], v[1]) for v in signature[1-label]),
                    'exact window': any(np.array_equal(window, w) for w in windows[1-label]),
                }
                for method, collision in checks.items():
                    rows.append((length, label, origin, method, collision))
    return pd.DataFrame(rows, columns=['length','label','cut_index','method','collision'])


def patch_array(image):
    size = image.shape[0]
    return image.reshape(size, -1, size).transpose(1, 0, 2).reshape(-1, size*size)


def aligned_distances(query, candidates, sign_free=False):
    # candidates have shape (classes, shifts, tiles, features).
    if sign_free:
        inner = np.sum(candidates * query, axis=-1)
        square = np.sum(candidates**2 + query**2, axis=(-1,-2)) - 2*np.sum(np.abs(inner), axis=-1)
        return np.sqrt(np.maximum(square.min(axis=1), 0))
    return np.linalg.norm((candidates-query).reshape(*candidates.shape[:2],-1), axis=-1).min(axis=1)


def query_images(replicates=48, seed=1507):
    """Yield the same paired synthetic views to later experiments."""
    rng = np.random.default_rng(seed)
    for label, prototype in enumerate(homometric()):
        for trajectory in range(replicates):
            labels = np.roll(prototype, rng.integers(len(prototype)))
            signs = rng.choice([-1,1], size=len(labels))
            noise = rng.normal(size=(8, 8*len(labels)))
            for family in FAMILIES:
                base = render(labels, tile_library(family))
                for condition, noise_ratio in CONDITIONS.items():
                    image = base.copy()
                    if condition != 'clean':
                        image *= np.repeat(signs, base.shape[0])
                    if noise_ratio:
                        image += noise_ratio*np.sqrt(np.mean(base**2))*noise
                    yield (label, trajectory, family, condition), image


def evaluate(replicates=48, seed=1507):
    wave = tile_library(FAMILIES[0])
    gallery_images = [render(x, wave) for x in homometric()]
    gallery_q = [tile_q(image) for image in gallery_images]
    gallery_stats = [cyclic_statistics(q) for q in gallery_q]
    q_orbits = np.array([orbit(q[:,None]) for q in gallery_q])
    image_orbits = np.array([orbit(patch_array(im)) for im in gallery_images])
    rows = []
    for key, image in query_images(replicates, seed):
        q = tile_q(image)
        pair,triple = cyclic_statistics(q)
        raw = patch_array(image)
        scores = [np.array([relative_distance(pair,s[0]) for s in gallery_stats]),
                  np.array([relative_distance(triple,s[1]) for s in gallery_stats]),
                  aligned_distances(q[:,None],q_orbits),
                  aligned_distances(raw,image_orbits),
                  aligned_distances(raw,image_orbits,sign_free=True)]
        label = key[0]
        for method, distance in zip(METHODS,scores):
            tie = np.isclose(distance,distance.min(),atol=1e-9,rtol=0)
            credit = tie[label]/tie.sum()
            rows.append((*key,method,credit))
    return pd.DataFrame(rows,columns=['label','trajectory','motif','condition','method','top1'])


def summarize(frame, draws=4000, seed=420):
    n = frame.trajectory.nunique()
    rng = np.random.default_rng(seed)
    index = (rng.integers(n,size=(draws,2,n))+np.arange(2)[None,:,None]*n).reshape(draws,-1)
    rows = []
    for key, group in frame.groupby(['motif','condition','method']):
        values = group.sort_values(['label','trajectory']).top1.to_numpy()
        low,high = np.quantile(values[index].mean(1),[.025,.975])
        rows.append((*key,values.mean(),low,high))
    return pd.DataFrame(rows,columns=['motif','condition','method','top1','low95','high95'])


def run(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    census = crop_census()
    data = evaluate()
    summary = summarize(data)
    for name,frame in [('crop_census',census),('retrieval_queries',data),('retrieval_summary',summary)]:
        frame.to_csv(output/(name+'.csv'),index=False)
    return census,data,summary


if __name__ == '__main__':
    _,_,summary = run(Path(__file__).resolve().parents[1]/'results'/'stage3')
    print(summary.pivot(index=['motif','condition'],columns='method',values='top1').round(3))
