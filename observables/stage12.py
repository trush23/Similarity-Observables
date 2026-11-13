"""Frozen hierarchical retrieval on the official CIFAR-100 test split."""
import argparse
import hashlib
import json
import os
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .stage6 import CONDITIONS
from .stage8 import KEYS, rank
from .stage10 import cifar_views
from .stage11 import CommonOffset, ROOT, average_precision, encode

SEED = 12427
ARCHIVE_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'
RESULTS = ROOT/'results'/'stage12'
METHODS = ['magnitude', 'phase', 'aligned ZNCC', 'encoder 224']


def load_cifar100(path=None):
    path = Path(path or os.environ.get('CIFAR100_ARCHIVE',
                                      Path.home()/'.keras/datasets/cifar-100-python.tar.gz'))
    with path.open('rb') as file:
        digest = hashlib.file_digest(file, 'md5').hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError('Expected the official CIFAR-100 Python archive')
    # Load only the authenticated test and metadata members, without extracting files.
    with tarfile.open(path, 'r:gz') as archive:
        payload = archive.extractfile('cifar-100-python/test').read()
        meta = pickle.loads(archive.extractfile('cifar-100-python/meta').read(), encoding='bytes')
    test = pickle.loads(payload, encoding='bytes')
    x = test[b'data']
    fine, coarse = (np.asarray(test[key]) for key in (b'fine_labels', b'coarse_labels'))
    if (x.shape != (10000, 3072) or x.dtype != np.uint8
            or not np.array_equal(np.bincount(fine), [100]*100)
            or not np.array_equal(np.bincount(coarse), [500]*20)):
        raise ValueError('Unexpected CIFAR-100 test layout')
    mapping = np.array([np.unique(coarse[fine == k]).item() for k in range(100)])
    if not np.array_equal(np.bincount(mapping), [5]*20):
        raise ValueError('Unexpected fine-to-coarse hierarchy')
    names = [[s.decode('ascii') for s in meta[key]]
             for key in (b'fine_label_names', b'coarse_label_names')]
    return dict(x_test=x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1),
                fine=fine, coarse=coarse, mapping=mapping, fine_names=names[0], coarse_names=names[1],
                test_sha256=hashlib.sha256(payload).hexdigest())


def galleries(data):
    rng = np.random.default_rng(SEED)
    images, fine, mapping = (data[key] for key in ('x_test', 'fine', 'mapping'))
    fine_by_coarse = [rng.permutation(np.flatnonzero(mapping == c)) for c in range(20)]
    pools = [iter(rng.permutation(np.flatnonzero(fine == k))) for k in range(100)]
    used = set()
    for repeat in range(3):
        for slot, superclasses in enumerate(rng.permutation(20).reshape(4, 5)):
            ids = []
            for coarse in superclasses:
                for label in np.delete(fine_by_coarse[coarse], repeat):
                    for _ in range(2):
                        for idx in pools[label]:
                            key = images[idx].tobytes()
                            if key not in used:
                                used.add(key)
                                ids.append(idx)
                                break
                        else:
                            raise ValueError('Insufficient unique test images')
            ids = rng.permutation(ids)
            yield 'cifar100', 4*repeat+slot, ids, images[ids]/255., fine[ids]


def observations(images, ids, panel):
    return [cifar_views(image, condition, [SEED, int(source), k])[1]
            for image, source in zip(images, ids) for k, condition in enumerate(CONDITIONS)]


def classical(data, output):
    pairs, members = [], []
    for panel, group, ids, images, labels in galleries(data):
        print(group, 'shared offsets', flush=True)
        gallery = CommonOffset(images)
        queries = observations(images, ids, panel)
        members.extend((group, i, f, data['coarse'][i], data['fine_names'][f],
                        data['coarse_names'][data['coarse'][i]]) for i, f in zip(ids, labels))
        if group == 0:
            np.savez_compressed(output/'gallery_preview.npz', source=ids, images=data['x_test'][ids])
        for j, source in enumerate(ids):
            for k, condition in enumerate(CONDITIONS):
                scores = gallery.compare(queries[j*len(CONDITIONS)+k])
                pairs.extend((panel, group, source, condition, candidate, int(dy), int(dx), zncc, m, a)
                             for candidate, (dy, dx, zncc, m, a) in zip(ids, scores))
    pd.DataFrame(pairs, columns=KEYS+['dy', 'dx', 'aligned ZNCC', 'mag_sq', 'angular_sq']).to_csv(
        output/'classical_pairs.csv.gz', index=False)
    pd.DataFrame(members, columns=['gallery', 'source', 'fine', 'coarse', 'fine_name', 'coarse_name']).to_csv(
        output/'membership.csv', index=False)
    (output/'manifest.json').write_text(json.dumps(
        dict(dataset='CIFAR-100 official test', archive_md5=ARCHIVE_MD5, test_sha256=data['test_sha256'],
             seed=SEED, galleries=12, superclasses_per_gallery=5, fine_classes_per_superclass=4,
             images_per_fine_class=2, color='RGB', conditions=list(CONDITIONS),
             search_radius=3, min_overlap=.65), indent=2)+'\n')


def tasks(ids, source, fine, coarse):
    other = ids != source
    same_fine = fine[ids] == fine[source]
    same_coarse = coarse[ids] == coarse[source]
    fine_mask, coarse_mask = other & same_coarse, other & ~same_fine
    return [('fine', fine_mask, same_fine[fine_mask]), ('coarse', coarse_mask, same_coarse[coarse_mask])]


def pair_accuracy(scores, truth):
    """Probability that a random positive outranks a random negative, ties split."""
    differences = scores[~truth][None, :] - scores[truth][:, None]
    return np.mean((differences > 0) + .5*(differences == 0))


def summarize(data, output):
    pairs = pd.read_csv(output/'classical_pairs.csv.gz')
    frozen = pd.read_csv(output/'encoder_224.csv.gz')
    pd.testing.assert_frame_equal(pairs[KEYS], frozen[KEYS])
    pairs['encoder 224'] = frozen['encoder 224']
    pairs['magnitude'] = np.sqrt(pairs.mag_sq)
    pairs['phase'] = np.sqrt(pairs.mag_sq+pairs.angular_sq)
    records = []
    for (_, gallery, source, condition), group in pairs.groupby(KEYS[:-1], sort=False):
        ids = group.candidate.to_numpy()
        for task, mask, truth in tasks(ids, source, data['fine'], data['coarse']):
            assert (mask.sum(), truth.sum()) == ((7, 1) if task == 'fine' else (38, 6))
            scores = group.loc[mask, METHODS].to_numpy()
            hits = rank(scores, truth)
            records.extend((gallery, source, condition, task, method, top1,
                            average_precision(d, truth), pair_accuracy(d, truth))
                           for method, top1, d in zip(METHODS, hits, scores.T))
    metrics = ['top1', 'ap', 'pair_accuracy']
    d = pd.DataFrame(records, columns=['gallery', 'source', 'condition', 'task', 'method', *metrics])
    d['split'] = np.where(d.condition.eq('clean'), 'clean', 'transformed')
    d.to_csv(output/'decisions.csv.gz', index=False)
    d.groupby(['split', 'task', 'method'])[metrics].mean().to_csv(output/'summary.csv')
    d.groupby(['condition', 'task', 'method'])[metrics].mean().to_csv(output/'condition_summary.csv')
    means = d.groupby(['split', 'task', 'gallery', 'method'])[metrics].mean()
    rng = np.random.default_rng(SEED+2)
    contrasts = []
    for split in ('clean', 'transformed'):
        for metric in metrics:
            wide = means.loc[split, metric].unstack('method')
            delta = (wide.phase-wide.magnitude).unstack('task')
            effects = {'fine': delta.fine, 'coarse': delta.coarse}
            if metric == 'pair_accuracy':
                effects['coarse minus fine'] = delta.coarse-delta.fine
            draws = rng.integers(len(delta), size=(4000, len(delta)))
            for task, values in effects.items():
                low, high = np.quantile(values.to_numpy()[draws].mean(1), [.025, .975])
                contrasts.append((split, task, metric, values.mean(), low, high))
    pd.DataFrame(contrasts, columns=['split', 'task', 'metric', 'delta', 'low95', 'high95']).to_csv(
        output/'phase_contrasts.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['classical', 'encoder', 'summarize'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    data = load_cifar100(args.data)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.part == 'classical':
        classical(data, RESULTS)
    elif args.part == 'encoder':
        encode(data, RESULTS, 224, args.threads, args.device,
               generate=galleries, make_queries=observations, seed=SEED)
    else:
        summarize(data, RESULTS)
