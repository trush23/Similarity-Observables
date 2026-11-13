"""CIFAR-10 common-offset scorecards and held-out encoder fusion."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .open_images import Observation, overlap_slices
from .stage6 import CONDITIONS
from .stage8 import KEYS, rank, replication_galleries, zncc_distance
from .stage10 import SEED as PREVIOUS_SEED
from .stage10 import LUMA, cifar_views, load_cifar, local_fields

SEED = 11427
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results'/'stage11'
PRIOR = ROOT/'results'/'stage10'/'cifar10'
FIELD_METHODS = ['magnitude', 'phase']
FUSION_METHODS = ['encoder 56', 'encoder 224', *FIELD_METHODS, 'encoder 224 + phase']


def galleries(data, panel):
    old = pd.read_csv(PRIOR/'membership.csv').query("panel == 'gray'")
    if panel == 'fresh':
        yield from replication_galleries(data, set(old.source), seed=SEED, panel=panel)
    else:
        for gallery, rows in old.groupby('gallery'):
            ids = rows.source.to_numpy()
            yield panel, gallery, ids, data['x_test'][ids]/255., data['y_test'][ids]


def observations(images, ids, panel):
    seed = SEED if panel == 'fresh' else PREVIOUS_SEED
    channel = 1 if panel == 'fresh' else 0
    return [cifar_views(image, condition, [seed, int(source), k])[channel]
            for image, source in zip(images, ids) for k, condition in enumerate(CONDITIONS)]


def encode(data, output, size, threads, device, *, generate=None, make_queries=observations,
           seed=SEED):
    import torch
    import torchvision
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import ResNet50_Weights, resnet50

    torch.set_num_threads(threads)
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights).eval().to(device=device, memory_format=torch.channels_last)
    model.fc = nn.Identity()
    transform = weights.transforms()
    mean, std = (torch.tensor(v, device=device)[None, :, None, None] for v in
                 (transform.mean, transform.std))
    cache = Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/', 1)[-1]
    with cache.open('rb') as file:
        checksum = hashlib.file_digest(file, 'sha256').hexdigest()
    expected = json.loads((PRIOR/'frozen_manifest.json').read_text())['models']['ResNet-50']['sha256']
    if checksum != expected:
        raise ValueError('Checkpoint differs from Stage 10')
    records = []
    panels = galleries(data, 'fresh') if generate is None else generate(data)
    with torch.inference_mode():
        for panel, group, ids, images, _ in panels:
            start = perf_counter()
            queries = make_queries(images, ids, panel)
            array = np.stack([q.image for q in queries]).astype('float32').transpose(0, 3, 1, 2)
            vectors = []
            for batch in np.array_split(array, int(np.ceil(len(array)/32))):
                x = torch.from_numpy(batch).to(device)
                if size == 224:
                    x = transform(x)
                else:
                    x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
                    x = (x-mean)/std
                vectors.append(F.normalize(model(x.contiguous(memory_format=torch.channels_last)), dim=1).cpu().numpy())
            vectors = np.concatenate(vectors)
            scores = (1 - np.clip(vectors @ vectors[::len(CONDITIONS)].T, -1, 1))/2
            for j, source in enumerate(ids):
                for k, condition in enumerate(CONDITIONS):
                    records.extend((panel, group, source, condition, candidate, score)
                                   for candidate, score in zip(ids, scores[j*len(CONDITIONS)+k]))
            print(size, group, f'{perf_counter()-start:.1f}s', flush=True)
    pd.DataFrame(records, columns=KEYS+[f'encoder {size}']).to_csv(output/f'encoder_{size}.csv.gz', index=False)
    (output/f'encoder_{size}.json').write_text(json.dumps(
        dict(size=size, resize=transform.resize_size if size==224 else [56], device=device,
             checkpoint_sha256=checksum, torch=torch.__version__, torchvision=torchvision.__version__,
             test_sha256=data['test_sha256'], seed=seed, threads=threads), indent=2)+'\n')


OFFSETS = sorted([(y, x) for y in range(-3, 4) for x in range(-3, 4)],
                 key=lambda offset: (sum(v*v for v in offset), *offset))


def squared_components(left, right, mask):
    """Per-site contributions to the squared normalized field distance."""
    left = np.where(mask, left, 0.)
    right = np.where(mask, right, 0.)
    a, b = abs(left), abs(right)
    scale = np.square(np.linalg.norm(a, axis=(-2, -1)) + np.linalg.norm(b, axis=(-2, -1)))
    scale = np.where(scale > 1e-24, scale, np.inf)[..., None, None]
    magnitude = (a-b)**2/scale
    angular = 2*np.maximum(a*b - (left*right.conj()).real, 0.)/scale
    return magnitude, angular


class CommonOffset:
    """Select a translation by ZNCC, then score both Q variants at that offset."""
    def __init__(self, images):
        self.images = images if images.ndim == 4 else images[..., None]
        self.q = np.stack([local_fields(Observation(image, np.ones(image.shape[:2], dtype=bool)))[0]
                           for image in images])
        self.windows = [overlap_slices(images.shape[1:3], [(0, 0), offset]) for offset in OFFSETS]

    def compare(self, query):
        pixels = query.image if query.image.ndim == 3 else query.image[..., None]
        count = len(self.images)
        chosen = np.full(count, -1)
        best = np.full(count, np.inf)
        for i, (a, b) in enumerate(self.windows):
            valid = query.valid[a]
            if valid.sum() < .65 * query.valid.size:
                continue
            left = pixels[a].reshape(valid.shape[0], -1)
            right = self.images[:, b[0], b[1]].reshape(count, valid.shape[0], -1)
            d = zncc_distance(left, right, np.repeat(valid, pixels.shape[-1], axis=1))
            improve = d < best - 1e-12
            chosen[improve], best[improve] = i, d[improve]
        if np.any(chosen < 0):
            raise ValueError('A query has insufficient support for common-offset comparison')
        q = local_fields(query)[0]
        parts = np.zeros((count, 2))
        for i in np.unique(chosen):
            a, b = self.windows[i]
            use = chosen == i
            maps = squared_components(q[a], self.q[use, b[0], b[1]], query.valid[a])
            parts[use] = np.stack([m.sum(axis=(-2, -1)) for m in maps], axis=1)
        return np.column_stack([np.array(OFFSETS)[chosen], best, parts])


def compute_fields(data, output, panel):
    records, members = [], []
    for tag, group, ids, images, labels in galleries(data, panel):
        print(tag, group, 'shared offsets', flush=True)
        scorer = CommonOffset(images if panel == 'fresh' else images @ LUMA)
        queries = observations(images, ids, panel)
        members.extend((tag, group, source, label) for source, label in zip(ids, labels))
        for j, source in enumerate(ids):
            for k, condition in enumerate(CONDITIONS):
                scores = scorer.compare(queries[j*len(CONDITIONS)+k])
                records.extend((tag, group, source, condition, candidate, int(dy), int(dx), zncc, mag, angular)
                               for candidate, (dy, dx, zncc, mag, angular) in zip(ids, scores))
    pd.DataFrame(records, columns=KEYS+['dy', 'dx', 'zncc', 'mag_sq', 'angular_sq']).to_csv(
        output/f'{panel}_pairs.csv.gz', index=False)
    pd.DataFrame(members, columns=['panel', 'gallery', 'source', 'label']).to_csv(
        output/f'{panel}_membership.csv', index=False)


def average_precision(scores, truth):
    """Expected non-interpolated AP under uniform permutations of exact score ties."""
    order = np.argsort(scores, kind='stable')
    sorted_truth = truth[order]
    positions = np.arange(1, len(order)+1)
    cumulative = np.cumsum(sorted_truth)
    contributions = np.where(sorted_truth, cumulative/positions, 0.)
    sorted_scores = scores[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0)+1, len(order)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        size = b-a
        if size == 1:
            continue
        relevant = sorted_truth[a:b].sum()
        before = cumulative[a] - sorted_truth[a]
        expected_previous = (positions[a:b]-a-1)*(relevant-1)/(size-1)
        contributions[a:b] = (relevant/size)*(before+1+expected_previous)/positions[a:b]
    return contributions.sum()/truth.sum()


def score_panel(data, output, panel):
    fields = pd.read_csv(output/f'{panel}_pairs.csv.gz')
    methods = FUSION_METHODS if panel == 'fresh' else FIELD_METHODS
    if panel == 'fresh':
        for size in (56, 224):
            enc = pd.read_csv(output/f'encoder_{size}.csv.gz')
            pd.testing.assert_frame_equal(fields[KEYS], enc[KEYS])
            fields[f'encoder {size}'] = enc[f'encoder {size}']
    fields['magnitude'] = np.sqrt(fields.mag_sq)
    fields['phase'] = np.sqrt(fields.mag_sq + fields.angular_sq)
    labels = data['y_test']
    records, switches = [], []
    for key, group in fields.groupby(KEYS[:-1], sort=False):
        _, _, source, _ = key
        group = group[group.candidate != source].copy()
        candidates = group.candidate.to_numpy()
        truth = labels[candidates] == labels[source]
        assert len(truth) == 39 and truth.sum() == 3
        if panel == 'fresh':
            group['encoder 224 + phase'] = (
                rankdata(group['encoder 224']) + rankdata(group['phase']))/2
        scores = group[methods].to_numpy()
        top = rank(scores, truth)
        records.extend((*key, name, hit, average_precision(distances, truth))
                       for name, hit, distances in zip(methods, top, scores.T))
        if panel == 'audit':
            mag, phase = group.magnitude.to_numpy(), group.phase.to_numpy()
            i, j = mag.argmin(), phase.argmin()
            if truth[i] != truth[j] and not any(
                    np.isclose(values, values.min(), rtol=0, atol=1e-8).sum() > 1
                    for values in (mag, phase)):
                pos, neg = (j, i) if truth[j] else (i, j)
                a, b = group.iloc[pos], group.iloc[neg]
                switches.append((*key, 'rescue' if truth[j] else 'harm', a.candidate, b.candidate,
                                 b.mag_sq-a.mag_sq, b.angular_sq-a.angular_sq))
    decisions = pd.DataFrame(records, columns=KEYS[:-1]+['method', 'top1', 'ap'])
    decisions.to_csv(output/f'{panel}_decisions.csv.gz', index=False)
    if panel == 'audit':
        pd.DataFrame(switches, columns=KEYS[:-1]+['switch', 'positive', 'negative',
                                                  'mag_margin', 'angular_margin']).to_csv(
            output/'audit_switches.csv', index=False)
    return decisions


def summarize(data, output):
    summaries, contrasts, conditions = [], [], []
    rng = np.random.default_rng(SEED+2)
    for panel in ('audit', 'fresh'):
        decisions = score_panel(data, output, panel)
        conditions.append(decisions.groupby(['panel', 'condition', 'method'])[['top1', 'ap']].mean().reset_index())
        comparisons = [('phase', 'magnitude')] if panel == 'audit' else [
            ('encoder 224 + phase', 'encoder 224'), ('encoder 224', 'encoder 56'),
            ('phase', 'magnitude')]
        for split, rows in [('clean', decisions.query("condition == 'clean'")),
                            ('transformed', decisions.query("condition != 'clean'"))]:
            summaries.append(rows.groupby(['panel', 'method'])[['top1', 'ap']].mean().assign(
                split=split).reset_index())
            means = rows.groupby(['gallery', 'method'])[['top1', 'ap']].mean()
            for metric in ('top1', 'ap'):
                wide = means[metric].unstack('method')
                for left, right in comparisons:
                    delta = (wide[left]-wide[right]).to_numpy()
                    draws = rng.choice(delta, (4000, len(delta)), replace=True).mean(1)
                    contrasts.append((panel, split, metric, left, right, delta.mean(),
                                      *np.quantile(draws, [.025, .975])))
    pd.concat(summaries).to_csv(output/'summary.csv', index=False)
    pd.concat(conditions).to_csv(output/'condition_summary.csv', index=False)
    pd.DataFrame(contrasts, columns=['panel', 'split', 'metric', 'left', 'right', 'delta', 'low95',
                                    'high95']).to_csv(output/'contrasts.csv', index=False)
    audit, fresh = (pd.read_csv(output/f'{panel}_membership.csv') for panel in ('audit', 'fresh'))
    assert audit.source.nunique() == fresh.source.nunique() == 480
    assert set(audit.source).isdisjoint(fresh.source)
    assert {data['x_test'][i].tobytes() for i in audit.source}.isdisjoint(
        {data['x_test'][i].tobytes() for i in fresh.source})
    (output/'manifest.json').write_text(json.dumps(
        dict(fresh_seed=SEED, audit_seed=PREVIOUS_SEED, galleries_per_panel=12, sources_per_gallery=40,
             test_sha256=data['test_sha256'], fusion_weights=[.5,.5], overlap_min=.65,
             radius=3, excluded_test_sources=480, audit='grayscale', fresh='RGB',
             conditions=list(CONDITIONS)), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['224', '56', 'audit', 'fresh', 'summarize'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = load_cifar(args.data)
    if args.part in ('224', '56'):
        encode(data, RESULTS, int(args.part), args.threads, args.device)
    elif args.part == 'summarize':
        summarize(data, RESULTS)
    else:
        compute_fields(data, RESULTS, args.part)
