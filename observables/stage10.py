"""Matched phase and gradient-weighting experiments on synthetic edges and CIFAR-10."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .open_images import Observation, finite_gradient, overlap_slices, window_distance
from .stage6 import CONDITIONS, view
from .stage8 import FROZEN, KEYS, rank, replication_galleries, run_frozen, zncc_distance

SEED = 10427
FLOOR = .05
TEST_MD5 = '40351d587109b95175f43aff81a1287e'
LUMA = np.array([.299, .587, .114])
PRIMARY = ['p1 phase', 'p1 magnitude', 'p2 phase', 'p2 magnitude']
UNDAMPED = [f'{name} no floor' for name in PRIMARY]
PIXELS = ['aligned pixels', 'aligned ZNCC']
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def local_fields(obs, *, include_undamped=False):
    """Local axial moments with a fixed gradient floor in magnitude units."""
    pixels = obs.image[..., None] if obs.image.ndim == 2 else obs.image
    gradients = np.array([finite_gradient(np.where(obs.valid, channel, 0.), obs.valid)
                          for channel in np.moveaxis(pixels, -1, 0)])
    magnitude = np.abs(gradients)
    square = gradients**2
    linear = np.divide(square, magnitude, out=np.zeros_like(square), where=magnitude > 1e-12)
    support = uniform_filter(obs.valid.astype(float), 5, mode='constant') * len(gradients)
    damped, undamped = [], []
    for p, moment in ((1, linear), (2, square)):
        mass = uniform_filter((magnitude**p).sum(0), 5, mode='constant')
        numerator = (uniform_filter(moment.real.sum(0), 5, mode='constant')
                     + 1j * uniform_filter(moment.imag.sum(0), 5, mode='constant'))
        denominators = [mass + FLOOR**p * support]
        if include_undamped:
            denominators.append(mass)
        for output, denominator in zip((damped, undamped), denominators):
            field = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                              where=(denominator > 1e-12) & obs.valid)
            output.extend((field, np.abs(field)))
    return damped + undamped


class Aligned:
    """Same open offsets and 65% common support across all field comparisons."""
    def __init__(self, images, include_undamped=False):
        self.include_undamped = include_undamped
        self.windows = [overlap_slices(images.shape[1:3], [(0, 0), (y, x)])
                        for y in range(-3, 4) for x in range(-3, 4)]
        self.fields = [np.stack(fields) for fields in zip(*(
            local_fields(Observation(image, np.ones(image.shape[:2], dtype=bool)),
                         include_undamped=include_undamped)
            for image in images))]
        # Concatenate channels along the width of each comparison window.
        self.images = images if images.ndim == 4 else images[..., None]

    def compare(self, query):
        fields = local_fields(query, include_undamped=self.include_undamped)
        q_pixels = query.image if query.image.ndim == 3 else query.image[..., None]
        count = len(self.images)
        minimum = np.ones((count, len(fields) + 2))
        for a, b in self.windows:
            shared = query.valid[a]
            if shared.sum() < .65 * query.valid.size:
                continue
            left = q_pixels[a].reshape(shared.shape[0], -1)
            right = self.images[:, b[0], b[1]].reshape(count, shared.shape[0], -1)
            pixel_mask = np.repeat(shared, q_pixels.shape[-1], axis=1)
            distances = [window_distance(left, right, pixel_mask), zncc_distance(left, right, pixel_mask)]
            distances.extend(window_distance(field[a], gallery[:, b[0], b[1]], shared)
                             for field, gallery in zip(fields, self.fields))
            np.minimum(minimum, np.column_stack(distances), out=minimum)
        return minimum


def load_cifar(path=None):
    path = Path(path or os.environ.get('CIFAR10_TEST_BATCH',
                                       Path.home()/'.keras/datasets/cifar-10-batches-py/test_batch'))
    payload = path.read_bytes()
    # Authenticate the known data file before unpickling.
    if hashlib.md5(payload).hexdigest() != TEST_MD5:
        raise ValueError('Expected the official CIFAR-10 Python test batch')
    batch = pickle.loads(payload, encoding='bytes')
    x = batch[b'data']
    y = np.asarray(batch[b'labels'])
    if x.shape != (10000, 3072) or x.dtype != np.uint8 or not np.array_equal(np.bincount(y), [1000]*10):
        raise ValueError('Unexpected CIFAR-10 test batch layout')
    return {'x_test': x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), 'y_test': y,
            'test_sha256': hashlib.sha256(payload).hexdigest()}


def cifar_views(image, condition, seed):
    gray = view(image @ LUMA, condition, np.random.default_rng(seed))
    # Reset each channel to the same seed for identical geometry, mask and noise.
    channels = [view(channel, condition, np.random.default_rng(seed))
                for channel in np.moveaxis(image, -1, 0)]
    return gray, Observation(np.stack([obs.image for obs in channels], axis=-1), gray.valid)


def cifar_galleries(data, excluded):
    for _, group, ids, rgb, labels in replication_galleries(data, excluded, seed=SEED):
        gray, color = zip(*(cifar_views(image, condition, [SEED, int(source), k])
                            for image, source in zip(rgb, ids)
                            for k, condition in enumerate(CONDITIONS)))
        for panel, gallery, queries in [('gray', rgb @ LUMA, gray), ('rgb', rgb, color)]:
            yield panel, group, ids, gallery, labels, queries


def edge(angle, shift=0., width=.9):
    y, x = np.mgrid[-16:17, -16:17]
    position = x*np.cos(angle) + y*np.sin(angle) - shift
    return np.tanh(position/width)


def noisy(image, rng, sd):
    return Observation.full(np.clip(image + rng.normal(0, sd, image.shape), 0, 1))


def synthetic_angles(output):
    rows = []
    rng = np.random.default_rng(SEED)
    # Common random noise across strength, angle separation, and all estimators.
    for trial in range(160):
        angle = rng.uniform(0, np.pi)
        noise = rng.standard_normal((33, 33))
        primary = edge(angle)
        for separation in (45, 90):
            crossing = edge(angle + np.deg2rad(separation))
            for ratio in (0., .25, .5, .75, 1., 1.25):
                clean = .5 + .19 * (primary + ratio * crossing)
                for sd in (0., .03, .08):
                    obs = Observation.full(np.clip(clean + sd*noise, 0, 1))
                    fields = local_fields(obs, include_undamped=True)
                    for name, field in zip((PRIMARY+UNDAMPED)[::2], fields[::2]):
                        q = field[16, 16]
                        error = abs(np.angle(q * np.exp(-2j*angle))) * 90 / np.pi if abs(q)>1e-12 else np.nan
                        rows.append((trial, separation, ratio, sd, name, error, abs(q)))
    pd.DataFrame(rows, columns=['trial', 'separation_deg', 'ratio', 'noise_sd', 'method',
                                'axis_error_deg', 'magnitude']).to_csv(output/'angle_trials.csv.gz', index=False)


def synthetic_galleries():
    rng = np.random.default_rng(SEED+1)
    for group in range(12):
        labels = np.repeat(np.arange(6), 4)
        ids = np.arange(group*len(labels), (group+1)*len(labels))
        gallery, queries = [], []
        for label in labels:
            theta = label*np.pi/6
            shift, width = rng.uniform(-4, 4), rng.uniform(.65, 2.5)
            strength = rng.uniform(.11, .23)
            ramp = edge(theta, shift, width)
            crossing = edge(theta + np.pi/3, rng.uniform(-4, 4), rng.uniform(.65, 2.5))
            clean = .5 + strength*ramp
            gallery.append(clean)
            queries.extend([Observation.full(clean + .65*strength*crossing),
                            noisy(clean, rng, .08),
                            noisy(clean + .65*strength*crossing, rng, .08),
                            noisy(.5 + .45*strength*ramp, rng, .08)])
        yield 'synthetic', group, ids, np.stack(gallery), labels, queries


def collect(galleries, conditions, output, *, synthetic=False):
    rows, memberships = [], []
    names = PIXELS + PRIMARY + (UNDAMPED if synthetic else [])
    for panel, group, ids, gallery, labels, queries in galleries:
        print(panel, group, 'aligned', flush=True)
        scorer = Aligned(gallery, include_undamped=synthetic)
        memberships.extend((panel, group, source, label) for source, label in zip(ids, labels))
        for j, source in enumerate(ids):
            for k, condition in enumerate(conditions):
                if condition == 'clean':
                    continue
                scores = scorer.compare(queries[j*len(conditions)+k])
                rows.extend((panel, group, source, condition, candidate, *row)
                            for candidate, row in zip(ids, scores))
    pairs = pd.DataFrame(rows, columns=KEYS+names)
    membership = pd.DataFrame(memberships, columns=['panel', 'gallery', 'source', 'label'])
    pairs.to_csv(output/'classical_pairs.csv.gz', index=False)
    membership.to_csv(output/'membership.csv', index=False)


def summarize(output, *, synthetic=False):
    pairs = pd.read_csv(output/'classical_pairs.csv.gz')
    membership = pd.read_csv(output/'membership.csv')
    methods = PIXELS + PRIMARY + UNDAMPED if synthetic else PIXELS + PRIMARY + list(FROZEN)
    if not synthetic:
        frozen = pd.read_csv(output/'frozen_pairs.csv.gz')
        assert pairs[KEYS].equals(frozen[KEYS])
        pairs = pairs.merge(frozen, on=KEYS, validate='one_to_one')
    membership_keys = ['panel', 'gallery', 'source']
    labeled = pairs.merge(membership, on=membership_keys, validate='many_to_one')
    candidate_labels = membership.rename(columns={'source': 'candidate', 'label': 'candidate_label'})
    labeled = labeled.merge(candidate_labels, on=['panel', 'gallery', 'candidate'], validate='many_to_one')
    assert len(labeled) == len(pairs)
    rows = []
    for key, group in labeled.groupby(KEYS[:-1], sort=False):
        scores = group[methods].to_numpy()
        identity = group.source.to_numpy() == group.candidate.to_numpy()
        category = group.label.to_numpy() == group.candidate_label.to_numpy()
        assert identity.sum() == 1 and category.sum() == 4
        rows.extend((*key, method, exact, same) for method, exact, same in
                    zip(methods, rank(scores, identity), rank(scores[~identity], category[~identity])))
    decisions = pd.DataFrame(rows, columns=KEYS[:-1]+['method', 'identity', 'class'])
    decisions.to_csv(output/'decisions.csv.gz', index=False)
    decisions.groupby(['panel', 'method'])[['identity', 'class']].mean().to_csv(output/'summary.csv')
    decisions.groupby(['panel', 'condition', 'method'])[['identity', 'class']].mean().to_csv(
        output/'condition_summary.csv')
    contrasts = {
        'p1 phase gain': {'p1 phase': 1, 'p1 magnitude': -1},
        'p2 phase gain': {'p2 phase': 1, 'p2 magnitude': -1},
        'p1 minus p2 phase': {'p1 phase': 1, 'p2 phase': -1},
        'phase-gain interaction': {'p1 phase': 1, 'p1 magnitude': -1, 'p2 phase': -1, 'p2 magnitude': 1},
    }
    if synthetic:
        contrasts.update({f'{method} floor effect': {method: 1, f'{method} no floor': -1}
                          for method in PRIMARY})
    means = decisions.groupby(['panel', 'gallery', 'method'])[['identity', 'class']].mean()
    effects = []
    rng = np.random.default_rng(SEED+2)
    for endpoint in ('identity', 'class'):
        wide = means[endpoint].unstack('method')
        panel_effects = {panel: {name: sum(coef*group[method] for method, coef in weights.items())
                                 for name, weights in contrasts.items()}
                         for panel, group in wide.groupby(level='panel')}
        if not synthetic:
            panel_effects['rgb minus gray'] = {
                name: panel_effects['rgb'][name].droplevel('panel') -
                panel_effects['gray'][name].droplevel('panel') for name in contrasts}
        for panel, panel_contrasts in panel_effects.items():
            for name, delta in panel_contrasts.items():
                draws = rng.choice(delta.to_numpy(), (4000, len(delta)), replace=True).mean(1)
                effects.append((panel, endpoint, name, delta.mean(), *np.quantile(draws, [.025, .975])))
    pd.DataFrame(effects, columns=['panel', 'endpoint', 'contrast', 'difference', 'low95', 'high95']).to_csv(
        output/'contrasts.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['synthetic', 'classical', 'frozen', 'summarize'])
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]/'results'/'stage10'
    output = root/('synthetic' if args.part == 'synthetic' else 'cifar10')
    output.mkdir(parents=True, exist_ok=True)
    if args.part == 'synthetic':
        synthetic_angles(output)
        collect(synthetic_galleries(), ('crossing', 'noise', 'crossing + noise', 'weak + noise'),
                output, synthetic=True)
        summarize(output, synthetic=True)
    elif args.part == 'classical':
        data = load_cifar(args.data)
        collect(cifar_galleries(data, set()), CONDITIONS, output)
        metadata = dict(seed=SEED, test_md5=TEST_MD5, test_sha256=data['test_sha256'],
                        gallery_count=12, gallery_size=40, luma_coefficients=LUMA.tolist(),
                        perturbations=list(CONDITIONS[1:]), floor=FLOOR, window=5,
                        radius=3, min_coverage=.65, numpy=np.__version__, pandas=pd.__version__)
        (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    elif args.part == 'frozen':
        run_frozen(load_cifar(args.data), set(), output, generate=cifar_galleries)
    else:
        summarize(output)
