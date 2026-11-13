from itertools import permutations

import numpy as np

from observables.open_images import window_distance
from observables.stage11 import average_precision, squared_components


def test_squared_components_reconstruct_both_distances():
    rng = np.random.default_rng(214)
    left, right = [rng.normal(size=(7, 9)) + 1j*rng.normal(size=(7, 9)) for _ in range(2)]
    mask = rng.random(left.shape) > .2
    right = np.stack((right, -left, left, np.zeros_like(left)))
    magnitude, angular = squared_components(left, right, mask)
    np.testing.assert_allclose(magnitude.sum((-2, -1)), window_distance(abs(left), abs(right), mask)**2)
    np.testing.assert_allclose((magnitude+angular).sum((-2, -1)),
                               window_distance(left, right, mask)**2, atol=1e-14)
    assert np.all(angular >= 0)
    assert not np.any(squared_components(left*0, right*0, mask))


def test_average_precision_marginalizes_within_score_ties():
    scores = np.array([0, 1, 1, 1, 2, 2])
    truth = np.array([True, False, True, True, False, True])
    ranks = np.arange(1, len(scores)+1)
    actual = []
    for middle in permutations([1, 2, 3]):
        for last in permutations([4, 5]):
            ordered = truth[np.array([0, *middle, *last])]
            actual.append((np.cumsum(ordered)[ordered]/ranks[ordered]).mean())
    np.testing.assert_allclose(average_precision(scores, truth), np.mean(actual))
