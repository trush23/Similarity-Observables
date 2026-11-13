import numpy as np

from observables.open_images import Observation, describe
from observables.stage8 import structure_field
from observables.stage10 import Aligned, local_fields


def test_stage10_grayscale_fields_match_previous_methods_with_open_boundaries():
    rng = np.random.default_rng(41)
    image = rng.random((28, 28))
    valid = rng.random(image.shape) > .15
    obs = Observation(image, valid)
    linear, linear_magnitude, quadratic, quadratic_magnitude = local_fields(obs)
    np.testing.assert_allclose(linear, describe(obs)['q'], atol=1e-14)
    np.testing.assert_allclose(quadratic, structure_field(obs), atol=1e-14)
    np.testing.assert_allclose(linear_magnitude, abs(linear))
    np.testing.assert_allclose(quadratic_magnitude, abs(quadratic))


def test_repeating_grayscale_channels_preserves_all_distances():
    rng = np.random.default_rng(42)
    gray = rng.random((3, 12, 13))
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    valid = rng.random(gray.shape[1:]) > .1
    gray_scores = Aligned(gray[:2], include_undamped=True).compare(Observation(gray[2], valid))
    rgb_scores = Aligned(rgb[:2], include_undamped=True).compare(Observation(rgb[2], valid))
    np.testing.assert_allclose(gray_scores, rgb_scores, atol=1e-14)
    assert np.all(gray_scores[:, 2::2] + 1e-14 >= gray_scores[:, 3::2])
