import numpy as np
import pytest
from observables.open_images import (Observation,translate,finite_gradient,open_moment,
                                     similarity_profile,Gallery)


def test_translation_does_not_wrap():
    image = np.zeros((28,28))
    image[10,27] = 1
    shifted = translate(Observation.full(image),(0,2))
    assert shifted.image.sum()==0
    assert not shifted.valid[:,:2].any()


def test_crop_has_no_artificial_gradient():
    image = np.ones((28,28))
    valid = np.ones_like(image,dtype=bool)
    valid[:,:6] = False
    image[~valid] = 0
    np.testing.assert_array_equal(finite_gradient(image,valid),0)


def test_open_moments_match_direct_sums():
    rng = np.random.default_rng(26)
    a = rng.normal(size=(12,12))+1j*rng.normal(size=(12,12))
    mask = rng.random(a.shape)>.2
    values = [(a[i,j]*a[i+2,j-1].conjugate()).real
              for i in range(10) for j in range(1,12) if mask[i,j] and mask[i+2,j-1]]
    assert open_moment(a,mask,[(0,0),(2,-1)])==pytest.approx(np.mean(values))


def test_known_shift_and_swap_symmetry():
    image = np.zeros((28,28))
    image[7:19,9:12] = 1
    image[17:19,9:20] = .7
    a = Observation.full(image)
    b = translate(a,(3,-2))
    forward,offset,coverage = similarity_profile(b,a)
    reverse,_,_ = similarity_profile(a,b)
    assert forward['aligned pixels']==pytest.approx(1.)
    assert forward['aligned local Q']==pytest.approx(1.)
    np.testing.assert_array_equal(offset[0],[-3,2])
    assert coverage[0]==pytest.approx(25*26/784)
    np.testing.assert_allclose(list(forward.values()),list(reverse.values()))


def test_low_overlap_is_reported():
    a = Observation.full(np.ones((28,28)))
    b = Observation.full(np.ones((28,28)))
    b.valid[3:] = False
    _,_,coverage = Gallery([a]).compare(b)
    assert not coverage.any()
