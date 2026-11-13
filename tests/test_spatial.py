import numpy as np
import pytest
from scipy.fft import dctn
from observables.spatial import (LAYOUTS,mosaic,patches,patch_moments,q_field,q_correlation,
                                orbit_distance,representations,candidate_distances)
from observables.core import moments

def test_equal_patch_multisets_and_global_moments():
    originals=[mosaic(name) for name in LAYOUTS]
    first=sorted(map(tuple,patches(originals[0])))
    for im in originals:
        assert sorted(map(tuple,patches(im)))==first
        for a,b in zip(patch_moments(im),patch_moments(originals[0])):
            np.testing.assert_allclose(a,b,atol=1e-12)
    global_q=[representations(a)['global_Q'] for a in originals]
    for q in global_q[1:]:np.testing.assert_allclose(q,global_q[0],atol=1e-12)

def test_orthogonal_patch_transform():
    x=patches(mosaic('checker'))
    y=dctn(x.reshape(-1,8,8),axes=(1,2),norm='ortho').reshape(-1,64)
    np.testing.assert_allclose(x@x.T,y@y.T,atol=1e-12)
    assert np.linalg.norm(moments(x)[1])==pytest.approx(np.linalg.norm(moments(y)[1]))

def test_translation_and_sign():
    a=mosaic('domains');b=-np.roll(a,(3,7),(0,1))
    qa,qb=q_field(a),q_field(b)
    np.testing.assert_allclose(np.roll(qa,(3,7),(0,1)),qb,atol=1e-12)
    np.testing.assert_allclose(q_correlation(qa),q_correlation(qb),atol=1e-12)
    assert orbit_distance(qa,qb)<1e-7
    assert orbit_distance(a,b,True)<1e-7

def test_fft_registration_against_exhaustive():
    rng=np.random.default_rng(34)
    a=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    b=rng.normal(size=(5,5))+1j*rng.normal(size=(5,5))
    direct=min(np.linalg.norm(a-np.roll(b,(i,j),(0,1)))/(np.linalg.norm(a)+np.linalg.norm(b))
               for i in range(5) for j in range(5))
    assert orbit_distance(a,b)==pytest.approx(direct)

def test_correlation_direct():
    q=q_field(mosaic('diagonal bands'))
    c=q_correlation(q)
    for lag in [(0,0),(1,2),(8,16)]:
        direct=np.mean(q*np.conj(np.roll(q,lag,(0,1)))).real
        assert c[lag]==pytest.approx(direct)

def test_orbit_retrieval():
    gallery=[representations(mosaic(n)) for n in LAYOUTS]
    for label,name in enumerate(LAYOUTS):
        a=-np.rot90(np.roll(mosaic(name),(3,7),(0,1)))
        scores=candidate_distances(a,gallery)
        for key in ['registered_Q','registered_pixels','Q_correlation']:
            assert np.argmin(scores[key])==label
            assert scores[key][label]<1e-7
