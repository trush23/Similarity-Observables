import numpy as np
import pytest
from observables.core import (moments,tensor_moment,spherical_kernel,angular_distance,
                             power,autocorrelation,bispectrum,periodic_correlation,relative_distance)
from observables.images import orientation,profile
from observables.patterns import homometric,stripe,symbolic_sequence

@pytest.mark.parametrize('dimension',[2,3,8,64])
def test_q_identity(dimension):
    rng=np.random.default_rng(dimension)
    u,v=rng.normal(size=(2,dimension))
    u/=np.linalg.norm(u);v/=np.linalg.norm(v)
    _,qa=moments(u[None]);_,qb=moments(v[None])
    actual=np.sum(qa*qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))
    assert actual==pytest.approx((dimension*np.dot(u,v)**2-1)/(dimension-1))
    assert actual==pytest.approx(spherical_kernel(u@v,2,dimension))

@pytest.mark.parametrize('order',[1,2,3,4])
def test_tensor_kernel_identity(order):
    rng=np.random.default_rng(31)
    a,b=rng.normal(size=(7,3)),rng.normal(size=(9,3))
    a/=np.linalg.norm(a,axis=1)[:,None];b/=np.linalg.norm(b,axis=1)[:,None]
    assert np.sum(tensor_moment(a,order)*tensor_moment(b,order))==pytest.approx(np.mean((a@b.T)**order))

def test_weighted_trace_rotation():
    rng=np.random.default_rng(9);a=rng.normal(size=(16,4));w=rng.uniform(size=16)
    r,_=np.linalg.qr(rng.normal(size=(4,4)))
    m,q=moments(a,w);mr,qr=moments(a@r.T,w)
    assert np.trace(q)==pytest.approx(0,abs=1e-14)
    np.testing.assert_allclose(mr,r@m)
    np.testing.assert_allclose(qr,r@q@r.T)

def test_angular_collision():
    angles=np.arange(4)*np.pi/2
    a=np.column_stack([np.cos(angles),np.sin(angles)])
    angles+=np.pi/4;b=np.column_stack([np.cos(angles),np.sin(angles)])
    assert angular_distance(a,b,1)<1e-12
    assert angular_distance(a,b,2)<1e-12
    assert angular_distance(a,b,4)==pytest.approx(4.)

def test_homometry():
    a,b=homometric()
    assert not any(np.array_equal(a,np.roll(c,k)) for c in (b,b[::-1]) for k in range(12))
    np.testing.assert_allclose(power(a),power(b),atol=1e-12)
    np.testing.assert_allclose(autocorrelation(a),autocorrelation(b),atol=1e-12)
    assert np.linalg.norm(bispectrum(a)-bispectrum(b))>1
    np.testing.assert_allclose(bispectrum(a),bispectrum(np.roll(a,3)),atol=1e-12)

def test_wiener_khinchin():
    x=np.random.default_rng(8).normal(size=(8,8));x-=x.mean()
    c=autocorrelation(x)
    for dy,dx in [(0,0),(1,2),(3,4)]:
        assert c[dy,dx]==pytest.approx(np.mean(x*np.roll(x,(dy,dx),(0,1))))

def test_sequence_order():
    a,b=symbolic_sequence('blocks'),symbolic_sequence('alternating')
    np.testing.assert_allclose(moments(a)[1],moments(b)[1],atol=1e-14)
    assert periodic_correlation(a,1)[0,0]==pytest.approx(.875)
    assert periodic_correlation(b,1)[0,0]==pytest.approx(-1)

def test_image_symmetries():
    a=stripe();p=profile(a,-a);shift=profile(a,np.roll(a,3,axis=1))
    assert p['nematic']==pytest.approx(1)
    assert shift['power']==pytest.approx(1)
    assert relative_distance(np.zeros(2),np.zeros(2))==0
    assert orientation(np.zeros((8,8)))[1]==0

@pytest.mark.parametrize('a,w',[([[0,0]],None),([[1,0]],[-1]),([[1,0]],[0]),([[float('nan'),0]],None)])
def test_invalid_orientation(a,w):
    with pytest.raises(ValueError):moments(a,w)

def test_periodic_gradient_polar_constraint():
    image=np.random.default_rng(90).normal(size=(32,32))
    angular,energy=orientation(image)
    assert abs(angular[0])<1e-12 and energy>0
