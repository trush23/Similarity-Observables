"""Stage 2: spatial organization under a fixed, inspectable image representation."""
import numpy as np
from scipy.ndimage import uniform_filter
from .images import gradient, tiled_orientation
from .core import moments, relative_distance

LAYOUTS = ('domains', 'alternating', 'checker', 'diagonal bands')
GRIDS = (2,4,8,16)

def layout(name, cells=8):
    y,x=np.mgrid[:cells,:cells]
    if name=='domains':
        return (x>=cells//2).astype(int)
    if name=='alternating':
        return x%2
    if name=='checker':
        return (x+y)%2
    if name=='diagonal bands':
        return ((x+y)//2)%2
    raise ValueError(name)

def tiles(size=8):
    if size<4:
        raise ValueError('Tiles require at least four pixels per side')
    y,x=np.mgrid[:size,:size]
    window=np.outer(np.hanning(size),np.hanning(size))
    a=window*np.cos(2*np.pi*2*x/(size-1))
    a[[0,-1],:]=0
    a[:,[0,-1]]=0
    return np.stack([a,a.T])

def mosaic(name, cells=8, tile_size=8):
    labels=layout(name,cells)
    library=tiles(tile_size)
    return library[labels].transpose(0,2,1,3).reshape(cells*tile_size,cells*tile_size)

def patches(image,size=8):
    a=np.asarray(image,dtype=float)
    if a.ndim!=2 or a.shape[0]%size or a.shape[1]%size:
        raise ValueError('A 2D image with dimensions divisible by patch size is required')
    return a.reshape(a.shape[0]//size,size,a.shape[1]//size,size).transpose(0,2,1,3).reshape(-1,size*size)

def patch_moments(image):
    x=patches(image)
    valid=np.linalg.norm(x,axis=1)>1e-12
    if not valid.any():
        return np.zeros(x.shape[1]),np.zeros((x.shape[1],x.shape[1]))
    return moments(x[valid])

def q_field(image,window=8):
    """Magnitude-weighted sliding nematic moment, with periodic boundaries."""
    gx,gy=gradient(image)
    z=gx+1j*gy
    mag=np.abs(z)
    numerator=np.divide(z*z,mag,out=np.zeros_like(z),where=mag>1e-12)
    real=uniform_filter(numerator.real,size=window,mode='wrap')
    imag=uniform_filter(numerator.imag,size=window,mode='wrap')
    mass=uniform_filter(mag,size=window,mode='wrap')
    return np.divide(real+1j*imag,mass,out=np.zeros_like(z),where=mass>1e-12)

def q_correlation(field):
    """Real trace contraction of uncentered complex nematic fields."""
    return np.fft.ifft2(np.abs(np.fft.fft2(field))**2).real/field.size

def orbit_distance(a,b,sign_free=False):
    """Minimum normalized RMS over periodic translations; rotation handled outside."""
    a,b=np.asarray(a),np.asarray(b)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    if na+nb<=1e-12:
        return 0.
    products=np.fft.ifft2(np.fft.fft2(a)*np.conj(np.fft.fft2(b))).real
    best=np.max(np.abs(products)) if sign_free else np.max(products)
    residual=max(0.,na*na+nb*nb-2*best)
    return np.sqrt(residual)/(na+nb)

def whole_cosine(a,b):
    a,b=np.ravel(a),np.ravel(b)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm>1e-12 else float(np.array_equal(a,b))

def representations(image):
    mean,q=patch_moments(image)
    field=q_field(image)
    return dict(pixels=image,patch_mean=mean,patch_Q=q,
                global_Q=tiled_orientation(image,1)[:,1],
                **{f'local_Q_{g}':tiled_orientation(image,g)[:,1] for g in GRIDS},
                field=field,correlation=q_correlation(field))

def candidate_distances(image,gallery):
    """Every orbit method minimizes over all four rotations and all translations."""
    rotations=[representations(np.rot90(image,k)) for k in range(4)]
    r=rotations[0]
    output={}
    for key in ('pixels','patch_mean','patch_Q','global_Q',*[f'local_Q_{g}' for g in GRIDS]):
        output[key]=np.array([relative_distance(r[key],g[key]) for g in gallery])
    output['whole_Q']=np.array([1-whole_cosine(image,g['pixels'])**2 for g in gallery])
    output['Q_correlation']=np.array([min(relative_distance(a['correlation'],g['correlation'])
                                          for a in rotations) for g in gallery])
    output['registered_Q']=np.array([min(orbit_distance(a['field'],g['field']) for a in rotations) for g in gallery])
    output['registered_pixels']=np.array([min(orbit_distance(a['pixels'],g['pixels'],True) for a in rotations) for g in gallery])
    return output
