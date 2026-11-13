"""Finite image measurements with explicit valid support and no wraparound."""
from dataclasses import dataclass
import numpy as np
from scipy.fft import dctn
from scipy.ndimage import uniform_filter
from .core import relative_distance

LAGS = ((0,1),(1,0),(1,1),(1,-1),(0,2),(2,0),(0,4),(4,0))
COMPONENTS = ('pixels', 'polar m1', 'global Q', 'angular m4', 'HOG',
              'local Q', 'open Q pairs', 'open Q triplets', 'DCT power',
              'aligned pixels', 'aligned local Q')
ABLATIONS = ('aligned Q magnitude','aligned gradient strength')

@dataclass
class Observation:
    image: np.ndarray
    valid: np.ndarray

    @classmethod
    def full(cls, image):
        a = np.asarray(image,dtype=float)
        return cls(a,np.ones(a.shape,dtype=bool))


def overlap_slices(shape, offsets):
    """Slices for x, x+r, … on one finite grid with common valid extent."""
    lower = np.maximum(0,-np.min(offsets,axis=0))
    upper = np.minimum(shape,np.asarray(shape)-np.max(offsets,axis=0))
    if np.any(upper<=lower):
        raise ValueError('Offsets have empty overlap')
    return [tuple(slice(int(lo+d),int(hi+d)) for lo,hi,d in zip(lower,upper,offset))
            for offset in offsets]


def translate(observation, offset):
    dst,src = overlap_slices(observation.image.shape,[(0,0),tuple(-v for v in offset)])
    image = np.zeros_like(observation.image)
    valid = np.zeros_like(observation.valid)
    image[dst] = observation.image[src]
    valid[dst] = observation.valid[src]
    return Observation(image,valid)


def finite_gradient(image, valid):
    """Centered derivatives inside support, one-sided at its boundary."""
    gradients = []
    for axis in (1,0):
        x = np.moveaxis(image,axis,0)
        m = np.moveaxis(valid,axis,0)
        derivative = np.zeros_like(x)
        edge = (x[1:]-x[:-1])*(m[1:]&m[:-1])
        derivative[1:] += edge
        derivative[:-1] += edge
        count = np.zeros(x.shape,dtype=int)
        count[1:] += m[1:]&m[:-1]
        count[:-1] += m[1:]&m[:-1]
        derivative = np.divide(derivative,count,out=derivative,where=count>0)
        gradients.append(np.moveaxis(derivative,0,axis))
    return gradients[0]+1j*gradients[1]


def open_moment(field, valid, offsets):
    windows = overlap_slices(field.shape,offsets)
    support = np.logical_and.reduce([valid[s] for s in windows])
    if not support.any():
        return 0.
    values = [field[s][support] for s in windows]
    if len(values)==2:
        return float(np.mean((values[0]*values[1].conj()).real))
    return float(np.mean(np.prod([v.real for v in values],axis=0)))


def describe(observation):
    image = np.asarray(observation.image,dtype=float)
    valid = np.asarray(observation.valid,dtype=bool)
    if image.ndim!=2 or min(image.shape)<9 or valid.shape!=image.shape or not valid.any():
        raise ValueError('Expected a 2D image at least 9 × 9 with nonempty valid support')
    image = np.where(valid,image,0.)
    if not np.isfinite(image).all():
        raise ValueError('Observed values must be finite')
    g = finite_gradient(image,valid)
    mass = np.abs(g)
    direction = np.divide(g,mass,out=np.zeros_like(g),where=mass>1e-12)
    total = mass.sum()
    moments = [np.array([(mass*direction**k).sum()/total if total>1e-12 else 0j])
               for k in (1,2,4)]
    local_mass = uniform_filter(mass,size=5,mode='constant')
    support = uniform_filter(valid.astype(float),size=5,mode='constant')
    quadrupole = mass*direction**2
    numerator = (uniform_filter(quadrupole.real,5,mode='constant')
                 +1j*uniform_filter(quadrupole.imag,5,mode='constant'))
    denom = local_mass+.05*support
    q = np.divide(numerator,denom,out=np.zeros_like(numerator),where=(denom>1e-12)&valid)
    pairs = np.array([open_moment(q,valid,[(0,0),lag]) for lag in LAGS])
    triples = np.array([open_moment(q,valid,[(0,0),r,s]) for r in LAGS for s in LAGS])
    # Signed gradient polarity is discarded by the modulo-pi binning.
    bins = np.floor(np.mod(np.angle(g),np.pi)*9/np.pi).astype(int).clip(0,8)
    histograms = []
    for row in np.array_split(np.arange(image.shape[0]),4):
        for col in np.array_split(np.arange(image.shape[1]),4):
            cell = np.ix_(row,col)
            hist = np.bincount(bins[cell].ravel(),weights=mass[cell].ravel(),minlength=9)
            histograms.append(hist/(np.linalg.norm(hist)+.05))
    spectral = dctn(np.where(valid,image-image[valid].mean(),0.),norm='ortho')**2
    strength = np.divide(local_mass,denom,out=np.zeros_like(local_mass),where=(denom>1e-12)&valid)
    return dict(pixels=image,valid=valid,q=q,strength=strength,features=[*moments,np.concatenate(histograms),pairs,triples,spectral])


def window_distance(a,b,mask):
    """One distance per candidate over its shared support."""
    left = np.where(mask,a,0.)
    right = np.where(mask,b,0.)
    norm = lambda x: np.sqrt(np.sum(np.abs(x)**2,axis=(-2,-1)))
    total = norm(left)+norm(right)
    return np.divide(norm(left-right),total,out=np.zeros_like(total),where=total>1e-12)


class Gallery:
    """Cache finite-image descriptors and compare using a shared search policy."""
    def __init__(self, references, radius=3, min_coverage=.65):
        self.references = [describe(ref) for ref in references]
        self.shape = self.references[0]['pixels'].shape
        if radius<0 or 2*radius>=min(self.shape) or not 0<min_coverage<=1:
            raise ValueError('Invalid search radius or overlap threshold')
        self.radius = radius
        self.min_coverage = min_coverage
        self.valid = np.stack([r['valid'] for r in self.references])
        self.images = np.stack([r['pixels'] for r in self.references])
        self.q = np.stack([r['q'] for r in self.references])
        self.strength = np.stack([r['strength'] for r in self.references])
        self.windows = [(offset,overlap_slices(self.shape,[(0,0),offset]))
                        for offset in [(y,x) for y in range(-radius,radius+1)
                                       for x in range(-radius,radius+1)]]

    def compare(self, observation, ablations=False):
        query = describe(observation)
        if query['pixels'].shape!=self.shape:
            raise ValueError('Images must share the comparison canvas')
        shared = self.valid&query['valid']
        fixed_pixels = window_distance(query['pixels'],self.images,shared)
        fixed_q = window_distance(query['q'],self.q,shared)
        fields = [(query['pixels'],self.images),(query['q'],self.q)]
        if ablations:
            fields += [(np.abs(query['q']),np.abs(self.q)),(query['strength'],self.strength)]
        minimum = np.full((len(fields),len(self.references)),np.inf)
        offsets = np.zeros((len(fields),len(self.references),2),dtype=int)
        coverage = np.zeros((len(fields),len(self.references)))
        for offset,(left,right) in self.windows:
            mask = self.valid[:,right[0],right[1]]&query['valid'][left]
            fraction = mask.sum((1,2))/np.prod(self.shape)
            eligible = fraction>=self.min_coverage
            if not eligible.any():
                continue
            for i,(a,b) in enumerate(fields):
                distance = window_distance(a[left],b[:,right[0],right[1]],mask)
                improved = eligible&(distance<minimum[i])
                minimum[i,improved] = distance[improved]
                coverage[i,improved] = fraction[improved]
                offsets[i,improved] = offset
        # Insufficient overlap remains explicit in coverage and produces max distance.
        minimum = np.where(np.isfinite(minimum),minimum,1.)
        other = np.array([[relative_distance(a,b) for a,b in zip(query['features'],r['features'])]
                          for r in self.references])
        scores = np.column_stack([fixed_pixels,other[:,:4],fixed_q,other[:,4:],minimum.T])
        scores[~shared.any((1,2)),0] = 1.
        scores[~shared.any((1,2)),5] = 1.
        return scores,offsets,coverage


def similarity_profile(a,b,radius=3,min_coverage=.65):
    """Return named agreements and alignment diagnostics for a single pair."""
    scores,offsets,coverage = Gallery([b],radius,min_coverage).compare(a)
    return dict(zip(COMPONENTS,1-scores[0])),offsets[:,0],coverage[:,0]
