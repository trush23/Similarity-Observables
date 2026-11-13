import numpy as np

def stripe(size=64, wave=(4, 0), phase=0., square=False):
    y, x = np.mgrid[:size, :size]
    signal = np.cos(2*np.pi*(wave[0]*x+wave[1]*y)/size+phase)
    if square:
        signal = np.where(signal >= 0, 1., -1.)
    return signal

def checker(size=64, cells=8):
    y, x = np.mgrid[:size, :size]
    return (-1.)**((x//(size//cells))+(y//(size//cells)))

def motif_gallery():
    a = stripe(square=True)
    b = checker()
    shuffled = a.ravel()[np.random.default_rng(41).permutation(a.size)].reshape(a.shape)
    patch = a.copy()
    patch[16:48,16:48] = stripe(size=32,wave=(0,2),square=True)
    return {'vertical':a, 'horizontal':a.T, 'diagonal':stripe(wave=(4,4),square=True),
            'checkerboard':b, 'shifted':np.roll(a,5,axis=1),
            'contrast reversed':-a, 'fine stripes':stripe(wave=(8,0),square=True),
            'patch replacement':patch,'pixel shuffled':shuffled}

def homometric():
    a, b = np.zeros(12), np.zeros(12)
    a[[0,1,4,6]] = 1
    b[[0,1,3,7]] = 1
    return a, b

def symbolic_sequence(kind, size=32):
    if kind == 'blocks':
        theta = np.repeat([0, np.pi], size//2)
    elif kind == 'alternating':
        theta = np.tile([0, np.pi], size//2)
    else:
        raise ValueError(kind)
    return np.column_stack([np.cos(theta), np.sin(theta)])
