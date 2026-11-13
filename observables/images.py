"""One explicit representation: periodic central-difference image gradients."""
import numpy as np
from .core import circular_moments, power, similarity

def gradient(image):
    a = np.asarray(image, dtype=float)
    gx = (np.roll(a, -1, axis=1)-np.roll(a, 1, axis=1))/2
    gy = (np.roll(a, -1, axis=0)-np.roll(a, 1, axis=0))/2
    return gx, gy

def orientation(image):
    gx, gy = gradient(image)
    magnitude = np.hypot(gx, gy)
    active = magnitude > 1e-12
    if not active.any():
        return np.zeros(3, dtype=complex), 0.
    theta = np.arctan2(gy[active], gx[active])
    return circular_moments(theta, magnitude[active]), float(magnitude.mean())

def tiled_orientation(image, grid=4):
    a = np.asarray(image)
    if a.shape[0] % grid or a.shape[1] % grid:
        raise ValueError("Image dimensions must be divisible by grid")
    # Compute the field once, so grid-cell boundaries create no artificial gradients.
    gx, gy = gradient(a)
    h, w = a.shape[0]//grid, a.shape[1]//grid
    output = []
    for row in range(grid):
        for col in range(grid):
            x = gx[row*h:(row+1)*h, col*w:(col+1)*w]
            y = gy[row*h:(row+1)*h, col*w:(col+1)*w]
            mag = np.hypot(x,y)
            valid = mag > 1e-12
            z = circular_moments(np.arctan2(y[valid],x[valid]),mag[valid]) if valid.any() else np.zeros(3,complex)
            output.append(z)
    return np.array(output)

def radial_power(image):
    p = power(image)
    fy, fx = np.meshgrid(np.fft.fftfreq(p.shape[0])*p.shape[0],
                         np.fft.fftfreq(p.shape[1])*p.shape[1], indexing='ij')
    radius = np.rint(np.hypot(fx,fy)).astype(int)
    # Shell sums preserve total power; shell radius rounded to nearest frequency index.
    return np.bincount(radius.ravel(), weights=p.ravel())

def descriptor(image):
    angular, energy = orientation(image)
    return dict(mean=np.array([np.mean(image)]),
                variance=np.array([np.var(image)]), gradient_energy=np.array([energy]),
                polar=angular[:1], nematic=angular[1:2], fourth=angular[2:],
                local_nematic=tiled_orientation(image)[:,1],
                power=power(image), radial_power=radial_power(image))

def profile(a, b):
    x, y = descriptor(a), descriptor(b)
    return {key:similarity(x[key],y[key]) for key in x}
