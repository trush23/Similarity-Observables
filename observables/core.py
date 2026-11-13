"""Explicit conventions for finite, weighted orientational configurations."""
import numpy as np
from scipy.special import eval_gegenbauer

def directions(vectors, weights=None):
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite nonempty N by d array, d >= 2")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0):
        raise ValueError("Zero vectors have undefined orientation")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative and have positive total")
    return x / norms[:, None], w / w.sum()

def moments(vectors, weights=None):
    u, w = directions(vectors, weights)
    mean = w @ u
    second = np.einsum("n,ni,nj->ij", w, u, u)
    return mean, second - np.eye(u.shape[1]) / u.shape[1]

def tensor_moment(vectors, order, weights=None):
    if order not in (1, 2, 3, 4):
        raise ValueError("Explicit tensor implementation supports orders 1 through 4")
    u, w = directions(vectors, weights)
    terms = u
    for _ in range(1, order):
        terms = np.einsum("ni,nj->nij", terms, u).reshape(len(u), -1)
    return (w @ terms).reshape((u.shape[1],) * order)

def spherical_kernel(t, order, dimension):
    """Degree-order zonal harmonic normalized to H(1)=1."""
    if dimension < 2 or int(dimension) != dimension or order < 0 or int(order) != order:
        raise ValueError("Integer dimension >= 2 and nonnegative integer order required")
    t = np.clip(np.asarray(t, dtype=float), -1, 1)
    if dimension == 2:
        return np.cos(order * np.arccos(t))
    alpha = (dimension - 2) / 2
    return eval_gegenbauer(order, alpha, t) / eval_gegenbauer(order, alpha, 1.)

def angular_kernel_mean(a, b, order, wa=None, wb=None):
    u, p = directions(a, wa)
    v, q = directions(b, wb)
    if u.shape[1] != v.shape[1]:
        raise ValueError("Configurations require the same feature dimension")
    return float(p @ spherical_kernel(u @ v.T, order, u.shape[1]) @ q)

def angular_distance(a, b, order, wa=None, wb=None):
    """Squared distance between degree-order mean harmonic embeddings."""
    value = (angular_kernel_mean(a, a, order, wa, wa)
             + angular_kernel_mean(b, b, order, wb, wb)
             - 2 * angular_kernel_mean(a, b, order, wa, wb))
    if value < -1e-10:
        raise ArithmeticError("Unexpected negative kernel distance")
    return max(0., value)

def circular_moments(theta, weights=None, orders=(1, 2, 4)):
    theta = np.asarray(theta, dtype=float).ravel()
    u, w = directions(np.column_stack([np.cos(theta), np.sin(theta)]), weights)
    return np.array([w @ (u[:, 0] + 1j*u[:, 1])**k for k in orders])

def periodic_correlation(field, lag):
    """Uncentered, directed matrix correlation, with periodic sequence boundary."""
    x = np.asarray(field, dtype=float)
    return x.T @ np.roll(x, -lag, axis=0) / len(x)

def power(image, center=True):
    x = np.asarray(image, dtype=float)
    if center:
        x = x - x.mean()
    return np.abs(np.fft.fftn(x))**2 / x.size

def autocorrelation(image, center=True):
    return np.fft.ifftn(power(image, center=center)).real

def bispectrum(sequence):
    """Uncentered cyclic third-order spectrum, normalized by N."""
    x = np.asarray(sequence, dtype=float)
    f = np.fft.fft(x)
    indices = np.arange(len(x))
    return f[:, None] * f[None, :] * f[(indices[:, None] + indices) % len(x)].conj() / len(x)

def relative_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = np.linalg.norm(a) + np.linalg.norm(b)
    return float(np.linalg.norm(a-b)/scale) if scale > 1e-12 else 0.

def similarity(a, b):
    """Bounded descriptive agreement; equality maps to 1, maximum distance to 0."""
    return 1 - relative_distance(a, b)
