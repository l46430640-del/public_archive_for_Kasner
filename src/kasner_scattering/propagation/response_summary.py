"""Numerical equations and integration routines for the local EMS model."""
import numpy as np

def sample(values, points, wanted):
    h = points[1] - points[0]
    indices = np.rint((wanted - points[0]) / h).astype(int)
    if np.max(abs(points[indices] - wanted)) > 1e-08:
        raise ValueError('comparison requires shared mesh nodes')
    return values[..., indices]
