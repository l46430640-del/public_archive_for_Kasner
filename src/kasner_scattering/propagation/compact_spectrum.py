"""Numerical equations and integration routines for the local EMS model."""
from functools import lru_cache
import mpmath as mp
import numpy as np
from scipy.special import roots_legendre
import sympy as sp

def transform(frequency, nodes=512):
    frequency = np.atleast_1d(frequency)
    points, weights = roots_legendre(nodes)
    weights *= np.exp(1 - 1 / (1 - points ** 2))
    output = np.empty_like(frequency, dtype=float)
    for start in range(0, len(frequency), 256):
        output[start:start + 256] = np.cos(np.outer(frequency[start:start + 256], points)) @ weights
    return output

def transform_mp(frequency, digits=50):
    with mp.workdps(digits):
        frequency = mp.mpf(str(frequency))
        count = max(32, int(abs(frequency) / 4))

        def integrand(x):
            return mp.exp(1 - 1 / (1 - x * x)) * mp.cos(frequency * x) if x < 1 else mp.mpf(0)
        return 2 * mp.quad(integrand, [mp.mpf(i) / count for i in range(count + 1)])

@lru_cache(maxsize=4)
def derivative_norm(order=6):
    """Total variation of f^(n-1), using isolated polynomial roots of f^n.

    The reported upper value is rounded upward by more than one unit after
    60-digit evaluation. Root brackets have width at most 1e-50.
    """
    if order not in (4, 6, 8):
        raise ValueError('unsupported derivative-norm order')
    x = sp.Symbol('x')
    log_derivative = -2 * x / (1 - x * x) ** 2
    rational = sp.Integer(1)
    previous = rational
    for _ in range(order):
        previous, rational = (rational, sp.cancel(sp.diff(rational, x) + log_derivative * rational))
    numerator, _ = sp.fraction(rational)
    brackets = sp.polys.polytools.intervals(numerator, eps=sp.Rational(1, 10 ** 50))
    previous_fn = sp.lambdify(x, previous, 'mpmath')
    with mp.workdps(60):
        values = [mp.mpf(0)]
        for (left, right), _ in brackets:
            if left <= -1 or right >= 1:
                continue
            midpoint = (left + right) / 2
            point = mp.mpf(str(sp.N(midpoint, 65)))
            values.append(previous_fn(point) * mp.exp(1 - 1 / (1 - point * point)))
        values.append(mp.mpf(0))
        norm = sum((abs(a - b) for a, b in zip(values, values[1:])))
        return {'order': order, 'norm_estimate': float(norm), 'upward_rounded_norm': float(mp.ceil(norm) + 1), 'interior_critical_points': len(values) - 2, 'root_bracket_width': 1e-50, 'evaluation_digits': 60}

def transform_tail(cutoff, order=6):
    norm = derivative_norm(order)['upward_rounded_norm']
    return (norm / (np.pi * (order - 1) * cutoff ** (order - 1)), norm / (np.pi * (order - 2) * cutoff ** (order - 2)))
