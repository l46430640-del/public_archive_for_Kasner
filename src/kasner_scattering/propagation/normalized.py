"""Numerical equations and integration routines for the local EMS model."""
from functools import lru_cache
import numpy as np
import sympy as sp
from .conformal import symbolic_system

@lru_cache(maxsize=2)
def normalized_system(coupling='exp_square'):
    original = symbolic_system(coupling)
    if coupling == 'constant':
        return original
    x, v, w = (original['fields'], original['vt'], original['vy'])
    psi = x[4]
    factor = sp.exp(-psi ** 2 / 2)
    mapping = sp.Matrix([*x[:5], factor * x[5], factor * x[6]])
    jac = mapping.jacobian(x)
    old_v, old_w = (jac * v, jac * w)
    replacements = dict(zip((*x, *v, *w), (*mapping, *old_v, *old_w)))
    force = original['acceleration'].subs(replacements, simultaneous=True)
    hessian_terms = sp.Matrix([(v.T * sp.hessian(value, x) * v - w.T * sp.hessian(value, x) * w)[0] for value in mapping])
    stable = lambda value: sp.powsimp(sp.expand(value), combine='exp', force=True)
    new_force = (jac.inv() * (force - hessian_terms)).applyfunc(stable)
    auxiliary = original['auxiliary'].subs(dict(zip(x, mapping)), simultaneous=True).applyfunc(stable)
    return {**original, 'acceleration': new_force, 'auxiliary': auxiliary, 'mapping': mapping, 'mapping_jacobian': jac, 'kinetic': (jac.T * original['kinetic'].subs(dict(zip(x, mapping)), simultaneous=True) * jac).applyfunc(sp.simplify), 'potential': original['potential'].subs(dict(zip(x, mapping)), simultaneous=True)}

def to_normalized(state, coupling='exp_square'):
    result = state.copy()
    if coupling == 'constant':
        return result
    psi, psi_t = (state[4], state[11])
    factor = np.exp(psi ** 2 / 2)
    result[5:7] = factor * state[5:7]
    result[12:14] = factor * (state[12:14] + psi * psi_t * state[5:7])
    return result

def to_potential(state, coupling='exp_square'):
    result = state.copy()
    if coupling == 'constant':
        return result
    psi, psi_t = (state[4], state[11])
    factor = np.exp(-psi ** 2 / 2)
    result[5:7] = factor * state[5:7]
    result[12:14] = factor * (state[12:14] - psi * psi_t * state[5:7])
    return result
