"""Numerical equations and integration routines for the local EMS model."""
from functools import lru_cache
import numpy as np
import sympy as sp
from .conformal import symbolic_system

@lru_cache(maxsize=4)
def compiled_step(coupling, representation='potential'):
    from numba import njit
    system = symbolic_system(coupling)
    if representation == 'normalized':
        from .normalized import normalized_system
        system = normalized_system(coupling)
    elif representation != 'potential':
        raise ValueError('unsupported field representation')
    force = njit(sp.lambdify((system['fields'], system['vt'], system['vy'], system['charges']), list(system['acceleration']), 'numpy', cse=True))
    auxiliary = njit(sp.lambdify((system['fields'], system['charges']), list(system['auxiliary']), 'numpy', cse=True))

    @njit
    def fast_rhs(state, spacing, charges):
        result = np.zeros_like(state)
        for j in range(state.shape[1]):
            gradient, second = (np.zeros(7), np.zeros(7))
            if 2 <= j < state.shape[1] - 2:
                for i in range(7):
                    gradient[i] = (8 * (state[i, j + 1] - state[i, j - 1]) - (state[i, j + 2] - state[i, j - 2])) / (12 * spacing)
                    second[i] = (16 * (state[i, j - 1] - state[i, j] + (state[i, j + 1] - state[i, j])) - (state[i, j - 2] - state[i, j]) - (state[i, j + 2] - state[i, j])) / (12 * spacing ** 2)
            f = force(state[:7, j], state[7:14, j], gradient, charges)
            u = auxiliary(state[:7, j], charges)
            for i in range(7):
                result[i, j] = state[i + 7, j]
                result[i + 7, j] = f[i] + second[i]
            for i in range(3):
                result[i + 14, j] = u[i]
        return result

    @njit
    def step(state, dt, spacing, charges):
        k1 = fast_rhs(state, spacing, charges)
        k2 = fast_rhs(state + dt * k1 / 2, spacing, charges)
        k3 = fast_rhs(state + dt * k2 / 2, spacing, charges)
        k4 = fast_rhs(state + dt * k3, spacing, charges)
        return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return step
