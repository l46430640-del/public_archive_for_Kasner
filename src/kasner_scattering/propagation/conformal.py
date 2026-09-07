"""Numerical equations and integration routines for the local EMS model."""
from __future__ import annotations
from functools import lru_cache
import numpy as np
import sympy as sp
FIELD_NAMES = ('log_area', 'sigma', 'P', 'Q', 'psi', 'a1', 'a2')

@lru_cache(maxsize=2)
def symbolic_system(coupling: str='exp_square') -> dict:
    r, sigma, p, shape_q, psi, a1, a2 = x = sp.symbols('r sigma P Q psi a1 a2', real=True)
    electric, j1, j2 = charges = sp.symbols('charge J1 J2', real=True)
    vt = sp.Matrix(sp.symbols('v0:7', real=True))
    vy = sp.Matrix(sp.symbols('w0:7', real=True))
    rho = sp.exp(r)
    if coupling not in {'exp_square', 'constant'}:
        raise ValueError('unsupported coupling')
    z = sp.exp(psi ** 2) if coupling == 'exp_square' else sp.Integer(1)
    shape = sp.Matrix([[sp.exp(p), sp.exp(p) * shape_q], [sp.exp(p) * shape_q, sp.exp(-p) + sp.exp(p) * shape_q ** 2]])
    inverse_shape = shape.inv().applyfunc(sp.simplify)
    kinetic = sp.zeros(7)
    kinetic[0, 0] = -rho
    kinetic[0, 1] = kinetic[1, 0] = -2 * rho
    kinetic[2, 2] = rho
    kinetic[3, 3] = rho * sp.exp(2 * p)
    kinetic[4, 4] = 4 * rho
    kinetic[5:7, 5:7] = 4 * z * inverse_shape
    twist_momentum = sp.Matrix([j1 - electric * a1, j2 - electric * a2])
    potential = sp.exp(2 * sigma) * (electric ** 2 / (8 * rho * z) + (twist_momentum.T * inverse_shape * twist_momentum)[0] / (2 * rho ** 2))
    derivatives = [kinetic.diff(value) for value in x]
    force = sp.Matrix([(vt.T * derivatives[i] * vt - vy.T * derivatives[i] * vy)[0] / 2 - sum((derivatives[k][i, j] * (vt[k] * vt[j] - vy[k] * vy[j]) for j in range(7) for k in range(7)), sp.Integer(0)) - sp.diff(potential, x[i]) for i in range(7)])
    acceleration = (kinetic.inv() * force).applyfunc(sp.simplify)
    twist_velocity = sp.exp(2 * sigma - 2 * r) * inverse_shape * twist_momentum
    base_velocity = electric * sp.exp(2 * sigma - r) / (4 * z)
    base_velocity -= (sp.Matrix([a1, a2]).T * twist_velocity)[0]
    auxiliary = sp.Matrix([*twist_velocity, base_velocity])
    return {'fields': x, 'charges': charges, 'vt': vt, 'vy': vy, 'kinetic': kinetic, 'potential': potential, 'shape': shape, 'acceleration': acceleration, 'auxiliary': auxiliary}

@lru_cache(maxsize=2)
def compiled_system(coupling: str='exp_square') -> tuple:
    system = symbolic_system(coupling)
    args = (*system['fields'], *system['vt'], *system['vy'], *system['charges'])
    force = [sp.lambdify(args, value, 'numpy', cse=True) for value in system['acceleration']]
    aux_args = (*system['fields'], *system['charges'])
    auxiliary = [sp.lambdify(aux_args, value, 'numpy', cse=True) for value in system['auxiliary']]
    return (force, auxiliary)

def nonlinear_force(position: np.ndarray, velocity: np.ndarray, gradient: np.ndarray, charges=(0.0, 0.0, 0.0), coupling='exp_square') -> np.ndarray:
    funcs, _ = compiled_system(coupling)
    arguments = (*position, *velocity, *gradient, *charges)
    return np.stack([np.broadcast_to(fn(*arguments), position.shape[1:]) for fn in funcs])

def auxiliary_velocity(position: np.ndarray, charges=(0.0, 0.0, 0.0), coupling='exp_square') -> np.ndarray:
    _, funcs = compiled_system(coupling)
    args = (*position, *charges)
    return np.stack([np.broadcast_to(fn(*args), position.shape[1:]) for fn in funcs])

def null_constraints(position: np.ndarray, velocity: np.ndarray, gradient: np.ndarray, acceleration: np.ndarray, mixed: np.ndarray, second_space: np.ndarray, coupling='exp_square') -> np.ndarray:
    """Return C_+ and C_- divided by area, with D_+/-=dT+/-dy."""
    r, sigma, p, shape_q, psi, _, _ = position
    z = np.exp(psi ** 2) if coupling == 'exp_square' else np.ones_like(psi)
    sinv00 = np.exp(-p) + np.exp(p) * shape_q ** 2
    sinv01 = -np.exp(p) * shape_q
    sinv11 = np.exp(p)
    output = []
    for sign in (1, -1):
        d = velocity + sign * gradient
        r_dd = acceleration[0] + 2 * sign * mixed[0] + second_space[0]
        matter = sinv00 * d[5] ** 2 + 2 * sinv01 * d[5] * d[6] + sinv11 * d[6] ** 2
        output.append(r_dd + 0.5 * d[0] ** 2 - 2 * d[1] * d[0] + 0.5 * (d[2] ** 2 + np.exp(2 * p) * d[3] ** 2) + 2 * d[4] ** 2 + 2 * z * np.exp(-r) * matter)
    return np.stack(output)

def homogeneous_constraint_velocity(position: np.ndarray, velocity: np.ndarray, charges=(0.0, 0.0, 0.0), coupling='exp_square') -> np.ndarray:
    result = np.asarray(velocity, dtype=float).copy()
    if abs(result[0]) < 1e-14:
        raise ValueError('area velocity cannot vanish in this initial-data chart')
    result[1] = 0
    zero = np.zeros_like(result)
    acc = nonlinear_force(position, result, zero, charges, coupling)
    residual = null_constraints(position, result, zero, acc, zero, zero, coupling)[0]
    result[1] = residual / (2 * result[0])
    return result

def geometric_spectrum(position: np.ndarray, velocity: np.ndarray, auxiliary: np.ndarray, charges=(0.0, 0.0, 0.0), coupling='exp_square') -> np.ndarray:
    """Kasner eigenvalues on constant-T slices; scalar is sqrt(2)*n(psi)/|K|."""
    r, sigma, p, shape_q, _, _, _ = position
    rho = np.exp(r)
    shape = np.array([[np.exp(p), np.exp(p) * shape_q], [np.exp(p) * shape_q, np.exp(-p) + np.exp(p) * shape_q ** 2]])
    orbit = rho * shape
    shape_dot = np.array([[np.exp(p) * velocity[2], np.exp(p) * (velocity[2] * shape_q + velocity[3])], [np.exp(p) * (velocity[2] * shape_q + velocity[3]), -np.exp(-p) * velocity[2] + np.exp(p) * (velocity[2] * shape_q ** 2 + 2 * shape_q * velocity[3])]])
    orbit_dot = rho * (shape_dot + velocity[0] * shape)
    c_dot = auxiliary_velocity(position, charges, coupling)[:2]
    orbit_frame = np.linalg.inv(np.linalg.cholesky(orbit).T)
    rate = np.zeros((3, 3))
    rate[0, 0] = velocity[1]
    rate[0, 1:] = rate[1:, 0] = 0.5 * np.exp(-sigma) * orbit_frame.T @ orbit @ c_dot
    rate[1:, 1:] = 0.5 * orbit_frame.T @ orbit_dot @ orbit_frame
    trace = velocity[0] + velocity[1]
    if trace == 0:
        raise ValueError('Kasner spectrum is undefined on a zero-expansion slice')
    spatial = np.sort(np.linalg.eigvalsh(rate) / trace)
    return np.append(spatial, np.sqrt(2) * velocity[4] / abs(trace))
