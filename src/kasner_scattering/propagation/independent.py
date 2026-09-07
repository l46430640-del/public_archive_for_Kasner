"""Numerical equations and integration routines for the local EMS model."""
from math import factorial
import numpy as np
import sympy as sp
from .conformal import symbolic_system
from .four_dimensional import residuals

def weights(order):
    offsets = np.arange(-3, 4, dtype=float)
    target = np.zeros(7)
    target[order] = factorial(order)
    return np.linalg.solve(np.stack([offsets ** power for power in range(7)]), target)

def sixth_derivatives(values, h):
    first, second = (np.zeros_like(values), np.zeros_like(values))
    center = values[..., 3:-3]
    for offset, w1, w2 in zip(range(-3, 4), weights(1), weights(2)):
        shifted = values[..., 3 + offset:values.shape[-1] - 3 + offset] - center
        first[..., 3:-3] += w1 * shifted / h
        second[..., 3:-3] += w2 * shifted / h ** 2
    return (first, second)

def point_residual(solution, n, h, charges, time=1.0, time_spacing=0.01, residual_evaluator=None):
    center = n // 2
    times = time + np.arange(-3, 4) * time_spacing
    sampled = solution.sol(times).T.reshape(7, 17, n)[:, :, center - 3:center + 4]
    full = np.concatenate([sampled[:, :7], sampled[:, 14:]], axis=1)
    w1, w2 = (weights(1), weights(2))
    first = np.zeros((4, 10))
    second = np.zeros((4, 4, 10))
    first[0, :7] = sampled[3, 7:14, 3]
    first[0, 7:] = np.einsum('i,ij->j', w1, full[:, 7:, 3] - full[3, 7:, 3]) / time_spacing
    first[1] = np.einsum('i,ji->j', w1, full[3] - full[3, :, 3, None]) / h
    second[0, 0, :7] = np.einsum('i,ij->j', w1, sampled[:, 7:14, 3] - sampled[3, 7:14, 3]) / time_spacing
    second[0, 0, 7:] = np.einsum('i,ij->j', w2, full[:, 7:, 3] - full[3, 7:, 3]) / time_spacing ** 2
    second[1, 1] = np.einsum('i,ji->j', w2, full[3] - full[3, :, 3, None]) / h ** 2
    second[0, 1, :7] = second[1, 0, :7] = np.einsum('i,ji->j', w1, sampled[3, 7:14] - sampled[3, 7:14, 3, None]) / h
    mixed_difference = full[:, 7:] - full[3, 7:] - full[:, 7:, 3, None] + full[3, 7:, 3, None]
    second[0, 1, 7:] = second[1, 0, 7:] = np.einsum('i,j,ikj->k', w1, w1, mixed_difference) / (h * time_spacing)
    point = full[3, :, 3].copy()
    point[7:] = 0.0
    first[1, 7:] = 0.0
    second[1, 1, 7:] = 0.0
    evaluator = residuals if residual_evaluator is None else residual_evaluator
    result = evaluator(point, first, second)
    output = {key: value for key, value in result.items() if np.isscalar(value)}
    system = symbolic_system()
    v, w, kinetic = (system['vt'], system['vy'], system['kinetic'])
    energy = (v.T * kinetic * v + w.T * kinetic * w)[0] / 2 + system['potential']
    flux = (v.T * kinetic * w)[0]
    energy_fn, flux_fn = [sp.lambdify((*system['fields'], *v, *w, *system['charges']), expr, 'numpy', cse=True) for expr in (energy, flux)]
    full_sample = solution.sol(times).T.reshape(7, 17, n)
    energy_samples, scalar_energies, em_energies = ([], [], [])
    scalar_fluxes, em_fluxes = ([], [])
    for sample in full_sample:
        gradient, _ = sixth_derivatives(sample[:7], h)
        args = (*sample[:7], *sample[7:14], *gradient, *charges)
        energy_samples.append(energy_fn(*args)[center])
        rho, z = (np.exp(sample[0]), np.exp(sample[4] ** 2))
        sinv00 = np.exp(-sample[2]) + np.exp(sample[2]) * sample[3] ** 2
        sinv01, sinv11 = (-np.exp(sample[2]) * sample[3], np.exp(sample[2]))
        at, ay = (sample[12:14], gradient[5:7])

        def product(a, b):
            return sinv00 * a[0] * b[0] + sinv01 * (a[0] * b[1] + a[1] * b[0]) + sinv11 * a[1] * b[1]
        scalar_energies.append((2 * rho * (sample[11] ** 2 + gradient[4] ** 2))[center])
        em_energies.append((2 * z * (product(at, at) + product(ay, ay)))[center])
        scalar_fluxes.append(4 * rho * sample[11] * gradient[4])
        em_fluxes.append(4 * z * product(at, ay))
    sample = full_sample[3]
    gradient, _ = sixth_derivatives(sample[:7], h)
    flux_samples = flux_fn(*sample[:7], *sample[7:14], *gradient, *charges)[center - 3:center + 4]
    et, fy = (w1 @ energy_samples / time_spacing, w1 @ flux_samples / h)
    output['reduced_energy_balance_absolute'] = float(abs(et - fy))
    output['reduced_energy_balance_relative'] = None
    q, velocity, spatial = (sample[:7, center], sample[7:14, center], gradient[:, center])
    replacements = dict(zip((*system['fields'], *system['charges']), (*q, *charges)))
    inverse_shape = system['shape'].inv()
    sinv = np.asarray(inverse_shape.subs(replacements), float)
    sinv_t = np.asarray(sum((inverse_shape.diff(field) * velocity[i] for i, field in enumerate(system['fields'])), sp.zeros(2)).subs(replacements), float)
    potential_gradient = np.array([float(sp.diff(system['potential'], field).subs(replacements)) for field in system['fields']])
    at, ay, z, rho = (velocity[5:7], spatial[5:7], np.exp(q[4] ** 2), np.exp(q[0]))
    exchange = 4 * q[4] * z * velocity[4] * (at @ sinv @ at - ay @ sinv @ ay)
    scalar_work = 2 * rho * velocity[0] * (spatial[4] ** 2 - velocity[4] ** 2) + exchange - velocity[4] * potential_gradient[4]
    em_work = -exchange - 2 * z * (at @ sinv_t @ at - ay @ sinv_t @ ay) - at @ potential_gradient[5:7]
    scalar_balance = w1 @ scalar_energies / time_spacing - w1 @ scalar_fluxes[3][center - 3:center + 4] / h
    em_balance = w1 @ em_energies / time_spacing - w1 @ em_fluxes[3][center - 3:center + 4] / h
    output['scalar_maxwell_exchange'] = float(exchange)
    output['scalar_energy_balance_absolute'] = float(abs(scalar_balance - scalar_work))
    output['maxwell_energy_balance_absolute'] = float(abs(em_balance - em_work))
    output['scalar_work'] = float(scalar_work)
    output['maxwell_work'] = float(em_work)
    output['scalar_energy_balance_relative'] = float(abs(scalar_balance - scalar_work) / max(abs(scalar_work), 1e-300))
    output['maxwell_energy_balance_relative'] = float(abs(em_balance - em_work) / max(abs(em_work), 1e-300))
    return output
