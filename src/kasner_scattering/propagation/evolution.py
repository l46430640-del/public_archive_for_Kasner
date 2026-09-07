"""Numerical equations and integration routines for the local EMS model."""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_simpson
from .conformal import auxiliary_velocity, nonlinear_force, null_constraints

def derivatives(values, spacing):
    first, second = (np.zeros_like(values), np.zeros_like(values))
    first[..., 2:-2] = (8 * (values[..., 3:-1] - values[..., 1:-3]) - (values[..., 4:] - values[..., :-4])) / (12 * spacing)
    center = values[..., 2:-2]
    second[..., 2:-2] = (16 * (values[..., 1:-3] - center + (values[..., 3:-1] - center)) - (values[..., :-4] - center) - (values[..., 4:] - center)) / (12 * spacing ** 2)
    return (first, second)

def constraint_initial_data(grid, position, velocity, charges=(0.0, 0.0, 0.0), coupling='exp_square'):
    """Solve both constraints with spatially constant initial area and area rate.

    The spatial conformal factor comes from the momentum constraint, not from
    a fictitious extra matter potential. Its central value fixes the scale.
    """
    x, v = (np.array(position, float, copy=True), np.array(velocity, float, copy=True))
    if np.ptp(x[0]) > 1e-14 or np.ptp(v[0]) > 1e-14 or np.min(np.abs(v[0])) < 1e-12:
        raise ValueError('this data chart requires constant nonzero area rate')
    h = grid[1] - grid[0]
    grad, _ = derivatives(x, h)
    r, _, p, q, psi, _, _ = x
    sinv00, sinv01, sinv11 = (np.exp(-p) + np.exp(p) * q ** 2, -np.exp(p) * q, np.exp(p))
    z = np.exp(psi ** 2) if coupling == 'exp_square' else np.ones_like(psi)
    flux = v[2] * grad[2] + np.exp(2 * p) * v[3] * grad[3] + 4 * v[4] * grad[4] + 4 * z * np.exp(-r) * (sinv00 * v[5] * grad[5] + sinv01 * (v[5] * grad[6] + v[6] * grad[5]) + sinv11 * v[6] * grad[6])
    sigma_y = flux / (2 * v[0])
    primitive = cumulative_simpson(sigma_y, x=grid, initial=0)
    x[1] += primitive - primitive[len(grid) // 2]
    grad, second = derivatives(x, h)
    v[1] = 0
    mixed, _ = derivatives(v, h)
    acc = second + nonlinear_force(x, v, grad, charges, coupling)
    constraint = null_constraints(x, v, grad, acc, mixed, second, coupling).mean(axis=0)
    v[1] = constraint / (2 * v[0])
    return np.concatenate([x, v, np.zeros((3, len(grid)))])

def rhs(state, spacing, charges=(0.0, 0.0, 0.0), coupling='exp_square'):
    position, velocity = (state[:7], state[7:14])
    grad, second = derivatives(position, spacing)
    return np.concatenate([velocity, second + nonlinear_force(position, velocity, grad, charges, coupling), auxiliary_velocity(position, charges, coupling)])

def rk4_step(state, step, spacing, charges=(0.0, 0.0, 0.0), coupling='exp_square'):
    k1 = rhs(state, spacing, charges, coupling)
    k2 = rhs(state + step * k1 / 2, spacing, charges, coupling)
    k3 = rhs(state + step * k2 / 2, spacing, charges, coupling)
    k4 = rhs(state + step * k3, spacing, charges, coupling)
    return state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def diagnostics(state, spacing, mask, charges=(0.0, 0.0, 0.0), coupling='exp_square'):
    x, v = (state[:7], state[7:14])
    grad, second = derivatives(x, spacing)
    mixed, _ = derivatives(v, spacing)
    acc = second + nonlinear_force(x, v, grad, charges, coupling)
    constraints = null_constraints(x, v, grad, acc, mixed, second, coupling)
    trace = v[0] + v[1]
    z = np.exp(x[4] ** 2) if coupling == 'exp_square' else np.ones_like(x[4])
    sinv00 = np.exp(-x[2]) + np.exp(x[2]) * x[3] ** 2
    sinv01, sinv11 = (-np.exp(x[2]) * x[3], np.exp(x[2]))
    contractions = [sinv00 * vec[5] ** 2 + 2 * sinv01 * vec[5] * vec[6] + sinv11 * vec[6] ** 2 for vec in (v, grad)]
    energy = z * np.exp(-x[0]) * sum(contractions) / trace ** 2
    return {'max_null_constraint': float(np.max(np.abs(constraints[:, mask]) / trace[mask] ** 2)), 'max_transverse_em_fraction': float(np.max(energy[mask])), 'min_area': float(np.min(np.exp(x[0, mask])))}

@dataclass
class Run:
    grid: np.ndarray
    state: np.ndarray
    snapshots: list
    step: float
    steps: int

class EvolutionFailure(FloatingPointError):

    def __init__(self, time, invalid_time, run, invalid_indices):
        super().__init__(f'nonfinite state at time {invalid_time:g}')
        self.last_finite_time = time
        self.invalid_time = invalid_time
        self.run = run
        self.invalid_indices = invalid_indices

def evolve(grid, initial, end_time, step, observation_radius, charges=(0.0, 0.0, 0.0), coupling='exp_square', samples=9, accelerated=True, representation='potential'):
    h = grid[1] - grid[0]
    if end_time <= 0 or step <= 0 or observation_radius < 0:
        raise ValueError('invalid evolution domain')
    if step / h > 0.6:
        raise ValueError('requested CFL exceeds the tested range')
    if min(-grid[0], grid[-1]) <= observation_radius + end_time + 20 * h:
        raise ValueError('observation region lacks a protected causal past')
    count = int(np.ceil(end_time / step))
    dt = end_time / count
    snapshots, state = ([], initial.copy())
    physical = lambda value: value
    if representation == 'normalized':
        if not accelerated:
            raise ValueError('normalized evolution currently requires the compiled stepper')
        from .normalized import to_normalized, to_potential
        state = to_normalized(state, coupling)
        physical = lambda value: to_potential(value, coupling)
    elif representation != 'potential':
        raise ValueError('unsupported field representation')
    stepper = rk4_step
    if accelerated:
        from .accelerated import compiled_step
        fast = compiled_step(coupling, representation)
        stepper = lambda state, dt, h, charges, _: fast(state, dt, h, charges)
    wanted = set(np.linspace(0, count, samples, dtype=int))
    mask = np.abs(grid) <= observation_radius + h / 2
    for index in range(count + 1):
        if index in wanted:
            output = physical(state)
            snapshots.append({'time': index * dt, **diagnostics(output, h, mask, charges, coupling), 'center_state': output[:, len(grid) // 2].tolist()})
        if index < count:
            candidate = stepper(state, dt, h, charges, coupling)
            if not np.all(np.isfinite(candidate)):
                raise EvolutionFailure(index * dt, (index + 1) * dt, Run(grid, physical(state), snapshots, dt, index), np.argwhere(~np.isfinite(candidate))[:20].tolist())
            state = candidate
    return Run(grid, physical(state), snapshots, dt, count)
