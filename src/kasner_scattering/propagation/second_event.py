"""Numerical equations and integration routines for the local EMS model."""
from .io import save_exchange as save, EXCHANGE_OUTPUT as OUTPUT
from dataclasses import asdict
import json
from time import perf_counter
import numpy as np
from .accelerated import compiled_step
from .conformal import geometric_spectrum, nonlinear_force, null_constraints
from .contracts import EvolutionResult, PacketSpec
from .evolution import derivatives
from .inputs import file_hash
from .response_experiment import frozen_case, grid_for, nonlinear_initial
POINTS = np.array([-34.8, -17.6, 0.0, 17.6, 34.8])

def diagonal_spectrum(v):
    trace = v[0] + v[1]
    return np.stack([v[1] / trace, (v[0] + v[2]) / (2 * trace), (v[0] - v[2]) / (2 * trace), np.sqrt(2) * v[4] / abs(trace)])

def diagonal_rate(v, acc):
    trace, at = (v[0] + v[1], acc[0] + acc[1])
    return np.stack([(acc[1] * trace - v[1] * at) / trace ** 2, ((acc[0] + acc[2]) * trace - (v[0] + v[2]) * at) / (2 * trace ** 2), ((acc[0] - acc[2]) * trace - (v[0] - v[2]) * at) / (2 * trace ** 2), np.sqrt(2) / abs(trace) * (acc[4] - v[4] * at / trace)])

def observables(state, spacing, indices, charges, coupling='exp_square'):
    x, v = (state[:7], state[7:14])
    grad, second = derivatives(x, spacing)
    mixed, _ = derivatives(v, spacing)
    x, v, grad, second, mixed = (a[:, indices] for a in (x, v, grad, second, mixed))
    local_grad = np.zeros_like(grad)
    local_grad[5:] = grad[5:]
    homogeneous = nonlinear_force(x, v, local_grad, charges, coupling)
    force = nonlinear_force(x, v, grad, charges, coupling)
    vacuum_v = v.copy()
    vacuum_v[5:] = 0.0
    vacuum = nonlinear_force(x, vacuum_v, np.zeros_like(grad), (0.0, 0.0, 0.0), coupling)
    trace = v[0] + v[1]
    z = np.exp(x[4] ** 2) if coupling == 'exp_square' else np.ones_like(trace)
    inverse00 = np.exp(-x[2]) + np.exp(x[2]) * x[3] ** 2
    inverse01, inverse11 = (-np.exp(x[2]) * x[3], np.exp(x[2]))

    def em(vector):
        return z * np.exp(-x[0]) * (inverse00 * vector[5] ** 2 + 2 * inverse01 * vector[5] * vector[6] + inverse11 * vector[6] ** 2) / trace ** 2
    spectra = np.array([geometric_spectrum(x[:, k], v[:, k], state[14:, j], charges, coupling) for k, j in enumerate(indices)])
    diagonal = diagonal_spectrum(v).T
    ordered = np.c_[np.sort(diagonal[:, :3], axis=1), diagonal[:, 3]]
    constraints = null_constraints(x, v, grad, second + force, mixed, second, coupling)
    return {'spectrum': spectra, 'axes_spectrum': diagonal, 'volume': -(x[0] + x[1]), 'magnetic': em(grad), 'electric': em(v), 'spatial_fraction': (grad[2] ** 2 / 4 + np.exp(2 * x[2]) * grad[3] ** 2 / 4 + grad[4] ** 2) / trace ** 2, 'matter_rate': diagonal_rate(v, homogeneous - vacuum).T, 'spatial_rate': diagonal_rate(v, second + force - homogeneous).T, 'vacuum_rate': diagonal_rate(v, vacuum).T, 'total_rate': diagonal_rate(v, second + force).T, 'constraint': np.max(abs(constraints), axis=0) / trace ** 2, 'axis_error': np.max(abs(spectra - ordered), axis=1), 'state': state[:, indices].T}

def run(label, spacing=0.4, cfl=0.4, end=340.0, domain_factor=4.0, coupling='exp_square'):
    start = perf_counter()
    protocol = json.loads((OUTPUT / 'protocol.json').read_text())
    if end not in protocol['windows']:
        raise ValueError('unregistered end time')
    _, case = frozen_case(label)
    grid, _ = grid_for(spacing, end, domain_factor)
    indices = np.rint((POINTS - grid[0]) / spacing).astype(int)
    if np.max(abs(grid[indices] - POINTS)) > 1e-08:
        raise ValueError('observation curves must be exact common mesh nodes')
    state, charges = nonlinear_initial(grid, case['mixing'], coupling=coupling)
    initial = state.copy()
    count = int(np.ceil(end / (cfl * spacing)))
    dt = end / count
    wanted = set(np.linspace(0, count, int(np.ceil(end / 0.25)) + 1, dtype=int))
    stepper = compiled_step(coupling)
    histories, times, failure = ({}, [], None)
    checkpoint_index = int(np.floor(220.0 / dt))
    checkpoint = None
    for i in range(count + 1):
        if i == checkpoint_index:
            checkpoint = state.copy()
        if i in wanted:
            obs = observables(state, spacing, indices, charges, coupling)
            for key, value in obs.items():
                histories.setdefault(key, []).append(value)
            times.append(i * dt)
        if i != count:
            candidate = stepper(state, dt, spacing, charges)
            if not np.isfinite(candidate).all():
                failure = {'last_finite_time': i * dt, 'invalid_time': (i + 1) * dt, 'invalid_indices': np.argwhere(~np.isfinite(candidate))[:20].tolist()}
                break
            state = candidate
    name = f'event_{label}_h{spacing:g}_c{cfl:g}_T{end:g}_L{domain_factor:g}_{coupling}'
    result = EvolutionResult(status='NUMERICAL_FAILURE' if failure else 'FINITE_EVENT_HISTORY_NO_PLATFORM_ACCEPTANCE', model='constrained_two_Killing_EMS', diagnostics={'packet': asdict(PacketSpec(1e-06, envelope='compact', input_spectrum={'mixing': case['mixing'], 'soft_frequency_ratio': 0.01})), 'spacing': spacing, 'step': dt, 'end_time': end, 'half_domain': float(grid[-1]), 'charges': list(charges), 'failure': failure, 'coupling': coupling}, validity={'complete_horizon_input': False, 'local_constraints_constructed': True, 'platform_analysis_pending': True}).to_dict()
    result['runtime_seconds'] = perf_counter() - start
    result['source_partition_version'] = 2
    result['checkpoint_time'] = checkpoint_index * dt if checkpoint is not None else None
    result['protocol_sha256'] = file_hash(OUTPUT / 'protocol.json')
    extra = {'checkpoint': checkpoint} if checkpoint is not None else {}
    return save(name, result, grid=grid, initial=initial, final=state, points=POINTS, **extra, times=np.array(times), **{key: np.array(value) for key, value in histories.items()})
