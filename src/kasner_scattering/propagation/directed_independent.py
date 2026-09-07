"""Numerical equations and integration routines for the local EMS model."""
from time import perf_counter
import numpy as np
from scipy.integrate import DOP853
from .conformal import auxiliary_velocity, nonlinear_force
from .directed import save
from .directed_analysis import load
from .independent import point_residual, weights
from .independent_segment import SavedSlices
from .second_event import diagonal_spectrum
from .second_independent import balanced_residuals

def run(label, cells=256):
    started = perf_counter()
    name = f'nonlinear_{label}_N{cells}_c0.4_L4'
    record, reference = load(name)
    table, _ = load('frozen_prediction')
    spacing = record['diagnostics']['spacing']
    charges = record['diagnostics']['charges']
    end = table['end_time']
    grid, initial, points = (reference['grid'], reference['initial'], reference['points'])
    shape = initial.shape
    indices = np.rint((points - grid[0]) / spacing).astype(int)
    first_weights, second_weights = (weights(1), weights(2))

    def fun(t, flat):
        state = flat.reshape(shape)
        position = state[:7]
        grad, second = (np.zeros_like(position), np.zeros_like(position))
        center = position[:, 3:-3]
        for offset, w1, w2 in zip(range(-3, 4), first_weights, second_weights):
            shifted = position[:, 3 + offset:position.shape[1] - 3 + offset] - center
            grad[:, 3:-3] += w1 * shifted / spacing
            second[:, 3:-3] += w2 * shifted / spacing ** 2
        return np.concatenate([state[7:14], second + nonlinear_force(position, state[7:14], grad, charges), auxiliary_velocity(position, charges)]).ravel()
    solver = DOP853(fun, 0.0, initial.ravel(), end, max_step=0.025, rtol=2e-11, atol=1e-18)
    centers = table['comparison_times'][:2]
    jet_steps = (0.04, 0.02)
    jets = np.unique(np.concatenate([t + np.arange(-3, 4) * step for t in centers for step in jet_steps]))
    samples = reference['times'][1:]
    wanted = np.unique(np.r_[samples, jets])
    saved, spectra, times = ({}, [], [])
    steps = 0
    for t in wanted:
        while solver.t < t and solver.status == 'running':
            solver.step()
            steps += 1
        if solver.status == 'failed':
            raise RuntimeError('independent integration failed')
        state = solver.dense_output()(t).reshape(shape)
        if not np.isfinite(state).all():
            raise FloatingPointError('nonfinite independent state')
        if np.min(abs(jets - t)) < 1e-09:
            saved[round(float(t), 10)] = state.copy()
        if np.min(abs(samples - t)) < 1e-09:
            axes = diagonal_spectrum(state[7:14, indices]).T
            spectra.append(np.c_[np.sort(axes[:, :3], axis=1), axes[:, 3]])
            times.append(t)
    checks = []
    for pindex in (0, len(points) // 2, len(points) - 1):
        j = indices[pindex]
        local = {key: state[:, j - 6:j + 7].ravel() for key, state in saved.items()}
        for t in centers:
            for step in jet_steps:
                checks.append({'position': float(points[pindex]), 'time': float(t), 'jet_step': step, **point_residual(SavedSlices(local), 13, spacing, charges, float(t), step, residual_evaluator=balanced_residuals)})
    spectra = np.array(spectra)
    return save(f'independent_{label}_N{cells}', {'status': 'INDEPENDENT_FROM_INITIAL_SURFACE_COMPLETED', 'method': 'FD6_DOP853', 'shared_initial_constrained_data': True, 'shared_evolved_checkpoint': False, 'initial_record': name, 'start_time': 0.0, 'end_time': end, 'cells_per_width': cells, 'spacing': spacing, 'steps': steps, 'max_step': 0.025, 'max_spectrum_difference': float(np.max(abs(spectra - reference['spectrum'][1:]))), 'four_dimensional_residuals': checks, 'scope': 'independent discretization of same local equations, four-dimensional residual of reconstructed two-Killing fields; not angular/horizon matching', 'runtime_seconds': perf_counter() - started, 'complete_horizon_input': False}, times=np.array(times), points=points, spectrum=spectra, grid=grid, final=solver.y.reshape(shape))
