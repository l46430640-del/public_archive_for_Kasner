"""Numerical equations and integration routines for the local EMS model."""
from .io import EXCHANGE_OUTPUT as OUTPUT
import json
from time import perf_counter
import numpy as np
from scipy.integrate import DOP853
from .conformal import auxiliary_velocity, geometric_spectrum, nonlinear_force
from .four_dimensional import residuals
from .independent import point_residual, weights
from .independent_segment import SavedSlices
from .inputs import ROOT, file_hash
from .second_event import POINTS, save

def balanced_residuals(position, first, second):
    """Constant EMS scale symmetry plus constant orbit-coordinate rescaling.

    g -> C^2 g, A -> C A and X^A -> C X^A imply sigma -> sigma+log C,
    (C_y^A,b_y) -> C(C_y^A,b_y), with r,P,Q,psi,a_A unchanged. C is fixed
    at the evaluated point, not differentiated. This avoids huge curvatures
    in coordinate arithmetic without changing relative field residuals.
    """
    q, d, dd = (position.copy(), first.copy(), second.copy())
    factor = np.exp(-q[1])
    q[1] = 0.0
    q[7:] *= factor
    d[:, 7:] *= factor
    dd[:, :, 7:] *= factor
    return residuals(q, d, dd)

def run(label='strong', spacing=0.2, end=300.0, max_step=0.025, domain_factor=1.0):
    start_clock = perf_counter()
    path = OUTPUT / f'event_{label}_h{spacing:g}_c0.4_T380_L4_exp_square.json'
    record = json.loads(path.read_text())
    if record['diagnostics']['failure']:
        raise ValueError('cannot use failed reference')
    with np.load(ROOT / record['raw_file']) as raw:
        grid, initial = (raw['grid'], raw['checkpoint'])
        reference_times, reference_spectra = (raw['times'], raw['spectrum'])
    start = record['checkpoint_time']
    radius = domain_factor * (max(abs(POINTS)) + (end - start) + 24 * spacing)
    mask = abs(grid) <= np.ceil(radius / spacing) * spacing
    grid, initial = (grid[mask], initial[:, mask])
    shape = initial.shape
    charges = record['diagnostics']['charges']
    first_weights, second_weights = (weights(1), weights(2))

    def fun(time, flat):
        state = flat.reshape(shape)
        position = state[:7]
        grad, second = (np.zeros_like(position), np.zeros_like(position))
        center = position[:, 3:-3]
        for offset, w1, w2 in zip(range(-3, 4), first_weights, second_weights):
            shifted = position[:, 3 + offset:position.shape[1] - 3 + offset] - center
            grad[:, 3:-3] += w1 * shifted / spacing
            second[:, 3:-3] += w2 * shifted / spacing ** 2
        return np.concatenate([state[7:14], second + nonlinear_force(position, state[7:14], grad, charges), auxiliary_velocity(position, charges)]).ravel()
    solver = DOP853(fun, start, initial.ravel(), end, max_step=max_step, rtol=2e-11, atol=1e-18)
    jet_centers = np.array([246.0, 259.0, 280.0, 296.0])
    jet_centers = jet_centers[(jet_centers > start + 1) & (jet_centers < end - 1)]
    jet_steps = (0.04, 0.02)
    jet_times = np.unique(np.concatenate([t + np.arange(-3, 4) * step for t in jet_centers for step in jet_steps]))
    samples = reference_times[(reference_times > start) & (reference_times <= end)]
    wanted = np.unique(np.r_[samples, jet_times, end])
    saved, spectra = ({}, [])
    indices = np.rint((POINTS - grid[0]) / spacing).astype(int)
    steps = 0
    for t in wanted:
        while solver.t < t and solver.status == 'running':
            solver.step()
            steps += 1
            if steps % 1000 == 0:
                print(json.dumps({'case': label, 'h': spacing, 'steps': steps, 'time': solver.t, 'dt': solver.step_size, 'elapsed': perf_counter() - start_clock}), flush=True)
        if solver.status == 'failed':
            raise RuntimeError('independent evolution failed')
        state = solver.dense_output()(t).reshape(shape)
        if not np.isfinite(state).all():
            raise FloatingPointError('nonfinite independent state')
        if np.min(abs(jet_times - t)) < 1e-09:
            saved[round(float(t), 10)] = state.copy()
        if np.min(abs(samples - t)) < 1e-09:
            spectra.append(np.array([geometric_spectrum(state[:7, j], state[7:14, j], state[14:, j], charges) for j in indices]))
    checks = []
    for point, j in zip(POINTS[[0, 2, 4]], indices[[0, 2, 4]]):
        local = {key: state[:, j - 6:j + 7].ravel() for key, state in saved.items()}
        for t in jet_centers:
            for step in jet_steps:
                checks.append({'position': float(point), 'time': float(t), 'jet_step': step, **point_residual(SavedSlices(local), 13, spacing, charges, float(t), step, residual_evaluator=balanced_residuals)})
    spectra = np.array(spectra)
    reference = reference_spectra[(reference_times > start) & (reference_times <= end)]
    suffix = f'_L{domain_factor:g}' if domain_factor != 1.0 else ''
    return save(f'independent_{label}_h{spacing:g}_T{end:g}_dt{max_step:g}{suffix}', {'status': 'INDEPENDENT_SEGMENT_COMPLETED', 'method': 'FD6_DOP853', 'shared_early_FD4_checkpoint': True, 'checkpoint_sha256': file_hash(path), 'start_time': start, 'end_time': end, 'spacing': spacing, 'max_step': max_step, 'segment_domain_factor': domain_factor, 'steps': steps, 'max_spectrum_difference': float(np.max(abs(spectra - reference))), 'center_spectrum_difference': float(np.max(abs(spectra[:, 2] - reference[:, 2]))), 'four_dimensional_residuals': checks, 'balanced_coordinate_jets': True, 'runtime_seconds': perf_counter() - start_clock, 'complete_horizon_input': False, 'p_plus': None, 'scattering_error': None}, times=samples, points=POINTS, spectrum=spectra, final=solver.y.reshape(shape), grid=grid)
