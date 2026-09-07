"""Numerical equations and integration routines for the local EMS model."""
import json
from time import perf_counter
import numpy as np
from scipy.integrate import DOP853
from .conformal import auxiliary_velocity, geometric_spectrum, nonlinear_force
from .independent import point_residual, weights
from .independent_segment import SavedSlices
from .inputs import ROOT, file_hash
from .response_experiment import OUTPUT, run_nonlinear, save

def run(spacing=0.2):
    started = perf_counter()
    prefix = f'nonlinear_strong_h{spacing:g}_c0.4_p0.00000_L4_T'
    checkpoint_path = OUTPUT / f'{prefix}1.25_exp_square.json'
    if not checkpoint_path.exists():
        run_nonlinear('strong', spacing=spacing, end_factor=1.25)
    checkpoint = json.loads(checkpoint_path.read_text())
    full = json.loads((OUTPUT / f'{prefix}1.4_exp_square.json').read_text())
    if checkpoint['diagnostics']['numerical_failure'] or full['diagnostics']['numerical_failure']:
        raise ValueError('independent segment needs valid completed reference trajectories')
    data = np.load(ROOT / checkpoint['raw_file'])
    grid, initial = (data['grid'], data['final'])
    shape = initial.shape
    charges = checkpoint['diagnostics']['charges']
    first, second_weights = (weights(1), weights(2))

    def fun(time, flat):
        state = flat.reshape(shape)
        position = state[:7]
        grad, second = (np.zeros_like(position), np.zeros_like(position))
        center = position[:, 3:-3]
        for offset, w1, w2 in zip(range(-3, 4), first, second_weights):
            shifted = position[:, 3 + offset:position.shape[1] - 3 + offset] - center
            grad[:, 3:-3] += w1 * shifted / spacing
            second[:, 3:-3] += w2 * shifted / spacing ** 2
        return np.concatenate([state[7:14], second + nonlinear_force(position, state[7:14], grad, charges), auxiliary_velocity(position, charges)]).ravel()
    start, end = (checkpoint['diagnostics']['end_time'], full['diagnostics']['end_time'])
    solver = DOP853(fun, start, initial.ravel(), end, max_step=0.02, rtol=2e-11, atol=1e-17)
    observation_times = np.array([end - 18, end - 10, end - 3])
    dt_jet = 0.02
    times = np.unique(np.concatenate([t + np.arange(-3, 4) * dt_jet for t in observation_times]))
    saved = {}
    for time in times:
        while solver.t < time and solver.status == 'running':
            solver.step()
        if solver.status == 'failed':
            raise RuntimeError('independent segment failed')
        saved[round(float(time), 10)] = solver.dense_output()(time)
    while solver.status == 'running':
        solver.step()
    if solver.status == 'failed' or not np.isfinite(solver.y).all():
        raise RuntimeError('independent segment did not finish')
    final = solver.y.reshape(shape)
    center = len(grid) // 2
    spectrum = geometric_spectrum(final[:7, center], final[7:14, center], final[14:, center], charges)
    reference_data = np.load(ROOT / full['raw_file'])
    reference = reference_data['final_spectrum'][len(reference_data['points']) // 2]
    residuals = []
    for point in (-34.8, 0.0, 34.8):
        j = int(np.rint((point - grid[0]) / spacing))
        local = {key: value.reshape(shape)[:, j - 6:j + 7].ravel() for key, value in saved.items()}
        for t in observation_times:
            residuals.append({'time': float(t), 'position': float(grid[j]), **point_residual(SavedSlices(local), 13, spacing, charges, float(t), dt_jet)})
    return save(f'independent_strong_h{spacing:g}', {'model': 'FD6_DOP853_local_interaction_segment', 'shared_early_FD4_input': True, 'checkpoint': str(checkpoint_path.relative_to(ROOT)), 'checkpoint_sha256': file_hash(checkpoint_path), 'spacing': spacing, 'max_step': 0.02, 'start_time': start, 'end_time': end, 'final_center_spectrum': spectrum.tolist(), 'center_difference_from_FD4': float(np.max(abs(spectrum - reference))), 'four_dimensional_residuals': residuals, 'runtime_seconds': perf_counter() - started, 'complete_horizon_input': False, 'p_plus': None, 'scattering_error': None}, grid=grid, final=final, sample_times=times, sample_states=np.stack([saved[round(float(t), 10)] for t in times]))
