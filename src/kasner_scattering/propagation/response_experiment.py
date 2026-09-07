"""Numerical equations and integration routines for the local EMS model."""
from .io import save_reference as save, REF_OUTPUT as OUTPUT
from .inputs import frozen_case
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .conformal import geometric_spectrum
from .contracts import EvolutionResult, PacketSpec
from .evolution import EvolutionFailure, constraint_initial_data, evolve
from .inputs import file_hash
from .response_model import background_initial, energy_coefficients, initial_expansion, normalization, profiles, reference_record, response_step, scalar_coefficients
from .scales import scale_record

def grid_for(spacing, end, domain_factor=4.0):
    scale = scale_record(reference_record())
    observation = scale['packet_width'] / 4
    half = max(domain_factor * scale['packet_width'], observation + end + 24 * spacing)
    n = int(np.ceil(half / spacing))
    return (np.arange(-n, n + 1) * spacing, observation)

def predict(spacing=0.4, cfl=0.4, phase=0.0, end_factor=1.4, fixed_kasner=False):
    started = perf_counter()
    record = reference_record()
    scale = scale_record(record)
    end = end_factor * scale['frozen_unit_wall_clock_T']
    grid, observation = grid_for(spacing, end)
    mask = abs(grid) <= observation + spacing / 2
    bg, state, charge = initial_expansion(record, grid, phase, fixed_kasner)
    count = int(np.ceil(end / (cfl * spacing)))
    dt = end / count
    wanted = set(np.linspace(0, count, 97, dtype=int))
    saved, backgrounds, times, coefs = ([], [], [], [])
    for index in range(count + 1):
        if index in wanted:
            saved.append(state[:, mask].copy())
            backgrounds.append(bg.copy())
            times.append(index * dt)
            coefs.append(scalar_coefficients(bg, state[:, mask])[1])
        if index != count:
            bg, state = response_step(bg, state, spacing, dt, charge)
            if not np.isfinite(state).all() or not np.isfinite(bg).all():
                raise FloatingPointError(f'nonfinite amplitude expansion at step {index}')
    p0, final = scalar_coefficients(bg, state[:, mask])
    name = f'prediction_h{spacing:g}_c{cfl:g}_p{phase:.5f}_T{end_factor:g}'
    if fixed_kasner:
        name += '_Kasner'
    return save(name, {'model': 'local_EMS_amplitude_expansion_order_two', 'fixed_Kasner': fixed_kasner, 'phase': phase, 'spacing': spacing, 'step': dt, 'end_time': end, 'steps': count, 'charge': charge, 'background_final': bg.tolist(), 'background_p_scalar': p0, 'max_scalar_coefficient_over_background': (np.max(abs(final), axis=1) / abs(p0)).tolist(), 'energy_coefficients': energy_coefficients(phase).tolist(), 'runtime_seconds': perf_counter() - started, 'physical_horizon_embedding': False}, grid=grid, final=state, points=grid[mask], times=np.array(times), backgrounds=np.array(backgrounds), states=np.array(saved), scalar_coefficients=np.array(coefs))

def nonlinear_initial(grid, mixing, phase=0.0, coupling='exp_square'):
    record = reference_record()
    bg, charge = background_initial(record)
    uc, ec, _, ul, el, _ = profiles(record, grid, phase)
    norm = normalization(mixing, phase)
    u, e = (norm * (uc + mixing * ul), norm * (ec + mixing * el))
    x, v = (np.zeros((7, len(grid))), np.zeros((7, len(grid))))
    x[4], v[0], v[4] = (bg[2], bg[3], bg[5])
    root_z = np.exp(bg[2] ** 2 / 2) if coupling == 'exp_square' else 1.0
    x[5], v[5] = (u / root_z, e / root_z)
    charges = (charge, 0.0, 0.0)
    return (constraint_initial_data(grid, x, v, charges, coupling), charges)

def run_nonlinear(label, spacing=0.4, cfl=0.4, phase=0.0, domain_factor=4.0, end_factor=1.4, coupling='exp_square'):
    started = perf_counter()
    frozen, case = frozen_case(label)
    end = end_factor * scale_record(reference_record())['frozen_unit_wall_clock_T']
    grid, observation = grid_for(spacing, end, domain_factor)
    initial, charges = nonlinear_initial(grid, case['mixing'], phase, coupling)
    failure = None
    try:
        run = evolve(grid, initial, end, cfl * spacing, observation, charges, coupling, samples=97)
    except EvolutionFailure as exc:
        run = exc.run
        failure = {'last_finite_time': exc.last_finite_time, 'invalid_time': exc.invalid_time, 'invalid_indices': exc.invalid_indices}
    mask = abs(grid) <= observation + spacing / 2

    def spectra(state):
        return np.array([geometric_spectrum(state[:7, j], state[7:14, j], state[14:, j], charges, coupling) for j in np.flatnonzero(mask)])
    spec = PacketSpec(1e-06, phase=phase, envelope='compact', input_spectrum={'family': 'compact_carrier_plus_smooth_soft', 'mixing': case['mixing'], 'soft_frequency_ratio': 0.01}, energy_normalization={'target': frozen['target_transverse_reduced_energy'], 'factor': float(normalization(case['mixing'], phase))})
    name = f'nonlinear_{label}_h{spacing:g}_c{cfl:g}_p{phase:.5f}_L{domain_factor:g}_T{end_factor:g}_{coupling}'
    summary = EvolutionResult(status='NUMERICAL_FAILURE' if failure else 'FINITE_TIME_LOCAL_RESPONSE_NO_PLATFORM_CLAIM', model='constrained_nonlinear_two_Killing_EMS', diagnostics={'packet': asdict(spec), 'spacing': spacing, 'step': run.step, 'steps': run.steps, 'end_time': end, 'half_domain': float(grid[-1]), 'observation_radius': observation, 'charges': list(charges), 'coupling': coupling, 'snapshots': run.snapshots, 'numerical_failure': failure, 'runtime_seconds': perf_counter() - started}, prediction={'selection_sha256': file_hash(OUTPUT / 'frozen_selection.json'), 'case': case}, validity={'local_constraints_constructed': True, 'complete_horizon_input': False, 'linear_prediction_after_order_one_response_is_not_physical_energy': True}).to_dict()
    arrays = {'grid': grid, 'initial': initial, 'final': run.state, 'points': grid[mask], 'initial_spectrum': spectra(initial)}
    if not failure:
        arrays['final_spectrum'] = spectra(run.state)
    return save(name, summary, **arrays)
