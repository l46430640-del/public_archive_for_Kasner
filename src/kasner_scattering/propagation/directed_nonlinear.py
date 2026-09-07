"""Numerical equations and integration routines for the local EMS model."""
from .io import DIRECTED_OUTPUT as OUTPUT
from .inputs import frozen_directed as frozen
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .accelerated import compiled_step
from .contracts import EvolutionResult, PacketSpec
from .directed import energy_coefficients, grid_for, normalization, protocol, record_for_delta, save
from .evolution import constraint_initial_data
from .inputs import file_hash
from .response_model import background_initial, profiles
from .scales import scale_record
from .second_event import observables

def nonlinear_initial(grid, mixing):
    record = record_for_delta()
    bg, charge = background_initial(record)
    uc, ec, _, ul, el, _ = profiles(record, grid)
    norm = normalization(mixing)
    u, e = (norm * (uc + mixing * ul), norm * (ec + mixing * el))
    x, v = (np.zeros((7, len(grid))), np.zeros((7, len(grid))))
    x[4], v[0], v[4] = (bg[2], bg[3], bg[5])
    root_z = np.exp(bg[2] ** 2 / 2)
    x[5], v[5] = (u / root_z, e / root_z)
    charges = (charge, 0.0, 0.0)
    return (constraint_initial_data(grid, x, v, charges), charges)

def run(label, cells=128, cfl=0.4, domain_factor=4.0):
    start = perf_counter()
    prediction = frozen()
    end = prediction['end_time']
    scale = scale_record(record_for_delta())
    spacing = scale['packet_width'] / cells
    grid, radius = grid_for(cells, end, domain_factor)
    points = np.arange(-32, 33) * scale['packet_width'] / 128
    indices = np.rint((points - grid[0]) / spacing).astype(int)
    if np.max(abs(grid[indices] - points)) > 1e-10:
        raise ValueError('observation curves must be common mesh nodes')
    mixing = protocol()['mixing'][label]
    state, charges = nonlinear_initial(grid, mixing)
    initial = state.copy()
    count = int(np.ceil(end / (cfl * spacing)))
    dt = end / count
    stride = max(1, count // 600)
    stepper = compiled_step('exp_square')
    histories, times = ({}, [])
    failure = None
    for i in range(count + 1):
        if i % stride == 0 or i == count:
            obs = observables(state, spacing, indices, charges)
            for key, value in obs.items():
                histories.setdefault(key, []).append(value)
            times.append(i * dt)
        if i != count:
            candidate = stepper(state, dt, spacing, charges)
            if not np.isfinite(candidate).all():
                failure = {'last_finite_time': i * dt, 'invalid_time': (i + 1) * dt, 'invalid_indices': np.argwhere(~np.isfinite(candidate))[:20].tolist()}
                break
            state = candidate
    result = EvolutionResult(status='NUMERICAL_FAILURE' if failure else 'FINITE_TIME_LOCAL_RESPONSE_NO_PLATFORM_CLAIM', model='constrained_two_Killing_EMS', diagnostics={'packet': asdict(PacketSpec(1e-05, envelope='compact', input_spectrum={'mixing': mixing, 'soft_frequency_ratio': 0.01}, energy_normalization={'target': float(energy_coefficients()[0]), 'factor': float(normalization(mixing)), 'definition': 'initial transverse reduced EM Hamiltonian'})), 'spacing': spacing, 'cells_per_width': cells, 'step': dt, 'end_time': end, 'half_domain': float(grid[-1]), 'charges': list(charges), 'failure': failure, 'coupling': 'exp_square'}, prediction={'frozen_table_sha256': file_hash(OUTPUT / 'frozen_prediction.json')}, validity={'local_constraints_constructed': True, 'complete_horizon_input': False, 'full_nonlinear_matching': False, 'platform_search_performed': False}).to_dict()
    result['runtime_seconds'] = perf_counter() - start
    result['source_partition_version'] = 2
    return save(f'nonlinear_{label}_N{cells}_c{cfl:g}_L{domain_factor:g}', result, grid=grid, initial=initial, final=state, points=points, times=np.array(times), **{key: np.array(value) for key, value in histories.items()})
