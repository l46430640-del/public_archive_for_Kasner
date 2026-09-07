"""Numerical equations and integration routines for the local EMS model."""
from .io import save_directed as save, DIRECTED_OUTPUT as OUTPUT
from dataclasses import asdict
from functools import lru_cache
import json
from time import perf_counter
import numpy as np
from scipy.special import roots_legendre
from .contracts import PacketSpec
from .inputs import HERE
from .response_model import background_rhs, initial_expansion, profiles, response_step, scalar_coefficients
from .scales import scale_record

def protocol():
    return json.loads((HERE / 'directed_protocol.json').read_text())

def record_for_delta(delta=1e-05):
    data = json.loads((HERE / 'matching_data.json').read_text())
    return next((row for row in data['records'] if row['background']['delta_target'] == delta))

@lru_cache(maxsize=8)
def energy_coefficients(delta=1e-05, nodes=2048):
    record = record_for_delta(delta)
    width = scale_record(record)['packet_width']
    points, weights = roots_legendre(nodes)
    _, e, m, _, el, ml = profiles(record, width * points, 0.0)
    return 2 * width * np.array([weights @ (e * e + m * m), 2 * weights @ (e * el + m * ml), weights @ (el * el + ml * ml)])

def normalization(mixing, delta=1e-05):
    energy = energy_coefficients(delta)
    return np.sqrt(energy[0] / np.polynomial.polynomial.polyval(mixing, energy))

def combine(coefficients, mixing, delta=1e-05, coherent=True):
    weights = [1.0, mixing if coherent else 0.0, mixing ** 2]
    return normalization(mixing, delta) ** 2 * np.einsum('i,...ij->...j', weights, coefficients)

def grid_for(cells, end, domain_factor=4.0):
    width = scale_record(record_for_delta())['packet_width']
    spacing = width / cells
    radius = width / 4
    half = max(domain_factor * width, radius + end + 24 * spacing)
    n = int(np.ceil(half / spacing))
    return (np.arange(-n, n + 1) * spacing, radius)

def background_scales(bg, charge, k, width):
    rhs = background_rhs(bg, charge)
    beta = charge ** 2 * np.exp(-bg[2] ** 2 - 2 * bg[0] + 2 * bg[1]) / 8
    g = bg[2] * bg[5]
    pump = bg[5] ** 2 + bg[2] * rhs[5] + g * g - 2 * beta
    return [g, pump, beta, pump / k ** 2, g * width, -(bg[0] + bg[1])]

def predict(cells=128, cfl=0.4, fixed_kasner=False):
    started = perf_counter()
    config, record = (protocol(), record_for_delta())
    scale = scale_record(record)
    end = config['maximum_end_over_frozen_clock'] * scale['frozen_unit_wall_clock_T']
    grid, radius = grid_for(cells, end)
    h = scale['packet_width'] / cells
    mask = abs(grid) <= radius + h / 2
    bg, state, charge = initial_expansion(record, grid, fixed_kasner=fixed_kasner)
    count = int(np.ceil(end / (cfl * h)))
    dt = end / count
    stride = max(1, count // 512)
    times, backgrounds, coefficients, scales = ([], [], [], [])
    hit = None
    last_valid = None
    for index in range(count + 1):
        p0, coeff = scalar_coefficients(bg, state[:, mask])
        strengths = [float(np.max(abs(combine(coeff, mix))) / abs(p0)) for mix in config['mixing'].values()]
        if max(strengths) >= config['stop_relative_scalar_response']:
            hit = {'first_above_step': index, 'time': index * dt, 'strengths': strengths}
            bg, state = last_valid
            break
        if index % stride == 0 or index == count:
            times.append(index * dt)
            backgrounds.append(bg.copy())
            coefficients.append(coeff)
            scales.append(background_scales(bg, charge, scale['k'], scale['packet_width']))
        if index != count:
            last_valid = (bg.copy(), state.copy())
            bg, state = response_step(bg, state, h, dt, charge)
            if not np.isfinite(state).all() or not np.isfinite(bg).all():
                raise FloatingPointError(f'nonfinite preflight at step {index}')
    final_time = (hit['first_above_step'] - 1) * dt if hit else end
    if times[-1] != final_time:
        times.append(final_time)
        backgrounds.append(bg.copy())
        coefficients.append(scalar_coefficients(bg, state[:, mask])[1])
        scales.append(background_scales(bg, charge, scale['k'], scale['packet_width']))
    p0, coef = scalar_coefficients(bg, state[:, mask])
    cases = {}
    for label, mix in config['mixing'].items():
        full, incoherent = (combine(coef, mix), combine(coef, mix, coherent=False))
        cases[label] = {'mixing': mix, 'normalization': float(normalization(mix)), 'final_max_relative_response': float(np.max(abs(full)) / abs(p0)), 'final_max_relative_cross': float(np.max(abs(full - incoherent)) / abs(p0)), 'center_relative_response': float(full[len(full) // 2] / abs(p0))}
    spec = PacketSpec(1e-05, envelope='compact', phase=0.0, input_spectrum={'mixing': config['mixing'], 'soft_frequency_ratio': 0.01}, energy_normalization={'target': float(energy_coefficients()[0]), 'definition': 'initial transverse reduced EM Hamiltonian'})
    return save(f'prediction_N{cells}_c{cfl:g}' + ('_Kasner' if fixed_kasner else ''), {'status': 'PREFLIGHT_ONLY', 'packet': asdict(spec), 'scale': scale, 'model': 'same charged no-packet background' if not fixed_kasner else 'fixed vacuum Kasner', 'spacing': h, 'cells_per_width': cells, 'cfl': cfl, 'step': dt, 'maximum_end': end, 'valid_end': final_time, 'first_invalid_response_step': hit, 'charge': charge, 'background_final_p_scalar': p0, 'cases': cases, 'energy_coefficients': energy_coefficients().tolist(), 'background_scale_columns': ['g', 'pump', 'beta', 'pump_over_k2', 'g_times_width', 'volume_clock'], 'local_constraints_constructed_order': 2, 'complete_horizon_input': False, 'full_nonlinear_matching': False, 'output_platform': None, 'scattering_error': None, 'runtime_seconds': perf_counter() - started}, times=np.array(times), points=grid[mask], backgrounds=np.array(backgrounds), scalar_coefficients=np.array(coefficients), background_scales=np.array(scales), grid=grid, final=state)
