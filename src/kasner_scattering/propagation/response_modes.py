"""Numerical equations and integration routines for the local EMS model."""
import json
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp, simpson, quad
from .compact_spectrum import transform, transform_tail
from .evolution import derivatives
from .inputs import ROOT
from .response_experiment import OUTPUT, save
from .response_model import normalization, reference_record
from .response_summary import sample
from .scales import projected_components, scale_record

def run(dxi=0.25, cutoff=600.0, quadrature=1536):
    started = perf_counter()
    record, scale = (reference_record(), scale_record(reference_record()))
    a, v, width, k = [scale[key] for key in ('area_rate', 'scalar_velocity', 'packet_width', 'k')]
    psi0 = record['background']['psi']
    end = 1.4 * scale['frozen_unit_wall_clock_T']
    xi = np.arange(-int(round(cutoff / dxi)), int(round(cutoff / dxi)) + 1) * dxi
    weights = transform(xi, quadrature) / (2 * np.pi)
    frequencies = np.array([k + xi / width, k / 100 + xi / width])
    n = frequencies.size

    def fun(t, flat):
        fields = flat.reshape(4, 2, len(xi))
        f, ft, b, bt = fields
        rho = 1 - a * t
        psi = psi0 - v / a * np.log(rho)
        pump = (v * v * (1 + psi * psi) + a * v * psi) / (rho * rho)
        rate = pump - frequencies ** 2
        return np.array([ft, rate * f, bt, rate * b]).ravel()
    initial = np.array([np.ones((2, len(xi))), np.zeros((2, len(xi))), np.zeros((2, len(xi))), np.ones((2, len(xi)))])
    solver = solve_ivp(fun, (0.0, end), initial.ravel(), method='DOP853', rtol=2e-12, atol=1e-14, max_step=0.05, t_eval=[end])
    if not solver.success:
        raise RuntimeError(solver.message)
    f, ft, b, bt = solver.y[:, -1].reshape(4, 2, len(xi))
    comp = projected_components(record)
    c = comp['soft_magnetic'] / (-1j * k)
    amplitudes = np.array([np.full(len(xi), c), np.full(len(xi), abs(c))])
    momenta = np.array([np.full(len(xi), -comp['transverse_electric'] + psi0 * v * c), (psi0 * v + 1j * frequencies[1]) * abs(c)])
    u, ut = (f * amplitudes + b * momenta, ft * amplitudes + bt * momenta)
    points = np.arange(-87, 88) * 0.4
    rho = 1 - a * end
    psi = psi0 - v / a * np.log(rho)
    g = psi * v / rho
    outputs = []
    for index in range(2):
        phase = np.exp(-1j * np.outer(points, frequencies[index]))
        outputs.append(scale['packet_amplitude'] * np.array([simpson(phase * (ut[index] - g * u[index]) * weights, x=xi, axis=1), simpson(phase * (-1j * frequencies[index]) * u[index] * weights, x=xi, axis=1)]))
    outputs = np.array(outputs)
    tail_mass, tail_moment = transform_tail(cutoff)

    def pump(t):
        rho = 1 - a * t
        psi = psi0 - v / a * np.log(rho)
        return (v * v * (1 + psi * psi) + a * v * psi) / (rho * rho)
    tail_bounds = []
    for frequency, oscillator_tail in ((k, (abs(-comp['transverse_electric'] + psi0 * v * c) + k * abs(c)) * tail_mass + abs(c) * tail_moment / width), (k / 100, abs(c) * ((abs(psi0 * v) + 2 * k / 100) * tail_mass + 2 * tail_moment / width))):
        qmin = cutoff / width - frequency
        if qmin <= 0:
            raise ValueError('tail estimate requires frequencies outside zero')
        growth = np.exp(quad(pump, 0, end, epsabs=1e-11)[0] / (2 * qmin))
        magnetic = scale['packet_amplitude'] * growth * oscillator_tail
        tail_bounds.append([magnetic * (1 + abs(g) / qmin), magnetic])
    tail_bounds = np.array(tail_bounds)
    frozen = json.loads((OUTPUT / 'frozen_selection.json').read_text())
    pred = json.loads((OUTPUT / 'prediction_h0.1_c0.4_p0.00000_T1.4.json').read_text())
    raw = np.load(ROOT / pred['raw_file'])
    fields = raw['final']
    grad, _ = derivatives(fields[[0, 2]], 0.1)
    bg = raw['backgrounds'][-1]
    fd = np.array([fields[[1, 3]] - bg[2] * bg[5] * fields[[0, 2]], grad]).transpose(1, 0, 2)
    checks = []
    for case in frozen['phase_zero_cases']:
        m = case['mixing']
        norm = normalization(m)
        continuum = norm * (outputs[0] + m * outputs[1]).real
        numerical = norm * (fd[0] + m * fd[1])
        numerical = sample(numerical, raw['grid'], points)
        checks.append({'case': case['label'], 'relative_em_field_error': float(np.max(abs(continuum - numerical)) / np.max(abs(continuum))), 'tail_absolute_em_bound': (norm * (tail_bounds[0] + abs(m) * tail_bounds[1])).tolist(), 'max_continuum_em_field': float(np.max(abs(continuum)))})
    return save(f'continuum_modes_dxi{dxi:g}_K{cutoff:g}_Q{quadrature}', {'model': 'linear_Maxwell_on_prescribed_Kasner', 'frequency_spacing': dxi, 'cutoff': cutoff, 'transform_nodes': quadrature, 'comparisons': checks, 'runtime_seconds': perf_counter() - started, 'comparison_scope': 'continuum Kasner versus FD4 charged-background linear fields, not independent nonlinear geometry', 'tail_bound_scope': 'frequency truncation only, not transform quadrature or solver roundoff', 'low_modes_removed': False}, points=points, complex_basis_fields=outputs)
