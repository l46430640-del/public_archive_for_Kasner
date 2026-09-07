"""Numerical equations and integration routines for the local EMS model."""
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp, simpson
from .compact_spectrum import transform, transform_mp, transform_tail
from .directed import background_scales, normalization, protocol, record_for_delta, save
from .directed_analysis import load
from .evolution import derivatives
from .response_model import background_initial, background_rhs
from .scales import projected_components, scale_record

def run(dxi=0.25, cutoff=600.0, nodes=1536):
    start = perf_counter()
    frozen, raw_table = load('frozen_prediction')
    record = record_for_delta()
    scale = scale_record(record)
    bg, charge = background_initial(record)
    width, k, amplitude = [scale[key] for key in ('packet_width', 'k', 'packet_amplitude')]
    xi = np.arange(-int(round(cutoff / dxi)), int(round(cutoff / dxi)) + 1) * dxi
    weights = transform(xi, nodes) / (2 * np.pi)
    frequencies = np.array([k + xi / width, k / 100 + xi / width])
    basis0 = np.array([np.ones_like(frequencies), np.zeros_like(frequencies), np.zeros_like(frequencies), np.ones_like(frequencies)])
    shape = basis0.shape

    def fun(t, flat):
        background = flat[:6]
        db = background_rhs(background, charge)
        f, ft, b, bt = flat[6:].reshape(shape)
        pump = background_scales(background, charge, k, width)[1]
        rate = pump - frequencies ** 2
        return np.r_[db, np.array([ft, rate * f, bt, rate * b]).ravel()]
    times = np.unique(np.r_[0.0, scale['frozen_unit_wall_clock_T'], raw_table['times']])
    solution = solve_ivp(fun, (0.0, times[-1]), np.r_[bg, basis0.ravel()], method='DOP853', t_eval=times, rtol=2e-12, atol=1e-14, max_step=0.03)
    if not solution.success:
        raise RuntimeError(solution.message)
    comp = projected_components(record)
    c = comp['soft_magnetic'] / (-1j * k)
    initial_a = np.array([np.full(len(xi), c), np.full(len(xi), abs(c))])
    momenta = np.array([np.full(len(xi), -comp['transverse_electric'] + bg[2] * bg[5] * c), (bg[2] * bg[5] + 1j * frequencies[1]) * abs(c)])
    points = raw_table['points']
    phases = np.exp(-1j * points[None, :, None] * frequencies[:, None, :])
    fields = []
    for column in solution.y.T:
        background = column[:6]
        f, ft, b, bt = column[6:].reshape(shape)
        u, ut = (f * initial_a + b * momenta, ft * initial_a + bt * momenta)
        e = ut - background[2] * background[5] * u
        m = -1j * frequencies * u
        fields.append(amplitude * np.array([simpson(phases * value[:, None, :] * weights, x=xi, axis=-1) for value in (u, e, m)]))
    fields = np.array(fields)
    pred, raw = load('prediction_N512_c0.4')
    h = pred['spacing']
    final = raw['final']
    grad, _ = derivatives(final[[0, 2]], h)
    final_bg = raw['backgrounds'][-1]
    fd = np.array([final[[0, 2]], final[[1, 3]] - final_bg[2] * final_bg[5] * final[[0, 2]], grad])
    indices = np.rint((points - raw['grid'][0]) / h).astype(int)
    checks = []
    for label, mix in protocol()['mixing'].items():
        norm = normalization(mix)
        continuum = norm * (fields[-1, :, 0] + mix * fields[-1, :, 1]).real
        finite = norm * (fd[:, 0, indices] + mix * fd[:, 1, indices])
        checks.append({'case': label, 'absolute_field_error': np.max(abs(continuum - finite), axis=1).tolist(), 'relative_em_error': float(np.max(abs(continuum[1:] - finite[1:])) / np.max(abs(continuum[1:]))), 'left_em_components': continuum[1:, 0].tolist()})
    tail_at_carrier = float(transform_mp(k * width, 50))
    tail_mass, tail_moment = transform_tail(cutoff)

    def growth_rhs(t, y):
        back = y[:6]
        return np.r_[background_rhs(back, charge), background_scales(back, charge, k, width)[1]]
    integral = solve_ivp(growth_rhs, (0.0, times[-1]), np.r_[bg, 0.0], rtol=1e-12, atol=1e-14).y[-1, -1]
    bounds = []
    g = final_bg[2] * final_bg[5]
    for frequency, oscillator_tail in ((k, (abs(-comp['transverse_electric'] + bg[2] * bg[5] * c) + k * abs(c)) * tail_mass + abs(c) * tail_moment / width), (k / 100, abs(c) * ((abs(bg[2] * bg[5]) + 2 * k / 100) * tail_mass + 2 * tail_moment / width))):
        qmin = cutoff / width - frequency
        if qmin <= 0:
            raise ValueError('cutoff must put the omitted frequencies outside zero')
        magnetic = amplitude * np.exp(integral / (2 * qmin)) * oscillator_tail
        bounds.append([magnetic * (1 + abs(g) / qmin), magnetic])
    return save(f'modes_dxi{dxi:g}_K{cutoff:g}_Q{nodes}', {'status': 'CONTINUUM_LINEAR_CHECK', 'delta': 1e-05, 'frequency_spacing': dxi, 'cutoff': cutoff, 'quadrature_nodes': nodes, 'comparisons': checks, 'carrier_kw': k * width, 'soft_kw': k * width / 100, 'compact_transform_at_carrier_50digits': tail_at_carrier, 'compact_transform_at_carrier_double': float(transform(np.array([k * width]), nodes)[0]), 'compact_transform_at_soft': float(transform(np.array([k * width / 100]), nodes)[0]), 'basis_tail_absolute_em_bounds': bounds, 'scope': 'charged homogeneous background; independent continuum Maxwell propagation, not second-order geometry; tail bounds exclude arithmetic and quadrature error', 'no_modes_removed': True, 'runtime_seconds': perf_counter() - start}, times=times, points=points, complex_basis_fields=fields, backgrounds=solution.y[:6].T, frequencies=frequencies, final_fundamental=solution.y[6:, -1].reshape(shape))
