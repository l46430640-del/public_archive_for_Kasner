"""Numerical equations and integration routines for the local EMS model."""
from ..background import Coupling, solve_branch_at_delta, solve_interior
from ..transfer import interior_background_derivatives, odd_potential_from_interior, _physical_component_matrices
from .io import save_carrier as save
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.integrate import solve_ivp
from .inputs import encode, file_hash

def horizon_coefficients(exterior, ell=2):
    coupling, charge = (exterior.coupling, exterior.charge)
    phi, chi = (exterior.phi_h, exterior.chi_h_normalized)
    z, em = (coupling.z(phi), np.exp(-chi))
    h1 = em * (-1 + charge * charge / z)
    beta = em * charge * charge * coupling.d_inv_z(phi) / (2 * h1)
    hx2 = em * (1 + charge * charge / z - charge * charge / z * coupling.d_log_z(phi) * beta - beta * beta * (-1 + charge * charge / z))
    f1, f2 = (-h1, -(hx2 / 2 + 2 * h1))
    lam = ell * (ell + 1)
    v1 = h1 * np.array([[(lam - 2) * em + h1, 2 * np.sqrt(lam - 2) * charge * em / np.sqrt(z)], [2 * np.sqrt(lam - 2) * charge * em / np.sqrt(z), em * (lam + 4 * charge * charge / z) + h1 * beta * coupling.d_log_z(phi) / 2]])
    return (f1, f2, v1)

def regular_transfer(interior, omega, start, endpoint, rtol=2e-11, max_step=0.002):
    """V=e^{-i omega r_*} H; H(0)=I in one continuous EF-regular basis."""
    f1, f2, v1 = horizon_coefficients(interior.exterior)
    h_derivative = v1 / (f1 * (f1 - 2j * omega))
    h = np.eye(2) + start * h_derivative
    f_start = -interior.solution.sol(start)[3] * np.exp(2 * start)
    rstar_start = np.log(start) / f1 - f2 * start / f1 ** 2
    phase = np.exp(-1j * omega * rstar_start)
    fields = phase * h
    momentum = phase * (f_start * h_derivative - 1j * omega * h)
    initial = np.r_[fields.ravel(), momentum.ravel()]
    ext = interior.exterior

    def rhs(x, flat):
        bg = interior.solution.sol(x)
        f = -bg[3] * np.exp(2 * x)
        potential = odd_potential_from_interior(x, bg, ext.charge, ext.coupling, 2)
        field, pi = (flat[:4].reshape(2, 2), flat[4:].reshape(2, 2))
        return np.r_[(pi / f).ravel(), ((potential - omega * omega * np.eye(2)) @ field / f).ravel()]
    solution = solve_ivp(rhs, (start, endpoint), initial, method='DOP853', rtol=rtol, atol=rtol * 0.01, max_step=max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    field, pi = (solution.y[:4, -1].reshape(2, 2), solution.y[4:, -1].reshape(2, 2))
    bg = interior.solution.sol(endpoint)
    derivative = interior_background_derivatives(endpoint, bg, ext.charge, ext.coupling)
    components = _physical_component_matrices(endpoint, bg, derivative, field, pi, omega, ext.coupling, 2)
    flux = (field.conj().T @ pi - pi.conj().T @ field) / 2j
    return {'omega_times_mass': omega * ext.mass, 'start': start, 'endpoint': endpoint, 'canonical': encode(field), 'momentum': encode(pi), 'components': {key: encode(value) for key, value in components.items()}, 'flux_absolute_error': float(np.max(abs(flux + omega * np.eye(2)))), 'field_norm': float(np.linalg.norm(field)), 'momentum_norm': float(np.linalg.norm(pi)), 'soft_magnetic_norm': float(np.linalg.norm(components['soft_magnetic'])), 'electric_norm': float(np.linalg.norm(components['angular_electric'])), 'evaluations': solution.nfev}

def run():
    started = perf_counter()
    ext = solve_branch_at_delta(1e-06, Coupling('exp_square', 1.0), r_max=180.0)
    interior = solve_interior(ext, log_z_max=3.0, kasner_tolerance=1e-06, sustain_log_z=1.0, rtol=2e-12, atol=1e-13, max_step=0.002, horizon_series_order=2)
    stop = float(interior.sigma_k)
    frequencies = [0.0, 1e-05, 3e-05, 0.0001, 0.0003, 0.001, 0.002, 0.003, 0.01, 0.03, 0.1, 0.2]
    rows = [regular_transfer(interior, omega / ext.mass, 1e-07, stop) for omega in frequencies]
    checks = []
    for omega in (0.0, 0.002, 0.2):
        for start in (4e-07, 2e-07):
            checks.append(regular_transfer(interior, omega / ext.mass, start, stop))
        checks.append(regular_transfer(interior, omega / ext.mass, 1e-07, stop, rtol=2e-12, max_step=0.001))
    negative = regular_transfer(interior, -0.002 / ext.mass, 1e-07, stop)
    f1, f2, v1 = horizon_coefficients(ext)
    horizon_checks = []
    for x in (4e-07, 2e-07, 1e-07):
        bg = interior.solution.sol(x)
        potential = odd_potential_from_interior(x, bg, ext.charge, ext.coupling, 2)
        horizon_checks.append({'x': x, 'relative_V1_error': float(np.linalg.norm(potential / x - v1) / np.linalg.norm(v1)), 'f_minus_series': float(-bg[3] * np.exp(2 * x) - f1 * x - f2 * x * x)})
    return save('horizon_regular_band', {'model': 'linear_spherical_odd_regular_horizon_transfer_only', 'delta': 1e-06, 'normalization': 'H_horizon=identity, V=exp(-i omega rstar) H, same basis at omega=0', 'rstar_convention': 'rstar-log(x)/f1 tends to zero at horizon x=0', 'mass': ext.mass, 'charge': ext.charge, 'matching_surface': stop, 'rows': rows, 'checks': checks, 'negative_frequency_check': negative, 'horizon_series_checks': horizon_checks, 'runtime_seconds': perf_counter() - started, 'complete_linear_packet': False, 'nonlinear_metric_ready': False, 'full_horizon_second_order_ready': False, 'linear_band_data_available': True, 'limitations': ['No frequency-convolved finite packet or metric-momentum pullback.', 'No second-order even input, ADM absorption, or angular forced response.', 'Nonconstant F_AB cannot be inserted into the two-Killing evolution.'], 'horizon_source_sha256': file_hash(Path(__file__))})
