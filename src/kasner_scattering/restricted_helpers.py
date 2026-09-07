"""Numerical equations and integration routines for the local EMS model."""
from __future__ import annotations
from hashlib import sha256
import json
from math import cos, pi, sin, sqrt
from typing import Any
import numpy as np
from .scattering import BounceConfig, SoftTransfer, canonical_to_kasner_jacobian, exp_square_reduced_map, explicit_inner_reflection, horizon_vectors, inner_scaling, kasner_momenta, nonlinear_input, solve_bounce
DELTAS = tuple((10.0 ** (-9.0 + 0.5 * index) for index in range(11)))
FREQUENCIES = (0.1, 0.2, 0.4)
OPEN_DELTAS = DELTAS
OPEN_FREQUENCIES = tuple((0.195 + 0.001 * index for index in range(11)))
CONTROL_DELTAS = DELTAS[:7]
KAPPAS = (0.1, 0.25, 0.5)
CP1_RADIUS = 0.1
SPATIAL_PHASE = pi / 4

def digest(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode('ascii')).hexdigest()

def with_digest(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result['record_sha256'] = digest(result)
    return result

def _encode_complex(values: np.ndarray) -> list[dict[str, float]]:
    return [{'re': float(value.real), 'im': float(value.imag)} for value in np.asarray(values, dtype=complex)]

def _directions(sample: SoftTransfer) -> dict[str, np.ndarray]:
    maximum = horizon_vectors(sample)['max_soft']
    orthogonal = np.asarray([-np.conjugate(maximum[1]), np.conjugate(maximum[0])], dtype=complex)
    result = {'max_soft': maximum}
    for index, phase in enumerate((0.0, 0.5 * pi, pi, 1.5 * pi)):
        result[f'cp1_boundary_{index}'] = cos(CP1_RADIUS) * maximum + sin(CP1_RADIUS) * np.exp(1j * phase) * orthogonal
    return result

def _rebuild_trajectory(target_delta: float, sample: SoftTransfer, direction_id: str, direction: np.ndarray, kappa: float, config: BounceConfig) -> dict[str, Any]:
    eta_h = sample.delta ** (0.5 + kappa)
    input_data = nonlinear_input(sample, direction, eta_h, SPATIAL_PHASE)
    result = solve_bounce(input_data, config)
    if result.post_kasner is None or result.s_peak is None:
        raise RuntimeError(f'missing post plateau for delta={target_delta}, omega={sample.omega_m}, direction={direction_id}, kappa={kappa}: {result.outcome}')
    reduced = exp_square_reduced_map(result.pre_kasner, delta=sample.delta, varphi_k=input_data.varphi_k, initial_wall_fraction=input_data.initial_wall_fraction, alpha_sq=input_data.alpha_sq, wall_axis=input_data.wall_axis, epsilon_on=config.epsilon_on, epsilon_exit=config.epsilon_exit)
    if reduced is None:
        raise RuntimeError('finite-q one-wall predictor failed')
    field_amplitude = sqrt(2.0 * input_data.initial_wall_fraction)
    scaling = inner_scaling(sample.delta, field_amplitude)
    l_b = scaling['L_B']
    predicted_t_peak = reduced.s_onset + reduced.scaled_peak_time / sqrt(sample.delta)
    theta_peak = result.s_peak * sqrt(sample.delta) / sqrt(l_b)
    predicted_theta = predicted_t_peak * sqrt(sample.delta) / sqrt(l_b)
    p_minus = np.asarray(result.pre_kasner, dtype=float)
    p_plus = np.asarray(result.post_kasner, dtype=float)
    predicted = np.asarray(reduced.predicted_p_plus, dtype=float)
    reflected = explicit_inner_reflection(p_minus)
    epsilon = sqrt(sample.delta)
    rescaled_error = (p_plus - reflected) / epsilon
    q_b = scaling['q_B_leading']
    canonical = kasner_momenta(p_minus)
    record = {'input': {'coupling': 'exp_square', 'target_delta': target_delta, 'measured_delta': sample.delta, 'omega_M': sample.omega_m, 'kappa': kappa, 'direction_id': direction_id, 'direction': _encode_complex(direction), 'eta_H': eta_h, 'field_amplitude': field_amplitude, 'wall_fraction': input_data.initial_wall_fraction}, 'packet_energy_tau_1': sqrt(pi) * eta_h ** 2 * sample.delta ** (-0.5), 'clock': {'L_B': l_b, 's_K': 0.0, 's_peak': result.s_peak, 't_peak': result.s_peak, 'predicted_t_peak': predicted_t_peak, 'Theta_peak': theta_peak, 'predicted_Theta_peak': predicted_theta, 'relative_error': abs(result.s_peak / predicted_t_peak - 1.0)}, 'plateaus': {'outcome': result.outcome, 'p_minus': p_minus.tolist(), 'p_plus': p_plus.tolist(), 'kasner_in': result.kasner_in_window, 'kasner_out': result.kasner_out_window, 'constraint_residual': result.constraint_residual, 'energy_error': result.energy_error}, 'finite_q_map': {'q_B': q_b, 'incoming_canonical_momentum': canonical.tolist(), 'canonical_to_kasner_jacobian': canonical_to_kasner_jacobian(canonical).tolist(), 'predicted_p_plus': predicted.tolist(), 'asymptotic_reflection': reflected.tolist(), 'relative_error': float(np.linalg.norm(p_plus - predicted) / max(np.linalg.norm(p_plus - p_minus), np.finfo(float).tiny)), 'axis_permutation_invariant_distance': float(np.linalg.norm(np.sort(p_plus[:3]) - np.sort(predicted[:3])) ** 2 + (p_plus[3] - predicted[3]) ** 2) ** 0.5, 'q_B_weighted_rescaled_correction': (q_b * rescaled_error).tolist()}, 'inner_limit': {'p_plus_reflection_limit': reflected.tolist(), 'rescaled_reflection_error': float(np.linalg.norm(rescaled_error)), 'scalar_rescaled_error': float(abs(rescaled_error[3])), 'spatial_rescaled_error': float(np.linalg.norm(rescaled_error[:3])), 'sqrt_LB_weighted_scalar_error': float(sqrt(l_b) * abs(rescaled_error[3])), 'sqrt_LB_weighted_spatial_error': float(sqrt(l_b) * np.linalg.norm(rescaled_error[:3])), 'hamiltonian_to_finite_q_relative_error': float(np.linalg.norm(p_plus - predicted) / max(np.linalg.norm(p_plus - p_minus), np.finfo(float).tiny))}}
    return with_digest(record)
