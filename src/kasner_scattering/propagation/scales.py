"""Numerical equations and integration routines for the local EMS model."""
import numpy as np
from scipy.integrate import quad
from .contracts import PacketSpec
from .inputs import decode

def projected_components(record):
    polarization = decode(record['polarization'])
    conversion = record['background']['knorm_over_ktrace']
    return {key: complex(decode(value) @ polarization) * conversion for key, value in record['components'].items()}

def kasner_parameters(record):
    beta = record['background']['beta']
    return (2 / (beta ** 2 + 3), (beta ** 2 - 1) / (beta ** 2 + 3), 2 * beta / (beta ** 2 + 3))

def scale_record(record, spec=None):
    bg = record['background']
    spec = PacketSpec(bg['delta_target']) if spec is None else spec
    if spec.delta != bg['delta_target'] or spec.omega_times_mass != bg['omega_times_mass']:
        raise ValueError('packet delta and physical frequency must match the regenerated transfer')
    ps, pt, v = kasner_parameters(record)
    a, psi0, k = (2 * ps, bg['psi'], bg['frequency_over_ktrace'])
    components = projected_components(record)
    b0 = abs(components['soft_magnetic']) * spec.amplitude
    linear = 2 * psi0 * v - a
    s_clock = (-linear + np.sqrt(linear ** 2 - 8 * v ** 2 * np.log(b0))) / (2 * v ** 2)
    rho_clock = np.exp(-a * s_clock)
    time_clock = -np.expm1(-a * s_clock) / a
    psi_clock = psi0 + v * s_clock
    g = psi_clock * v / rho_clock
    gt = (v ** 2 + a * v * psi_clock) / rho_clock ** 2
    width = bg['length_conversion'] * bg['mass'] * spec.duration

    def potential(t):
        rho = 1 - a * t
        psi = psi0 - v / a * np.log(rho)
        return (v ** 2 * (1 + psi ** 2) + a * v * psi) / rho ** 2
    energy_bound_log = quad(potential, 0, time_clock, epsabs=1e-12)[0] / k
    return {'delta': spec.delta, 'k': k, 'ps': ps, 'pt': pt, 'scalar_velocity': v, 'area_rate': a, 'packet_width': width, 'packet_amplitude': spec.amplitude, 'frozen_unit_wall_clock_s': s_clock, 'frozen_unit_wall_clock_T': time_clock, 'area_at_clock': rho_clock, 'pump_over_carrier_at_clock': g / k, 'pump_potential_over_k2_at_clock': (gt + g ** 2) / k ** 2, 'canonical_oscillator_energy_log_bound': energy_bound_log, 'frozen_efolding_carrier_phase': k * rho_clock / (2 * v * psi_clock - a), 'packet_residence_over_clock': width / time_clock, 'mass_drift_scale_E_over_Mdelta': spec.gaussian_energy / spec.delta, 'hard_to_soft_magnetic_ratio': abs(components['hard_magnetic'] / components['soft_magnetic']), 'coordinate_omega': bg['coordinate_frequency']}
