"""Numerical equations and integration routines for the local EMS model."""
from __future__ import annotations
from ..background import Coupling, critical_charge_to_mass, solve_branch_at_delta, solve_interior
from ..transfer import interior_background_derivatives, integrate_transfer
from .inputs import encode, decode
DELTAS = (1e-05, 1e-06)
from dataclasses import asdict
from typing import Any
import numpy as np
from .contracts import MatchingData

def generate_inputs() -> dict[str, Any]:
    records = []
    for delta in DELTAS:
        print(f'fresh full odd input delta={delta:g}', flush=True)
        coupling = Coupling('exp_square', 1.0)
        exterior = solve_branch_at_delta(delta, coupling, r_max=180.0)
        interior = solve_interior(exterior, log_z_max=3.0, kasner_tolerance=1e-06, sustain_log_z=1.0, rtol=2e-12, atol=1e-13, max_step=0.002, horizon_series_order=2)
        coordinate_frequency = 0.2 / exterior.mass
        transfer = integrate_transfer(interior, coordinate_frequency, 2)
        x = interior.sigma_k
        state = np.asarray(interior.solution.sol(x), dtype=float)
        derivative = interior_background_derivatives(x, state, exterior.charge, coupling)
        psi, beta, chi, h = state
        z = np.exp(x)
        n_squared = h * z * np.exp(chi)
        d_log_a_dx = derivative[3] / h + 1.0 - beta ** 2
        kt = -0.5 * np.sqrt(-n_squared) * z * d_log_a_dx
        ks = np.sqrt(-n_squared) * z
        ktrace = kt + 2 * ks
        knorm = np.sqrt(kt ** 2 + 2 * ks ** 2)
        a_parallel = np.sqrt(-n_squared) * np.exp(-chi)
        background = {'delta_target': delta, 'delta_measured': exterior.charge_to_mass / critical_charge_to_mass(1.0) - 1, 'mass': exterior.mass, 'charge': exterior.charge, 'omega_times_mass': 0.2, 'coordinate_frequency': coordinate_frequency, 'sigma_k': x, 'psi': psi, 'beta': beta, 'chi': chi, 'h': h, 'state_derivative': derivative.tolist(), 'ktrace': float(ktrace), 'knorm': float(knorm), 'a_parallel': float(a_parallel), 'a_transverse': float(1 / z), 'length_conversion': float(a_parallel * ktrace), 'frequency_over_ktrace': float(coordinate_frequency / (a_parallel * ktrace)), 'knorm_over_ktrace': float(knorm / ktrace), 'p_spatial': [float(ks / ktrace), float(ks / ktrace), float(kt / ktrace)], 'p_scalar': float(np.sqrt(2) * beta * np.sqrt(-n_squared) * z / ktrace)}
        components = {'soft_magnetic': encode(transfer.soft_magnetic_matrix), 'hard_magnetic': encode(transfer.hard_magnetic_matrix), 'transverse_electric': encode(transfer.angular_electric_matrix), 'odd_curvature': encode(transfer.odd_curvature_matrix)}
        records.append(MatchingData(background=background, canonical=encode(transfer.canonical_matrix), canonical_momentum=encode(transfer.canonical_momentum_matrix), components=components, polarization=[], flux_residual=transfer.flux_error, source_hashes={}, omissions=['Full master-to-conformal metric and momentum pullback not derived.', 'y-dependent F_AB is not an exact two-Killing Maxwell flux.', 'Second-order horizon-generated even data are not supplied.']))
    row = decode(records[-1].components['soft_magnetic'])
    polarization = row.conj() / np.linalg.norm(row)
    for record in records:
        record.polarization = encode(polarization)
    return {'schema_version': 1, 'role': 'EXPLORATORY_COMPLETE_LINEAR_INPUT', 'frequency_convention': 'omega_times_ADM_mass=0.2, coordinate omega=0.2/ADM_mass; r_H=1 and asymptotic chi=0. This corrects only new inputs, not the coordinate-frequency convention of published data.', 'polarization_reference_delta': 1e-06, 'records': [asdict(record) for record in records]}
