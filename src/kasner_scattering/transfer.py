"""Odd-parity Einstein-Maxwell-scalar system and radial transfer.

The implementation specializes Eqs. (4.25)-(4.27) of Gannouji and Baez,
JHEP 02 (2022) 020, to

    f1 = 2,  f2 = 4 X + 4 Z(phi) F,

and to the Li-Sun-Yang background variables (x, phi, beta, chi, h), where
x=log(z), beta=dphi/dx, and h=z^-3 exp(-chi) f.  The two canonical variables
are ordered as (V_g, V_e).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .background import Coupling, InteriorSolution


@dataclass(frozen=True)
class OddSectorMetadata:
    ell: int
    lambda_value: int
    physical_degrees_of_freedom: int = 2
    scalar_odd_amplitude: int = 0
    horizon_channels: tuple[str, str] = ("grav_led", "em_led")


def odd_sector_metadata(ell: int) -> OddSectorMetadata:
    if ell < 2:
        raise ValueError("the higher-multipole odd system requires ell >= 2")
    return OddSectorMetadata(ell=ell, lambda_value=ell * (ell + 1))


def interior_background_derivatives(
    log_z: float,
    state: np.ndarray,
    charge: float,
    coupling: Coupling,
) -> np.ndarray:
    """Return d(phi,beta,chi,h)/dx for an interior background state."""

    phi, beta, chi, h_value = state
    z_coord = exp(log_z)
    e_minus_chi = exp(-chi) if chi < 740.0 else 0.0
    z_value = coupling.z(phi)
    h_x = e_minus_chi * (-1.0 / z_coord + charge * charge * z_coord / z_value)
    beta_x = -h_x * beta / h_value + (
        z_coord
        * e_minus_chi
        * charge
        * charge
        * coupling.d_inv_z(phi)
        / (2.0 * h_value)
    )
    return np.array([beta, beta_x, beta * beta, h_x], dtype=float)


def odd_potential_from_interior(
    log_z: float,
    state: np.ndarray,
    charge: float,
    coupling: Coupling,
    ell: int,
) -> np.ndarray:
    """Return the real symmetric canonical 2x2 potential in the interior."""

    metadata = odd_sector_metadata(ell)
    lam = metadata.lambda_value
    phi, beta, chi, h_value = [float(value) for value in state]
    z_coord = exp(log_z)
    z_value = coupling.z(phi)
    e_minus_chi = exp(-chi) if chi < 740.0 else 0.0
    _, beta_x, _, h_x = interior_background_derivatives(
        log_z, state, charge, coupling
    )

    # D=dr/dr_*=N exp(-chi)=h z and A=N exp(-2chi)=h z exp(-chi).
    a_over_r2 = h_value * z_coord**3 * e_minus_chi
    s1 = h_value * z_coord**2
    s1_x = z_coord**2 * (h_x + 2.0 * h_value)

    log_z_phi = coupling.d_log_z(phi)
    log_z_phi_phi = coupling.dd_log_z(phi)
    s2 = 0.5 * h_value * z_coord**2 * beta * log_z_phi
    s2_x = 0.5 * z_coord**2 * (
        (h_x + 2.0 * h_value) * beta * log_z_phi
        + h_value * beta_x * log_z_phi
        + h_value * beta * beta * log_z_phi_phi
    )

    # -dS/dr_* = h z^2 dS/dx because dr/dr_*=h z and dr/dx=-r.
    v11 = (lam - 2.0) * a_over_r2 + h_value * z_coord**2 * s1_x + s1 * s1
    g_value = (
        4.0
        * h_value
        * charge
        * charge
        * z_coord**5
        * e_minus_chi
        / z_value
    )
    v22 = lam * a_over_r2 + g_value + h_value * z_coord**2 * s2_x + s2 * s2
    v12 = (
        2.0
        * sqrt(lam - 2.0)
        * h_value
        * charge
        * z_coord**4
        * e_minus_chi
        / sqrt(z_value)
    )
    return np.array([[v11, v12], [v12, v22]], dtype=float)


def rn_odd_potential(
    radius: float, mass: float, charge: float, ell: int
) -> np.ndarray:
    """Return the canonical RN potential in the action normalization."""

    metadata = odd_sector_metadata(ell)
    lam = metadata.lambda_value
    lapse = 1.0 - 2.0 * mass / radius + charge * charge / radius**2
    v11 = lapse / radius**2 * (
        lam - 6.0 * mass / radius + 4.0 * charge * charge / radius**2
    )
    v22 = lapse / radius**2 * (
        lam + 4.0 * charge * charge / radius**2
    )
    v12 = (
        2.0
        * sqrt(lam - 2.0)
        * lapse
        * charge
        / radius**3
    )
    return np.array([[v11, v12], [v12, v22]], dtype=float)


def rn_interior_state(
    log_z: float, mass: float, charge: float
) -> np.ndarray:
    """Return (phi,beta,chi,h) for RN in x=log(z) variables."""

    z_coord = exp(log_z)
    lapse = 1.0 - 2.0 * mass * z_coord + charge * charge * z_coord**2
    h_value = lapse / z_coord
    return np.array([0.0, 0.0, 0.0, h_value], dtype=float)


def canonical_to_physical(
    log_z: float,
    state: np.ndarray,
    canonical: np.ndarray,
    coupling: Coupling,
    ell: int,
) -> tuple[complex, complex]:
    """Return longitudinal magnetic and odd-curvature amplitudes.

    The first result is sqrt(Z) times the magnetic field along the interior
    longitudinal (Schwarzschild-time) axis.  It comes from F_AB and therefore
    carries the hard magnetic wall beta^3.  It is retained for compatibility;
    the complete component reconstruction lives in ``transfer.py``.
    The second is the Gerlach-Sengupta orbit-curvature amplitude q/r^2,
    proportional to the odd magnetic Weyl harmonic.  Both are linear in the
    canonical fields and independent of Regge-Wheeler gauge.
    """

    metadata = odd_sector_metadata(ell)
    lam = metadata.lambda_value
    _, _, chi, _ = state
    z_coord = exp(log_z)
    v_g, v_e = canonical
    magnetic = lam * z_coord**2 * v_e / sqrt(8.0 * (lam - 2.0))
    q_auxiliary = exp(-float(chi)) * z_coord * v_g / sqrt(2.0)
    odd_curvature = sqrt(lam * (lam - 2.0)) * z_coord**2 * q_auxiliary
    return complex(magnetic), complex(odd_curvature)


def potential_is_symmetric(matrix: np.ndarray, tolerance: float = 1.0e-13) -> bool:
    return bool(np.max(np.abs(matrix - matrix.T)) < tolerance)




@dataclass
class TransferResult:
    ell: int
    omega_m: float
    sigma_k: float
    transfer_matrix: np.ndarray
    canonical_matrix: np.ndarray
    canonical_momentum_matrix: np.ndarray
    soft_magnetic_matrix: np.ndarray
    hard_magnetic_matrix: np.ndarray
    angular_electric_matrix: np.ndarray
    odd_curvature_matrix: np.ndarray
    singular_values: np.ndarray
    magnetic_row_norm: float
    hard_magnetic_row_norm: float
    angular_electric_row_norm: float
    curvature_row_norm: float
    flux_error: float
    constraint_residual: float
    horizon_normalization: str


def _extrinsic_curvature_norm(
    log_z: float, state: np.ndarray, derivative: np.ndarray
) -> float:
    """Return sqrt(K_ij K^ij) on a constant-r interior hypersurface."""

    _, beta, chi, h_value = [float(value) for value in state]
    h_x = float(derivative[3])
    z_coord = exp(log_z)
    d_value = h_value * z_coord
    lapse_n = d_value * exp(chi)
    if lapse_n >= 0.0:
        raise ValueError("the matching surface is not inside the event horizon")
    root_minus_n = sqrt(-lapse_n)
    d_log_a_dx = h_x / h_value + 1.0 - beta * beta
    k_t = -0.5 * root_minus_n * z_coord * d_log_a_dx
    k_s = root_minus_n * z_coord
    return sqrt(k_t * k_t + 2.0 * k_s * k_s)


def _physical_component_matrices(
    log_z: float,
    state: np.ndarray,
    derivative: np.ndarray,
    canonical_matrix: np.ndarray,
    momentum_matrix: np.ndarray,
    omega_m: float,
    coupling: Coupling,
    ell: int,
) -> dict[str, np.ndarray]:
    """Reconstruct gauge-invariant orthonormal Maxwell/Weyl rows.

    On a constant-r interior slice axis 3 is the Schwarzschild-time direction.
    With the local angular frame aligned with X_A, F_rA gives E_1, F_tA gives
    the soft B_2 wall, and F_AB gives the hard B_3 wall.  The normalization is
    harmonic RMS and every Maxwell row is multiplied by sqrt(Z)/|K|.
    """

    lam = ell * (ell + 1)
    z_coord = exp(log_z)
    phi, beta, chi, h_value = [float(value) for value in state]
    k_norm = _extrinsic_curvature_norm(log_z, state, derivative)
    lapse_n = h_value * z_coord * exp(chi)
    if lapse_n >= 0.0:
        raise ValueError("the matching surface is not inside the event horizon")
    root_minus_n = sqrt(-lapse_n)
    normalization = sqrt(8.0 * (lam - 2.0))

    hard_magnetic_factor = (
        lam * z_coord**2 / sqrt(8.0 * (lam - 2.0)) / k_norm
    )
    soft_magnetic_factor = (
        omega_m
        * sqrt(lam)
        * exp(chi)
        * z_coord
        / root_minus_n
        / normalization
        / k_norm
    )
    s2 = 0.5 * h_value * z_coord**2 * beta * coupling.d_log_z(phi)
    angular_electric_factor = (
        sqrt(lam) * root_minus_n / h_value / normalization / k_norm
    )
    q_factor = exp(-chi) * z_coord / sqrt(2.0)
    curvature_factor = (
        sqrt(lam * (lam - 2.0)) * z_coord**2 * q_factor / k_norm**2
    )
    return {
        "soft_magnetic": -1j * soft_magnetic_factor * canonical_matrix[1, :],
        "hard_magnetic": hard_magnetic_factor * canonical_matrix[1, :],
        "angular_electric": angular_electric_factor
        * (momentum_matrix[1, :] + s2 * canonical_matrix[1, :]),
        "odd_curvature": curvature_factor * canonical_matrix[0, :],
    }


def integrate_transfer(
    interior: InteriorSolution,
    omega_m: float,
    ell: int = 2,
    *,
    horizon_basis: np.ndarray | None = None,
    matching_log_z: float | None = None,
    rtol: float = 5.0e-14,
    atol: float = 5.0e-16,
) -> TransferResult:
    """Integrate the two regular odd channels to the selected Kasner surface."""

    if omega_m < 0.0:
        raise ValueError("omega_m must be non-negative")
    if ell < 2:
        raise ValueError("ell=1 is a separate constrained system")

    basis = np.eye(2, dtype=complex) if horizon_basis is None else np.asarray(
        horizon_basis, dtype=complex
    )
    if basis.shape != (2, 2):
        raise ValueError("horizon_basis must be a 2x2 matrix")

    x_start = float(interior.log_z[0])
    x_stop = interior.sigma_k if matching_log_z is None else float(matching_log_z)
    if not x_start < x_stop <= float(interior.log_z[-1]):
        raise ValueError("matching_log_z lies outside the integrated interior")
    background_start = np.asarray(interior.solution.sol(x_start), dtype=float)
    if omega_m > 0.0:
        psi0 = basis / sqrt(omega_m)
        pi0 = -1j * sqrt(omega_m) * basis
        normalization = "unit ingoing symplectic flux"
    else:
        psi0 = basis.copy()
        pi0 = np.zeros((2, 2), dtype=complex)
        normalization = "unit static canonical field (control only)"

    initial_flux = (psi0.conj().T @ pi0 - pi0.conj().T @ psi0) / (2j)
    y0 = np.concatenate([psi0.ravel(), pi0.ravel()])
    charge = interior.exterior.charge
    coupling = interior.exterior.coupling

    def rhs(log_z: float, y: np.ndarray) -> np.ndarray:
        background = np.asarray(interior.solution.sol(log_z), dtype=float)
        h_value = float(background[3])
        z_coord = exp(log_z)
        dx_drstar = -h_value * z_coord**2
        potential = odd_potential_from_interior(
            log_z, background, charge, coupling, ell
        )
        psi = y[:4].reshape(2, 2)
        pi = y[4:].reshape(2, 2)
        psi_x = pi / dx_drstar
        pi_x = (potential - omega_m * omega_m * np.eye(2)) @ psi / dx_drstar
        return np.concatenate([psi_x.ravel(), pi_x.ravel()])

    solution = solve_ivp(
        rhs,
        (x_start, x_stop),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=0.001,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    final = solution.y[:, -1]
    psi = final[:4].reshape(2, 2)
    pi = final[4:].reshape(2, 2)
    final_flux = (psi.conj().T @ pi - pi.conj().T @ psi) / (2j)
    flux_scale = max(1.0, float(np.max(np.abs(initial_flux))))
    flux_error = float(np.max(np.abs(final_flux - initial_flux)) / flux_scale)

    background = np.asarray(interior.solution.sol(x_stop), dtype=float)
    # OdeSolution has no public derivative method. Re-evaluate the exact
    # first-order background equations.
    derivative = interior_background_derivatives(x_stop, background, charge, coupling)
    components = _physical_component_matrices(
        x_stop,
        background,
        derivative,
        psi,
        pi,
        omega_m,
        coupling,
        ell,
    )
    transfer = np.vstack([components["soft_magnetic"], components["odd_curvature"]])
    singular_values = np.linalg.svd(transfer, compute_uv=False)

    # After eliminating h0 and h1, numerical preservation of the original odd
    # constraint is equivalent to self-adjointness of the canonical potential.
    # Check it on every accepted integration node rather than estimating an ODE
    # derivative from two adaptive steps.
    constraint_residual = 0.0
    for node in solution.t:
        node_background = np.asarray(interior.solution.sol(node), dtype=float)
        node_potential = odd_potential_from_interior(
            float(node), node_background, charge, coupling, ell
        )
        scale = max(1.0, float(np.max(np.abs(node_potential))))
        constraint_residual = max(
            constraint_residual,
            float(np.max(np.abs(node_potential - node_potential.T)) / scale),
        )

    return TransferResult(
        ell=ell,
        omega_m=omega_m,
        sigma_k=x_stop,
        transfer_matrix=transfer,
        canonical_matrix=psi,
        canonical_momentum_matrix=pi,
        soft_magnetic_matrix=components["soft_magnetic"],
        hard_magnetic_matrix=components["hard_magnetic"],
        angular_electric_matrix=components["angular_electric"],
        odd_curvature_matrix=components["odd_curvature"],
        singular_values=singular_values,
        magnetic_row_norm=float(np.linalg.norm(transfer[0, :])),
        hard_magnetic_row_norm=float(np.linalg.norm(components["hard_magnetic"])),
        angular_electric_row_norm=float(np.linalg.norm(components["angular_electric"])),
        curvature_row_norm=float(np.linalg.norm(transfer[1, :])),
        flux_error=flux_error,
        constraint_residual=constraint_residual,
        horizon_normalization=normalization,
    )


def integrate_transfer_with_sensitivity(
    interior: InteriorSolution,
    omega: float,
    ell: int = 2,
) -> dict[str, Any]:
    """Integrate the odd system together with its frequency derivative."""

    start = float(interior.log_z[0])
    stop = float(interior.sigma_k)
    psi0 = np.eye(2, dtype=complex) / sqrt(omega)
    pi0 = -1j * sqrt(omega) * np.eye(2, dtype=complex)
    initial_flux = (psi0.conj().T @ pi0 - pi0.conj().T @ psi0) / (2j)
    psi_omega0 = -0.5 * np.eye(2, dtype=complex) / omega**1.5
    pi_omega0 = -0.5j * np.eye(2, dtype=complex) / sqrt(omega)
    y0 = np.concatenate(
        [psi0.ravel(), pi0.ravel(), psi_omega0.ravel(), pi_omega0.ravel()]
    )
    charge = interior.exterior.charge
    coupling = interior.exterior.coupling

    def radial_rhs(x: float, y: np.ndarray) -> np.ndarray:
        background = np.asarray(interior.solution.sol(x), dtype=float)
        radial_factor = -float(background[3]) * exp(2.0 * x)
        potential = odd_potential_from_interior(
            x, background, charge, coupling, ell
        )
        psi = y[:4].reshape(2, 2)
        momentum = y[4:8].reshape(2, 2)
        psi_omega = y[8:12].reshape(2, 2)
        momentum_omega = y[12:16].reshape(2, 2)
        return np.concatenate(
            [
                (momentum / radial_factor).ravel(),
                (
                    (potential - omega * omega * np.eye(2))
                    @ psi
                    / radial_factor
                ).ravel(),
                (momentum_omega / radial_factor).ravel(),
                (
                    (
                        (potential - omega * omega * np.eye(2))
                        @ psi_omega
                        - 2.0 * omega * psi
                    )
                    / radial_factor
                ).ravel(),
            ]
        )

    solution = solve_ivp(
        radial_rhs,
        (start, stop),
        y0,
        method="DOP853",
        rtol=2.0e-13,
        atol=2.0e-15,
        max_step=5.0e-4,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    final = np.asarray(solution.sol(stop), dtype=complex)
    psi = final[:4].reshape(2, 2)
    momentum = final[4:8].reshape(2, 2)
    psi_omega = final[8:12].reshape(2, 2)
    final_flux = (psi.conj().T @ momentum - momentum.conj().T @ psi) / (2j)
    flux_error = float(np.max(np.abs(final_flux - initial_flux)))

    background = np.asarray(interior.solution.sol(stop), dtype=float)
    background_prime = interior_background_derivatives(
        stop, background, charge, coupling
    )
    components = _physical_component_matrices(
        stop,
        background,
        background_prime,
        psi,
        momentum,
        omega,
        coupling,
        ell,
    )
    row = np.asarray(components["soft_magnetic"], dtype=complex)
    reference = int(np.argmax(np.abs(psi[1, :])))
    if abs(psi[1, reference]) <= 1.0e-30:
        raise RuntimeError("the soft Maxwell row vanished during reconstruction")
    reconstruction = row[reference] / (-1j * omega * psi[1, reference])
    row_omega_raw = (
        row / omega - 1j * reconstruction * omega * psi_omega[1, :]
    )
    common_phase_rate = float(
        np.imag(np.vdot(row, row_omega_raw)) / np.vdot(row, row).real
    )
    row_omega = row_omega_raw - 1j * common_phase_rate * row

    points = np.linspace(max(start + 2.0e-5, 0.02), stop - 1.0e-3, 41)
    defects: list[float] = []
    for point in points:
        step = 1.0e-5 * (1.0 + float(point))
        numerical = (
            -np.asarray(solution.sol(point + 2.0 * step))
            + 8.0 * np.asarray(solution.sol(point + step))
            - 8.0 * np.asarray(solution.sol(point - step))
            + np.asarray(solution.sol(point - 2.0 * step))
        ) / (12.0 * step)
        exact = radial_rhs(float(point), np.asarray(solution.sol(point)))
        defects.append(
            float(
                np.linalg.norm(numerical[:8] - exact[:8])
                / max(1.0, np.linalg.norm(exact[:8]))
            )
        )

    return {
        "omega_M": float(omega),
        "soft_magnetic_row": row,
        "soft_magnetic_norm": float(np.linalg.norm(row)),
        "frequency_sensitivity_row": row_omega,
        "frequency_sensitivity_norm": float(np.linalg.norm(row_omega)),
        "raw_frequency_sensitivity_norm": float(np.linalg.norm(row_omega_raw)),
        "common_phase_rate": common_phase_rate,
        "flux_residual": flux_error,
        "strong_equation_defect": max(defects),
        "integration_steps": int(solution.t.size),
        "matching_log_z": stop,
    }
