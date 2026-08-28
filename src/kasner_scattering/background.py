"""Static EMS backgrounds on the scalarized-connected fundamental branch.

The equations use

    ds^2 = -N(r) exp(-2 chi(r)) dt^2 + dr^2/N(r) + r^2 dOmega^2,
    N(r) = 1 - 2 m(r)/r,

with event horizon r_H = 1.  The scalar normalization is the one used in
arXiv:2512.19377.  The public critical point is the exact asymptotic RN
zero mode.  Nonlinear exterior shots use ``u=1/r`` and impose the scalar
source at the true compactified boundary ``u=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log, sqrt
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import mpmath as mp


@dataclass(frozen=True)
class Coupling:
    coupling_id: str
    alpha_sq: float = 1.0

    def z(self, phi: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(phi)
        if self.coupling_id == "exp_square":
            value = np.exp(np.minimum(self.alpha_sq * x * x, 700.0))
        elif self.coupling_id == "bounded_rational":
            value = 1.0 + self.alpha_sq * x * x / (1.0 + x * x)
        elif self.coupling_id == "einstein_maxwell":
            value = np.ones_like(x, dtype=float)
        else:
            raise ValueError(f"unknown coupling: {self.coupling_id}")
        return float(value) if np.ndim(value) == 0 else value

    def d_inv_z(self, phi: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(phi)
        if self.coupling_id == "exp_square":
            value = -2.0 * self.alpha_sq * x * np.exp(
                -np.minimum(self.alpha_sq * x * x, 700.0)
            )
        elif self.coupling_id == "bounded_rational":
            z_value = self.z(x)
            dz = 2.0 * self.alpha_sq * x / (1.0 + x * x) ** 2
            value = -dz / z_value**2
        elif self.coupling_id == "einstein_maxwell":
            value = np.zeros_like(x, dtype=float)
        else:
            raise ValueError(f"unknown coupling: {self.coupling_id}")
        return float(value) if np.ndim(value) == 0 else value

    def dd_inv_z(self, phi: float | np.ndarray) -> float | np.ndarray:
        """Return the second scalar derivative of ``1/Z``."""

        x = np.asarray(phi)
        if self.coupling_id == "exp_square":
            inverse = np.exp(-np.minimum(self.alpha_sq * x * x, 700.0))
            value = (-2.0 * self.alpha_sq + 4.0 * self.alpha_sq**2 * x * x) * inverse
        elif self.coupling_id == "bounded_rational":
            z_value = self.z(x)
            dz = 2.0 * self.alpha_sq * x / (1.0 + x * x) ** 2
            ddz = 2.0 * self.alpha_sq * (1.0 - 3.0 * x * x) / (1.0 + x * x) ** 3
            value = 2.0 * dz * dz / z_value**3 - ddz / z_value**2
        elif self.coupling_id == "einstein_maxwell":
            value = np.zeros_like(x, dtype=float)
        else:
            raise ValueError(f"unknown coupling: {self.coupling_id}")
        return float(value) if np.ndim(value) == 0 else value

    def d_log_z(self, phi: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(phi)
        if self.coupling_id == "exp_square":
            value = 2.0 * self.alpha_sq * x
        elif self.coupling_id == "bounded_rational":
            dz = 2.0 * self.alpha_sq * x / (1.0 + x * x) ** 2
            value = dz / self.z(x)
        elif self.coupling_id == "einstein_maxwell":
            value = np.zeros_like(x, dtype=float)
        else:
            raise ValueError(f"unknown coupling: {self.coupling_id}")
        return float(value) if np.ndim(value) == 0 else value

    def dd_log_z(self, phi: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(phi)
        if self.coupling_id == "exp_square":
            value = np.full_like(x, 2.0 * self.alpha_sq, dtype=float)
        elif self.coupling_id == "bounded_rational":
            z_value = self.z(x)
            dz = 2.0 * self.alpha_sq * x / (1.0 + x * x) ** 2
            ddz = (
                2.0
                * self.alpha_sq
                * (1.0 - 3.0 * x * x)
                / (1.0 + x * x) ** 3
            )
            value = ddz / z_value - (dz / z_value) ** 2
        elif self.coupling_id == "einstein_maxwell":
            value = np.zeros_like(x, dtype=float)
        else:
            raise ValueError(f"unknown coupling: {self.coupling_id}")
        return float(value) if np.ndim(value) == 0 else value


@dataclass
class ExteriorSolution:
    coupling: Coupling
    phi_h: float
    charge: float
    mass: float
    charge_to_mass: float
    chi_infinity: float
    scalar_charge: float
    source_residual: float
    solution: object

    @property
    def chi_h_normalized(self) -> float:
        return -self.chi_infinity


@dataclass
class InteriorSolution:
    exterior: ExteriorSolution
    log_z: np.ndarray
    phi: np.ndarray
    beta: np.ndarray
    chi: np.ndarray
    h: np.ndarray
    residual: np.ndarray
    sigma_k_index: int
    solution: object

    @property
    def sigma_k(self) -> float:
        return float(self.log_z[self.sigma_k_index])

    @property
    def beta_k(self) -> float:
        return float(self.beta[self.sigma_k_index])


def rn_mass(charge: float, horizon_radius: float = 1.0) -> float:
    return 0.5 * (horizon_radius + charge * charge / horizon_radius)


def rn_charge_to_mass(charge: float) -> float:
    return charge / rn_mass(charge)


def _horizon_scalar_derivative(
    phi_h: float, charge: float, coupling: Coupling
) -> float:
    z_h = coupling.z(phi_h)
    n_prime = 1.0 - charge * charge / z_h
    if n_prime <= 0.0:
        raise ValueError("the horizon is extremal or super-extremal")
    return charge * charge * coupling.d_inv_z(phi_h) / (2.0 * n_prime)


def _integrate_exterior(
    charge: float,
    phi_h: float,
    coupling: Coupling,
    *,
    r_max: float = 300.0,
    rtol: float = 2.0e-12,
    atol: float = 1.0e-14,
) -> tuple[float, object]:
    z_h = coupling.z(phi_h)
    if charge * charge >= z_h:
        return float("nan"), None

    phi_prime_h = _horizon_scalar_derivative(phi_h, charge, coupling)
    eps = 1.0e-9
    mass_prime_h = charge * charge / (2.0 * z_h)
    y0 = np.array(
        [
            0.5 + mass_prime_h * eps,
            phi_h + phi_prime_h * eps,
            phi_prime_h,
            -phi_prime_h * phi_prime_h * eps,
        ],
        dtype=float,
    )

    def rhs(radius: float, y: np.ndarray) -> np.ndarray:
        mass, phi, phi_prime, chi = y
        z_value = coupling.z(phi)
        lapse = 1.0 - 2.0 * mass / radius
        mass_prime = (
            0.5 * radius * radius * lapse * phi_prime * phi_prime
            + charge * charge / (2.0 * radius * radius * z_value)
        )
        lapse_prime = -2.0 * mass_prime / radius + 2.0 * mass / radius**2
        chi_prime = -radius * phi_prime * phi_prime
        scalar_force = (
            charge
            * charge
            * coupling.d_inv_z(phi)
            / (2.0 * radius**4 * lapse)
        )
        phi_second = scalar_force - (
            -chi_prime + 2.0 / radius + lapse_prime / lapse
        ) * phi_prime
        return np.array([mass_prime, phi_prime, phi_second, chi_prime])

    solution = solve_ivp(
        rhs,
        (1.0 + eps, r_max),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=3.0,
    )
    if not solution.success:
        return float("nan"), solution
    phi_end = solution.y[1, -1]
    phi_prime_end = solution.y[2, -1]
    return float(phi_end + r_max * phi_prime_end), solution


def _integrate_exterior_compactified(
    charge: float,
    phi_h: float,
    coupling: Coupling,
    *,
    rtol: float = 2.0e-12,
    atol: float = 1.0e-14,
    max_step: float = 0.05,
) -> tuple[float, object]:
    """Integrate the nonlinear exterior to the true boundary ``u=1/r=0``.

    The state is ``(m, phi, dphi/du, chi)``.  In this representation the
    source-free asymptotic condition is simply ``phi(0)=0`` and the ADM mass
    and scalar charge are read directly at the endpoint.
    """

    z_h = coupling.z(phi_h)
    if charge * charge >= z_h:
        return float("nan"), None
    phi_prime_h = _horizon_scalar_derivative(phi_h, charge, coupling)
    phi_u_h = -phi_prime_h
    # The near-critical branch amplifies horizon truncation errors in the
    # interior Kasner slope.  Starting at 1e-10 keeps the first-order exterior
    # horizon remainder below the binary64 integration floor for delta>=1e-9.
    eps = 1.0e-12
    xi0 = log(eps)
    y0 = np.array(
        [
            0.5 + charge * charge * eps / (2.0 * z_h),
            phi_h - phi_u_h * eps,
            phi_u_h,
            -phi_u_h * phi_u_h * eps,
        ],
        dtype=float,
    )

    def rhs(xi_value: float, y: np.ndarray) -> np.ndarray:
        one_minus_u = exp(xi_value)
        u_value = 1.0 - one_minus_u
        mass, phi, phi_u, chi = y
        z_value = coupling.z(phi)
        lapse = 1.0 - 2.0 * mass * u_value
        mass_u = -0.5 * lapse * phi_u * phi_u - charge * charge / (2.0 * z_value)
        chi_u = u_value * phi_u * phi_u
        lapse_u = -2.0 * mass - 2.0 * u_value * mass_u
        weighted_u = lapse_u - lapse * chi_u
        phi_uu = (
            0.5 * charge * charge * coupling.d_inv_z(phi) - weighted_u * phi_u
        ) / lapse
        factor = -one_minus_u
        return factor * np.array([mass_u, phi_u, phi_uu, chi_u])

    solution = solve_ivp(
        rhs,
        (xi0, 0.0),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not solution.success:
        return float("nan"), solution
    return float(solution.y[1, -1]), solution


def _first_charge_bracket(
    boundary_value: Callable[[float], float], upper: float
) -> tuple[float, float]:
    grid = np.linspace(0.02, upper, 100)
    previous_q = grid[0]
    previous_value = boundary_value(previous_q)
    for charge in grid[1:]:
        value = boundary_value(charge)
        if np.isfinite(previous_value) and np.isfinite(value):
            if previous_value == 0.0 or previous_value * value < 0.0:
                return float(previous_q), float(charge)
        previous_q, previous_value = charge, value
    raise RuntimeError("no fundamental scalarized charge bracket was found")


@lru_cache(maxsize=32)
def cutoff_critical_horizon_charge(
    alpha_sq: float = 1.0, r_max: float = 300.0
) -> float:
    """Return the finite-cutoff Robin approximation to the RN zero mode.

    This function is retained solely for convergence comparisons. It must not be
    used to define the physical distance ``delta`` from scalarization.
    """

    def linear_boundary(charge: float) -> float:
        n_prime_h = 1.0 - charge * charge
        phi_prime_h = -alpha_sq * charge * charge / n_prime_h
        eps = 1.0e-9

        def rhs(radius: float, y: np.ndarray) -> np.ndarray:
            phi, phi_prime = y
            lapse = (1.0 - 1.0 / radius) * (
                1.0 - charge * charge / radius
            )
            lapse_prime = (
                (1.0 / radius**2) * (1.0 - charge * charge / radius)
                + (1.0 - 1.0 / radius) * charge * charge / radius**2
            )
            phi_second = -(
                2.0 / radius + lapse_prime / lapse
            ) * phi_prime - alpha_sq * charge * charge * phi / (
                radius**4 * lapse
            )
            return np.array([phi_prime, phi_second])

        sol = solve_ivp(
            rhs,
            (1.0 + eps, r_max),
            [1.0 + phi_prime_h * eps, phi_prime_h],
            method="DOP853",
            rtol=2.0e-12,
            atol=1.0e-14,
            max_step=3.0,
        )
        return float(sol.y[0, -1] + r_max * sol.y[1, -1])

    upper = 1.0 - 1.0e-6
    bracket = _first_charge_bracket(linear_boundary, upper)
    return float(brentq(linear_boundary, *bracket, xtol=2.0e-14, rtol=2.0e-14))


@lru_cache(maxsize=32)
def exact_critical_horizon_charge(
    alpha_sq: float = 1.0, precision_digits: int = 100
) -> mp.mpf:
    """Return the asymptotic fundamental RN zero-mode charge for ``r_H=1``.

    The horizon-regular solution is a Legendre function of degree ``nu``
    satisfying ``nu (nu+1)=-alpha_sq``.  Its value at infinity vanishes at
    the scalarization threshold.  The calculation is performed afresh at the
    requested precision and is independent of a finite outer boundary.
    """

    if alpha_sq <= 0.25:
        raise ValueError("the fundamental scalar zero mode requires alpha_sq > 1/4")
    if precision_digits < 40:
        raise ValueError("precision_digits must be at least 40")
    with mp.workdps(precision_digits):
        alpha = mp.mpf(str(alpha_sq))
        degree = (-1 + mp.sqrt(1 - 4 * alpha)) / 2

        def source(a_value: mp.mpf) -> mp.mpf:
            argument = (1 + a_value) / (1 - a_value)
            return mp.re(mp.legenp(degree, 0, argument, type=3))

        # Locate the first sign change in Q_H^2.  Keeping this search in the
        # exact Legendre representation avoids importing any finite-radius
        # bracket from the nonlinear shooting code.
        grid = [mp.mpf("0.02") + j * mp.mpf("0.97") / 600 for j in range(601)]
        left = grid[0]
        f_left = source(left)
        bracket: tuple[mp.mpf, mp.mpf] | None = None
        for right in grid[1:]:
            f_right = source(right)
            if f_left == 0 or f_left * f_right < 0:
                bracket = (left, right)
                break
            left, f_left = right, f_right
        if bracket is None:
            raise RuntimeError("no asymptotic fundamental scalar zero mode was found")
        a_root = mp.findroot(source, bracket, solver="anderson")
        return +mp.sqrt(a_root)


def critical_horizon_charge(
    alpha_sq: float = 1.0, r_max: float | None = None
) -> float:
    """Return the exact asymptotic RN threshold charge.

    ``r_max`` is accepted only for source compatibility with earlier callers;
    it has no effect.  Finite-cutoff studies must call
    :func:`cutoff_critical_horizon_charge` explicitly.
    """

    del r_max
    return float(exact_critical_horizon_charge(alpha_sq, 100))


def solve_exterior(
    phi_h: float,
    coupling: Coupling,
    *,
    r_max: float = 300.0,
    rtol: float = 2.0e-12,
    atol: float = 1.0e-14,
    root_xtol: float = 2.0e-13,
    root_rtol: float = 2.0e-13,
    max_step: float = 0.05,
) -> ExteriorSolution:
    """Shoot the source-free fundamental scalarized branch for fixed phi_H."""

    if phi_h <= 0.0:
        raise ValueError("phi_h must be positive on the selected Z2 branch")

    def boundary(charge: float) -> float:
        value, _ = _integrate_exterior_compactified(
            charge,
            phi_h,
            coupling,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        return value

    upper = min(sqrt(coupling.z(phi_h)) * (1.0 - 1.0e-6), 1.25)
    q_critical = critical_horizon_charge(coupling.alpha_sq)
    # A wide interval can contain the first two scalar zero modes, giving the
    # same sign at both ends.  Stay close to the fundamental zero mode and only
    # fall back to a scan when the nonlinear branch has moved substantially.
    width = max(0.025, 0.055 * q_critical)
    left = max(0.02, q_critical - width)
    right = min(upper, q_critical + width)
    try:
        if boundary(left) * boundary(right) >= 0.0:
            left, right = _first_charge_bracket(boundary, upper)
    except (FloatingPointError, ValueError):
        left, right = _first_charge_bracket(boundary, upper)

    charge = float(brentq(boundary, left, right, xtol=root_xtol, rtol=root_rtol))
    residual, solution = _integrate_exterior_compactified(
        charge,
        phi_h,
        coupling,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    mass, _phi_source, scalar_charge, chi_infinity = solution.y[:, -1]
    return ExteriorSolution(
        coupling=coupling,
        phi_h=phi_h,
        charge=charge,
        mass=float(mass),
        charge_to_mass=float(charge / mass),
        chi_infinity=float(chi_infinity),
        scalar_charge=float(scalar_charge),
        source_residual=float(residual),
        solution=solution,
    )


def solve_exterior_cutoff(
    phi_h: float,
    coupling: Coupling,
    *,
    r_max: float,
    rtol: float = 2.0e-12,
    atol: float = 1.0e-14,
) -> ExteriorSolution:
    """Reproduce the finite-radius Robin shooting convention.

    This entry point exists only for convergence comparison. It cannot define
    the asymptotic scalarization threshold.
    """

    def boundary(charge: float) -> float:
        residual, _ = _integrate_exterior(
            charge,
            phi_h,
            coupling,
            r_max=r_max,
            rtol=rtol,
            atol=atol,
        )
        return residual

    upper = min(sqrt(coupling.z(phi_h)) * (1.0 - 1.0e-6), 1.25)
    left, right = _first_charge_bracket(boundary, upper)
    charge = float(brentq(boundary, left, right, xtol=2.0e-14, rtol=2.0e-14))
    residual, solution = _integrate_exterior(
        charge,
        phi_h,
        coupling,
        r_max=r_max,
        rtol=rtol,
        atol=atol,
    )
    mass, _phi, _phi_prime, chi_infinity = solution.y[:, -1]
    scalar_charge = -r_max * r_max * solution.y[2, -1]
    return ExteriorSolution(
        coupling=coupling,
        phi_h=phi_h,
        charge=charge,
        mass=float(mass),
        charge_to_mass=float(charge / mass),
        chi_infinity=float(chi_infinity),
        scalar_charge=float(scalar_charge),
        source_residual=float(residual),
        solution=solution,
    )


def critical_charge_to_mass(
    alpha_sq: float = 1.0, r_max: float | None = None
) -> float:
    """Return the exact asymptotic critical charge-to-mass ratio."""

    return rn_charge_to_mass(critical_horizon_charge(alpha_sq, r_max))


def cutoff_critical_charge_to_mass(
    alpha_sq: float = 1.0, r_max: float = 300.0
) -> float:
    """Return the finite-cutoff comparison value of the critical ratio."""

    return rn_charge_to_mass(cutoff_critical_horizon_charge(alpha_sq, r_max))


def solve_branch_at_delta(
    delta: float,
    coupling: Coupling,
    *,
    r_max: float = 300.0,
) -> ExteriorSolution:
    """Invert the scalarized branch to q/q_c-1 = delta."""

    if delta <= 0.0:
        raise ValueError("delta must be positive")
    q_critical = critical_charge_to_mass(coupling.alpha_sq)
    cache: dict[float, ExteriorSolution] = {}
    tight = delta <= 1.0e-8
    exterior_kwargs = {
        "r_max": r_max,
        "rtol": 2.0e-13 if tight else 2.0e-12,
        "atol": 5.0e-16 if tight else 1.0e-14,
        "root_xtol": 5.0e-15 if tight else 2.0e-13,
        "root_rtol": 5.0e-15 if tight else 2.0e-13,
        "max_step": 0.05,
    }

    def objective(log_phi_h: float) -> float:
        phi_h = exp(log_phi_h)
        # The near-critical branch resolves O(delta) changes in q/M from an
        # O(sqrt(delta)) horizon scalar.  Rounding log(phi_H) here used to
        # quantize the outer shooting problem at precisely the 10^-9 anchor.
        # Cache the actual binary64 abscissa returned by Brent instead.
        key = float(log_phi_h)
        if key not in cache:
            cache[key] = solve_exterior(phi_h, coupling, **exterior_kwargs)
        return cache[key].charge_to_mass / q_critical - 1.0 - delta

    low, high = log(1.0e-6), log(0.4)
    f_low, f_high = objective(low), objective(high)
    if f_low > 0.0 or f_high < 0.0:
        raise RuntimeError(
            f"failed to bracket phi_h for delta={delta}: {f_low}, {f_high}"
        )
    root = float(
        brentq(
            objective,
            low,
            high,
            xtol=5.0e-14 if tight else 2.0e-11,
            rtol=5.0e-15 if tight else 2.0e-12,
        )
    )
    key = float(root)
    return cache.get(key) or solve_exterior(exp(root), coupling, **exterior_kwargs)


def solve_interior(
    exterior: ExteriorSolution,
    *,
    log_z_max: float = 8.0,
    kasner_tolerance: float = 1.0e-6,
    sustain_log_z: float = 1.0,
    rtol: float = 2.0e-11,
    atol: float = 1.0e-13,
    max_step: float = 0.005,
    horizon_series_order: int = 1,
) -> InteriorSolution:
    """Continue a shot exterior through the horizon in x=log(z)."""

    coupling = exterior.coupling
    phi_h = exterior.phi_h
    charge = exterior.charge
    chi_h = exterior.chi_h_normalized
    z_h_value = coupling.z(phi_h)
    h_prime_h = exp(-chi_h) * (-1.0 + charge * charge / z_h_value)
    phi_z_h = (
        exp(-chi_h)
        * charge
        * charge
        * coupling.d_inv_z(phi_h)
        / (2.0 * h_prime_h)
    )
    eps = 1.0e-10
    z0 = 1.0 + eps
    x0 = log(z0)
    if horizon_series_order not in (1, 2):
        raise ValueError("horizon_series_order must be 1 or 2")
    if horizon_series_order == 1:
        y0 = np.array(
            [
                phi_h + phi_z_h * eps,
                z0 * phi_z_h,
                chi_h + phi_z_h * phi_z_h * eps,
                h_prime_h * eps,
            ]
        )
    else:
        inverse_z_prime = coupling.d_inv_z(phi_h)
        inverse_z_second = coupling.dd_inv_z(phi_h)
        bracket0 = -1.0 + charge * charge / z_h_value
        bracket1 = 2.0 + charge * charge * inverse_z_prime * phi_z_h
        h_second_coefficient = 0.5 * exp(-chi_h) * (
            bracket1 - phi_z_h * phi_z_h * bracket0
        )
        electric_first = 0.5 * exp(-chi_h) * charge * charge * (
            (1.0 - phi_z_h * phi_z_h) * inverse_z_prime
            + inverse_z_second * phi_z_h
        )
        beta_first = (
            electric_first - 2.0 * h_second_coefficient * phi_z_h
        ) / (2.0 * h_prime_h)
        phi_second_coefficient = 0.5 * (beta_first - phi_z_h)
        chi_second_coefficient = 0.5 * (
            2.0 * phi_z_h * beta_first - phi_z_h * phi_z_h
        )
        y0 = np.array(
            [
                phi_h + phi_z_h * eps + phi_second_coefficient * eps**2,
                phi_z_h + beta_first * eps,
                chi_h + phi_z_h * phi_z_h * eps + chi_second_coefficient * eps**2,
                h_prime_h * eps + h_second_coefficient * eps**2,
            ]
        )

    def rhs(log_z_value: float, y: np.ndarray) -> np.ndarray:
        phi, beta, chi, h_value = y
        z_coord = exp(log_z_value)
        e_minus_chi = exp(-chi) if chi < 740.0 else 0.0
        z_value = coupling.z(phi)
        h_x_curvature = -e_minus_chi / z_coord
        h_x_electric = e_minus_chi * charge * charge * z_coord / z_value
        h_x = h_x_curvature + h_x_electric
        beta_x = -h_x * beta / h_value + (
            z_coord
            * e_minus_chi
            * charge
            * charge
            * coupling.d_inv_z(phi)
            / (2.0 * h_value)
        )
        return np.array([beta, beta_x, beta * beta, h_x])

    solution = solve_ivp(
        rhs,
        (x0, log_z_max),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    def kasner_residual(x_value: float, state: np.ndarray) -> float:
        derivative = rhs(float(x_value), state)
        beta_value = state[1]
        h_value = state[3]
        z_coord = exp(float(x_value))
        e_minus_chi = exp(-float(state[2])) if state[2] < 740.0 else 0.0
        z_value = coupling.z(float(state[0]))
        curvature_ratio = abs((-e_minus_chi / z_coord) / h_value)
        electric_ratio = abs(
            (e_minus_chi * charge * charge * z_coord / z_value) / h_value
        )
        beta_residual = abs(derivative[1] / beta_value) if beta_value else np.inf
        h_residual = abs(derivative[3] / h_value) if h_value else np.inf
        return max(
            beta_residual, h_residual, curvature_ratio, electric_ratio
        )

    # Locate the geometric matching surface on a deterministic dense-output
    # grid.  Selecting an adaptive solver node made Sigma_K drift when the ODE
    # tolerance changed, even though the underlying dense solutions agreed.
    scan_step = min(1.0e-3, max_step)
    scan = np.arange(x0, log_z_max - sustain_log_z + scan_step, scan_step)
    scan = scan[scan <= log_z_max - sustain_log_z]
    scan_residual = np.asarray(
        [kasner_residual(float(value), solution.sol(float(value))) for value in scan]
    )
    sigma_value: float | None = None
    for index in range(1, len(scan)):
        if scan_residual[index - 1] >= kasner_tolerance > scan_residual[index]:
            left, right = float(scan[index - 1]), float(scan[index])
            candidate = float(
                brentq(
                    lambda value: kasner_residual(value, solution.sol(value))
                    - kasner_tolerance,
                    left,
                    right,
                    xtol=2.0e-14,
                    rtol=2.0e-14,
                )
            )
            candidate = float(np.nextafter(candidate, right))
            sustain = np.linspace(candidate, candidate + sustain_log_z, 65)
            if max(
                kasner_residual(float(value), solution.sol(float(value)))
                for value in sustain
            ) <= kasner_tolerance * (1.0 + 1.0e-8):
                sigma_value = candidate
                break
    if sigma_value is None:
        # A solution can enter the plateau before the first scanned point.
        sustain = np.linspace(x0, x0 + sustain_log_z, 65)
        if max(
            kasner_residual(float(value), solution.sol(float(value)))
            for value in sustain
        ) < kasner_tolerance:
            sigma_value = x0
    if sigma_value is None:
        raise RuntimeError("no sustained Kasner matching surface was found")

    log_z_values = solution.t
    phi, beta, chi, h_values = solution.y
    residual = np.asarray(
        [
            kasner_residual(float(x_value), state)
            for x_value, state in zip(log_z_values, solution.y.T)
        ]
    )
    sigma_index = int(np.searchsorted(log_z_values, sigma_value))
    if sigma_index >= len(log_z_values) or abs(log_z_values[sigma_index] - sigma_value) > 1.0e-14:
        state = np.asarray(solution.sol(sigma_value), dtype=float)
        log_z_values = np.insert(log_z_values, sigma_index, sigma_value)
        phi = np.insert(phi, sigma_index, state[0])
        beta = np.insert(beta, sigma_index, state[1])
        chi = np.insert(chi, sigma_index, state[2])
        h_values = np.insert(h_values, sigma_index, state[3])
        residual = np.insert(
            residual, sigma_index, kasner_residual(sigma_value, state)
        )

    return InteriorSolution(
        exterior=exterior,
        log_z=log_z_values,
        phi=phi,
        beta=beta,
        chi=chi,
        h=h_values,
        residual=residual,
        sigma_k_index=sigma_index,
        solution=solution,
    )


def critical_slope(deltas: np.ndarray, values: np.ndarray) -> float:
    """Return the least-squares logarithmic power in values ~ delta^slope."""

    return float(np.polyfit(np.log(deltas), np.log(np.abs(values)), 1)[0])


def direct_zero_mode_source(
    horizon_charge: float,
    *,
    horizon_epsilon: float = 1.0e-8,
    rtol: float = 2.0e-13,
) -> float:
    """Evaluate the RN scalar source by direct compactified integration."""

    charge_squared = horizon_charge * horizon_charge
    start = 1.0 - horizon_epsilon
    derivative_horizon = charge_squared / (1.0 - charge_squared)
    initial = [1.0 - derivative_horizon * horizon_epsilon, derivative_horizon]

    def zero_mode_rhs(u_value: float, state: np.ndarray) -> np.ndarray:
        scalar, scalar_u = state
        coefficient = (1.0 - u_value) * (1.0 - charge_squared * u_value)
        coefficient_u = -(1.0 + charge_squared) + 2.0 * charge_squared * u_value
        return np.asarray(
            [
                scalar_u,
                (-coefficient_u * scalar_u - charge_squared * scalar)
                / coefficient,
            ],
            dtype=float,
        )

    solution = solve_ivp(
        zero_mode_rhs,
        (start, 0.0),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=rtol * 1.0e-2,
        max_step=0.01,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return float(solution.y[0, -1])


def direct_critical_charge_to_mass() -> tuple[float, list[dict[str, float]]]:
    """Solve the compactified zero-mode equation at three resolutions."""

    records: list[dict[str, float]] = []
    for epsilon, tolerance in (
        (1.0e-6, 2.0e-11),
        (1.0e-7, 2.0e-12),
        (1.0e-8, 2.0e-13),
    ):
        root = brentq(
            lambda charge: direct_zero_mode_source(
                charge,
                horizon_epsilon=epsilon,
                rtol=tolerance,
            ),
            0.89,
            0.90,
            xtol=5.0e-15,
            rtol=5.0e-15,
        )
        records.append(
            {
                "horizon_epsilon": epsilon,
                "relative_tolerance": tolerance,
                "horizon_charge": root,
                "critical_charge_to_mass": rn_charge_to_mass(root),
                "source_residual": direct_zero_mode_source(
                    root,
                    horizon_epsilon=epsilon,
                    rtol=tolerance,
                ),
            }
        )
    return records[-1]["critical_charge_to_mass"], records
