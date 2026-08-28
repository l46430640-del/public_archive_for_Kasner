"""Constraint-complete diagonal Bianchi-I EMS wall Hamiltonian.

The canonical scalar is ``varphi=sqrt(2)*psi_EMS``. This is the normalization
in which the Kasner constraint reads ``sum(p_a^2)+p_varphi^2=1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import cosh, exp, isfinite, log, log1p, pi, sqrt, tanh
from typing import Any

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import eval_legendre
import sympy as sp


@dataclass(frozen=True)
class WallHamiltonian:
    coupling_id: str
    alpha_sq: float = 1.0
    magnetic_coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)
    electric_coefficients: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.alpha_sq < 0.0:
            raise ValueError("alpha_sq must be non-negative")
        if self.coupling_id not in {
            "exp_square",
            "bounded_rational",
            "constant",
            "linear_dilaton",
        }:
            raise ValueError(f"unsupported coupling: {self.coupling_id}")
        if any(value < 0.0 for value in self.magnetic_coefficients):
            raise ValueError("magnetic coefficients are squared fluxes")
        if any(value < 0.0 for value in self.electric_coefficients):
            raise ValueError("electric coefficients are squared displacements")

    def log_z(self, varphi: float) -> float:
        psi = varphi / sqrt(2.0)
        if self.coupling_id == "exp_square":
            return self.alpha_sq * psi * psi
        if self.coupling_id == "bounded_rational":
            return log1p(self.alpha_sq * psi * psi / (1.0 + psi * psi))
        if self.coupling_id == "linear_dilaton":
            return sqrt(2.0) * self.alpha_sq * psi
        return 0.0

    def d_log_z_d_varphi(self, varphi: float) -> float:
        if self.coupling_id == "exp_square":
            return self.alpha_sq * varphi
        if self.coupling_id == "bounded_rational":
            psi = varphi / sqrt(2.0)
            denominator = (1.0 + psi * psi) * (
                1.0 + (1.0 + self.alpha_sq) * psi * psi
            )
            return sqrt(2.0) * self.alpha_sq * psi / denominator
        if self.coupling_id == "linear_dilaton":
            return self.alpha_sq
        return 0.0


def kinetic_hamiltonian(momentum: np.ndarray) -> float:
    """Return 1/2(pi.G^-1.pi + pi_varphi^2)."""

    momentum = np.asarray(momentum, dtype=float)
    if momentum.shape != (4,):
        raise ValueError("momentum must be (pi_1,pi_2,pi_3,pi_varphi)")
    spatial = momentum[:3]
    return 0.5 * (
        float(spatial @ spatial)
        - 0.5 * float(np.sum(spatial)) ** 2
        + float(momentum[3]) ** 2
    )


def velocities_from_momenta(momentum: np.ndarray) -> np.ndarray:
    momentum = np.asarray(momentum, dtype=float)
    spatial = momentum[:3]
    volume_momentum = float(np.sum(spatial))
    return np.array(
        [
            spatial[0] - 0.5 * volume_momentum,
            spatial[1] - 0.5 * volume_momentum,
            spatial[2] - 0.5 * volume_momentum,
            momentum[3],
        ],
        dtype=float,
    )


def kasner_momenta(kasner: np.ndarray) -> np.ndarray:
    kasner = np.asarray(kasner, dtype=float)
    if kasner.shape != (4,):
        raise ValueError("kasner must be (p1,p2,p3,p_varphi)")
    volume_rate = float(np.sum(kasner[:3]))
    return np.array(
        [
            kasner[0] - volume_rate,
            kasner[1] - volume_rate,
            kasner[2] - volume_rate,
            kasner[3],
        ],
        dtype=float,
    )


def _safe_exp(log_value: float) -> float:
    if log_value > 700.0:
        return float("inf")
    if log_value < -745.0:
        return 0.0
    return exp(log_value)


def wall_log_values(
    coordinates: np.ndarray, model: WallHamiltonian
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=float)
    beta = coordinates[:3]
    log_z_value = model.log_z(float(coordinates[3]))
    magnetic = np.full(3, -np.inf, dtype=float)
    electric = np.full(3, -np.inf, dtype=float)
    for axis, coefficient in enumerate(model.magnetic_coefficients):
        if coefficient > 0.0:
            magnetic[axis] = np.log(coefficient) + log_z_value - 2.0 * beta[axis]
    for axis, coefficient in enumerate(model.electric_coefficients):
        if coefficient > 0.0:
            electric[axis] = np.log(coefficient) - log_z_value - 2.0 * beta[axis]
    return magnetic, electric


def wall_values(
    coordinates: np.ndarray, model: WallHamiltonian
) -> tuple[np.ndarray, np.ndarray]:
    magnetic_logs, electric_logs = wall_log_values(coordinates, model)
    return (
        np.asarray([_safe_exp(value) for value in magnetic_logs]),
        np.asarray([_safe_exp(value) for value in electric_logs]),
    )


def hamiltonian(state: np.ndarray, model: WallHamiltonian) -> float:
    state = np.asarray(state, dtype=float)
    magnetic, electric = wall_values(state[:4], model)
    return kinetic_hamiltonian(state[4:]) + float(np.sum(magnetic) + np.sum(electric))


def rhs(_s: float, state: np.ndarray, model: WallHamiltonian) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    coordinates = state[:4]
    momentum = state[4:]
    magnetic, electric = wall_values(coordinates, model)
    dlogz = model.d_log_z_d_varphi(float(coordinates[3]))
    derivative = np.zeros(8, dtype=float)
    derivative[:4] = velocities_from_momenta(momentum)
    derivative[4:7] = 2.0 * (magnetic + electric)
    derivative[7] = -dlogz * float(np.sum(magnetic) - np.sum(electric))
    return derivative


def constraint_complete_initial_state(
    kasner: np.ndarray,
    coordinates: np.ndarray,
    model: WallHamiltonian,
) -> np.ndarray:
    """Adjust only volume momentum along the branch continuous at zero flux."""

    kasner = np.asarray(kasner, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (4,):
        raise ValueError("coordinates must be (beta1,beta2,beta3,varphi)")
    if abs(float(np.sum(kasner[:3])) - 1.0) > 1.0e-10:
        raise ValueError("the supplied Kasner volume rate must equal one")
    if abs(float(kasner[:3] @ kasner[:3] + kasner[3] ** 2) - 1.0) > 1.0e-10:
        raise ValueError("the supplied Kasner vector violates the null constraint")
    magnetic, electric = wall_values(coordinates, model)
    potential = float(np.sum(magnetic) + np.sum(electric))
    shift = (2.0 / 3.0) * (1.0 - sqrt(1.0 + 3.0 * potential))
    momentum = kasner_momenta(kasner)
    momentum[:3] += shift
    state = np.concatenate([coordinates, momentum])
    if abs(hamiltonian(state, model)) > 5.0e-13 * max(1.0, potential):
        raise ArithmeticError("failed to select the continuous Hamiltonian root")
    return state


def normalized_kasner(momentum: np.ndarray) -> np.ndarray:
    """Return a volume-normalized velocity vector for plateau diagnostics."""

    velocity = velocities_from_momenta(momentum)
    volume_rate = float(np.sum(velocity[:3]))
    if volume_rate <= 0.0:
        raise ValueError("contracting Kasner branch requires positive beta-volume rate")
    return velocity / volume_rate


@dataclass(frozen=True)
class BounceInput:
    coupling_id: str
    alpha_sq: float
    kasner: tuple[float, float, float, float]
    varphi_k: float
    initial_wall_fraction: float
    wall_axis: int = 0
    initial_background_electric_fraction: float = 0.0
    background_electric_axis: int = 2


@dataclass(frozen=True)
class BounceConfig:
    epsilon_on: float = 1.0e-12
    epsilon_exit: float = 1.0e-12
    rtol: float = 2.0e-12
    atol: float = 2.0e-14
    max_bounce_duration: float = 1.0e5
    max_step: float = 0.1
    runaway_fraction: float = 1.0e4
    post_plateau_volume: float = 1.0
    plateau_drift_tolerance: float = 1.0e-6
    plateau_constraint_tolerance: float = 1.0e-10


@dataclass
class BounceResult:
    outcome: str
    s_linear: float | None
    s_start: float | None
    s_peak: float | None
    s_bounce: float | None
    s_exit: float | None
    pre_kasner: np.ndarray
    post_kasner: np.ndarray | None
    max_wall_fraction: float
    wall_velocity_before: float
    wall_velocity_after: float | None
    kasner_shift: float | None
    constraint_residual: float
    energy_error: float
    precision_digits: int
    final_state: np.ndarray | None
    sample_s: np.ndarray
    sample_log_wall: np.ndarray
    kasner_in_window: dict[str, float | bool] = field(default_factory=dict)
    kasner_out_window: dict[str, float | bool] = field(default_factory=dict)
    next_wall_onset: float | None = None
    evolved_channel_impulses: dict[str, np.ndarray | float] = field(default_factory=dict)
    sample_states: np.ndarray = field(default_factory=lambda: np.empty((0, 8)))


def _free_log_wall(input_data: BounceInput, s_value: float) -> float:
    kasner = np.asarray(input_data.kasner, dtype=float)
    model = WallHamiltonian(input_data.coupling_id, input_data.alpha_sq)
    varphi = input_data.varphi_k + kasner[3] * s_value
    return (
        log(input_data.initial_wall_fraction)
        + model.log_z(varphi)
        - model.log_z(input_data.varphi_k)
        - 2.0 * kasner[input_data.wall_axis] * s_value
    )


def find_linear_onset(input_data: BounceInput, epsilon_on: float) -> float | None:
    """Return the first free-flight time at which the wall reaches epsilon_on."""

    if input_data.initial_wall_fraction <= 0.0:
        return None
    if input_data.initial_wall_fraction >= epsilon_on:
        return 0.0
    target = log(epsilon_on)
    value0 = _free_log_wall(input_data, 0.0) - target

    if input_data.coupling_id == "exp_square":
        kasner = np.asarray(input_data.kasner, dtype=float)
        velocity = kasner[3]
        quadratic = 0.5 * input_data.alpha_sq * velocity * velocity
        linear = (
            input_data.alpha_sq * input_data.varphi_k * velocity
            - 2.0 * kasner[input_data.wall_axis]
        )
        constant = value0
        if quadratic == 0.0:
            return -constant / linear if linear > 0.0 else None
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return None
        roots = [
            (-linear - sqrt(discriminant)) / (2.0 * quadratic),
            (-linear + sqrt(discriminant)) / (2.0 * quadratic),
        ]
        positive = [root for root in roots if root >= 0.0]
        return min(positive) if positive else None

    upper = 1.0
    previous_s = 0.0
    previous_value = value0
    while upper <= 1.0e9:
        value = _free_log_wall(input_data, upper) - target
        if value >= 0.0 and previous_value < 0.0:
            return float(
                brentq(
                    lambda sample: _free_log_wall(input_data, sample) - target,
                    previous_s,
                    upper,
                    xtol=1.0e-12,
                )
            )
        previous_s = upper
        previous_value = value
        upper *= 2.0
    return None


def wall_velocity(state: np.ndarray, model: WallHamiltonian, axis: int) -> float:
    velocity = velocities_from_momenta(np.asarray(state[4:], dtype=float))
    return float(
        velocity[axis]
        - 0.5 * model.d_log_z_d_varphi(float(state[3])) * velocity[3]
    )


def _wall_log(state: np.ndarray, model: WallHamiltonian, axis: int) -> float:
    magnetic, _ = wall_log_values(state[:4], model)
    return float(magnetic[axis])


def _recentered_initial_state(
    input_data: BounceInput, s_on: float, start_fraction: float
) -> tuple[np.ndarray, WallHamiltonian]:
    kasner = np.asarray(input_data.kasner, dtype=float)
    varphi_on = input_data.varphi_k + kasner[3] * s_on
    base_model = WallHamiltonian(input_data.coupling_id, input_data.alpha_sq)
    coefficient = start_fraction * exp(-base_model.log_z(varphi_on))
    magnetic = [0.0, 0.0, 0.0]
    magnetic[input_data.wall_axis] = coefficient
    electric = [0.0, 0.0, 0.0]
    if input_data.initial_background_electric_fraction > 0.0:
        delta_log_z = base_model.log_z(varphi_on) - base_model.log_z(
            input_data.varphi_k
        )
        electric_fraction = input_data.initial_background_electric_fraction * exp(
            -delta_log_z
            - 2.0
            * kasner[input_data.background_electric_axis]
            * s_on
        )
        electric[input_data.background_electric_axis] = electric_fraction * exp(
            base_model.log_z(varphi_on)
        )
    model = WallHamiltonian(
        input_data.coupling_id,
        input_data.alpha_sq,
        magnetic_coefficients=tuple(magnetic),
        electric_coefficients=tuple(electric),
    )
    coordinates = np.array([0.0, 0.0, 0.0, varphi_on], dtype=float)
    return constraint_complete_initial_state(kasner, coordinates, model), model


def _kasner_residual(kasner: np.ndarray) -> float:
    kasner = np.asarray(kasner, dtype=float)
    return float(
        abs(np.sum(kasner[:3]) - 1.0)
        + abs(float(kasner[:3] @ kasner[:3] + kasner[3] ** 2) - 1.0)
    )


def _next_free_soft_onset(
    state: np.ndarray,
    model: WallHamiltonian,
    axis: int,
    threshold: float,
) -> float | None:
    """Return affine time to the next free-flight crossing of ``threshold``."""

    if model.coupling_id != "exp_square":
        return None
    velocity = velocities_from_momenta(state[4:])
    linear = (
        model.alpha_sq * float(state[3]) * velocity[3]
        - 2.0 * velocity[axis]
    )
    quadratic = 0.5 * model.alpha_sq * velocity[3] ** 2
    offset = _wall_log(state, model, axis) - log(threshold)
    if quadratic <= 0.0:
        return -offset / linear if linear > 0.0 and offset < 0.0 else None
    discriminant = linear * linear - 4.0 * quadratic * offset
    if discriminant < 0.0:
        return None
    roots = [
        (-linear - sqrt(discriminant)) / (2.0 * quadratic),
        (-linear + sqrt(discriminant)) / (2.0 * quadratic),
    ]
    positive = [root for root in roots if root > 1.0e-8]
    return min(positive) if positive else None


def solve_bounce(
    input_data: BounceInput, config: BounceConfig = BounceConfig()
) -> BounceResult:
    kasner = np.asarray(input_data.kasner, dtype=float)
    s_on = find_linear_onset(input_data, config.epsilon_on)
    if s_on is None:
        taub_identity = (
            abs(kasner[input_data.wall_axis]) < 1.0e-14
            and abs(kasner[3]) < 1.0e-14
            and np.count_nonzero(np.abs(kasner[:3] - 1.0) < 1.0e-14) == 1
            and np.count_nonzero(np.abs(kasner[:3]) < 1.0e-14) == 2
        )
        return BounceResult(
            outcome="TAUB_IDENTITY_CONTROL" if taub_identity else "NO_ONSET",
            s_linear=None,
            s_start=None,
            s_peak=None,
            s_bounce=None,
            s_exit=None,
            pre_kasner=kasner,
            post_kasner=None,
            max_wall_fraction=input_data.initial_wall_fraction,
            wall_velocity_before=float("nan"),
            wall_velocity_after=None,
            kasner_shift=None,
            constraint_residual=0.0,
            energy_error=0.0,
            precision_digits=15,
            final_state=None,
            sample_s=np.empty(0),
            sample_log_wall=np.empty(0),
        )

    start_fraction = max(config.epsilon_on, input_data.initial_wall_fraction)
    initial, model = _recentered_initial_state(input_data, s_on, start_fraction)
    u_before = wall_velocity(initial, model, input_data.wall_axis)
    initial_energy = hamiltonian(initial, model)
    if u_before >= 0.0:
        return BounceResult(
            outcome="NO_ONSET",
            s_linear=s_on,
            s_start=s_on,
            s_peak=None,
            s_bounce=None,
            s_exit=None,
            pre_kasner=kasner,
            post_kasner=None,
            max_wall_fraction=start_fraction,
            wall_velocity_before=u_before,
            wall_velocity_after=None,
            kasner_shift=None,
            constraint_residual=abs(initial_energy),
            energy_error=0.0,
            precision_digits=15,
            final_state=initial,
            sample_s=np.asarray([s_on]),
            sample_log_wall=np.asarray([log(start_fraction)]),
        )
    log_runaway = log(config.runaway_fraction)

    def augmented_rhs(time: float, augmented: np.ndarray) -> np.ndarray:
        physical = np.asarray(augmented[:8], dtype=float)
        derivative = np.zeros(16, dtype=float)
        derivative[:8] = rhs(time, physical, model)
        magnetic, electric = wall_values(physical[:4], model)
        dlogz = model.d_log_z_d_varphi(float(physical[3]))
        soft = float(magnetic[input_data.wall_axis])
        derivative[8 + input_data.wall_axis] = 2.0 * soft
        derivative[11] = -dlogz * soft
        derivative[12:15] = 2.0 * electric
        derivative[15] = dlogz * float(np.sum(electric))
        return derivative

    augmented_initial = np.concatenate([initial, np.zeros(8, dtype=float)])

    def peak_event(_time: float, state: np.ndarray) -> float:
        return wall_velocity(state[:8], model, input_data.wall_axis)

    peak_event.direction = 1.0
    peak_event.terminal = True

    def runaway_event(_time: float, state: np.ndarray) -> float:
        return _wall_log(state[:8], model, input_data.wall_axis) - log_runaway

    runaway_event.direction = 1.0
    runaway_event.terminal = True

    stage_one = solve_ivp(
        augmented_rhs,
        (0.0, config.max_bounce_duration),
        augmented_initial,
        method="DOP853",
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
        events=(peak_event, runaway_event),
        dense_output=True,
    )
    if not stage_one.success:
        outcome = "NUMERICAL_FAILURE"
        peak_time = None
        final_state = stage_one.y[:8, -1]
        stages = [stage_one]
    elif len(stage_one.t_events[1]) > 0 or len(stage_one.t_events[0]) == 0:
        outcome = "MAGNETIC_RUNAWAY"
        peak_time = None
        final_state = stage_one.y[:8, -1]
        stages = [stage_one]
    else:
        peak_time = float(stage_one.t_events[0][0])
        peak_state = np.asarray(stage_one.y_events[0][0], dtype=float)

        effective_exit = min(
            config.epsilon_exit,
            0.01 * config.plateau_constraint_tolerance,
        )

        def exit_event(_time: float, state: np.ndarray) -> float:
            return _wall_log(state[:8], model, input_data.wall_axis) - log(effective_exit)

        exit_event.direction = -1.0
        exit_event.terminal = True
        stage_two = solve_ivp(
            augmented_rhs,
            (peak_time, config.max_bounce_duration),
            peak_state,
            method="DOP853",
            rtol=config.rtol,
            atol=config.atol,
            max_step=config.max_step,
            events=(exit_event, runaway_event),
            dense_output=True,
        )
        stages = [stage_one, stage_two]
        exit_time = None
        if not stage_two.success:
            outcome = "NUMERICAL_FAILURE"
        elif len(stage_two.t_events[1]) > 0:
            outcome = "MAGNETIC_RUNAWAY"
        elif len(stage_two.t_events[0]) == 0:
            outcome = "NO_POST_PLATEAU"
        else:
            exit_time = float(stage_two.t_events[0][0])
            exit_augmented = np.asarray(stage_two.y_events[0][0], dtype=float)
            exit_state = exit_augmented[:8]
            start_volume = float(np.sum(exit_state[:3]))

            def plateau_event(_time: float, state: np.ndarray) -> float:
                return (
                    float(np.sum(state[:3]))
                    - start_volume
                    - config.post_plateau_volume
                )

            plateau_event.direction = 1.0
            plateau_event.terminal = True
            stage_three = solve_ivp(
                augmented_rhs,
                (exit_time, exit_time + config.max_bounce_duration),
                exit_augmented,
                method="DOP853",
                rtol=config.rtol,
                atol=config.atol,
                max_step=config.max_step,
                events=(plateau_event, runaway_event),
                dense_output=True,
            )
            stages.append(stage_three)
            if not stage_three.success:
                outcome = "NUMERICAL_FAILURE"
            elif len(stage_three.t_events[1]) > 0:
                outcome = "MAGNETIC_RUNAWAY"
            elif len(stage_three.t_events[0]) == 0:
                outcome = "NO_POST_PLATEAU"
            else:
                outcome = "CANDIDATE_BOUNCE"
        final_state = stage_two.y[:8, -1]
        if len(stages) == 3:
            final_state = stages[-1].y[:8, -1]

    all_states = np.hstack([stage.y[:8] for stage in stages])
    all_times = np.concatenate([stage.t for stage in stages])
    all_logs = np.asarray(
        [_wall_log(state, model, input_data.wall_axis) for state in all_states.T]
    )
    max_log = float(np.max(all_logs))
    max_fraction = exp(max_log) if max_log < 700.0 else float("inf")
    constraint_values = np.asarray(
        [abs(hamiltonian(state, model)) for state in all_states.T]
    )
    constraint_residual = float(np.max(constraint_values))
    energy_error = float(np.max(np.abs(constraint_values - abs(initial_energy))))
    u_after = wall_velocity(final_state, model, input_data.wall_axis)
    post = None
    shift = None
    kasner_in_window = {
        "duration_N": 1.0,
        "kasner_drift": 0.0,
        "kasner_residual": _kasner_residual(kasner),
        "passed": _kasner_residual(kasner) < config.plateau_constraint_tolerance,
    }
    kasner_out_window: dict[str, float | bool] = {}
    next_wall_onset = None
    if outcome == "CANDIDATE_BOUNCE":
        post = normalized_kasner(final_state[4:])
        shift = float(np.linalg.norm(post - kasner))
        exit_augmented = np.asarray(stage_two.y_events[0][0], dtype=float)
        exit_state = exit_augmented[:8]
        exit_kasner = normalized_kasner(exit_state[4:])
        out_drift = float(np.linalg.norm(post - exit_kasner))
        out_residual = _kasner_residual(post)
        plateau_logs = np.asarray(
            [
                _wall_log(state[:8], model, input_data.wall_axis)
                for state in stages[-1].y.T
            ]
        )
        plateau_wall_max = float(np.exp(min(700.0, np.max(plateau_logs))))
        plateau_passed = (
            out_drift < config.plateau_drift_tolerance
            and out_residual < config.plateau_constraint_tolerance
        )
        kasner_out_window = {
            "duration_N": config.post_plateau_volume,
            "kasner_drift": out_drift,
            "kasner_residual": out_residual,
            "maximum_wall_fraction": plateau_wall_max,
            "passed": plateau_passed,
        }
        next_local = _next_free_soft_onset(
            final_state,
            model,
            input_data.wall_axis,
            config.epsilon_on,
        )
        if next_local is not None:
            next_wall_onset = s_on + float(stages[-1].t[-1]) + next_local
        if not plateau_passed:
            outcome = "NO_POST_PLATEAU"
        elif u_before >= 0.0 or u_after <= 0.0:
            outcome = "NO_REFLECTION"
        else:
            outcome = "KASNER_TRANSITION"

    finite_times = all_times[np.isfinite(all_times)]
    evolved_channel_impulses: dict[str, np.ndarray | float] = {}
    if "exit_augmented" in locals():
        soft_impulse = np.asarray(exit_augmented[8:12], dtype=float)
        background_impulse = np.asarray(exit_augmented[12:16], dtype=float)
        actual_kick = np.asarray(exit_state[4:] - initial[4:], dtype=float)
        closure = float(
            np.linalg.norm(actual_kick - soft_impulse - background_impulse)
            / max(np.linalg.norm(actual_kick), np.finfo(float).tiny)
        )
        evolved_channel_impulses = {
            "soft_magnetic": soft_impulse,
            "background_electric": background_impulse,
            "actual_momentum_kick": actual_kick,
            "closure_error": closure,
        }
    return BounceResult(
        outcome=outcome,
        s_linear=s_on,
        s_start=s_on,
        s_peak=None if peak_time is None else s_on + peak_time,
        s_bounce=None if peak_time is None else s_on + peak_time,
        s_exit=(
            None
            if outcome in {"MAGNETIC_RUNAWAY", "NO_POST_PLATEAU", "NUMERICAL_FAILURE"}
            else s_on + float(exit_time if exit_time is not None else finite_times[-1])
        ),
        pre_kasner=kasner,
        post_kasner=post,
        max_wall_fraction=max_fraction,
        wall_velocity_before=u_before,
        wall_velocity_after=u_after,
        kasner_shift=shift,
        constraint_residual=constraint_residual,
        energy_error=energy_error,
        precision_digits=15,
        final_state=final_state,
        sample_s=s_on + all_times,
        sample_log_wall=all_logs,
        kasner_in_window=kasner_in_window,
        kasner_out_window=kasner_out_window,
        next_wall_onset=next_wall_onset,
        evolved_channel_impulses=evolved_channel_impulses,
        sample_states=all_states.T,
    )


def mpmath_constraint_residual(
    input_data: BounceInput,
    duration: float,
    *,
    epsilon_on: float = 1.0e-12,
    precision_digits: int = 80,
) -> float:
    """Independent arbitrary-precision Taylor integration at a fixed duration."""

    s_on = find_linear_onset(input_data, epsilon_on)
    if s_on is None:
        raise ValueError("the selected input has no nonlinear onset")
    mp.mp.dps = precision_digits
    alpha = mp.mpf(str(input_data.alpha_sq))
    epsilon = mp.mpf(str(epsilon_on))
    s_on_mp = mp.mpf(str(s_on))
    kasner = [mp.mpf(str(value)) for value in input_data.kasner]
    varphi_on = mp.mpf(str(input_data.varphi_k)) + kasner[3] * s_on_mp
    if input_data.coupling_id == "exp_square":
        logz_on = alpha * varphi_on * varphi_on / 2
    elif input_data.coupling_id == "constant":
        logz_on = mp.mpf("0")
    else:
        raise ValueError("mpmath anchor supports exp_square and constant controls")
    coefficient = epsilon * mp.exp(-logz_on)
    volume_rate = sum(kasner[:3])
    spatial_momentum = [value - volume_rate for value in kasner[:3]]
    base_kinetic = (
        sum(value * value for value in spatial_momentum)
        - sum(spatial_momentum) ** 2 / 2
        + kasner[3] ** 2
    ) / 2
    base_constraint = base_kinetic + epsilon
    shift = mp.mpf(2) * (
        volume_rate - mp.sqrt(volume_rate**2 + 3 * base_constraint)
    ) / 3
    spatial_momentum = [value + shift for value in spatial_momentum]
    y0 = [mp.mpf("0"), mp.mpf("0"), mp.mpf("0"), varphi_on]
    y0.extend(spatial_momentum + [kasner[3]])
    axis = input_data.wall_axis

    def derivatives(_time, state):
        beta = state[:3]
        varphi = state[3]
        momentum = state[4:]
        volume = sum(momentum[:3])
        velocity = [momentum[index] - volume / 2 for index in range(3)]
        velocity.append(momentum[3])
        if input_data.coupling_id == "exp_square":
            logz = alpha * varphi * varphi / 2
            dlogz = alpha * varphi
        elif input_data.coupling_id == "constant":
            logz = mp.mpf("0")
            dlogz = mp.mpf("0")
        else:
            raise ValueError("mpmath anchor supports exp_square and constant controls")
        wall = coefficient * mp.exp(logz - 2 * beta[axis])
        dpi = [mp.mpf("0"), mp.mpf("0"), mp.mpf("0"), -dlogz * wall]
        dpi[axis] = 2 * wall
        return velocity + dpi

    solution = mp.odefun(
        derivatives,
        mp.mpf("0"),
        y0,
        tol=mp.mpf(10) ** (-(precision_digits - 15)),
        degree=40,
    )
    final = solution(mp.mpf(str(duration)))
    spatial = final[4:7]
    kinetic = (
        sum(value * value for value in spatial)
        - sum(spatial) ** 2 / 2
        + final[7] ** 2
    ) / 2
    if input_data.coupling_id == "exp_square":
        logz_final = alpha * final[3] ** 2 / 2
    else:
        logz_final = mp.mpf("0")
    potential = coefficient * mp.exp(logz_final - 2 * final[axis])
    return float(abs(kinetic + potential))


def kasner_from_beta(beta: float) -> tuple[float, float, float, float]:
    """Return the exact scalar-Kasner family and scalar velocity."""

    denominator = beta * beta + 3.0
    p_s = 2.0 / denominator
    p_t = (beta * beta - 1.0) / denominator
    p_scalar = 2.0 * sqrt(2.0) * beta / denominator
    return p_s, p_t, p_scalar, p_scalar / sqrt(2.0)


@dataclass(frozen=True)
class SoftTransfer:
    coupling_id: str
    alpha_sq: float
    ell: int
    omega_m: float
    delta: float
    sigma_k: float
    transfer_row: tuple[complex, complex]
    beta_k: float
    phi_k: float
    flux_error: float
    constraint_residual: float

    @property
    def row_norm(self) -> float:
        return float(np.linalg.norm(np.asarray(self.transfer_row, dtype=complex)))

    @property
    def kasner(self) -> tuple[float, float, float, float]:
        p_s, p_t, p_varphi, _ = kasner_from_beta(self.beta_k)
        return p_s, p_s, p_t, p_varphi



def horizon_vectors(sample: SoftTransfer) -> dict[str, np.ndarray]:
    row = np.asarray(sample.transfer_row, dtype=complex)
    norm = np.linalg.norm(row)
    maximum = np.conjugate(row) / norm if norm > 0.0 else np.zeros(2, dtype=complex)
    return {
        "max_soft": maximum,
        "grav_led": np.array([1.0, 0.0], dtype=complex),
        "em_led": np.array([0.0, 1.0], dtype=complex),
        "equal_complex": np.array([1.0, 1.0j], dtype=complex) / sqrt(2.0),
    }


def local_transfer_coefficient(sample: SoftTransfer, horizon_vector: np.ndarray) -> complex:
    vector = np.asarray(horizon_vector, dtype=complex)
    if vector.shape != (2,):
        raise ValueError("horizon_vector must have two canonical channels")
    return complex(np.asarray(sample.transfer_row, dtype=complex) @ vector)


def _harmonic_derivative(ell: int, angle: float) -> float:
    x_value = np.cos(angle)
    if abs(np.sin(angle)) < 1.0e-14:
        return 0.0
    p_l = eval_legendre(ell, x_value)
    p_previous = eval_legendre(ell - 1, x_value)
    derivative_x = ell * (x_value * p_l - p_previous) / (x_value * x_value - 1.0)
    return float(abs(-np.sin(angle) * derivative_x))


@lru_cache(maxsize=None)
def _harmonic_normalization(ell: int) -> float:
    step = 1.0e-6
    grid = np.linspace(step, pi - step, 4001)
    return max(_harmonic_derivative(ell, float(angle)) for angle in grid)


def harmonic_factor(ell: int, theta: float) -> float:
    """Normalized local magnitude of the m=0 axial vector harmonic."""

    if not 0.0 <= theta <= pi:
        raise ValueError("theta must be a colatitude in [0,pi]")
    value = _harmonic_derivative(ell, theta) / _harmonic_normalization(ell)
    return 0.0 if value < 1.0e-14 else value


def nonlinear_input(
    sample: SoftTransfer,
    horizon_vector: np.ndarray,
    eta_h: float,
    theta: float,
) -> BounceInput:
    coefficient = abs(local_transfer_coefficient(sample, horizon_vector))
    local_amplitude = eta_h * coefficient * harmonic_factor(sample.ell, theta)
    return BounceInput(
        coupling_id=sample.coupling_id,
        alpha_sq=sample.alpha_sq,
        kasner=sample.kasner,
        varphi_k=sqrt(2.0) * sample.phi_k,
        initial_wall_fraction=0.5 * local_amplitude * local_amplitude,
        wall_axis=1,
        initial_background_electric_fraction=1.0e-8,
        background_electric_axis=2,
    )


DEWITT_INVERSE = np.block(
    [
        [np.eye(3) - 0.5 * np.ones((3, 3)), np.zeros((3, 1))],
        [np.zeros((1, 3)), np.ones((1, 1))],
    ]
)


@dataclass(frozen=True)
class ReflectionDiagnostic:
    predicted_p_plus: np.ndarray
    absolute_error: float
    relative_error: float
    passed: bool


@dataclass(frozen=True)
class ReducedMapResult:
    predicted_p_plus: np.ndarray
    s_onset: float
    scaled_peak_time: float
    scaled_exit_time: float
    scaled_peak_fraction: float


def kasner_residual(point: np.ndarray) -> float:
    point = np.asarray(point, dtype=float)
    return float(
        abs(np.sum(point[:3]) - 1.0)
        + abs(float(point[:3] @ point[:3] + point[3] ** 2) - 1.0)
    )


def fixed_wall_reflection(
    p_minus: np.ndarray,
    wall_axis: int,
    scalar_slope: float = 0.0,
) -> np.ndarray:
    """Return the Weyl reflection for ``beta_axis-scalar_slope*varphi/2``."""

    p_minus = np.asarray(p_minus, dtype=float)
    wall = np.zeros(4, dtype=float)
    wall[wall_axis] = 1.0
    wall[3] = -0.5 * scalar_slope
    raised = DEWITT_INVERSE @ wall
    norm = float(wall @ raised)
    if abs(norm) < 1.0e-14:
        raise ValueError("the selected wall covector is null")
    reflected = p_minus - 2.0 * float(wall @ p_minus) * raised / norm
    volume_rate = float(np.sum(reflected[:3]))
    if volume_rate <= 0.0:
        raise ValueError("reflection left the contracting Kasner branch")
    return reflected / volume_rate


def reflection_diagnostic(
    p_minus: np.ndarray,
    p_plus: np.ndarray,
    predicted_p_plus: np.ndarray,
    *,
    relative_tolerance: float = 0.05,
) -> ReflectionDiagnostic:
    p_minus = np.asarray(p_minus, dtype=float)
    p_plus = np.asarray(p_plus, dtype=float)
    predicted = np.asarray(predicted_p_plus, dtype=float)
    absolute = float(np.linalg.norm(p_plus - predicted))
    predicted_shift = float(np.linalg.norm(predicted - p_minus))
    relative = absolute / max(predicted_shift, np.finfo(float).tiny)
    return ReflectionDiagnostic(predicted, absolute, relative, relative < relative_tolerance)


def _exp_square_onset(
    p_minus: np.ndarray,
    varphi_k: float,
    initial_fraction: float,
    epsilon_on: float,
    alpha_sq: float,
    wall_axis: int,
) -> float | None:
    if initial_fraction <= 0.0:
        return None
    if initial_fraction >= epsilon_on:
        return 0.0
    p_minus = np.asarray(p_minus, dtype=float)
    quadratic = 0.5 * alpha_sq * p_minus[3] ** 2
    linear = alpha_sq * varphi_k * p_minus[3] - 2.0 * p_minus[wall_axis]
    constant = log(initial_fraction / epsilon_on)
    if quadratic == 0.0:
        return -constant / linear if linear > 0.0 else None
    discriminant = linear * linear - 4.0 * quadratic * constant
    if discriminant < 0.0:
        return None
    roots = [
        (-linear - sqrt(discriminant)) / (2.0 * quadratic),
        (-linear + sqrt(discriminant)) / (2.0 * quadratic),
    ]
    positive = [root for root in roots if root >= 0.0]
    return min(positive) if positive else None


def exp_square_reduced_map(
    p_minus: np.ndarray,
    *,
    delta: float,
    varphi_k: float,
    initial_wall_fraction: float,
    alpha_sq: float,
    wall_axis: int,
    epsilon_on: float = 1.0e-12,
    epsilon_exit: float = 1.0e-12,
) -> ReducedMapResult | None:
    """Integrate the independently rescaled near-Taub one-wall scattering map.

    The transformation is exact for the diagonal exponential-square one-wall
    Hamiltonian.  It removes the vanishing ``sqrt(delta)`` momentum scale and
    is independent of the matched-asymptotic solver in ``bounce_solver``.
    """

    if delta <= 0.0:
        return None
    p_minus = np.asarray(p_minus, dtype=float)
    onset = _exp_square_onset(
        p_minus,
        varphi_k,
        initial_wall_fraction,
        epsilon_on,
        alpha_sq,
        wall_axis,
    )
    if onset is None:
        return None
    scale = sqrt(delta)
    dominant_axis = int(np.argmax(p_minus[:3]))
    taub = np.zeros(4, dtype=float)
    taub[dominant_axis] = 1.0
    taub_momentum = kasner_momenta(taub)
    momentum = kasner_momenta(p_minus)
    start_fraction = max(epsilon_on, initial_wall_fraction)
    volume_shift = (2.0 / 3.0) * (1.0 - sqrt(1.0 + 3.0 * start_fraction))
    momentum[:3] += volume_shift
    scaled_momentum = (momentum - taub_momentum) / scale
    varphi_on = varphi_k + p_minus[3] * onset
    initial_log = log(start_fraction / delta)
    state = np.concatenate([scaled_momentum, [varphi_on, initial_log]])

    def reduced_rhs(_x_value: float, values: np.ndarray) -> np.ndarray:
        scaled_pi = values[:4]
        varphi = float(values[4])
        wall_value = float(np.exp(min(700.0, values[5])))
        spatial_sum = float(np.sum(scaled_pi[:3]))
        scaled_axis_velocity = scaled_pi[wall_axis] - 0.5 * spatial_sum
        scaled_wall_velocity = (
            scaled_axis_velocity
            - 0.5 * alpha_sq * varphi * scaled_pi[3]
        )
        derivative = np.zeros(6, dtype=float)
        derivative[wall_axis] = 2.0 * wall_value
        derivative[3] = -alpha_sq * varphi * wall_value
        derivative[4] = scaled_pi[3]
        derivative[5] = -2.0 * scaled_wall_velocity
        return derivative

    def peak_event(_x_value: float, values: np.ndarray) -> float:
        scaled_pi = values[:4]
        return float(
            scaled_pi[wall_axis]
            - 0.5 * np.sum(scaled_pi[:3])
            - 0.5 * alpha_sq * values[4] * scaled_pi[3]
        )

    peak_event.direction = 1.0
    peak_event.terminal = True
    stage_one = solve_ivp(
        reduced_rhs,
        (0.0, 1.0e5),
        state,
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=2.0,
        events=peak_event,
    )
    if not stage_one.success or len(stage_one.t_events[0]) == 0:
        return None
    peak_time = float(stage_one.t_events[0][0])
    peak_state = np.asarray(stage_one.y_events[0][0], dtype=float)
    exit_log = log(epsilon_exit / delta)

    def exit_event(_x_value: float, values: np.ndarray) -> float:
        return float(values[5] - exit_log)

    exit_event.direction = -1.0
    exit_event.terminal = True
    stage_two = solve_ivp(
        reduced_rhs,
        (peak_time, 1.0e5),
        peak_state,
        method="DOP853",
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=2.0,
        events=exit_event,
    )
    if not stage_two.success or len(stage_two.t_events[0]) == 0:
        return None
    exit_time = float(stage_two.t_events[0][0])
    exit_state = np.asarray(stage_two.y_events[0][0], dtype=float)
    predicted_momentum = taub_momentum + scale * exit_state[:4]
    return ReducedMapResult(
        predicted_p_plus=normalized_kasner(predicted_momentum),
        s_onset=onset,
        scaled_peak_time=peak_time,
        scaled_exit_time=exit_time,
        scaled_peak_fraction=float(np.exp(peak_state[5])),
    )


@dataclass(frozen=True)
class InnerParameters:
    delta: float
    alpha: float
    amplitude: float
    varphi_star: float
    q_b: float
    p_scalar_in: float


def inner_scaling(
    delta: float,
    amplitude: float,
    *,
    alpha: float = 1.0,
    c0: float = 1.0,
) -> dict[str, float]:
    """Return the contracting-layer variables fixed by dominant balance."""

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0,1)")
    if not 0.0 < abs(amplitude) < 1.0:
        raise ValueError("the wall amplitude must lie in (0,1)")
    l_b = log(1.0 / abs(amplitude))
    psi_star = sqrt(2.0 * l_b) / alpha
    varphi_star = sqrt(2.0) * psi_star
    q_b = alpha * alpha * abs(varphi_star) / 2.0
    x_slope = 4.0 * sqrt(2.0) * alpha * sqrt(l_b) / c0
    return {
        "L_B": l_b,
        "psi_star_leading": psi_star,
        "varphi_star_leading": varphi_star,
        "q_B_leading": q_b,
        "x_log_wall_slope": x_slope,
        "x_width": 1.0 / x_slope,
        "z_width": q_b / x_slope,
    }


def symbolic_inner_identities() -> dict[str, Any]:
    """Construct the leading equations and exact Liouville solution."""

    q_b, alpha = sp.symbols("q_B alpha", positive=True)
    chi, p_hat, omega_hat = sp.symbols("chi p_hat Omega_hat", real=True)
    varphi = 2 * q_b / alpha**2 + chi / q_b
    scalar_force = sp.limit(-alpha**2 * varphi * omega_hat / q_b, q_b, sp.oo)
    wall_transport = sp.limit(alpha**2 * varphi * p_hat / q_b, q_b, sp.oo)
    z, p_in = sp.symbols("z p_in", real=True, positive=True)
    profile_p = -p_in * sp.tanh(p_in * z)
    profile_wall = p_in**2 * sp.sech(p_in * z) ** 2 / 2
    profile_chi = -sp.log(sp.cosh(p_in * z))
    residuals = [
        sp.simplify(sp.diff(profile_p, z) + 2 * profile_wall),
        sp.simplify(sp.diff(profile_chi, z) - profile_p),
        sp.simplify(sp.diff(sp.log(profile_wall), z) - 2 * profile_p),
        sp.simplify(profile_p**2 + 2 * profile_wall - p_in**2),
    ]
    return {
        "scalar_force": scalar_force,
        "centered_scalar": p_hat,
        "wall_transport": wall_transport,
        "profile_residuals": residuals,
    }


def analytic_inner_profile(
    z: np.ndarray | float,
    p_scalar_in: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centered scalar, momentum, and wall fraction."""

    values = np.asarray(z, dtype=float)
    momentum_scale = float(p_scalar_in)
    chi = -np.log(np.cosh(momentum_scale * values))
    momentum = -momentum_scale * np.tanh(momentum_scale * values)
    wall = 0.5 * momentum_scale**2 / np.cosh(momentum_scale * values) ** 2
    return chi, momentum, wall


def explicit_inner_reflection(p_minus: np.ndarray) -> np.ndarray:
    """Reflect the scalar Kasner component in the singular inner limit."""

    values = np.asarray(p_minus, dtype=float).copy()
    if values.shape != (4,):
        raise ValueError("p_minus must have four Kasner components")
    values[3] *= -1.0
    return values


def finite_q_orbit(
    q_b: float,
    p_scalar_in: float,
    *,
    alpha: float = 1.0,
    z_max: float | None = None,
    rtol: float = 2.0e-12,
    atol: float = 2.0e-14,
) -> dict[str, Any]:
    """Integrate the finite-q centered scalar-wall equations."""

    if q_b <= 0.0 or p_scalar_in <= 0.0:
        raise ValueError("q_B and incoming scalar momentum must be positive")
    extent = max(12.0 / p_scalar_in, 12.0) if z_max is None else float(z_max)
    chi0, p0, omega0 = analytic_inner_profile(-extent, p_scalar_in)
    initial = np.asarray([float(chi0), float(p0), float(omega0)], dtype=float)

    def inner_rhs(_z: float, state: np.ndarray) -> np.ndarray:
        chi, p_hat, omega_hat = state
        coefficient = 2.0 + alpha * alpha * chi / (q_b * q_b)
        return np.asarray(
            [p_hat, -coefficient * omega_hat, coefficient * p_hat * omega_hat],
            dtype=float,
        )

    solution = solve_ivp(
        inner_rhs,
        (-extent, extent),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=min(0.02, 0.1 / p_scalar_in),
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = solution.y[:, -1]
    invariant = solution.y[1] ** 2 + 2.0 * solution.y[2]
    return {
        "q_B": float(q_b),
        "p_scalar_in": float(p_scalar_in),
        "z_extent": extent,
        "p_scalar_out": float(final[1]),
        "reflection_error": float(abs(final[1] + p_scalar_in)),
        "invariant_drift": float(np.max(np.abs(invariant - invariant[0]))),
        "solution": solution,
    }


def canonical_to_kasner_jacobian(momentum: np.ndarray) -> np.ndarray:
    """Jacobian of volume-normalized Kasner exponents with respect to momentum."""

    values = np.asarray(momentum, dtype=float)
    if values.shape != (4,):
        raise ValueError("momentum must contain three spatial and one scalar entry")
    velocity = velocities_from_momenta(values)
    volume_velocity = float(np.sum(velocity[:3]))
    if abs(volume_velocity) <= 1.0e-14:
        raise ValueError("volume velocity must be nonzero")
    velocity_jacobian = np.block(
        [
            [np.eye(3) - 0.5 * np.ones((3, 3)), np.zeros((3, 1))],
            [np.zeros((1, 3)), np.ones((1, 1))],
        ]
    )
    volume_gradient = np.sum(velocity_jacobian[:3, :], axis=0)
    jacobian = np.zeros((4, 4), dtype=float)
    for index in range(4):
        jacobian[index, :] = (
            velocity_jacobian[index, :] * volume_velocity
            - velocity[index] * volume_gradient
        ) / volume_velocity**2
    return jacobian
