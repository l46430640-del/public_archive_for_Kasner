"""Public verification, reconstruction, and plotting pipeline."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from math import cos, log, pi, sin, sqrt
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
import sympy as sp

from .background import (
    Coupling,
    critical_charge_to_mass,
    direct_critical_charge_to_mass,
    exact_critical_horizon_charge,
    solve_branch_at_delta,
    solve_interior,
)
from .scattering import (
    BounceConfig,
    BounceInput,
    SoftTransfer,
    canonical_to_kasner_jacobian,
    exp_square_reduced_map,
    explicit_inner_reflection,
    finite_q_orbit,
    horizon_vectors,
    inner_scaling,
    kasner_momenta,
    nonlinear_input,
    solve_bounce,
    symbolic_inner_identities,
)
from .transfer import integrate_transfer, integrate_transfer_with_sensitivity


DELTAS = tuple(10.0 ** (-9.0 + 0.5 * index) for index in range(11))
FREQUENCIES = (0.1, 0.2, 0.4)
OPEN_DELTAS = DELTAS
OPEN_FREQUENCIES = tuple(0.195 + 0.001 * index for index in range(11))
CONTROL_DELTAS = DELTAS[:7]
KAPPAS = (0.1, 0.25, 0.5)
CP1_RADIUS = 0.1
SPATIAL_PHASE = pi / 4


def repository_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "data").is_dir():
        return source_root
    current = Path.cwd().resolve()
    if (current / "data").is_dir():
        return current
    raise RuntimeError("run this command from the reproduction repository")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_trajectories(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def file_digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def with_digest(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = digest(result)
    return result


def _check_embedded_digest(value: dict[str, Any], field: str) -> bool:
    copy = dict(value)
    recorded = copy.pop(field)
    return recorded == digest(copy)


def _kasner_residual(point: Iterable[float]) -> float:
    values = np.asarray(tuple(point), dtype=float)
    return float(
        abs(np.sum(values[:3]) - 1.0)
        + abs(float(values[:3] @ values[:3] + values[3] ** 2) - 1.0)
    )


def _analytic_checks() -> dict[str, bool]:
    beta, delta, c0 = sp.symbols("beta delta c0", positive=True)
    denominator = beta**2 + 3
    p_s = 2 / denominator
    p_t = (beta**2 - 1) / denominator
    p_scalar = 2 * sp.sqrt(2) * beta / denominator
    gaussian = sp.integrate(
        sp.exp(-sp.Symbol("v", real=True) ** 2 / sp.Symbol("T", positive=True) ** 2),
        (sp.Symbol("v", real=True), -sp.oo, sp.oo),
    )
    inner = symbolic_inner_identities()
    return {
        "linear_kasner_constraint": sp.simplify(2 * p_s + p_t - 1) == 0,
        "quadratic_kasner_constraint": sp.simplify(
            2 * p_s**2 + p_t**2 + p_scalar**2 - 1
        )
        == 0,
        "critical_scaling_substitution": sp.simplify(
            p_s.subs(beta, c0 / sp.sqrt(delta))
            - 2 * delta / (c0**2 + 3 * delta)
        )
        == 0,
        "gaussian_flux": sp.simplify(
            gaussian - sp.sqrt(sp.pi) * sp.Symbol("T", positive=True)
        )
        == 0,
        "inner_scalar_force": sp.sstr(inner["scalar_force"]) == "-2*Omega_hat",
        "inner_wall_transport": sp.sstr(inner["wall_transport"]) == "2*p_hat",
        "inner_profile": all(value == 0 for value in inner["profile_residuals"]),
    }


def verify(root: Path | None = None, *, verify_figure: bool = True) -> dict[str, Any]:
    """Verify the committed data without running the expensive reconstruction."""

    root = root or repository_root()
    data = root / "data"
    backgrounds = read_json(data / "backgrounds.json")
    transfers = read_json(data / "transfers.json")
    controls = read_json(data / "controls.json")
    summary = read_json(data / "summary.json")
    trajectories = read_trajectories(data / "trajectories.jsonl")

    checks: dict[str, bool] = {}
    checks.update(_analytic_checks())
    for name, value in (
        ("background_content_hash", backgrounds),
        ("transfer_content_hash", transfers),
        ("control_content_hash", controls),
        ("summary_content_hash", summary),
    ):
        checks[name] = _check_embedded_digest(value, "content_sha256")
    checks["background_record_hashes"] = all(
        _check_embedded_digest(row, "record_sha256")
        for row in backgrounds["records"]
    )
    checks["transfer_record_hashes"] = all(
        _check_embedded_digest(row, "record_sha256")
        for row in transfers["records"]
    )
    checks["trajectory_record_hashes"] = all(
        _check_embedded_digest(row, "record_sha256") for row in trajectories
    )
    checks["control_record_hashes"] = all(
        _check_embedded_digest(row, "record_sha256") for row in controls["records"]
    )

    q_critical = float(backgrounds["threshold"]["critical_charge_to_mass"])
    checks["exact_threshold"] = abs(critical_charge_to_mass() / q_critical - 1.0) < 2.0e-15
    checks["backgrounds_above_threshold"] = all(
        row["target_delta"] > 0.0
        and row["measured_delta"] > 0.0
        and row["relative_delta_error"] < 1.0e-6
        for row in backgrounds["records"]
    )
    checks["background_constraints"] = max(
        row["kasner_matching_residual"] for row in backgrounds["records"]
    ) < 1.000001e-6
    checks["transfer_open_set"] = (
        transfers["frequency_interval"] == [0.195, 0.205]
        and transfers["polarization_ball_radius"] == 0.1
        and min(row["resolved_row_norm_lower"] for row in transfers["neighborhoods"])
        > 0.0
        and min(row["cp1_amplitude_lower"] for row in transfers["neighborhoods"])
        > 0.0
    )
    checks["transfer_flux"] = max(row["flux_residual"] for row in transfers["records"]) < 1.0e-9
    checks["transfer_equation_defect"] = 0.0 < max(
        row["strong_equation_defect"] for row in transfers["records"]
    ) < 2.0e-5
    checks["transfer_independent_check"] = max(
        transfers["independent_check"]["relative_errors"]
    ) < 1.0e-8

    checks["trajectory_count"] = len(trajectories) == 495
    checks["two_kasner_plateaus"] = all(
        row["plateaus"]["outcome"] == "KASNER_TRANSITION"
        and row["plateaus"]["kasner_in"]["passed"]
        and row["plateaus"]["kasner_out"]["passed"]
        for row in trajectories
    )
    checks["trajectory_constraints"] = max(
        row["plateaus"]["constraint_residual"] for row in trajectories
    ) < 1.0e-10
    checks["kasner_constraints"] = max(
        max(
            _kasner_residual(row["plateaus"]["p_minus"]),
            _kasner_residual(row["plateaus"]["p_plus"]),
        )
        for row in trajectories
    ) < 1.0e-10
    checks["peak_clock"] = (
        summary["maximum_peak_clock_relative_error"] < 0.02
        and summary["maximum_normalized_clock_drift_last_three_decades"] < 0.05
    )
    checks["finite_q_map"] = summary["maximum_finite_q_map_relative_error"] < 0.02
    checks["vanishing_packet_energy"] = all(
        slope > 0.0 for slope in summary["packet_energy_log_slopes"]
    )
    allowed_outcomes = {"KASNER_TRANSITION", "NO_ONSET", "NO_POST_PLATEAU"}
    checks["control_semantics"] = len(controls["records"]) == 14 and all(
        row["outcome"] in allowed_outcomes
        and row["rescaled_impulse_upper"] > 0.0
        and (
            (row["map_value_status"] == "MEASURED_TRANSITION")
            == (row["rescaled_map_distance"] is not None)
        )
        for row in controls["records"]
    )

    manifest_path = root / "MANIFEST.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        checks["manifest_hashes"] = all(
            (root / row["path"]).stat().st_size == row["bytes"]
            and file_digest(root / row["path"]) == row["sha256"]
            for row in manifest["files"]
        )
    if verify_figure:
        checks["figure_present"] = all(
            (root / "figures" / name).is_file()
            for name in ("results_summary.pdf", "results_summary.png")
        )

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("verification failed: " + ", ".join(failed))
    return {
        "status": "PASS",
        "checks": checks,
        "trajectory_count": len(trajectories),
        "threshold_charge_to_mass": q_critical,
    }


def _encode_complex(values: np.ndarray) -> list[dict[str, float]]:
    return [
        {"re": float(value.real), "im": float(value.imag)}
        for value in np.asarray(values, dtype=complex)
    ]


def _directions(sample: SoftTransfer) -> dict[str, np.ndarray]:
    maximum = horizon_vectors(sample)["max_soft"]
    orthogonal = np.asarray(
        [-np.conjugate(maximum[1]), np.conjugate(maximum[0])], dtype=complex
    )
    result = {"max_soft": maximum}
    for index, phase in enumerate((0.0, 0.5 * pi, pi, 1.5 * pi)):
        result[f"cp1_boundary_{index}"] = (
            cos(CP1_RADIUS) * maximum
            + sin(CP1_RADIUS) * np.exp(1j * phase) * orthogonal
        )
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rebuild_trajectory(
    target_delta: float,
    sample: SoftTransfer,
    direction_id: str,
    direction: np.ndarray,
    kappa: float,
    config: BounceConfig,
) -> dict[str, Any]:
    eta_h = sample.delta ** (0.5 + kappa)
    input_data = nonlinear_input(sample, direction, eta_h, SPATIAL_PHASE)
    result = solve_bounce(input_data, config)
    if result.post_kasner is None or result.s_peak is None:
        raise RuntimeError(
            f"missing post plateau for delta={target_delta}, omega={sample.omega_m}, "
            f"direction={direction_id}, kappa={kappa}: {result.outcome}"
        )
    reduced = exp_square_reduced_map(
        result.pre_kasner,
        delta=sample.delta,
        varphi_k=input_data.varphi_k,
        initial_wall_fraction=input_data.initial_wall_fraction,
        alpha_sq=input_data.alpha_sq,
        wall_axis=input_data.wall_axis,
        epsilon_on=config.epsilon_on,
        epsilon_exit=config.epsilon_exit,
    )
    if reduced is None:
        raise RuntimeError("finite-q one-wall predictor failed")
    field_amplitude = sqrt(2.0 * input_data.initial_wall_fraction)
    scaling = inner_scaling(sample.delta, field_amplitude)
    l_b = scaling["L_B"]
    predicted_t_peak = reduced.s_onset + reduced.scaled_peak_time / sqrt(sample.delta)
    theta_peak = result.s_peak * sqrt(sample.delta) / sqrt(l_b)
    predicted_theta = predicted_t_peak * sqrt(sample.delta) / sqrt(l_b)
    p_minus = np.asarray(result.pre_kasner, dtype=float)
    p_plus = np.asarray(result.post_kasner, dtype=float)
    predicted = np.asarray(reduced.predicted_p_plus, dtype=float)
    reflected = explicit_inner_reflection(p_minus)
    epsilon = sqrt(sample.delta)
    rescaled_error = (p_plus - reflected) / epsilon
    q_b = scaling["q_B_leading"]
    canonical = kasner_momenta(p_minus)
    record = {
        "input": {
            "coupling": "exp_square",
            "target_delta": target_delta,
            "measured_delta": sample.delta,
            "omega_M": sample.omega_m,
            "kappa": kappa,
            "direction_id": direction_id,
            "direction": _encode_complex(direction),
            "eta_H": eta_h,
            "field_amplitude": field_amplitude,
            "wall_fraction": input_data.initial_wall_fraction,
        },
        "packet_energy_tau_1": sqrt(pi) * eta_h**2 * sample.delta**-0.5,
        "clock": {
            "L_B": l_b,
            "s_K": 0.0,
            "s_peak": result.s_peak,
            "t_peak": result.s_peak,
            "predicted_t_peak": predicted_t_peak,
            "Theta_peak": theta_peak,
            "predicted_Theta_peak": predicted_theta,
            "relative_error": abs(result.s_peak / predicted_t_peak - 1.0),
        },
        "plateaus": {
            "outcome": result.outcome,
            "p_minus": p_minus.tolist(),
            "p_plus": p_plus.tolist(),
            "kasner_in": result.kasner_in_window,
            "kasner_out": result.kasner_out_window,
            "constraint_residual": result.constraint_residual,
            "energy_error": result.energy_error,
        },
        "finite_q_map": {
            "q_B": q_b,
            "incoming_canonical_momentum": canonical.tolist(),
            "canonical_to_kasner_jacobian": canonical_to_kasner_jacobian(canonical).tolist(),
            "predicted_p_plus": predicted.tolist(),
            "asymptotic_reflection": reflected.tolist(),
            "relative_error": float(
                np.linalg.norm(p_plus - predicted)
                / max(np.linalg.norm(p_plus - p_minus), np.finfo(float).tiny)
            ),
            "axis_permutation_invariant_distance": float(
                np.linalg.norm(np.sort(p_plus[:3]) - np.sort(predicted[:3])) ** 2
                + (p_plus[3] - predicted[3]) ** 2
            ) ** 0.5,
            "q_B_weighted_rescaled_correction": (
                q_b * rescaled_error
            ).tolist(),
        },
        "inner_limit": {
            "p_plus_reflection_limit": reflected.tolist(),
            "rescaled_reflection_error": float(np.linalg.norm(rescaled_error)),
            "scalar_rescaled_error": float(abs(rescaled_error[3])),
            "spatial_rescaled_error": float(np.linalg.norm(rescaled_error[:3])),
            "sqrt_LB_weighted_scalar_error": float(sqrt(l_b) * abs(rescaled_error[3])),
            "sqrt_LB_weighted_spatial_error": float(sqrt(l_b) * np.linalg.norm(rescaled_error[:3])),
            "hamiltonian_to_finite_q_relative_error": float(
                np.linalg.norm(p_plus - predicted)
                / max(np.linalg.norm(p_plus - p_minus), np.finfo(float).tiny)
            ),
        },
    }
    return with_digest(record)


def _summarize(
    backgrounds: dict[str, Any],
    transfers: dict[str, Any],
    trajectories: list[dict[str, Any]],
    controls: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[tuple[float, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        value = row["input"]
        if value["target_delta"] in (1.0e-7, 1.0e-8, 1.0e-9):
            groups[(value["omega_M"], value["kappa"], value["direction_id"])].append(row)
    drifts: list[float] = []
    slopes: list[float] = []
    for rows in groups.values():
        ordered = sorted(rows, key=lambda row: row["input"]["measured_delta"])
        ratios = [row["clock"]["Theta_peak"] / row["clock"]["predicted_Theta_peak"] for row in ordered]
        drifts.append((max(ratios) - min(ratios)) / abs(sum(ratios) / len(ratios)))
        x = np.log([row["input"]["measured_delta"] for row in ordered])
        y = np.log([row["packet_energy_tau_1"] for row in ordered])
        slopes.append(float(np.polyfit(x, y, 1)[0]))
    summary = {
        "schema_version": 1,
        "scope": "first isolated Taub-degenerate magnetic transition from regular finite-energy odd horizon data",
        "full_ems_embedding_condition": "norm(Delta P_ext)/sqrt(delta) -> 0",
        "threshold_charge_to_mass": backgrounds["threshold"]["critical_charge_to_mass"],
        "frequency_interval": transfers["frequency_interval"],
        "polarization_ball_radius": transfers["polarization_ball_radius"],
        "minimum_transfer_lower_bound": min(row["resolved_row_norm_lower"] for row in transfers["neighborhoods"]),
        "minimum_polarization_lower_bound": min(row["cp1_amplitude_lower"] for row in transfers["neighborhoods"]),
        "trajectory_count": len(trajectories),
        "two_plateau_count": sum(row["plateaus"]["outcome"] == "KASNER_TRANSITION" for row in trajectories),
        "maximum_constraint_residual": max(row["plateaus"]["constraint_residual"] for row in trajectories),
        "maximum_peak_clock_relative_error": max(row["clock"]["relative_error"] for row in trajectories),
        "maximum_normalized_clock_drift_last_three_decades": max(drifts),
        "maximum_finite_q_map_relative_error": max(row["finite_q_map"]["relative_error"] for row in trajectories),
        "packet_energy_log_slopes": sorted(set(round(value, 12) for value in slopes)),
        "control_record_count": len(controls["records"]),
        "control_outcomes": sorted(set(row["outcome"] for row in controls["records"])),
    }
    summary["content_sha256"] = digest(summary)
    return summary


def rebuild(output: Path) -> dict[str, Any]:
    """Recompute all headline results into ``output`` without touching frozen data."""

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    coupling = Coupling("exp_square", 1.0)
    critical_ratio = critical_charge_to_mass(1.0)
    exact_charge = exact_critical_horizon_charge(1.0, 100)
    exact_ratio = 2 * exact_charge / (1 + exact_charge * exact_charge)
    direct_ratio, direct_rows = direct_critical_charge_to_mass()
    threshold = {
        "horizon_charge": mp.nstr(exact_charge, 100),
        "critical_charge_to_mass": mp.nstr(exact_ratio, 100),
        "legendre_degree": "-1/2+i sqrt(3)/2",
        "zero_source_condition": "Re LegendreP_nu((1+Q_H^2)/(1-Q_H^2))=0",
        "legendre_residual": "recomputed by exact_critical_horizon_charge",
        "working_precision_digits": 100,
        "direct_compactified_ode": direct_rows,
        "direct_relative_error": abs(direct_ratio / float(exact_ratio) - 1.0),
    }

    background_records: list[dict[str, Any]] = []
    interiors: dict[float, Any] = {}
    samples: list[tuple[float, SoftTransfer]] = []
    for target_delta in DELTAS:
        exterior = solve_branch_at_delta(target_delta, coupling, r_max=180.0)
        interior = solve_interior(
            exterior,
            log_z_max=3.0,
            kasner_tolerance=1.0e-6,
            sustain_log_z=1.0,
            rtol=2.5e-13 if target_delta <= 1.0e-8 else 2.0e-12,
            atol=5.0e-15 if target_delta <= 1.0e-8 else 1.0e-13,
            max_step=5.0e-4 if target_delta <= 1.0e-8 else 0.002,
            horizon_series_order=2,
        )
        interiors[target_delta] = interior
        state_k = np.asarray(interior.solution.sol(interior.sigma_k), dtype=float)
        measured_delta = exterior.charge_to_mass / critical_ratio - 1.0
        record = {
            "coupling_id": "exp_square",
            "alpha_sq": 1.0,
            "target_delta": target_delta,
            "measured_delta": measured_delta,
            "relative_delta_error": abs(measured_delta / target_delta - 1.0),
            "critical_charge_to_mass": critical_ratio,
            "phi_h": exterior.phi_h,
            "charge": exterior.charge,
            "mass": exterior.mass,
            "charge_to_mass": exterior.charge_to_mass,
            "scalar_charge": exterior.scalar_charge,
            "source_residual": exterior.source_residual,
            "sigma_k": interior.sigma_k,
            "phi_k": float(state_k[0]),
            "beta_k": float(state_k[1]),
            "chi_k": float(state_k[2]),
            "h_k": float(state_k[3]),
            "kasner_matching_residual": float(interior.residual[interior.sigma_k_index]),
        }
        background_records.append(with_digest(record))
        for frequency in FREQUENCIES:
            result = integrate_transfer(interior, frequency, 2)
            samples.append(
                (
                    target_delta,
                    SoftTransfer(
                        coupling_id="exp_square",
                        alpha_sq=1.0,
                        ell=2,
                        omega_m=frequency,
                        delta=measured_delta,
                        sigma_k=interior.sigma_k,
                        transfer_row=tuple(complex(value) for value in result.soft_magnetic_matrix),
                        beta_k=float(state_k[1]),
                        phi_k=float(state_k[0]),
                        flux_error=result.flux_error,
                        constraint_residual=result.constraint_residual,
                    ),
                )
            )
    backgrounds = {
        "schema_version": 1,
        "model": "Z(psi)=exp(psi^2), alpha^2=1, r_H=1",
        "threshold": threshold,
        "records": background_records,
    }
    backgrounds["content_sha256"] = digest(backgrounds)
    _write_json(output / "backgrounds.json", backgrounds)

    transfer_records: list[dict[str, Any]] = []
    neighborhoods: list[dict[str, Any]] = []
    for target_delta in OPEN_DELTAS:
        rows = []
        interior = interiors[target_delta]
        for frequency in OPEN_FREQUENCIES:
            result = integrate_transfer_with_sensitivity(interior, frequency)
            record = {
                key: value
                for key, value in result.items()
                if key not in {"soft_magnetic_row", "frequency_sensitivity_row"}
            }
            record["soft_magnetic_row"] = _encode_complex(result["soft_magnetic_row"])
            record["frequency_sensitivity_row"] = _encode_complex(result["frequency_sensitivity_row"])
            record["target_delta"] = target_delta
            record["measured_delta"] = interior.exterior.charge_to_mass / critical_ratio - 1.0
            record = with_digest(record)
            transfer_records.append(record)
            rows.append(record)
        center = next(row for row in rows if abs(row["omega_M"] - 0.2) < 1.0e-14)
        derivative_bound = 1.25 * max(row["frequency_sensitivity_norm"] for row in rows)
        row_lower = min(
            center["soft_magnetic_norm"] - 0.005 * derivative_bound,
            *(row["soft_magnetic_norm"] for row in rows),
        )
        neighborhoods.append(
            {
                "target_delta": target_delta,
                "frequency_interval": [0.195, 0.205],
                "center_row_norm": center["soft_magnetic_norm"],
                "frequency_sensitivity_norm_bound": derivative_bound,
                "resolved_row_norm_lower": row_lower,
                "cp1_radius": CP1_RADIUS,
                "cp1_amplitude_lower": row_lower * cos(CP1_RADIUS),
            }
        )
    transfers = {
        "schema_version": 1,
        "ell": 2,
        "horizon_normalization": "unit ingoing symplectic flux",
        "frequency_interval": [0.195, 0.205],
        "polarization_ball_radius": CP1_RADIUS,
        "records": transfer_records,
        "neighborhoods": neighborhoods,
    }
    transfers["content_sha256"] = digest(transfers)
    _write_json(output / "transfers.json", transfers)

    config = BounceConfig(
        epsilon_on=1.0e-12,
        epsilon_exit=1.0e-12,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=5.0,
        max_bounce_duration=1.0e5,
    )
    trajectories: list[dict[str, Any]] = []
    for target_delta, sample in samples:
        for direction_id, direction in _directions(sample).items():
            for kappa in KAPPAS:
                trajectories.append(
                    _rebuild_trajectory(
                        target_delta,
                        sample,
                        direction_id,
                        direction,
                        kappa,
                        config,
                    )
                )
    with (output / "trajectories.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in trajectories:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    background_by_delta = {row["target_delta"]: row for row in background_records}
    selected = [
        row
        for row in trajectories
        if row["input"]["omega_M"] == 0.2
        and row["input"]["kappa"] == 0.25
        and row["input"]["direction_id"] == "max_soft"
        and row["input"]["target_delta"] in CONTROL_DELTAS
    ]
    control_records = []
    control_config = BounceConfig(
        epsilon_on=1.0e-12,
        epsilon_exit=1.0e-12,
        rtol=2.0e-12,
        atol=2.0e-14,
        max_step=1.0,
        max_bounce_duration=1.0e5,
    )
    for source in selected:
        value = source["input"]
        p_minus = source["plateaus"]["p_minus"]
        phi_k = background_by_delta[value["target_delta"]]["phi_k"]
        for coupling_id in ("bounded_rational", "constant"):
            input_data = BounceInput(
                coupling_id=coupling_id,
                alpha_sq=1.0,
                kasner=tuple(p_minus),
                varphi_k=sqrt(2.0) * phi_k,
                initial_wall_fraction=value["wall_fraction"],
                wall_axis=0,
            )
            result = solve_bounce(input_data, control_config)
            z_ratio = 2.0 / (1.0 + phi_k * phi_k / (1.0 + phi_k * phi_k)) if coupling_id == "bounded_rational" else 1.0
            impulse_bound = value["wall_fraction"] * z_ratio / p_minus[0]
            record = {
                "coupling_id": coupling_id,
                "target_delta": value["target_delta"],
                "measured_delta": value["measured_delta"],
                "omega_M": 0.2,
                "kappa": 0.25,
                "input_wall_fraction": value["wall_fraction"],
                "outcome": result.outcome,
                "rescaled_map_distance": (
                    float(result.kasner_shift) / sqrt(value["measured_delta"])
                    if result.post_kasner is not None and result.kasner_shift is not None
                    else None
                ),
                "rescaled_impulse_upper": impulse_bound / sqrt(value["measured_delta"]),
                "constraint_residual": float(result.constraint_residual),
                "map_value_status": "MEASURED_TRANSITION" if result.post_kasner is not None else "UPPER_BOUND_ONLY",
            }
            control_records.append(with_digest(record))
    controls = {"schema_version": 1, "records": control_records}
    controls["content_sha256"] = digest(controls)
    _write_json(output / "controls.json", controls)

    summary = _summarize(backgrounds, transfers, trajectories, controls)
    _write_json(output / "summary.json", summary)
    compare = compare_rebuild(output)
    _write_json(output / "comparison.json", compare)
    if compare["status"] != "PASS":
        raise RuntimeError("full reconstruction did not match the committed results")
    return compare


def compare_rebuild(output: Path, root: Path | None = None) -> dict[str, Any]:
    """Compare a full reconstruction with the committed machine results."""

    root = root or repository_root()
    frozen_backgrounds = read_json(root / "data" / "backgrounds.json")
    rebuilt_backgrounds = read_json(output / "backgrounds.json")
    frozen_transfers = read_json(root / "data" / "transfers.json")
    rebuilt_transfers = read_json(output / "transfers.json")
    frozen_trajectories = read_trajectories(root / "data" / "trajectories.jsonl")
    rebuilt_trajectories = read_trajectories(output / "trajectories.jsonl")

    errors: dict[str, float] = {}
    errors["threshold"] = abs(
        float(rebuilt_backgrounds["threshold"]["critical_charge_to_mass"])
        / float(frozen_backgrounds["threshold"]["critical_charge_to_mass"])
        - 1.0
    )
    frozen_bg = {row["target_delta"]: row for row in frozen_backgrounds["records"]}
    errors["background"] = float(max(
        abs(row[field] / frozen_bg[row["target_delta"]][field] - 1.0)
        for row in rebuilt_backgrounds["records"]
        for field in ("phi_h", "phi_k", "beta_k", "charge", "mass")
        if frozen_bg[row["target_delta"]][field] != 0.0
    ))
    frozen_transfer = {
        (row["target_delta"], row["omega_M"]): row
        for row in frozen_transfers["records"]
    }
    errors["transfer"] = float(max(
        abs(
            row["soft_magnetic_norm"]
            / frozen_transfer[(row["target_delta"], row["omega_M"])]["soft_magnetic_norm"]
            - 1.0
        )
        for row in rebuilt_transfers["records"]
    ))
    def key(row: dict[str, Any]) -> tuple[float, float, float, str]:
        value = row["input"]
        return value["target_delta"], value["omega_M"], value["kappa"], value["direction_id"]
    frozen_trajectory = {key(row): row for row in frozen_trajectories}
    errors["trajectory"] = float(max(
        np.linalg.norm(
            np.asarray(row["plateaus"]["p_plus"])
            - np.asarray(frozen_trajectory[key(row)]["plateaus"]["p_plus"])
        )
        / max(np.linalg.norm(np.asarray(frozen_trajectory[key(row)]["plateaus"]["p_plus"])), np.finfo(float).tiny)
        for row in rebuilt_trajectories
    ))
    checks = {
        "threshold_below_1e12": bool(errors["threshold"] < 1.0e-12),
        "background_below_2e8": bool(errors["background"] < 2.0e-8),
        "transfer_below_2e8": bool(errors["transfer"] < 2.0e-8),
        "trajectory_below_2e6": bool(errors["trajectory"] < 2.0e-6),
        "trajectory_count": len(rebuilt_trajectories) == 495,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "relative_errors": errors,
        "checks": checks,
    }


def plot_results(output: Path, root: Path | None = None) -> list[Path]:
    """Generate one compact scientific summary from committed public data."""

    root = root or repository_root()
    output.mkdir(parents=True, exist_ok=True)
    backgrounds = read_json(root / "data" / "backgrounds.json")
    transfers = read_json(root / "data" / "transfers.json")
    trajectories = read_trajectories(root / "data" / "trajectories.jsonl")
    controls = read_json(root / "data" / "controls.json")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 1.7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    markers = ("o", "s", "^")
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.4), constrained_layout=True)

    axis = axes[0, 0]
    bg_rows = sorted(backgrounds["records"], key=lambda row: row["measured_delta"])
    x = np.asarray([row["measured_delta"] for row in bg_rows])
    y = np.asarray([row["beta_k"] * sqrt(row["measured_delta"]) for row in bg_rows])
    axis.semilogx(x, y, "o-", color=colors[0])
    axis.axhline(float(np.mean(y[:3])), color="0.35", linestyle="--")
    axis.set(xlabel=r"$\delta$", ylabel=r"$\beta_K\sqrt{\delta}$")
    axis.text(0.04, 0.92, "(a)", transform=axis.transAxes, fontweight="bold")

    axis = axes[0, 1]
    for index, delta in enumerate((1.0e-9, 1.0e-7, 1.0e-4)):
        rows = sorted(
            [row for row in transfers["records"] if row["target_delta"] == delta],
            key=lambda row: row["omega_M"],
        )
        axis.plot(
            [row["omega_M"] for row in rows],
            [row["soft_magnetic_norm"] for row in rows],
            marker=markers[index],
            color=colors[index],
            label=rf"$\delta=10^{{{int(round(np.log10(delta)))}}}$",
        )
    axis.axvspan(0.195, 0.205, color="#56B4E9", alpha=0.18)
    axis.set(xlabel=r"carrier frequency $\omega M$", ylabel=r"$\|T_B\|$")
    axis.legend(frameon=False)
    axis.text(0.04, 0.92, "(b)", transform=axis.transAxes, fontweight="bold")

    axis = axes[0, 2]
    for index, kappa in enumerate(KAPPAS):
        rows = sorted(
            [
                row
                for row in trajectories
                if row["input"]["omega_M"] == 0.2
                and row["input"]["direction_id"] == "max_soft"
                and row["input"]["kappa"] == kappa
            ],
            key=lambda row: row["input"]["measured_delta"],
        )
        axis.semilogx(
            [row["input"]["measured_delta"] for row in rows],
            [
                1.0e8
                * (
                    row["clock"]["Theta_peak"]
                    / row["clock"]["predicted_Theta_peak"]
                    - 1.0
                )
                for row in rows
            ],
            marker=markers[index],
            color=colors[index],
            label=rf"$\kappa={kappa}$",
        )
    axis.axhline(0.0, color="0.25", linestyle="--")
    axis.set(
        xlabel=r"$\delta$",
        ylabel=r"$10^8(\Theta_{\rm peak}/\Theta_{\rm pred}-1)$",
    )
    axis.legend(frameon=False)
    axis.text(0.04, 0.92, "(c)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1, 0]
    for index, frequency in enumerate(FREQUENCIES):
        rows = sorted(
            [
                row
                for row in trajectories
                if row["input"]["omega_M"] == frequency
                and row["input"]["direction_id"] == "max_soft"
                and row["input"]["kappa"] == 0.25
            ],
            key=lambda row: row["finite_q_map"]["q_B"],
        )
        axis.semilogy(
            [1.0 / row["finite_q_map"]["q_B"] for row in rows],
            [row["finite_q_map"]["relative_error"] for row in rows],
            marker=markers[index],
            color=colors[index],
            label=rf"$\omega M={frequency}$",
        )
    axis.set(xlabel=r"$q_B^{-1}$", ylabel="finite-wall map error")
    axis.legend(frameon=False)
    axis.text(0.04, 0.92, "(d)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1, 1]
    main_rows = sorted(
        [
            row
            for row in trajectories
            if row["input"]["omega_M"] == 0.2
            and row["input"]["direction_id"] == "max_soft"
            and row["input"]["kappa"] == 0.25
        ],
        key=lambda row: row["input"]["measured_delta"],
    )
    axis.loglog(
        [row["input"]["measured_delta"] for row in main_rows],
        [
            np.linalg.norm(
                np.asarray(row["plateaus"]["p_plus"])
                - np.asarray(row["plateaus"]["p_minus"])
            )
            / sqrt(row["input"]["measured_delta"])
            for row in main_rows
        ],
        "o-",
        color=colors[1],
        label=r"$e^{\psi^2}$",
    )
    for index, coupling in enumerate(("bounded_rational", "constant")):
        rows = sorted(
            [row for row in controls["records"] if row["coupling_id"] == coupling],
            key=lambda row: row["measured_delta"],
        )
        axis.loglog(
            [row["measured_delta"] for row in rows],
            [row["rescaled_map_distance"] if row["rescaled_map_distance"] is not None else row["rescaled_impulse_upper"] for row in rows],
            marker=("s", "^")[index],
            linestyle="--",
            color=colors[index],
            markerfacecolor="white",
            label=("bounded $Z$", "$Z=1$")[index],
        )
    axis.set(xlabel=r"$\delta$", ylabel="rescaled transition or bound")
    axis.legend(frameon=False)
    axis.text(0.04, 0.92, "(e)", transform=axis.transAxes, fontweight="bold")

    axis = axes[1, 2]
    z = np.linspace(-4.0, 4.0, 500)
    chi = -np.log(np.cosh(z))
    momentum = -np.tanh(z)
    wall = 0.5 / np.cosh(z) ** 2
    axis.plot(z, wall / max(wall), color=colors[1], label=r"$\widehat\Omega_B$")
    axis.plot(z, momentum, color=colors[0], linestyle="--", label=r"$\widehat\pi_\psi$")
    axis.plot(z, chi / max(abs(chi)), color=colors[2], linestyle=":", label=r"$\chi$")
    axis.set(xlabel=r"$z$", ylabel="normalized inner profile")
    axis.legend(frameon=False)
    axis.text(0.04, 0.92, "(f)", transform=axis.transAxes, fontweight="bold")

    pdf = output / "results_summary.pdf"
    png = output / "results_summary.png"
    figure.savefig(
        pdf,
        metadata={"Creator": "kasner-scattering", "CreationDate": None, "ModDate": None},
    )
    figure.savefig(png, dpi=220, metadata={"Software": "kasner-scattering"})
    plt.close(figure)
    return [pdf, png]


def run_wolfram(root: Path | None = None) -> dict[str, Any]:
    """Run the optional independent Wolfram Language calculation."""

    root = root or repository_root()
    executable = shutil.which("wolframscript")
    if executable is None:
        raise RuntimeError("wolframscript is not available on PATH")
    output = root / "build" / "wolfram"
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [executable, "-file", str(root / "wolfram" / "crosscheck.wls"), str(output)],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])
    payload = read_json(output / "crosscheck.json")
    if not all(payload["checks"].values()):
        raise RuntimeError("the Wolfram cross-check failed")
    def number(value: Any) -> float:
        if not isinstance(value, str):
            return float(value)
        cleaned = re.sub(r"`[^*]*", "", value).replace("*^", "e")
        return float(cleaned)

    backgrounds = read_json(root / "data" / "backgrounds.json")
    transfers = read_json(root / "data" / "transfers.json")
    frozen_threshold = mp.mpf(backgrounds["threshold"]["critical_charge_to_mass"])
    wolfram_threshold = mp.mpf(
        re.sub(r"`.*$", "", payload["threshold"]["critical_charge_to_mass"])
    )
    threshold_relative_error = abs(wolfram_threshold / frozen_threshold - 1)
    transfer_errors = []
    for anchor in payload["transfer"]:
        frozen = min(
            (
                row
                for row in transfers["records"]
                if row["target_delta"] == 1.0e-6
            ),
            key=lambda row: abs(row["omega_M"] - float(anchor["omega_M"])),
        )
        python_row = np.asarray(
            [complex(value["re"], value["im"]) for value in frozen["soft_magnetic_row"]]
        )
        wolfram_row = np.asarray(
            [complex(number(value["re"]), number(value["im"])) for value in anchor["soft_magnetic_row"]]
        )
        overlap = np.vdot(wolfram_row, python_row)
        phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
        transfer_errors.append(
            float(
                np.linalg.norm(python_row - phase * wolfram_row)
                / np.linalg.norm(python_row)
            )
        )
    payload["cross_engine"] = {
        "threshold_relative_error": str(threshold_relative_error),
        "maximum_transfer_relative_error": max(transfer_errors),
        "checks": {
            "threshold_below_1e30": threshold_relative_error < mp.mpf("1e-30"),
            "transfer_below_1e8": max(transfer_errors) < 1.0e-8,
        },
    }
    if not all(payload["cross_engine"]["checks"].values()):
        raise RuntimeError("the Python/Wolfram comparison failed")
    _write_json(output / "comparison.json", payload["cross_engine"])
    return payload
