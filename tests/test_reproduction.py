from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from kasner_scattering.pipeline import read_json, read_trajectories, verify
from kasner_scattering.scattering import (
    analytic_inner_profile,
    canonical_to_kasner_jacobian,
    explicit_inner_reflection,
    finite_q_orbit,
    kasner_momenta,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_verifies() -> None:
    assert verify(ROOT)["status"] == "PASS"


def test_trajectory_count_and_semantics() -> None:
    rows = read_trajectories(ROOT / "data" / "trajectories.jsonl")
    assert len(rows) == 225
    assert all(row["plateaus"]["outcome"] == "KASNER_TRANSITION" for row in rows)
    assert all(row["packet_energy_tau_1"] > 0.0 for row in rows)


def test_scalar_reflection_and_canonical_jacobian() -> None:
    row = read_trajectories(ROOT / "data" / "trajectories.jsonl")[0]
    p_minus = np.asarray(row["plateaus"]["p_minus"])
    assert np.allclose(
        explicit_inner_reflection(p_minus),
        row["finite_q_map"]["asymptotic_reflection"],
        rtol=0.0,
        atol=1.0e-15,
    )
    momentum = kasner_momenta(p_minus)
    assert np.allclose(
        canonical_to_kasner_jacobian(momentum),
        row["finite_q_map"]["canonical_to_kasner_jacobian"],
        rtol=0.0,
        atol=2.0e-15,
    )


def test_liouville_profile_and_finite_q_orbit() -> None:
    z = np.linspace(-4.0, 4.0, 501)
    _chi, momentum, wall = analytic_inner_profile(z, 1.0)
    assert np.max(wall) == pytest.approx(0.5)
    assert momentum[0] > 0.999
    assert momentum[-1] < -0.999
    orbit = finite_q_orbit(5.0, 1.0)
    assert orbit["reflection_error"] < 0.05
    assert orbit["invariant_drift"] < 1.0e-8


def test_control_bounds_do_not_encode_missing_maps_as_zero() -> None:
    controls = read_json(ROOT / "data" / "controls.json")
    for row in controls["records"]:
        if row["map_value_status"] == "UPPER_BOUND_ONLY":
            assert row["rescaled_map_distance"] is None
            assert row["rescaled_impulse_upper"] > 0.0


def test_hash_mutation_is_detected(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ("backgrounds.json", "transfers.json", "controls.json", "summary.json"):
        (data / name).write_bytes((ROOT / "data" / name).read_bytes())
    (data / "trajectories.jsonl").write_bytes(
        (ROOT / "data" / "trajectories.jsonl").read_bytes()
    )
    mutated = read_json(data / "summary.json")
    mutated["trajectory_count"] = 224
    (data / "summary.json").write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(RuntimeError, match="summary_content_hash"):
        verify(tmp_path, verify_figure=False)
