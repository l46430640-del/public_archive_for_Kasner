"""Restricted diagonal Hamiltonian records in their historical conventions."""
import json
from math import sqrt
from pathlib import Path

import numpy as np

from .background import Coupling, critical_charge_to_mass, solve_branch_at_delta, solve_interior
from .scattering import BounceConfig, SoftTransfer
from .transfer import integrate_transfer
from .restricted_helpers import DELTAS, FREQUENCIES, KAPPAS, _directions, _rebuild_trajectory


def rebuild(output):
    output = Path(output)
    root = Path(__file__).resolve().parents[2]
    if (root / "build").resolve() not in output.resolve().parents:
        raise ValueError("restricted reconstruction output must be below build/")
    output.mkdir()
    frozen = [json.loads(line) for line in (root / "data/trajectories.jsonl").read_text().splitlines()]
    def key(row):
        return tuple(row["input"][field] for field in ("target_delta", "omega_M", "kappa", "direction_id"))
    lookup = {key(row): row for row in frozen}
    config = BounceConfig(epsilon_on=1e-12, epsilon_exit=1e-12, rtol=2e-12,
                          atol=2e-14, max_step=5., max_bounce_duration=1e5)
    coupling = Coupling("exp_square", 1.)
    critical = critical_charge_to_mass(1.)
    count, max_tangent_difference = 0, 0.
    with (output / "trajectories.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for delta in DELTAS:
            exterior = solve_branch_at_delta(delta, coupling, r_max=180.)
            interior = solve_interior(exterior, log_z_max=3., kasner_tolerance=1e-6,
                sustain_log_z=1., rtol=2.5e-13 if delta <= 1e-8 else 2e-12,
                atol=5e-15 if delta <= 1e-8 else 1e-13,
                max_step=5e-4 if delta <= 1e-8 else .002, horizon_series_order=2)
            state = interior.solution.sol(interior.sigma_k)
            measured = exterior.charge_to_mass / critical - 1
            for omega in FREQUENCIES:
                transfer = integrate_transfer(interior, omega, 2)
                sample = SoftTransfer(coupling_id="exp_square", alpha_sq=1., ell=2,
                    omega_m=omega, delta=measured, sigma_k=interior.sigma_k,
                    transfer_row=tuple(transfer.soft_magnetic_matrix), beta_k=float(state[1]),
                    phi_k=float(state[0]), flux_error=transfer.flux_error,
                    constraint_residual=transfer.constraint_residual)
                for name, direction in _directions(sample).items():
                    for kappa in KAPPAS:
                        row = _rebuild_trajectory(delta, sample, name, direction, kappa, config)
                        old = lookup[key(row)]
                        difference = np.linalg.norm(np.array(row["plateaus"]["p_plus"])
                                                    - old["plateaus"]["p_plus"]) / sqrt(measured)
                        max_tangent_difference = max(max_tangent_difference, float(difference))
                        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                        count += 1
    # Check agreement in the same tangent normalization as the wall results.
    if count != 495 or max_tangent_difference >= 1e-6:
        raise RuntimeError("restricted trajectories differ from the frozen calculation")
    report = {"status": "PASS", "count": count, "max_tangent_reproduction_difference": max_tangent_difference,
              "scope": "restricted diagonal Hamiltonian; main retains perpendicular electric energy, comparison omits it"}
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return report
