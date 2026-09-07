"""Compare reconstructed trajectories with their frozen observation arrays."""
import json
from pathlib import Path

import numpy as np

from .propagation.inputs import ROOT


def directed_diagnostics(folder):
    """Check the published diagnostic ceilings separately from spectrum errors."""
    limits = {"einstein_relative": 9.21e-10, "maxwell_relative": 1.43e-10,
              "scalar_over_expansion_squared": 8.86e-13,
              "reduced_energy_balance_absolute": 9.35e-12}
    rows = []
    for label in ("weak", "strong"):
        record = json.loads((Path(folder) / f"independent_{label}_N512.json").read_text())
        assert len(record["four_dimensional_residuals"]) == 12
        values = {key: max(abs(r[key]) for r in record["four_dimensional_residuals"]) for key in limits}
        rows.append({"input": label, "maxima": values, "reported_ceilings": limits,
                     "within_reported_ceilings": all(values[key] <= limits[key] for key in limits),
                     "independent_spectrum_difference": record["max_spectrum_difference"],
                     "spectrum_within_reported_error": record["max_spectrum_difference"] < 1.29e-10})
    return {"status": "PASS" if all(r["within_reported_ceilings"] and r["spectrum_within_reported_error"] for r in rows) else "REVIEW_REQUIRED",
            "scope": "Derivative-fit residual ceilings and independent spectrum error are separate checks", "rows": rows}


def compare(output):
    output = Path(output).resolve()
    if (ROOT / "build").resolve() not in output.parents:
        raise ValueError("comparison outputs must be strictly below build/")
    status = json.loads((output / "rebuild_status.json").read_text())
    if status["status"] != "COMPLETED_REQUIRES_COMPARISON":
        raise ValueError("the prescribed integrations are incomplete")
    rows = []
    from .propagation.studies import jobs
    for group in status["studies"]:
        prescribed = jobs(group)
        recorded = [r for r in status["jobs"] if r["study"] == group]
        if len(prescribed) != len(recorded) or any(
            actual["status"] != "COMPLETED" or any(actual[key] != expected[key]
                for key in ("module", "function", "parameters"))
            for actual, expected in zip(recorded, prescribed)
        ):
            raise ValueError("the reconstructed study does not cover its prescribed jobs: " + group)
    for group in status["studies"]:
        if group in ("carrier", "restricted-wall"):
            continue
        for generated in sorted((output / group).glob("*.npz")):
            candidates = [ROOT / f"data/native/{group}/{generated.name}",
                          ROOT / f"data/response/{group}/{generated.name}",
                          ROOT / f"data/{group}/{generated.name}"]
            frozen = next((p for p in candidates if p.is_file()), None)
            if frozen is None:
                if generated.stem == "frozen_prediction" or "_T1.25_" in generated.stem:
                    continue
                raise ValueError("missing frozen counterpart: " + generated.name)
            with np.load(generated, allow_pickle=False) as new, np.load(frozen, allow_pickle=False) as old:
                differences = {}
                indices = None
                if "points" in old.files and "points" in new.files and old["points"].shape != new["points"].shape:
                    indices = np.array([np.argmin(abs(new["points"]-point)) for point in old["points"]])
                    np.testing.assert_allclose(new["points"][indices], old["points"], rtol=0, atol=1e-12)
                for key in old.files:
                    if key not in new.files:
                        raise ValueError("missing reconstructed array: " + generated.name + "/" + key)
                    value, expected = new[key], old[key]
                    if indices is not None:
                        if key in ("points", "initial_spectrum", "final_spectrum"):
                            value = value[indices]
                        elif key in ("states", "scalar_coefficients"):
                            value = value[..., indices]
                    if value.shape != expected.shape:
                        raise ValueError("observation shape mismatch: " + generated.name + "/" + key)
                    diff = float(np.max(abs(value-expected)))
                    if key in ("spectrum", "axes_spectrum", "initial_spectrum", "final_spectrum"):
                        limit = {"reference": 2e-12, "directed": 2e-11, "exchange": 5e-10, "wall": 1e-12}[group]
                        if diff > limit:
                            raise ValueError(f"spectral reconstruction mismatch: {generated.name}/{key}: {diff} > {limit}")
                    else:
                        scale = max(float(np.max(abs(expected))), np.finfo(float).tiny)
                        if diff > 2e-11*scale + 1e-15:
                            raise ValueError(f"array reconstruction mismatch: {generated.name}/{key}: {diff}")
                    differences[key] = diff
                rows.append({"study": group, "result": generated.stem, "maximum_absolute_differences": differences})
    if "carrier" in status["studies"]:
        from .verification import read
        new = read(output / "carrier/matching_data.json")["records"]
        old = read(ROOT / "data/inputs/matching_data.json")["records"]
        carrier_rows = []
        for r, reference in zip(new, old):
            assert r["background"]["delta_target"] == reference["background"]["delta_target"]
            for key in ("canonical", "canonical_momentum", "polarization"):
                np.testing.assert_allclose(r[key], reference[key], rtol=2e-10, atol=1e-13)
            for key in reference["components"]:
                np.testing.assert_allclose(r["components"][key], reference["components"][key], rtol=2e-10, atol=1e-13)
            carrier_rows.append({"delta": r["background"]["delta_target"], "field_and_momentum": "PASS"})
        assert len(new) == len(old) == 2
        rows.extend(carrier_rows)
        band = read(output / "carrier/horizon_regular_band.json")
        reference_band = read(ROOT / "data/inputs/regular_horizon.json")
        regular_rows = []
        for key in ("rows", "checks", "negative_frequency_check"):
            values = band[key] if isinstance(band[key], list) else [band[key]]
            references = reference_band[key] if isinstance(reference_band[key], list) else [reference_band[key]]
            assert len(values) == len(references)
            for value, expected in zip(values, references):
                for field in ("omega_times_mass", "start", "endpoint", "canonical", "momentum"):
                    np.testing.assert_allclose(value[field], expected[field], rtol=2e-10, atol=1e-13)
                for component in expected["components"]:
                    np.testing.assert_allclose(value["components"][component], expected["components"][component], rtol=2e-10, atol=1e-13)
                regular_rows.append({"omega_times_mass": value["omega_times_mass"],
                                     "start": value["start"], "field_and_momentum": "PASS"})
        rows.append({"regular_horizon_transfer": regular_rows})
    if not rows and "restricted-wall" not in status["studies"]:
        raise ValueError("no reconstructed arrays were compared")
    report = {"status": "PASS", "studies": status["studies"], "comparisons": rows,
              "scope": "reconstruction agreement; physical discretization errors require the separate convergence comparisons"}
    if "directed" in status["studies"]:
        report["four_dimensional_diagnostics"] = directed_diagnostics(output / "directed")
        if report["four_dimensional_diagnostics"]["status"] != "PASS":
            report["status"] = "REVIEW_REQUIRED"
    (output / "comparison.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    if report["status"] != "PASS":
        raise ValueError("reconstructed diagnostics exceed reported ceilings; inspect comparison.json")
    return report
