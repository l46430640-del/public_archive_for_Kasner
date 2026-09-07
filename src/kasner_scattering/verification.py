"""Recompute the displayed observables from immutable numerical evidence."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .propagation.inputs import ROOT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_check(root=ROOT):
    manifest = read(root / "MANIFEST.json")
    names = [row["path"] for row in manifest["files"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate manifest paths")
    for row in manifest["files"]:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("manifest path leaves the archive")
        if not path.is_file() or path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("integrity mismatch: " + row["path"])
    return len(manifest["files"])


def response_checks(root=ROOT):
    result = []
    evidence = root / "data/response"
    for group, labels in (("reference", ("weak", "near", "strong")), ("directed", ("weak", "strong"))):
        with np.load(evidence / group / "linear.npz", allow_pickle=False) as linear:
            for label in labels:
                with np.load(evidence / group / (label + ".npz"), allow_pickle=False) as data:
                    mixing = float(data["mixing"])
                    if group == "reference":
                        norm = float(data["normalization_squared"])
                        coeff = linear["scalar_coefficients"][-1]
                        actual = np.load(evidence / group / f"nonlinear_{label}_h0.1_c0.4_p0.00000_L4_T1.4_exp_square.npz")["final_spectrum"][:, 3] - data["p0"]
                    else:
                        norm = float(data["normalization"])**2
                        coeff = CubicSpline(linear["times"], linear["scalar_coefficients"], axis=0)(data["times"])
                        coeff = CubicSpline(linear["points"], coeff, axis=-1)(data["points"])
                        b = linear["backgrounds"]
                        background = CubicSpline(linear["times"], np.sqrt(2)*b[:, 5]/abs(b[:, 3]+b[:, 4]))(data["times"])
                        with np.load(evidence / group / f"nonlinear_{label}_N512_c0.4_L4.npz") as raw:
                            actual = CubicSpline(raw["times"], raw["spectrum"][:, :, 3], axis=0)(data["times"])-background[:, None]
                    parts = norm * coeff * np.array([1., mixing, mixing**2])[:, None]
                    coherent = parts.sum(axis=-2)
                    omitted = parts[..., 0, :] + parts[..., 2, :]
                    for value, key in ((parts, "quadratic_parts"), (coherent, "coherent"), (omitted, "omitted_cross"), (actual, "nonlinear")):
                        np.testing.assert_allclose(value, data[key], rtol=0, atol=1e-17, err_msg=f"{group}/{label}/{key}")
                    denominator = np.max(abs(coherent), axis=-1)
                    error = np.max(abs(actual-coherent), axis=-1)/denominator
                    omitted_error = np.max(abs(actual-omitted), axis=-1)/denominator
                    np.testing.assert_allclose(error, data["coherent_error_fraction"], rtol=1e-14)
                    np.testing.assert_allclose(omitted_error, data["omitted_error_fraction"], rtol=1e-14)
                    result.append({"background": group, "input": label, "coherent_percent_error": (100*error).tolist(),
                                   "omitted_percent_error": (100*omitted_error).tolist()})
    return result


def exchange_checks(root=ROOT):
    folder = root / "data/response/exchange"
    windows = read(folder / "windows.json")
    for row in windows["curves"]:
        with np.load(folder / (row["case"]+"_history.npz")) as history:
            j = np.argmin(abs(history["points"]-row["position"]))
            left, right = row["fwhm_indices"]
            peak = row["peak_index"]
            b = history["magnetic"][:, j]
            assert b[left] < b[peak]/2 <= b[left+1]
            assert b[right] < b[peak]/2 <= b[right-1]
        with np.load(folder / f"{row['case']}_curve{j}.npz") as data:
            integrals = data["weights"] @ data["rates"]
            np.testing.assert_allclose(integrals, list(row["integrals"].values()), rtol=0, atol=2e-16)
            measured = data["spectrum"][-1, 3]-data["spectrum"][0, 3]
            np.testing.assert_allclose(measured, row["scalar_change"], rtol=0, atol=1e-16)
            dt = np.diff(data["times"])
            np.testing.assert_array_equal(data["weights"], np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2])
    assert len(windows["curves"]) == 15
    assert not any(r["accepted_at_1e_minus4"] for r in windows["platform_screen"])
    return {"curves": 15, "accepted_output_platform": False}


def table_checks(root=ROOT):
    with (root / "data/tables/response_profiles.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for panel, group, label, time_index in (("a", "reference", "weak", None), ("b", "directed", "strong", 0)):
        subset = [r for r in rows if r["panel"] == panel]
        with np.load(root / f"data/response/{group}/{label}.npz") as data:
            for field in ("coherent", "omitted_cross", "nonlinear"):
                expected = data[field] if time_index is None else data[field][time_index]
                # The figure export and quadratic decomposition sum in a
                # different order; use the existing 1e-17 arithmetic bound.
                np.testing.assert_allclose([float(r[field]) for r in subset], expected, rtol=0, atol=1e-17)
            np.testing.assert_array_equal([float(r["y"]) for r in subset], data["points"])
    with (root / "data/tables/letter_ordering.csv").open() as stream:
        ordering = list(csv.DictReader(stream))
    for i, row in enumerate(ordering):
        group = "reference" if i == 0 else "directed"
        for label in ("weak", "strong"):
            with np.load(root / f"data/response/{group}/{label}.npz") as data:
                for field, key in (("prediction", "coherent"), ("nonlinear", "nonlinear")):
                    value = data[key] if i == 0 else data[key][i-1]
                    np.testing.assert_allclose(np.max(abs(value))/abs(data["p0"]), float(row[f"{label}_{field}"]), rtol=1e-14)
    assert len(rows) == 240 and len(ordering) == 4
    windows = read(root / "data/response/exchange/windows.json")["curves"]
    with (root / "data/tables/first_exchange.csv").open() as stream:
        exchange = list(csv.DictReader(stream))
    for row, label in zip(exchange, ("weak", "near", "strong")):
        w = next(w for w in windows if w["case"] == label and w["position"] == 0.)
        for key, value in w["integrals"].items():
            np.testing.assert_allclose(float(row[key]), value, rtol=0, atol=2e-16)
        np.testing.assert_allclose([float(row["T_start"]), float(row["T_end"])], w["window"], rtol=0, atol=1e-12)
    sources = read(root / "data/tables/signed_sources.json")
    with np.load(root / "data/response/directed/strong.npz") as data:
        for row in sources:
            i = np.argmin(abs(data["times"]-row["time"]))
            j = np.argmin(abs(data["points"]-row["point"]))
            np.testing.assert_allclose(data["quadratic_parts"][i, :, j]/abs(data["p0"]),
                                       row["cc_cl_ll_over_p0"], rtol=0, atol=1e-15)
        with (root / "data/tables/prediction_errors.csv").open() as stream:
            for i, row in enumerate(csv.DictReader(stream)):
                np.testing.assert_allclose(float(row["T"]), data["times"][i], rtol=0, atol=1e-13)
                for column, key in (("full_profile_relative_error", "coherent_error_fraction"),
                                    ("omitted_cross_relative_error", "omitted_error_fraction")):
                    np.testing.assert_allclose(float(row[column]), data[key][i], rtol=1e-14)
                estimate = next(r["numeric_absolute"] for r in read(root / "data/response/directed/comparison_rows.json")["rows"]
                                if r["case"] == "strong" and r["time"] == data["times"][i])
                assert float(row["combined_numerical_estimate"]) == estimate
    with (root / "data/tables/directed.csv").open() as stream:
        for row, comparison in zip(csv.DictReader(stream), ordering[1:]):
            for field in ("weak_prediction", "weak_nonlinear", "strong_prediction", "strong_nonlinear"):
                assert row[field] == comparison[field]
    energy = read(root / "data/native/reference/prediction_h0.1_c0.4_p0.00000_T1.4.json")["energy_coefficients"]
    with (root / "data/tables/reference.csv").open() as stream:
        for row, label in zip(csv.DictReader(stream), ("weak", "near", "strong")):
            with np.load(root / f"data/response/reference/{label}.npz") as data:
                for field, key in (("predicted_relative", "coherent"), ("nonlinear_relative", "nonlinear")):
                    np.testing.assert_allclose(float(row[field]), np.max(abs(data[key]))/abs(data["p0"]), rtol=1e-14)
                fraction = float(data["mixing"])**2*energy[2]/energy[0]
                np.testing.assert_allclose(float(row["soft_self_energy_fraction"]), fraction, rtol=1e-14)
    from .propagation.scales import scale_record, projected_components
    from .propagation.response_model import background_initial
    records = read(root / "data/inputs/matching_data.json")["records"]
    with (root / "data/tables/background_initial_data.csv").open() as stream:
        for row in csv.DictReader(stream):
            record = next(r for r in records if r["background"]["delta_target"] == float(row["delta"]))
            bg, charge = background_initial(record)
            scale, components = scale_record(record), projected_components(record)
            values = {"beta": record["background"]["beta"], "psi_initial": bg[2], "q": charge,
                      "k": scale["k"], "w": scale["packet_width"],
                      "B_re": components["soft_magnetic"].real, "B_im": components["soft_magnetic"].imag,
                      "E_re": components["transverse_electric"].real, "E_im": components["transverse_electric"].imag}
            for field, value in values.items():
                np.testing.assert_allclose(float(row[field]), value, rtol=2e-14, atol=1e-16)
    trajectories = [json.loads(line) for line in (root / "data/trajectories.jsonl").read_text().splitlines()]
    with (root / "data/tables/isolated_wall_495.csv").open() as stream:
        wall_rows = list(csv.DictReader(stream))
    assert len(wall_rows) == len(trajectories) == 495
    for row in wall_rows:
        trajectory = trajectories[int(row["record_index"])]
        values = {"delta": trajectory["input"]["measured_delta"], "L_B": trajectory["clock"]["L_B"],
                  "constraint_residual": trajectory["plateaus"]["constraint_residual"],
                  "finite_q_relative_error": trajectory["finite_q_map"]["relative_error"]}
        for field, value in values.items():
            if field == "finite_q_relative_error":
                # Preserve both historical norm evaluations; their difference
                # is at the arithmetic floor, below the resolved wall effect.
                np.testing.assert_allclose(float(row[field]), value, rtol=0, atol=32*np.finfo(float).eps)
            else:
                assert float(row[field]) == value
        for field, key in (("incoming_plateau", "kasner_in"), ("outgoing_plateau", "kasner_out")):
            assert (row[field] == "True") == trajectory["plateaus"][key]["passed"]
    return {"figure_points": len(rows), "ordering_rows": len(ordering)}


def carrier_checks(root=ROOT):
    from .propagation.inputs import decode
    data = read(root / "data/inputs/carrier.json")
    reference = next(r for r in data["records"] if r["background"]["delta_target"] == 1e-6)
    row = decode(reference["components"]["soft_magnetic"])
    pol = row.conj()/np.linalg.norm(row)
    results = []
    for record in data["records"]:
        b = record["background"]
        np.testing.assert_allclose(pol, decode(record["polarization"]), rtol=0, atol=2e-16)
        v, pi = decode(record["canonical"]), decode(record["canonical_momentum"])
        z, chi, h, beta, psi, kn = np.exp(b["sigma_k"]), b["chi"], b["h"], b["beta"], b["psi"], b["knorm"]
        metric_n = h*z*np.exp(chi)
        soft = -1j*b["coordinate_frequency"]*np.sqrt(6)*np.exp(chi)*z/(np.sqrt(-metric_n)*np.sqrt(32)*kn)*v[1]
        electric = np.sqrt(6)*np.sqrt(-metric_n)/(h*np.sqrt(32)*kn)*(pi[1]+h*z*z*beta*psi*v[1])
        np.testing.assert_allclose(soft, decode(record["components"]["soft_magnetic"]), rtol=2e-14, atol=1e-16)
        np.testing.assert_allclose(electric, decode(record["components"]["transverse_electric"]), rtol=2e-14, atol=1e-16)
        k = b["coordinate_frequency"]/(b["a_parallel"]*b["ktrace"])
        width = b["a_parallel"]*b["ktrace"]*b["mass"]/np.sqrt(b["delta_target"])
        np.testing.assert_allclose(k*width, .2/np.sqrt(b["delta_target"]), rtol=2e-15)
        results.append({"delta": b["delta_target"], "kw": k*width})
    return results


def regular_horizon_checks(root=ROOT):
    from .propagation.inputs import decode
    band = read(root / "data/inputs/regular_horizon.json")
    assert len(band["rows"]) == 12 and len(band["checks"]) == 9
    assert band["linear_band_data_available"] and not band["full_horizon_second_order_ready"]
    for row in band["rows"] + band["checks"]:
        field, momentum = decode(row["canonical"]), decode(row["momentum"])
        flux = (field.conj().T @ momentum - momentum.conj().T @ field)/(2j)
        omega = row["omega_times_mass"]/band["mass"]
        error = float(np.max(abs(flux+omega*np.eye(2))))
        np.testing.assert_allclose(error, row["flux_absolute_error"], rtol=0, atol=2e-16)
    zero = band["rows"][0]
    assert zero["omega_times_mass"] == 0
    assert np.max(abs(decode(zero["components"]["soft_magnetic"]))) == 0
    negative = band["negative_frequency_check"]
    positive = next(row for row in band["rows"] if abs(row["omega_times_mass"] + negative["omega_times_mass"]) < 1e-15)
    for field in ("canonical", "momentum"):
        np.testing.assert_allclose(decode(negative[field]), decode(positive[field]).conj(), rtol=2e-12, atol=1e-14)
    return {"frequencies": 12, "start_and_accuracy_comparisons": 9,
            "zero_frequency_in_same_basis": True, "negative_frequency_conjugation": True}


def continuous_field_checks(root=ROOT):
    from .propagation.evolution import derivatives
    from .propagation.response_summary import sample
    result = []
    for group in ("reference", "directed"):
        folder = root / f"data/native/{group}"
        prediction_name = "prediction_h0.1_c0.4_p0.00000_T1.4" if group == "reference" else "prediction_N512_c0.4"
        metadata = read(folder / (prediction_name+".json"))
        with np.load(folder / (prediction_name+".npz")) as raw:
            final, bg = raw["final"], raw["backgrounds"][-1]
            spacing = .1 if group == "reference" else metadata["spacing"]
            gradient, _ = derivatives(final[[0, 2]], spacing)
            em = final[[1, 3]] - bg[2]*bg[5]*final[[0, 2]]
            finite = np.array([em, gradient]).transpose(1, 0, 2) if group == "reference" else np.array([final[[0, 2]], em, gradient])
            for path in sorted(folder.glob("*modes*.json")):
                saved = read(path)
                with np.load(path.with_suffix(".npz")) as modes:
                    basis = modes["complex_basis_fields"] if group == "reference" else modes["complex_basis_fields"][-1]
                    for row in saved["comparisons"]:
                        with np.load(root / f"data/response/{group}/{row['case']}.npz") as control:
                            mixing = float(control["mixing"])
                            norm = np.sqrt(float(control["normalization_squared"])) if group == "reference" else float(control["normalization"])
                        if group == "reference":
                            continuum = norm*(basis[0]+mixing*basis[1]).real
                            fd = sample(norm*(finite[0]+mixing*finite[1]), raw["grid"], modes["points"])
                            error = np.max(abs(continuum-fd))/np.max(abs(continuum))
                            key = "relative_em_field_error"
                        else:
                            continuum = norm*(basis[:, 0]+mixing*basis[:, 1]).real
                            indices = np.rint((modes["points"]-raw["grid"][0])/spacing).astype(int)
                            fd = norm*(finite[:, 0, indices]+mixing*finite[:, 1, indices])
                            error = np.max(abs(continuum[1:]-fd[1:]))/np.max(abs(continuum[1:]))
                            key = "relative_em_error"
                        np.testing.assert_allclose(error, row[key], rtol=2e-8, atol=1e-15)
                        result.append({"background": group, "frequency_spacing": saved["frequency_spacing"],
                                       "input": row["case"], "relative_em_error": float(error)})
    return result


def convergence_checks(root=ROOT):
    def load(group, name):
        path = root / f"data/native/{group}/{name}.npz"
        if not path.exists():
            path = root / f"data/response/{group}/{name}.npz"
        with np.load(path) as data:
            return dict(data)
    report = {}
    ref = root / "data/response/reference"
    for label in ("weak", "near", "strong"):
        values = [np.load(ref / f"nonlinear_{label}_h{h:g}_c0.4_p0.00000_L4_T1.4_exp_square.npz")["final_spectrum"] for h in (.4, .2, .1)]
        diffs = [float(np.max(abs(a-b))) for a,b in zip(values[:-1], values[1:])]
        expected = next(r for r in read(ref / "comparison_rows.json")["rows"] if r["case"] == label)["space_differences"]
        np.testing.assert_allclose(diffs, expected, rtol=1e-12, atol=1e-16)
        report["reference_"+label] = diffs
    directed = read(root / "data/response/directed/nonlinear_verification.json")
    for label in ("weak", "strong"):
        values = [load("directed", f"nonlinear_{label}_N{n}_c{c:g}_L4")["spectrum"][-1]
                  for n,c in ((128,.4),(256,.4),(512,.4),(256,.2),(256,.1))]
        space = [float(np.max(abs(values[a]-values[b]))) for a,b in ((0,1),(1,2))]
        temporal = [float(np.max(abs(values[a]-values[b]))) for a,b in ((1,3),(3,4))]
        expected = next(r for r in directed["cases"] if r["case"] == label)
        np.testing.assert_allclose(space, expected["space_final_differences"], rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(temporal, expected["time_final_differences"], rtol=1e-12, atol=1e-16)
        baseline = load("directed", f"nonlinear_{label}_N256_c0.4_L4")
        domain = load("directed", f"nonlinear_{label}_N256_c0.4_L8")
        assert np.max(abs(baseline["spectrum"]-domain["spectrum"])) == expected["domain_doubling_spectrum_difference"]
        report["directed_"+label] = {"space": space, "time": temporal}
    exchange = read(root / "data/response/exchange/verification.json")
    for row in exchange["cases"]:
        histories = [load("exchange", f"event_{row['label']}_h{h:g}_c0.4_T380_L4_exp_square") for h in (.4,.2,.1)]
        shared = np.round(histories[0]["times"],8)
        for data in histories[1:]:
            shared = np.intersect1d(shared,np.round(data["times"],8))
        indices = [np.array([np.flatnonzero(abs(d["times"]-t)<1e-7)[0] for t in shared]) for d in histories]
        values = [d["spectrum"][ids] for d,ids in zip(histories,indices)]
        diffs = [float(np.max(abs(a-b))) for a,b in zip(values[:-1],values[1:])]
        np.testing.assert_allclose(diffs, row["convergence"]["spectrum"]["differences"], rtol=1e-12, atol=1e-16)
        report["exchange_"+row["label"]] = diffs
    return report


def verify(root=ROOT, check_manifest=True):
    root = Path(root)
    result = {"status": "PASS"}
    if check_manifest:
        result["manifest_files"] = manifest_check(root)
    result["response"] = response_checks(root)
    result["exchange"] = exchange_checks(root)
    result["tables"] = table_checks(root)
    result["carrier"] = carrier_checks(root)
    result["regular_horizon"] = regular_horizon_checks(root)
    result["continuous_fields"] = continuous_field_checks(root)
    from .comparison import directed_diagnostics
    result["frozen_four_dimensional_diagnostics"] = directed_diagnostics(root / "data/native/directed")
    assert result["frozen_four_dimensional_diagnostics"]["status"] == "PASS"
    result["convergence"] = convergence_checks(root)
    from .propagation.wall_verification import verify_wall
    wall = verify_wall(root / "data/wall", write_outputs=False)
    assert wall["all_eight_accepted"]
    frozen_wall = read(root / "data/wall/verification.json")
    for actual, expected in zip(wall["rows"], frozen_wall["rows"]):
        for key in ("q_B", "tangent_error", "full_spectrum_numerical_difference", "full_spectrum_window_difference", "platform_drift", "finite_window_residual_bound"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-16)
    result["magnetic_controls"] = {"count": len(wall["rows"]), "all_eight_accepted": True}
    files, arrays = 0, 0
    for path in (root / "data").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                value = data[key]
                if value.dtype.kind not in "buifc":
                    raise ValueError("unexpected nonnumeric frozen array: " + path.name + "/" + key)
                if not np.isfinite(value).all():
                    raise ValueError("nonfinite frozen evidence: " + path.name + "/" + key)
                arrays += 1
        files += 1
    result["finite_arrays"] = arrays
    result["array_files"] = files
    trajectories = [json.loads(line) for line in (root / "data/trajectories.jsonl").read_text().splitlines()]
    assert len(trajectories) == 495
    result["restricted_trajectories"] = len(trajectories)
    return result
