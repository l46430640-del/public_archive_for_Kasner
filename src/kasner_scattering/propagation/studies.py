"""Prescribed numerical comparisons, with no fit or parameter reselection."""
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import io
from .inputs import HERE, ROOT


def jobs(study):
    result = []
    def add(module, function, **kwargs):
        result.append({"module": module, "function": function, "parameters": kwargs})
    if study == "reference":
        for h, c in ((.4, .4), (.2, .4), (.1, .4), (.2, .2), (.2, .1)):
            add("response_experiment", "predict", spacing=h, cfl=c)
        add("response_experiment", "predict", spacing=.2, phase=np.pi/2)
        add("response_experiment", "predict", spacing=.2, fixed_kasner=True)
        for label in ("weak", "near", "strong"):
            for h in (.4, .2, .1):
                add("response_experiment", "run_nonlinear", label=label, spacing=h)
        for c in (.2, .1):
            add("response_experiment", "run_nonlinear", label="strong", spacing=.2, cfl=c)
        add("response_experiment", "run_nonlinear", label="strong", spacing=.2, domain_factor=6.)
        add("response_experiment", "run_nonlinear", label="strong", spacing=.2, coupling="constant")
        for h in (.4, .2, .1):
            add("response_experiment", "run_nonlinear", label="strong", spacing=h, phase=np.pi/2)
        for h in (.2, .1):
            add("response_independent", "run", spacing=h)
        for dxi, cutoff, quadrature in ((.25, 600., 1536), (.125, 900., 2048)):
            add("response_modes", "run", dxi=dxi, cutoff=cutoff, quadrature=quadrature)
    elif study == "directed":
        for n, c in ((128, .4), (256, .4), (512, .4), (256, .2), (256, .1)):
            add("directed", "predict", cells=n, cfl=c)
        add("directed", "predict", cells=256, fixed_kasner=True)
        for label in ("weak", "strong"):
            for n, c, length in ((128, .4, 4.), (256, .4, 4.), (512, .4, 4.),
                                 (256, .2, 4.), (256, .1, 4.), (256, .4, 8.)):
                add("directed_nonlinear", "run", label=label, cells=n, cfl=c, domain_factor=length)
            for n in (256, 512):
                add("directed_independent", "run", label=label, cells=n)
        for dxi, cutoff, nodes in ((.25, 600., 1536), (.125, 900., 2048)):
            add("directed_modes", "run", dxi=dxi, cutoff=cutoff, nodes=nodes)
    elif study == "exchange":
        for label in ("weak", "near", "strong"):
            for h in (.4, .2, .1):
                add("second_event", "run", label=label, spacing=h, end=380.)
        for label in ("weak", "strong"):
            for c in (.2, .1):
                add("second_event", "run", label=label, spacing=.2, cfl=c, end=380.)
        add("second_event", "run", label="strong", spacing=.2, end=380., domain_factor=6.)
        add("second_event", "run", label="strong", spacing=.2, end=380., coupling="constant")
        for label, h, length in (("weak", .2, 1.), ("weak", .1, 5.), ("strong", .2, 5.), ("strong", .1, 5.)):
            add("second_independent", "run", label=label, spacing=h, end=300., domain_factor=length)
    return result


def wall_study(output):
    from .single_wall import integrate
    from .wall_verification import verify_wall
    folder = output / "wall"
    folder.mkdir()
    config = json.loads((ROOT / "data/wall/protocol.json").read_text())
    backgrounds = json.loads((HERE / "background_points.json").read_text())["records"]
    count = 0
    for delta in config["wall_delta_targets"]:
        beta = min(backgrounds, key=lambda r: abs(r["target_delta"]-delta))["beta_k"]
        for phi in config["wall_initial_scalar"]:
            for method, tol in [("DOP853", t) for t in config["main_relative_tolerances"]] + [("Radau", config["independent_relative_tolerance"])]:
                name = f"d{delta:.0e}_phi{phi}_{method}_r{tol:.0e}"
                record, arrays = integrate(delta, beta, phi, method, tol)
                io.save(folder, name, record, arrays)
                count += 1
    report = verify_wall(folder)
    if not report["all_eight_accepted"]:
        raise RuntimeError("one or more magnetic controls failed their existing acceptance criteria")
    return {"integrations": count, "all_eight_accepted": True}


def carrier_study():
    from .carrier import generate_inputs
    from .horizon_low_frequency import run
    data = generate_inputs()
    io.save_carrier("matching_data", data)
    run()
    return {"backgrounds": len(data["records"]), "continuous_regular_horizon_basis": True}


def run(study, output):
    from ..runtime import single_threaded
    with single_threaded() as runtime:
        return _run(study, output, runtime)


def _run(study, output, runtime):
    output = io.configure(output)
    groups = ("reference", "directed", "exchange", "wall", "carrier", "restricted-wall") if study == "all" else (study,)
    report = {"status": "RUNNING", "studies": list(groups), "jobs": [], "numerical_runtime": runtime}
    status_file = output / "rebuild_status.json"
    def record_status():
        status_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    record_status()
    try:
        for group in groups:
            if group == "wall":
                report["wall"] = wall_study(output)
            elif group == "carrier":
                report["carrier"] = carrier_study()
            elif group == "restricted-wall":
                from ..restricted import rebuild
                report["restricted_wall"] = rebuild(output / "restricted-wall")
            else:
                for job in jobs(group):
                    started = perf_counter()
                    entry = {"study": group, **job, "status": "RUNNING"}
                    report["jobs"].append(entry)
                    record_status()
                    module = importlib.import_module(f"{__package__}.{job['module']}")
                    result = getattr(module, job["function"])(**job["parameters"])
                    entry["seconds"] = perf_counter()-started
                    if result.get("status") == "NUMERICAL_FAILURE":
                        raise RuntimeError("nonfinite evolution; see saved result")
                    entry["status"] = "COMPLETED"
                    record_status()
        report["status"] = "COMPLETED_REQUIRES_COMPARISON"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = repr(exc)
        raise
    finally:
        record_status()
    return report
