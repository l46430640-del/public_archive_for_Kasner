"""Explicit output directories; committed evidence is always read-only."""
import importlib
import json
from pathlib import Path
import shutil

import numpy as np

from .inputs import HERE, ROOT, file_hash

REF_OUTPUT = ROOT / "build/reference"
DIRECTED_OUTPUT = ROOT / "build/directed"
EXCHANGE_OUTPUT = ROOT / "build/exchange"
CARRIER_OUTPUT = ROOT / "build/carrier"


def configure(output):
    global REF_OUTPUT, DIRECTED_OUTPUT, EXCHANGE_OUTPUT, CARRIER_OUTPUT
    output = Path(output).resolve()
    build = (ROOT / "build").resolve()
    if output == build or not output.is_relative_to(build):
        raise ValueError("choose a new directory strictly inside this repository's build directory")
    output.mkdir(parents=True, exist_ok=False)
    REF_OUTPUT, DIRECTED_OUTPUT, EXCHANGE_OUTPUT, CARRIER_OUTPUT = [output / name for name in ("reference", "directed", "exchange", "carrier")]
    for path in (REF_OUTPUT, DIRECTED_OUTPUT, EXCHANGE_OUTPUT, CARRIER_OUTPUT):
        path.mkdir()
    for source, target in (("reference_selection.json", REF_OUTPUT / "frozen_selection.json"),
                           ("directed_protocol.json", DIRECTED_OUTPUT / "protocol.json"),
                           ("directed_prediction.json", DIRECTED_OUTPUT / "frozen_prediction.json"),
                           ("exchange_protocol.json", EXCHANGE_OUTPUT / "protocol.json")):
        shutil.copyfile(HERE / source, target)
    frozen = ROOT / "data/response/directed/weak.npz"
    with np.load(frozen) as data:
        protocol_path = DIRECTED_OUTPUT / "frozen_prediction.json"
        protocol = json.loads(protocol_path.read_text())
        archive = DIRECTED_OUTPUT / "frozen_prediction.npz"
        np.savez_compressed(archive, times=np.array(protocol["comparison_times"]), points=data["points"])
        protocol["raw_file"] = archive.relative_to(ROOT).as_posix()
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    locations = {
        "response_experiment": REF_OUTPUT, "response_independent": REF_OUTPUT, "response_modes": REF_OUTPUT,
        "directed": DIRECTED_OUTPUT, "directed_nonlinear": DIRECTED_OUTPUT, "directed_analysis": DIRECTED_OUTPUT,
        "second_event": EXCHANGE_OUTPUT, "second_independent": EXCHANGE_OUTPUT,
    }
    for name, path in locations.items():
        module = importlib.import_module(f"{__package__}.{name}")
        module.OUTPUT = path
    return output


def save(folder, name, result, arrays):
    folder = Path(folder).resolve()
    if (ROOT / "build").resolve() not in folder.parents:
        raise ValueError("result output must be strictly below build/")
    if "/" in name or "\\" in name or Path(name).name != name or name in (".", ".."):
        raise ValueError("a result name cannot contain a path")
    path = folder / (name + ".json")
    raw = folder / (name + ".npz")
    if path.exists() or raw.exists():
        raise FileExistsError(f"refusing to replace a result: {name}")
    if arrays:
        np.savez_compressed(raw, **arrays)
        result["raw_file"] = raw.relative_to(ROOT).as_posix()
        result["raw_sha256"] = file_hash(raw)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"result": name, "status": result.get("status", "completed")}), flush=True)
    return result


def save_reference(name, result, **arrays):
    return save(REF_OUTPUT, name, result, arrays)


def save_directed(name, result, **arrays):
    return save(DIRECTED_OUTPUT, name, result, arrays)


def save_exchange(name, result, **arrays):
    return save(EXCHANGE_OUTPUT, name, result, arrays)


def save_carrier(name, result, **arrays):
    return save(CARRIER_OUTPUT, name, result, arrays)
