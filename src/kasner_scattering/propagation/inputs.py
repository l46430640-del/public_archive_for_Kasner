"""Read the prescribed carrier, packet parameters and comparison protocol."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "data/inputs"


def encode(values):
    return np.stack((np.asarray(values).real, np.asarray(values).imag), axis=-1).tolist()


def decode(values):
    raw = np.asarray(values, dtype=float)
    return raw[..., 0] + 1j * raw[..., 1]


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def frozen_case(label):
    data = json.loads((HERE / "reference_selection.json").read_text())
    case = next(row for row in data["phase_zero_cases"] if row["label"] == label)
    return data, case


def frozen_directed():
    return json.loads((HERE / "directed_prediction.json").read_text())
