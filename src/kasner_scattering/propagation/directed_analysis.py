"""Numerical equations and integration routines for the local EMS model."""
from .io import DIRECTED_OUTPUT as OUTPUT
import json
import numpy as np
from scipy.interpolate import CubicSpline
from .directed import combine, protocol
from .inputs import ROOT

def load(name):
    path = OUTPUT / f'{name}.json'
    record = json.loads(path.read_text())
    with np.load(ROOT / record['raw_file']) as raw:
        arrays = dict(raw)
    return (record, arrays)

def sample(values, old_times, old_points, times, points):
    time_sample = CubicSpline(old_times, values, axis=0)(times)
    return CubicSpline(old_points, time_sample, axis=-1)(points)

def predicted(name, times, points, coherent=True):
    record, data = load(name)
    coeff = sample(data['scalar_coefficients'], data['times'], data['points'], times, points)
    return np.array([combine(coeff, mix, coherent=coherent) for mix in protocol()['mixing'].values()])
