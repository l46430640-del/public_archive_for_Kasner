import json
import os
import subprocess
import sys

from kasner_scattering.runtime import single_threaded


def test_active_thread_limit():
    from threadpoolctl import threadpool_info
    with single_threaded() as runtime:
        assert runtime == {"blas_threads": 1, "verified": True}
        assert all(p["num_threads"] == 1 for p in threadpool_info() if p["user_api"] == "blas")
        assert "filepath" not in json.dumps(runtime)


def test_numpy_loaded_before_package():
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    code = """
import numpy as np
from threadpoolctl import threadpool_info
from kasner_scattering.runtime import single_threaded
before = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
assert before and all(n == 2 for n in before)
with single_threaded():
    assert all(p['num_threads'] == 1 for p in threadpool_info() if p['user_api'] == 'blas')
    matrix = np.stack([np.arange(-3, 4, dtype=float)**i for i in range(7)])
    target = np.zeros(7)
    target[1] = 1
    assert np.isfinite(np.linalg.solve(matrix, target)).all()
"""
    subprocess.run([sys.executable, "-c", code], env=environment, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
