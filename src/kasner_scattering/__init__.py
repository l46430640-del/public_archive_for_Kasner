"""Reproduction package for coherent critical Kasner response."""

import os

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "1"

__version__ = "2.0.0"
