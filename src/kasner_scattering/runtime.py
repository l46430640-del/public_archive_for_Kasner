"""Apply and verify the numerical thread setting used for reconstruction."""
from contextlib import contextmanager
import threading


@contextmanager
def single_threaded():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("start numerical reconstruction in the main thread of a dedicated process")
    import numpy
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise RuntimeError("single-thread BLAS initialization could not be verified")
        yield {"blas_threads": 1, "verified": True}
