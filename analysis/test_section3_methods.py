import numpy as np

from lorentzian_core import lorentzian_histogram

from thread_lorentz import run_threaded
from mp_lorentz import run_multiproc
from ppe_lorentz import run_ppe
from async_lorentz import run_async

# Optional imports (comment out until installed)
# from dask_lorentz import run_dask
# from joblib_lorentz import run_joblib
# from mpire_lorentz import run_mpire
# from numba_lorentz import run_numba

def main():
    n = 500_000
    bins = 100
    xmin, xmax = -10.0, 10.0
    seed = 123

    base = lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

    tests = [
        ("threading", run_threaded(n, n_threads=4, bins=bins, xmin=xmin, xmax=xmax, seed=seed)),
        ("multiprocessing", run_multiproc(n, n_cores=4, bins=bins, xmin=xmin, xmax=xmax, seed=seed)),
        ("processpoolexecutor", run_ppe(n, max_workers=4, bins=bins, xmin=xmin, xmax=xmax, seed=seed)),
        ("asyncio", run_async(n, n_tasks=4, bins=bins, xmin=xmin, xmax=xmax, seed=seed)),
    ]

    for name, out in tests:
        diff = np.max(np.abs(out - base))
        print(f"{name:18s} max|diff| vs baseline = {diff}")

if __name__ == "__main__":
    main()