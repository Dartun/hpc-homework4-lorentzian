import numpy as np
import multiprocessing as mp
from lorentzian_core import lorentzian_histogram, split_chunks


def _worker(args):
    n, bins, xmin, xmax, seed = args
    return lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)


def run_multiproc(n, n_cores=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    """
    Run Lorentzian sampling in parallel using multiprocessing Pool.
    Each process computes a local histogram; we sum at the end (no locks).
    """
    chunks = split_chunks(n, n_cores)
    work = [(int(chunks[i]), bins, xmin, xmax, seed + i) for i in range(n_cores)]

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(_worker, work)

    return np.sum(results, axis=0, dtype=np.int64)