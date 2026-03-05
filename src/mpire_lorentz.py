import numpy as np
from mpire import WorkerPool
from lorentzian_core import lorentzian_histogram, split_chunks

def run_mpire(n, n_jobs=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, n_jobs)
    args = [(int(chunks[i]), bins, xmin, xmax, seed + i) for i in range(n_jobs)]

    with WorkerPool(n_jobs=n_jobs) as pool:
        results = pool.starmap(lorentzian_histogram, args)

    return np.sum(results, axis=0, dtype=np.int64)