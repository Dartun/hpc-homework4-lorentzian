import numpy as np
import multiprocessing as mp
from functools import partial
from lorentzian_core import lorentzian_histogram, split_chunks

def run_multiproc(n, n_cores=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, n_cores)

    # Fix default args except n and seed
    func = partial(lorentzian_histogram, bins=bins, xmin=xmin, xmax=xmax)

    # Important on Windows: must be inside __main__ guard in the caller script
    with mp.Pool(processes=n_cores) as pool:
        results = pool.starmap(func, [(int(chunks[i]), seed + i) for i in range(n_cores)])

    return np.sum(results, axis=0, dtype=np.int64)