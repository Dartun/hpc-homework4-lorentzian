import numpy as np
from joblib import Parallel, delayed
from lorentzian_core import lorentzian_histogram, split_chunks

def run_joblib(n, n_jobs=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(lorentzian_histogram)(int(chunks[i]), bins=bins, xmin=xmin, xmax=xmax, seed=seed + i)
        for i in range(n_jobs)
    )
    return np.sum(results, axis=0, dtype=np.int64)