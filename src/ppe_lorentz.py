import numpy as np
from concurrent.futures import ProcessPoolExecutor
from lorentzian_core import lorentzian_histogram, split_chunks

def run_ppe(n, max_workers=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(lorentzian_histogram, int(chunks[i]), bins, xmin, xmax, seed + i)
            for i in range(max_workers)
        ]
        results = [f.result() for f in futures]

    return np.sum(results, axis=0, dtype=np.int64)