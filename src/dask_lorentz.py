import numpy as np
import dask
from dask import delayed
from lorentzian_core import lorentzian_histogram, split_chunks

@delayed
def delayed_hist(n, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    return lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

def run_dask(n, n_tasks=4, bins=100, xmin=-10.0, xmax=10.0, seed=0, scheduler="processes"):
    chunks = split_chunks(n, n_tasks)
    tasks = [delayed_hist(int(chunks[i]), bins, xmin, xmax, seed + i) for i in range(n_tasks)]
    results = dask.compute(*tasks, scheduler=scheduler)
    return np.sum(results, axis=0, dtype=np.int64)