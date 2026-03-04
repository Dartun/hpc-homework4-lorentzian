import asyncio
import numpy as np
from lorentzian_core import lorentzian_histogram, split_chunks

async def async_lorentzian_histogram(n, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    # CPU-bound: just call sync function
    return lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)

async def get_counts(n, n_tasks=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, n_tasks)
    tasks = [
        asyncio.create_task(async_lorentzian_histogram(int(chunks[i]), bins, xmin, xmax, seed + i))
        for i in range(n_tasks)
    ]
    results = await asyncio.gather(*tasks)
    return np.sum(results, axis=0, dtype=np.int64)

def run_async(n, n_tasks=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    return asyncio.run(get_counts(n, n_tasks, bins, xmin, xmax, seed))