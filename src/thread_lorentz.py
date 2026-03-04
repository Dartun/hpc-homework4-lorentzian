import threading
import numpy as np
from lorentzian_core import lorentzian_histogram, split_chunks

def add_chunk(n, counts, lock, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    local = lorentzian_histogram(n, bins=bins, xmin=xmin, xmax=xmax, seed=seed)
    with lock:
        counts += local

def run_threaded(n, n_threads=4, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    chunks = split_chunks(n, n_threads)

    counts = np.zeros(bins, dtype=np.int64)
    lock = threading.Lock()
    threads = []

    for i in range(n_threads):
        t = threading.Thread(
            target=add_chunk,
            args=(int(chunks[i]), counts, lock, bins, xmin, xmax, seed + i),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return counts