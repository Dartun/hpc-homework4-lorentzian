import numpy as np

def lorentzian_histogram(n, bins=100, xmin=-10.0, xmax=10.0, seed=0):
    """
    Baseline: sample n points from Lorentzian/Cauchy using inverse transform
    and histogram into (bins, [xmin,xmax]). Returns counts (int64).
    """
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    x = 1.0 / np.tan(np.pi * u)  # cot(pi*u)

    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)

def split_chunks(n, p):
    """
    Split integer n across p workers as evenly as possible.
    """
    chunks = (n // p) * np.ones(p, dtype=int)
    chunks[: n % p] += 1
    return chunks