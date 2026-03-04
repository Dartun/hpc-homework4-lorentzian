import argparse
import time
import numpy as np
from mpi4py import MPI


def lorentzian_histogram(n, bins=100, xmin=-10.0, xmax=10.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    u = rng.random(n)
    x = 1.0 / np.tan(np.pi * u)
    counts, _ = np.histogram(x, bins=bins, range=(xmin, xmax))
    return counts.astype(np.int64)


def split_chunks(n_total, size):
    chunks = (n_total // size) * np.ones(size, dtype=int)
    chunks[: n_total % size] += 1
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_total", type=int, default=10_000_000, help="total samples across all ranks")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--xmin", type=float, default=-10.0)
    ap.add_argument("--xmax", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42, help="base seed for SeedSequence")
    ap.add_argument("--out", type=str, default="results/mpi_lorentzian_histogram.txt")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Independent RNG stream per rank (good practice)
    ss = np.random.SeedSequence(args.seed)
    child_seeds = ss.spawn(size)
    rng = np.random.default_rng(child_seeds[rank])

    chunks = split_chunks(args.n_total, size)
    n_local = int(chunks[rank])

    if rank == 0:
        print(f"MPI ranks = {size}")
        print(f"Total samples n_total = {args.n_total}")
        print(f"Bins={args.bins}, range=[{args.xmin}, {args.xmax}]")

    comm.Barrier()
    t0 = time.time()

    local_counts = lorentzian_histogram(
        n_local, bins=args.bins, xmin=args.xmin, xmax=args.xmax, rng=rng
    )

    global_counts = np.empty_like(local_counts)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

    comm.Barrier()
    t1 = time.time()

    if rank == 0:
        runtime = t1 - t0
        sps = args.n_total / runtime if runtime > 0 else float("inf")
        print(f"Runtime: {runtime:.3f} s")
        print(f"Samples/sec: {sps:.3e}")

        # Save: bin_center, global_count
        edges = np.linspace(args.xmin, args.xmax, args.bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        out_arr = np.column_stack((centers, global_counts))

        import os
        os.makedirs("results", exist_ok=True)
        np.savetxt(args.out, out_arr, fmt=["%.6f", "%d"])
        print("Saved:", args.out)


if __name__ == "__main__":
    main()